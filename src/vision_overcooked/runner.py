from __future__ import annotations

import json
from pathlib import Path

from vision_overcooked.adapters import (
    AgentInvocationError,
    AgentResponseFormatError,
    EnvironmentAdapter,
    EvaluationAdapter,
    MacroActionExecutor,
    build_agent,
    role_action_guide,
    validate_plan_string,
)
from vision_overcooked.schemas import AgentTurnResponse, PilotRunConfig, RunRecord, TurnRecord


def _counter_summary(env: EnvironmentAdapter) -> str:
    state = env.current_state()
    counter_objects = env.mdp.get_counter_objects_dict(
        state, list(env.mdp.terrain_pos_dict["X"])
    )
    if not counter_objects:
        return "reachable_counters=empty"
    parts = []
    for name, positions in sorted(counter_objects.items()):
        if positions:
            parts.append(f"{name}:{len(positions)}")
    return "reachable_counters=" + (", ".join(parts) if parts else "empty")


def _utensil_summary(env: EnvironmentAdapter) -> str:
    state = env.current_state()
    utensil_states = env.mdp.get_utensil_states(state)
    segments = []
    for bucket in ("empty", "cooking", "ready", "full", "partially_full", "wrong"):
        values = utensil_states.get(bucket, [])
        if values:
            segments.append(f"{bucket}={','.join(values)}")
    return "utensils=" + ("; ".join(segments) if segments else "none")


def _recipe_requirement_summary(env: EnvironmentAdapter, order: str) -> str:
    for utensil_kind, recipes in env.mdp.recipe_config["recipes"].items():
        if order in recipes:
            ingredients = recipes[order]["recipe"]
            cook_time = recipes[order]["cook_time"]
            return (
                f"recipe_requirement=to make {order}, use {utensil_kind} with ingredients "
                f"{', '.join(ingredients)}; cook_time={cook_time}"
            )
    return f"recipe_requirement=unknown for {order}"


def _player_summary(env: EnvironmentAdapter) -> str:
    state = env.current_state()
    summaries = []
    for role, index in (("chef", 0), ("assistant", 1)):
        player = state.players[index]
        held = player.held_object.name if player.held_object is not None else "nothing"
        summaries.append(f"{role}_holds={held}")
    return "; ".join(summaries)


def _role_context(
    role: str,
    order: str,
    state_string: str,
    timestep: int,
    action_guide: str,
    state_summary: str,
) -> str:
    recipe_line = order if role == "chef" else "unknown_to_assistant"
    return (
        f"role={role}\n"
        f"timestep={timestep}\n"
        f"recipe_context={recipe_line}\n"
        f"state=\n{state_string}\n"
        f"state_summary={state_summary}\n"
        f"action_guide={action_guide}\n"
    )


def run_pilot_experiment(config: PilotRunConfig) -> RunRecord:
    env = EnvironmentAdapter(layout=config.layout, horizon=config.horizon)
    evaluation = EvaluationAdapter(Path(config.results_root))
    snapshot = env.reset(config.order)
    executor = MacroActionExecutor(env)

    chef_agent = build_agent(config.chef)
    assistant_agent = build_agent(config.assistant)
    turns: list[TurnRecord] = []
    cumulative_score = 0.0
    last_messages = {"chef": None, "assistant": None}
    success = False

    for timestep in range(config.horizon):
        prompt_texts = {}
        raw_responses = {}
        parsed_responses = {}
        validator_errors = {"chef": [], "assistant": []}
        state_summary = " | ".join(
            [
                _recipe_requirement_summary(env, config.order),
                _player_summary(env),
                _counter_summary(env),
                _utensil_summary(env),
            ]
        )

        for role, agent in (("chef", chef_agent), ("assistant", assistant_agent)):
            teammate_role = "assistant" if role == "chef" else "chef"
            feedback = []
            parsed = None
            raw = ""
            prompt_text = ""
            invocation_error: AgentInvocationError | None = None
            teammate_message = last_messages[teammate_role]
            if teammate_role in parsed_responses:
                teammate_message = parsed_responses[teammate_role].say
            for _ in range(config.max_retries + 1):
                try:
                    result = agent.act(
                        snapshot.frame,
                        _role_context(
                            role,
                            config.order,
                            snapshot.state_string,
                            snapshot.timestep,
                            role_action_guide(role, env.mdp),
                            state_summary,
                        ),
                        teammate_message,
                        feedback,
                    )
                except AgentResponseFormatError as exc:
                    prompt_text = exc.prompt_text
                    raw = exc.raw_response
                    error_message = str(exc)
                    feedback = [error_message]
                    validator_errors[role].append(error_message)
                    continue
                except AgentInvocationError as exc:
                    prompt_text = exc.prompt_text
                    raw = exc.raw_response
                    error_message = str(exc)
                    feedback = [error_message]
                    validator_errors[role].append(error_message)
                    invocation_error = exc
                    continue
                raw = result.raw_response
                prompt_text = result.prompt_text
                parsed = result.parsed_response
                validation = validate_plan_string(parsed.plan, role=role, mdp=env.mdp)
                if validation.valid:
                    break
                feedback = validation.errors
                validator_errors[role].extend(validation.errors)
                if config.max_retries == 0:
                    break
            if parsed is None:
                if invocation_error is not None:
                    raise RuntimeError(
                        f"Agent {role} failed after {config.max_retries + 1} attempts: {invocation_error}"
                    ) from invocation_error
                parsed = AgentTurnResponse(
                    analysis="Failed to produce valid structured output after retries.",
                    plan="[NONE]",
                    say="[NOTHING]",
                )
            else:
                final_validation = validate_plan_string(parsed.plan, role=role, mdp=env.mdp)
                if not final_validation.valid:
                    validator_errors[role].extend(final_validation.errors)
                    parsed = parsed.model_copy(update={"plan": "[NONE]"})
            prompt_texts[role] = prompt_text
            raw_responses[role] = raw
            parsed_responses[role] = parsed

        for role in ("chef", "assistant"):
            executor.accept_proposal(role, parsed_responses[role].plan)
        chef_execution = executor.execute_current_intent("chef")
        assistant_execution = executor.execute_current_intent("assistant")
        validator_errors["chef"].extend(chef_execution.errors)
        validator_errors["assistant"].extend(assistant_execution.errors)
        joint_action = [
            chef_execution.macro_action,
            assistant_execution.macro_action,
        ]
        low_level_actions = [
            chef_execution.low_level_action,
            assistant_execution.low_level_action,
        ]
        snapshot, reward, done = env.step(
            tuple(low_level_actions),
            config.order,
            tuple(joint_action),
        )
        executor.finalize_step()
        cumulative_score += reward

        turn = TurnRecord(
            timestep=timestep,
            state_string=snapshot.state_string,
            prompt_texts=prompt_texts,
            raw_responses=raw_responses,
            parsed_responses=parsed_responses,
            validator_errors=validator_errors,
            joint_action=joint_action,
            low_level_actions=low_level_actions,
            score_delta=reward,
            cumulative_score=cumulative_score,
            order=config.order,
        )
        turns.append(turn)
        last_messages = {
            "chef": parsed_responses["chef"].say,
            "assistant": parsed_responses["assistant"].say,
        }
        if done:
            success = reward > 0 or success
            break

    record = RunRecord(
        run_name=config.run_name,
        layout=config.layout,
        order=config.order,
        level=config.level,
        success=success,
        horizon=config.horizon,
        turns=turns,
    )

    runs_dir = Path(config.results_root) / "runs" / config.run_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"{config.order}.json"
    run_path.write_text(json.dumps(record.model_dump(mode="json"), indent=2))

    evaluation_path = evaluation.evaluate(record)
    updated = record.model_copy(
        update={
            "benchmark_log_path": str((Path(config.results_root) / "legacy_logs" / config.run_name / config.order).resolve()),
            "evaluation_result_path": str(evaluation_path),
        }
    )
    run_path.write_text(json.dumps(updated.model_dump(mode="json"), indent=2))
    return updated
