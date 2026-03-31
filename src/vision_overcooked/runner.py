from __future__ import annotations

import json
from pathlib import Path

from vision_overcooked.adapters import (
    AgentInvocationError,
    AgentResponseFormatError,
    EnvironmentAdapter,
    EvaluationAdapter,
    build_agent,
    validate_plan_string,
)
from vision_overcooked.schemas import AgentTurnResponse, PilotRunConfig, RunRecord, TurnRecord


def _role_context(role: str, order: str, state_string: str, timestep: int) -> str:
    recipe_line = order if role == "chef" else "unknown_to_assistant"
    return (
        f"role={role}\n"
        f"timestep={timestep}\n"
        f"recipe_context={recipe_line}\n"
        f"state=\n{state_string}\n"
    )


def run_pilot_experiment(config: PilotRunConfig) -> RunRecord:
    env = EnvironmentAdapter(layout=config.layout, horizon=config.horizon)
    evaluation = EvaluationAdapter(Path(config.results_root))
    snapshot = env.reset(config.order)

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

        for role, agent in (("chef", chef_agent), ("assistant", assistant_agent)):
            teammate_role = "assistant" if role == "chef" else "chef"
            feedback = []
            parsed = None
            raw = ""
            prompt_text = ""
            invocation_error: AgentInvocationError | None = None
            for _ in range(config.max_retries + 1):
                try:
                    result = agent.act(
                        snapshot.frame,
                        _role_context(role, config.order, snapshot.state_string, snapshot.timestep),
                        last_messages[teammate_role],
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
                validation = validate_plan_string(parsed.plan)
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
            prompt_texts[role] = prompt_text
            raw_responses[role] = raw
            parsed_responses[role] = parsed

        chef_validation = validate_plan_string(parsed_responses["chef"].plan)
        assistant_validation = validate_plan_string(parsed_responses["assistant"].plan)
        joint_action = [
            chef_validation.execution_action,
            assistant_validation.execution_action,
        ]
        snapshot, reward, done = env.step(tuple(joint_action), config.order)
        cumulative_score += reward

        turn = TurnRecord(
            timestep=timestep,
            state_string=snapshot.state_string,
            prompt_texts=prompt_texts,
            raw_responses=raw_responses,
            parsed_responses=parsed_responses,
            validator_errors=validator_errors,
            joint_action=joint_action,
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
