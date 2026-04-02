from pathlib import Path

import pytest

from vision_overcooked.adapters import MacroActionExecutor
from vision_overcooked.adapters.environment import EnvironmentAdapter
from vision_overcooked.adapters.agent import OpenAIVisionAgent
from vision_overcooked.adapters.upstream_bridge import VisionBackedModule
from vision_overcooked.runner import run_pilot_experiment
from vision_overcooked.schemas import PilotRunConfig


def test_runner_executes_pilot_scaffold(tmp_path: Path):
    config = PilotRunConfig(
        run_name="pytest-pilot",
        layout="new_env",
        order="boiled_egg",
        level=1,
        horizon=2,
        max_retries=0,
        results_root=str(tmp_path),
        debug_capture_frames=False,
        persist_sampled_frames=False,
        chef={
            "role": "chef",
            "backend": "static",
            "model_name": "pytest-static",
            "static_response": {
                "analysis": "chef analysis",
                "plan": "[NONE]",
                "say": "[NOTHING]",
            },
        },
        assistant={
            "role": "assistant",
            "backend": "static",
            "model_name": "pytest-static",
            "static_response": {
                "analysis": "assistant analysis",
                "plan": "[NONE]",
                "say": "[NOTHING]",
            },
        },
    )
    result = run_pilot_experiment(config)
    assert result.order == "boiled_egg"
    assert result.evaluation_result_path is not None
    assert Path(result.evaluation_result_path).exists()


class _FakeResponse:
    def __init__(self, output_text: str):
        self.output_text = output_text


class _FakeResponsesAPI:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.outputs:
            raise AssertionError("No fake outputs remaining.")
        next_output = self.outputs.pop(0)
        if next_output == "__RAISE__":
            raise RuntimeError("temporary API failure")
        return _FakeResponse(next_output)


class _FakeClient:
    def __init__(self, outputs: list[str]):
        self.responses = _FakeResponsesAPI(outputs)


def test_runner_executes_openai_backend_with_mocked_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    clients = [
        _FakeClient(["Chef analysis: chef ok\nChef plan: [NONE]\nChef say: go"]),
        _FakeClient(["Assistant analysis: assistant ok\nAssistant plan: pickup(egg,ingredient_dispenser)\nAssistant say: [NOTHING]"]),
    ]

    def fake_create_client(self):
        return clients.pop(0)

    monkeypatch.setattr(OpenAIVisionAgent, "_create_client", fake_create_client)

    config = PilotRunConfig(
        run_name="pytest-openai-pilot",
        layout="new_env",
        order="boiled_egg",
        level=1,
        horizon=1,
        max_retries=0,
        results_root=str(tmp_path),
        debug_capture_frames=False,
        persist_sampled_frames=False,
        chef={
            "role": "chef",
            "backend": "openai_vision",
            "model_name": "gpt-4.1-mini",
        },
        assistant={
            "role": "assistant",
            "backend": "openai_vision",
            "model_name": "gpt-4.1-mini",
        },
    )
    result = run_pilot_experiment(config)
    assert result.turns[0].parsed_responses["assistant"].plan == "pickup(egg,ingredient_dispenser)"
    assert "Role:" in result.turns[0].prompt_texts["chef"]
    assert result.turns[0].joint_action[1] == "pickup(egg,ingredient_dispenser)"
    assert result.turns[0].low_level_actions[1] in {"NORTH", "SOUTH", "EAST", "WEST", "INTERACT", "STAY"}
    assert Path(result.evaluation_result_path).exists()


def test_runner_executes_upstream_vision_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class _FakeTurnResult:
        def __init__(self, role: str):
            self.low_level_action = "STAY"
            self.macro_action = "[NONE]"
            self.raw_response = (
                f"{role.title()} analysis: ok\n{role.title()} say: [NOTHING]\n{role.title()} plan: [NONE]"
            )
            self.prompt_text = "prompt"
            self.parsed_response = {
                "analysis": "ok",
                "plan": "[NONE]",
                "say": "[NOTHING]",
            }
            self.validator_errors = []

    class _FakeController:
        def __init__(self, env, chef, assistant):
            self.env = env

        def set_frame(self, frame):
            self.frame = frame

        def act(self, role: str):
            from vision_overcooked.schemas import AgentTurnResponse

            result = _FakeTurnResult(role)
            result.parsed_response = AgentTurnResponse.model_validate(result.parsed_response)
            return result

    monkeypatch.setattr("vision_overcooked.runner.UpstreamVisionController", _FakeController)

    config = PilotRunConfig(
        run_name="pytest-upstream-vision",
        controller="upstream_vision",
        layout="new_env",
        order="boiled_egg",
        level=1,
        horizon=1,
        max_retries=0,
        results_root=str(tmp_path),
        chef={
            "role": "chef",
            "backend": "openai_vision",
            "model_name": "gpt-4.1-mini",
        },
        assistant={
            "role": "assistant",
            "backend": "openai_vision",
            "model_name": "gpt-4.1-mini",
        },
    )
    result = run_pilot_experiment(config)
    assert result.turns[0].joint_action == ["[NONE]", "[NONE]"]
    assert result.turns[0].low_level_actions == ["STAY", "STAY"]
    assert Path(result.evaluation_result_path).exists()


def test_upstream_vision_module_sends_system_and_image(
    monkeypatch: pytest.MonkeyPatch,
):
    class _FakeResponse:
        def __init__(self, output_text: str):
            self.output_text = output_text

    class _FakeResponsesAPI:
        def __init__(self):
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return _FakeResponse(
                "Chef analysis: ok\nChef say: [NOTHING]\nChef plan: [NONE]"
            )

    class _FakeClient:
        def __init__(self):
            self.responses = _FakeResponsesAPI()

    class _FakeUpstreamModule:
        def __init__(self, role_messages, model, retrival_method, K):
            self.current_user_message = {"content": "Upstream user prompt"}
            self.instruction_head_list = role_messages

        def query_messages(self, rethink):
            return [
                {"role": "system", "content": "Upstream system prompt"},
                {"role": "user", "content": "Upstream user prompt"},
            ]

    class _Owner:
        current_frame = pytest.importorskip("numpy").zeros((2, 2, 3), dtype="uint8")

    fake_client = _FakeClient()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("vision_overcooked.adapters.upstream_bridge.OpenAI", lambda api_key: fake_client)

    module = VisionBackedModule(
        _FakeUpstreamModule,
        owner=_Owner(),
        role_messages=[{"role": "system", "content": ""}],
        model="gpt-4.1-mini",
        api_key_env="OPENAI_API_KEY",
    )
    monkeypatch.setattr(
        VisionBackedModule,
        "_frame_to_data_url",
        lambda self, frame: "data:image/png;base64,FAKE",
    )

    output, _ = module.query()
    assert output.startswith("Chef analysis:")
    call = fake_client.responses.calls[0]
    assert call["input"][0] == {"role": "system", "content": "Upstream system prompt"}
    user_content = call["input"][1]["content"]
    assert user_content[0] == {"type": "input_text", "text": "Upstream user prompt"}
    assert user_content[1]["type"] == "input_image"


def test_runner_retries_invalid_json_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    clients = [
        _FakeClient([
            "not json",
            "Chef analysis: chef retry ok\nChef plan: wait(1)\nChef say: waiting",
        ]),
        _FakeClient(["Assistant analysis: assistant ok\nAssistant plan: pickup(egg,ingredient_dispenser)\nAssistant say: [NOTHING]"]),
    ]

    def fake_create_client(self):
        return clients.pop(0)

    monkeypatch.setattr(OpenAIVisionAgent, "_create_client", fake_create_client)

    config = PilotRunConfig(
        run_name="pytest-openai-retry",
        layout="new_env",
        order="boiled_egg",
        level=1,
        horizon=1,
        max_retries=1,
        results_root=str(tmp_path),
        debug_capture_frames=False,
        persist_sampled_frames=False,
        chef={
            "role": "chef",
            "backend": "openai_vision",
            "model_name": "gpt-4.1-mini",
        },
        assistant={
            "role": "assistant",
            "backend": "openai_vision",
            "model_name": "gpt-4.1-mini",
        },
    )
    result = run_pilot_experiment(config)
    assert result.turns[0].parsed_responses["chef"].plan == "wait(1)"
    assert any("upstream text format" in msg for msg in result.turns[0].validator_errors["chef"])


def test_runner_falls_back_to_safe_noop_after_exhausted_parse_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    clients = [
        _FakeClient(["not json", "Chef analysis: chef invalid\nChef plan: pickup(egg)\nChef say: bad"]),
        _FakeClient(["Assistant analysis: assistant ok\nAssistant plan: pickup(egg,ingredient_dispenser)\nAssistant say: [NOTHING]"]),
    ]

    def fake_create_client(self):
        return clients.pop(0)

    monkeypatch.setattr(OpenAIVisionAgent, "_create_client", fake_create_client)

    config = PilotRunConfig(
        run_name="pytest-openai-fallback",
        layout="new_env",
        order="boiled_egg",
        level=1,
        horizon=1,
        max_retries=1,
        results_root=str(tmp_path),
        debug_capture_frames=False,
        persist_sampled_frames=False,
        chef={
            "role": "chef",
            "backend": "openai_vision",
            "model_name": "gpt-4.1-mini",
        },
        assistant={
            "role": "assistant",
            "backend": "openai_vision",
            "model_name": "gpt-4.1-mini",
        },
    )
    result = run_pilot_experiment(config)
    assert result.turns[0].parsed_responses["chef"].plan == "[NONE]"
    assert result.turns[0].joint_action[0] == "[NONE]"


def test_macro_executor_counts_down_wait():
    adapter = EnvironmentAdapter(horizon=4)
    adapter.reset("boiled_egg")
    executor = MacroActionExecutor(adapter)
    executor.accept_proposal("chef", "wait(2)")
    first = executor.execute_current_intent("chef")
    second = executor.execute_current_intent("chef")
    assert first.macro_action == "wait(2)"
    assert first.low_level_action == "STAY"
    assert second.completed is True


def test_macro_executor_pickup_intent_persists_until_item_is_acquired():
    adapter = EnvironmentAdapter(horizon=4)
    adapter.reset("boiled_egg")
    executor = MacroActionExecutor(adapter)
    executor.accept_proposal("assistant", "pickup(egg,ingredient_dispenser)")
    result = executor.execute_current_intent("assistant")
    assert result.macro_action == "pickup(egg,ingredient_dispenser)"
    assert result.low_level_action in {"NORTH", "SOUTH", "EAST", "WEST", "INTERACT", "STAY"}
    intent = executor.intent_state["assistant"].current_plan
    assistant = adapter.current_state().players[1]
    assert intent == "pickup(egg,ingredient_dispenser)" or (
        assistant.has_object() and assistant.get_object().name == "egg"
    )


def test_macro_executor_rejects_impossible_pickup_from_empty_counter():
    adapter = EnvironmentAdapter(horizon=4)
    adapter.reset("boiled_egg")
    executor = MacroActionExecutor(adapter)
    error = executor._validate_against_state("chef", "pickup(egg,counter)", adapter.current_state())
    assert error is not None
    assert "not available on any reachable counter" in error


def test_macro_executor_accepts_upstream_style_multi_step_plan_and_executes_first_action():
    adapter = EnvironmentAdapter(horizon=4)
    adapter.reset("boiled_egg")
    executor = MacroActionExecutor(adapter)
    first = "pickup(egg,ingredient_dispenser);place_obj_on_counter()".split(";")[0]
    executor.accept_proposal("assistant", first)
    result = executor.execute_current_intent("assistant")
    assert result.macro_action == "pickup(egg,ingredient_dispenser)"
