from pathlib import Path

import pytest

from vision_overcooked.adapters.agent import OpenAIVisionAgent
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
        _FakeClient(['{"analysis":"chef ok","plan":"NORTH","say":"go"}']),
        _FakeClient(['{"analysis":"assistant ok","plan":"STAY","say":"[NOTHING]"}']),
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
    assert result.turns[0].parsed_responses["chef"].plan == "NORTH"
    assert "Role:" in result.turns[0].prompt_texts["chef"]
    assert Path(result.evaluation_result_path).exists()


def test_runner_retries_invalid_json_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    clients = [
        _FakeClient([
            "not json",
            '{"analysis":"chef retry ok","plan":"WEST","say":"moving"}',
        ]),
        _FakeClient(['{"analysis":"assistant ok","plan":"STAY","say":"[NOTHING]"}']),
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
    assert result.turns[0].parsed_responses["chef"].plan == "WEST"
    assert any("valid JSON" in msg for msg in result.turns[0].validator_errors["chef"])


def test_runner_falls_back_to_safe_noop_after_exhausted_parse_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    clients = [
        _FakeClient(["not json", "still not json"]),
        _FakeClient(['{"analysis":"assistant ok","plan":"STAY","say":"[NOTHING]"}']),
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
    assert result.turns[0].joint_action[0] == "STAY"
