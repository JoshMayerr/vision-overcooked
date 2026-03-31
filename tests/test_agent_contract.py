import pytest

from vision_overcooked.adapters.agent import OpenAIVisionAgent, validate_plan_string
from vision_overcooked.schemas import AgentConfig, AgentTurnResponse


def test_agent_turn_response_requires_strict_json():
    payload = '{"analysis":"ok","plan":"[NONE]","say":"[NOTHING]"}'
    parsed = AgentTurnResponse.from_raw_json(payload)
    assert parsed.plan == "[NONE]"


def test_agent_turn_response_rejects_non_json():
    with pytest.raises(ValueError, match="valid JSON"):
        AgentTurnResponse.from_raw_json("not json")


def test_agent_turn_response_rejects_missing_required_field():
    with pytest.raises(ValueError, match="schema validation"):
        AgentTurnResponse.from_raw_json('{"analysis":"ok","plan":"[NONE]"}')


@pytest.mark.parametrize("plan", ["[NONE]", "NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT", "wait(1)"])
def test_validator_accepts_supported_plan_strings(plan: str):
    result = validate_plan_string(plan)
    assert result.valid is True


def test_validator_rejects_unsupported_macro_action():
    result = validate_plan_string("pickup(onion,counter)")
    assert result.valid is False
    assert result.execution_action == "STAY"


def test_openai_agent_builds_prompt_with_context_and_feedback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = OpenAIVisionAgent(
        AgentConfig(role="chef", backend="openai_vision", model_name="gpt-4.1-mini")
    )

    prompt_text = agent._build_prompt_text(
        "role=chef\ntimestep=0\nstate=demo",
        "bring the egg",
        ["Plan cannot be empty."],
    )

    assert "role=chef" in prompt_text
    assert "bring the egg" in prompt_text
    assert "Plan cannot be empty." in prompt_text
    assert "analysis, plan, say" in prompt_text


def test_openai_agent_request_input_includes_image(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = OpenAIVisionAgent(
        AgentConfig(role="assistant", backend="openai_vision", model_name="gpt-4.1-mini")
    )
    frame = pytest.importorskip("numpy").zeros((4, 4, 3), dtype="uint8")

    request_input = agent._build_request_input(frame, "prompt")

    assert request_input[0]["content"][0] == {"type": "input_text", "text": "prompt"}
    image_part = request_input[0]["content"][1]
    assert image_part["type"] == "input_image"
    assert str(image_part["image_url"]).startswith("data:image/png;base64,")


def test_static_backend_requires_static_response():
    with pytest.raises(ValueError, match="Static agents require a static_response"):
        AgentConfig(role="chef", backend="static", model_name="pytest-static")


def test_openai_backend_does_not_require_static_response():
    config = AgentConfig(role="chef", backend="openai_vision", model_name="gpt-4.1-mini")
    assert config.static_response is None
