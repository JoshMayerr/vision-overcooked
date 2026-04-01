import pytest

from vision_overcooked.adapters import validate_plan_string
from vision_overcooked.adapters.agent import OpenAIVisionAgent
from vision_overcooked.adapters.environment import EnvironmentAdapter
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


@pytest.mark.parametrize(
    "role, plan",
    [
        ("chef", "[NONE]"),
        ("chef", "wait(1)"),
        ("chef", "pickup(egg,counter)"),
        ("chef", "put_obj_in_utensil(pot0)"),
        ("chef", "cook(pot0)"),
        ("assistant", "pickup(egg,ingredient_dispenser)"),
        ("assistant", "place_obj_on_counter()"),
        ("assistant", "cut(chopping_board0)"),
    ],
)
def test_validator_accepts_supported_plan_strings(role: str, plan: str):
    adapter = EnvironmentAdapter()
    result = validate_plan_string(plan, role=role, mdp=adapter.mdp)
    assert result.valid is True


def test_validator_rejects_unsupported_macro_action():
    adapter = EnvironmentAdapter()
    result = validate_plan_string("SOUTH", role="chef", mdp=adapter.mdp)
    assert result.valid is False
    assert "Low-level movement actions are unsupported" in result.errors[0]


def test_validator_rejects_role_incompatible_macro_action():
    adapter = EnvironmentAdapter()
    result = validate_plan_string("cook(pot0)", role="assistant", mdp=adapter.mdp)
    assert result.valid is False


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
    assert "one or more benchmark macro-actions" in prompt_text


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
