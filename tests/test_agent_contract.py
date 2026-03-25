import pytest

from vision_overcooked.adapters.agent import validate_plan_string
from vision_overcooked.schemas import AgentTurnResponse


def test_agent_turn_response_requires_strict_json():
    payload = '{"analysis":"ok","plan":"[NONE]","say":"[NOTHING]"}'
    parsed = AgentTurnResponse.from_raw_json(payload)
    assert parsed.plan == "[NONE]"


@pytest.mark.parametrize("plan", ["[NONE]", "NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT", "wait(1)"])
def test_validator_accepts_supported_plan_strings(plan: str):
    result = validate_plan_string(plan)
    assert result.valid is True


def test_validator_rejects_unsupported_macro_action():
    result = validate_plan_string("pickup(onion,counter)")
    assert result.valid is False
    assert result.execution_action == "STAY"
