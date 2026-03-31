from pathlib import Path

from vision_overcooked.adapters.evaluation import EvaluationAdapter
from vision_overcooked.schemas import AgentTurnResponse, RunRecord, TurnRecord


def test_evaluation_adapter_exports_legacy_log(tmp_path: Path):
    adapter = EvaluationAdapter(tmp_path)
    turn = TurnRecord(
        timestep=0,
        state_string="test-state",
        raw_responses={"chef": "{}", "assistant": "{}"},
        parsed_responses={
            "chef": AgentTurnResponse(analysis="a", plan="[NONE]", say="[NOTHING]"),
            "assistant": AgentTurnResponse(analysis="b", plan="pickup(egg,ingredient_dispenser)", say="[NOTHING]"),
        },
        validator_errors={"chef": [], "assistant": []},
        joint_action=["[NONE]", "pickup(egg,ingredient_dispenser)"],
        low_level_actions=["STAY", "WEST"],
        score_delta=0,
        cumulative_score=0,
        order="boiled_egg",
    )
    record = RunRecord(
        run_name="test-run",
        layout="new_env",
        order="boiled_egg",
        level=1,
        success=False,
        horizon=1,
        turns=[turn],
    )
    path = adapter.export_legacy_log(record)
    assert path.exists()
    assert "experiment_" in path.name
