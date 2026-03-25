from pathlib import Path

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
