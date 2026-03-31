from __future__ import annotations

import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

from vision_overcooked.paths import EVALUATIONS_DIR, LEGACY_LOGS_DIR, UPSTREAM_SRC
from vision_overcooked.schemas import RunRecord

if not hasattr(np, "Inf"):
    np.Inf = np.inf


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_legacy_eval_module():
    module_name = "_vision_overcooked_legacy_eval_utils"
    module_path = UPSTREAM_SRC / "eval_utils.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy evaluation module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class EvaluationAdapter:
    def __init__(self, results_root: Path | None = None):
        root = results_root or Path("results")
        self.legacy_root = Path(root) / "legacy_logs"
        self.evaluation_root = Path(root) / "evaluations"
        self.legacy_root.mkdir(parents=True, exist_ok=True)
        self.evaluation_root.mkdir(parents=True, exist_ok=True)

    def export_legacy_log(self, record: RunRecord) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        order_dir = self.legacy_root / record.run_name / record.order
        order_dir.mkdir(parents=True, exist_ok=True)
        log_path = order_dir / f"experiment_{timestamp}_{record.order}.json"

        total_action_list = [[], []]
        content = []
        total_timestamp = []
        for turn in record.turns:
            total_timestamp.append(turn.timestep)
            for agent_index, role in enumerate(["chef", "assistant"]):
                plan = turn.parsed_responses[role].plan.strip()
                if plan not in {"[NONE]", "wait(1)", "STAY"}:
                    total_action_list[agent_index].append(
                        {"timestamp": turn.timestep, "action": plan}
                    )
            content.append(
                {
                    "timestamp": turn.timestep,
                    "order_list": [record.order],
                    "actions": [
                        turn.parsed_responses["chef"].plan,
                        turn.parsed_responses["assistant"].plan,
                    ],
                    "map": turn.state_string,
                    "statistical_data": {
                        "score": turn.score_delta,
                        "communication": [
                            {"call": 0, "turn": [], "token": []},
                            {"call": 0, "turn": [], "token": []},
                        ],
                        "error": [
                            {
                                "format_error": {"error_num": 0, "error_message": []},
                                "validator_error": {
                                    "error_num": len(turn.validator_errors["chef"]),
                                    "error_message": turn.validator_errors["chef"],
                                },
                            },
                            {
                                "format_error": {"error_num": 0, "error_message": []},
                                "validator_error": {
                                    "error_num": len(turn.validator_errors["assistant"]),
                                    "error_message": turn.validator_errors["assistant"],
                                },
                            },
                        ],
                        "error_correction": [
                            {
                                "format_correction": {
                                    "correction_num": 0,
                                    "correction_tokens": [],
                                },
                                "validator_correction": {
                                    "correction_num": 0,
                                    "reflection_obtain": [],
                                    "correction_tokens": [],
                                },
                            },
                            {
                                "format_correction": {
                                    "correction_num": 0,
                                    "correction_tokens": [],
                                },
                                "validator_correction": {
                                    "correction_num": 0,
                                    "reflection_obtain": [],
                                    "correction_tokens": [],
                                },
                            },
                        ],
                    },
                    "content": {
                        "observation": [[], []],
                        "reflection": [
                            [turn.parsed_responses["chef"].analysis],
                            [turn.parsed_responses["assistant"].analysis],
                        ],
                        "content": [[], []],
                        "action_list": [
                            [turn.parsed_responses["chef"].plan],
                            [turn.parsed_responses["assistant"].plan],
                        ],
                        "original_log": json.dumps(turn.raw_responses),
                    },
                }
            )

        payload = {
            "total_timestamp": total_timestamp,
            "total_order_finished": [record.order] if record.success else [],
            "total_score": record.turns[-1].cumulative_score if record.turns else 0,
            "total_action_list": total_action_list,
            "content": content,
        }
        log_path.write_text(json.dumps(payload, indent=2))
        return log_path

    def evaluate(self, record: RunRecord) -> Path:
        log_path = self.export_legacy_log(record)
        log_dir = log_path.parent.resolve()
        save_dir = (self.evaluation_root / record.run_name / record.order).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        with _temporary_cwd(UPSTREAM_SRC):
            legacy_module = _load_legacy_eval_module()
            exp_log = legacy_module.ExpLog(str(log_dir))
            evaluation = legacy_module.Evaluation(
                order_name_list=[record.order] * len(exp_log),
                exp_log=exp_log,
            )
            legacy_result = evaluation.evaluate(str(save_dir))

        normalized = {
            "run_name": record.run_name,
            "order": record.order,
            "level": record.level,
            "success_rate": legacy_result[record.order]["task_metrics"]["success_rate"],
            "tes_f1": {
                "chef": legacy_result[record.order]["average"]["similarity_and_redundancy"]["agent_0"]["mean_f1"],
                "assistant": legacy_result[record.order]["average"]["similarity_and_redundancy"]["agent_1"]["mean_f1"],
                "overall": legacy_result[record.order]["average"]["similarity_and_redundancy"]["overall"]["mean_f1"],
            },
            "initiating_capability": legacy_result[record.order]["statistic"]["initiate_collaboration"],
            "responding_capability": legacy_result[record.order]["statistic"]["respond_collaboration"],
            "legacy_log_path": str(log_path),
        }
        normalized_path = save_dir / "normalized_metrics.json"
        normalized_path.write_text(json.dumps(normalized, indent=2))
        return normalized_path
