from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich import print

from vision_overcooked.baselines import baseline_summary, validate_literature_baselines
from vision_overcooked.paths import EXPERIMENT_CONFIG_DIR
from vision_overcooked.runner import run_pilot_experiment
from vision_overcooked.schemas import PilotRunConfig
from vision_overcooked.adapters.environment import EnvironmentAdapter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vision Overcooked project scaffold")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-baselines")
    validate.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Optional override for the literature baseline JSONL path.",
    )

    list_tasks = subparsers.add_parser("list-tasks")
    list_tasks.add_argument("--layout", default="new_env")

    run = subparsers.add_parser("run-pilot")
    run.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_CONFIG_DIR / "pilot_qwen_vl.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-baselines":
        records = validate_literature_baselines(args.path) if args.path else validate_literature_baselines()
        print(json.dumps({"summary": baseline_summary(args.path) if args.path else baseline_summary(), "records": records}, indent=2))
        return

    if args.command == "list-tasks":
        adapter = EnvironmentAdapter(layout=args.layout)
        print(json.dumps(adapter.available_tasks(), indent=2))
        return

    if args.command == "run-pilot":
        config = PilotRunConfig.from_yaml(args.config)
        record = run_pilot_experiment(config)
        print(json.dumps(record.model_dump(mode="json"), indent=2))
        return


if __name__ == "__main__":
    main()
