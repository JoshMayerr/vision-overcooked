from __future__ import annotations

import json
from pathlib import Path

from .paths import DATA_EXTERNAL_DIR
from .schemas import LiteratureBaselineRecord

BASELINE_PATH = DATA_EXTERNAL_DIR / "collab_overcooked_literature_baselines.jsonl"


def load_literature_baselines(
    path: Path = BASELINE_PATH,
) -> list[LiteratureBaselineRecord]:
    records: list[LiteratureBaselineRecord] = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(LiteratureBaselineRecord.model_validate_json(line))
    return records


def validate_literature_baselines(path: Path = BASELINE_PATH) -> list[dict[str, object]]:
    return [record.model_dump(mode="json") for record in load_literature_baselines(path)]


def baseline_summary(path: Path = BASELINE_PATH) -> dict[str, object]:
    records = load_literature_baselines(path)
    return {
        "path": str(path),
        "record_count": len(records),
        "models": sorted(record.model_name for record in records),
    }
