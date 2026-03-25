from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = ROOT / "third_party" / "collab_overcooked"
UPSTREAM_SRC = THIRD_PARTY_ROOT / "src"
UPSTREAM_PROMPTS = UPSTREAM_SRC / "prompts"
UPSTREAM_RECIPE_DIR = UPSTREAM_PROMPTS / "recipe"
UPSTREAM_REFERENCE_DIR = UPSTREAM_PROMPTS / "reference"
DOCS_DIR = ROOT / "docs"
PAPERS_DIR = DOCS_DIR / "papers"
DATA_EXTERNAL_DIR = ROOT / "data" / "external"
EXPERIMENT_CONFIG_DIR = ROOT / "experiments" / "configs"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
LEGACY_LOGS_DIR = RESULTS_DIR / "legacy_logs"
EVALUATIONS_DIR = RESULTS_DIR / "evaluations"
