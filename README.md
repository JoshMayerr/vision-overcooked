# Vision Overcooked

`vision-overcooked` is a project-owned research scaffold for comparing visually grounded agents against text-only coordination baselines in the Collab-Overcooked environment.

## Repo layout

- `third_party/collab_overcooked/`: preserved upstream benchmark and vendored Overcooked environment.
- `src/vision_overcooked/`: project-owned adapters, runner, schemas, and CLI.
- `experiments/configs/`: declarative pilot configs.
- `data/external/`: literature-only baseline datasets and provenance.
- `results/`: generated run logs, benchmark-compatible exports, and evaluation outputs.
- `docs/`: paper assets and implementation notes.

## What is implemented

- A root `uv` project with the upstream environment wired in as an editable dependency.
- A strict JSON agent contract with `analysis`, `plan`, and `say`.
- A project-owned environment adapter with a lightweight frame renderer.
- A benchmark compatibility layer that exports run logs into the legacy Collab-Overcooked evaluation format.
- A pilot runner that preserves asymmetric task knowledge and can execute safe no-op pilot runs.
- Seed literature-baseline records with canonical schema and provenance fields.

## Quick start

Install and sync with `uv`:

```bash
uv sync
```

Validate the literature baseline seed file:

```bash
uv run vision-overcooked validate-baselines
```

List benchmark tasks detected from the preserved prompt assets:

```bash
uv run vision-overcooked list-tasks
```

Run the pilot scaffold:

```bash
uv run vision-overcooked run-pilot --config experiments/configs/pilot_qwen_vl.yaml
```

The pilot runner writes:

- raw turn logs under `results/runs/`
- benchmark-compatible legacy exports under `results/legacy_logs/`
- normalized evaluation outputs under `results/evaluations/`

## Notes

- The upstream benchmark is intentionally isolated under `third_party/` and is not the main development surface.
- The default pilot uses safe no-op execution for unsupported or absent model outputs so the environment, logging, and evaluation pipeline can be exercised before full action execution is implemented.
- The baseline dataset is intentionally lightweight in this pass; the schema and ingestion path are in place, while full table extraction from the paper can be filled in incrementally.
