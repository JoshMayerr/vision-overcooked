# Phase 1: OpenAI Backend

## Goal

Prove that a real model can run through the full pipeline end to end.

This phase is about development speed, not final benchmarking. The purpose is to validate the agent contract, prompt structure, parsing, retries, and action execution before adding the extra complexity of local inference.

## Why Start Here

- Hosted inference is faster to integrate and debug than a local cluster workflow.
- Most early failures will come from prompt design, schema violations, response parsing, and action mapping rather than model hosting.
- A working OpenAI path gives the project a fast feedback loop for improving the runner and logs.

## What To Build

- Add an `openai` backend alongside the existing `static` backend.
- Implement prompt construction for both `chef` and `assistant`.
- Add an API client layer for model calls.
- Parse responses into the strict `analysis`, `plan`, and `say` schema.
- Retry when outputs are malformed or use unsupported plans.
- Log raw prompts and raw model responses for debugging.

## Success Criteria

- `run-pilot` completes with model-generated outputs.
- At least some turns produce actions other than `STAY`.
- Invalid outputs are handled gracefully through retries or safe fallback behavior.
- A failed run is easy to inspect from saved logs.

## Notes

- It is acceptable to begin with a text-only OpenAI agent before adding vision input.
- OpenAI results should be treated as a development scaffold, not necessarily the final experiment setup.
