# Phase 3: Local SCC Backend

## Goal

Run real experiments on a local model stack suitable for SCC jobs and larger-scale evaluation.

This phase is about reproducibility, cost control, and scaling beyond development-time hosted API usage.

## Why This Comes After Phase 1 and Phase 2

- Local inference adds operational complexity on top of the agent logic.
- If prompts, parsing, and action mapping are still unstable, SCC debugging will be slow and noisy.
- A shared backend interface makes the local path much easier to implement correctly.

## What To Build

- Add a `local` backend that follows the same contract as the `openai` backend.
- Support the selected SCC inference path, such as a local Hugging Face model, a vLLM server, or another cluster-hosted endpoint.
- Keep job-friendly config and output behavior for batch runs.
- Add smoke-test configurations for short SCC validation runs before large experiments.

## Requirements

- The local backend must return the same structured `analysis`, `plan`, and `say` fields.
- Logging and metrics should remain comparable to the OpenAI development path.
- Failures should still degrade cleanly to retries or safe fallback behavior.

## Success Criteria

- The same experiment scaffold works with `backend: local`.
- Local runs produce the same kinds of artifacts as hosted runs.
- Small SCC smoke tests pass before larger experiment batches are launched.
- Final experiments can be run without changing the core runner.

## Practical Strategy

- Start with a tiny smoke-test config on SCC.
- Confirm model loading, inference latency, and output formatting.
- Only then scale to repeated runs, broader task coverage, and benchmark comparisons.
