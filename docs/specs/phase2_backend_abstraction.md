# Phase 2: Backend Abstraction

## Goal

Make model providers swappable without changing the experiment loop.

This phase turns the initial OpenAI integration into a clean backend architecture so the project can support both hosted and local models without rewriting core logic.

## Why This Matters

- A quick prototype can become hard to replace if provider-specific logic leaks into the runner.
- The project will eventually need both hosted and local inference paths.
- Clear boundaries make debugging easier and keep experiments comparable across providers.

## What To Clean Up

- Separate prompt construction from model invocation.
- Separate model invocation from response parsing.
- Separate response parsing from action normalization.
- Keep one common backend interface used by the runner.
- Add config fields for backend-specific options such as model name, temperature, token limits, and vision/text mode.
- Improve observability by saving prompt payloads, validation failures, and retry counts.
- Add tests for malformed outputs and backend-specific edge cases.

## Target Shape

The runner should not need to know whether the backend is `static`, `openai`, or `local`.

The backend layer should be responsible for:

- receiving the agent context
- calling the model provider
- returning structured content in the common schema

## Success Criteria

- Changing providers is mostly a config change.
- The same pilot loop works for multiple backends.
- Debugging artifacts are rich enough to compare failures across providers.
- The code is ready for a local-model backend without major refactoring.
