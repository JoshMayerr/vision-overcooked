from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


class LevelMetrics(BaseModel):
    level: int = Field(ge=1, le=6)
    success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    progress_completeness: float | None = Field(default=None, ge=0.0, le=1.0)
    initiating_capability: float | None = Field(default=None, ge=0.0, le=1.0)
    responding_capability: float | None = Field(default=None, ge=0.0, le=1.0)


class LiteratureBaselineRecord(BaseModel):
    source_id: str
    source_title: str
    source_path: str
    extraction_status: str
    model_name: str
    model_family: str
    setting: str
    notes: str = ""
    levels: list[LevelMetrics] = Field(default_factory=list)


class AgentTurnResponse(BaseModel):
    analysis: str
    plan: str
    say: str

    @classmethod
    def from_raw_json(cls, raw: str) -> "AgentTurnResponse":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Agent output must be valid JSON: {exc}") from exc
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Agent output failed schema validation: {exc}") from exc


class AgentConfig(BaseModel):
    role: Literal["chef", "assistant"]
    backend: Literal["static", "openai_vision"] = "static"
    model_name: str
    static_response: AgentTurnResponse | None = None
    api_key_env: str | None = None

    @model_validator(mode="after")
    def validate_static_response(self) -> "AgentConfig":
        if self.backend == "static" and self.static_response is None:
            raise ValueError("Static agents require a static_response.")
        return self


class PilotRunConfig(BaseModel):
    run_name: str
    layout: str = "new_env"
    order: str
    level: int = Field(ge=1, le=6)
    horizon: int = Field(default=12, ge=1)
    max_retries: int = Field(default=1, ge=0, le=5)
    results_root: str = "results"
    debug_capture_frames: bool = False
    persist_sampled_frames: bool = False
    chef: AgentConfig
    assistant: AgentConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "PilotRunConfig":
        import yaml

        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data)


class TurnRecord(BaseModel):
    timestep: int
    state_string: str
    prompt_texts: dict[str, str] = Field(default_factory=dict)
    raw_responses: dict[str, str]
    parsed_responses: dict[str, AgentTurnResponse]
    validator_errors: dict[str, list[str]]
    joint_action: list[str]
    score_delta: float
    cumulative_score: float
    order: str
    frame_path: str | None = None


class RunRecord(BaseModel):
    run_name: str
    layout: str
    order: str
    level: int
    success: bool
    horizon: int
    turns: list[TurnRecord]
    benchmark_log_path: str | None = None
    evaluation_result_path: str | None = None
