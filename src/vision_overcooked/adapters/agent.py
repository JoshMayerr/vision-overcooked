from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from openai import OpenAI

from vision_overcooked.schemas import AgentConfig, AgentTurnResponse

PLAN_PATTERN = re.compile(r"^\[NONE\]$|^wait\(\d+\)$|^(NORTH|SOUTH|EAST|WEST|STAY|INTERACT)$")


@dataclass
class PlanValidationResult:
    plan: str
    valid: bool
    errors: list[str]
    execution_action: str
    benchmark_action: str | None


def validate_plan_string(plan: str) -> PlanValidationResult:
    normalized = plan.strip()
    if not normalized:
        return PlanValidationResult(
            plan=plan,
            valid=False,
            errors=["Plan cannot be empty."],
            execution_action="STAY",
            benchmark_action=None,
        )
    if PLAN_PATTERN.match(normalized):
        if normalized in {"[NONE]", "wait(1)", "wait(2)", "wait(3)"} or normalized.startswith("wait("):
            return PlanValidationResult(
                plan=normalized,
                valid=True,
                errors=[],
                execution_action="STAY",
                benchmark_action=None,
            )
        benchmark_action = None if normalized == "STAY" else normalized
        return PlanValidationResult(
            plan=normalized,
            valid=True,
            errors=[],
            execution_action=normalized,
            benchmark_action=benchmark_action,
        )
    return PlanValidationResult(
        plan=normalized,
        valid=False,
        errors=[
            "Unsupported plan string. Use one of [NONE], wait(n), NORTH, SOUTH, EAST, WEST, STAY, or INTERACT."
        ],
        execution_action="STAY",
        benchmark_action=None,
    )


class AgentAdapter(Protocol):
    role: str
    model_name: str

    def act(
        self,
        frame: np.ndarray,
        text_context: str,
        teammate_message: str | None,
        validator_feedback: list[str] | None = None,
    ) -> tuple[str, AgentTurnResponse]:
        ...


class StaticJSONAgent:
    def __init__(self, config: AgentConfig):
        self.role = config.role
        self.model_name = config.model_name
        self.response = config.static_response
        assert self.response is not None

    def act(
        self,
        frame: np.ndarray,
        text_context: str,
        teammate_message: str | None,
        validator_feedback: list[str] | None = None,
    ) -> tuple[str, AgentTurnResponse]:
        raw = json.dumps(self.response.model_dump(mode="json"))
        return raw, self.response


class OpenAICompatibleVisionAgent:
    def __init__(self, config: AgentConfig):
        self.role = config.role
        self.model_name = config.model_name
        self.endpoint = config.endpoint or "http://localhost:8000/v1"
        api_key = os.environ.get(config.api_key_env or "OPENAI_API_KEY", "local-dev-token")
        self.client = OpenAI(base_url=self.endpoint, api_key=api_key)

    def _frame_to_data_url(self, frame: np.ndarray) -> str:
        import pygame

        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        encoded = pygame.image.tobytes(surface, "RGB")
        payload = base64.b64encode(encoded).decode("utf-8")
        return f"data:image/raw-rgb;base64,{payload}"

    def act(
        self,
        frame: np.ndarray,
        text_context: str,
        teammate_message: str | None,
        validator_feedback: list[str] | None = None,
    ) -> tuple[str, AgentTurnResponse]:
        feedback = "\n".join(validator_feedback or [])
        teammate_text = teammate_message or "[NOTHING]"
        prompt = (
            "Return strict JSON with keys analysis, plan, say.\n"
            f"Context:\n{text_context}\n"
            f"Teammate message:\n{teammate_text}\n"
            f"Validator feedback:\n{feedback or '[NONE]'}\n"
        )
        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": self._frame_to_data_url(frame)},
                    ],
                }
            ],
        )
        raw = response.output_text
        return raw, AgentTurnResponse.from_raw_json(raw)


def build_agent(config: AgentConfig) -> AgentAdapter:
    if config.backend == "static":
        return StaticJSONAgent(config)
    return OpenAICompatibleVisionAgent(config)
