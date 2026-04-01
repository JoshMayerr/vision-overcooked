from __future__ import annotations

import base64
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from openai import OpenAI

from vision_overcooked.schemas import AgentConfig, AgentTurnResponse

@dataclass
class AgentActResult:
    prompt_text: str
    raw_response: str
    parsed_response: AgentTurnResponse


class AgentResponseFormatError(ValueError):
    def __init__(self, message: str, *, prompt_text: str, raw_response: str):
        super().__init__(message)
        self.prompt_text = prompt_text
        self.raw_response = raw_response


class AgentInvocationError(RuntimeError):
    def __init__(self, message: str, *, prompt_text: str, raw_response: str = ""):
        super().__init__(message)
        self.prompt_text = prompt_text
        self.raw_response = raw_response


class AgentAdapter(Protocol):
    role: str
    model_name: str

    def act(
        self,
        frame: np.ndarray,
        text_context: str,
        teammate_message: str | None,
        validator_feedback: list[str] | None = None,
    ) -> AgentActResult:
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
    ) -> AgentActResult:
        raw = json.dumps(self.response.model_dump(mode="json"))
        prompt_text = (
            f"STATIC BACKEND\nrole={self.role}\n"
            "No model prompt was sent. Using configured static_response."
        )
        return AgentActResult(prompt_text=prompt_text, raw_response=raw, parsed_response=self.response)


class OpenAIVisionAgent:
    PLAN_INSTRUCTIONS = (
        "Return strict JSON with exactly three string keys: analysis, plan, say.\n"
        "Do not wrap the JSON in markdown or any extra text.\n"
        "The plan must be one or more benchmark macro-actions written as strings separated by semicolons.\n"
        "Supported forms are [NONE], wait(n), pickup(obj,source), put_obj_in_utensil(utensil), fill_dish_with_food(utensil), place_obj_on_counter(), deliver_soup(), check_recipe(), and role-appropriate utensil operations such as cook(pot0).\n"
        "Use say for coordination. If chef cannot access an ingredient directly, chef should ask assistant to fetch it from ingredient_dispenser and place it on the counter. Assistant should follow direct requests from chef when they are legal."
    )

    def __init__(self, config: AgentConfig):
        self.role = config.role
        self.model_name = config.model_name
        self.api_key_env = config.api_key_env or "OPENAI_API_KEY"
        self.client = self._create_client()

    def _create_client(self) -> OpenAI:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise AgentInvocationError(
                f"Missing OpenAI API key in environment variable {self.api_key_env}.",
                prompt_text="",
            )
        return OpenAI(api_key=api_key)

    def _frame_to_data_url(self, frame: np.ndarray) -> str:
        import pygame

        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            pygame.image.save(surface, tmp.name)
            payload = base64.b64encode(tmp.read()).decode("utf-8")
        return f"data:image/png;base64,{payload}"

    def _build_prompt_text(
        self,
        text_context: str,
        teammate_message: str | None,
        validator_feedback: list[str] | None,
    ) -> str:
        teammate_text = teammate_message or "[NOTHING]"
        feedback = "\n".join(validator_feedback or []) or "[NONE]"
        return (
            f"{self.PLAN_INSTRUCTIONS}\n\n"
            f"Role:\n{self.role}\n\n"
            f"Context:\n{text_context}\n"
            f"Teammate message:\n{teammate_text}\n\n"
            f"Validator feedback from previous attempts:\n{feedback}\n"
        )

    def _build_request_input(
        self,
        frame: np.ndarray,
        prompt_text: str,
    ) -> list[dict[str, object]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image", "image_url": self._frame_to_data_url(frame)},
                ],
            }
        ]

    def _response_text_format(self) -> dict[str, object]:
        return {
            "format": {
                "type": "json_schema",
                "name": "overcooked_turn_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "analysis": {"type": "string"},
                        "plan": {"type": "string"},
                        "say": {"type": "string"},
                    },
                    "required": ["analysis", "plan", "say"],
                    "additionalProperties": False,
                },
            }
        }

    def _parse_response(self, raw_response: str, prompt_text: str) -> AgentTurnResponse:
        try:
            return AgentTurnResponse.from_raw_json(raw_response)
        except ValueError as exc:
            raise AgentResponseFormatError(
                str(exc),
                prompt_text=prompt_text,
                raw_response=raw_response,
            ) from exc

    def act(
        self,
        frame: np.ndarray,
        text_context: str,
        teammate_message: str | None,
        validator_feedback: list[str] | None = None,
    ) -> AgentActResult:
        prompt_text = self._build_prompt_text(text_context, teammate_message, validator_feedback)
        request_input = self._build_request_input(frame, prompt_text)
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=request_input,
                text=self._response_text_format(),
            )
        except Exception as exc:
            raise AgentInvocationError(
                f"OpenAI Responses API call failed: {exc}",
                prompt_text=prompt_text,
            ) from exc
        raw_response = response.output_text
        parsed_response = self._parse_response(raw_response, prompt_text)
        return AgentActResult(
            prompt_text=prompt_text,
            raw_response=raw_response,
            parsed_response=parsed_response,
        )


OpenAICompatibleVisionAgent = OpenAIVisionAgent


def build_agent(config: AgentConfig) -> AgentAdapter:
    if config.backend == "static":
        return StaticJSONAgent(config)
    return OpenAIVisionAgent(config)
