from __future__ import annotations

import base64
import importlib
import os
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI
from overcooked_ai_py.planning.planners import MediumLevelActionManager

from vision_overcooked.paths import UPSTREAM_PROMPTS, UPSTREAM_SRC
from vision_overcooked.schemas import AgentConfig, AgentTurnResponse


_UPSTREAM_CACHE: tuple[object, object] | None = None
_UPSTREAM_KEYFILE = Path(tempfile.gettempdir()) / "vision_overcooked_upstream_openai_key.txt"


def _planner_params(mdp) -> dict[str, object]:
    counters = list(mdp.terrain_pos_dict["X"])
    return {
        "start_orientations": False,
        "wait_allowed": True,
        "counter_goals": counters,
        "counter_drop": counters,
        "counter_pickup": counters,
        "same_motion_goals": True,
    }


def _load_upstream_modules() -> tuple[object, object]:
    global _UPSTREAM_CACHE
    if _UPSTREAM_CACHE is not None:
        return _UPSTREAM_CACHE

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if str(UPSTREAM_SRC) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_SRC))
    if "pkg_resources" not in sys.modules:
        pkg_resources = types.ModuleType("pkg_resources")

        class _Distribution:
            version = "0.0.1"

        pkg_resources.get_distribution = lambda _name: _Distribution()
        sys.modules["pkg_resources"] = pkg_resources
    _UPSTREAM_KEYFILE.write_text(api_key)

    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "openai_key.txt").write_text(api_key)
        os.chdir(temp_path)
        try:
            collab = importlib.import_module("collab.collab")
            modules = importlib.import_module("collab.modules")
        finally:
            os.chdir(previous_cwd)

    collab.PROMPT_DIR = str(UPSTREAM_PROMPTS)
    collab.openai_key_file = str(_UPSTREAM_KEYFILE)
    modules.gpt4_key_file = str(_UPSTREAM_KEYFILE)
    _UPSTREAM_CACHE = (collab, modules)
    return _UPSTREAM_CACHE


class VisionBackedModule:
    def __init__(
        self,
        upstream_module_cls,
        owner,
        role_messages,
        model: str,
        api_key_env: str,
        retrival_method: str = "recent_k",
        K: int = 3,
    ):
        self._delegate = upstream_module_cls(
            role_messages,
            model=model,
            retrival_method=retrival_method,
            K=K,
        )
        self.owner = owner
        self.model = model
        self.api_key_env = api_key_env
        self.last_prompt_text = ""
        self.last_raw_response = ""
        self.name = ""

    def __setattr__(self, name: str, value):
        local_attrs = {
            "_delegate",
            "owner",
            "model",
            "api_key_env",
            "last_prompt_text",
            "last_raw_response",
            "name",
        }
        if name in local_attrs or "_delegate" not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        setattr(self._delegate, name, value)

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)

    def reset(self):
        self._delegate.reset()

    def _frame_to_data_url(self, frame: np.ndarray) -> str:
        import pygame

        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            pygame.image.save(surface, tmp.name)
            payload = base64.b64encode(Path(tmp.name).read_bytes()).decode("utf-8")
        return f"data:image/png;base64,{payload}"

    def query(
        self,
        key=None,
        proxy=None,
        stop=None,
        temperature=0.7,
        debug_mode="Y",
        trace=True,
        rethink=False,
        map="",
    ):
        del key, proxy, stop, temperature, debug_mode, map
        messages = self._delegate.query_messages(rethink)
        if not messages:
            raise RuntimeError("Upstream vision planner requires a prompt to query.")
        prompt_text = messages[-1]["content"]
        if trace is False and not rethink:
            prompt_text += " Based on the failure explanation and scene description, analyze and plan again."
        frame = self.owner.current_frame
        if frame is None:
            raise RuntimeError("Upstream vision planner requires a current frame before query().")

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing OpenAI API key in environment variable {self.api_key_env}."
            )

        client = OpenAI(api_key=api_key)
        input_messages = []
        if messages[0].get("role") == "system" and messages[0].get("content"):
            input_messages.append({"role": "system", "content": messages[0]["content"]})
        input_messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {
                        "type": "input_image",
                        "image_url": self._frame_to_data_url(frame),
                    },
                ],
            }
        )
        response = client.responses.create(model=self.model, input=input_messages)
        self.last_prompt_text = "\n\n".join(
            message["content"] if isinstance(message["content"], str) else str(message["content"])
            for message in input_messages
        )
        self.last_raw_response = response.output_text
        return response.output_text, 0


class UpstreamVisionAgent:
    def __init__(self, config: AgentConfig, actor: str):
        collab, modules = _load_upstream_modules()
        self.actor = actor
        self.config = config
        self.current_frame: np.ndarray | None = None
        self._collab = collab
        self._modules = modules
        self._agent = None

    def build(self, mlam):
        collab, modules = self._collab, self._modules
        outer = self

        class VisionLLMAgents(collab.LLMAgents):
            def create_gptmodule(
                self, module_name, file_type="txt", retrival_method="recent_k", K=10
            ):
                if module_name != "planner":
                    raise Exception(f"Module {module_name} not supported.")
                module = VisionBackedModule(
                    modules.Module,
                    owner=outer,
                    role_messages=[{"role": "system", "content": ""}],
                    model=outer.config.model_name,
                    api_key_env=outer.config.api_key_env or "OPENAI_API_KEY",
                    retrival_method=retrival_method,
                    K=K,
                )
                return module

        self._agent = VisionLLMAgents(
            mlam,
            layout=mlam.mdp.layout_name,
            model=self.config.model_name,
            retrival_method="recent_k",
            K=1,
            actor=self.actor,
            controller_mode="new",
        )
        return self._agent

    @property
    def planner(self):
        return self._agent.planner

    @property
    def current_ml_action(self) -> str | None:
        return self._agent.current_ml_action

    @property
    def failed_message(self) -> str:
        return getattr(self._agent, "failed_message", "")

    @property
    def turn_statistics_dict(self):
        return self._agent.turn_statistics_dict

    def set_frame(self, frame: np.ndarray) -> None:
        self.current_frame = frame

    def action(self, state):
        return self._agent.action(state)

    def set_agent_index(self, index: int) -> None:
        self._agent.set_agent_index(index)

    def reset(self, teammate) -> None:
        self._agent.reset(teammate._agent)


@dataclass
class UpstreamTurnResult:
    low_level_action: str
    macro_action: str
    raw_response: str
    prompt_text: str
    parsed_response: AgentTurnResponse
    validator_errors: list[str]


class UpstreamVisionController:
    def __init__(self, env_adapter, chef: AgentConfig, assistant: AgentConfig):
        self.env_adapter = env_adapter
        self.mlam = MediumLevelActionManager(env_adapter.mdp, _planner_params(env_adapter.mdp))
        self.chef = UpstreamVisionAgent(chef, actor="chef")
        self.assistant = UpstreamVisionAgent(assistant, actor="assistant")
        self.chef.build(self.mlam)
        self.assistant.build(self.mlam)
        self.chef._agent.teammate = self.assistant._agent
        self.assistant._agent.teammate = self.chef._agent
        self.chef.set_agent_index(0)
        self.assistant.set_agent_index(1)
        self.chef.reset(self.assistant)
        self.assistant.reset(self.chef)

    def set_frame(self, frame: np.ndarray) -> None:
        self.chef.set_frame(frame)
        self.assistant.set_frame(frame)

    def act(self, role: str):
        agent = self.chef if role == "chef" else self.assistant
        action, _ = agent.action(self.env_adapter.current_state())
        prompt_text = getattr(agent.planner, "last_prompt_text", "")
        raw_response = getattr(agent.planner, "last_raw_response", "")
        parsed = (
            AgentTurnResponse.from_upstream_text(raw_response, role)
            if raw_response
            else AgentTurnResponse(analysis="[NOTHING]", plan="[NONE]", say="[NOTHING]")
        )
        errors = []
        failed_message = agent.failed_message
        if failed_message and failed_message != "success":
            errors.append(failed_message.strip())
        from vision_overcooked.adapters.macro_actions import action_to_name

        macro_action = agent.current_ml_action or "[NONE]"
        return UpstreamTurnResult(
            low_level_action=action_to_name(action),
            macro_action=macro_action,
            raw_response=raw_response,
            prompt_text=prompt_text,
            parsed_response=parsed,
            validator_errors=errors,
        )
