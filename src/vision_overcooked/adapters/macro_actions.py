from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field

from overcooked_ai_py.data.layouts import read_layout_dict
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.planners import MediumLevelActionManager

LOW_LEVEL_ACTIONS = {"NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT"}
ZERO_ARG_ACTIONS = {"place_obj_on_counter", "deliver_soup", "check_recipe"}
ONE_ARG_ACTIONS = {"put_obj_in_utensil", "fill_dish_with_food"}


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


def action_to_name(action) -> str:
    if action == Direction.NORTH:
        return "NORTH"
    if action == Direction.SOUTH:
        return "SOUTH"
    if action == Direction.EAST:
        return "EAST"
    if action == Direction.WEST:
        return "WEST"
    if action == Action.INTERACT:
        return "INTERACT"
    return "STAY"


@dataclass
class MacroPlanValidationResult:
    plan: str
    normalized_plan: str
    valid: bool
    errors: list[str]


@dataclass
class MacroIntentState:
    current_plan: str = "[NONE]"
    remaining_wait: int = 0
    queued_plans: list[str] = field(default_factory=list)


@dataclass
class MacroExecutionResult:
    macro_action: str
    low_level_action: str
    completed: bool
    errors: list[str]


def _canonicalize_param(token: str, mdp) -> str:
    cleaned = token.strip().replace(" ", "").replace('"', "").replace("'", "")
    lowered = cleaned.lower()
    for utensil in mdp.generate_utensil_list():
        if lowered == utensil.lower():
            return utensil
    for ingredient in mdp.all_ingredients + ["dish"]:
        if lowered == ingredient.lower():
            return ingredient
    sources = {"counter", "ingredient_dispenser", "dish_dispenser"}
    if lowered in sources:
        return lowered
    return lowered


def _canonicalize_function_name(name: str) -> str:
    return name.strip().lower().replace(" ", "")


def _primary_plan_step(plan: str) -> str:
    for step in _plan_steps(plan):
        return step
    return ""


def _plan_steps(plan: str) -> list[str]:
    steps = []
    for step in plan.split(";"):
        candidate = step.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered.startswith("say:") or lowered.startswith("say("):
            continue
        if any(token in lowered for token in ("request(", "accept(", "deny(", "clarify(")):
            continue
        steps.append(candidate)
    return steps


def canonicalize_plan(plan: str, role: str, mdp) -> str:
    raw = _primary_plan_step(plan)
    if not raw:
        return raw
    if raw.upper() == "[NONE]":
        return "[NONE]"
    if raw.upper() in LOW_LEVEL_ACTIONS:
        return raw.upper()
    wait_match = re.fullmatch(r"wait\s*\(\s*(\d+)\s*\)", raw, re.IGNORECASE)
    if wait_match:
        return f"wait({wait_match.group(1)})"
    macro_match = re.fullmatch(r"([a-zA-Z_]+)\s*\((.*)\)", raw)
    if not macro_match:
        return raw
    name = _canonicalize_function_name(macro_match.group(1))
    param_blob = macro_match.group(2).strip()
    params = []
    if param_blob:
        params = [_canonicalize_param(token, mdp) for token in param_blob.split(",")]
    if name in ZERO_ARG_ACTIONS:
        return f"{name}()"
    if name in ONE_ARG_ACTIONS or name in _role_operation_names(role, mdp):
        return f"{name}({','.join(params)})"
    if name == "pickup":
        return f"pickup({','.join(params)})"
    return raw


def _role_operation_names(role: str, mdp) -> set[str]:
    layout = read_layout_dict(mdp.layout_name)
    key = "utensil_agent0" if role == "chef" else "utensil_agent1"
    return set(layout[key].values())


def role_action_guide(role: str, mdp) -> str:
    utensils = ", ".join(mdp.generate_utensil_list())
    operations = ", ".join(sorted(_role_operation_names(role, mdp))) or "[NONE]"
    ingredients = ", ".join(sorted(mdp.all_ingredients + ["dish"]))
    if role == "chef":
        sources = "counter"
        extra = (
            "You cannot access ingredient_dispenser or dish_dispenser directly. "
            "Use check_recipe(), pickup(...,counter), put_obj_in_utensil(...), cook/bake on your side, "
            "fill_dish_with_food(...), place_obj_on_counter(), deliver_soup(), wait(n), or [NONE]. "
            "If an ingredient is needed but not on the counter, ask the assistant in say to pick it up from ingredient_dispenser and place it on the counter."
        )
    else:
        sources = "ingredient_dispenser, dish_dispenser, counter"
        extra = (
            "You cannot use check_recipe(). "
            "Use pickup(...,ingredient_dispenser), pickup(dish,dish_dispenser), pickup(...,counter), "
            "put_obj_in_utensil(...), cut/stir on your side, place_obj_on_counter(), wait(n), or [NONE]. "
            "If chef asks for an ingredient from ingredient_dispenser or asks you to place something on the counter, prefer following that request."
        )
    return (
        f"Valid pickup sources for {role}: {sources}. "
        f"Valid utensil names on this layout: {utensils}. "
        f"Valid object names on this layout: {ingredients}. "
        f"Role-specific utensil operations: {operations}. "
        f"{extra}"
    )


def _build_role_action_space(role: str, mdp) -> set[str]:
    layout = read_layout_dict(mdp.layout_name)
    utensil_list = mdp.generate_utensil_list()
    if role == "chef":
        actions = {
            "check_recipe()",
            "place_obj_on_counter()",
            "fill_dish_with_food()",
            "deliver_soup()",
            "pickup(dish,counter)",
        }
        pick_places = {"counter"}
        utensil_map = layout["utensil_agent0"]
    else:
        actions = {
            "place_obj_on_counter()",
            "pickup(dish,counter)",
            "pickup(dish,dish_dispenser)",
        }
        pick_places = {"ingredient_dispenser", "dish_dispenser", "counter"}
        utensil_map = layout["utensil_agent1"]

    for utensil, operation in utensil_map.items():
        for utensil_name in utensil_list:
            if utensil in utensil_name:
                actions.add(f"{operation}({utensil_name})")
                actions.add(f"put_obj_in_utensil({utensil_name})")
                actions.add(f"fill_dish_with_food({utensil_name})")
                pick_places.add(utensil_name)

    ingredients = list(layout["ingredients"])
    for recipe_group in layout["recipes"].values():
        for result_name in recipe_group.keys():
            if result_name not in ingredients:
                ingredients.append(result_name)
    for ingredient in ingredients:
        for place in pick_places:
            actions.add(f"pickup({ingredient},{place})")

    actions.add("[NONE]")
    return actions


def validate_plan_string(plan: str, role: str = "chef", mdp=None) -> MacroPlanValidationResult:
    normalized = plan.strip()
    if not normalized:
        return MacroPlanValidationResult(
            plan=plan,
            normalized_plan=plan,
            valid=False,
            errors=["Plan cannot be empty."],
        )
    if normalized.upper() in LOW_LEVEL_ACTIONS:
        return MacroPlanValidationResult(
            plan=plan,
            normalized_plan=normalized.upper(),
            valid=False,
            errors=[
                "Low-level movement actions are unsupported. Use benchmark macro-actions like pickup(...), put_obj_in_utensil(...), cook(...), wait(n), or [NONE]."
            ],
        )
    if mdp is None:
        raise ValueError("validate_plan_string requires an mdp for macro-action validation.")
    canonical = canonicalize_plan(normalized, role, mdp)
    if canonical == "[NONE]":
        return MacroPlanValidationResult(plan=plan, normalized_plan=canonical, valid=True, errors=[])
    if re.fullmatch(r"wait\(\d+\)", canonical):
        return MacroPlanValidationResult(plan=plan, normalized_plan=canonical, valid=True, errors=[])
    allowed_actions = _build_role_action_space(role, mdp)
    if canonical in allowed_actions:
        return MacroPlanValidationResult(plan=plan, normalized_plan=canonical, valid=True, errors=[])
    return MacroPlanValidationResult(
        plan=plan,
        normalized_plan=canonical,
        valid=False,
        errors=[
            "Unsupported macro-action for this role/layout. Use [NONE], wait(n), pickup(obj,source), put_obj_in_utensil(utensil), fill_dish_with_food(utensil), place_obj_on_counter(), deliver_soup(), or the role's utensil operation such as cook(pot0)."
        ],
    )


class MacroActionExecutor:
    def __init__(self, env_adapter):
        self.env_adapter = env_adapter
        self.mlam = MediumLevelActionManager(env_adapter.mdp, _planner_params(env_adapter.mdp))
        self.intent_state = {
            "chef": MacroIntentState(),
            "assistant": MacroIntentState(),
        }

    def reset(self) -> None:
        self.mlam = MediumLevelActionManager(self.env_adapter.mdp, _planner_params(self.env_adapter.mdp))
        self.intent_state = {
            "chef": MacroIntentState(),
            "assistant": MacroIntentState(),
        }

    def accept_proposal(self, role: str, proposed_plan: str) -> str:
        steps = _plan_steps(proposed_plan)
        intent = self.intent_state[role]
        if intent.current_plan != "[NONE]" and (
            intent.queued_plans or not intent.current_plan.startswith("wait(")
        ):
            return intent.current_plan
        if not steps:
            return self.intent_state[role].current_plan
        canonical_steps = [canonicalize_plan(step, role, self.env_adapter.mdp) for step in steps]
        canonical = canonical_steps[0]
        if canonical == "[NONE]":
            return self.intent_state[role].current_plan
        wait_match = re.fullmatch(r"wait\((\d+)\)", canonical)
        remaining_wait = int(wait_match.group(1)) if wait_match else 0
        self.intent_state[role] = MacroIntentState(
            current_plan=canonical,
            remaining_wait=remaining_wait,
            queued_plans=canonical_steps[1:],
        )
        return canonical

    def execute_current_intent(self, role: str) -> MacroExecutionResult:
        state = self.env_adapter.current_state()
        intent = self.intent_state[role]
        if intent.current_plan == "[NONE]":
            return MacroExecutionResult("[NONE]", "STAY", True, [])
        if intent.current_plan.startswith("wait("):
            intent.remaining_wait = max(intent.remaining_wait - 1, 0)
            completed = intent.remaining_wait == 0
            if completed:
                self._advance_or_clear(role)
            return MacroExecutionResult(intent.current_plan, "STAY", completed, [])

        if self._is_completed(role, intent.current_plan, state):
            self._advance_or_clear(role)
            return MacroExecutionResult(intent.current_plan, "STAY", True, [])

        validation_error = self._validate_against_state(role, intent.current_plan, state)
        if validation_error is not None:
            self.intent_state[role] = MacroIntentState()
            return MacroExecutionResult("[NONE]", "STAY", True, [validation_error])

        motion_goals = self._find_motion_goals(role, intent.current_plan, state)
        if not motion_goals:
            self.intent_state[role] = MacroIntentState()
            return MacroExecutionResult(
                "[NONE]",
                "STAY",
                True,
                [f"No reachable motion goals for {intent.current_plan}."],
            )

        low_level = self._choose_lowest_cost_action(role, motion_goals, state)
        if low_level is None:
            return MacroExecutionResult(intent.current_plan, "STAY", False, [])
        return MacroExecutionResult(intent.current_plan, action_to_name(low_level), False, [])

    def finalize_step(self) -> dict[str, str]:
        state = self.env_adapter.current_state()
        completed_actions: dict[str, str] = {}
        for role, intent in self.intent_state.items():
            if intent.current_plan != "[NONE]" and not intent.current_plan.startswith("wait("):
                if self._is_completed(role, intent.current_plan, state):
                    completed_actions[role] = intent.current_plan
                    self._advance_or_clear(role)
        return completed_actions

    def _advance_or_clear(self, role: str) -> None:
        intent = self.intent_state[role]
        if intent.queued_plans:
            next_plan = intent.queued_plans.pop(0)
            wait_match = re.fullmatch(r"wait\((\d+)\)", next_plan)
            self.intent_state[role] = MacroIntentState(
                current_plan=next_plan,
                remaining_wait=int(wait_match.group(1)) if wait_match else 0,
                queued_plans=intent.queued_plans,
            )
            return
        self.intent_state[role] = MacroIntentState()

    def _player(self, role: str, state):
        return state.players[0 if role == "chef" else 1]

    def _agent_index(self, role: str) -> int:
        return 0 if role == "chef" else 1

    def _parse_action(self, action: str) -> tuple[str, list[str]]:
        if action == "[NONE]":
            return "[NONE]", []
        wait_match = re.fullmatch(r"wait\((\d+)\)", action)
        if wait_match:
            return "wait", [wait_match.group(1)]
        match = re.fullmatch(r"([a-z_]+)\((.*)\)", action)
        if not match:
            return action, []
        name = match.group(1)
        param_blob = match.group(2)
        params = [] if not param_blob else [part for part in param_blob.split(",") if part]
        return name, params

    def _validate_against_state(self, role: str, action: str, state) -> str | None:
        player = self._player(role, state)
        has_object = player.has_object()
        action_name, params = self._parse_action(action)
        counter_objects = self.env_adapter.mdp.get_counter_objects_dict(
            state, list(self.env_adapter.mdp.terrain_pos_dict["X"])
        )
        if action_name == "pickup":
            if len(params) != 2:
                return "pickup(...) requires object and source."
            if has_object:
                return f"{role} is already holding an object."
            obj, source = params
            if source == "counter":
                if obj not in counter_objects or len(counter_objects[obj]) == 0:
                    return f"{obj} is not available on any reachable counter."
            elif source == "ingredient_dispenser":
                if obj not in self.env_adapter.mdp.default_ingredients:
                    return f"{obj} cannot be picked up from ingredient_dispenser."
            elif source == "dish_dispenser":
                if obj != "dish":
                    return "Only dish can be picked up from dish_dispenser."
            elif source in self.env_adapter.mdp.generate_utensil_list():
                utensil_states = self.env_adapter.mdp.get_utensil_states(state)
                if source in utensil_states["empty"]:
                    return f"{source} is empty."
            return None
        if action_name == "put_obj_in_utensil":
            if not has_object:
                return f"{role} must be holding an object before using put_obj_in_utensil(...)."
            utensil = params[0]
            utensil_states = self.env_adapter.mdp.get_utensil_states(state)
            if utensil in utensil_states["cooking"] or utensil in utensil_states["ready"]:
                return f"{utensil} is already busy."
            return None
        if action_name == "fill_dish_with_food":
            if not has_object or player.get_object().name != "dish":
                return f"{role} must hold a dish before using fill_dish_with_food(...)."
            utensil = params[0]
            utensil_states = self.env_adapter.mdp.get_utensil_states(state)
            if utensil not in utensil_states["ready"]:
                return f"{utensil} is not ready to fill a dish."
            return None
        if action_name == "place_obj_on_counter":
            if not has_object:
                return f"{role} is not holding an object to place on a counter."
            if not self.mlam.place_obj_on_counter_actions(state):
                return "There is no reachable empty counter to place the object on."
            return None
        if action_name == "deliver_soup":
            if not has_object:
                return f"{role} is not holding a deliverable object."
            if self.env_adapter.current_order not in player.get_object().name:
                return f"{role} is not holding the current recipe output."
            return None
        if action_name == "check_recipe":
            return None if role == "chef" else "Only chef can use check_recipe()."
        if action_name in _role_operation_names(role, self.env_adapter.mdp):
            if has_object:
                return f"{role} must have empty hands before using {action_name}(...)."
            utensil = params[0]
            utensil_pos = self.env_adapter.mdp.utensil_state_dict[utensil]["position"]
            if not state.has_object(utensil_pos):
                return f"{utensil} has nothing in it to operate on."
            obj = state.get_object(utensil_pos)
            if not hasattr(obj, "state"):
                return f"{utensil} has no operable object."
            _, _, cook_time = obj.state
            if cook_time != 0:
                return f"{utensil} is already operating or completed."
            return None
        return None

    def _is_completed(self, role: str, action: str, state) -> bool:
        player = self._player(role, state)
        action_name, params = self._parse_action(action)
        if action_name == "pickup":
            return player.has_object() and player.get_object().name == params[0]
        if action_name in {"put_obj_in_utensil", "place_obj_on_counter", "deliver_soup"}:
            return not player.has_object()
        if action_name == "fill_dish_with_food":
            return player.held_object is not None and self.env_adapter.current_order in player.held_object.name
        if action_name in _role_operation_names(role, self.env_adapter.mdp):
            utensil_states = self.env_adapter.mdp.get_utensil_states(state)
            utensil = params[0]
            return utensil in utensil_states["cooking"] or utensil in utensil_states["ready"]
        if action_name == "check_recipe":
            return True
        return False

    def _find_motion_goals(self, role: str, action: str, state):
        agent_index = self._agent_index(role)
        counter_objects = self.env_adapter.mdp.get_counter_objects_dict(
            state, list(self.env_adapter.mdp.terrain_pos_dict["X"])
        )
        action_name, params = self._parse_action(action)
        player = self._player(role, state)
        if action_name == "pickup":
            return self.mlam.pickup_obj_actions(
                state,
                params[0],
                params[1],
                agent_index,
                counter_objects,
            )
        if action_name == "put_obj_in_utensil":
            return self.mlam.go_to_utensil_actions(state, params[0], agent_index)
        if action_name == "fill_dish_with_food":
            return self.mlam.go_to_utensil_actions(state, params[0], agent_index)
        if action_name == "place_obj_on_counter":
            motion_goals = self.mlam.place_obj_on_counter_actions(state)
            if motion_goals:
                return motion_goals
            return self.mlam._get_ml_actions_for_positions(self.env_adapter.mdp.get_empty_counter_locations(state))
        if action_name == "deliver_soup":
            return self.mlam.deliver_soup_actions()
        if action_name in _role_operation_names(role, self.env_adapter.mdp):
            return self.mlam.go_to_utensil_actions(state, params[0], agent_index)
        if action_name == "check_recipe":
            return self.mlam.wait_actions(player)
        return self.mlam.wait_actions(player)

    def _choose_lowest_cost_action(self, role: str, motion_goals, state):
        player = self._player(role, state)
        best_action = None
        min_cost = float("inf")
        for goal in motion_goals:
            if not self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, goal):
                continue
            action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(player.pos_and_or, goal)
            if plan_cost < min_cost:
                min_cost = plan_cost
                best_action = action_plan[0]
        return best_action
