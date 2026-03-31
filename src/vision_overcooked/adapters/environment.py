from __future__ import annotations

from dataclasses import dataclass

import numpy as np

if not hasattr(np, "Inf"):
    np.Inf = np.inf

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from vision_overcooked.paths import UPSTREAM_RECIPE_DIR


TERRAIN_COLORS = {
    " ": np.array([247, 242, 224], dtype=np.uint8),
    "X": np.array([90, 78, 65], dtype=np.uint8),
    "P": np.array([194, 76, 60], dtype=np.uint8),
    "D": np.array([90, 127, 160], dtype=np.uint8),
    "S": np.array([115, 158, 92], dtype=np.uint8),
    "O": np.array([184, 124, 55], dtype=np.uint8),
    "T": np.array([176, 71, 71], dtype=np.uint8),
    "I": np.array([108, 119, 176], dtype=np.uint8),
    "B": np.array([146, 103, 176], dtype=np.uint8),
    "C": np.array([110, 110, 110], dtype=np.uint8),
}
OBJECT_COLORS = {
    "dish": np.array([225, 225, 240], dtype=np.uint8),
    "egg": np.array([239, 226, 113], dtype=np.uint8),
    "mushroom": np.array([157, 116, 87], dtype=np.uint8),
    "sweet_potato": np.array([216, 140, 87], dtype=np.uint8),
    "potato": np.array([209, 190, 145], dtype=np.uint8),
    "bell_pepper": np.array([206, 81, 73], dtype=np.uint8),
    "pumpkin": np.array([226, 136, 57], dtype=np.uint8),
    "corn": np.array([235, 212, 90], dtype=np.uint8),
    "green_bean": np.array([92, 165, 95], dtype=np.uint8),
    "onion": np.array([209, 188, 96], dtype=np.uint8),
    "carrot": np.array([223, 126, 56], dtype=np.uint8),
    "broccoli": np.array([92, 170, 82], dtype=np.uint8),
    "chickpea": np.array([210, 190, 136], dtype=np.uint8),
    "cauliflower": np.array([230, 229, 212], dtype=np.uint8),
    "pea": np.array([92, 173, 87], dtype=np.uint8),
    "lentil": np.array([116, 86, 61], dtype=np.uint8),
    "spinach": np.array([65, 147, 83], dtype=np.uint8),
    "zucchini": np.array([114, 173, 91], dtype=np.uint8),
    "romaine_lettuce": np.array([134, 188, 88], dtype=np.uint8),
    "taro": np.array([181, 154, 177], dtype=np.uint8),
    "bean": np.array([156, 97, 69], dtype=np.uint8),
}
PLAYER_COLORS = [
    np.array([58, 120, 214], dtype=np.uint8),
    np.array([72, 166, 92], dtype=np.uint8),
]


@dataclass
class EnvironmentSnapshot:
    frame: np.ndarray
    state_string: str
    order: str
    timestep: int


class EnvironmentAdapter:
    def __init__(self, layout: str = "new_env", horizon: int = 120, tile_size: int = 32):
        self.layout = layout
        self.horizon = horizon
        self.tile_size = tile_size
        self.current_order: str | None = None
        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.env = OvercookedEnv(self.mdp, horizon=horizon)

    def available_tasks(self) -> list[dict[str, object]]:
        tasks = []
        for path in sorted(UPSTREAM_RECIPE_DIR.glob("*.txt")):
            prefix, _, name = path.stem.partition("_")
            try:
                level = int(prefix)
            except ValueError:
                level = 0
            tasks.append({"level": level, "order": name})
        return tasks

    def reset(self, order: str) -> EnvironmentSnapshot:
        self.current_order = order
        self.mdp = OvercookedGridworld.from_layout_name(self.layout)
        self.mdp.start_order_list = [order]
        self.mdp.one_task_mode = True
        self.env = OvercookedEnv(self.mdp, horizon=self.horizon)
        self.env.reset()
        return self.snapshot(order)

    def current_state(self):
        return self.env.state

    def snapshot(self, order: str) -> EnvironmentSnapshot:
        state = self.env.state
        state_string = self.env.mdp.state_string(state).replace("ø", "o")
        frame = self.render_frame()
        return EnvironmentSnapshot(
            frame=frame,
            state_string=state_string,
            order=order,
            timestep=self.env.t,
        )

    def step(self, joint_actions: tuple[str, str], order: str) -> tuple[EnvironmentSnapshot, float, bool]:
        motion_actions = tuple(self._to_motion_action(action) for action in joint_actions)
        _, reward, done, _ = self.env.step(motion_actions)
        return self.snapshot(order), reward, done

    def render_frame(self) -> np.ndarray:
        terrain = self.env.mdp.terrain_mtx
        height = len(terrain)
        width = len(terrain[0])
        frame = np.zeros((height * self.tile_size, width * self.tile_size, 3), dtype=np.uint8)

        for y, row in enumerate(terrain):
            for x, tile in enumerate(row):
                color = TERRAIN_COLORS.get(tile, TERRAIN_COLORS[" "])
                self._fill_tile(frame, x, y, color)

        for obj in self.env.state.objects.values():
            obj_color = OBJECT_COLORS.get(obj.name, np.array([45, 45, 45], dtype=np.uint8))
            self._fill_marker(frame, obj.position[0], obj.position[1], obj_color, margin=8)

        for index, player in enumerate(self.env.state.players):
            self._fill_marker(frame, player.position[0], player.position[1], PLAYER_COLORS[index], margin=4)
            if player.held_object is not None:
                held_color = OBJECT_COLORS.get(player.held_object.name, np.array([15, 15, 15], dtype=np.uint8))
                self._fill_marker(frame, player.position[0], player.position[1], held_color, margin=12)

        return frame

    def _fill_tile(self, frame: np.ndarray, x: int, y: int, color: np.ndarray) -> None:
        ys = y * self.tile_size
        xs = x * self.tile_size
        frame[ys : ys + self.tile_size, xs : xs + self.tile_size] = color

    def _fill_marker(
        self, frame: np.ndarray, x: int, y: int, color: np.ndarray, margin: int
    ) -> None:
        ys = y * self.tile_size + margin
        xs = x * self.tile_size + margin
        frame[
            ys : ys + self.tile_size - 2 * margin,
            xs : xs + self.tile_size - 2 * margin,
        ] = color

    def _to_motion_action(self, action_name: str):
        normalized = action_name.strip().upper()
        mapping = {
            "NORTH": Direction.NORTH,
            "SOUTH": Direction.SOUTH,
            "EAST": Direction.EAST,
            "WEST": Direction.WEST,
            "INTERACT": Action.INTERACT,
            "STAY": Action.STAY,
        }
        return mapping.get(normalized, Action.STAY)
