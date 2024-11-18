from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Ball


class AdversaryDoorPickup(AdversaryEnv):
    def __init__(self, max_steps: int | None = None, **kwargs):
        max_steps = 200
        super().__init__(
            mission_space=mission_space,
            width=11,
            num_cols=6,
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        self.grid.vert_wall(int(width/2),0)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self.goal_reward
                terminated = True

        return obs, reward, terminated, truncated, info
