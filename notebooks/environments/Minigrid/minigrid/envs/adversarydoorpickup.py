from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.envs.adversaries_base import AdversaryEnv
from minigrid.core.tasks import GoTo
from minigrid.core.world_object import Door, Box


class AdversaryDoorPickup(RoomGrid, AdversaryEnv):
    def __init__(self, success_reward=1, collision_penalty=-1, dense_reward: bool = False, max_steps: int | None = None, **kwargs):
        max_steps = 200
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=6,
            max_steps=max_steps,
            **kwargs,
        )
        self.success_reward = success_reward
        self.collision_penalty = collision_penalty
        self.dense_reward = dense_reward

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self.width = width

        self.agent_pos = (1, 1)
        self.agent_dir = 1

        self.put_obj(Door("yellow"), int(width/2), height - 2)
        object, _ = self.add_object(1, 0, kind="box")
        self.object = object

        green_adv = self.add_adversary(int(width/2) - 1, 1, "green", direction=1, tasks=[GoTo((int(width/2) - 1, 4)),
                                                                                         GoTo((1, 4)),
                                                                                         GoTo((1, 1)),
                                                                                         GoTo((int(width/2) - 1, 1))], repeating=True)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.object:
                reward = self.success_reward
                terminated = True

        if self.dense_reward and action == self.actions.toggle:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell and fwd_cell.type == "door":
                if fwd_cell.is_open:
                    reward += 0.1
                if not fwd_cell.is_open:
                    reward -= 0.11
        if self.dense_reward and self.agent_pos[0] < 7:
            reward -= 0.001 * (self.width - self.agent_pos[0])
        return obs, reward, terminated, truncated, info
