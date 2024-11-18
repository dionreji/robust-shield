from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, SlipperyNorth, SlipperyEast, SlipperySouth, SlipperyWest, Ball
from minigrid.envs.adversaries_base import AdversaryEnv
from minigrid.core.tasks import GoTo, DoNothing, PickUpObject, PlaceObject

import numpy as np

class OscillatingAdversaries(AdversaryEnv):

    def __init__(self, width=8, height=8, max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 200

        super().__init__(
            width=width, height=height, max_steps=max_steps, **kwargs
        )

    def _gen_grid(self, width, height):
        assert width >= 8 and height >= 8
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        goal_pos = np.array((int(width/2) - 1, height - 2))
        self.put_obj(Goal(), *goal_pos)

        self.adversaries = {}
        self.agent_pos = np.array((int(width/2), 1))
        self.agent_dir = 2

        yellow_adv = self.add_adversary(1, 3, "yellow", direction=0, tasks=[GoTo((3, 3)),
                                                                            GoTo((3, 1)),
                                                                            GoTo((1, 3))], repeating=True)
        green_adv = self.add_adversary(width - 2, 5, "green", direction=3, tasks=[GoTo((width - 2, 3)),
                                                                                  GoTo((width - 4, 3)),
                                                                                  GoTo((width - 4, 5))], repeating=True)


    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
