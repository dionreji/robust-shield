from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, SlipperyNorth, SlipperyEast, SlipperySouth, SlipperyWest, Ball
from minigrid.envs.adversaries_base import AdversaryEnv
from minigrid.core.tasks import GoTo, DoNothing, PickUpObject, PlaceObject

import numpy as np

class AdversarySimple(AdversaryEnv):

    """
    ## Description


    ## Registered Configurations

    - `MiniGrid-Adv-8x8-v0`
    - `MiniGrid-AdvLava-8x8-v0`
    - `MiniGrid-AdvSlipperyLava-8x8-v0`
    - `MiniGrid-AdvSimple-8x8-v0`

    """

    def __init__(self, width=7, height=6, generate_wall=True, generate_lava=False, generate_slippery=False ,max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 200

        self.generate_wall = generate_wall
        super().__init__(
            width=width, height=height, max_steps=max_steps, **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        goal_pos = np.array((width - 2, height - 2))
        ball_pos = np.array((width - 3, height - 2))

        self.put_obj(Goal(), *goal_pos)
        self.put_obj(Ball("yellow"), *ball_pos)
        ball = self.grid.get(*ball_pos)


        self.adversaries = {}
        self.agent_pos = np.array((width - 2, 1))
        self.agent_dir = 2

        self.grid.horz_wall(2, height - 3)

        blue_adv = self.add_adversary(1, 1, "yellow", direction=1, tasks=[GoTo((width - 4, height - 2)),
                                                                        PickUpObject(ball_pos, ball),
                                                                        GoTo((1,1)),
                                                                        PlaceObject((2, 1), ball),
                                                                        DoNothing(duration=2),
                                                                        PickUpObject((2, 1), ball),
                                                                        GoTo((width - 4, height - 2)),
                                                                        PlaceObject(ball_pos, ball)], repeating=True)



    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
