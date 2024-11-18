from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import (
    Ball,
    Box,
    Key,
    Slippery,
    SlipperyEast,
    SlipperySouth,
    SlipperyNorth,
    SlipperyWest,
    Lava,
    Goal,
    Point
 )

from minigrid.minigrid_env import MiniGridEnv

import numpy as np
import random

class Playground(MiniGridEnv):
    """
    An empty Playground environment for Graz Security Week
    """
    def __init__(self,
                size=12,
                width=None,
                height=None,
                fault_probability=0.0,
                per_step_penalty=0.0,
                probability_intended=1.0,
                probability_turn_intended=1.0,
                faulty_behavior=True,
                randomize_start=True,
                **kwargs):

        self.size = size
        self.fault_probability = fault_probability
        self.faulty_behavior = faulty_behavior
        self.previous_action = None
        self.per_step_penalty = per_step_penalty
        self.randomize_start = randomize_start
        self.probability_intended = probability_intended
        self.probability_turn_intended = probability_turn_intended

        if width is not None and height is not None:
            self.width = width
            self.height = height
        else:
            self.width = size
            self.height = size

        mission_space = MissionSpace(mission_func=lambda: "get to the green goal square")

        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            max_steps=200,
            see_through_walls=False,
            **kwargs
        )

    def fault(self):
        return True if random.random() < self.fault_probability else False

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.step_count > 0 and self.fault():
            action = self.previous_action
        self.previous_action = action
        obs, reward, terminated, trucated, info = super().step(action)
        return obs, reward - self.per_step_penalty, terminated, trucated, info

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        self.previous_action = None
        return super().reset(**kwargs)

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5
        # Create an empty grid
        self.grid = Grid(width, height)

        slippery_north = SlipperyNorth(probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended)
        slippery_east = SlipperyEast(probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended)
        slippery_south = SlipperySouth(probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended)
        slippery_west = SlipperyWest(probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended)



        # A rectangular wall around the environment
        self.grid.wall_rect(0, 0, width, height)

        # Change the goal position:
        self.put_obj(Goal(), width - 2, 1)

        # TODO: Add walls, pools of lava, etc.
        self.grid.horz_wall(2, 1, 3, slippery_north)
        self.grid.vert_wall(5, 2, 2, slippery_east)
        l = 4
        self.grid.horz_wall(self.size - l - 1, self.size - 2, l, Lava)

        self.put_obj(Lava(), 7, 3)


        if self.randomize_start:
            self.place_agent()
        else:
            self.agent_pos = np.array((1, height - 2))
            self.agent_dir = 3

    def disable_random_start(self):
        self.randomize_start = False

    def printGrid(self, init=False):
        grid = super().printGrid(init)
        properties_str = ""
        if self.faulty_behavior:
            properties_str += F"FaultProbability:{self.fault_probability}\n"
        properties_str += F"ProbTurnIntended:{self.probability_turn_intended}\n"
        properties_str += F"ProbForwardIntended:{self.probability_intended}\n"
        return  grid + properties_str


