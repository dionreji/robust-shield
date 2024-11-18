from __future__ import annotations
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import (
    SlipperyEast,
    SlipperySouth,
    SlipperyNorth,
    SlipperyWest,
    SlipperyNorthEast,
    Lava,
    Goal
 )
from minigrid.envs.adversaries_base import AdversaryEnv
from minigrid.core.tasks import FollowAgent, DoRandom, GoTo

from minigrid.minigrid_env import MiniGridEnv, is_slippery

import numpy as np
from itertools import product

class WindyCityEnv(MiniGridEnv):
    def __init__(self,
                randomize_start=True, size=10,
                width=24,
                height=22,
                probability_intended=8/9,
                probability_turn_intended=8/9,
                obstacle_type=Lava,
                goal_reward=1,
                failure_penalty=-1,
                per_step_penalty=0,
                dense_rewards=False,
                     **kwargs):

        self.obstacle_type = obstacle_type
        self.size = size
        self.probability_intended = probability_intended
        self.probability_turn_intended = probability_turn_intended

        if width is not None and height is not None:
            self.width = width
            self.height = height
        elif size is not None:
            self.width = size
            self.height = size
        else:
            raise ValueError(f"Please define either width and height or a size for square environments. The set values are width: {width}, height: {height}, size: {size}.")

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=200,
            # Set this to True for maximum speed
            see_through_walls=False,
            mission_space = mission_space,
            **kwargs
        )

        self.randomize_start = randomize_start
        self.goal_reward = goal_reward
        self.failure_penalty = failure_penalty
        self.dense_rewards = dense_rewards
        self.per_step_penalty = per_step_penalty

        self.trajectory = list()

    @staticmethod
    def _gen_mission():
        return "Finish your task while avoiding the adversaries"

    def disable_random_start(self):
        self.randomize_start = False

    def place_agent(self, spawn_on_slippery=False, agent_pos=None, agent_dir=0):
        max_tries = 10_000
        num_tries = 0

        if self.randomize_start == True:
            while True:
                num_tries += 1
                if num_tries > max_tries:
                    raise RecursionError("rejection sampling failed in place_obj")
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)

                cell = self.grid.get(*(x,y))
                if ( cell is None or
                    (cell.can_overlap() and
                        not isinstance(cell, Lava) and
                        not isinstance(cell, Goal) and
                        (spawn_on_slippery or not is_slippery(cell)) and
                        not (x in [7, 8, 9, 10] and y in [9, 10]))
                    ):
                    self.agent_pos = np.array((x, y))
                    self.agent_dir = np.random.randint(0, 4)
                    break
        elif agent_dir is None:
            self.agent_pos = np.array((1, 1))
            self.agent_dir = 0
        else:
            self.agent_pos = agent_pos
            self.agent_dir = agent_dir
        self.trajectory.append((self.agent_pos, self.agent_dir))

    def place_goal(self, goal_pos):
        self.goal_pos = goal_pos
        self.put_obj(Goal(), *self.goal_pos)


    def printGrid(self, init=False):
        grid = super().printGrid(init)

        properties_str = ""

        properties_str += F"ProbTurnIntended:{self.probability_turn_intended}\n"
        properties_str += F"ProbForwardIntended:{self.probability_intended}\n"

        return grid + properties_str

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.trajectory.append((action, self.agent_pos, self.agent_dir))
        if truncated and info["ran_into_lava"]:
            print(self.trajectory)
            print("truncated: ", info)
            self.trajectory = list()
        if truncated and info["reached_goal"]:
            print("truncated: ", info)
            self.trajectory = list()
        elif terminated and info["ran_into_lava"]:
            print(self.trajectory)
            print("terminated: ", info)
            self.trajectory = list()
        elif terminated:
            print("terminated: ", info)
            self.trajectory = list()
        elif truncated:
            print("truncated: ", info)
            self.trajectory = list()
        return obs, reward - self.per_step_penalty, terminated, truncated, info

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        return super().reset(**kwargs)

    def _place_building(self, col, row, width, height, obj_type=Lava):
        for i in range(col, width + col):
            self.grid.vert_wall(i, row, height, obj_type=obj_type)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        for i in range(1, height - 1):
            self.grid.horz_wall(1, i, width-2, obj_type=SlipperyNorthEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))

        self._place_building(13, 1, 4, 2)
        self.grid.vert_wall(12, 1, 2, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(13, 3, 4, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(17, 1, 2, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))

        self._place_building(7,  3, 3, 4)
        self.grid.vert_wall(6, 3, 4, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(10, 3, 4, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(7, 2, 3, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(7, 7, 3, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))

        self._place_building(15, 7, 6, 4)
        self.grid.vert_wall(14, 7, 4, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(14, 9, 2, obj_type=Lava)
        self.grid.vert_wall(20, 7, 4, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(13, 9, 2, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(15, 6, 5, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(14, 11, 6, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))


        self._place_building(5, 11, 5, 6)
        self.grid.vert_wall(4, 11, 6, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(10, 11, 6, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(5, 17, 5, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(5, 10, 5, obj_type=SlipperyWest("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(6, 9, 4, obj_type=SlipperyWest("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(9, 7, 4, obj_type=SlipperySouth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))

        self._place_building(21, 13, 2, 5)
        self.grid.vert_wall(20, 13, 5, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(21, 12, 2, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(21, 18, 2, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))



        self.place_agent(agent_pos=np.array((1, height -2)), agent_dir=0, spawn_on_slippery=True)
        self.place_goal(np.array((width - 2, 1)))
        if self.dense_rewards: self.run_bfs()


class WindyCityAdvEnv(AdversaryEnv):
    def __init__(self,
                randomize_start=True, size=10,
                width=15,
                height=15,
                probability_intended=8/9,
                probability_turn_intended=8/9,
                obstacle_type=Lava,
                goal_reward=1,
                failure_penalty=-1,
                per_step_penalty=0,
                dense_rewards=False,
                     **kwargs):

        self.obstacle_type = obstacle_type
        self.size = size
        self.probability_intended = probability_intended
        self.probability_turn_intended = probability_turn_intended

        if width is not None and height is not None:
            self.width = width
            self.height = height
        elif size is not None:
            self.width = size
            self.height = size
        else:
            raise ValueError(f"Please define either width and height or a size for square environments. The set values are width: {width}, height: {height}, size: {size}.")

        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=200,
            # Set this to True for maximum speed
            see_through_walls=False,
            **kwargs
        )

        self.randomize_start = randomize_start
        self.goal_reward = goal_reward
        self.failure_penalty = failure_penalty
        self.dense_rewards = dense_rewards
        self.per_step_penalty = per_step_penalty

        self.trajectory = list()

    def disable_random_start(self):
        self.randomize_start = False

    def place_agent(self, spawn_on_slippery=False, agent_pos=None, agent_dir=0):
        max_tries = 10_000
        num_tries = 0

        if self.randomize_start == True:
            while True:
                num_tries += 1
                if num_tries > max_tries:
                    raise RecursionError("rejection sampling failed in place_obj")
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)

                cell = self.grid.get(*(x,y))
                if ( cell is None or
                    (cell.can_overlap() and
                        not isinstance(cell, Lava) and
                        not isinstance(cell, Goal) and
                        (spawn_on_slippery or not is_slippery(cell)) and
                        not (x in [7, 8, 9, 10] and y in [9, 10]))
                    ):
                    self.agent_pos = np.array((x, y))
                    self.agent_dir = np.random.randint(0, 4)
                    break
        elif agent_dir is None:
            self.agent_pos = np.array((1, 1))
            self.agent_dir = 0
        else:
            self.agent_pos = agent_pos
            self.agent_dir = agent_dir
        self.trajectory.append((self.agent_pos, self.agent_dir))

    def place_goal(self, goal_pos):
        self.goal_pos = goal_pos
        self.put_obj(Goal(), *self.goal_pos)


    def printGrid(self, init=False):
        grid = super().printGrid(init)

        properties_str = ""

        properties_str += F"ProbTurnIntended:{self.probability_turn_intended}\n"
        properties_str += F"ProbForwardIntended:{self.probability_intended}\n"

        return grid + properties_str

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.trajectory.append((action, self.agent_pos, self.agent_dir, str(self.adversaries["blue"])))
        if truncated and info["ran_into_lava"]:
            print(self.trajectory)
            print("truncated: ", info)
            self.trajectory = list()
        if truncated and info["reached_goal"]:
            print("truncated: ", info)
            self.trajectory = list()
        elif terminated and info["ran_into_lava"]:
            print(self.trajectory)
            print("terminated: ", info)
            self.trajectory = list()
        elif terminated:
            print("terminated: ", info)
            self.trajectory = list()
        elif truncated:
            print("truncated: ", info)
            self.trajectory = list()
        return obs, reward - self.per_step_penalty, terminated, truncated, info

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        return super().reset(**kwargs)

    def _place_building(self, col, row, width, height, obj_type=Lava):
        for i in range(col, width + col):
            self.grid.vert_wall(i, row, height, obj_type=obj_type)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        for i in range(1, height - 1):
            self.grid.horz_wall(1, i, width-2, obj_type=SlipperyNorthEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))

        self._place_building(7, 1, 4, 1)
        self.grid.vert_wall(6, 1, 1, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(7, 2, 4, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(11, 1, 1, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))

        self._place_building(4, 5, 2, 1)
        self.grid.vert_wall(3, 5, 1, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(6, 5, 1, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(4, 4, 2, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(4, 6, 2, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))

        self._place_building(12, 7, 2, 3)
        self.grid.vert_wall(11, 7, 3, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(11, 6, 3, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(11, 10, 3, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))


        self._place_building(4, 10, 2, 2)
        self.grid.vert_wall(3, 10, 2, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(6, 10, 2, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(4, 12, 2, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(4, 9, 2, obj_type=SlipperyWest("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(5, 7, 3, obj_type=SlipperySouth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))


        #should spawn randomly
        x = np.random.choice([1,2,3,6,7,8,9])
        y = np.random.choice([6,7,8])
        self.add_adversary(x, y, "blue", direction=1, tasks=[FollowAgent("red", duration=2), DoRandom(duration=2)], repeating=True)

        self.place_agent(agent_pos=np.array((1, height -2)), agent_dir=0, spawn_on_slippery=True)
        self.place_goal(np.array((width - 2, 1)))
        if self.dense_rewards: self.run_bfs()

class WindyCity2Env(MiniGridEnv):
    def __init__(self,
                randomize_start=True, size=10,
                width=27,
                height=22,
                probability_intended=8/9,
                probability_turn_intended=8/9,
                obstacle_type=Lava,
                goal_reward=1,
                failure_penalty=-1,
                per_step_penalty=0,
                dense_rewards=False,
                two_player_winning_region_start=False,
                     **kwargs):

        self.obstacle_type = obstacle_type
        self.size = size
        self.probability_intended = probability_intended
        self.probability_turn_intended = probability_turn_intended

        if width is not None and height is not None:
            self.width = width
            self.height = height
        elif size is not None:
            self.width = size
            self.height = size
        else:
            raise ValueError(f"Please define either width and height or a size for square environments. The set values are width: {width}, height: {height}, size: {size}.")

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=200,
            # Set this to True for maximum speed
            see_through_walls=False,
            mission_space = mission_space,
            **kwargs
        )

        self.randomize_start = randomize_start
        self.two_player_winning_region_start = two_player_winning_region_start
        self.goal_reward = goal_reward
        self.failure_penalty = failure_penalty
        self.dense_rewards = dense_rewards
        self.per_step_penalty = per_step_penalty

        self.trajectory = list()

    @staticmethod
    def _gen_mission():
        return "Finish your task while avoiding the adversaries"

    def disable_random_start(self):
        self.randomize_start = False

    def place_agent(self, spawn_on_slippery=False, agent_pos=None, agent_dir=0):
        max_tries = 10_000
        num_tries = 0

        if self.two_player_winning_region_start == True:
            winning_region = list()
            winning_region += product([1,2,3,4], [y for y in range(1, self.height-1)])
            winning_region += product([x for x in range(1,12)], [1])
            winning_region += product([x for x in range(1,self.width-10)], [self.height-2])
            winning_region += product([x for x in range(self.width-6, self.width-1)], [1,2,3,4])
            winning_region += product([x for x in range(self.width-11, self.width-1)], [5])
            x, y= winning_region[np.random.choice(len(winning_region), 1)[0]]
            self.agent_pos = np.array((x,y))
            self.agent_dir = np.random.randint(0, 4)
            self.trajectory.append((self.agent_pos, self.agent_dir))
            return

        if self.randomize_start == True:
            while True:
                num_tries += 1
                if num_tries > max_tries:
                    raise RecursionError("rejection sampling failed in place_obj")
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)

                cell = self.grid.get(*(x,y))
                if ( cell is None or
                    (cell.can_overlap() and
                        not isinstance(cell, Lava) and
                        not isinstance(cell, Goal) and
                        (spawn_on_slippery or not is_slippery(cell)) and
                        not (x in [7, 8, 9, 10] and y in [9, 10]))
                    ):
                    self.agent_pos = np.array((x, y))
                    self.agent_dir = np.random.randint(0, 4)
                    break
        elif agent_dir is None:
            self.agent_pos = np.array((1, 1))
            self.agent_dir = 0
        else:
            self.agent_pos = agent_pos
            self.agent_dir = agent_dir
        self.trajectory.append((self.agent_pos, self.agent_dir))

    def place_goal(self, goal_pos):
        self.goal_pos = goal_pos
        self.put_obj(Goal(), *self.goal_pos)


    def printGrid(self, init=False):
        grid = super().printGrid(init)

        properties_str = ""

        properties_str += F"ProbTurnIntended:{self.probability_turn_intended}\n"
        properties_str += F"ProbForwardIntended:{self.probability_intended}\n"

        return grid + properties_str

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.trajectory.append((action, self.agent_pos, self.agent_dir))
        if truncated and info["ran_into_lava"]:
            print(self.trajectory)
            print("truncated: ", info)
            self.trajectory = list()
        if truncated and info["reached_goal"]:
            print("truncated: ", info)
            self.trajectory = list()
        elif terminated and info["ran_into_lava"]:
            print(self.trajectory)
            print("terminated: ", info)
            self.trajectory = list()
        elif terminated:
            print("terminated: ", info)
            self.trajectory = list()
        elif truncated:
            print("truncated: ", info)
            self.trajectory = list()
        return obs, reward - self.per_step_penalty, terminated, truncated, info

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        return super().reset(**kwargs)

    def _place_building(self, col, row, width, height, obj_type=Lava):
        for i in range(col, width + col):
            self.grid.vert_wall(i, row, height, obj_type=obj_type)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        for i in range(1, height - 1):
            self.grid.horz_wall(1, i, width-2, obj_type=SlipperyNorthEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))

        self.grid.horz_wall(1, 17, 15, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(1, 18, 16, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(1, 19, 17, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(1, 20, 18, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))

        self.grid.horz_wall(1, 7,  9, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(1, 8,  8, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(1, 9,  8, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(1, 10, 7, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))

        self._place_building(16, 1, 4, 2)
        self.grid.vert_wall(15, 1, 2, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(16, 3, 4, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(20, 1, 2, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))

        self._place_building(10,  3, 3, 4)
        #self.grid.vert_wall(9, 3, 4, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(13, 3, 4, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(10, 2, 3, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(10, 7, 3, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))

        self._place_building(16, 7, 8, 5)
        self.grid.vert_wall(15, 7, 4, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        #self.grid.vert_wall(17, 9, 3, obj_type=Lava)
        self.grid.vert_wall(24, 7, 5, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(15, 9, 3, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(16, 6, 7, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(16, 12, 7, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(22, 12, 1, obj_type=SlipperyNorthEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(23, 13, 1, obj_type=SlipperyNorthEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))


        self._place_building(8, 11, 5, 6)
        #self.grid.vert_wall(7, 11, 6, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(13, 11, 6, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(8, 17, 5, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(9, 10, 4, obj_type=SlipperyWest("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(10, 9, 3, obj_type=SlipperyWest("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.vert_wall(12, 7, 4, obj_type=SlipperySouth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))

        self._place_building(22, 14, 4, 4)
        self.grid.vert_wall(21, 14, 4, obj_type=SlipperyNorth("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(22, 13, 4, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
        self.grid.horz_wall(22, 18, 4, obj_type=SlipperyEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))


        #self.grid.vert_wall(22, 13, 1, obj_type=SlipperyNorthEast("white", probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))



        self.place_agent(agent_pos=np.array((18, height - 4)), agent_dir=3, spawn_on_slippery=True)
        self.place_goal(np.array((width - 2, 10)))
        if self.dense_rewards: self.run_bfs()





class WindyCitySmallAdv(AdversaryEnv):
    def __init__(self,
                randomize_start=True, size=10,
                width=11,
                height=9,
                probability_intended=1,
                probability_turn_intended=1,
                obstacle_type=Lava,
                goal_reward=1,
                failure_penalty=-1,
                per_step_penalty=0,
                dense_rewards=False,
                two_player_winning_region_start=False,
                     **kwargs):

        self.obstacle_type = obstacle_type
        self.size = size
        self.probability_intended = probability_intended
        self.probability_turn_intended = probability_turn_intended

        if width is not None and height is not None:
            self.width = width
            self.height = height
        elif size is not None:
            self.width = size
            self.height = size
        else:
            raise ValueError(f"Please define either width and height or a size for square environments. The set values are width: {width}, height: {height}, size: {size}.")

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=50,
            # Set this to True for maximum speed
            see_through_walls=False,
            #mission_space = mission_space,
            **kwargs
        )

        self.randomize_start = randomize_start
        self.two_player_winning_region_start = two_player_winning_region_start
        self.goal_reward = goal_reward
        self.failure_penalty = failure_penalty
        self.dense_rewards = dense_rewards
        self.per_step_penalty = per_step_penalty

        self.trajectory = list()

    @staticmethod
    def _gen_mission():
        return "Finish your task while avoiding the adversaries"

    def disable_random_start(self):
        self.randomize_start = False

    def place_agent(self, spawn_on_slippery=False, agent_pos=None, agent_dir=0):
        max_tries = 10_000
        num_tries = 0

        if self.randomize_start == True:
            while True:
                num_tries += 1
                if num_tries > max_tries:
                    raise RecursionError("rejection sampling failed in place_obj")
                x = np.random.randint(0, self.width)
                y = np.random.randint(5, self.height)

                cell = self.grid.get(*(x,y))
                if ( cell is None or
                    (cell.can_overlap() and
                        not isinstance(cell, Lava) and
                        not isinstance(cell, Goal) and
                        (spawn_on_slippery or not is_slippery(cell)) and
                        not (x in [7, 8, 9, 10] and y in [9, 10]))
                    ):
                    self.agent_pos = np.array((x, y))
                    self.agent_dir = np.random.randint(0, 4)
                    break
        elif agent_dir is None:
            self.agent_pos = np.array((1, 1))
            self.agent_dir = 0
        else:
            self.agent_pos = agent_pos
            self.agent_dir = agent_dir
        self.trajectory.append((self.agent_pos, self.agent_dir))

    def place_goal(self, goal_pos):
        self.goal_pos = goal_pos
        self.put_obj(Goal(), *self.goal_pos)


    def printGrid(self, init=False):
        grid = super().printGrid(init)

        properties_str = ""

        properties_str += F"ProbTurnIntended:{self.probability_turn_intended}\n"
        properties_str += F"ProbForwardIntended:{self.probability_intended}\n"

        return grid + properties_str

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.trajectory.append((action, self.agent_pos, self.agent_dir))
        if truncated and info["ran_into_lava"]:
            print(self.trajectory)
            print("truncated: ", info)
            self.trajectory = list()
        if truncated and info["reached_goal"]:
            print("truncated: ", info)
            self.trajectory = list()
        elif terminated and info["ran_into_lava"]:
            print(self.trajectory)
            print("terminated: ", info)
            self.trajectory = list()
        elif terminated and info["collision"]:
            print(self.trajectory)
            print("terminated: ", info)
            self.trajectory = list()
        elif terminated:
            print("terminated: ", info)
            self.trajectory = list()
        elif truncated:
            print("truncated: ", info)
            self.trajectory = list()
        return obs, reward - self.per_step_penalty, terminated, truncated, info

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        return super().reset(**kwargs)

    def _place_building(self, col, row, width, height, obj_type=Lava):
        for i in range(col, width + col):
            self.grid.vert_wall(i, row, height, obj_type=obj_type)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)


        self._place_building(3, 3, 5, 2)
        blue_adv = self.add_adversary(2, 4, "blue", direction=3, tasks=
                                      [GoTo((2,2)), GoTo((8,2)), GoTo((8,4)), GoTo((8,2)), GoTo((2,2)), GoTo((2,4))], repeating=True)


        self.place_agent(agent_pos=np.array((5, 5)), agent_dir=3, spawn_on_slippery=True)
        self.place_goal(np.array((width//2, 1)))
        if self.dense_rewards: self.run_bfs()



