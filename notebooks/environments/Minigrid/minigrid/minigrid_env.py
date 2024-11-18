from __future__ import annotations

import hashlib
import math
from abc import abstractmethod
from typing import Any, Iterable, SupportsFloat, TypeVar

import numpy

import gymnasium as gym
import numpy as np
import pygame
import pygame.freetype
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS, OBJECT_TO_STR
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Point, WorldObj, Slippery, SlipperyEast, SlipperyNorth, SlipperySouth, SlipperyWest, Lava, SlipperyNorthWest, SlipperyNorthEast, SlipperySouthWest, SlipperySouthEast
from minigrid.core.adversary import Adversary
from minigrid.core.tasks import DoRandom, Task, List
from minigrid.core.state import State

from collections import deque

T = TypeVar("T")

stay_at_pos_distribution = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

def is_slippery(cell : WorldObj):
    return isinstance(cell, (SlipperySouth, Slippery, SlipperyEast, SlipperyWest, SlipperyNorth, SlipperyNorthWest, SlipperySouthEast, SlipperyNorthEast, SlipperySouthWest))

class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        mission_space: MissionSpace,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
        **kwargs
    ):
        # Initialize mission
        self.mission = mission_space.sample()
        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size
        assert width is not None and height is not None

        # Action enumeration for this environment
        self.actions = Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "direction": spaces.Discrete(4),
                "mission": mission_space,
            }
        )

        # Range of possible rewards
        self.reward_range = (0, 1)

        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None

        # Environment configuration
        self.width = width
        self.height = height

        assert isinstance(
            max_steps, int
        ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
        self.max_steps = max_steps

        self.see_through_walls = see_through_walls

        # Current position and direction of the agent
        self.agent_pos: np.ndarray | tuple[int, int] = None
        self.agent_dir: int = None

        # Current grid and mission and carrying
        self.grid = Grid(width, height)
        self.carrying = None
        self.objects = list()
        self.doors = list()

        # dict of adversaries
        self.adversaries = dict()

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov

        # Custom
        self.background_tiles = dict()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1
        self.goal_pos  = (-1, -1)
        # Generate a new random grid at the start of each episode

        self.objects.clear()
        self.doors.clear()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )


        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, {}


    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode().tolist(), self.agent_pos, self.agent_dir]
        to_encode += [(adv.adversary_pos, adv.adversary_dir, adv.color) for adv in self.adversaries]
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        return sample_hash.hexdigest()[:size]

    def add_adversary(
       self,
       i: int,
       j: int,
       color: str,
       direction: int = 0,
       tasks: List[Task] = [DoRandom()],
       repeating=False
    ):
       """
       Adds an adversary to the grid
       """

       adv = Adversary((i,j), direction, color, tasks=tasks, repeating=repeating)
       self.adversaries[color] = adv
       return adv


    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def pprint_grid(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """
        if self.agent_pos is None or self.agent_dir is None or self.grid is None:
            raise ValueError(
                "The environment hasn't been `reset` therefore the `agent_pos`, `agent_dir` or `grid` are unknown."
            )

        # Map of object types to short string

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        output = ""

        for j in range(self.grid.height):
            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    output += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                tile = self.grid.get(i, j)

                if tile is None:
                    output += "  "
                    continue

                if tile.type == "door":
                    if tile.is_open:
                        output += "__"
                    elif tile.is_locked:
                        output += "L" + tile.color[0].upper()
                    else:
                        output += "D" + tile.color[0].upper()
                    continue

                output += OBJECT_TO_STR[tile.type] + tile.color[0].upper()

            if j < self.grid.height - 1:
                output += "\n"

        return output

    def printGrid(self, init=False):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """
        if init:
            self._gen_grid(self.grid.width, self.grid.height) # todo need to add this for minigrid2prism
        #print("Dimensions: {} x {}".format(self.grid.height, self.grid.width))
        #self._gen_grid(self.grid.width, self.grid.height)
        # Map of object types to short string

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        str = ""
        background_str = ""
        adversaries = {adv.adversary_pos: adv for adv in self.adversaries.values()} if self.adversaries else {}
        bfs_rewards = []

        for j in range(self.grid.height):
            for i in range(self.grid.width):
                b = self.grid.get_background(i, j)
                c = self.grid.get(i, j)

                if (i,j) in adversaries.keys():
                    a = adversaries[(i,j)]
                    str += OBJECT_TO_STR["adversary"] + a.color[0].upper()
                    if init:
                        background_str += "  "
                    continue

                if init:
                    if c and c.type == "wall":
                        background_str += OBJECT_TO_STR[c.type] + c.color[0].upper()
                    elif c and c.type in ["slipperynorth", "slipperyeast", "slipperysouth", "slipperywest", "slipperynorthwest", "slipperynortheast", "slipperysoutheast", "slipperysouthwest"]:
                         background_str += OBJECT_TO_STR[c.type] + c.color[0].upper()
                    elif b is None:
                        background_str += "  "
                    else:
                        if b.type != "floor":
                            type_str = OBJECT_TO_STR[b.type]
                        else:
                            type_str = " "

                        background_str += type_str + b.color.replace("light","")[0].upper()

                    if hasattr(self, "bfs_reward") and self.bfs_reward:
                         bfs_rewards.append(f"{i};{j};{self.bfs_reward[i + self.grid.width * j]}")

                if self.agent_pos is not None and i == self.agent_pos[0] and j == self.agent_pos[1]:

                    if init:
                        str += "XR"
                    else:
                        str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue


                if c is None:
                    str += "  "
                    continue

                if c.type == "door":
                    if c.is_open:
                        str += "__"
                    elif c.is_locked:
                        str += "L" + c.color[0].upper()
                    else:
                        str += "D" + c.color[0].upper()
                    continue


                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += "\n"
                if init:
                    background_str += "\n"


        seperator = "-" * self.grid.width * 2

        if init and hasattr(self, "bfs_reward") and self.bfs_reward:
            return str + "\n" + seperator + "\n" + background_str + "\n" + seperator + "\n" + ";".join(bfs_rewards) + "\n" + seperator + "\n"
        else:
            return str + "\n" + seperator + "\n" + background_str + "\n" + seperator + "\n" + seperator + "\n"

    def export_grid(self, filename="grid.txt"):
        with open(filename, "w") as gridFile:
            gridFile.write(self.printGrid(init=True))

    @abstractmethod
    def _gen_grid(self, width, height):
        pass

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low: int, high: int) -> int:
        """
        Generate random integer in [low,high[
        """

        return self.np_random.integers(low, high)

    def _rand_float(self, low: float, high: float) -> float:
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self) -> bool:
        """
        Generate random boolean value
        """

        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> list[T]:
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out: list[T] = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self) -> str:
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(
        self, x_low: int, x_high: int, y_low: int, y_high: int
    ) -> tuple[int, int]:
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.integers(x_low, x_high),
            self.np_random.integers(y_low, y_high),
        )

    def place_obj(
        self,
        obj: WorldObj | None,
        top: Point = None,
        size: tuple[int, int] = None,
        reject_fn=None,
        max_tries=math.inf,
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos = (
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
            )

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj: WorldObj, i: int, j: int):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)
        if obj.can_pickup():
            self.objects.append(obj)
            self.objects = sorted(self.objects, key=lambda object: object.color)
        if obj.type == "door":
            self.doors.append(obj)
            self.doors = sorted(self.doors, key=lambda object: object.color)

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos = (-1, -1)
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)

        return pos

    def disable_random_start(self):
        pass

    def add_slippery_tile(self, i: int, j: int, type: str):
            """
            Adds a slippery tile to the grid
            """

            if type=="slipperynorth":
                slippery_tile = SlipperyNorth()
            elif type=="slipperysouth":
                slippery_tile = SlipperySouth()
            elif type=="slipperyeast":
                slippery_tile = SlipperyEast()
            elif type=="slipperywest":
                slippery_tile = SlipperyWest()
            else:
                slippery_tile = SlipperyNorth()

            self.grid.set(i, j, slippery_tile)
            return (i, j)


    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert (
            self.agent_dir >= 0 and self.agent_dir < 4
        ), f"Invalid agent_dir: {self.agent_dir} is not within range(0, 4)"
        return DIR_TO_VEC[self.agent_dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos + self.dir_vec

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.agent_pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self, agent_view_size=None):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        if agent_view_size is None, use self.agent_view_size
        """

        agent_view_size = agent_view_size or self.agent_view_size

        # Facing right
        if self.agent_dir == 0:
            topX = self.agent_pos[0]
            topY = self.agent_pos[1] - agent_view_size // 2
        # Facing down
        elif self.agent_dir == 1:
            topX = self.agent_pos[0] - agent_view_size // 2
            topY = self.agent_pos[1]
        # Facing left
        elif self.agent_dir == 2:
            topX = self.agent_pos[0] - agent_view_size + 1
            topY = self.agent_pos[1] - agent_view_size // 2
        # Facing up
        elif self.agent_dir == 3:
            topX = self.agent_pos[0] - agent_view_size // 2
            topY = self.agent_pos[1] - agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return topX, topY, botX, botY

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size:
            return None

        return vx, vy

    def get_neighbours(self, i, j):
        neighbours = list()
        potential_neighbours = [(i-1,j), (i,j+1), (i+1,j), (i,j-1)]
        for n in potential_neighbours:
            cell = self.grid.get(*n)
            if cell is None or (cell.can_overlap()): #and not isinstance(cell, Lava)):
                neighbours.append(n)

        return neighbours

    def run_BFS_reward(grid):
        if not hasattr(grid, "goal_pos") or np.all(grid.goal_pos == (-1, -1)):
             return []

        starting_position = (grid.goal_pos[0], grid.goal_pos[1])
        max_distance = 0
        distances = [None] * grid.width * grid.height
        bfs_queue = deque([starting_position])
        traversed_cells = set()

        distances[starting_position[0] + grid.width * starting_position[1]] = 0
        while bfs_queue:
            current_cell = bfs_queue.pop()
            if current_cell in traversed_cells: continue
            traversed_cells.add(current_cell)
            current_distance = distances[current_cell[0] + grid.width * current_cell[1]]
            if current_distance > max_distance:
                max_distance = current_distance
            for neighbour in grid.get_neighbours(*current_cell):
                if neighbour in traversed_cells:
                    continue
                bfs_queue.appendleft(neighbour)
                if distances[neighbour[0] + grid.width * neighbour[1]] is None:
                    distances[neighbour[0] + grid.width * neighbour[1]] = current_distance + 1

        distances = [x if x else 0 for x in distances]
        # return [ (-x/1) for x in distances]
        return [ (1/4)* (-x/max_distance) if x != 0 else 0 for x in distances]

    def print_bfs_reward(self):
        rep = ""
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                rep += F"{self.bfs_reward[j * self.grid.height + i]:5.2f} "


            rep += '\n'

        print(rep)


    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()

        obs_grid, _ = Grid.decode(obs["image"])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        assert world_cell is not None

        return obs_cell is not None and obs_cell.type == world_cell.type

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False
        info = dict()
        need_position_update = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        current_cell = self.grid.get(*self.agent_pos)

        opened_door = False
        picked_up = False
        if action == self.actions.forward and is_slippery(current_cell):
            probabilities = current_cell.get_probabilities(self.agent_dir)
            possible_fwd_pos, prob = self.get_neighbours_prob(self.agent_pos, probabilities)
            fwd_pos_index = np.random.choice(len(possible_fwd_pos), 1, p=prob)
            fwd_pos = possible_fwd_pos[fwd_pos_index[0]]
            fwd_cell = self.grid.get(*fwd_pos)
            need_position_update = True
        # Rotate left
        elif action == self.actions.left:
            if is_slippery(current_cell):
                possible_fwd_pos, prob = self.get_neighbours_prob(self.agent_pos, current_cell.probabilities_turn)
                fwd_pos_index = np.random.choice(len(possible_fwd_pos), 1, p=prob)
                fwd_pos = possible_fwd_pos[fwd_pos_index[0]]
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_pos == (self.agent_pos[0], self.agent_pos[1]):
                    self.agent_dir -= 1
                    if self.agent_dir < 0:
                        self.agent_dir += 4
                else:
                    need_position_update = True
            else:
                self.agent_dir -= 1
                if self.agent_dir < 0:
                    self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            if is_slippery(current_cell):
                possible_fwd_pos, prob = self.get_neighbours_prob(self.agent_pos, current_cell.probabilities_turn)
                fwd_pos_index = np.random.choice(len(possible_fwd_pos), 1, p=prob)
                fwd_pos = possible_fwd_pos[fwd_pos_index[0]]
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_pos == (self.agent_pos[0], self.agent_pos[1]):
                    self.agent_dir = (self.agent_dir + 1) % 4
                else:
                    need_position_update = True
            else:
                self.agent_dir = (self.agent_dir + 1) % 4


        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
                fwd_cell = self.grid.get(*fwd_pos)
                need_position_update = True


        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)
                    picked_up = True

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)
                if fwd_cell.type == "door" and fwd_cell.is_open:
                    opened_door = True

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if need_position_update and (fwd_cell is None or fwd_cell.can_overlap()):
            self.agent_pos = tuple(fwd_pos)

        current_cell = self.grid.get(*self.agent_pos)

        collision = False
        if self.adversaries:
            for adversary in self.adversaries.values():
                if np.array_equal(self.agent_pos, adversary.adversary_pos):
                    collision = True


        reached_goal = False
        ran_into_lava = False
        if current_cell is not None and current_cell.type == "goal":
            terminated = True
            reached_goal = True
            try: reward = self.goal_reward
            except: reward = 1
        elif current_cell is not None and current_cell.type == "lava":
            terminated = True
            ran_into_lava = True
            try: reward = self.failure_penalty
            except: reward = -1
        elif collision:
            terminated = True
            try: reward = self.collision_penalty
            except: reward = -1
            self.agent_pos = tuple(fwd_pos)
        else:
            try: reward += self.bfs_reward[self.agent_pos[0] + self.grid.width * self.agent_pos[1]]
            except: pass

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        info["reached_goal"] = reached_goal
        info["ran_into_lava"] = ran_into_lava
        info["opened_door"] = opened_door
        info["picked_up"] = picked_up
        #if terminated:
        #    print(f"Terminated at: {self.agent_pos} {self.grid.get(*self.agent_pos)} {info}")
        if len(self.adversaries) > 0: info["collision"] = collision
        obs = self.gen_obs()
        return obs, reward, terminated, truncated, info

    def get_neighbours_prob(self, agent_pos, probabilities):
        neighbours = [tuple((x,y)) for x in range(agent_pos[0]-1, agent_pos[0]+2) for y in range(agent_pos[1]-1,agent_pos[1]+2)]
        probabilities_dict = dict(zip(neighbours, probabilities))

        for pos in probabilities_dict:
            cell = self.grid.get(*pos)
            if cell is not None and not cell.can_overlap():
                probabilities_dict[pos] = 0.0
        try:
            return list(probabilities_dict.keys()), [float(p) / sum(probabilities_dict.values()) for p in probabilities_dict.values()]
        except ZeroDivisionError as e:
            return list(probabilities_dict.keys()), stay_at_pos_distribution

    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {"image": image, "direction": self.agent_dir, "mission": self.mission}

        return obs

    def get_pov_render(self, tile_size):
        """
        Render an agent's POV observation for visualization
        """
        grid, vis_mask = self.gen_obs_grid()

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            adversaries=self.adversaries.values(),
            highlight_mask=vis_mask,
        )

        return img

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """
        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = (
            self.agent_pos
            + f_vec * (self.agent_view_size - 1)
            - r_vec * (self.agent_view_size // 2)
        )

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            adversaries=self.adversaries.values() if self.adversaries else [],
            highlight_mask=highlight_mask if highlight else None,
        )

        return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """

        if agent_pov:
            return self.get_pov_render(tile_size)
        else:
            return self.get_full_render(highlight, tile_size)

    def render(self):
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        screen_width = 2 * self.tile_size * self.grid.width
        screen_height = 2 * self.tile_size * self.grid.height
        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (screen_width, screen_height)
                )
                pygame.display.set_caption("minigrid")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            offset = 0
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (screen_width, screen_height))

            #font_size = 22
            #text = self.mission
            #font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            #text_rect = font.get_rect(text, size=font_size)
            #text_rect.center = bg.get_rect().center
            #text_rect.y = bg.get_height() - font_size * 1.5
            #font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img

    def get_symbolic_state(self):
        adversaries = tuple()
        balls = tuple()
        keys = tuple()
        boxes = tuple()
        doors = tuple()

        for obj in self.objects:
            if obj.type == "box":
                boxes += (obj.to_state(),)
            if obj.type == "ball":
                balls += (obj.to_state(),)
            if obj.type == "key":
                keys += (obj.to_state(),)
        for door in self.doors:
            doors += (door.to_state(),)


        for color in COLOR_NAMES:
            try:
                adversaries += (self.adversaries[color].to_state(),)
            except Exception as e:
                pass


        carrying = "" if not self.carrying else f"{self.carrying.color.capitalize()}{self.carrying.type.capitalize()}"
        state = State(colAgent=self.agent_pos[0], rowAgent=self.agent_pos[1], viewAgent=self.agent_dir, carrying=carrying, adversaries=adversaries, keys=keys, doors=doors)
        return state

    def close(self):
        if self.window:
            pygame.quit()
