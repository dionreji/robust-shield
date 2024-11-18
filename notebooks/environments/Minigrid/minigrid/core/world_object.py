from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
from enum import Enum

import numpy as np
import math

from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
    rotate_fn,
)
from minigrid.core.state import KeyState, BallState, BoxState, DoorState

if TYPE_CHECKING:
    from minigrid.minigrid_env import MiniGridEnv

Point = Tuple[int, int]


class WorldObj:

    """
    Base class for grid world objects
    """

    def __init__(self, type: str, color: str):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos: Point | None = None

        # Current position of the object
        self.cur_pos: Point | None = None

    def can_overlap(self) -> bool:
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self) -> bool:
        """Can the agent pick this up?"""
        return False

    def can_contain(self) -> bool:
        """Can this contain another object?"""
        return False

    def see_behind(self) -> bool:
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env: MiniGridEnv, pos: tuple[int, int]) -> bool:
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self) -> tuple[int, int, int]:
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx: int, color_idx: int, state: int) -> WorldObj | None:
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == "empty" or obj_type == "unseen":
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == "wall":
            v = Wall(color)
        elif obj_type == "floor":
            v = Floor(color)
        elif obj_type == "ball":
            v = Ball(color)
        elif obj_type == "key":
            v = Key(color)
        elif obj_type == "box":
            v = Box(color)
        elif obj_type == "door":
            v = Door(color, is_open, is_locked)
        elif obj_type == "goal":
            v = Goal()
        elif obj_type == "lava":
            v = Lava()
        elif obj_type == "slippery":
            v = Slippery(color)
        elif obj_type == "slipperyeast":
            v = SlipperyEast(color)
        elif obj_type == "slipperysouth":
            v = SlipperySouth(color)
        elif obj_type == "slipperywest":
            v = SlipperyWest(color)
        elif obj_type == "slipperynorth":
            v = SlipperyNorth(color)
        elif obj_type == "slipperynortheast":
            v = SlipperyNorthEast(color)
        elif obj_type == "slipperynorthwest":
            v = SlipperyNorthWest(color)
        elif obj_type == "slipperysoutheast":
            v = SlipperySouthEast(color)
        elif obj_type == "slipperysouthwest":
            v = SlipperySouthEast(color)
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r: np.ndarray) -> np.ndarray:
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Slippery(WorldObj):
    def __init__(self, color: str = "blue"):
        super().__init__("slippery", color)

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(6):
            ylo = 0.1  + 0.15 * i
            yhi = 0.20 + 0.15 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))

class SlipperySouth(WorldObj):
    def __init__(self, color: str = "blue", probability_intended=3/9, probability_turn_intended=6/9):
        super().__init__("slipperysouth", color)
        self.direction = 1

        # Field probabilties are stored in the order:
        # 0: Left Above - 1: Left - 2: Left Below
        # 3: Above - 4: Current - 5: Below
        # 6: Right Above - 7: Right - 8: Right Below
        self.probabilities_turn = [0.0, 0.0, 0.0, 0.0, probability_turn_intended, 1 - probability_turn_intended, 0.0, 0.0, 0.0]
        probability_displacement = 1 - probability_intended
        self.probabilities_0 =   [0, 0, 0, 0, 0, 1.0, 0, 0, 0]
        self.probabilities_90 =  [0, probability_intended, probability_displacement, 0, 0, 0, 0, 0, 0]
        self.probabilities_180 = [0, 0, 0, probability_intended, probability_displacement, 0, 0,  0, 0]
        self.probabilities_270 = [0, 0, 0, 0, 0, 0, 0, probability_intended, probability_displacement]


    def get_probabilities(self, agent_dir):
        if agent_dir == self.direction:
            return self.probabilities_0
        elif agent_dir == 2:
            return self.probabilities_90
        elif agent_dir == 3:
            return self.probabilities_180
        elif agent_dir == 0:
            return self.probabilities_270
        else:
            raise NotImplementedError("Agent directory not implemented")

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(6):
            ylo = 0.1  + 0.15 * i
            yhi = 0.20 + 0.15 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))

class SlipperyNorthWest(WorldObj):
    def __init__(self, color: str = "blue", probability_intended=6/9, probability_turn_intended=6/9):
        super().__init__("slipperynorthwest", color)

        # Field probabilties are stored in the order:
        # 0: Left Above - 1: Left - 2: Left Below
        # 3: Above - 4: Current - 5: Below
        # 6: Right Above - 7: Right - 8: Right Below

        self.probabilities_turn = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        pi = probability_intended
        pd = 1 - probability_intended
        self.probabilities_north = [pd, 0,  0, pi, 0, 0, 0, 0, 0]
        self.probabilities_west  = [pd, pi, 0, 0,  0, 0, 0, 0, 0]
        self.probabilities_south = [pd/3, pd/3, pd/3, 0, 0, pi, 0, 0, 0]
        self.probabilities_east  = [pd/3, 0, 0, pd/3, 0, 0, pd/3, pi, 0]

    def get_probabilities(self, agent_dir):
        if agent_dir == 3:
            return self.probabilities_north
        elif agent_dir == 2:
            return self.probabilities_west
        elif agent_dir == 1:
            return self.probabilities_south
        elif agent_dir == 0:
            return self.probabilities_east
        else:
            raise NotImplementedError("Agent directory not implemented")

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(6):
            ylo = 0.1  + 0.15 * i
            yhi = 0.20 + 0.15 * i
            fill_coords(img, rotate_fn(point_in_line(0.1, ylo, 0.3, yhi, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.3, yhi, 0.5, ylo, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))

class SlipperyNorthEast(WorldObj):
    def __init__(self, color: str = "blue", probability_intended=6/9, probability_turn_intended=6/9):
        super().__init__("slipperynortheast", color)

        # Field probabilties are stored in the order:
        # 0: Left Above - 1: Left - 2: Left Below
        # 3: Above - 4: Current - 5: Below
        # 6: Right Above - 7: Right - 8: Right Below

        self.probabilities_turn = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        pi = probability_intended
        pd = 1 - probability_intended

        self.probabilities_north = [0, 0, 0, pi, 0, 0, pd, 0, 0]
        self.probabilities_east  = [0, 0, 0, 0, 0, 0, pd, pi, 0]
        self.probabilities_west  = [pd/3, pi, 0, pd/3, 0, 0, pd/3, 0, 0]
        self.probabilities_south = [0, 0, 0, 0, 0, pi, pd/3, pd/3, pd/3]

    def get_probabilities(self, agent_dir):
        if agent_dir == 3:
            return self.probabilities_north
        elif agent_dir == 2:
            return self.probabilities_west
        elif agent_dir == 1:
            return self.probabilities_south
        elif agent_dir == 0:
            return self.probabilities_east
        else:
            raise NotImplementedError("Agent directory not implemented")

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        fill_coords(img, point_in_rect(0.1, 0.75, 0.2, 0.26), (0, 0, 0))
        fill_coords(img, point_in_rect(0.69, 0.75, 0.2, 0.75), (0, 0, 0))

        fill_coords(img, point_in_rect(0.1, 0.65, 0.3, 0.36), (0, 0, 0))
        fill_coords(img, point_in_rect(0.59, 0.65, 0.3, 0.75), (0, 0, 0))

        fill_coords(img, point_in_rect(0.1, 0.55, 0.4, 0.46), (0, 0, 0))
        fill_coords(img, point_in_rect(0.49, 0.55, 0.4, 0.75), (0, 0, 0))

class SlipperySouthWest(WorldObj):
    def __init__(self, color: str = "blue", probability_intended=6/9, probability_turn_intended=6/9):
        super().__init__("slipperysouthwest", color)

        # Field probabilties are stored in the order:
        # 0: Left Above - 1: Left - 2: Left Below
        # 3: Above - 4: Current - 5: Below
        # 6: Right Above - 7: Right - 8: Right Below

        self.probabilities_turn = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        pi = probability_intended
        pd = 1 - probability_intended
        self.probabilities_south = [0, 0, pd, 0, 0, pi, 0, 0, 0]
        self.probabilities_west  = [0, pi, pd, 0, 0, 0, 0, 0, 0]
        self.probabilities_east  = [0, 0, pd/3, 0, 0, pd/3, 0, pi, pd/3]
        self.probabilities_north = [pd/3, pd/3, pd/3, pi, 0, 0, 0, 0, 0]

    def get_probabilities(self, agent_dir):
        if agent_dir == 3:
            return self.probabilities_north
        elif agent_dir == 2:
            return self.probabilities_west
        elif agent_dir == 1:
            return self.probabilities_south
        elif agent_dir == 0:
            return self.probabilities_east
        else:
            raise NotImplementedError("Agent directory not implemented")

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(6):
            ylo = 0.1  + 0.15 * i
            yhi = 0.20 + 0.15 * i
            fill_coords(img, rotate_fn(point_in_line(0.1, ylo, 0.3, yhi, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.3, yhi, 0.5, ylo, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))

class SlipperySouthEast(WorldObj):
    def __init__(self, color: str = "blue", probability_intended=6/9, probability_turn_intended=6/9):
        super().__init__("slipperysoutwest", color)

        # Field probabilties are stored in the order:
        # 0: Left Above - 1: Left - 2: Left Below
        # 3: Above - 4: Current - 5: Below
        # 6: Right Above - 7: Right - 8: Right Below

        self.probabilities_turn = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        pi = probability_intended
        pd = 1 - probability_intended
        self.probabilities_south = [0, 0, 0, 0, 0, pi, 0, 0, pd]
        self.probabilities_east  = [0, 0, 0, 0, 0, 0, 0, pi, pd]
        self.probabilities_north = [0, 0, 0, pi, 0, 0, pd/3, pd/3, pd/3]
        self.probabilities_west  = [0, pi, pd/3, 0, 0, pd/3, 0, 0, pd/3]

    def get_probabilities(self, agent_dir):
        if agent_dir == 3:
            return self.probabilities_north
        elif agent_dir == 2:
            return self.probabilities_west
        elif agent_dir == 1:
            return self.probabilities_south
        elif agent_dir == 0:
            return self.probabilities_east
        else:
            raise NotImplementedError("Agent directory not implemented")

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(6):
            ylo = 0.1  + 0.15 * i
            yhi = 0.20 + 0.15 * i
            fill_coords(img, rotate_fn(point_in_line(0.1, ylo, 0.3, yhi, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.3, yhi, 0.5, ylo, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.5, ylo, 0.7, yhi, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.7, yhi, 0.9, ylo, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))

class SlipperyNorth(WorldObj):
    def __init__(self, color: str = "blue", probability_intended=6/9, probability_turn_intended=6/9):
        super().__init__("slipperynorth", color)
        self.offset = (0,-1)
        self.direction = 3

        # Field probabilties are stored in the order:
        # 0: Left Above - 1: Left - 2: Left Below
        # 3: Above - 4: Current - 5: Below
        # 6: Right Above - 7: Right - 8: Right Below

        self.probabilities_turn = [0.0, 0.0, 0.0, 1 - probability_turn_intended, probability_turn_intended, 0.0, 0.0, 0.0, 0.0]
        probability_displacement = 1 - probability_intended
        self.probabilities_0 =   [0,  0, 0, 1.0, 0, 0, 0, 0, 0]
        self.probabilities_90 =  [0, 0, 0, 0, 0, 0, probability_displacement, probability_intended, 0]
        self.probabilities_180 = [0, 0, 0, 0, probability_displacement, probability_intended, 0,  0, 0]
        self.probabilities_270 = [probability_displacement, probability_intended, 0, 0, 0, 0, 0, 0, 0]

    def get_probabilities(self, agent_dir):
        if agent_dir == self.direction:
            return self.probabilities_0
        elif agent_dir == 0: # Agent looks to east
            return self.probabilities_90
        elif agent_dir == 1: # Agent looks down
            return self.probabilities_180
        elif agent_dir == 2:
            return self.probabilities_270
        else:
            raise NotImplementedError("Agent directory not implemented")

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(6):
            ylo = 0.1  + 0.15 * i
            yhi = 0.20 + 0.15 * i
            fill_coords(img, rotate_fn(point_in_line(0.1, ylo, 0.3, yhi, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.3, yhi, 0.5, ylo, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.5, ylo, 0.7, yhi, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.7, yhi, 0.9, ylo, r=0.03), cx=0.5, cy=0.5, theta=math.pi), (0, 0, 0))

class SlipperyWest(WorldObj):
    def __init__(self, color: str = "blue", probability_intended=6/9, probability_turn_intended=6/9):
        super().__init__("slipperywest", color)
        self.offset = (-1,0)
        self.direction = 2

        # Field probabilties are stored in the order:
        # 0: Left Above - 1: Left - 2: Left Below
        # 3: Above - 4: Current - 5: Below
        # 6: Right Above - 7: Right - 8: Right Below

        self.probabilities_turn = [0.0, 1 - probability_turn_intended, 0.0, 0.0, probability_turn_intended, 0.0, 0.0, 0.0, 0.0]
        probability_displacement = 1 - probability_intended
        self.probabilities_0 =   [0, 1.0, 0, 0, 0, 0, 0, 0, 0]
        self.probabilities_90 =  [probability_displacement / 2, probability_displacement / 2, 0, probability_intended, 0 , 0, 0, 0, 0]
        self.probabilities_180 = [0, 0, 0, 0, probability_displacement, 0, 0, probability_intended, 0]
        self.probabilities_270 = [0, probability_displacement / 2 , probability_displacement / 2, 0, 0, probability_intended, 0, 0, 0]

    def can_overlap(self):
        return True

    def get_probabilities(self, agent_dir):
        if agent_dir == self.direction:
            return self.probabilities_0
        elif agent_dir == 3:
            return self.probabilities_90
        elif agent_dir == 0:
            return self.probabilities_180
        elif agent_dir == 1:
            return self.probabilities_270
        else:
            raise NotImplementedError("Agent directory not implemented")


    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(6):
            ylo = 0.1  + 0.15 * i
            yhi = 0.20 + 0.15 * i
            fill_coords(img, rotate_fn(point_in_line(0.1, ylo, 0.3, yhi, r=0.03), cx=0.5, cy=0.5, theta=0.5 * math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.3, yhi, 0.5, ylo, r=0.03), cx=0.5, cy=0.5, theta=0.5 * math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.5, ylo, 0.7, yhi, r=0.03), cx=0.5, cy=0.5, theta=0.5 * math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.7, yhi, 0.9, ylo, r=0.03), cx=0.5, cy=0.5, theta=0.5 * math.pi), (0, 0, 0))

class SlipperyEast(WorldObj):
    def __init__(self, color: str = "blue", probability_intended=6/9, probability_turn_intended=6/9):
        super().__init__("slipperyeast", color)
        self.offset = (1,0)
        self.direction = 0

        # Field probabilties are stored in the order:
        # 0: Left Above - 1: Left - 2: Left Below
        # 3: Above - 4: Current - 5: Below
        # 6: Right Above - 7: Right - 8: Right Below

        self.probabilities_turn = [0.0, 0.0, 0.0, 0.0, probability_turn_intended, 0.0, 0.0, 1 - probability_turn_intended, 0.0]
        probability_displacement = 1 - probability_intended
        self.probabilities_0 =   [0, 0, 0, 0, 0, 0, 0, 1.0, 0]
        self.probabilities_90 =  [0, 0, 0, 0, 0, probability_intended, 0,  probability_displacement / 2, probability_displacement / 2]
        self.probabilities_180 = [0, probability_intended, 0, 0, probability_displacement, 0, 0, 0, 0]
        self.probabilities_270 =   [0, 0, 0, probability_intended , 0, 0, probability_displacement / 2, probability_displacement / 2, 0]


    def get_probabilities(self, agent_dir):
        if agent_dir == self.direction:
            return self.probabilities_0
        elif agent_dir == 1:
            return self.probabilities_90
        elif agent_dir == 2:
            return self.probabilities_180
        elif agent_dir == 3:
            return self.probabilities_270
        else:
            raise NotImplementedError("Agent directory not implemented")


    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(6):
            ylo = 0.1  + 0.15 * i
            yhi = 0.20 + 0.15 * i
            fill_coords(img, rotate_fn(point_in_line(0.1, ylo, 0.3, yhi, r=0.03), cx=0.5, cy=0.5, theta=-0.5 * math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.3, yhi, 0.5, ylo, r=0.03), cx=0.5, cy=0.5, theta=-0.5 * math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.5, ylo, 0.7, yhi, r=0.03), cx=0.5, cy=0.5, theta=-0.5 * math.pi), (0, 0, 0))
            fill_coords(img, rotate_fn(point_in_line(0.7, yhi, 0.9, ylo, r=0.03), cx=0.5, cy=0.5, theta=-0.5 * math.pi), (0, 0, 0))

class Goal(WorldObj):
    def __init__(self):
        super().__init__("goal", "green")

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color: str = "blue"):
        super().__init__("floor", color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    def __init__(self):
        super().__init__("lava", "red")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(WorldObj):
    def __init__(self, color: str = "grey"):
        super().__init__("wall", color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Door(WorldObj):
    def __init__(self, color: str, is_open: bool = False, is_locked: bool = False):
        super().__init__("door", color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        # if door is closed and unlocked
        elif not self.is_open:
            state = 1
        else:
            raise ValueError(
                f"There is no possible state encoding for the state:\n -Door Open: {self.is_open}\n -Door Closed: {not self.is_open}\n -Door Locked: {self.is_locked}"
            )

        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)
    def to_state(self):
        return DoorState(self.color.capitalize(), not self.is_open)

class Key(WorldObj):
    def __init__(self, color: str = "blue"):
        super().__init__("key", color)

    def can_pickup(self):
        return True

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))

    def to_state(self):
        return KeyState(self.color.capitalize(), *self.cur_pos)

class Ball(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("ball", color)

    def can_pickup(self):
        return True

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

    def to_state(self):
        return BallState(self.color.capitalize(), *self.cur_pos)


class Box(WorldObj):
    def __init__(self, color, contains: WorldObj | None = None):
        super().__init__("box", color)
        self.contains = contains

    def can_pickup(self):
        return True

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(pos[0], pos[1], self.contains)
        return True

    def to_state():
        return BoxState(self.color.capitalize(), *self.cur_pos)
