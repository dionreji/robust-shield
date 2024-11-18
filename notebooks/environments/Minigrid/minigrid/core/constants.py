from __future__ import annotations

import numpy as np

TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "white": np.array([255, 255, 255]),
    "black": np.array([0, 0, 0]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5, "white": 6, "black": 7}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
    "adversary": 11,
    "slipperyeast": 12,
    "slipperysouth": 13,
    "slipperywest": 14,
    "slipperynorth": 15,
    "slipperynortheast": 16,
    "slipperynorthwest": 17,
    "slipperysoutheast": 18,
    "slipperysouthwest": 19,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
    "adv_view_east": 3,
    "adv_view_south": 4,
    "adv_view_west": 5,
    "adv_view_north":6
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

OBJECT_TO_STR = {
    "wall": "W",
    "floor": "F",
    "door": "D",
    "key": "K",
    "ball": "A",
    "box": "B",
    "goal": "G",
    "lava": "V",
    "adversary": "Z",
    "slippery": "S",
    "slipperyeast": "e",
    "slipperysouth": "s",
    "slipperywest": "w",
    "slipperynorth": "n" ,
    "slipperynortheast": "b",
    "slipperynorthwest": "a",
    "slipperysoutheast": "d",
    "slipperysouthwest": "c"
}
