from minigrid.core.world_object import WorldObj
from minigrid.core.tasks import DoRandom, TaskManager
import numpy as np
import math

from minigrid.core.state import AdversaryState


from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    DIR_TO_VEC
)
from minigrid.utils.rendering import (
    fill_coords,
    point_in_triangle,
    rotate_fn,
)

VIEW_TO_STATE_IDX = {
   0: 3,
   1: 4,
   2: 5,
   3: 6
}


class Adversary(WorldObj):
    def __init__(self, adversary_pos, adversary_dir=1, color="blue", tasks=[DoRandom()], repeating=False):
        super().__init__("adversary", color)
        self.adversary_pos = adversary_pos # TODO
        self.adversary_dir = adversary_dir # TODO
        self.color = color
        self.rgb = COLORS[self.color]
        self.task_manager = TaskManager(tasks, repeating=repeating)
        self.carrying = None
        self.name = color.capitalize()

    def render(self, img):
        tri_fn = point_in_triangle(
           (0.12, 0.19),
           (0.87, 0.50),
           (0.12, 0.81),
       )

       # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.adversary_dir)
        fill_coords(img, tri_fn, COLORS[self.color])

    def dir_vec(self):
        assert self.adversary_dir >= 0 and self.adversary_dir < 4
        return DIR_TO_VEC[self.adversary_dir]

    def can_overlap(self):
        return True

    def encode(self):
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], VIEW_TO_STATE_IDX[self.adversary_dir])

    def append_task(self, task):
        self.task_manager.tasks.append(task)

    def insert_task(self, task, position):
        self.task_manager.tasks.insert(position, task)

    def to_state(self):
        color = self.color.capitalize()
        if self.carrying:
            carrying = f"{self.carrying.color.capitalize()}{self.carrying.type.capitalize()}"
        else:
            carrying = ""
        return AdversaryState(color=color, col=self.adversary_pos[0], row=self.adversary_pos[1], view=self.adversary_dir, carrying=carrying)

    def get_action(self, env):
        return self.task_manager.get_best_action(self.adversary_pos, self.adversary_dir, self.carrying, env)

    def __str__(self):
        return f"({self.adversary_pos}, {self.adversary_dir})"
