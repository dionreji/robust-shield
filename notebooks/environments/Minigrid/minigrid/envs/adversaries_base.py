from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, SlipperyNorth, SlipperyEast, SlipperySouth, SlipperyWest, Ball
from minigrid.minigrid_env import MiniGridEnv, is_slippery
from minigrid.core.tasks import GoTo, DoNothing, PickUpObject, PlaceObject

import numpy as np

class AdversaryEnv(MiniGridEnv):

    """
    ## Description

    """

    def __init__(self, width=7, height=6, generate_wall=True, generate_lava=False, generate_slippery=False ,max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 10 * (width * height)**2
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.collision_penalty = -1
        super().__init__(
            mission_space=mission_space, width=width, height=height, max_steps=max_steps, **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "Finish your task while avoiding the adversaries"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)


    def step(self, action):
        delete_list = list()
        for position, box in self.background_tiles.items():
            if self.grid.get(*position) is None:
                self.grid.set(*position, box)
                self.grid.set_background(*position, None)
                delete_list.append(tuple(position))
        for position in delete_list:
            del self.background_tiles[position]

        obs, reward, terminated, truncated, info = super().step(action)

        agent_pos = self.agent_pos
        adv_penalty = 0

        if not terminated:
            for adversary in self.adversaries.values():
                collided = self.move_adversary(adversary, agent_pos)
                self.trajectory.append((adversary.color, adversary.adversary_pos, adversary.adversary_dir))
                if collided:
                    terminated = True
                    info["collision"] = True
                    try:
                        reward = self.collision_penalty
                    except e:
                        reward = -1


        return obs, reward, terminated, truncated, info

    def move_adversary(self, adversary, agent_pos):
        # fetch current location and forward location
        cur_pos = adversary.adversary_pos
        current_cell = self.grid.get(*adversary.adversary_pos)
        fwd_pos = cur_pos + adversary.dir_vec()
        fwd_cell = self.grid.get(*fwd_pos)
        collision = False
        need_position_update = False

        action = adversary.get_action(self)
        if action == self.actions.forward and is_slippery(current_cell):
            probabilities = current_cell.get_probabilities(adversary.adversary_dir)
            possible_fwd_pos, prob = self.get_neighbours_prob(adversary.adversary_pos, probabilities)
            fwd_pos_index = np.random.choice(len(possible_fwd_pos), 1, p=prob)
            fwd_pos = possible_fwd_pos[fwd_pos_index[0]]
            fwd_cell = self.grid.get(*fwd_pos)
            need_position_update = True

        if action == self.actions.left:
            adversary.adversary_dir -= 1
            if adversary.adversary_dir < 0:
                adversary.adversary_dir += 4

        # Rotate right
        elif action == self.actions.right:
            adversary.adversary_dir = (adversary.adversary_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_pos[0] == agent_pos[0] and fwd_pos[1] == agent_pos[1]:
                collision = True
            if fwd_cell is None or fwd_cell.can_overlap():
                adversary.adversary_pos = tuple(fwd_pos)

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if adversary.carrying is None:
                    adversary.carrying = fwd_cell
                    adversary.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and adversary.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], adversary.carrying)
                adversary.carrying.cur_pos = fwd_pos
                adversary.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if need_position_update and (fwd_cell is None or fwd_cell.can_overlap()):
            adversary.adversary_pos = tuple(fwd_pos)

        return collision
