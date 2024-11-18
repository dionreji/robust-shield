from abc import ABC
from typing import Iterable, List
from collections import deque

import numpy as np


from minigrid.core.world_object import Lava
from minigrid.core.actions import Actions
from minigrid.core.constants import DIR_TO_VEC


try:
    from astar import find_path
except:
    print("Install with:")
    print("pip install git+https://github.com/jrialland/python-astar.git")
    raise Exception("Need to install astar")
import numpy.random

class Task(ABC):
    # returns a bool, true if task is completed, false otherwise
    def completed(self, pos, dir, carrying, env):
        pass
    # Returns the best action to solve this task
    def get_best_action(self, pos, dir, carrying, env):
        pass
    # returns a string representing the task
    def __repr__(self):
        pass


def get_plan(pos, dir, carrying, env, goal_pos, avoid_agent=True):
    def neighbors_fnct(node):
        left = (node[0], node[1], node[3], -node[2])
        right = (node[0], node[1], -node[3], node[2])
        fwd_pos = node[0] + node[2], node[1] + node[3]
        forward_cell = env.grid.get(*fwd_pos)

        forward_pos_open = forward_cell is None or forward_cell.can_overlap()
        forward_pos_open = forward_pos_open and not isinstance(forward_cell, Lava)

        if forward_pos_open:
            forward = (node[0] + node[2], node[1] + node[3], node[2], node[3])
            return forward, left, right
        else:
            return left, right

    def avoid_agent_neighbors_fcnt(node):
        left = (node[0], node[1], node[3], -node[2])
        right = (node[0], node[1], -node[3], node[2])
        fwd_pos = node[0] + node[2], node[1] + node[3]
        forward_cell = env.grid.get(*fwd_pos)

        forward_pos_open = forward_cell is None or forward_cell.can_overlap()
        forward_pos_open = forward_pos_open and not isinstance(forward_cell, Lava)
        forward_pos_not_agent = fwd_pos[0] != env.agent_pos[0] or fwd_pos[1] != env.agent_pos[1]

        if forward_pos_open and forward_pos_not_agent:
            forward = (node[0] + node[2], node[1] + node[3], node[2], node[3])
            return forward, left, right
        else:
            return left, right

    dir_vec = DIR_TO_VEC[dir]
    start = (pos[0], pos[1], dir_vec[0], dir_vec[1])
    goal = (goal_pos[0], goal_pos[1], dir_vec[0], dir_vec[1])

    plan = find_path(
        start=start,
        goal=goal,
        neighbors_fnct=avoid_agent_neighbors_fcnt if avoid_agent else neighbors_fnct,
        reversePath=False,
        heuristic_cost_estimate_fnct=lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]),
        distance_between_fnct=lambda a, b: 1.0,
        is_goal_reached_fnct=lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 0
    )

    return list(plan) if plan is not None else None


class GoTo(Task):
    def __init__(self, goal_position):
        self.goal_position = goal_position
        self.plan = None

    def completed(self, pos, dir, carrying, env):
        return pos == self.goal_position

    def get_best_action(self, pos, dir, carrying, env):
        if pos == self.goal_position:
            return Actions.done
        # if farther than 1 unit away, Run A*
        if self.plan is None or len(self.plan) == 0:
            self.plan = get_plan(pos, dir, carrying, env, self.goal_position)

        # if we have a plan, but we are not in the state we should be, create new plan

        dir_vec = DIR_TO_VEC[dir]
        if self.plan is not None:
            current_state_maybe = self.plan.pop(0)
            if current_state_maybe[0] != pos[0] or \
                    current_state_maybe[1] != pos[1] or \
                    current_state_maybe[2] != dir_vec[0] or \
                    current_state_maybe[3] != dir_vec[1]:
                self.plan = None
                return self.get_best_action(pos, dir, carrying, env)

        if self.plan is None or len(self.plan) < 1:
            # yield because there might be another actor blocking the way to our goal
            self.plan = None
            return Actions.done

        next_state = self.plan[0]

        # decide how to achieve next state
        if abs(next_state[0] - pos[0]) == 1 or abs(next_state[1] - pos[1]) == 1:
            return Actions.forward
        elif next_state[2] == dir_vec[1] and next_state[3] == -dir_vec[0]:
            return Actions.left
        elif next_state[2] == -dir_vec[1] and next_state[3] == dir_vec[0]:
            return Actions.right
        else:  # something went wrong such as bumping into other agent, replan
            self.plan = None
            return self.get_best_action(pos, dir, carrying, env)

    def __repr__(self):
        return "Task: Go to position {}".format(self.goal_position)


class PickUpObject(Task):
    def __init__(self, obj_position, obj):
        self.obj_position = obj_position
        self.obj = obj
        self.plan = None
    def completed(self, pos, dir, carrying, env):
        return carrying == self.obj
    def get_best_action(self, pos, dir, carrying, env):
        assert abs(pos[0] - self.obj_position[0] + pos[1] - self.obj_position[1]) == 1, "Distance to the object needs to be exactly 1, please move the adversary (GoTo) first."
        delta_row = self.obj_position[0] - pos[0]
        delta_col = self.obj_position[1] - pos[1]
        dir_vec = DIR_TO_VEC[dir]
        if delta_row == dir_vec[0] and delta_col == dir_vec[1]:
            return Actions.pickup
        else:
            return Actions.left
    def __repr__(self):
        return "Task: Pick up object at position {}".format(self.obj_position)


class PlaceObject(Task):
    def __init__(self, obj_position, obj):
        self.obj_position = obj_position
        self.obj = obj

    def completed(self, pos, dir, carrying, env):
        return env.grid.get(*self.obj_position) == self.obj and carrying is None
    def get_best_action(self, pos, dir, carrying, env):
        assert abs(pos[0] - self.obj_position[0] + pos[1] - self.obj_position[1]) == 1, "Distance to the object needs to be exactly 1, please move the adversary (GoTo) first."
        delta_row = self.obj_position[0] - pos[0]
        delta_col = self.obj_position[1] - pos[1]
        dir_vec = DIR_TO_VEC[dir]
        if delta_row == dir_vec[0] and delta_col == dir_vec[1]:
            return Actions.drop
        else:
            return Actions.left
    def __repr__(self):
        return "Task: Place object at position {}".format(self.obj_position)

class DoNothing(Task):
    def __init__(self, duration=0):
        self.duration = duration
        self.steps = 0
    def completed(self, pos, dir, carrying, env):
        if self.duration == 0:
            return False
        elif self.duration == self.steps:
            return True
        return False
    def reset_steps(self):
        self.steps = 0
    def get_best_action(self, pos, dir, carrying, env):
        if self.duration != 0:
            self.steps += 1
        return Actions.done
    def __repr__(self):
        return "Task: Do nothing"

class DoRandom(Task):
    def __init__(self, duration=0):
        self.duration = duration
        self.steps = 0
    def completed(self, pos, dir, carrying, env):
        if self.duration == 0:
            return False
        elif self.duration == self.steps:
            return True
        return False
    def reset_steps(self):
        self.steps = 0
    def get_best_action(self, pos, dir, carrying, env):
        if self.duration != 0: self.steps += 1
        return numpy.random.random_integers(0, 2, 1)
    def __repr__(self):
        return "Task: Act randomly"

class FollowAgent(Task):
    def __init__(self, agent, duration=0):
        self.agent = agent
        self.duration = duration
        self.steps = 0
        self.follow_agent = False
        if self.agent == "red":
            self.follow_agent = True
        self.plan = None
    def completed(self, pos, dir, carrying, env):
        if self.duration == 0:
            return False
        elif self.duration == self.steps:
            return True
        return False
    def reset_steps(self):
        self.steps = 0
    def get_best_action(self, pos, dir, carrying, env):
        if self.duration != 0: self.steps += 1
        targets_position = np.array(self._get_targets_position(env))
        optimal_directions=self._get_optimal_directions(pos, targets_position)

        if dir in optimal_directions:
            self.plan = get_plan(pos, dir, carrying, env, targets_position, avoid_agent=False)
            if self.plan is not None:
                next_state = self.plan[1]
                dir_vec = DIR_TO_VEC[dir]
                if abs(next_state[0] - pos[0]) == 1 or abs(next_state[1] - pos[1]) == 1:
                    return Actions.forward
                elif next_state[2] == dir_vec[1] and next_state[3] == -dir_vec[0]:
                    return Actions.left
                elif next_state[2] == -dir_vec[1] and next_state[3] == dir_vec[0]:
                    return Actions.right
                else:
                    self.plan = None
                    return Actions.done
            else:
                return Actions.done

        return self._get_turn_action(dir, optimal_directions)

    def __repr__(self):
        return f"Task: Follow agent {self.agent}"

    def _get_targets_position(self, env):
        if self.follow_agent:
            return env.agent_pos
        return env.adversaries[self.agent].adversary_pos

    def _get_optimal_directions(self, pos, targets_position):
        offset = targets_position - np.array(pos)
        optimal_directions = list()
        if offset[0] < 0: optimal_directions.append(2)
        if offset[0] > 0: optimal_directions.append(0)
        if offset[1] < 0: optimal_directions.append(3)
        if offset[1] > 0: optimal_directions.append(1)
        return optimal_directions

    def _get_turn_action(self, dir, optimal_directions):
        diffs = [d - dir for d in optimal_directions]
        if -1 in diffs or 3 in diffs: return Actions.left
        if 1 in diffs or -3 in diffs: return Actions.right
        else:                         return numpy.random.random_integers(0, 1, 1) # target is behind me


class TaskManager:
    def __init__(self, tasks:List[Task], repeating=False):
        self.repeating = repeating
        if repeating:
            self.tasks = deque(tasks)
        else:
            self.tasks = tasks

    def get_best_action(self, pos, dir, carrying, env):
        if len(self.tasks) == 0:
            raise Exception("List of tasks empty")
        if self.tasks[0].completed(pos, dir, carrying, env) and not self.repeating:
            self.tasks.pop(0)
        elif self.tasks[0].completed(pos, dir, carrying, env) and self.repeating:
            done_task = self.tasks.popleft()
            if isinstance(done_task, (DoRandom, DoNothing, FollowAgent)): done_task.reset_steps()
            self.tasks.append(done_task)
        try:
            best_action = self.tasks[0].get_best_action(pos, dir, carrying, env)
        except IndexError as e:
            # The adversary has finished all its tasks and will yield
            self.tasks = [DoNothing()]
            best_action = self.tasks[0].get_best_action(pos, dir, carrying, env)
        return best_action
