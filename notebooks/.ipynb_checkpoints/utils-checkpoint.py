import stormpy
import stormpy.core
import stormpy.simulator

import stormpy.shields
import stormpy.logic

import stormpy.examples
import stormpy.examples.files

from enum import Enum
from abc import ABC

from PIL import Image, ImageDraw

import re
import sys
import tempfile, datetime, shutil
import numpy as np

import gymnasium as gym

from minigrid.core.actions import Actions
from minigrid.core.state import to_state, State

import os
import time

import argparse
from collections import deque
def tic():
    #Homemade version of matlab tic and toc functions: https://stackoverflow.com/a/18903019
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

class ShieldingConfig(Enum):
    Training = 'training'
    Evaluation = 'evaluation'
    Disabled = 'none'
    Full = 'full'

    def __str__(self) -> str:
        return self.value

def shield_needed(shielding):
    return shielding in [ShieldingConfig.Training, ShieldingConfig.Evaluation, ShieldingConfig.Full]

def shielded_evaluation(shielding):
    return shielding in [ShieldingConfig.Evaluation, ShieldingConfig.Full]

def shielded_training(shielding):
    return shielding in [ShieldingConfig.Training, ShieldingConfig.Full]

class ShieldHandler(ABC):
    def __init__(self) -> None:
        pass
    def create_shield(self, **kwargs) -> dict:
        pass


class MiniGridShieldHandler(ShieldHandler):
    def __init__(self, grid_to_prism_binary, grid_file, prism_path, formula, prism_config=None, shield_value=0.9, shield_comparison='absolute', nocleanup=False, prism_file=None) -> None:
        self.tmp_dir_name = f"shielding_files_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}_{next(tempfile._get_candidate_names())}"
        os.mkdir(self.tmp_dir_name)
        self.grid_file = self.tmp_dir_name + "/" + grid_file
        self.grid_to_prism_binary = grid_to_prism_binary
        self.prism_path = self.tmp_dir_name + "/" + prism_path
        self.prism_config = prism_config
        self.prism_file = prism_file
        self.action_dictionary = None

        self.formula = formula
        shield_comparison = stormpy.logic.ShieldComparison.ABSOLUTE if shield_comparison == "absolute" else stormpy.logic.ShieldComparison.RELATIVE
        self.shield_expression = stormpy.logic.ShieldExpression(stormpy.logic.ShieldingType.PRE_SAFETY, shield_comparison, shield_value)

        self.nocleanup = nocleanup
        
    def __del__(self):
        if not self.nocleanup:
            shutil.rmtree(self.tmp_dir_name)

    def __export_grid_to_text(self, env):
        with open(self.grid_file, "w") as f:
            f.write(env.printGrid(init=True))


    def __create_prism(self):
        if self.prism_file is not None:
            print(self.prism_file)
            print(self.prism_path)
            shutil.copyfile(self.prism_file, self.prism_path)
            return
        if self.prism_config is None:
            result = os.system(F"{self.grid_to_prism_binary} -i {self.grid_file} -o {self.prism_path}")
        else:
            result = os.system(F"{self.grid_to_prism_binary} -i {self.grid_file} -o {self.prism_path} -c {self.prism_config}")

        assert result == 0, "Prism file could not be generated"

    def __create_shield_dict(self):
        program = stormpy.parse_prism_program(self.prism_path)

        formulas = stormpy.parse_properties_for_prism_program(self.formula, program)
        options = stormpy.BuilderOptions([p.raw_formula for p in formulas])
        options.set_build_state_valuations(True)
        options.set_build_choice_labels(True)
        options.set_build_all_labels()
        print(f"LOG: Starting with explicit model creation...")
        tic()
        model = stormpy.build_sparse_model_with_options(program, options)
        toc()

        print(f"LOG: Starting with model checking...")
        tic()
        result = stormpy.model_checking(model, formulas[0], extract_scheduler=True, shield_expression=self.shield_expression)
        toc()

        assert result.has_shield
        shield = result.shield
        action_dictionary = dict()
        shield_scheduler = shield.construct()
        state_valuations = model.state_valuations
        choice_labeling = model.choice_labeling


        if self.nocleanup:
            stormpy.shields.export_shield(model, shield, self.tmp_dir_name + "/shield")

        print(f"LOG: Starting to translate shield...")
        tic()
        for stateID in model.states:
            choice = shield_scheduler.get_choice(stateID)
            choices = choice.choice_map

            state_valuation = state_valuations.get_string(stateID)

            ints = dict(re.findall(r'([a-zA-Z][_a-zA-Z0-9]+)=(-?[a-zA-Z0-9]+)', state_valuation))
            booleans = re.findall(r'(\!?)([a-zA-Z][_a-zA-Z0-9]+)[\s\t]+', state_valuation)
            booleans = {b[1]: False if b[0] == "!" else True for b in booleans}

            if int(ints.get("previousActionAgent", 7)) != 7:
                continue
            if int(ints.get("clock", 0)) != 0:
                continue
            state = to_state(ints, booleans)
            #print(f"{state} got added with actions:")
            # print(get_allowed_actions_mask([choice_labeling.get_labels_of_choice(model.get_choice_index(stateID, choice[1])) for choice in choices]))
            action_dictionary[state] = get_allowed_actions_mask([choice_labeling.get_labels_of_choice(model.get_choice_index(stateID, choice[1])) for choice in choices])

        toc()
        #print(f"{len(action_dictionary)} states in the shield")
        self.action_dictionary = action_dictionary

        # Remove shielding_files_* immediatelly, only to remove clutter for the demo
        if not self.nocleanup:
            shutil.rmtree(self.tmp_dir_name)
        return action_dictionary


    def create_shield(self, **kwargs):
        if self.action_dictionary is not None:
            #print("Returning already calculated shield")
            return self.action_dictionary
            
        env = kwargs["env"]
        self.__export_grid_to_text(env)
        self.__create_prism()
        print("Computing new shield")
        return self.__create_shield_dict()

#################################
# class PerceptionLossShieldHandler(MiniGridShieldHandler):
#     def __init__(self, k: int, n: int, *args, **kwargs):
#         """
#         Extends MiniGridShieldHandler to handle perception loss scenarios.
        
#         Args:
#             k (int): Maximum number of oblivious steps allowed
#             n (int): Total window size to consider
#             *args, **kwargs: Arguments passed to MiniGridShieldHandler
#         """
#         super().__init__(*args, **kwargs)
#         self.perception_simulator = PerceptionLossSimulator(k, n)
#         self._last_known_state = None
#         self._shield_cache =             #           # {}
        
    # def create_shield(self, **kwargs) -> dict:
    #     """Creates the base shield and initializes perception tracking."""
    #     self._base_shield = super().create_shield(**kwargs)
    #     self.perception_simulator.reset()
    #     return self._base_shield
    
    # def get_shield_actions(self, state: State) -> list:
    #     """
    #     Get allowed actions considering perception loss.
        
    #     Args:
    #         state (State): Current state observation
            
    #     Returns:
    #         list: Allowed actions mask considering perception loss
    #     """
    #     is_blanked = self.perception_simulator.step()
        
    #     if is_blanked:
    #         if self._last_known_state is None:
    #             # If we have no previous state, take most conservative action mask
    #             return self._get_conservative_actions()
    #         state_to_use = self._last_known_state
    #     else:
    #         self._last_known_state = state
    #         state_to_use = state
            
    #     try:
    #         return self._base_shield[state_to_use]
    #     except KeyError:
    #         return self._get_conservative_actions()
    
    # def _get_conservative_actions(self) -> list:
    #     """
    #     Get the most conservative set of actions that are safe across all possible states.
        
    #     Returns:
    #         list: Conservative action mask
    #     """
    #     # Initialize with all actions allowed
    #     conservative_mask = [1.0] * 7  
        
    #     # If we have a shield, find intersection of allowed actions across all states
    #     if hasattr(self, '_base_shield'):
    #         for state_actions in self._base_shield.values():
    #             for i in range(len(conservative_mask)):
    #                 conservative_mask[i] &= state_actions[i]
        
    #     return conservative_mask
    
#     def reset(self):
#         """Reset perception loss simulation and state tracking."""
#         self.perception_simulator.reset()
#         self._last_known_state = None


class PerceptionLossSimulator:
    def __init__(self, k: int, n: int):
        """
        Initialize the perception loss simulator.
        
        Args:
            k (int): Maximum number of blanked (oblivious) steps allowed
            n (int): Total window size to consider
        """
        if k > n:
            raise ValueError("k cannot be larger than n")
        if k < 0 or n < 0:
            raise ValueError("k and n must be non-negative")
            
        self.k = k  # max allowed blanked steps
        self.n = n  # window size
        self.current_window = deque(maxlen=n)  # False = normal, True = blanked
        self.blanked_count = 0  # current count of blanked steps in window
        self.total_steps=0
    def can_blank_next(self) -> bool:
        """Check if next step can be blanked while respecting k,n constraints."""
        if len(self.current_window) == self.n:
            # If window is full, we need to account for the oldest step that will be removed
            if self.current_window[0]:  # if oldest step was blanked
                effective_count = self.blanked_count - 1
            else:
                effective_count = self.blanked_count
        else:
            effective_count = self.blanked_count
            
        return effective_count < self.k
    
    def step(self) -> bool:
        """
        Simulate one step and determine if perception is lost (blanked).
        
        Returns:
            bool: True if step is blanked, False if normal
        """
        print(self.get_stats())
        self.total_steps+=1
        # If window is full, remove oldest and update count if needed
        if len(self.current_window) == self.n:
            if self.current_window.popleft():  # if removing a blanked step
                self.blanked_count -= 1
        
        # Check if we can blank and randomly decide
        can_blank = self.can_blank_next()
        is_blanked = can_blank and np.random.random() < self.k/self.n  # 30% chance if possible
        # Update window and count
        self.current_window.append(is_blanked)
        if is_blanked:
            self.blanked_count += 1
            print(self.total_steps)
        return is_blanked
    
    def get_window(self) -> list:
        """Return current window state."""
        return list(self.current_window)
    
    def get_stats(self) -> dict:
        """Return current statistics."""
        return {
            "window_size": len(self.current_window),
            "blanked_count": self.blanked_count,
            "window_state": list(self.current_window),
            "total_steps": self.total_steps,
        }
    
    def reset(self):
        """Reset simulator to initial state."""
        self.current_window.clear()
        self.blanked_count = 0
        self.total_steps=0
#################################
        

def rectangle_for_overlay(x, y, dir, tile_size, width=2, offset=0, thickness=0):
    if dir == 0: return (((x+1)*tile_size-width-thickness,y*tile_size+offset), ((x+1)*tile_size,(y+1)*tile_size-offset))
    if dir == 1: return ((x*tile_size+offset,(y+1)*tile_size-width-thickness), ((x+1)*tile_size-offset,(y+1)*tile_size))
    if dir == 2: return ((x*tile_size,y*tile_size+offset), (x*tile_size+width+thickness,(y+1)*tile_size-offset))
    if dir == 3: return ((x*tile_size+offset,y*tile_size), ((x+1)*tile_size-offset,y*tile_size+width+thickness))

def triangle_for_overlay(x,y, dir, tile_size):
    offset = tile_size/2
    if dir == 0: return [((x+1)*tile_size,y*tile_size), ((x+1)*tile_size,(y+1)*tile_size), ((x+1)*tile_size-offset, y*tile_size+tile_size/2)]
    if dir == 1: return [(x*tile_size,(y+1)*tile_size), ((x+1)*tile_size,(y+1)*tile_size), (x*tile_size+tile_size/2, (y+1)*tile_size-offset)]
    if dir == 2: return [(x*tile_size,y*tile_size), (x*tile_size,(y+1)*tile_size), (x*tile_size+offset, y*tile_size+tile_size/2)]
    if dir == 3: return [(x*tile_size,y*tile_size), ((x+1)*tile_size,y*tile_size), (x*tile_size+tile_size/2, y*tile_size+offset)]

def create_shield_overlay_image(env, shield):
    env.reset()
    img = Image.fromarray(env.render()).convert("RGBA")
    ts = env.tile_size
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    red = (255,0,0,200)
    for x in range(0, env.width):
        for y in range(0, env.height):
            for dir in range(0,4):
                try:
                    if shield[State(x, y, dir, "")][2] <= 0.0:
                        draw.polygon(triangle_for_overlay(x,y,dir,ts), fill=red)
                    #else:
                    #    draw.polygon(triangle_for_overlay(x,y,dir,ts), fill=(0, 200, 0, 96))

                except KeyError: pass
    img = Image.alpha_composite(img, overlay)
    img.show()
def expname(args):
    return f"{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}_{args.env}_{args.shielding}_{args.shield_comparison}_{args.shield_value}_{args.expname_suffix}"

def create_log_dir(args):
    log_dir = f"{args.log_dir}/{expname(args)}"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def get_allowed_actions_mask(actions):
    action_mask = [0.0] * 7
    actions_labels = [label for labels in actions for label in list(labels)]
    for action_label in actions_labels:
        if "move" in action_label:
            action_mask[2] = 1.0
        elif "left" in action_label:
            action_mask[0] = 1.0
        elif "right" in action_label:
            action_mask[1] = 1.0
        elif "pickup" in action_label:
            action_mask[3] = 1.0
        elif "drop" in action_label:
            action_mask[4] = 1.0
        elif "toggle" in action_label:
            action_mask[5] = 1.0
        elif "done" in action_label:
            action_mask[6] = 1.0
    return action_mask

def common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        help="gym environment to load",
                        choices=gym.envs.registry.keys(),
                        default="MiniGrid-LavaSlipperyCliff-16x13-v0")

    parser.add_argument("--grid_file", default="grid.txt")
    parser.add_argument("--prism_file", default=None)
    parser.add_argument("--prism_output_file", default="grid.prism")
    parser.add_argument("--log_dir", default="../log_results/")
    parser.add_argument("--formula", default="Pmax=? [G !AgentIsOnLava]")
    parser.add_argument("--shielding", type=ShieldingConfig, choices=list(ShieldingConfig), default=ShieldingConfig.Full)
    parser.add_argument("--steps", default=20_000, type=int)
    parser.add_argument("--shield_creation_at_reset", action=argparse.BooleanOptionalAction)
    parser.add_argument("--prism_config",  default=None)
    parser.add_argument("--shield_value", default=0.9, type=float)
    parser.add_argument("--shield_comparison", default='absolute', choices=['relative', 'absolute'])
    parser.add_argument("--nocleanup", action=argparse.BooleanOptionalAction)
    parser.add_argument("--expname_suffix", default="")
    return parser

class MiniWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.transpose(1,0,2), info

    def observations(self, obs):
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.transpose(1,0,2), reward, terminated, truncated, info
