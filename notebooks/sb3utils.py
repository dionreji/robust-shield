import gymnasium as gym
import numpy as np
import random
from moviepy.editor import ImageSequenceClip
import sys

from utils import MiniGridShieldHandler, common_parser
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import Image
from utils import PerceptionLossSimulator
from minigrid.core.actions import Actions
from minigrid.core.state import to_state, State

class MiniGridSbShieldingWrapper(gym.core.Wrapper):
    def __init__(
        self,
        env,
        shield_handler: MiniGridShieldHandler,
        create_shield_at_reset=False,
        k: int = 0,
        n: int = 10,
    ):
        """
        Initialize the MiniGrid shielding wrapper.
        
        Args:
            env: The environment to wrap
            shield_handler: Handler for shield creation and management
            create_shield_at_reset: Whether to create new shield on reset
            k: Maximum number of blanked steps allowed
            n: Window size for perception tracking
        """
        print("Entered into Modified Wrapper")
        super().__init__(env)
        self.k = k
        self.n = n
        self.shield_handler = shield_handler
        self.create_shield_at_reset = create_shield_at_reset
        self.simulator = PerceptionLossSimulator(self.k, self.n)  # Changed Simulator to simulator
        self.shield = self.shield_handler.create_shield(env=self.env)
        self._last_known_state=None
        self._last_known_step=0
        self._total_steps=0
        
    def _get_conservative_actions(self) -> list:
        """
        Get the most conservative set of actions that are safe across all possible states
        by finding the intersection of allowed actions across all states in the shield.
        
        Returns:
            list: Action mask where 1.0 indicates an action is safe in all states,
                and 0.0 indicates it's unsafe in at least one state
        """
        print("No last state known. Giving out the conservative actions")
        # If we don't have a base shield, return all actions as unsafe
        if not hasattr(self, '_base_shield') or not self.shield:
            return [0.0] * 7
        
        # Get the first state's action mask as initial conservative mask
        first_state = next(iter(self.shield))
        conservative_mask = list(self.shield[first_state])
        
        # For each remaining state in the shield
        for state in self.shield:
            current_actions = self._base_shield[state]
            # Update conservative mask to only include actions that are safe in both
            # the current conservative mask AND the current state's actions
            for action_idx in range(len(conservative_mask)):
                conservative_mask[action_idx] = (
                    1.0 if conservative_mask[action_idx] and current_actions[action_idx]
                    else 0.0
                )
            
            # Early termination if no actions are safe anymore
            if sum(conservative_mask) == 0:
                break
                
        return conservative_mask
############################
    def get_possible_states(self, start_state, depth):
        """
        Get all possible states reachable from start_state within given depth
        using safe actions at each step.
        
        Args:
            start_state: Initial state to start exploration from
            depth: Number of steps to explore
            
        Returns:
            set: Set of possible states reachable within depth steps
        """
        print("Finding possible states")
        if depth == 0:
            return {start_state}
            
        possible_states = {start_state}
        current_states = {start_state}
        
        for _ in range(depth):
            next_states = set()
            for state in current_states:
                # Get safe actions for current state
                try:
                    safe_actions = self.shield[state]
                except KeyError:
                    continue
                    
                # For each safe action, get next possible states
                for action_idx, is_safe in enumerate(safe_actions):
                    if is_safe:
                        new_env=self.env
                        next_state,x,y,z,w = new_env.step(action_idx)
                        print("symbolically simulated new state",new_env.get_symbolic_state())
                        next_states.add(new_env.get_symbolic_state())
            
            current_states = next_states
            possible_states.update(next_states)
            
        return possible_states
#############################
    def get_shield_action(self, state) -> list:
        """
        Get allowed actions considering perception loss.
        Args:
            state: Current state observation
        Returns:
            list: Allowed actions mask considering perception loss
        """
        print("Inside get_sheild_actions() calculating safe actions")
        self._total_steps += 1
        is_blanked = self.simulator.step()
        if is_blanked:
            if self._last_known_state is None:
                return self._get_conservative_actions()
                
            state_to_use = self._last_known_state
            continued_blank_steps = self._total_steps - self._last_known_step - 1
            window_state = self.simulator.get_stats()["window_state"]
            
            # Calculate consecutive blank states
            blank_states_till = []
            for i in range(len(window_state)):
                if i == 0:
                    blank_states_till.append(1 if window_state[i] else 0)
                else:
                    if window_state[i]:
                        blank_states_till.append(blank_states_till[i-1] + 1)
                    else:
                        blank_states_till.append(blank_states_till[i-1])
            
            total_blank_states = blank_states_till[-1]
            
            # Count possible future blank states within k limit
            possible_blank_states = 0
            for i in range(min(self.k,len(blank_states_till))):
                if total_blank_states - blank_states_till[i] < self.k:
                    possible_blank_states += 1
                else:
                    break
                    
            # Get all possible states we could be in after taking safe actions
            # for (continued_blank_steps + possible_blank_states) steps
            total_depth = continued_blank_steps + possible_blank_states
            possible_current_states = self.get_possible_states(state_to_use, total_depth)
            
            # Initialize action mask with all actions allowed
            safe_actions = [1.0] * len(self.shield[state_to_use])
            
            # An action is safe only if it's safe for ALL possible current states
            for possible_state in possible_current_states:
                try:
                    state_actions = self.shield[possible_state]
                    for i in range(len(safe_actions)):
                        # print(safe_actions[i])
                        # print("sfae ka type", type(safe_actions[i]))
                        safe_actions[i] = bool(safe_actions[i]) and bool(state_actions[i])

                except KeyError:
                    # If any state is not in shield, be conservative
                    return self._get_conservative_actions()
            
            return safe_actions
            
        else:
            self._last_known_state = state
            self._last_known_step = self._total_steps
            state_to_use = state
            
        try:
            return self.shield[state_to_use]
        except KeyError:
            print("KeyError")
            return self._get_conservative_actions()
    
    def create_action_mask(self):
        # try:
            # return self.shield[self.env.get_symbolic_state()]
            print("create_action_mask() --- 1")
            answer_list= self.get_shield_action(self.env.get_symbolic_state())
            print("Old Shield's Action-------->")
            print(self.shield[self.env.get_symbolic_state()])
            print("New Shield's Safe actions -------->")
            print(answer_list)
            if answer_list == [0.0] * 7:
                sys.exit("No safe actions possible, shield stopped.")
            return answer_list
        # except:
        #     if answer_list == [0.0] * 7:
        #         sys.exit("Couldn't create shield mask, shield stopped forcefully.")
            # print("create_action_mask() --- 2")
            # return [0.0] * 3 + [0.0] * 4

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)

        if self.create_shield_at_reset:
            shield = self.shield_handler.create_shield(env=self.env)
            self.shield = shield
        return obs, infos

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        info["no_shield_action"] = not self.shield.__contains__(self.env.get_symbolic_state())
        return obs, rew, done, truncated, info

def parse_sb3_arguments():
    parser = common_parser()
    args = parser.parse_args()

    return args

class ImageRecorderCallback(BaseCallback):
    def __init__(self, eval_env, render_freq, n_eval_episodes, evaluation_method, log_dir, deterministic=True, verbose=0):
        super().__init__(verbose)

        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self._evaluation_method = evaluation_method
        self._log_dir = log_dir

    def _on_training_start(self):
        image = self.training_env.render(mode="rgb_array")
        self.logger.record("trajectory/image", Image(image, "HWC"), exclude=("stdout", "log", "json", "csv"))

    def _on_step(self) -> bool:
        #if self.n_calls % self._render_freq == 0:
        #    self.record_video()
        return True

    def _on_training_end(self) -> None:
        self.record_video()

    def record_video(self) -> bool:
        screens = []
        def grab_screens(_locals, _globals) -> None:
            """
            Renders the environment in its current state, recording the screen in the captured `screens` list

            :param _locals: A dictionary containing all local variables of the callback's scope
            :param _globals: A dictionary containing all global variables of the callback's scope
            """
            screen = self._eval_env.render()
            screens.append(screen)
        self._evaluation_method(
            self.model,
            self._eval_env,
            callback=grab_screens,
            n_eval_episodes=self._n_eval_episodes,
            deterministic=self._deterministic,
        )

        clip = ImageSequenceClip(list(screens), fps=3)
        clip.write_gif(f"{self._log_dir}/{self.n_calls}.gif", fps=3)
        return True


class InfoCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.sum_goal = 0
        self.sum_lava = 0
        self.sum_collisions = 0
        self.sum_opened_door = 0
        self.sum_picked_up = 0
        self.no_shield_action = 0

    def _on_step(self) -> bool:
        infos = self.locals["infos"][0]
        if infos["reached_goal"]:
            self.sum_goal += 1
        if infos["ran_into_lava"]:
            self.sum_lava += 1
        self.logger.record("info/sum_reached_goal", self.sum_goal)
        self.logger.record("info/sum_ran_into_lava", self.sum_lava)
        if "collision" in infos:
            if infos["collision"]:
                self.sum_collisions += 1
            self.logger.record("info/sum_collision", self.sum_collisions)
        if "opened_door" in infos:
            if infos["opened_door"]:
                self.sum_opened_door += 1
            self.logger.record("info/sum_opened_door", self.sum_opened_door)
        if "picked_up" in infos:
            if infos["picked_up"]:
                self.sum_picked_up += 1
            self.logger.record("info/sum_picked_up", self.sum_picked_up)
        if "no_shield_action" in infos:
            if infos["no_shield_action"]:
                self.no_shield_action += 1
            self.logger.record("info/no_shield_action", self.no_shield_action)
        return True
