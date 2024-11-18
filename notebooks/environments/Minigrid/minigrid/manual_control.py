#!/usr/bin/env python3

from __future__ import annotations
import re
from tqdm import tqdm

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import Env

from minigrid.core.actions import Actions
from minigrid.core.state import to_state
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

def actionsToMiniGrid(actions):
    mask = [0] * 7
    for action in actions:
        if "turn_left" in action:
            mask[0] = 1
        elif "turn_right" in action:
            mask[1] = 1
        elif "move" in action:
            mask[2] = 1
        elif "pickup" in action:
            mask[3] = 1
        elif "drop" in action:
            mask[4] = 1
        elif "toggle" in action:
            mask[5] = 1
        elif "done" in action:
            mask[6] = 1
    return mask

class Shield:
    def __init__(self, shieldfile="current.shield"):
        self.shieldfile = shieldfile
        self.parse()

    def parse(self):
        self.shield = dict()
        self.shield_raw = dict()
        with open(self.shieldfile, "r") as shield:
            shield = shield.readlines()
            for line in tqdm(shield[3:-2]):
                state_valuation = line[line.find("[")+1:line.find("]")]
                actions = line[line.find("]")+1:]
                ints = dict(re.findall(r'([a-zA-Z][_a-zA-Z0-9]+)=(-?[a-zA-Z0-9]+)', state_valuation))
                booleans = re.findall(r'(\!?)([a-zA-Z][_a-zA-Z0-9]+)[\s\t]+', state_valuation)
                booleans = {b[1]: False if b[0] == "!" else True for b in booleans}
                actions = re.findall(r'{([a-zA-Z][_a-zA-Z0-9]+)}', actions)
                if int(ints.get("clock", 0)) != 0:
                    continue
                if int(ints.get("previousActionAgent", 3)) != 3:
                    continue
                self.shield[to_state(ints, booleans)] = actionsToMiniGrid(actions)
                self.shield_raw[to_state(ints, booleans)] = line[line.find("]")+1:]

    def get_action_mask(self, state):
        print(state)
        try:
            return self.shield[state]
        except:
            print("Unsafe State")
            return [0.0] * 7

    def get_action_mask_raw(self, state):
        try:
            return self.shield_raw[state]
        except:
            print("Not listed")

class ManualControl:
    def __init__(
        self,
        env: Env,
        seed=None,
        random_agent=False,
        shieldfile=None,
        enforce=False
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False
        self.random_agent = random_agent
        if shieldfile is not None:
            self.shield = Shield(shieldfile)
        self.enforce= enforce
        self.cumulative_reward = 0

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            if self.random_agent:
                index = np.random.choice(7, 1)[0]
                action = [Actions.left, Actions.right, Actions.forward, Actions.pickup, Actions.drop, Actions.toggle, Actions.done][index]
                print(Actions(action), end=" ")
                if hasattr(self, "shield") and self.enforce:
                    mask = self.shield.get_action_mask(self.env.get_symbolic_state())
                    if mask[Actions(action)] == 1.0:
                        self.step(action)
                    else:
                        print("blocked: ", Actions(action), end=" ")
                else:
                    self.step(action)
                print(" ")
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.env.close()
                        break
                    if event.type == pygame.KEYDOWN:
                        event.key = pygame.key.name(int(event.key))
                        self.key_handler(event)

    def step(self, action: Actions):
        _, reward, terminated, truncated, info = self.env.step(action)
        self.cumulative_reward += reward
        print(f"step={self.env.step_count}, reward={reward:.4f}, cumulative_reward={self.cumulative_reward:.4f}")
        print(info)
        if hasattr(self, "shield") and self.enforce:
            symbolic_state = self.env.get_symbolic_state()
            mask = self.shield.get_action_mask(symbolic_state)
            print(mask)
            print(self.shield.get_action_mask_raw(symbolic_state))

        if terminated:
            print("terminated!")
            input("")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        if hasattr(self, "shield") and self.enforce:
            symbolic_state = self.env.get_symbolic_state()
            mask = self.shield.get_action_mask(symbolic_state)
            print(mask)
            print(self.shield.get_action_mask_raw(symbolic_state))
        self.cumulative_reward = 0
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return
        if key == "f12":
            self.take_screenshot()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            symbolic_state = self.env.get_symbolic_state()
            if hasattr(self, "shield") and self.enforce:
                mask = self.shield.get_action_mask(symbolic_state)
                print(mask)
                print(self.shield.get_action_mask_raw(symbolic_state))
                if mask[Actions(action)] == 1.0:
                    self.step(action)
                else:
                    print(key)
            elif hasattr(self, "shield") and not self.enforce:
                mask = self.shield.get_action_mask(symbolic_state)
                print(mask)
                print(self.shield.get_action_mask_raw(symbolic_state))
            else:
                self.step(action)
        else:
            print(key)

    def take_screenshot(self):
        import datetime
        filename = f"{datetime.datetime.now().isoformat()}.png"
        print(f"Saving a screenshot to '{filename}'")
        window = self.env.window
        screenshot = pygame.Surface(window.get_size())
        screenshot.blit(window, (0,0))
        pygame.image.save(screenshot, filename)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="MiniGrid-MultiRoom-N6-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )
    parser.add_argument(
        "--random-agent",
        action="store_true",
        help="make the agent move around randomly"
    )
    parser.add_argument(
        "--shield-file",
        type=str,
        help="shield file to parse and load",
    )
    parser.add_argument(
        "--no-enforcement",
        action="store_true",
        help="do not enforce, but inform the user abouth shield violations"
    )


    args = parser.parse_args()

    env: MiniGridEnv = gym.make(
        args.env_id,
        tile_size=args.tile_size,
        render_mode="human",
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
    )

    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, args.tile_size)
        env = ImgObsWrapper(env)
    #env.disable_random_start()
    print(env.printGrid(init=True))
    manual_control = ManualControl(env, seed=args.seed, random_agent=args.random_agent, shieldfile=args.shield_file, enforce=args.no_enforcement == False)
    manual_control.start()
