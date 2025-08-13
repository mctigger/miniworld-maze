import numpy as np
import math
from gymnasium import spaces, utils

from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box

# Simulation parameters for robot configuration
# These assume a robot about 15cm tall with a pi camera module v2
# Note: Using default parameters since params system from drstrategy is not available

class SimToRealGoto(MiniWorldEnv, utils.EzPickle):
    """
    Environment designed for sim-to-real transfer.
    In this environment, the robot has to go to the red box.
    """

    def __init__(self, **kwargs):
        MiniWorldEnv.__init__(
            self,
            max_episode_steps=100,
            domain_rand=True,
            **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # 1-2 meter wide rink
        size = self.np_random.uniform(1, 2)

        wall_height = self.np_random.uniform(0.20, 0.50)

        box_size = self.np_random.uniform(0.07, 0.12)

        self.agent.radius = 0.11

        floor_textures = [
            'cardboard',
            'wood',
            'wood_planks',
        ]
        floor_tex = floor_textures[self.np_random.choice(len(floor_textures))]

        wall_textures = [
            'drywall',
            'stucco',
            'cardboard',
            # Chosen because they have visible lines/seams
            'concrete_tiles',
            'ceiling_tiles',
        ]
        wall_tex = wall_textures[self.np_random.choice(len(wall_textures))]

        # Create a long rectangular room
        room = self.add_rect_room(
            min_x=0,
            max_x=size,
            min_z=0,
            max_z=size,
            no_ceiling=True,
            wall_height=wall_height,
            wall_tex=wall_tex,
            floor_tex=floor_tex
        )

        self.box = self.place_entity(Box(color='red', size=box_size))

        # Place the agent a random distance away from the goal
        self.place_agent()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info