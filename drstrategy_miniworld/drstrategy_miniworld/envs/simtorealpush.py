import numpy as np
import math
from gymnasium import spaces, utils

from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box

class SimToRealPush(MiniWorldEnv, utils.EzPickle):
    """
    Environment designed for sim-to-real transfer.
    In this environment, the robot has to push the
    red box towards the yellow box.
    """

    def __init__(self, **kwargs):
        MiniWorldEnv.__init__(
            self,
            max_episode_steps=150,
            domain_rand=True,
            **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only the movement actions (left, right, forward, back)
        self.action_space = spaces.Discrete(self.actions.move_back+1)

    def _gen_world(self):
        # Size of the rink the robot is placed in
        size = self.np_random.uniform(1.6, 1.7)
        wall_height = self.np_random.uniform(0.42, 0.50)

        box1_size = self.np_random.uniform(0.075, 0.090)
        box2_size = self.np_random.uniform(0.075, 0.090)

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
            # Materials chosen because they have visible lines/seams
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

        # Target distance for the boxes
        min_dist = box1_size + box2_size
        self.goal_dist = 1.5 * min_dist

        # Avoid spawning boxes in the corners (where they can't be pushed)
        min_pos = 2 * 0.11  # Using agent radius

        # Place the red box to be pushed
        self.red_box = self.place_entity(
            Box(color='red', size=box1_size),
            min_x=min_pos, max_x=size-min_pos,
            min_z=min_pos, max_z=size-min_pos
        )

        # Place the yellow target box
        self.yellow_box = self.place_entity(
            Box(color='yellow', size=box2_size),
            min_x=min_pos, max_x=size-min_pos,
            min_z=min_pos, max_z=size-min_pos
        )

        # Place the agent
        self.place_agent()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        # Calculate distance between boxes
        box_dist = np.linalg.norm(self.red_box.pos - self.yellow_box.pos)

        # Reward for getting boxes close together
        if box_dist < self.goal_dist:
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info