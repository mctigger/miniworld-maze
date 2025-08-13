import numpy as np
import math
from gymnasium import spaces, utils

from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box, Ball, Key

class RoomObjs(MiniWorldEnv, utils.EzPickle):
    """
    Single room with multiple objects
    Inspired by the single room environment of
    the Generative Query Networks paper:
    https://deepmind.com/blog/neural-scene-representation-and-rendering/
    """

    def __init__(self, size=10, **kwargs):
        assert size >= 2
        self.size = size

        MiniWorldEnv.__init__(
            self,
            max_episode_steps=math.inf,
            **kwargs
        )
        utils.EzPickle.__init__(self, size=size, **kwargs)

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex='brick_wall',
            floor_tex='asphalt',
            no_ceiling=True,
        )

        # Reduce chances that objects are too close to see
        self.agent.radius = 1.5

        colorlist = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']

        self.place_entity(Box(color=colorlist[self.np_random.choice(len(colorlist))], size=0.9))

        self.place_entity(Ball(color=colorlist[self.np_random.choice(len(colorlist))], size=0.9))

        self.place_entity(Key(color=colorlist[self.np_random.choice(len(colorlist))]))

        self.place_agent()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        return obs, reward, termination, truncation, info