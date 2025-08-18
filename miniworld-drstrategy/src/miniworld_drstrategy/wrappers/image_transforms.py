"""Image transformation wrappers for Nine Rooms environments."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ImageToPyTorch(gym.ObservationWrapper):
    """Convert HWC to CHW format for PyTorch compatibility."""
    
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]), 
            dtype=np.uint8
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


