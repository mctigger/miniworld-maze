"""Image transformation wrappers for Nine Rooms environments."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2


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


class ResizeObservationGymnasium(gym.ObservationWrapper):
    """Resize observations to specified size."""
    
    def __init__(self, env, size):
        super().__init__(env)
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        
        if len(env.observation_space.shape) == 3:
            new_shape = (size[0], size[1], env.observation_space.shape[2])
        else:
            new_shape = (size[0], size[1])
            
        self.observation_space = spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, observation):
        return cv2.resize(observation, self.size, interpolation=cv2.INTER_AREA)