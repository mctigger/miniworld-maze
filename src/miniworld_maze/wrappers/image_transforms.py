"""Image transformation wrappers for Nine Rooms environments."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ImageToPyTorch(gym.ObservationWrapper):
    """Convert HWC to CHW format for PyTorch compatibility."""

    def __init__(self, env):
        """
        Initialize PyTorch-compatible image transformation wrapper.

        Transforms observation space from HWC (Height, Width, Channels) format
        to CHW (Channels, Height, Width) format expected by PyTorch models.

        Args:
            env: The environment to wrap
        """
        super(ImageToPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        """
        Transform observation from HWC to CHW format.

        Args:
            observation: Input observation in HWC format (H, W, C)

        Returns:
            np.ndarray: Observation in CHW format (C, H, W)
        """
        return np.transpose(observation, (2, 0, 1))
