#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import cv2

sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/miniworld-drstrategy')
from miniworld_gymnasium.envs.roomnav import NineRooms

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

class NineRoomsFullyPureGymnasium:
    """Minimal wrapper for NineRoomsDrStrategyEnv."""
    
    def __init__(self, name="NineRooms", obs_level=1, continuous=False, size=64):
        base_env = NineRooms(
            room_size=15,
            door_size=2.5,
            obs_level=obs_level,
            continuous=continuous
        )
        
        # Apply wrappers
        resized_env = ResizeObservationGymnasium(base_env, size)
        self._env = ImageToPyTorch(resized_env)
        self.shape = (size, size)
        
    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed)
    
    def step(self, action):
        return self._env.step(action)
        
    def render_on_pos(self, pos):
        # Get base environment
        base_env = self._env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
            
        # Save current position
        current_pos = base_env.agent.pos
        
        # Move agent to specified position
        base_env.place_agent(pos=pos)
        
        # Render observation
        obs = base_env.render_top_view(POMDP=True)
        
        # Restore agent position
        base_env.place_agent(pos=current_pos)
        
        # Apply resize wrapper
        obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        
        # Apply PyTorch wrapper (HWC to CHW) - but render_top_view might already return correct format
        if len(obs.shape) == 3 and obs.shape[2] <= 4:
            obs = np.transpose(obs, (2, 0, 1))
        
        return obs
    
    @property
    def observation_space(self):
        return self._env.observation_space
    
    @property
    def action_space(self):
        return self._env.action_space