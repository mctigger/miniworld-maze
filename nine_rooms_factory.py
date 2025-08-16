#!/usr/bin/env python3
"""
Factory for creating different Nine Rooms environment variants.
Supports: NineRooms, SpiralNineRooms, TwentyFiveRooms
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import cv2

sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/nine_rooms_pure_gymnasium_env')
from miniworld_gymnasium.envs.roomnav import NineRooms, SpiralNineRooms, TwentyFiveRooms

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

class NineRoomsEnvironmentWrapper:
    """Unified wrapper for all Nine Rooms environment variants."""
    
    def __init__(self, variant="NineRooms", obs_level=1, continuous=False, size=64, room_size=15, door_size=2.5):
        """
        Create a Nine Rooms environment variant.
        
        Args:
            variant: Environment variant ("NineRooms", "SpiralNineRooms", "TwentyFiveRooms")
            obs_level: Observation level (1 for RGB)
            continuous: Whether to use continuous actions
            size: Observation image size (will be resized to size x size)
            room_size: Size of each room in environment units
            door_size: Size of doors between rooms
        """
        self.variant = variant
        
        # Select the appropriate environment class
        env_classes = {
            "NineRooms": NineRooms,
            "SpiralNineRooms": SpiralNineRooms,
            "TwentyFiveRooms": TwentyFiveRooms
        }
        
        if variant not in env_classes:
            raise ValueError(f"Unknown variant '{variant}'. Available: {list(env_classes.keys())}")
        
        env_class = env_classes[variant]
        
        # Create base environment
        base_env = env_class(
            room_size=room_size,
            door_size=door_size,
            obs_level=obs_level,
            continuous=continuous
        )
        
        # Apply wrappers
        env = ResizeObservationGymnasium(base_env, size)
        env = ImageToPyTorch(env)
        
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    
    def reset(self, **kwargs):
        return self._env.reset(**kwargs)
    
    def step(self, action):
        return self._env.step(action)
    
    def render(self, mode='rgb_array'):
        return self._env.render()
    
    def close(self):
        return self._env.close()
    
    def render_on_pos(self, pos):
        """Render observation from a specific position."""
        # Get access to the base environment
        base_env = self._env
        while hasattr(base_env, 'env') or hasattr(base_env, '_env'):
            if hasattr(base_env, 'env'):
                base_env = base_env.env
            elif hasattr(base_env, '_env'):
                base_env = base_env._env
            else:
                break
        
        # Store original position
        original_pos = base_env.agent.pos.copy()
        
        # Move agent to target position
        base_env.place_agent(pos=pos)
        
        # Get observation through the wrapper chain
        obs = self._env.render()
        
        # Restore original position
        base_env.place_agent(pos=original_pos)
        
        # Apply wrapper transformations manually for consistency
        # First resize
        if hasattr(self._env, 'size'):
            obs = cv2.resize(obs, self._env.size, interpolation=cv2.INTER_AREA)
        
        # Then convert to PyTorch format (CHW)
        obs = np.transpose(obs, (2, 0, 1))
        
        return obs

def create_nine_rooms_env(variant="NineRooms", **kwargs):
    """
    Factory function to create Nine Rooms environment variants.
    
    Args:
        variant: Environment variant ("NineRooms", "SpiralNineRooms", "TwentyFiveRooms")
        **kwargs: Additional arguments passed to NineRoomsEnvironmentWrapper
    
    Returns:
        NineRoomsEnvironmentWrapper instance
    """
    return NineRoomsEnvironmentWrapper(variant=variant, **kwargs)

# For backward compatibility
def NineRoomsFullyPureGymnasium(name="NineRooms", obs_level=1, continuous=False, size=64):
    """Legacy function for backward compatibility."""
    return create_nine_rooms_env(variant="NineRooms", obs_level=obs_level, continuous=continuous, size=size)

if __name__ == "__main__":
    # Test all variants
    variants = ["NineRooms", "SpiralNineRooms", "TwentyFiveRooms"]
    
    for variant in variants:
        print(f"\nTesting {variant}...")
        env = create_nine_rooms_env(variant=variant, size=64)
        
        obs, info = env.reset()
        print(f"✅ {variant}: obs_shape={obs.shape}, action_space={env.action_space}")
        
        # Test a step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step: action={action}, reward={reward}, done={terminated or truncated}")
        
        env.close()
    
    print("\n🎉 All variants working correctly!")