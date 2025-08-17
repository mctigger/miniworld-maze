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

# Import from the clean, refactored modules
from .environments.factory import NineRoomsEnvironmentWrapper, create_nine_rooms_env

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