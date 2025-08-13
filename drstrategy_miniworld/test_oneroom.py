#!/usr/bin/env python3

import os
print("Starting test...")

# Set display for headless mode
os.environ['DISPLAY'] = ':99'

try:
    print("Importing drstrategy_miniworld...")
    import drstrategy_miniworld
    print("Import successful")
    
    print("Importing OneRoom...")
    from drstrategy_miniworld.envs import OneRoom
    print("OneRoom import successful")
    
    print("Creating OneRoom environment...")
    env = OneRoom()
    print("Environment created successfully")
    
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Environment reset successful. Obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    
    print("Taking a step...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step successful. Reward: {reward}")
    
    print("Closing environment...")
    env.close()
    print("Test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()