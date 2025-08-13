#!/usr/bin/env python3
"""
Test script to verify MiniWorld observations are actually changing.
Saves images to disk to visually verify changes.
"""

import numpy as np
from PIL import Image
from drstrategy_miniworld.envs import OneRoom

def save_obs_as_image(obs, filename):
    """Save observation as PNG image."""
    # Ensure correct format
    if obs.max() <= 1.0:
        obs = (obs * 255).astype(np.uint8)
    
    img = Image.fromarray(obs, 'RGB')
    img.save(filename)
    print(f"Saved {filename} (sum: {obs.sum()})")

def main():
    print("Testing MiniWorld observation changes...")
    
    env = OneRoom()
    
    # Reset and save initial observation
    obs, info = env.reset()
    save_obs_as_image(obs, "obs_0_reset.png")
    
    # Take several actions and save observations
    actions_to_try = [0, 1, 2, 0, 1, 2, 0, 1]  # turn left, turn right, move forward
    action_names = {0: "turn_left", 1: "turn_right", 2: "move_forward", 3: "move_back"}
    
    for i, action in enumerate(actions_to_try):
        print(f"\n--- Step {i+1}: {action_names.get(action, f'action_{action}')} ---")
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print agent state
        if hasattr(env, 'agent'):
            print(f"Agent pos: {env.agent.pos}")
            print(f"Agent dir: {env.agent.dir:.3f}")
        
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        
        # Save observation
        filename = f"obs_{i+1}_{action_names.get(action, f'action_{action}')}.png"
        save_obs_as_image(obs, filename)
        
        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()
            save_obs_as_image(obs, f"obs_{i+1}_reset_after_end.png")
    
    env.close()
    print(f"\nTest complete! Check the saved PNG files to see if observations change.")
    print("If images look identical, there's an environment issue.")
    print("If images are different, the issue is in the web visualization code.")

if __name__ == "__main__":
    main()