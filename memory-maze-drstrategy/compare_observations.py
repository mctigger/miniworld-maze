#!/usr/bin/env python3
"""
Script to capture and save observations for comparison between
DrStrategy and memory-maze-drstrategy implementations.
"""

import os
import numpy as np
from PIL import Image
import gymnasium as gym
import memory_maze

def capture_observations(env_id, num_steps=10, camera_resolution=256, output_prefix="obs"):
    """Capture observations from an environment and save as images."""
    
    # Set up environment
    os.environ['MUJOCO_GL'] = 'osmesa'
    
    print(f"Creating environment: {env_id}")
    if camera_resolution:
        env = gym.make(env_id, camera_resolution=camera_resolution)
    else:
        env = gym.make(env_id)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset environment
    obs, info = env.reset()
    
    # Create output directory
    output_dir = f"{output_prefix}_{env_id.replace('-', '_')}_res{camera_resolution}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving observations to: {output_dir}")
    
    # Save initial observation
    if isinstance(obs, dict) and 'image' in obs:
        image_obs = obs['image']
    else:
        image_obs = obs
    
    # Convert to uint8 if needed
    if image_obs.dtype != np.uint8:
        image_obs = np.clip(image_obs * 255, 0, 255).astype(np.uint8)
    
    # Save initial observation
    img = Image.fromarray(image_obs)
    img.save(os.path.join(output_dir, f"step_000_reset.png"))
    print(f"Saved step 000 (reset): {image_obs.shape}")
    
    # Take steps and save observations
    actions = [1, 2, 1, 3, 1, 2, 1, 3, 1, 2]  # Forward, turn left, forward, turn right, etc.
    
    for step in range(num_steps):
        action = actions[step % len(actions)]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if isinstance(obs, dict) and 'image' in obs:
            image_obs = obs['image']
        else:
            image_obs = obs
        
        # Convert to uint8 if needed
        if image_obs.dtype != np.uint8:
            image_obs = np.clip(image_obs * 255, 0, 255).astype(np.uint8)
        
        # Save observation
        img = Image.fromarray(image_obs)
        img.save(os.path.join(output_dir, f"step_{step+1:03d}_action{action}.png"))
        print(f"Saved step {step+1:03d} (action {action}): reward={reward:.3f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step+1}")
            break
    
    env.close()
    print(f"Completed! Check {output_dir} for images.\n")
    
    return output_dir


def main():
    """Capture observations from both implementations."""
    
    print("=" * 60)
    print("OBSERVATION COMPARISON SCRIPT")
    print("=" * 60)
    
    # Test parameters
    camera_resolution = 256
    num_steps = 5
    
    try:
        # Capture from our DrStrategy complex maze implementation
        print("\n1. Capturing from memory-maze-drstrategy (DrStrategy complex maze):")
        dir1 = capture_observations(
            "MemoryMaze-cmaze-7x7-fixed-layout-v0",
            num_steps=num_steps,
            camera_resolution=camera_resolution,
            output_prefix="our_drstrategy"
        )
        
        # Capture from simple four-room layout for comparison
        print("2. Capturing from memory-maze-drstrategy (simple four rooms):")
        dir2 = capture_observations(
            "MemoryMaze-four-rooms-7x7-fixed-layout-v0", 
            num_steps=num_steps,
            camera_resolution=camera_resolution,
            output_prefix="our_simple"
        )
        
        print("=" * 60)
        print("COMPARISON RESULTS:")
        print(f"DrStrategy complex maze images: {dir1}")
        print(f"Simple four rooms images: {dir2}")
        print("\nYou can now:")
        print("1. Compare the wall textures visually")
        print("2. Check if complex maze has colorful walls vs simple yellow walls")
        print("3. Verify floor texture variations in complex maze")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during capture: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()