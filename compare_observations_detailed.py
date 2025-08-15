#!/usr/bin/env python3
"""
Detailed comparison of observations between original DrStrategy and pure Gymnasium implementations.
"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add paths for both implementations
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy-refactor')
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy-refactor/drstrategy')
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy-refactor/drstrategy_envs')
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy-refactor/drstrategy_envs/drstrategy_envs')

from nine_rooms_fully_pure_gymnasium import NineRoomsFullyPureGymnasium

def get_original_observations():
    """Get observations from original DrStrategy implementation."""
    print("Getting observations from original DrStrategy...")
    
    # Import original DrStrategy
    from drstrategy.envs import RoomNav
    
    # Create original environment
    env = RoomNav(name="NineRooms", obs_level=1, continuous=False, size=64)
    
    # Get base environment for direct access
    base_env = env._env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    
    # Reset and get observations
    obs_original = base_env.reset()
    observations = [obs_original]
    
    # Take same sequence of actions
    actions = [1, 1, 2]  # forward, forward, turn_left
    for action in actions:
        step_result = base_env.step(action)
        if isinstance(step_result, tuple):
            obs = step_result[0]
        else:
            obs = step_result
        
        # Apply same transformations as wrapper chain
        import cv2
        obs_resized = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
        obs_chw = np.moveaxis(obs_resized, 2, 0)  # Convert to CHW
        observations.append(obs_chw)
    
    # Get render_on_pos observation
    render_obs = env.render_on_pos([7.5, 0.0, 22.5])
    observations.append(render_obs)
    
    return observations

def get_gymnasium_observations():
    """Get observations from pure Gymnasium implementation."""
    print("Getting observations from pure Gymnasium implementation...")
    
    # Create pure gymnasium environment
    env = NineRoomsFullyPureGymnasium(name="NineRooms", obs_level=1, continuous=False, size=64)
    
    # Reset and get observations
    obs, info = env.reset(seed=42)  # Use fixed seed for reproducibility
    observations = [obs]
    
    # Take same sequence of actions
    actions = [1, 1, 2]  # forward, forward, turn_left
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
    
    # Get render_on_pos observation
    render_obs = env.render_on_pos([7.5, 0.0, 22.5])
    observations.append(render_obs)
    
    return observations

def compare_observations(original_obs, gymnasium_obs):
    """Compare observations pixel by pixel."""
    print(f"\nComparing {len(original_obs)} observation pairs...")
    
    all_identical = True
    
    for i, (orig, gym) in enumerate(zip(original_obs, gymnasium_obs)):
        print(f"\n--- Observation {i} ---")
        
        # Handle different formats
        if len(orig.shape) == 3 and orig.shape[0] == 3:  # CHW format
            orig_hwc = np.transpose(orig, (1, 2, 0))
        else:
            orig_hwc = orig
            
        if len(gym.shape) == 3 and gym.shape[0] == 3:  # CHW format
            gym_hwc = np.transpose(gym, (1, 2, 0))
        else:
            gym_hwc = gym
        
        print(f"Original shape: {orig.shape} -> HWC: {orig_hwc.shape}")
        print(f"Gymnasium shape: {gym.shape} -> HWC: {gym_hwc.shape}")
        print(f"Original dtype: {orig_hwc.dtype}, range: [{orig_hwc.min()}, {orig_hwc.max()}]")
        print(f"Gymnasium dtype: {gym_hwc.dtype}, range: [{gym_hwc.min()}, {gym_hwc.max()}]")
        
        # Check if shapes match
        if orig_hwc.shape != gym_hwc.shape:
            print(f"❌ Shape mismatch: {orig_hwc.shape} vs {gym_hwc.shape}")
            all_identical = False
            continue
        
        # Check if arrays are identical
        if np.array_equal(orig_hwc, gym_hwc):
            print("✅ Observations are IDENTICAL pixel-for-pixel!")
        else:
            print("❌ Observations differ!")
            all_identical = False
            
            # Calculate differences
            diff = np.abs(orig_hwc.astype(float) - gym_hwc.astype(float))
            max_diff = diff.max()
            mean_diff = diff.mean()
            num_diff_pixels = (diff > 0).sum()
            total_pixels = diff.size
            
            print(f"   Max difference: {max_diff}")
            print(f"   Mean difference: {mean_diff:.3f}")
            print(f"   Different pixels: {num_diff_pixels}/{total_pixels} ({100*num_diff_pixels/total_pixels:.2f}%)")
            
            # Save difference visualization
            if max_diff > 0:
                diff_vis = (diff * 255 / max_diff).astype(np.uint8)
                Image.fromarray(diff_vis).save(f'difference_obs_{i}.png')
                print(f"   Saved difference image: difference_obs_{i}.png")
        
        # Save individual observations for manual inspection
        Image.fromarray(orig_hwc).save(f'original_obs_{i}.png')
        Image.fromarray(gym_hwc).save(f'gymnasium_obs_{i}.png')
        print(f"   Saved: original_obs_{i}.png, gymnasium_obs_{i}.png")
    
    return all_identical

def main():
    """Main comparison function."""
    print("=" * 60)
    print("DETAILED OBSERVATION COMPARISON")
    print("Original DrStrategy vs Pure Gymnasium Implementation")
    print("=" * 60)
    
    try:
        # Get observations from both implementations
        original_obs = get_original_observations()
        
        # Reset random seed to ensure same conditions
        np.random.seed(42)
        gymnasium_obs = get_gymnasium_observations()
        
        # Compare observations
        all_identical = compare_observations(original_obs, gymnasium_obs)
        
        print("\n" + "=" * 60)
        if all_identical:
            print("🎉 SUCCESS: All observations are IDENTICAL!")
            print("📊 The pure Gymnasium implementation produces exactly the same output as the original DrStrategy!")
        else:
            print("⚠️  DIFFERENCES FOUND: Some observations differ between implementations")
            print("🔍 Check the saved images and difference visualizations for details")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()