#!/usr/bin/env python3
"""
Controlled comparison with fixed seed and deterministic actions.
"""

import sys
import numpy as np
from PIL import Image

# Add paths for both implementations
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy-refactor')
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy-refactor/drstrategy')
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy-refactor/drstrategy_envs')
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy-refactor/drstrategy_envs/drstrategy_envs')

from nine_rooms_fully_pure_gymnasium import NineRoomsFullyPureGymnasium

def test_controlled_comparison():
    """Test with controlled conditions."""
    print("=" * 60)
    print("CONTROLLED COMPARISON - Fixed seed, deterministic actions")
    print("=" * 60)
    
    # Set global random seed
    np.random.seed(12345)
    
    # Test 1: Compare initial observations with same seed
    print("\n1. Comparing initial observations with fixed seed...")
    
    # Original DrStrategy
    from drstrategy.envs import RoomNav
    env_orig = RoomNav(name="NineRooms", obs_level=1, continuous=False, size=64)
    
    # Gymnasium implementation  
    env_gym = NineRoomsFullyPureGymnasium(name="NineRooms", obs_level=1, continuous=False, size=64)
    
    # Reset both with same seed (though original doesn't use it in reset)
    base_env_orig = env_orig._env
    while hasattr(base_env_orig, 'env'):
        base_env_orig = base_env_orig.env
    base_env_orig.seed(12345)
    obs_orig = base_env_orig.reset()
    
    obs_gym, _ = env_gym.reset(seed=12345)
    
    # Apply same resizing to original
    import cv2
    obs_orig_resized = cv2.resize(obs_orig, (64, 64), interpolation=cv2.INTER_AREA)
    obs_orig_chw = np.moveaxis(obs_orig_resized, 2, 0)
    
    # Compare
    obs_gym_hwc = np.transpose(obs_gym, (1, 2, 0))
    obs_orig_hwc = np.transpose(obs_orig_chw, (1, 2, 0))
    
    if np.array_equal(obs_orig_hwc, obs_gym_hwc):
        print("✅ Initial observations are IDENTICAL!")
    else:
        diff = np.abs(obs_orig_hwc.astype(float) - obs_gym_hwc.astype(float))
        print(f"❌ Initial observations differ - max diff: {diff.max()}, mean: {diff.mean():.3f}")
        
        # Save for inspection
        Image.fromarray(obs_orig_hwc).save('controlled_original_init.png')
        Image.fromarray(obs_gym_hwc).save('controlled_gymnasium_init.png')
        diff_vis = (diff * 255 / diff.max()).astype(np.uint8)
        Image.fromarray(diff_vis).save('controlled_diff_init.png')
    
    # Test 2: Compare render_on_pos with same position
    print("\n2. Comparing render_on_pos with same position...")
    
    test_pos = [7.5, 0.0, 22.5]
    render_orig = env_orig.render_on_pos(test_pos)
    render_gym = env_gym.render_on_pos(test_pos)
    
    if np.array_equal(render_orig, render_gym):
        print("✅ render_on_pos observations are IDENTICAL!")
    else:
        diff = np.abs(render_orig.astype(float) - render_gym.astype(float))
        print(f"❌ render_on_pos observations differ - max diff: {diff.max()}, mean: {diff.mean():.3f}")
        
        # Save for inspection
        Image.fromarray(render_orig).save('controlled_original_render.png')
        Image.fromarray(render_gym).save('controlled_gymnasium_render.png')
        diff_vis = (diff * 255 / diff.max()).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
        Image.fromarray(diff_vis).save('controlled_diff_render.png')
    
    # Test 3: Check if the environments have same room configuration
    print("\n3. Comparing room configurations...")
    
    print(f"Original rooms: {len(env_orig.rooms)}")
    print(f"Gymnasium rooms: {len(env_gym.rooms)}")
    
    rooms_match = True
    for i, (orig_room, gym_room) in enumerate(zip(env_orig.rooms, env_gym.rooms)):
        if not np.allclose(orig_room, gym_room):
            print(f"❌ Room {i} differs: {orig_room} vs {gym_room}")
            rooms_match = False
    
    if rooms_match:
        print("✅ Room configurations are identical!")
    
    # Test 4: Check goal positions
    print("\n4. Comparing goal positions...")
    
    goals_match = True
    for i, (orig_goals, gym_goals) in enumerate(zip(env_orig.goal_positions, env_gym.goal_positions)):
        for j, (orig_goal, gym_goal) in enumerate(zip(orig_goals, gym_goals)):
            if not np.allclose(orig_goal, gym_goal):
                print(f"❌ Goal {i},{j} differs: {orig_goal} vs {gym_goal}")
                goals_match = False
    
    if goals_match:
        print("✅ Goal positions are identical!")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_controlled_comparison()