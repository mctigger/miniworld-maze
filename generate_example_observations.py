#!/usr/bin/env python3
"""
Generate example partial and full observations using the pure Gymnasium Nine Rooms implementation.
"""

import sys
import numpy as np
from PIL import Image

# Add path for our pure gymnasium implementation
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences')

from nine_rooms_fully_pure_gymnasium import NineRoomsFullyPureGymnasium

def generate_example_observations():
    """Generate comprehensive example observations."""
    print("=" * 70)
    print("GENERATING EXAMPLE OBSERVATIONS - NINE ROOMS ENVIRONMENT")
    print("Pure Gymnasium Implementation (Zero gym dependency)")
    print("=" * 70)
    
    # Create environment
    print("Creating Nine Rooms environment...")
    env = NineRoomsFullyPureGymnasium(name="NineRooms", obs_level=1, continuous=False, size=64)
    
    # Get base environment for direct render access
    base_env = env._env
    while hasattr(base_env, 'env') or hasattr(base_env, '_env'):
        if hasattr(base_env, 'env'):
            base_env = base_env.env
        elif hasattr(base_env, '_env'):
            base_env = base_env._env
        else:
            break
    
    print(f"Environment created - base type: {type(base_env)}")
    print(f"Environment size: {base_env.max_x - base_env.min_x:.1f} x {base_env.max_z - base_env.min_z:.1f} units")
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Environment reset - standard obs shape: {obs.shape}")
    
    # ===== FULL OBSERVATION EXAMPLES =====
    print(f"\n{'='*50}")
    print("FULL ENVIRONMENT OBSERVATIONS (All 9 Rooms)")
    print(f"{'='*50}")
    
    # 1. Full view with agent at starting position
    print("\n1. Full environment view with agent at start position")
    full_view_start = base_env.render_top_view(POMDP=False)
    Image.fromarray(full_view_start).save('example_full_view_start.png')
    print(f"   ✓ Saved: example_full_view_start.png")
    print(f"   ✓ Agent position: {base_env.agent.pos}")
    print(f"   ✓ Image size: {full_view_start.shape}")
    
    # 2. Full view without agent (clean maze view)
    print("\n2. Full environment view without agent (clean maze)")
    full_view_clean = base_env.render_top_view(POMDP=False, render_ag=False)
    Image.fromarray(full_view_clean).save('example_full_view_clean.png')
    print(f"   ✓ Saved: example_full_view_clean.png")
    print(f"   ✓ Shows: Complete 9-room layout with obstacles")
    
    # 3. Full view with agent in center
    print("\n3. Full environment view with agent in center")
    center_x = (base_env.min_x + base_env.max_x) / 2
    center_z = (base_env.min_z + base_env.max_z) / 2
    base_env.place_agent(pos=[center_x, 0.0, center_z])
    full_view_center = base_env.render_top_view(POMDP=False)
    Image.fromarray(full_view_center).save('example_full_view_center.png')
    print(f"   ✓ Saved: example_full_view_center.png")
    print(f"   ✓ Agent moved to center: [{center_x:.1f}, 0.0, {center_z:.1f}]")
    
    # ===== PARTIAL OBSERVATION EXAMPLES =====
    print(f"\n{'='*50}")
    print("PARTIAL OBSERVATIONS (POMDP Views)")
    print(f"{'='*50}")
    
    # Reset agent to start position for consistent partial views
    base_env.place_agent(pos=[2.5, 0.0, 2.5])
    
    # 1. Partial view from starting position (top-left room)
    print("\n1. Partial view from starting position (top-left room)")
    partial_start = base_env.render_top_view(POMDP=True)
    Image.fromarray(partial_start).save('example_partial_start.png')
    print(f"   ✓ Saved: example_partial_start.png")
    print(f"   ✓ Agent position: {base_env.agent.pos}")
    print(f"   ✓ View radius: 2.5 units (5x5 unit window)")
    
    # 2. Partial view from center room
    print("\n2. Partial view from center room")
    base_env.place_agent(pos=[center_x, 0.0, center_z])
    partial_center = base_env.render_top_view(POMDP=True)
    Image.fromarray(partial_center).save('example_partial_center.png')
    print(f"   ✓ Saved: example_partial_center.png")
    print(f"   ✓ Agent position: [{center_x:.1f}, 0.0, {center_z:.1f}]")
    print(f"   ✓ Shows: Center room with partial views of adjacent rooms")
    
    # 3. Partial view from bottom-right room
    print("\n3. Partial view from bottom-right room")
    bottom_right_x = base_env.max_x - 2.5
    bottom_right_z = base_env.max_z - 2.5
    base_env.place_agent(pos=[bottom_right_x, 0.0, bottom_right_z])
    partial_bottom_right = base_env.render_top_view(POMDP=True)
    Image.fromarray(partial_bottom_right).save('example_partial_bottom_right.png')
    print(f"   ✓ Saved: example_partial_bottom_right.png")
    print(f"   ✓ Agent position: [{bottom_right_x:.1f}, 0.0, {bottom_right_z:.1f}]")
    print(f"   ✓ Shows: Bottom-right room (different colors and obstacles)")
    
    # 4. Partial view at room boundary (doorway)
    print("\n4. Partial view at room boundary (doorway)")
    doorway_x = 15.0  # Between room 0 and 1
    doorway_z = 7.5   # Middle of room boundary
    base_env.place_agent(pos=[doorway_x, 0.0, doorway_z])
    partial_doorway = base_env.render_top_view(POMDP=True)
    Image.fromarray(partial_doorway).save('example_partial_doorway.png')
    print(f"   ✓ Saved: example_partial_doorway.png")
    print(f"   ✓ Agent position: [{doorway_x:.1f}, 0.0, {doorway_z:.1f}]")
    print(f"   ✓ Shows: Doorway between two rooms")
    
    # ===== WRAPPED OBSERVATIONS =====
    print(f"\n{'='*50}")
    print("WRAPPED OBSERVATIONS (Through Gymnasium Interface)")
    print(f"{'='*50}")
    
    # Reset environment and get wrapped observations
    obs, info = env.reset(seed=42)
    
    # 1. Standard gymnasium observation (CHW format, 64x64)
    print("\n1. Standard Gymnasium observation (processed)")
    obs_hwc = np.transpose(obs, (1, 2, 0))  # Convert CHW to HWC for saving
    Image.fromarray(obs_hwc).save('example_gymnasium_standard.png')
    print(f"   ✓ Saved: example_gymnasium_standard.png")
    print(f"   ✓ Format: CHW -> {obs.shape} (PyTorch compatible)")
    print(f"   ✓ Resolution: 64x64 (resized from 80x80)")
    print(f"   ✓ Data type: {obs.dtype}")
    
    # 2. Observations after agent movement
    print("\n2. Gymnasium observations after movement")
    actions = [1, 1, 2]  # forward, forward, turn_left
    action_names = ['forward', 'forward', 'turn_left']
    
    for i, (action, name) in enumerate(zip(actions, action_names)):
        obs, reward, terminated, truncated, info = env.step(action)
        obs_hwc = np.transpose(obs, (1, 2, 0))
        Image.fromarray(obs_hwc).save(f'example_gymnasium_step_{i+1}_{name}.png')
        print(f"   ✓ Saved: example_gymnasium_step_{i+1}_{name}.png")
        print(f"     Action: {name} (action={action}), Reward: {reward}")
    
    # ===== RENDER_ON_POS EXAMPLES =====
    print(f"\n{'='*50}")
    print("RENDER_ON_POS EXAMPLES (Goal Visualization)")
    print(f"{'='*50}")
    
    # Test render_on_pos at different locations
    test_positions = [
        ([7.5, 0.0, 7.5], "center of top-middle room"),
        ([37.5, 0.0, 37.5], "center of bottom-right room"),
        ([22.1, 0.0, 22.1], "center of environment"),
        ([7.5, 0.0, 22.5], "middle-left room"),
    ]
    
    for i, (pos, description) in enumerate(test_positions):
        print(f"\n{i+1}. Render at {description}")
        render_obs = env.render_on_pos(pos)
        Image.fromarray(render_obs).save(f'example_render_on_pos_{i+1}.png')
        print(f"   ✓ Saved: example_render_on_pos_{i+1}.png")
        print(f"   ✓ Position: {pos}")
        print(f"   ✓ Shows: View from {description}")
    
    # ===== SUMMARY =====
    print(f"\n{'='*70}")
    print("SUMMARY - GENERATED EXAMPLE OBSERVATIONS")
    print(f"{'='*70}")
    
    print(f"\n📸 FULL ENVIRONMENT VIEWS (3 images):")
    print(f"   • example_full_view_start.png - Complete 9-room view with agent at start")
    print(f"   • example_full_view_clean.png - Clean maze layout without agent")
    print(f"   • example_full_view_center.png - Complete view with agent in center")
    
    print(f"\n🔍 PARTIAL OBSERVATIONS (4 images):")
    print(f"   • example_partial_start.png - POMDP view from top-left room")
    print(f"   • example_partial_center.png - POMDP view from center room")
    print(f"   • example_partial_bottom_right.png - POMDP view from bottom-right")
    print(f"   • example_partial_doorway.png - POMDP view at room boundary")
    
    print(f"\n🏋️ GYMNASIUM WRAPPED OBSERVATIONS (4 images):")
    print(f"   • example_gymnasium_standard.png - Standard reset observation")
    print(f"   • example_gymnasium_step_1_forward.png - After forward action")
    print(f"   • example_gymnasium_step_2_forward.png - After second forward")
    print(f"   • example_gymnasium_step_3_turn_left.png - After turning left")
    
    print(f"\n🎯 RENDER_ON_POS EXAMPLES (4 images):")
    print(f"   • example_render_on_pos_1.png - View from top-middle room")
    print(f"   • example_render_on_pos_2.png - View from bottom-right room") 
    print(f"   • example_render_on_pos_3.png - View from environment center")
    print(f"   • example_render_on_pos_4.png - View from middle-left room")
    
    print(f"\n🎉 Total: 15 example observations generated!")
    print(f"📦 All images demonstrate the pure Gymnasium Nine Rooms environment")
    print(f"✨ Zero dependency on old gym package - fully modern implementation!")

if __name__ == "__main__":
    generate_example_observations()