#!/usr/bin/env python3
"""
Generate example partial and full observations for all three Nine Rooms environment variants:
- NineRooms (classic 3x3 grid)
- SpiralNineRooms (3x3 grid with spiral connections)
- TwentyFiveRooms (5x5 grid)
"""

import sys
import numpy as np
from PIL import Image
import os

# Add path for our implementations
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences')

from nine_rooms_factory import create_nine_rooms_env

def generate_variant_observations(variant_name, output_dir=None):
    """Generate comprehensive example observations for a specific environment variant."""
    if output_dir is None:
        output_dir = f"{variant_name.lower()}_observations"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"GENERATING EXAMPLE OBSERVATIONS - {variant_name.upper()}")
    print("Pure Gymnasium Implementation (Zero gym dependency)")
    print("=" * 70)
    
    # Create environment
    print(f"Creating {variant_name} environment...")
    env = create_nine_rooms_env(variant=variant_name, size=64)
    
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
    print(f"Room connections: {len(base_env.connections)}")
    print(f"Room textures: {len(base_env.textures)}")
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Environment reset - standard obs shape: {obs.shape}")
    
    # ===== FULL OBSERVATION EXAMPLES =====
    print(f"\n{'='*50}")
    print(f"FULL ENVIRONMENT OBSERVATIONS ({variant_name})")
    print(f"{'='*50}")
    
    # 1. Full view with agent at starting position
    print("\n1. Full environment view with agent at start position")
    full_view_start = base_env.render_top_view(POMDP=False)
    Image.fromarray(full_view_start).save(f'{output_dir}/full_view_start.png')
    print(f"   ✓ Saved: {output_dir}/full_view_start.png")
    print(f"   ✓ Agent position: {base_env.agent.pos}")
    print(f"   ✓ Image size: {full_view_start.shape}")
    
    # 2. Full view without agent (clean maze view)
    print("\n2. Full environment view without agent (clean maze)")
    full_view_clean = base_env.render_top_view(POMDP=False, render_ag=False)
    Image.fromarray(full_view_clean).save(f'{output_dir}/full_view_clean.png')
    print(f"   ✓ Saved: {output_dir}/full_view_clean.png")
    print(f"   ✓ Shows: Complete {variant_name} layout with obstacles")
    
    # 3. Full view with agent in center
    print("\n3. Full environment view with agent in center")
    center_x = (base_env.min_x + base_env.max_x) / 2
    center_z = (base_env.min_z + base_env.max_z) / 2
    base_env.place_agent(pos=[center_x, 0.0, center_z])
    full_view_center = base_env.render_top_view(POMDP=False)
    Image.fromarray(full_view_center).save(f'{output_dir}/full_view_center.png')
    print(f"   ✓ Saved: {output_dir}/full_view_center.png")
    print(f"   ✓ Agent moved to center: [{center_x:.1f}, 0.0, {center_z:.1f}]")
    
    # ===== PARTIAL OBSERVATION EXAMPLES =====
    print(f"\n{'='*50}")
    print("PARTIAL OBSERVATIONS (POMDP Views)")
    print(f"{'='*50}")
    
    # Reset agent to start position for consistent partial views
    base_env.place_agent(pos=[2.5, 0.0, 2.5])
    
    # 1. Partial view from starting position
    print("\n1. Partial view from starting position")
    partial_start = base_env.render_top_view(POMDP=True)
    Image.fromarray(partial_start).save(f'{output_dir}/partial_start.png')
    print(f"   ✓ Saved: {output_dir}/partial_start.png")
    print(f"   ✓ Agent position: {base_env.agent.pos}")
    print(f"   ✓ View radius: 2.5 units (5x5 unit window)")
    
    # 2. Partial view from center
    print("\n2. Partial view from center")
    base_env.place_agent(pos=[center_x, 0.0, center_z])
    partial_center = base_env.render_top_view(POMDP=True)
    Image.fromarray(partial_center).save(f'{output_dir}/partial_center.png')
    print(f"   ✓ Saved: {output_dir}/partial_center.png")
    print(f"   ✓ Agent position: [{center_x:.1f}, 0.0, {center_z:.1f}]")
    print(f"   ✓ Shows: Center area with partial views of adjacent rooms")
    
    # 3. Partial view from corner/edge
    print("\n3. Partial view from corner/edge area")
    corner_x = base_env.max_x - 2.5
    corner_z = base_env.max_z - 2.5
    base_env.place_agent(pos=[corner_x, 0.0, corner_z])
    partial_corner = base_env.render_top_view(POMDP=True)
    Image.fromarray(partial_corner).save(f'{output_dir}/partial_corner.png')
    print(f"   ✓ Saved: {output_dir}/partial_corner.png")
    print(f"   ✓ Agent position: [{corner_x:.1f}, 0.0, {corner_z:.1f}]")
    print(f"   ✓ Shows: Corner/edge area view")
    
    # 4. Partial view at strategic location (depends on variant)
    print("\n4. Partial view at strategic location")
    if variant_name == "NineRooms":
        strategic_x, strategic_z = 15.0, 7.5  # Room boundary
        description = "room boundary (doorway)"
    elif variant_name == "SpiralNineRooms":
        strategic_x, strategic_z = 22.5, 15.0  # Center of spiral
        description = "spiral center area"
    else:  # TwentyFiveRooms
        strategic_x, strategic_z = 37.5, 37.5  # Mid-outer area
        description = "mid-outer area"
    
    base_env.place_agent(pos=[strategic_x, 0.0, strategic_z])
    partial_strategic = base_env.render_top_view(POMDP=True)
    Image.fromarray(partial_strategic).save(f'{output_dir}/partial_strategic.png')
    print(f"   ✓ Saved: {output_dir}/partial_strategic.png")
    print(f"   ✓ Agent position: [{strategic_x:.1f}, 0.0, {strategic_z:.1f}]")
    print(f"   ✓ Shows: {description}")
    
    # ===== WRAPPED OBSERVATIONS =====
    print(f"\n{'='*50}")
    print("WRAPPED OBSERVATIONS (Through Gymnasium Interface)")
    print(f"{'='*50}")
    
    # Reset environment and get wrapped observations
    obs, info = env.reset(seed=42)
    
    # 1. Standard gymnasium observation (CHW format, 64x64)
    print("\n1. Standard Gymnasium observation (processed)")
    obs_hwc = np.transpose(obs, (1, 2, 0))  # Convert CHW to HWC for saving
    Image.fromarray(obs_hwc).save(f'{output_dir}/gymnasium_standard.png')
    print(f"   ✓ Saved: {output_dir}/gymnasium_standard.png")
    print(f"   ✓ Format: CHW -> {obs.shape} (PyTorch compatible)")
    print(f"   ✓ Resolution: 64x64 (resized from 80x80)")
    print(f"   ✓ Data type: {obs.dtype}")
    
    # 2. Observations after agent movement
    print("\n2. Gymnasium observations after movement")
    actions = [2, 2, 1]  # move_forward, move_forward, turn_right
    action_names = ['move_forward', 'move_forward', 'turn_right']
    
    for i, (action, name) in enumerate(zip(actions, action_names)):
        obs, reward, terminated, truncated, info = env.step(action)
        obs_hwc = np.transpose(obs, (1, 2, 0))
        Image.fromarray(obs_hwc).save(f'{output_dir}/gymnasium_step_{i+1}_{name}.png')
        print(f"   ✓ Saved: {output_dir}/gymnasium_step_{i+1}_{name}.png")
        print(f"     Action: {name} (action={action}), Reward: {reward}")
    
    # ===== RENDER_ON_POS EXAMPLES =====
    print(f"\n{'='*50}")
    print("RENDER_ON_POS EXAMPLES (Goal Visualization)")
    print(f"{'='*50}")
    
    # Test render_on_pos at different locations (adapted per variant)
    if variant_name == "NineRooms":
        test_positions = [
            ([7.5, 0.0, 7.5], "top-middle room center"),
            ([37.5, 0.0, 37.5], "bottom-right room center"),
            ([22.5, 0.0, 22.5], "environment center"),
            ([7.5, 0.0, 22.5], "middle-left room center"),
        ]
    elif variant_name == "SpiralNineRooms":
        test_positions = [
            ([7.5, 0.0, 7.5], "top-left room (spiral start)"),
            ([37.5, 0.0, 37.5], "bottom-right room (spiral end)"),
            ([22.5, 0.0, 7.5], "top-right room"),
            ([7.5, 0.0, 37.5], "bottom-left room"),
        ]
    else:  # TwentyFiveRooms
        test_positions = [
            ([37.5, 0.0, 37.5], "room (1,1) - near corner"),
            ([112.5, 0.0, 112.5], "room (4,4) - far corner"),
            ([75.0, 0.0, 75.0], "center room (2,2)"),
            ([37.5, 0.0, 112.5], "room (1,4) - opposite corner"),
        ]
    
    for i, (pos, description) in enumerate(test_positions):
        print(f"\n{i+1}. Render at {description}")
        render_obs = env.render_on_pos(pos)
        
        # Convert CHW to HWC for PIL
        if len(render_obs.shape) == 3 and render_obs.shape[0] == 3:
            render_obs = np.transpose(render_obs, (1, 2, 0))
            
        Image.fromarray(render_obs).save(f'{output_dir}/render_on_pos_{i+1}.png')
        print(f"   ✓ Saved: {output_dir}/render_on_pos_{i+1}.png")
        print(f"   ✓ Position: {pos}")
        print(f"   ✓ Shows: View from {description}")
    
    # ===== VARIANT SUMMARY =====
    print(f"\n{'='*70}")
    print(f"SUMMARY - {variant_name.upper()} OBSERVATIONS")
    print(f"{'='*70}")
    
    print(f"\n📸 FULL ENVIRONMENT VIEWS (3 images):")
    print(f"   • {output_dir}/full_view_start.png - Complete layout with agent at start")
    print(f"   • {output_dir}/full_view_clean.png - Clean maze layout without agent")
    print(f"   • {output_dir}/full_view_center.png - Complete view with agent in center")
    
    print(f"\n🔍 PARTIAL OBSERVATIONS (4 images):")
    print(f"   • {output_dir}/partial_start.png - POMDP view from starting position")
    print(f"   • {output_dir}/partial_center.png - POMDP view from center")
    print(f"   • {output_dir}/partial_corner.png - POMDP view from corner/edge")
    print(f"   • {output_dir}/partial_strategic.png - POMDP view from strategic location")
    
    print(f"\n🏋️ GYMNASIUM WRAPPED OBSERVATIONS (4 images):")
    print(f"   • {output_dir}/gymnasium_standard.png - Standard reset observation")
    print(f"   • {output_dir}/gymnasium_step_1_move_forward.png - After forward action")
    print(f"   • {output_dir}/gymnasium_step_2_move_forward.png - After second forward")
    print(f"   • {output_dir}/gymnasium_step_3_turn_right.png - After turning right")
    
    print(f"\n🎯 RENDER_ON_POS EXAMPLES (4 images):")
    for i, (pos, description) in enumerate(test_positions):
        print(f"   • {output_dir}/render_on_pos_{i+1}.png - View from {description}")
    
    print(f"\n✨ Total: 15 example observations generated for {variant_name}!")
    
    env.close()
    return output_dir

def generate_all_variants_observations():
    """Generate observations for all three environment variants."""
    print("=" * 80)
    print("GENERATING COMPREHENSIVE OBSERVATIONS FOR ALL NINE ROOMS VARIANTS")
    print("=" * 80)
    
    variants = ["NineRooms", "SpiralNineRooms", "TwentyFiveRooms"]
    output_dirs = []
    
    for variant in variants:
        print(f"\n🎯 Processing {variant}...")
        try:
            output_dir = generate_variant_observations(variant)
            output_dirs.append((variant, output_dir, True))
            print(f"✅ {variant} completed successfully!")
        except Exception as e:
            print(f"❌ Error processing {variant}: {e}")
            output_dirs.append((variant, None, False))
    
    # ===== FINAL SUMMARY =====
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - ALL VARIANTS")
    print(f"{'='*80}")
    
    total_images = 0
    successful_variants = 0
    
    for variant, output_dir, success in output_dirs:
        if success:
            print(f"\n✅ {variant.upper()}:")
            print(f"   📁 Directory: {output_dir}/")
            print(f"   📸 Images: 15 observations generated")
            print(f"   🎨 Includes: Full views, partial views, gymnasium wrapped, render_on_pos")
            total_images += 15
            successful_variants += 1
        else:
            print(f"\n❌ {variant.upper()}: Failed to generate observations")
    
    print(f"\n🎉 GENERATION COMPLETE!")
    print(f"✅ Successfully processed: {successful_variants}/{len(variants)} variants")
    print(f"📸 Total images generated: {total_images}")
    
    if successful_variants == len(variants):
        print(f"\n🌟 All variants completed successfully!")
        print(f"📦 Each variant demonstrates:")
        print(f"   • Full environment topology and room connections")
        print(f"   • Partial observation mechanics (POMDP views)")
        print(f"   • Gymnasium wrapper functionality") 
        print(f"   • Positional rendering capabilities")
        
        print(f"\n🔬 VARIANT CHARACTERISTICS:")
        print(f"• NineRooms: Classic 3x3 grid with 12 connections")
        print(f"• SpiralNineRooms: 3x3 grid with 8 spiral connections")
        print(f"• TwentyFiveRooms: Large 5x5 grid with 40 connections")
    else:
        print(f"\n⚠️  Some variants failed. Check error messages above.")

if __name__ == "__main__":
    generate_all_variants_observations()