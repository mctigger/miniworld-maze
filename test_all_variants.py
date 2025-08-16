#!/usr/bin/env python3
"""
Test script demonstrating all Nine Rooms environment variants.
"""

import sys
import numpy as np
from PIL import Image

# Add path for our implementations
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences')

from nine_rooms_factory import create_nine_rooms_env

def test_environment_variant(variant_name, num_steps=5):
    """Test a specific environment variant and save sample observations."""
    print(f"\n{'='*50}")
    print(f"TESTING {variant_name.upper()}")
    print(f"{'='*50}")
    
    # Create environment
    env = create_nine_rooms_env(variant=variant_name, size=64)
    
    # Environment info
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Get base environment for connection info
    base_env = env._env
    while hasattr(base_env, 'env') or hasattr(base_env, '_env'):
        if hasattr(base_env, 'env'):
            base_env = base_env.env
        elif hasattr(base_env, '_env'):
            base_env = base_env._env
        else:
            break
    
    print(f"Room connections: {len(base_env.connections)}")
    print(f"Room textures: {len(base_env.textures)}")
    
    # Reset and get initial observation
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    
    # Save initial observation
    obs_hwc = np.transpose(obs, (1, 2, 0))  # Convert CHW to HWC
    Image.fromarray(obs_hwc).save(f'{variant_name.lower()}_initial.png')
    print(f"✅ Saved: {variant_name.lower()}_initial.png")
    
    # Test movement
    print(f"\nTesting {num_steps} random actions:")
    total_reward = 0
    for step in range(num_steps):
        action = np.random.randint(0, 3)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        action_names = {0: "turn_left", 1: "turn_right", 2: "move_forward"}
        print(f"  Step {step+1}: {action_names[action]} -> reward={reward}")
        
        if terminated or truncated:
            print(f"  Episode ended at step {step+1}")
            break
    
    print(f"Total reward: {total_reward}")
    
    # Save final observation
    obs_hwc = np.transpose(obs, (1, 2, 0))
    Image.fromarray(obs_hwc).save(f'{variant_name.lower()}_final.png')
    print(f"✅ Saved: {variant_name.lower()}_final.png")
    
    env.close()
    return True

def main():
    """Test all environment variants."""
    print("TESTING ALL NINE ROOMS ENVIRONMENT VARIANTS")
    print("=" * 70)
    
    variants = ["NineRooms", "SpiralNineRooms", "TwentyFiveRooms"]
    results = []
    
    for variant in variants:
        try:
            success = test_environment_variant(variant)
            results.append((variant, success))
        except Exception as e:
            print(f"❌ Error testing {variant}: {e}")
            results.append((variant, False))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    all_passed = True
    for variant, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{variant:20s}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 ALL VARIANTS WORKING CORRECTLY!")
        print(f"📸 Generated observation samples:")
        for variant in variants:
            if any(r[0] == variant and r[1] for r in results):
                print(f"   • {variant.lower()}_initial.png - Initial observation")
                print(f"   • {variant.lower()}_final.png - Final observation") 
    else:
        print(f"\n⚠️  Some variants failed to work correctly.")
    
    print(f"\nEnvironment Characteristics:")
    print(f"• NineRooms: 3x3 grid of rooms (classic layout)")
    print(f"• SpiralNineRooms: 3x3 grid with spiral connection pattern")
    print(f"• TwentyFiveRooms: 5x5 grid of rooms (larger environment)")

if __name__ == "__main__":
    main()