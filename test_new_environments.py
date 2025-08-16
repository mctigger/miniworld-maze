#!/usr/bin/env python3
"""
Test the SpiralNineRooms and TwentyFiveRooms implementations.
"""

import sys
import numpy as np

# Add paths
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences')

from nine_rooms_pure_gymnasium_env.miniworld_gymnasium.envs.roomnav import SpiralNineRooms, TwentyFiveRooms


def test_environment(env_class, env_name):
    """Test a specific environment implementation."""
    print(f"\n=== Testing {env_name} ===")
    
    try:
        # Create environment
        env = env_class(room_size=15, door_size=2.5, obs_level=1, continuous=False)
        print(f"✅ Environment created: {type(env).__name__}")
        
        # Check basic properties
        print(f"Action space: {env.action_space}")
        print(f"Max episode steps: {env.max_episode_steps}")
        print(f"Room size: {env.room_size}")
        print(f"Door size: {env.door_size}")
        print(f"Connections: {len(env.connections)} room connections")
        print(f"Textures: {len(env.textures)} room textures")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Reset successful")
        print(f"Observation shape: {obs.shape}")
        print(f"Observation dtype: {obs.dtype}")
        print(f"Observation range: [{obs.min()}, {obs.max()}]")
        
        # Test a few steps
        total_reward = 0
        for step in range(5):
            action = np.random.randint(0, 3)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            print(f"Step {step + 1}: action={action}, reward={reward}")
            if done:
                print(f"Episode terminated at step {step + 1}")
                break
        
        print(f"✅ Steps completed. Total reward: {total_reward}")
        
        # Test world generation (check if agent is placed correctly)
        agent_pos = env.agent.pos
        print(f"Agent position: ({agent_pos[0]:.2f}, {agent_pos[2]:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {env_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_connections():
    """Compare connections between environments."""
    print(f"\n=== Comparing Environment Connections ===")
    
    # Create environments
    spiral_env = SpiralNineRooms()
    twenty_five_env = TwentyFiveRooms()
    
    print(f"SpiralNineRooms connections: {spiral_env.connections}")
    print(f"Number of connections: {len(spiral_env.connections)}")
    
    print(f"\nTwentyFiveRooms connections (first 10): {twenty_five_env.connections[:10]}...")
    print(f"Number of connections: {len(twenty_five_env.connections)}")
    
    # Verify spiral pattern in SpiralNineRooms
    expected_spiral = [(0,1), (0,3), (1,2), (2,5), (3,6), (4,5), (6,7), (7,8)]
    if spiral_env.connections == expected_spiral:
        print("✅ SpiralNineRooms has correct spiral connection pattern")
    else:
        print("⚠️  SpiralNineRooms connection pattern differs from expected")
        
    # Verify 25-room grid connectivity
    if len(twenty_five_env.connections) == 40:  # 5x5 grid should have 40 connections (20 horizontal + 20 vertical)
        print("✅ TwentyFiveRooms has correct number of connections for 5x5 grid")
    else:
        print(f"⚠️  TwentyFiveRooms has {len(twenty_five_env.connections)} connections (expected 40)")


def main():
    """Main test function."""
    print("TESTING NEW ENVIRONMENT VARIANTS")
    print("=" * 50)
    
    # Test environments
    results = []
    
    # Test SpiralNineRooms
    results.append(test_environment(SpiralNineRooms, "SpiralNineRooms"))
    
    # Test TwentyFiveRooms
    results.append(test_environment(TwentyFiveRooms, "TwentyFiveRooms"))
    
    # Compare connections
    compare_connections()
    
    # Summary
    print(f"\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if all(results):
        print("🎉 ALL TESTS PASSED!")
        print("✅ SpiralNineRooms: Working correctly")
        print("✅ TwentyFiveRooms: Working correctly")
        print("✅ Both environments are ready for use")
    else:
        print("⚠️  SOME TESTS FAILED")
        for i, (env_name, result) in enumerate([("SpiralNineRooms", results[0]), ("TwentyFiveRooms", results[1])]):
            if result:
                print(f"✅ {env_name}: PASSED")
            else:
                print(f"❌ {env_name}: FAILED")


if __name__ == "__main__":
    main()