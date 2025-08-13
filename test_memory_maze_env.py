#!/usr/bin/env python3
"""
Test script for DrStrategy Memory-Maze environment.
This script creates different variants of the memory-maze environment 
and runs a few test actions to verify functionality.
"""

import sys
import os
import numpy as np

# Add the DrStrategy paths to Python path
sys.path.append('/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy')
sys.path.append('/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy_envs')

def test_environment(task_name, num_steps=10):
    """Test a specific memory-maze environment variant."""
    print(f"\n=== Testing {task_name} ===")
    
    try:
        # Set environment variables for MuJoCo
        os.environ['MUJOCO_GL'] = 'osmesa'  # Use osmesa for headless rendering
        
        # Import the DrStrategy environment wrapper
        from drstrategy.envs import MemoryMaze
        
        # Create environment
        print(f"Creating environment: {task_name}")
        env = MemoryMaze(
            task=task_name,
            discrete_actions=True,  # Use discrete actions for easier testing
            no_wall_patterns=False,
            different_floor_textures=True,
            override_high_walls=False,
            sky=True,
            time_limit=100  # Short time limit for testing
        )
        
        print("Environment created successfully!")
        
        # Print environment info
        print(f"Action space: {env.action_space}")
        print(f"Observation space keys: {list(env.observation_space.keys()) if hasattr(env.observation_space, 'keys') else 'N/A'}")
        print(f"Max episode steps: {env.max_num_steps}")
        print(f"Discrete actions: {env.discrete_actions}")
        
        # Reset environment
        print("Resetting environment...")
        obs = env.reset()
        print(f"Initial observation keys: {list(obs.keys())}")
        print(f"Agent position: {obs.get('agent_pos', 'N/A')}")
        print(f"Agent direction: {obs.get('agent_dir', 'N/A')}")
        print(f"Target position: {obs.get('target_pos', 'N/A')}")
        print(f"Target color: {obs.get('target_color', 'N/A')}")
        print(f"Image shape: {obs.get('image', np.array([])).shape}")
        
        # Run a few test actions
        print(f"\nRunning {num_steps} test actions...")
        total_reward = 0
        
        for step in range(num_steps):
            # Sample random action
            action = env.action_space.sample()
            
            # Take step
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"Step {step+1}: action={action}, reward={reward:.3f}, done={done}")
            print(f"  Agent pos: {obs.get('agent_pos', 'N/A')}")
            
            if done:
                print("Episode finished early!")
                break
        
        print(f"Total reward: {total_reward:.3f}")
        print(f"Test completed successfully for {task_name}!")
        
        # Clean up
        env.close() if hasattr(env, 'close') else None
        return True
        
    except Exception as e:
        print(f"Error testing {task_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test multiple environment variants."""
    print("Testing DrStrategy Memory-Maze Environments")
    print("=" * 50)
    
    # Test different environment variants
    test_variants = [
        'mzx7x7',      # 7x7 maze
        'mzx15x15',    # 15x15 maze  
        '4x7x7',       # Four rooms 7x7
        '4x15x15',     # Four rooms 15x15
    ]
    
    success_count = 0
    total_tests = len(test_variants)
    
    for variant in test_variants:
        success = test_environment(variant, num_steps=5)
        if success:
            success_count += 1
        print("-" * 30)
    
    print(f"\nTest Results: {success_count}/{total_tests} environments working")
    
    if success_count == total_tests:
        print("✅ All tests passed! DrStrategy memory-maze environments are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)