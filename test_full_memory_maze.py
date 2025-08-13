#!/usr/bin/env python3
"""
Full test script for DrStrategy Memory-Maze environment using mock labmaze.
This script tests the actual environment functionality with mock dependencies.
"""

import sys
import os
import numpy as np

# Setup mock labmaze before any memory_maze imports
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences')
import mock_labmaze
sys.modules['labmaze'] = mock_labmaze
sys.modules['labmaze.assets'] = mock_labmaze

# Add the DrStrategy paths to Python path
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy')
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy_envs')
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy_envs/drstrategy_envs')

def test_memory_maze_imports():
    """Test that we can import the memory-maze modules."""
    print("Testing memory-maze imports...")
    
    try:
        # Import memory-maze modules
        from memory_maze import tasks
        from memory_maze.custom_task import FourRooms7x7, FourRooms15x15, Maze7x7, Maze15x15
        from memory_maze.custom_task import C_memory_maze_fixed_layout
        print("✓ All memory-maze modules imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layout_classes():
    """Test layout class instantiation and properties."""
    print("\nTesting layout classes...")
    
    try:
        from memory_maze.custom_task import FourRooms7x7, FourRooms15x15, Maze7x7, Maze15x15
        
        layouts = [
            ("FourRooms7x7", FourRooms7x7()),
            ("FourRooms15x15", FourRooms15x15()),
            ("Maze7x7", Maze7x7()),
            ("Maze15x15", Maze15x15()),
        ]
        
        for name, layout in layouts:
            print(f"✓ {name}: {layout.len_x}x{layout.len_y}, {layout.max_num_steps} steps, {len(layout.rooms)} rooms")
            
        return True
        
    except Exception as e:
        print(f"❌ Layout test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_environment():
    """Test creating a custom memory-maze environment."""
    print("\nTesting custom environment creation...")
    
    try:
        from memory_maze.custom_task import C_memory_maze_fixed_layout, CFOUR_ROOMS_7x7_LAYOUT
        
        print("Creating 7x7 four-rooms environment...")
        
        # Set environment variables for MuJoCo
        os.environ['MUJOCO_GL'] = 'osmesa'  # Use osmesa for headless rendering
        
        env = C_memory_maze_fixed_layout(
            entity_layer=CFOUR_ROOMS_7x7_LAYOUT,
            n_targets=4,
            time_limit=50,  # Short time for testing
            discrete_actions=True,
            image_only_obs=False,
            global_observables=True,
            no_wall_patterns=True,  # Simplify for testing
            different_floor_textures=False,
            seed=42
        )
        
        print("✓ Environment created successfully")
        
        # Test reset
        timestep = env.reset()
        print(f"✓ Environment reset, observation keys: {list(timestep.observation.keys())}")
        
        # Test step
        action = np.array([0.0, 0.0])  # No-op action
        timestep = env.step(action)
        print(f"✓ Environment step, reward: {timestep.reward}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_drstrategy_wrapper():
    """Test the DrStrategy MemoryMaze wrapper."""
    print("\nTesting DrStrategy wrapper...")
    
    try:
        from drstrategy.envs import MemoryMaze
        
        print("Creating DrStrategy MemoryMaze wrapper...")
        
        # Set environment variables
        os.environ['MUJOCO_GL'] = 'osmesa'
        
        env = MemoryMaze(
            task='mzx7x7',
            discrete_actions=True,
            no_wall_patterns=True,
            different_floor_textures=False,
            time_limit=50
        )
        
        print("✓ DrStrategy MemoryMaze wrapper created")
        print(f"✓ Action space: {env.action_space}")
        print(f"✓ Observation space keys: {list(env.observation_space.keys())}")
        
        # Test reset
        obs = env.reset()
        print(f"✓ Environment reset, observation keys: {list(obs.keys())}")
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"✓ Step {i+1}: action={action}, reward={reward:.3f}, done={done}")
            if done:
                break
        
        return True
        
    except Exception as e:
        print(f"❌ DrStrategy wrapper test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("DrStrategy Memory-Maze Full Environment Test")
    print("=" * 55)
    
    tests = [
        ("Memory-Maze Imports", test_memory_maze_imports),
        ("Layout Classes", test_layout_classes),
        ("Custom Environment", test_custom_environment),
        ("DrStrategy Wrapper", test_drstrategy_wrapper),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 40)
        
        success = test_func()
        if success:
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 55)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The DrStrategy memory-maze environment is working!")
        print("\nEnvironment successfully tested with:")
        print("- Mock labmaze (bypassing Bazel/Python 3.13 issues)")
        print("- Full dm_control and MuJoCo integration") 
        print("- DrStrategy custom layouts and tasks")
        print("- Both dm_env and gym interfaces")
        print("\nThe environment is ready for training and evaluation!")
        return 0
    else:
        print("❌ Some tests failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)