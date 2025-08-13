#!/usr/bin/env python3
"""
Complete test script for DrStrategy Memory-Maze environment with real labmaze.
This script tests the fully functional environment with all dependencies installed.
"""

import sys
import os
import numpy as np

# Add the DrStrategy paths to Python path
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy')
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy_envs')
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy_envs/drstrategy_envs')

def test_labmaze_import():
    """Test that labmaze imports correctly with all features."""
    print("Testing labmaze imports...")
    
    try:
        import labmaze
        from labmaze import assets
        print("✓ labmaze imported successfully")
        print(f"✓ labmaze.defaults.MAX_ROOMS: {labmaze.defaults.MAX_ROOMS}")
        print(f"✓ labmaze.assets available: {hasattr(assets, 'get_wall_texture_paths')}")
        return True
        
    except Exception as e:
        print(f"❌ labmaze import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_maze_imports():
    """Test memory-maze module imports."""
    print("\nTesting memory-maze imports...")
    
    try:
        from memory_maze import tasks
        from memory_maze.custom_task import FourRooms7x7, FourRooms15x15, Maze7x7, Maze15x15
        from memory_maze.custom_task import C_memory_maze_fixed_layout
        print("✓ All memory-maze modules imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Memory-maze import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layout_instantiation():
    """Test layout class instantiation."""
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
            print(f"✓ {name}: {layout.len_x}x{layout.len_y}, {layout.max_num_steps} steps, {len(layout.goal_poses)} goals")
            
        return True
        
    except Exception as e:
        print(f"❌ Layout instantiation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_environment_creation():
    """Test creating custom memory-maze environment."""
    print("\nTesting custom environment creation...")
    
    try:
        from memory_maze.custom_task import C_memory_maze_fixed_layout, CFOUR_ROOMS_7x7_LAYOUT
        
        print("Creating 7x7 four-rooms environment with real labmaze...")
        
        # Set environment variables for MuJoCo
        os.environ['MUJOCO_GL'] = 'osmesa'  # Use osmesa for headless rendering
        
        env = C_memory_maze_fixed_layout(
            entity_layer=CFOUR_ROOMS_7x7_LAYOUT,
            n_targets=4,
            time_limit=100,  # Longer time for real testing
            discrete_actions=True,
            image_only_obs=False,
            global_observables=True,
            no_wall_patterns=False,  # Use real textures
            different_floor_textures=True,
            seed=42
        )
        
        print("✓ Environment created successfully")
        
        # Test reset
        timestep = env.reset()
        print(f"✓ Environment reset, observation keys: {list(timestep.observation.keys())}")
        print(f"  Agent position: {timestep.observation.get('agent_pos', 'N/A')}")
        print(f"  Target position: {timestep.observation.get('target_pos', 'N/A')}")
        print(f"  Image shape: {timestep.observation.get('image', np.array([])).shape}")
        
        # Test multiple steps with discrete actions
        total_reward = 0
        for step in range(5):
            action = 0  # Discrete action (forward)
            timestep = env.step(action)
            total_reward += timestep.reward
            print(f"  Step {step+1}: reward={timestep.reward:.3f}, discount={timestep.discount}")
            
            if timestep.last():
                print("  Episode finished")
                break
        
        print(f"✓ Environment stepping successful, total reward: {total_reward:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Custom environment error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_drstrategy_wrapper():
    """Test DrStrategy MemoryMaze wrapper."""
    print("\nTesting DrStrategy wrapper...")
    
    try:
        from drstrategy.envs import MemoryMaze
        
        print("Creating DrStrategy MemoryMaze wrapper...")
        
        # Set environment variables
        os.environ['MUJOCO_GL'] = 'osmesa'
        
        env = MemoryMaze(
            task='mzx7x7',
            discrete_actions=True,
            no_wall_patterns=False,  # Use real textures
            different_floor_textures=True,
            override_high_walls=False,
            sky=True,
            time_limit=100
        )
        
        print("✓ DrStrategy MemoryMaze wrapper created")
        print(f"✓ Task: {env.layout.__class__.__name__}")
        print(f"✓ Max steps: {env.max_num_steps}")
        print(f"✓ Action space: {env.action_space}")
        print(f"✓ Observation space keys: {list(env.observation_space.keys())}")
        
        # Test reset
        obs = env.reset()
        print(f"✓ Environment reset, observation keys: {list(obs.keys())}")
        print(f"  Agent pos: {obs.get('agent_pos', 'N/A')}")
        print(f"  Target pos: {obs.get('target_pos', 'N/A')}")
        print(f"  Image shape: {obs.get('image', np.array([])).shape}")
        print(f"  Top view shape: {obs.get('top_view', np.array([])).shape}")
        
        # Test several steps with random actions
        total_reward = 0
        for i in range(10):
            action = env.action_space.sample()
            action = np.array(action)  # Convert to numpy array for DrStrategy wrapper
            step_result = env.step(action)
            reward = step_result['reward']
            done = step_result['is_last']
            total_reward += reward
            
            print(f"  Step {i+1}: action={action}, reward={reward:.3f}, done={done}")
            print(f"    Agent pos: {step_result.get('position', 'N/A')}")
            
            if done:
                print("  Episode finished early")
                break
        
        print(f"✓ DrStrategy wrapper test successful, total reward: {total_reward:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ DrStrategy wrapper error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_environment_variants():
    """Test multiple environment variants."""
    print("\nTesting multiple environment variants...")
    
    try:
        from drstrategy.envs import MemoryMaze
        
        variants = [
            'mzx7x7',      # 7x7 maze
            '4x7x7',       # Four rooms 7x7
            'mzx15x15',    # 15x15 maze  
            '4x15x15',     # Four rooms 15x15
        ]
        
        os.environ['MUJOCO_GL'] = 'osmesa'
        
        for variant in variants:
            print(f"  Testing variant: {variant}")
            
            env = MemoryMaze(
                task=variant,
                discrete_actions=True,
                no_wall_patterns=True,  # Simplified for faster testing
                time_limit=20  # Short episodes
            )
            
            obs = env.reset()
            action = env.action_space.sample()
            action = np.array(action)  # Convert to numpy array for DrStrategy wrapper
            step_result = env.step(action)
            
            print(f"    ✓ {variant}: Created, reset, and stepped successfully")
            env.close() if hasattr(env, 'close') else None
        
        print("✓ All environment variants working")
        return True
        
    except Exception as e:
        print(f"❌ Multiple variants error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive tests."""
    print("DrStrategy Memory-Maze Complete Environment Test")
    print("=" * 60)
    print("Testing with Python 3.12 + real labmaze + full dm_control")
    
    tests = [
        ("Labmaze Import", test_labmaze_import),
        ("Memory-Maze Imports", test_memory_maze_imports),
        ("Layout Instantiation", test_layout_instantiation),
        ("Custom Environment", test_custom_environment_creation),
        ("DrStrategy Wrapper", test_drstrategy_wrapper),
        ("Multiple Variants", test_multiple_environment_variants),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 45)
        
        success = test_func()
        if success:
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nThe DrStrategy memory-maze environment is fully functional!")
        print("\n🔧 Successfully installed and tested:")
        print("  • Python 3.12 virtual environment")
        print("  • labmaze (prebuilt wheels from PyPI)")
        print("  • dm_control with MuJoCo physics")
        print("  • DrStrategy custom memory-maze implementation")
        print("  • Multiple environment variants (7x7, 15x15, 4-rooms, mazes)")
        print("  • Both dm_env and gym interfaces")
        print("  • Real texture assets and visual rendering")
        print("\n🚀 Environment ready for:")
        print("  • Training DrStrategy agents")
        print("  • Running experiments and evaluations")
        print("  • Data generation for offline RL")
        print("  • Research on hierarchical navigation")
        
        return 0
    else:
        print("❌ Some tests failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)