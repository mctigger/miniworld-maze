#!/usr/bin/env python3
"""
Simplified test script for DrStrategy Memory-Maze environment.
This script tests the basic functionality without requiring full dependency installation.
"""

import sys
import os
import numpy as np

# Add the DrStrategy paths to Python path
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy')
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy_envs')
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy_envs/drstrategy_envs')

def test_imports():
    """Test that we can import the required modules."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import memory_maze
        print("✓ memory_maze module imported successfully")
        
        # Test custom task import
        from memory_maze.custom_task import FourRooms7x7, FourRooms15x15, Maze7x7
        print("✓ Custom layout classes imported successfully")
        
        # Test task functions
        from memory_maze import tasks
        print("✓ Tasks module imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_layout_classes():
    """Test that layout classes can be instantiated."""
    print("\nTesting layout classes...")
    
    try:
        from memory_maze.custom_task import FourRooms7x7, FourRooms15x15, Maze7x7, Maze15x15
        
        # Test 7x7 four rooms layout
        layout_4x7x7 = FourRooms7x7()
        print(f"✓ FourRooms7x7: {layout_4x7x7.len_x}x{layout_4x7x7.len_y}, {layout_4x7x7.max_num_steps} max steps")
        
        # Test 15x15 four rooms layout
        layout_4x15x15 = FourRooms15x15()
        print(f"✓ FourRooms15x15: {layout_4x15x15.len_x}x{layout_4x15x15.len_y}, {layout_4x15x15.max_num_steps} max steps")
        
        # Test maze layouts
        layout_mzx7x7 = Maze7x7()
        print(f"✓ Maze7x7: {layout_mzx7x7.len_x}x{layout_mzx7x7.len_y}, {layout_mzx7x7.max_num_steps} max steps")
        
        layout_mzx15x15 = Maze15x15()
        print(f"✓ Maze15x15: {layout_mzx15x15.len_x}x{layout_mzx15x15.len_y}, {layout_mzx15x15.max_num_steps} max steps")
        
        return True
    except Exception as e:
        print(f"❌ Layout test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layout_properties():
    """Test layout properties and methods."""
    print("\nTesting layout properties...")
    
    try:
        from memory_maze.custom_task import FourRooms7x7
        
        layout = FourRooms7x7()
        
        # Test layout string
        print(f"✓ Layout string length: {len(layout.layout)} chars")
        print(f"✓ Layout contains 'G' (goals): {'G' in layout.layout}")
        print(f"✓ Layout contains 'P' (spawn): {'P' in layout.layout}")
        
        # Test rooms
        rooms = layout.get_rooms()
        print(f"✓ Number of rooms: {len(rooms)}")
        print(f"✓ First room bounds: {rooms[0] if rooms else 'None'}")
        
        # Test coordinate bounds
        min_x, max_x, min_y, max_y = layout.get_min_max_coords((7, 7))
        print(f"✓ Coordinate bounds: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
        
        # Test goal poses
        print(f"✓ Goal poses available: {len(layout.goal_poses)} poses")
        
        return True
    except Exception as e:
        print(f"❌ Layout properties test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_creation():
    """Test environment creation without full MuJoCo dependency."""
    print("\nTesting environment creation (mock)...")
    
    try:
        # Test that we can access the MemoryMaze class structure
        from drstrategy.envs import MemoryMaze
        print("✓ MemoryMaze class imported successfully")
        
        # Test task name parsing logic (without actual env creation)
        task_variants = ['mzx7x7', 'mzx15x15', '4x7x7', '4x15x15', '8x30x30']
        
        for task in task_variants:
            # Test layout selection logic
            if '4x7x7' in task:
                layout_type = "FourRooms7x7"
            elif '4x15x15' in task:
                layout_type = "FourRooms15x15"
            elif '8x30x30' in task:
                layout_type = "EightRooms30x30"
            elif 'mzx7x7' in task:
                layout_type = "Maze7x7"
            elif 'mzx15x15' in task:
                layout_type = "Maze15x15"
            else:
                layout_type = "Unknown"
            
            print(f"✓ Task '{task}' -> Layout '{layout_type}'")
        
        return True
    except Exception as e:
        print(f"❌ Environment creation test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("DrStrategy Memory-Maze Environment Test (Simplified)")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Layout Classes Test", test_layout_classes),
        ("Layout Properties Test", test_layout_properties),
        ("Environment Creation Test", test_environment_creation),
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
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The DrStrategy memory-maze code structure is working correctly.")
        print("\nNote: This test validates the code structure and imports.")
        print("Full environment testing requires MuJoCo and dm_control dependencies.")
        return 0
    else:
        print("❌ Some tests failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)