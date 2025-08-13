#!/usr/bin/env python3
"""
Structure-only test script for DrStrategy Memory-Maze environment.
This script validates the code structure without requiring dependencies.
"""

import sys
import os
import ast
import re

def test_file_structure():
    """Test that key files exist and have expected content."""
    print("Testing file structure...")
    
    base_path = "/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy_envs/drstrategy_envs/memory-maze"
    
    required_files = [
        "memory_maze/__init__.py",
        "memory_maze/custom_task.py",
        "memory_maze/tasks.py",
        "memory_maze/maze.py",
        "memory_maze/wrappers.py",
        "setup.py",
    ]
    
    for file_path in required_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True

def test_custom_task_content():
    """Test that custom_task.py has expected classes and functions."""
    print("\nTesting custom_task.py content...")
    
    file_path = "/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy_envs/drstrategy_envs/memory-maze/memory_maze/custom_task.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for key classes and functions
        expected_items = [
            "class FourRooms7x7",
            "class FourRooms15x15", 
            "class EightRooms30x30",
            "class Maze7x7",
            "class Maze15x15",
            "class CMemoryMazeTask",
            "def C_memory_maze_fixed_layout",
            "CFOUR_ROOMS_7x7_LAYOUT",
            "CFOUR_ROOMS_15x15_LAYOUT",
        ]
        
        for item in expected_items:
            if item in content:
                print(f"✓ Found {item}")
            else:
                print(f"❌ Missing {item}")
                return False
        
        # Check file size (should be substantial)
        file_size = len(content)
        print(f"✓ File size: {file_size} characters ({file_size//1000}KB)")
        
        if file_size < 10000:  # Less than 10KB seems too small
            print("❌ File seems too small for expected content")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading custom_task.py: {e}")
        return False

def test_layout_strings():
    """Test that layout strings are properly defined."""
    print("\nTesting layout definitions...")
    
    file_path = "/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy_envs/drstrategy_envs/memory-maze/memory_maze/custom_task.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Test specific layout patterns
        layout_tests = [
            ("CFOUR_ROOMS_7x7_LAYOUT", ["*", "P", "G"]),
            ("CFOUR_ROOMS_15x15_LAYOUT", ["*", "P", "G"]),
            ("CMaze_7x7_LAYOUT", ["*", "P", "G"]),
            ("CMaze_15x15_LAYOUT", ["*", "P", "G"]),
        ]
        
        for layout_name, required_chars in layout_tests:
            if layout_name in content:
                print(f"✓ Found layout definition: {layout_name}")
                
                # Extract layout string (simplified)
                pattern = rf'{layout_name}\s*=\s*"""([^"]+)"""'
                match = re.search(pattern, content)
                if match:
                    layout_str = match.group(1)
                    for char in required_chars:
                        if char in layout_str:
                            print(f"  ✓ Contains '{char}' character")
                        else:
                            print(f"  ❌ Missing '{char}' character")
                            return False
                else:
                    print(f"  ❌ Could not extract layout string")
                    return False
            else:
                print(f"❌ Missing layout: {layout_name}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing layouts: {e}")
        return False

def test_drstrategy_integration():
    """Test DrStrategy-specific integration files."""
    print("\nTesting DrStrategy integration...")
    
    drstrategy_path = "/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy"
    
    # Test envs.py integration
    envs_file = os.path.join(drstrategy_path, "envs.py")
    
    if not os.path.exists(envs_file):
        print("❌ envs.py not found")
        return False
    
    with open(envs_file, 'r') as f:
        content = f.read()
    
    # Check for MemoryMaze class and key methods
    integration_items = [
        "class MemoryMaze:",
        "def __init__",
        "def warp_to",
        "'4x7x7' in task",
        "'mzx7x7' in task", 
        "'mzx15x15' in task",
        "from memory_maze.custom_task import",
    ]
    
    for item in integration_items:
        if item in content:
            print(f"✓ Found in envs.py: {item}")
        else:
            print(f"❌ Missing in envs.py: {item}")
            return False
    
    # Test config files
    config_file = os.path.join(drstrategy_path, "configs/memorymaze_pixels.yaml")
    if os.path.exists(config_file):
        print("✓ memorymaze_pixels.yaml config exists")
    else:
        print("❌ memorymaze_pixels.yaml config missing")
        return False
    
    # Test runner scripts
    runner_files = [
        "runner/run_drstrategy_mzx7x7.sh",
        "runner/run_drstrategy_mzx15x15.sh",
    ]
    
    for runner_file in runner_files:
        full_path = os.path.join(drstrategy_path, runner_file)
        if os.path.exists(full_path):
            print(f"✓ Runner script exists: {runner_file}")
        else:
            print(f"❌ Runner script missing: {runner_file}")
            return False
    
    return True

def test_data_generation():
    """Test data generation scripts."""
    print("\nTesting data generation scripts...")
    
    data_path = "/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy_envs/drstrategy_envs/memory-maze/data"
    
    if not os.path.exists(data_path):
        print("❌ Data directory not found")
        return False
    
    # Check for data generation scripts
    data_files = os.listdir(data_path)
    script_types = {
        'single_room': 0,
        'two_rooms': 0,
        'four_rooms': 0,
        'ten_rooms': 0,
        'twenty_rooms': 0,
    }
    
    for file_name in data_files:
        if file_name.endswith('.py'):
            for script_type in script_types:
                if script_type in file_name:
                    script_types[script_type] += 1
    
    for script_type, count in script_types.items():
        if count > 0:
            print(f"✓ Found {count} {script_type} scripts")
        else:
            print(f"❌ No {script_type} scripts found")
    
    total_scripts = sum(script_types.values())
    print(f"✓ Total data generation scripts: {total_scripts}")
    
    return total_scripts > 10  # Should have many scripts

def main():
    """Run all structure tests."""
    print("DrStrategy Memory-Maze Structure Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Custom Task Content", test_custom_task_content),
        ("Layout Definitions", test_layout_strings),
        ("DrStrategy Integration", test_drstrategy_integration),
        ("Data Generation", test_data_generation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 30)
        
        success = test_func()
        if success:
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All structure tests passed!")
        print("The DrStrategy memory-maze implementation appears to be complete and well-structured.")
        print("\nKey findings:")
        print("- Custom layout classes implemented (FourRooms, Maze variants)")
        print("- Multiple maze configurations (7x7, 15x15, 30x30)")
        print("- Integration with DrStrategy framework")
        print("- Comprehensive data generation pipeline")
        print("- Training and evaluation scripts")
        print("\nTo run the actual environment, you would need:")
        print("- dm_control and MuJoCo dependencies")
        print("- labmaze package (requires Bazel build)")
        return 0
    else:
        print("❌ Some structure tests failed.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)