#!/usr/bin/env python3
"""
DrStrategy Memory-Maze Environment Launcher

A clean launcher script to run different memory-maze environment variants
with real labmaze assets and DrStrategy integration.

Usage:
    python run_memory_maze.py --env mzx7x7 --steps 100
    python run_memory_maze.py --env 4x15x15 --steps 50 --render
"""

import sys
import os
import argparse
import time
import numpy as np

# Add DrStrategy paths
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy')
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy_envs')
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy_envs/drstrategy_envs')

def setup_environment(headless=False):
    """Set up the Python environment paths."""
    os.environ['PYTHONPATH'] = ':'.join([
        '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy',
        '/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy',
        os.environ.get('PYTHONPATH', '')
    ])
    
    if headless:
        # Set environment variables for headless operation
        os.environ['MUJOCO_GL'] = 'osmesa'
        os.environ['DISPLAY'] = ''

def create_environment(env_name, time_limit=100, enable_rendering=False):
    """
    Create a DrStrategy MemoryMaze environment.
    
    Args:
        env_name: Environment variant (mzx7x7, 4x7x7, mzx15x15, 4x15x15, 8x30x30)
        time_limit: Maximum steps per episode
        enable_rendering: Whether to enable visual features
        
    Returns:
        Configured DrStrategy MemoryMaze environment
    """
    # Import using the PYTHONPATH approach that worked in the test
    from drstrategy.envs import MemoryMaze
    
    # Environment configuration based on variant
    config = {
        'task': env_name,
        'discrete_actions': True,
        'time_limit': time_limit,
        'no_wall_patterns': not enable_rendering,  # Enable patterns for visual appeal
        'different_floor_textures': enable_rendering,
        'override_high_walls': False,
        'sky': enable_rendering,
    }
    
    print(f"Creating MemoryMaze environment: {env_name}")
    print(f"Configuration: {config}")
    
    env = MemoryMaze(**config)
    return env

def print_environment_info(env):
    """Print detailed information about the environment."""
    print("\n" + "="*60)
    print("ENVIRONMENT INFORMATION")
    print("="*60)
    
    print(f"Task: {getattr(env._env, 'task', 'Unknown')}")
    print(f"Max steps: {getattr(env, 'max_num_steps', 'Unknown')}")
    print(f"Action space: {env.action_space}")
    
    # Get observation space info
    obs_space = env._env.observation_spec()
    print(f"Observation space keys: {list(obs_space.keys())}")
    
    for key, spec in obs_space.items():
        if hasattr(spec, 'shape'):
            print(f"  {key}: {spec.shape} ({spec.dtype})")
        else:
            print(f"  {key}: {spec}")

def run_environment(env, num_steps=100, interactive=False):
    """
    Run the environment for a specified number of steps.
    
    Args:
        env: The environment instance
        num_steps: Number of steps to run
        interactive: Whether to wait for user input between steps
    """
    print("\n" + "="*60)
    print("RUNNING ENVIRONMENT")
    print("="*60)
    
    # Reset environment
    obs = env.reset()
    print(f"Environment reset. Initial observation keys: {list(obs.keys())}")
    
    if 'position' in obs:
        print(f"Initial agent position: {obs['position']}")
    
    total_reward = 0
    step_count = 0
    
    print(f"\nRunning for {num_steps} steps...")
    if interactive:
        print("Press Enter after each step, or 'q' to quit early.")
    
    for step in range(num_steps):
        # Sample random action
        action = env.action_space.sample()
        action = np.array(action)  # Convert to numpy array for DrStrategy
        
        # Take step
        step_result = env.step(action)
        
        reward = step_result['reward']
        done = step_result['is_last']
        total_reward += reward
        step_count += 1
        
        # Print step information
        position = step_result.get('position', [0, 0, 0])
        print(f"Step {step_count:3d}: action={action}, reward={reward:.3f}, "
              f"pos=[{position[0]:.2f}, {position[2]:.2f}], done={done}")
        
        if interactive:
            user_input = input()
            if user_input.lower() == 'q':
                print("Exiting early...")
                break
        
        if done:
            print(f"Episode finished at step {step_count}")
            break
        
        # Small delay for better visualization
        if not interactive:
            time.sleep(0.1)
    
    print(f"\nEpisode completed!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Average reward: {total_reward/step_count:.3f}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Launch DrStrategy Memory-Maze Environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variants:
  mzx7x7    - 7x7 maze navigation
  4x7x7     - 7x7 four-room navigation  
  mzx15x15  - 15x15 maze navigation
  4x15x15   - 15x15 four-room navigation
  8x30x30   - 30x30 eight-room navigation

Examples:
  python run_memory_maze.py --env mzx7x7 --steps 100
  python run_memory_maze.py --env 4x15x15 --steps 50 --render --interactive
        """
    )
    
    parser.add_argument('--env', type=str, required=True,
                        choices=['mzx7x7', '4x7x7', 'mzx15x15', '4x15x15', '8x30x30'],
                        help='Environment variant to run')
    
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of steps to run (default: 100)')
    
    parser.add_argument('--time-limit', type=int, default=100,
                        help='Episode time limit (default: 100)')
    
    parser.add_argument('--render', action='store_true',
                        help='Enable visual features (wall patterns, textures, etc.)')
    
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode - wait for Enter between steps')
    
    parser.add_argument('--info', action='store_true',
                        help='Show detailed environment information only')
    
    parser.add_argument('--headless', action='store_true',
                        help='Run without graphics (for systems without display)')
    
    args = parser.parse_args()
    
    print("DrStrategy Memory-Maze Environment Launcher")
    print("="*60)
    
    if args.info:
        # Show basic info without creating environment
        print("\n" + "="*60)
        print("ENVIRONMENT INFORMATION")
        print("="*60)
        
        env_info = {
            'mzx7x7': {'type': '7x7 maze', 'max_steps': 500, 'layout': 'Maze7x7'},
            '4x7x7': {'type': '7x7 four-room', 'max_steps': 500, 'layout': 'FourRooms7x7'},
            'mzx15x15': {'type': '15x15 maze', 'max_steps': 1000, 'layout': 'Maze15x15'},
            '4x15x15': {'type': '15x15 four-room', 'max_steps': 1000, 'layout': 'FourRooms15x15'},
            '8x30x30': {'type': '30x30 eight-room', 'max_steps': 2000, 'layout': 'EightRooms30x30'},
        }
        
        info = env_info.get(args.env, {'type': 'Unknown', 'max_steps': 'Unknown', 'layout': 'Unknown'})
        print(f"Environment: {args.env}")
        print(f"Type: {info['type']}")
        print(f"Layout: {info['layout']}")
        print(f"Default max steps: {info['max_steps']}")
        print(f"Your time limit: {args.time_limit}")
        print(f"Action space: Discrete(6) - move forward/backward, turn left/right, strafe left/right")
        print(f"Observation space: RGB images (64x64x3), agent position, target info, maze layout")
        
        print("\n✅ Environment information displayed successfully!")
        
    else:
        # Setup environment and create it
        setup_environment(headless=args.headless)
        
        try:
            # Create environment
            env = create_environment(
                env_name=args.env,
                time_limit=args.time_limit,
                enable_rendering=args.render and not args.headless
            )
            
            # Print environment information
            print_environment_info(env)
            
            # Run the environment
            run_environment(
                env=env,
                num_steps=args.steps,
                interactive=args.interactive
            )
            
            print("\n🎉 Environment run completed successfully!")
        
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Error: {error_msg}")
            
            if 'gladLoadGL' in error_msg or 'glGetError' in error_msg or 'DISPLAY' in error_msg:
                print("\n💡 Graphics Error Detected:")
                print("This appears to be a graphics/OpenGL issue. Try one of these solutions:")
                print("1. Use --headless flag (though this environment requires graphics)")
                print("2. Run on a system with proper OpenGL support")
                print("3. Use X11 forwarding if running remotely: ssh -X")
                print("4. Install mesa-utils: sudo apt-get install mesa-utils")
                print("\nNote: The test script worked because it didn't render graphics.")
                print("The memory-maze environment requires OpenGL for visual observations.")
            
            print("\nUse --info flag to see what environments are available without initializing them.")
            
            return 1
    
    return 0

if __name__ == '__main__':
    exit(main())