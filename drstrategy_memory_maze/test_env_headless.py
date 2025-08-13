#!/usr/bin/env python3
"""Headless testing script for DrStrategy Memory Maze environments.

This script tests environments without requiring a display, suitable for 
automated testing and CI/CD environments.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Any

import numpy as np
import gymnasium as gym

import drstrategy_memory_maze


def test_environment(env_id: str, max_steps: int = 200) -> Dict[str, Any]:
    """Test an environment and collect statistics.
    
    Args:
        env_id: Gymnasium environment ID
        max_steps: Maximum steps to test
        
    Returns:
        Dictionary containing test results and statistics
    """
    print(f"Testing environment: {env_id}")
    
    # Create environment
    env = gym.make(env_id)
    
    # Collect statistics
    stats = {
        'env_id': env_id,
        'action_space': str(env.action_space),
        'observation_space': str(env.observation_space),
        'episodes': 0,
        'total_steps': 0,
        'total_reward': 0.0,
        'episode_lengths': [],
        'episode_rewards': [],
        'observation_shapes': {},
        'errors': []
    }
    
    try:
        # Test multiple episodes
        total_steps = 0
        episode = 0
        
        while total_steps < max_steps:
            episode += 1
            
            # Reset environment
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            
            # Collect observation shapes
            for key, value in obs.items():
                if key not in stats['observation_shapes']:
                    stats['observation_shapes'][key] = value.shape if hasattr(value, 'shape') else str(type(value))
            
            # Run episode
            while total_steps < max_steps:
                # Take random action
                action = env.action_space.sample()
                
                try:
                    step_obs, reward, terminated, truncated, step_info = env.step(action)
                    
                    episode_reward += reward
                    episode_steps += 1
                    total_steps += 1
                    
                    # Validate observation structure
                    if not isinstance(step_obs, dict):
                        stats['errors'].append(f"Episode {episode}: Observation is not a dict")
                    
                    if terminated or truncated:
                        break
                        
                except Exception as e:
                    stats['errors'].append(f"Episode {episode}, Step {episode_steps}: {str(e)}")
                    break
            
            # Record episode statistics
            stats['episode_lengths'].append(episode_steps)
            stats['episode_rewards'].append(episode_reward)
            
            print(f"  Episode {episode}: {episode_steps} steps, reward={episode_reward:.3f}")
        
        # Calculate final statistics
        stats['episodes'] = episode
        stats['total_steps'] = total_steps
        stats['total_reward'] = sum(stats['episode_rewards'])
        stats['avg_episode_length'] = np.mean(stats['episode_lengths']) if stats['episode_lengths'] else 0
        stats['avg_episode_reward'] = np.mean(stats['episode_rewards']) if stats['episode_rewards'] else 0
        
    except Exception as e:
        stats['errors'].append(f"Fatal error: {str(e)}")
        
    finally:
        env.close()
    
    return stats


def print_test_results(stats: Dict[str, Any]) -> None:
    """Print formatted test results."""
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {stats['env_id']}")
    print(f"{'='*60}")
    
    print(f"Action Space: {stats['action_space']}")
    print(f"Observation Space: {stats['observation_space']}")
    print()
    
    print(f"Episodes Run: {stats['episodes']}")
    print(f"Total Steps: {stats['total_steps']}")
    print(f"Average Episode Length: {stats['avg_episode_length']:.1f}")
    print()
    
    print(f"Total Reward: {stats['total_reward']:.3f}")
    print(f"Average Episode Reward: {stats['avg_episode_reward']:.3f}")
    print()
    
    print("Observation Shapes:")
    for key, shape in stats['observation_shapes'].items():
        print(f"  {key}: {shape}")
    print()
    
    if stats['errors']:
        print(f"ERRORS ({len(stats['errors'])}):")
        for error in stats['errors'][:10]:  # Show max 10 errors
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    else:
        print("✅ NO ERRORS - Environment working correctly!")
    
    print(f"{'='*60}\n")


def test_all_environments(max_steps: int = 200) -> None:
    """Test all available environments."""
    available_envs = [
        'DrStrategy-MemoryMaze-4x7x7-v0',
        'DrStrategy-MemoryMaze-4x15x15-v0',
        'DrStrategy-MemoryMaze-8x30x30-v0', 
        'DrStrategy-MemoryMaze-mzx7x7-v0',
        'DrStrategy-MemoryMaze-mzx15x15-v0'
    ]
    
    print("DrStrategy Memory Maze Environment Testing")
    print("=" * 60)
    print(f"Testing {len(available_envs)} environments with {max_steps} steps each")
    print("=" * 60)
    
    all_results = []
    
    for env_id in available_envs:
        start_time = time.time()
        stats = test_environment(env_id, max_steps)
        test_time = time.time() - start_time
        stats['test_time'] = test_time
        
        print_test_results(stats)
        all_results.append(stats)
    
    # Summary
    print("SUMMARY")
    print("=" * 60)
    
    total_errors = sum(len(stats['errors']) for stats in all_results)
    working_envs = sum(1 for stats in all_results if len(stats['errors']) == 0)
    
    print(f"Environments tested: {len(all_results)}")
    print(f"Working correctly: {working_envs}/{len(all_results)}")
    print(f"Total errors found: {total_errors}")
    
    if working_envs == len(all_results):
        print("✅ ALL ENVIRONMENTS WORKING CORRECTLY!")
    else:
        print("❌ Some environments have issues - check error details above")
    
    print("\nPerformance Summary:")
    for stats in all_results:
        print(f"  {stats['env_id']}: "
              f"{stats['test_time']:.1f}s, "
              f"{stats['total_steps']/stats['test_time']:.1f} steps/sec")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Headless testing of DrStrategy Memory Maze environments'
    )
    
    available_envs = [
        'DrStrategy-MemoryMaze-4x7x7-v0',
        'DrStrategy-MemoryMaze-4x15x15-v0',
        'DrStrategy-MemoryMaze-8x30x30-v0',
        'DrStrategy-MemoryMaze-mzx7x7-v0', 
        'DrStrategy-MemoryMaze-mzx15x15-v0'
    ]
    
    parser.add_argument(
        '--env',
        choices=available_envs + ['all'],
        default='all',
        help='Environment to test, or "all" for all environments (default: %(default)s)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=200,
        help='Maximum steps per environment (default: %(default)s)'
    )
    
    args = parser.parse_args()
    
    if args.env == 'all':
        test_all_environments(args.steps)
    else:
        stats = test_environment(args.env, args.steps)
        print_test_results(stats)


if __name__ == '__main__':
    main()