#!/usr/bin/env python3
"""
Memory Maze Observation Capture Script (Standalone Version)

This script iterates through the Memory Maze environment and saves the first 50 
observations as PNG files in their own directory. Adapted for the standalone
memory-maze-drstrategy package.
"""

import argparse
import os
import sys
import traceback
import numpy as np
from pathlib import Path
from typing import Optional


def setup_mujoco_backend(backend: str) -> None:
    """Set MUJOCO_GL environment variable."""
    if backend != "auto":
        os.environ["MUJOCO_GL"] = backend
        print(f"Set MUJOCO_GL={backend}")
    elif "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
        print("Set MUJOCO_GL=egl (default)")
    else:
        print(f"Using existing MUJOCO_GL={os.environ['MUJOCO_GL']}")


def find_memory_maze_env(requested_env_id: str):
    """Find a suitable Memory Maze environment ID."""
    import gymnasium as gym
    
    # First try the requested ID
    try:
        gym.make(requested_env_id)
        return requested_env_id
    except gym.error.UnregisteredEnv:
        pass
    
    # Scan registry for MemoryMaze environments containing "7x7"
    registry = gym.envs.registry
    candidates = []
    
    for env_id in registry.keys():
        if "MemoryMaze" in env_id and "7x7" in env_id:
            candidates.append(env_id)
    
    if candidates:
        selected = candidates[0]
        print(f"Environment '{requested_env_id}' not found, using '{selected}'")
        return selected
    
    # Fallback to any MemoryMaze environment
    for env_id in registry.keys():
        if "MemoryMaze" in env_id:
            candidates.append(env_id)
    
    if candidates:
        selected = candidates[0]
        print(f"No 7x7 MemoryMaze found, using '{selected}'")
        return selected
    
    raise RuntimeError("No MemoryMaze environments found in registry")


def save_observation(observation: np.ndarray, save_path: str, step_num: int) -> bool:
    """Save observation as PNG file."""
    try:
        from PIL import Image
        
        # Handle different observation formats
        if observation.ndim == 3:
            if observation.shape[0] == 3:  # CHW format
                img_array = np.transpose(observation, (1, 2, 0))  # Convert to HWC
            else:  # Already HWC
                img_array = observation
        else:
            print(f"Warning: Cannot save frame with shape {observation.shape}")
            return False
        
        # Ensure uint8 format
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        filename = f"observation_{step_num:03d}.png"
        full_path = os.path.join(save_path, filename)
        img.save(full_path)
        return True
        
    except ImportError:
        print("Error: Pillow not available, cannot save frames")
        return False
    except Exception as e:
        print(f"Warning: Failed to save frame {step_num}: {e}")
        return False


def capture_observations(env_id: str, num_observations: int, output_dir: str, backend: str) -> None:
    """Capture and save observations from the environment."""
    
    # Set up backend
    setup_mujoco_backend(backend)
    
    # Import after setting MUJOCO_GL
    import gymnasium as gym
    import memory_maze  # This triggers environment registration
    
    # Find suitable environment
    actual_env_id = find_memory_maze_env(env_id)
    print(f"Selected env_id: {actual_env_id}")
    print(f"Using MUJOCO_GL backend: {os.environ.get('MUJOCO_GL', 'unset')}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving observations to: {output_path.absolute()}")
    
    # Create environment
    env = gym.make(actual_env_id)
    
    try:
        # Reset environment
        observation = env.reset()
        
        # Handle both old and new gym API
        if isinstance(observation, tuple):
            observation, info = observation
        
        # Find image-like observation
        obs_array = None
        obs_key = None
        if isinstance(observation, dict):
            # Look for image-like observations in dict
            for key, value in observation.items():
                if isinstance(value, np.ndarray) and value.ndim == 3 and value.dtype == np.uint8:
                    obs_array = value
                    obs_key = key
                    break
        elif isinstance(observation, np.ndarray) and observation.ndim == 3:
            obs_array = observation
        
        if obs_array is None:
            raise RuntimeError("No suitable image observation found")
        
        print(f"Observation {'key' if obs_key else 'array'}: {obs_key or 'direct'}")
        print(f"Observation dtype: {obs_array.dtype}, shape: {obs_array.shape}")
        
        # Save observations
        saved_count = 0
        step_count = 0
        
        while saved_count < num_observations:
            # Get current observation
            current_obs = observation[obs_key] if obs_key else observation
            
            # Save observation
            if save_observation(current_obs, str(output_path), saved_count):
                saved_count += 1
                if saved_count % 10 == 0:
                    print(f"Saved {saved_count}/{num_observations} observations")
            
            # Break if we have enough
            if saved_count >= num_observations:
                break
            
            # Take a step
            action = env.action_space.sample()
            step_result = env.step(action)
            
            # Handle both old and new gym API
            if len(step_result) == 4:
                observation, reward, done, info = step_result
                terminated = done
                truncated = False
            else:
                observation, reward, terminated, truncated, info = step_result
            
            step_count += 1
            
            # Reset if episode ended
            if terminated or truncated:
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    observation, info = reset_result
                else:
                    observation = reset_result
        
        print(f"Successfully saved {saved_count} observations in {step_count} steps")
        
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Capture Memory Maze observations (standalone version)")
    parser.add_argument("--env-id", default="MemoryMaze-7x7-v0", 
                       help="Environment ID (default: MemoryMaze-7x7-v0)")
    parser.add_argument("--num-observations", type=int, default=50,
                       help="Number of observations to capture (default: 50)")
    parser.add_argument("--output-dir", default="memory_maze_observations",
                       help="Output directory for PNG files (default: memory_maze_observations)")
    parser.add_argument("--backend", default="auto", 
                       choices=["auto", "egl", "osmesa", "glfw"],
                       help="MUJOCO_GL backend (default: auto)")
    
    args = parser.parse_args()
    
    try:
        capture_observations(args.env_id, args.num_observations, args.output_dir, args.backend)
        sys.exit(0)
        
    except Exception as e:
        if args.backend == "auto" and ("EGL" in str(e) or "GL" in str(e) or "OpenGL" in str(e)):
            print(f"EGL/GL error detected: {e}")
            print("Retrying with OSMesa backend...")
            try:
                capture_observations(args.env_id, args.num_observations, args.output_dir, "osmesa")
                sys.exit(0)
            except Exception as osmesa_e:
                print(f"FAILURE: OSMesa fallback also failed: {osmesa_e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            print(f"FAILURE: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()