#!/usr/bin/env python3
"""
Example script for generating and saving top-down views as PNG files.

This script demonstrates:
1. Generating a single top-down view and saving it as PNG
2. Running 5 steps and saving observation (left) and desired_goal (right) side by side
"""

from typing import List
import numpy as np
from PIL import Image
import gymnasium as gym

from miniworld_maze import ObservationLevel
import miniworld_maze  # noqa: F401


def save_single_topdown_view(env: gym.Env) -> str:
    """Generate a single top-down full view and save it as PNG."""
    print("🎯 Generating single top-down full view...")
    
    # Reset environment to ensure consistent state
    env.reset(seed=42)
    
    # Use render_top_view directly for full view
    obs_image = env.unwrapped.render_top_view(POMDP=False)
    
    # Save the observation as PNG
    filename = "single_topdown_view.png"
    Image.fromarray(obs_image).save(filename)
    print(f"   ✅ Saved single top-down full view to: {filename}")
    
    return filename


def save_steps_comparison(env: gym.Env) -> List[str]:
    """Run 5 steps and save observation/desired_goal side by side."""
    print("🚶 Running 5 steps and saving observation/desired_goal comparisons...")
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    saved_files = []
    
    # Run 5 steps
    for step in range(5):
        # Take a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Get observation and desired_goal
        observation = obs["observation"]
        desired_goal = obs["desired_goal"]
        
        # Create side-by-side comparison
        # observation (left) and desired_goal (right)
        combined_width = observation.shape[1] + desired_goal.shape[1]
        combined_height = max(observation.shape[0], desired_goal.shape[0])
        
        # Create blank canvas
        combined_image = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
        
        # Place observation on the left
        combined_image[:observation.shape[0], :observation.shape[1]] = observation
        
        # Place desired_goal on the right
        combined_image[:desired_goal.shape[0], observation.shape[1]:] = desired_goal
        
        # Save the combined image
        filename = f"step_{step + 1}_obs_vs_desired.png"
        Image.fromarray(combined_image).save(filename)
        saved_files.append(filename)
        
        print(f"   Step {step + 1}: reward={reward:.3f}, saved to {filename}")
        
        # Reset if episode ended
        if terminated or truncated:
            obs, _ = env.reset()
    
    return saved_files


def main() -> None:
    """Main function demonstrating both features."""
    print("🖼️  Top-Down View Generator")
    print("=" * 50)
    
    # Create a single environment instance
    env = gym.make(
        "NineRooms-v0",
        obs_level=ObservationLevel.TOP_DOWN_PARTIAL,
        obs_width=256,
        obs_height=256,
    )
    
    try:
        # 1. Generate single top-down view
        single_file = save_single_topdown_view(env)
        
        print("\n" + "=" * 50)
        
        # 2. Run 5 steps and save comparisons
        step_files = save_steps_comparison(env)
        
        print("\n" + "=" * 50)
        print("🎉 Generation complete!")
        print("📁 Files created:")
        print(f"   • {single_file} (single top-down view)")
        for file in step_files:
            print(f"   • {file} (observation vs desired_goal)")
        
        print("\n💡 Tip: Open the PNG files to see the different views!")
        
    finally:
        env.close()


if __name__ == "__main__":
    main()