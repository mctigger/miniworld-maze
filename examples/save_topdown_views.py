#!/usr/bin/env python3
"""
Example script for generating and saving top-down views as PNG files.

This script demonstrates:
1. Generating a single top-down view and saving it as PNG
2. Running 5 steps and saving observation (left) and desired_goal (right) side by side
3. Executing 1000 random steps and visualizing agent trajectory on top-down view
"""

from typing import List, Tuple
import numpy as np
from PIL import Image
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def collect_trajectory_and_visualize(env: gym.Env) -> str:
    """Execute 1000 random steps and visualize agent trajectory on top-down view."""
    print("🔍 Collecting agent trajectory over 1000 random steps...")
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    # Get actual scene bounds from environment (same as render_top_view uses)
    env_unwrapped = env.unwrapped
    scene_min_x = env_unwrapped.min_x - 1
    scene_max_x = env_unwrapped.max_x + 1  
    scene_min_z = env_unwrapped.min_z - 1
    scene_max_z = env_unwrapped.max_z + 1
    
    print(f"   Environment bounds: x=[{scene_min_x:.1f}, {scene_max_x:.1f}], z=[{scene_min_z:.1f}, {scene_max_z:.1f}]")
    
    # Collect trajectory positions
    trajectory_positions = []
    
    # Execute 1000 random steps
    for step in range(1000):
        # Get current agent position
        agent_pos = env_unwrapped.agent.pos
        trajectory_positions.append((agent_pos[0], agent_pos[2]))  # x, z coordinates
        
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Reset if episode ended
        if terminated or truncated:
            obs, _ = env.reset()
        
        if (step + 1) % 200 == 0:
            print(f"   Completed {step + 1}/1000 steps")
    
    print(f"   ✅ Collected {len(trajectory_positions)} position points")
    
    # Get top-down view for background
    env.reset(seed=42)
    background_image = env_unwrapped.render_top_view(POMDP=False)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Display background image using actual scene bounds
    # Use 'upper' origin to match how single_topdown_view.png is saved (standard image coordinates)
    ax.imshow(background_image, extent=[scene_min_x, scene_max_x, scene_max_z, scene_min_z], origin='upper')
    
    # Extract x and z coordinates (no normalization needed now)
    x_coords = [pos[0] for pos in trajectory_positions]
    z_coords = [pos[1] for pos in trajectory_positions]
    
    # Plot trajectory using real world coordinates
    ax.plot(x_coords, z_coords, 'r-', linewidth=2, alpha=0.7, label='Agent Trajectory')
    
    # Mark start and end points
    if trajectory_positions:
        ax.scatter(x_coords[0], z_coords[0], c='green', s=100, marker='o', 
                  label='Start', zorder=5, edgecolors='black', linewidths=2)
        ax.scatter(x_coords[-1], z_coords[-1], c='red', s=100, marker='s', 
                  label='End', zorder=5, edgecolors='black', linewidths=2)
    
    # Add colormap for trajectory progression (plasma colormap for better visibility)
    colors = plt.cm.plasma(np.linspace(0, 1, len(trajectory_positions)))
    for i in range(len(trajectory_positions) - 1):
        ax.plot([x_coords[i], x_coords[i+1]], 
               [z_coords[i], z_coords[i+1]], 
               color=colors[i], linewidth=2, alpha=0.8)
    
    # Formatting using scene bounds (flipped Z to match image orientation)
    ax.set_xlim(scene_min_x, scene_max_x)
    ax.set_ylim(scene_max_z, scene_min_z)
    ax.set_title('Agent Trajectory over 1000 Random Steps', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Position (world coordinates)', fontsize=12)
    ax.set_ylabel('Z Position (world coordinates)', fontsize=12)
    ax.legend(loc='upper right')
    
    # Save the visualization
    filename = "agent_trajectory_visualization.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved trajectory visualization to: {filename}")
    
    return filename


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
        
        # 3. Collect trajectory and visualize
        trajectory_file = collect_trajectory_and_visualize(env)
        
        print("\n" + "=" * 50)
        print("🎉 Generation complete!")
        print("📁 Files created:")
        print(f"   • {single_file} (single top-down view)")
        for file in step_files:
            print(f"   • {file} (observation vs desired_goal)")
        print(f"   • {trajectory_file} (agent trajectory visualization)")
        
        print("\n💡 Tip: Open the PNG files to see the different views!")
        
    finally:
        env.close()


if __name__ == "__main__":
    main()