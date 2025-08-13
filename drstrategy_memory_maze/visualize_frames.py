#!/usr/bin/env python3
"""Frame-saving visualization script for DrStrategy Memory Maze environments.

This script executes random actions and saves visualization frames to disk,
suitable for headless environments and creating video sequences.

Usage:
    python visualize_frames.py [--env ENV_ID] [--steps STEPS] [--output DIR]

Examples:
    python visualize_frames.py --env DrStrategy-MemoryMaze-4x7x7-v0 --output frames/
    python visualize_frames.py --steps 200 --fps 10
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gymnasium as gym

import drstrategy_memory_maze


class FrameSaver:
    """Saves visualization frames to disk for headless environments."""
    
    def __init__(
        self, 
        env_id: str = 'DrStrategy-MemoryMaze-4x7x7-v0',
        output_dir: str = 'frames',
        max_steps: int = 100
    ):
        """Initialize the frame saver.
        
        Args:
            env_id: Gymnasium environment ID
            output_dir: Directory to save frames
            max_steps: Maximum steps to run
        """
        self.env_id = env_id
        self.output_dir = Path(output_dir)
        self.max_steps = max_steps
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environment
        print(f"Creating environment: {env_id}")
        self.env = gym.make(env_id)
        print(f"Action space: {self.env.action_space}")
        print(f"Observation space keys: {list(self.env.observation_space.spaces.keys())}")
        
        # Initialize tracking variables
        self.current_obs: Optional[Dict[str, np.ndarray]] = None
        self.current_info: Optional[Dict[str, Any]] = None
        self.episode_reward = 0.0
        self.episode_step = 0
        self.episode_count = 0
        self.total_steps = 0
        self.frame_count = 0
        
        # Set up matplotlib figure (non-interactive)
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle(f'DrStrategy Memory Maze: {env_id}', fontsize=16)
        
        # Configure subplots
        self._setup_plots()
        
        # Initialize environment
        self._reset_environment()

    def _setup_plots(self) -> None:
        """Set up the matplotlib subplot layout."""
        # Top-left: Environment observation image
        self.ax_image = self.axes[0, 0]
        self.ax_image.set_title('Agent Observation (Image)')
        self.ax_image.set_xticks([])
        self.ax_image.set_yticks([])
        
        # Top-right: Target color display
        self.ax_target = self.axes[0, 1] 
        self.ax_target.set_title('Target Color')
        self.ax_target.set_xlim(0, 1)
        self.ax_target.set_ylim(0, 1)
        self.ax_target.set_xticks([])
        self.ax_target.set_yticks([])
        
        # Bottom-left: Episode statistics
        self.ax_stats = self.axes[1, 0]
        self.ax_stats.set_title('Episode Statistics')
        self.ax_stats.axis('off')
        
        # Bottom-right: Action history (bar chart)
        self.ax_actions = self.axes[1, 1]
        self.ax_actions.set_title('Action History (Last 20)')
        self.ax_actions.set_xlabel('Action')
        self.ax_actions.set_ylabel('Count')
        
        # Initialize action tracking
        if hasattr(self.env.action_space, 'n'):
            self.action_names = [f'Action {i}' for i in range(self.env.action_space.n)]
        else:
            self.action_names = ['Continuous']
        
        self.recent_actions = []
        
        plt.tight_layout()

    def _reset_environment(self) -> None:
        """Reset the environment and update tracking variables."""
        print(f"Starting episode {self.episode_count + 1}")
        self.current_obs, self.current_info = self.env.reset(seed=None)
        self.episode_reward = 0.0
        self.episode_step = 0
        self.episode_count += 1

    def _take_random_action(self) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a random action and return step results."""
        action = self.env.action_space.sample()
        
        # Track action for visualization
        if hasattr(self.env.action_space, 'n'):
            self.recent_actions.append(action)
            if len(self.recent_actions) > 20:
                self.recent_actions.pop(0)
        
        # Take step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update tracking
        self.current_obs = obs
        self.current_info = info
        self.episode_reward += reward
        self.episode_step += 1
        self.total_steps += 1
        
        return action, reward, terminated, truncated, info

    def _update_image_display(self) -> None:
        """Update the agent observation image display."""
        if self.current_obs is None:
            return
            
        image = self.current_obs.get('image', np.zeros((64, 64, 3), dtype=np.uint8))
        
        self.ax_image.clear()
        self.ax_image.set_title('Agent Observation (Image)')
        self.ax_image.imshow(image)
        self.ax_image.set_xticks([])
        self.ax_image.set_yticks([])
        
        # Add step count overlay
        step_count = self.current_obs.get('step_count', [0])[0]
        self.ax_image.text(5, 10, f'Step: {step_count}', 
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          fontsize=10, color='black')

    def _update_target_display(self) -> None:
        """Update the target color display."""
        if self.current_obs is None:
            return
            
        target_color = self.current_obs.get('target_color', np.array([1.0, 0.0, 0.0]))
        
        self.ax_target.clear()
        self.ax_target.set_title('Target Color')
        
        # Create a colored rectangle
        rect = Rectangle((0.1, 0.1), 0.8, 0.8, 
                        facecolor=target_color, 
                        edgecolor='black', 
                        linewidth=2)
        self.ax_target.add_patch(rect)
        
        # Add RGB values as text
        rgb_text = f'RGB: ({target_color[0]:.2f}, {target_color[1]:.2f}, {target_color[2]:.2f})'
        self.ax_target.text(0.5, 0.05, rgb_text, 
                           horizontalalignment='center',
                           fontsize=10)
        
        self.ax_target.set_xlim(0, 1)
        self.ax_target.set_ylim(0, 1)
        self.ax_target.set_xticks([])
        self.ax_target.set_yticks([])

    def _update_stats_display(self) -> None:
        """Update the statistics display."""
        self.ax_stats.clear()
        self.ax_stats.set_title('Episode Statistics')
        self.ax_stats.axis('off')
        
        # Prepare statistics text
        stats_text = [
            f'Episode: {self.episode_count}',
            f'Episode Step: {self.episode_step}',
            f'Total Steps: {self.total_steps}',
            f'Episode Reward: {self.episode_reward:.3f}',
            f'Frame: {self.frame_count}',
            ''
        ]
        
        # Add environment info
        if self.current_info:
            stats_text.extend([
                'Environment Info:',
                f'Max Steps: {self.current_info.get("max_steps", "N/A")}',
                f'Step Count: {self.current_info.get("step_count", "N/A")}',
            ])
        
        # Display text
        text_str = '\n'.join(stats_text)
        self.ax_stats.text(0.05, 0.95, text_str, 
                          transform=self.ax_stats.transAxes,
                          verticalalignment='top',
                          fontsize=11,
                          fontfamily='monospace')

    def _update_action_display(self) -> None:
        """Update the action history display."""
        if not self.recent_actions:
            return
            
        self.ax_actions.clear()
        self.ax_actions.set_title('Action History (Last 20)')
        
        if hasattr(self.env.action_space, 'n'):
            # Discrete actions - show histogram
            action_counts = np.bincount(self.recent_actions, 
                                      minlength=self.env.action_space.n)
            
            bars = self.ax_actions.bar(range(len(action_counts)), action_counts)
            self.ax_actions.set_xlabel('Action ID')
            self.ax_actions.set_ylabel('Count')
            self.ax_actions.set_xticks(range(len(action_counts)))
            
            # Color the most recent action differently
            if self.recent_actions:
                most_recent = self.recent_actions[-1]
                bars[most_recent].set_color('red')

    def _save_frame(self) -> None:
        """Save current frame to disk."""
        # Update all displays
        self._update_image_display()
        self._update_target_display() 
        self._update_stats_display()
        self._update_action_display()
        
        plt.tight_layout()
        
        # Save frame
        frame_path = self.output_dir / f'frame_{self.frame_count:06d}.png'
        self.fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        
        self.frame_count += 1
        
        # Print progress occasionally
        if self.frame_count % 20 == 0:
            print(f"Saved {self.frame_count} frames...")

    def run(self) -> None:
        """Run the frame saving loop."""
        print(f"Saving frames to: {self.output_dir}")
        print(f"Running for {self.max_steps} steps...")
        
        start_time = time.time()
        
        try:
            while self.total_steps < self.max_steps:
                # Take action and get results
                action, reward, terminated, truncated, info = self._take_random_action()
                
                # Save frame
                self._save_frame()
                
                # Check for episode end
                if terminated or truncated:
                    print(f"Episode {self.episode_count} finished:")
                    print(f"  Steps: {self.episode_step}")
                    print(f"  Reward: {self.episode_reward:.3f}")
                    print(f"  Terminated: {terminated}, Truncated: {truncated}")
                    
                    # Save final frame of episode
                    self._save_frame()
                    
                    # Reset for new episode if we have steps remaining
                    if self.total_steps < self.max_steps:
                        self._reset_environment()
        
        except KeyboardInterrupt:
            print("\nFrame saving interrupted by user")
        
        finally:
            self.env.close()
            
            elapsed_time = time.time() - start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\nFrame saving complete!")
            print(f"Saved {self.frame_count} frames in {elapsed_time:.1f} seconds")
            print(f"Average rate: {fps:.1f} frames/second")
            print(f"Output directory: {self.output_dir.absolute()}")
            
            # Create a simple video script
            self._create_video_script()

    def _create_video_script(self) -> None:
        """Create a script to convert frames to video."""
        script_content = f"""#!/bin/bash
# Script to convert frames to video using ffmpeg

# Create video at 20 FPS
ffmpeg -r 20 -i frame_%06d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -r 20 -pix_fmt yuv420p {self.env_id.replace('-', '_')}_visualization.mp4

echo "Video created: {self.env_id.replace('-', '_')}_visualization.mp4"

# Alternative: Create GIF (smaller file, lower quality)
ffmpeg -r 5 -i frame_%06d.png -vf "scale=800:-1,palettegen" palette.png
ffmpeg -r 5 -i frame_%06d.png -i palette.png -lavfi "scale=800:-1,paletteuse" {self.env_id.replace('-', '_')}_visualization.gif
rm palette.png

echo "GIF created: {self.env_id.replace('-', '_')}_visualization.gif"
"""
        
        script_path = self.output_dir / 'create_video.sh'
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        
        print(f"Video creation script saved to: {script_path}")
        print(f"To create video, run: cd {self.output_dir} && ./create_video.sh")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Frame-saving visualization of DrStrategy Memory Maze environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Get available environment IDs
    available_envs = [
        'DrStrategy-MemoryMaze-4x7x7-v0',
        'DrStrategy-MemoryMaze-4x15x15-v0', 
        'DrStrategy-MemoryMaze-8x30x30-v0',
        'DrStrategy-MemoryMaze-mzx7x7-v0',
        'DrStrategy-MemoryMaze-mzx15x15-v0'
    ]
    
    parser.add_argument(
        '--env', 
        default='DrStrategy-MemoryMaze-4x7x7-v0',
        choices=available_envs,
        help='Environment to visualize (default: %(default)s)'
    )
    
    parser.add_argument(
        '--steps',
        type=int, 
        default=100,
        help='Maximum total steps to run (default: %(default)s)'
    )
    
    parser.add_argument(
        '--output',
        default='frames',
        help='Output directory for frames (default: %(default)s)'
    )
    
    args = parser.parse_args()
    
    print("DrStrategy Memory Maze Frame Saver")
    print("=" * 50)
    print(f"Environment: {args.env}")
    print(f"Max steps: {args.steps}")
    print(f"Output directory: {args.output}")
    print()
    print("Press Ctrl+C to stop early")
    print()
    
    # Create and run frame saver
    saver = FrameSaver(
        env_id=args.env,
        output_dir=args.output,
        max_steps=args.steps
    )
    
    saver.run()


if __name__ == '__main__':
    main()