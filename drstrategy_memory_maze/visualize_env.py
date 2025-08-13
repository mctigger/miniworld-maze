#!/usr/bin/env python3
"""Live visualization script for DrStrategy Memory Maze environments.

This script executes random actions at 20fps and displays the agent's observations
in real-time. Used for qualitative testing to verify environments work correctly.

Usage:
    python visualize_env.py [--env ENV_ID] [--steps STEPS] [--fps FPS]

Examples:
    python visualize_env.py --env DrStrategy-MemoryMaze-4x7x7-v0
    python visualize_env.py --env DrStrategy-MemoryMaze-mzx15x15-v0 --steps 1000
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import gymnasium as gym

import drstrategy_memory_maze


class EnvironmentVisualizer:
    """Real-time visualizer for DrStrategy Memory Maze environments."""
    
    def __init__(
        self, 
        env_id: str = 'DrStrategy-MemoryMaze-4x7x7-v0',
        fps: float = 20.0,
        max_steps: int = 1000
    ):
        """Initialize the visualizer.
        
        Args:
            env_id: Gymnasium environment ID
            fps: Target frames per second for visualization
            max_steps: Maximum steps per episode
        """
        self.env_id = env_id
        self.fps = fps
        self.frame_interval = 1000.0 / fps  # milliseconds
        self.max_steps = max_steps
        
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
        
        # Set up matplotlib figure
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
            self.action_counts = np.zeros(self.env.action_space.n)
        else:
            self.action_names = ['Continuous']
            self.action_counts = np.zeros(1)
        
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
        else:
            # Continuous actions - show recent values
            self.ax_actions.plot(self.recent_actions[-20:], 'o-')
            self.ax_actions.set_xlabel('Recent Steps')
            self.ax_actions.set_ylabel('Action Value')

    def _update_display(self) -> None:
        """Update all visualization components."""
        self._update_image_display()
        self._update_target_display()
        self._update_stats_display()
        self._update_action_display()
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run_interactive(self) -> None:
        """Run interactive visualization with matplotlib."""
        plt.ion()  # Turn on interactive mode
        
        try:
            start_time = time.time()
            frame_count = 0
            
            while self.total_steps < self.max_steps:
                frame_start = time.time()
                
                # Take action and get results
                action, reward, terminated, truncated, info = self._take_random_action()
                
                # Update display
                self._update_display()
                
                # Check for episode end
                if terminated or truncated:
                    print(f"Episode {self.episode_count} finished:")
                    print(f"  Steps: {self.episode_step}")
                    print(f"  Reward: {self.episode_reward:.3f}")
                    print(f"  Terminated: {terminated}, Truncated: {truncated}")
                    print()
                    
                    # Wait a moment before starting new episode
                    time.sleep(1.0)
                    self._reset_environment()
                
                # Control frame rate
                frame_time = time.time() - frame_start
                target_frame_time = 1.0 / self.fps
                
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)
                
                frame_count += 1
                
                # Print FPS occasionally
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    actual_fps = frame_count / elapsed
                    print(f"Running at {actual_fps:.1f} FPS (target: {self.fps:.1f})")
        
        except KeyboardInterrupt:
            print("\nVisualization interrupted by user")
        
        finally:
            self.env.close()
            plt.ioff()

    def run_animation(self) -> None:
        """Run as matplotlib animation (alternative approach)."""
        def animate(frame):
            """Animation function called by matplotlib."""
            if self.total_steps >= self.max_steps:
                return []
                
            # Take action
            action, reward, terminated, truncated, info = self._take_random_action()
            
            # Update display
            self._update_display()
            
            # Handle episode end
            if terminated or truncated:
                print(f"Episode {self.episode_count} finished: {self.episode_step} steps, {self.episode_reward:.3f} reward")
                time.sleep(0.5)  # Brief pause
                self._reset_environment()
            
            return []
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig, animate, 
            interval=self.frame_interval,
            blit=False,
            repeat=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nVisualization interrupted by user")
        finally:
            self.env.close()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Live visualization of DrStrategy Memory Maze environments',
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
        '--fps',
        type=float,
        default=20.0,
        help='Target frames per second (default: %(default)s)'
    )
    
    parser.add_argument(
        '--steps',
        type=int, 
        default=1000,
        help='Maximum total steps to run (default: %(default)s)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['interactive', 'animation'],
        default='interactive',
        help='Visualization mode (default: %(default)s)'
    )
    
    args = parser.parse_args()
    
    print("DrStrategy Memory Maze Environment Visualizer")
    print("=" * 50)
    print(f"Environment: {args.env}")
    print(f"Target FPS: {args.fps}")
    print(f"Max steps: {args.steps}")
    print(f"Mode: {args.mode}")
    print()
    print("Press Ctrl+C to stop visualization")
    print()
    
    # Create and run visualizer
    visualizer = EnvironmentVisualizer(
        env_id=args.env,
        fps=args.fps,
        max_steps=args.steps
    )
    
    if args.mode == 'animation':
        visualizer.run_animation()
    else:
        visualizer.run_interactive()
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()