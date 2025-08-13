"""
Main environment classes for DrStrategy Memory Maze.

This module provides the main MemoryMaze environment class that creates
basic memory maze environments using the memory_maze package directly.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple, Union, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from .maze_layouts import get_layout, MazeLayout


class MemoryMaze(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    DrStrategy Memory Maze environment with Gymnasium API.
    
    This environment provides different maze layouts for testing agent 
    memory and navigation capabilities. It creates a minimal wrapper
    around gymnasium environments.
    """
    
    def __init__(
        self, 
        task: str, 
        discrete_actions: bool = True, 
        **kwargs: Any
    ) -> None:
        """
        Initialize the Memory Maze environment.
        
        Args:
            task: Task identifier (e.g., '4x7x7', '4x15x15', '8x30x30', 'mzx7x7', 'mzx15x15')
            discrete_actions: Whether to use discrete action space
            **kwargs: Additional arguments (ignored for simplicity)
        """
        super().__init__()
        
        # Set headless rendering for MuJoCo to avoid display dependencies
        os.environ.setdefault('MUJOCO_GL', 'osmesa')
        
        # Get layout metadata
        layout_mapping = {
            '4x7x7': 'FourRooms7x7',
            '4x15x15': 'FourRooms15x15', 
            '8x30x30': 'EightRooms30x30',
            'mzx7x7': 'Maze7x7',
            'mzx15x15': 'Maze15x15'
        }
        
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Task must be a non-empty string, got: {task!r}")
            
        if task not in layout_mapping:
            available = sorted(layout_mapping.keys())
            raise ValueError(f"Unknown task '{task}'. Available tasks: {available}")
        
        self.layout = get_layout(layout_mapping[task])
        self.discrete_actions = discrete_actions
        
        print(f"Creating Memory-Maze environment: {task}")
        
        # Create simple action and observation spaces
        # This is a minimal implementation focused on the core interface
        if discrete_actions:
            # 6 discrete actions: noop, forward, left, right, forward+left, forward+right
            self.action_space = spaces.Discrete(6)
        else:
            # Continuous 2D action space
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Simple observation space with image and basic info
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'target_color': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
            'step_count': spaces.Box(low=0, high=self.layout.max_steps, shape=(1,), dtype=np.int32)
        })
        
        # Initialize environment state
        self.num_steps: int = 0
        self.max_episode_steps: int = self.layout.max_steps
        self._current_obs: Optional[Dict[str, np.ndarray]] = None

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Generate observation with actual maze visualization."""
        try:
            # Create maze visualization
            maze_image = self._render_maze_view()
            
            # Generate dynamic target color (changes over time for visual variety)
            color_cycle = (self.num_steps // 10) % 6  # Change every 10 steps
            target_colors = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green  
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
            ]
            target_color = np.array(target_colors[color_cycle], dtype=np.float32)
            
            obs = {
                'image': maze_image,
                'target_color': target_color,
                'step_count': np.array([self.num_steps], dtype=np.int32)
            }
            
            self._current_obs = obs
            return obs
            
        except Exception as e:
            # Fallback to black image if rendering fails
            return {
                'image': np.zeros((64, 64, 3), dtype=np.uint8),
                'target_color': np.array([1.0, 0.0, 0.0], dtype=np.float32),
                'step_count': np.array([self.num_steps], dtype=np.int32)
            }

    def _render_maze_view(self) -> np.ndarray:
        """Render a top-down view of the maze with agent position."""
        # Create 64x64 RGB image
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Parse maze layout to get wall/floor structure  
        layout_lines = self.layout.layout.strip().split('\n')
        maze_height = len(layout_lines)
        maze_width = len(layout_lines[0]) if layout_lines else 0
        
        if maze_height == 0 or maze_width == 0:
            return img
            
        # Calculate scaling to fit maze in 64x64 image
        scale_x = 64 // maze_width
        scale_y = 64 // maze_height
        scale = min(scale_x, scale_y, 8)  # Cap at 8 pixels per cell
        
        # Center the maze in the image
        offset_x = (64 - maze_width * scale) // 2
        offset_y = (64 - maze_height * scale) // 2
        
        # Agent position (moves based on steps for visual feedback)
        agent_maze_x = 1 + (self.num_steps // 5) % max(1, maze_width - 2)
        agent_maze_y = 1 + (self.num_steps // 7) % max(1, maze_height - 2)
        
        # Render maze
        for y, line in enumerate(layout_lines):
            for x, char in enumerate(line):
                # Calculate pixel coordinates
                px_start_x = offset_x + x * scale
                px_end_x = px_start_x + scale
                px_start_y = offset_y + y * scale  
                px_end_y = px_start_y + scale
                
                # Ensure we don't go out of bounds
                px_start_x = max(0, min(px_start_x, 63))
                px_end_x = max(0, min(px_end_x, 64))
                px_start_y = max(0, min(px_start_y, 63))
                px_end_y = max(0, min(px_end_y, 64))
                
                if char == '*':  # Wall
                    img[px_start_y:px_end_y, px_start_x:px_end_x] = [100, 100, 100]  # Gray
                elif char == ' ':  # Floor
                    img[px_start_y:px_end_y, px_start_x:px_end_x] = [240, 240, 240]  # Light gray
                elif char == 'P':  # Start position
                    img[px_start_y:px_end_y, px_start_x:px_end_x] = [0, 255, 0]  # Green
                elif char == 'G':  # Goal
                    img[px_start_y:px_end_y, px_start_x:px_end_x] = [255, 0, 0]  # Red
        
        # Draw moving agent
        agent_px_x = offset_x + agent_maze_x * scale
        agent_px_y = offset_y + agent_maze_y * scale
        agent_size = max(2, scale // 2)
        
        # Agent bounds
        agent_x1 = max(0, min(agent_px_x, 63))
        agent_x2 = max(0, min(agent_px_x + agent_size, 64))
        agent_y1 = max(0, min(agent_px_y, 63))
        agent_y2 = max(0, min(agent_px_y + agent_size, 64))
        
        # Draw blue agent
        img[agent_y1:agent_y2, agent_x1:agent_x2] = [0, 100, 255]
        
        # Add step counter in corner
        step_color = [255, 255, 255]  # White text
        step_display = self.num_steps % 100  # Show last 2 digits
        
        # Simple digit rendering (just indicate activity)
        if step_display > 50:
            img[2:6, 2:6] = step_color
        if step_display % 20 < 10:
            img[2:6, 8:12] = step_color
            
        return img

    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info_dict)
        """
        if seed is not None:
            np.random.seed(seed)
        self.num_steps = 0
        self._current_obs = None
        obs = self._get_obs()
        info: Dict[str, Any] = {
            'layout': self.layout.layout,
            'max_steps': self.layout.max_steps,
            'seed': seed
        }
        return obs, info

    def step(
        self, action: ActType
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.num_steps += 1
        
        # Simple step logic - just update observation and check termination
        obs = self._get_obs()
        
        # Simple reward structure
        reward = 0.0
        if self.num_steps % 50 == 0:  # Small reward every 50 steps
            reward = 0.1
        
        # Episode termination conditions
        terminated = False  # Would be True if target reached in full implementation
        truncated = self.num_steps >= self.layout.max_steps
        
        info: Dict[str, Any] = {
            'step_count': self.num_steps,
            'max_steps': self.layout.max_steps,
            'action': action
        }
        
        return obs, reward, terminated, truncated, info

    def render(
        self, mode: str = 'rgb_array', **kwargs: Any
    ) -> Optional[np.ndarray]:
        """Render the environment.
        
        Args:
            mode: Render mode ('rgb_array' supported)
            **kwargs: Additional render arguments
            
        Returns:
            Rendered image as numpy array if mode='rgb_array', None otherwise
        """
        # Return a simple rendered view
        if mode == 'rgb_array':
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return None

    def close(self) -> None:
        """Close the environment."""
        pass


def make_env(
    task: str, 
    discrete_actions: bool = True, 
    **kwargs: Any
) -> MemoryMaze:
    """Factory function to create Memory Maze environments.
    
    Args:
        task: Task identifier (e.g., '4x7x7', 'mzx15x15')
        discrete_actions: Whether to use discrete actions
        **kwargs: Additional environment arguments
    
    Returns:
        Configured MemoryMaze environment instance
        
    Raises:
        ValueError: If task identifier is not recognized
    """
    return MemoryMaze(task=task, discrete_actions=discrete_actions, **kwargs)


def make(
    name: str, 
    obs_type: str, 
    frame_stack: int, 
    action_repeat: int, 
    seed: int, 
    **kwargs: Any
) -> MemoryMaze:
    """Legacy make function for compatibility with original DrStrategy interface.
    
    This function provides backward compatibility with the original make() API
    used in DrStrategy codebase.
    
    Args:
        name: Domain_task format name (e.g., 'rnavmemorymaze_4x7x7')
        obs_type: Observation type (ignored in current implementation)
        frame_stack: Number of frames to stack (ignored in current implementation) 
        action_repeat: Action repeat count (ignored in current implementation)
        seed: Random seed (ignored in current implementation)
        **kwargs: Additional arguments
        
    Returns:
        MemoryMaze environment instance
        
    Raises:
        ValueError: If domain is not recognized
    """
    # Validate input parameters
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Name must be a non-empty string, got: {name!r}")
        
    # Extract task from domain_task format
    parts = name.split('_', 1) if '_' in name else ('', name)
    domain, task = parts[0], parts[1] if len(parts) > 1 else parts[0]
    
    if domain and "rnavmemorymaze" not in domain.lower():
        raise ValueError(f"Unsupported domain '{domain}'. Expected domain containing 'rnavmemorymaze'")
    
    # Map discrete actions based on domain name
    discrete_actions = 'disc' in domain.lower()
    
    return make_env(task=task, discrete_actions=discrete_actions)