#!/usr/bin/env python3
"""
Dr. Strategy 2D Top-Down Environments

This module implements the correct 2D top-down room navigation environments 
from the Dr. Strategy paper, without requiring MiniWorld dependencies.

These environments provide:
- 2D bird's-eye view observations (top-down perspective) 
- Discrete room-based navigation
- Goal-directed tasks with room-specific objectives
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io


class DrStrategy2DEnv(gym.Env):
    """
    Base class for Dr. Strategy 2D top-down room navigation environments.
    
    This implements the core functionality for:
    - 2D grid-world with rooms connected by doors
    - Agent navigation with discrete actions
    - 2D top-down visual observations  
    - Goal-based reward system
    """
    
    def __init__(
        self,
        room_grid_size: Tuple[int, int],
        connections: List[Tuple[int, int]],
        room_textures: List[str],
        observation_size: int = 64,
        room_size: float = 15.0,
        door_size: float = 2.5,
        max_steps: int = 1000,
    ):
        super().__init__()
        
        self.grid_rows, self.grid_cols = room_grid_size
        self.num_rooms = self.grid_rows * self.grid_cols
        self.connections = connections
        self.room_textures = room_textures
        self.observation_size = observation_size
        self.room_size = room_size
        self.door_size = door_size
        self.max_steps = max_steps
        
        # Action space: 0=stay, 1=forward, 2=turn_left, 3=turn_right, 4=forward+left, 5=forward+right
        self.action_space = spaces.Discrete(6)
        
        # Observation space: 2D RGB image (CHW format for PyTorch compatibility)
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(3, observation_size, observation_size), 
            dtype=np.uint8
        )
        
        # Agent state
        self.agent_pos = np.array([0.0, 0.0])  # [x, z] position in world coordinates
        self.agent_dir = 0.0  # Direction in radians
        self.current_room = 0
        self.step_count = 0
        
        # Room geometry (world coordinates)
        self.rooms = []
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                room_idx = i * self.grid_cols + j
                x_min = self.room_size * j
                x_max = self.room_size * (j + 0.95)
                z_min = self.room_size * i  
                z_max = self.room_size * (i + 0.95)
                self.rooms.append({
                    'idx': room_idx,
                    'bounds': [x_min, x_max, z_min, z_max],
                    'center': [(x_min + x_max) / 2, (z_min + z_max) / 2],
                    'texture': self.room_textures[room_idx % len(self.room_textures)]
                })
        
        # Build adjacency list from connections
        self.adjacency = {i: [] for i in range(self.num_rooms)}
        for room1, room2 in self.connections:
            self.adjacency[room1].append(room2)
            self.adjacency[room2].append(room1)
        
        # Goal management
        self.goal_positions = self._generate_goal_positions()
        self.room_goal_cnt = [0] * self.num_rooms
        self.room_goal_success = [0] * self.num_rooms
        self._goal_room = None
        self._goal_position = None
        self._goal_is_given = False
        
        # Color map for rooms
        self.room_colors = self._generate_room_colors()
        
    def _generate_goal_positions(self) -> List[List[List[float]]]:
        """Generate goal positions for each room."""
        goal_positions = []
        
        for room_idx in range(self.num_rooms):
            room = self.rooms[room_idx]
            center_x, center_z = room['center']
            
            # For 9-room environments, add 2 goals per room
            if self.num_rooms == 9:
                goals = [
                    [center_x - 0.5, 0.0, center_z - 0.5],  # Offset goal 1
                    [center_x + 1.0, 0.0, center_z + 1.0],  # Offset goal 2  
                ]
            else:
                # For other environments, 1 goal per room at center
                goals = [
                    [center_x, 0.0, center_z]
                ]
            
            goal_positions.append(goals)
        
        return goal_positions
    
    def _generate_room_colors(self) -> Dict[str, Tuple[float, float, float]]:
        """Generate color mapping for room textures."""
        colors = {
            'beige': (0.96, 0.96, 0.86),
            'lightbeige': (0.98, 0.98, 0.90),
            'lightgray': (0.83, 0.83, 0.83),
            'copperred': (0.72, 0.45, 0.20),
            'skyblue': (0.53, 0.81, 0.92),
            'lightcobaltgreen': (0.56, 0.93, 0.56),
            'oakbrown': (0.59, 0.29, 0.00),
            'navyblue': (0.00, 0.00, 0.50),
            'cobaltgreen': (0.24, 0.70, 0.44),
            'crimson': (0.86, 0.08, 0.24),
            'beanpaste': (0.64, 0.16, 0.16),
            'lightnavyblue': (0.68, 0.85, 0.90),
            'turquoise': (0.25, 0.88, 0.82),
            'violet': (0.93, 0.51, 0.93),
            'morningglory': (0.60, 0.40, 0.80),
            'silver': (0.75, 0.75, 0.75),
            'magenta': (1.00, 0.00, 1.00),
            'sunnyyellow': (1.00, 1.00, 0.00),
            'blueberry': (0.31, 0.31, 0.87),
            'seablue': (0.00, 0.75, 1.00),
            'lemongrass': (0.80, 1.00, 0.20),
            'orchid': (0.85, 0.44, 0.84),
            'redbean': (0.70, 0.13, 0.13),
            'orange': (1.00, 0.65, 0.00),
            'realblueberry': (0.19, 0.56, 0.84),
            'lime': (0.00, 1.00, 0.00),
        }
        return colors
        
    def _get_room_from_position(self, pos: np.ndarray) -> int:
        """Get room index from world position."""
        x, z = pos[0], pos[1]
        
        for room in self.rooms:
            x_min, x_max, z_min, z_max = room['bounds']
            if x_min <= x <= x_max and z_min <= z <= z_max:
                return room['idx']
        
        # If outside all rooms, return closest room
        distances = []
        for room in self.rooms:
            center_x, center_z = room['center']
            dist = np.sqrt((x - center_x)**2 + (z - center_z)**2)
            distances.append(dist)
        
        return int(np.argmin(distances))
    
    def _is_valid_move(self, from_room: int, to_room: int) -> bool:
        """Check if movement between rooms is allowed."""
        return to_room in self.adjacency[from_room]
    
    def _render_2d_observation(self) -> np.ndarray:
        """Render 2D top-down observation."""
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.room_size * self.grid_cols)
        ax.set_ylim(0, self.room_size * self.grid_rows)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw rooms
        for room in self.rooms:
            x_min, x_max, z_min, z_max = room['bounds']
            width = x_max - x_min
            height = z_max - z_min
            
            # Get room color
            texture = room['texture']
            color = self.room_colors.get(texture, (0.8, 0.8, 0.8))
            
            # Draw room rectangle
            rect = patches.Rectangle(
                (x_min, z_min), width, height,
                linewidth=2, edgecolor='black', facecolor=color, alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add room number
            center_x, center_z = room['center']
            ax.text(center_x, center_z, str(room['idx']), 
                   ha='center', va='center', fontsize=12, weight='bold')
        
        # Draw connections (doors)
        for room1_idx, room2_idx in self.connections:
            room1 = self.rooms[room1_idx]
            room2 = self.rooms[room2_idx]
            
            # Draw line between room centers
            x1, z1 = room1['center']
            x2, z2 = room2['center']
            
            ax.plot([x1, x2], [z1, z2], 'g-', linewidth=4, alpha=0.6)
        
        # Draw agent
        agent_x, agent_z = self.agent_pos
        
        # Agent direction arrow
        arrow_length = 2.0
        dx = arrow_length * np.cos(self.agent_dir)
        dz = arrow_length * np.sin(self.agent_dir)
        
        ax.arrow(agent_x, agent_z, dx, dz, 
                head_width=1.0, head_length=1.0, fc='red', ec='red', linewidth=2)
        
        # Agent position dot
        ax.plot(agent_x, agent_z, 'ro', markersize=8)
        
        # Draw goal if set
        if self._goal_is_given and self._goal_position is not None:
            goal_x, _, goal_z = self._goal_position
            ax.plot(goal_x, goal_z, 'b*', markersize=15, label='Goal')
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        
        # Load image and convert to numpy array
        img = Image.open(buf)
        img = img.resize((self.observation_size, self.observation_size))
        img_array = np.array(img)[:, :, :3]  # Remove alpha channel if present
        
        plt.close(fig)
        buf.close()
        
        # Convert from HWC to CHW format (PyTorch style)
        return np.transpose(img_array, (2, 0, 1))
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset agent to room 0 center
        self.current_room = 0
        self.agent_pos = np.array(self.rooms[0]['center'])
        self.agent_dir = 0.0
        self.step_count = 0
        
        # Reset goal state
        self._goal_is_given = False
        self._goal_room = None
        self._goal_position = None
        
        # Generate observation
        obs = self._render_2d_observation()
        
        return obs, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        
        self.step_count += 1
        
        # Execute action
        reward = 0.0
        
        if action == 0:  # Stay
            pass
        elif action == 1:  # Forward
            self._move_forward()
        elif action == 2:  # Turn left
            self.agent_dir += np.pi / 2
        elif action == 3:  # Turn right  
            self.agent_dir -= np.pi / 2
        elif action == 4:  # Forward + left
            self._move_forward()
            self.agent_dir += np.pi / 2
        elif action == 5:  # Forward + right
            self._move_forward()
            self.agent_dir -= np.pi / 2
        
        # Normalize direction
        self.agent_dir = self.agent_dir % (2 * np.pi)
        
        # Update current room
        new_room = self._get_room_from_position(self.agent_pos)
        self.current_room = new_room
        
        # Check goal achievement
        if self._goal_is_given and self._goal_position is not None:
            goal_x, _, goal_z = self._goal_position
            dist = np.sqrt((self.agent_pos[0] - goal_x)**2 + (self.agent_pos[1] - goal_z)**2)
            
            if dist < 2.0:  # Close enough to goal
                reward = 1.0
                self.room_goal_success[self._goal_room] += 1
        
        # Generate observation
        obs = self._render_2d_observation()
        
        # Check termination
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        info = {
            'room': self.current_room,
            'position': self.agent_pos.copy(),
            'direction': self.agent_dir,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _move_forward(self, distance: float = 2.0):
        """Move agent forward in current direction."""
        dx = distance * np.cos(self.agent_dir)
        dz = distance * np.sin(self.agent_dir)
        
        new_pos = self.agent_pos + np.array([dx, dz])
        
        # Check if new position is valid (within connected rooms)
        old_room = self._get_room_from_position(self.agent_pos)
        new_room = self._get_room_from_position(new_pos)
        
        # Allow movement within same room or to connected rooms
        if old_room == new_room or self._is_valid_move(old_room, new_room):
            # Ensure new position is within room bounds
            room = self.rooms[new_room]
            x_min, x_max, z_min, z_max = room['bounds']
            
            # Clamp to room boundaries with small margin
            margin = 1.0
            new_pos[0] = np.clip(new_pos[0], x_min + margin, x_max - margin)
            new_pos[1] = np.clip(new_pos[1], z_min + margin, z_max - margin)
            
            self.agent_pos = new_pos
        
    def get_goal(self, room_idx: Optional[int] = None, goal_idx: Optional[int] = None) -> np.ndarray:
        """Set goal and return goal image."""
        self._goal_is_given = True
        
        if room_idx is None:
            self._goal_room = np.random.randint(self.num_rooms)
        else:
            self._goal_room = room_idx
        
        if goal_idx is None:
            goal_idx = np.random.randint(len(self.goal_positions[self._goal_room]))
        
        self._goal_position = self.goal_positions[self._goal_room][goal_idx]
        self.room_goal_cnt[self._goal_room] += 1
        
        # Return goal visualization (same as current observation for simplicity)
        return self._render_2d_observation()
    
    def is_goal_achieved(self, pos: np.ndarray, goal_pos: Optional[np.ndarray] = None, 
                        threshold: float = 2.0, direction: Optional[np.ndarray] = None) -> bool:
        """Check if goal is achieved."""
        if goal_pos is None:
            if self._goal_position is None:
                return False
            goal_pos = np.array([self._goal_position[0], self._goal_position[2]])
        
        distance = np.sqrt(np.sum((pos - goal_pos[:2])**2))
        
        achieved = distance < threshold
        if achieved and self._goal_is_given:
            self.room_goal_success[self._goal_room] += 1
            
        return achieved
    
    def get_room_goal_ratio(self) -> np.ndarray:
        """Get success ratio for each room."""
        return np.array(self.room_goal_success) / (np.array(self.room_goal_cnt) + 1e-6)
    
    def render(self, mode: str = 'rgb_array') -> Optional[np.ndarray]:
        """Render environment."""
        if mode == 'rgb_array':
            obs = self._render_2d_observation()
            # Convert CHW back to HWC for display
            return np.transpose(obs, (1, 2, 0))
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")


class DrStrategy2DNineRooms(DrStrategy2DEnv):
    """Dr. Strategy NineRooms environment with 2D top-down observations."""
    
    def __init__(self, observation_size: int = 64, **kwargs):
        # 3x3 room grid with full connectivity
        connections = [(0,1), (0,3), (1,2), (1,4), (2,5), (3,4), (3,6), (4,5), (4,7), (5,8), (6,7), (7,8)]
        
        textures = ['beige','lightbeige', 'lightgray',
                   'copperred', 'skyblue', 'lightcobaltgreen',
                   'oakbrown', 'navyblue', 'cobaltgreen']
        
        super().__init__(
            room_grid_size=(3, 3),
            connections=connections,
            room_textures=textures,
            observation_size=observation_size,
            **kwargs
        )


class DrStrategy2DSpiralNineRooms(DrStrategy2DEnv):
    """Dr. Strategy SpiralNineRooms environment with 2D top-down observations."""
    
    def __init__(self, observation_size: int = 64, **kwargs):
        # 3x3 room grid with spiral connectivity
        connections = [(0,1), (0,3), (1,2), (2,5), (3,6), (4,5), (6,7), (7,8)]
        
        textures = ['beige','lightbeige', 'lightgray',
                   'copperred', 'skyblue', 'lightcobaltgreen',
                   'oakbrown', 'navyblue', 'cobaltgreen']
        
        super().__init__(
            room_grid_size=(3, 3),
            connections=connections,
            room_textures=textures,
            observation_size=observation_size,
            **kwargs
        )


class DrStrategy2DTwentyFiveRooms(DrStrategy2DEnv):
    """Dr. Strategy TwentyFiveRooms environment with 2D top-down observations."""
    
    def __init__(self, observation_size: int = 64, **kwargs):
        # 5x5 room grid with full connectivity
        connections = [(0,1), (0,5), (1,2), (1,6), (2,3), (2,7), (3,4), (3,8), (4,9),
                      (5,6), (5,10), (6,7), (6,11), (7,8), (7,12), (8,9), (8,13), (9,14),
                      (10,11), (10,15), (11,12), (11,16), (12,13), (12,17), (13,14), (13,18), (14,19),
                      (15,16), (15,20), (16,17), (16,21), (17,18), (17,22), (18,19), (18,23), (19,24),
                      (20,21), (21,22), (22,23), (23,24)]
        
        textures = ['crimson','beanpaste', 'cobaltgreen', 'lightnavyblue', 'skyblue', 
                   'lightcobaltgreen','oakbrown', 'copperred', 'lightgray', 'lime',
                   'turquoise', 'violet', 'beige', 'morningglory', 'silver',
                   'magenta','sunnyyellow', 'blueberry', 'lightbeige', 'seablue',
                   'lemongrass', 'orchid', 'redbean', 'orange', 'realblueberry']
        
        super().__init__(
            room_grid_size=(5, 5),
            connections=connections,
            room_textures=textures,
            observation_size=observation_size,
            **kwargs
        )