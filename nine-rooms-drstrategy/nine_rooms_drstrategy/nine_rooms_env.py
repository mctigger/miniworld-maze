"""
Nine Rooms DrStrategy Environment

Faithful port of the NineRooms environment from the original DrStrategy implementation,
providing 2D top-down observations with room-based navigation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union
import math
from .render_utils import DrStrategyRenderer




class NineRoomsDrStrategyEnv(gym.Env):
    """
    Nine Rooms navigation environment from DrStrategy, providing 2D top-down observations.
    
    This is a faithful port of the original NineRooms environment from:
    drstrategy/drstrategy_envs/drstrategy_envs/miniworld/envs/roomnav.py
    
    The environment features:
    - 3x3 grid of rooms (numbered 0-8)
    - Rooms connected by doors according to adjacency rules
    - Each room contains 9 colored boxes arranged in a 3x3 grid
    - Agent navigation with partial observability
    - 2D top-down RGB observations showing local environment view
    """

    def __init__(
        self,
        observation_size: int = 64,
        room_size: float = 15.0,  # Matches original implementation
        door_size: float = 2.5,   # Matches original implementation
        max_steps: int = 1000,
        continuous: bool = False,  # Default to discrete for compatibility
        **kwargs
    ):
        super().__init__()
        
        # Environment parameters (matching original NineRooms)
        self.observation_size = observation_size
        self.room_size = room_size
        self.door_size = door_size
        self.max_steps = max_steps
        self.continuous = continuous
        
        # Room connections (from original NineRooms.__init__)
        self.connections = [(0,1), (0,3), (1,2), (1,4), (2,5), (3,4), (3,6), (4,5), (4,7), (5,8), (6,7), (7,8)]
        
        # Room textures (from original NineRooms.__init__)
        self.room_textures = ['beige','lightbeige', 'lightgray',
                             'copperred', 'skyblue', 'lightcobaltgreen',
                             'oakbrown', 'navyblue', 'cobaltgreen']
        
        # Action space - mimicking original discrete actions
        if continuous:
            # For continuous mode, use continuous actions (forward/backward, turn)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            # For discrete mode, use discrete actions (stay, forward, turn_left, turn_right, etc.)
            self.action_space = spaces.Discrete(6)  # 0: stay, 1: forward, 2: left, 3: right, 4: back_left, 5: back_right
        
        # Observation space - 2D top-down RGB view (CHW format)
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(3, observation_size, observation_size), 
            dtype=np.uint8
        )
        
        # Add shape attribute to match original RoomNav behavior
        # Original uses size=64 by default, stored as (height, width) for cv2.resize
        self.shape = (observation_size, observation_size)
        
        # Room setup (3x3 grid, rooms 0-8) - EXACTLY matching original layout
        self.grid_rows, self.grid_cols = 3, 3
        self.num_rooms = 9
        
        # Room textures (from original NineRooms.__init__ - EXACT order)
        # Room layout: | 0 | 1 | 2 |  <- row 0 (i=0): beige, lightbeige, lightgray
        #              | 3 | 4 | 5 |  <- row 1 (i=1): copperred, skyblue, lightcobaltgreen  
        #              | 6 | 7 | 8 |  <- row 2 (i=2): oakbrown, navyblue, cobaltgreen
        self.room_textures = ['beige','lightbeige', 'lightgray',
                             'copperred', 'skyblue', 'lightcobaltgreen', 
                             'oakbrown', 'navyblue', 'cobaltgreen']
        
        self.rooms = self._setup_rooms()
        
        # Build room adjacency from connections
        self.adjacency = {i: [] for i in range(self.num_rooms)}
        for room1, room2 in self.connections:
            self.adjacency[room1].append(room2)
            self.adjacency[room2].append(room1)
        
        # Agent state
        self.agent_pos = np.array([0.0, 0.0])  # [x, z] world coordinates
        self.agent_dir = 0.0  # Direction in radians
        self.current_room = 0
        self.step_count = 0
        
        # Goal management
        self.goal_positions = self._generate_goal_positions()
        self.room_goal_cnt = [0] * self.num_rooms
        self.room_goal_success = [0] * self.num_rooms
        self._goal_room = None
        self._goal_position = None
        self._goal_is_given = False
        
        # Initialize faithful DrStrategy renderer (copying original OpenGL logic exactly)
        self.dr_renderer = DrStrategyRenderer(observation_size=observation_size)
        
        # Boxes setup (9 boxes per room in 3x3 arrangement - from original _gen_world)
        # Must be after renderer initialization since we use renderer's color names
        self.boxes = self._setup_boxes()
        
    def _setup_rooms(self) -> List[Dict]:
        """Setup room geometry - EXACTLY matching original _gen_world."""
        rooms = []
        for i in range(3):  # 3 rows
            for j in range(3):  # 3 columns
                room_idx = 3*i + j  # Room numbering: 0,1,2 / 3,4,5 / 6,7,8
                
                # Room bounds - EXACTLY matching original: room_size*j, room_size*(j+0.95)
                min_x = self.room_size * j
                max_x = self.room_size * (j + 0.95)
                min_z = self.room_size * i  
                max_z = self.room_size * (i + 0.95)
                
                # Room texture assignment: self.textures[3*i+j] from original
                texture = self.room_textures[3*i + j]
                
                rooms.append({
                    'idx': room_idx,
                    'bounds': [min_x, max_x, min_z, max_z],
                    'center': [(min_x + max_x) / 2, (min_z + max_z) / 2],
                    'texture': texture
                })
        return rooms
    
    def _setup_boxes(self) -> List[Dict]:
        """Setup colored boxes in each room (from original _gen_world)."""
        boxes = []
        
        for i in range(3):  # room rows
            for j in range(3):  # room columns
                room_idx = 3*i + j
                room = self.rooms[room_idx]
                
                start_x = self.room_size * j
                start_z = self.room_size * i
                
                # Original: 9 boxes per room in 3x3 arrangement  
                for k in range(9):
                    # Original position calculation:
                    # pos=[_start_x+ (self.room_size/3)*(k%3)+0.16*self.room_size, 0, _start_y + (self.room_size/3)*(k//3)+0.16*self.room_size]
                    box_x = start_x + (self.room_size/3)*(k%3) + 0.16*self.room_size
                    box_z = start_z + (self.room_size/3)*(k//3) + 0.16*self.room_size
                    
                    # Original color calculation: (k+1+(i+1)*(j+1))%9
                    color_idx = (k + 1 + (i+1)*(j+1)) % len(self.dr_renderer.box_color_names)
                    color_name = self.dr_renderer.box_color_names[color_idx]
                    
                    boxes.append({
                        'room': room_idx,
                        'pos': [box_x, 0.0, box_z],
                        'color': color_name,
                        'size': 2 * self.room_size / 15  # From original size calculation
                    })
        return boxes
    
    def _generate_goal_positions(self) -> List[List[List[float]]]:
        """Generate goal positions for each room (matching original)."""
        goal_positions = []
        
        for room_idx in range(self.num_rooms):
            room = self.rooms[room_idx]
            center_x, center_z = room['center']
            
            # Two goal positions per room (matching original NineRooms goal setup)
            goals = [
                [center_x - 0.5, 0.0, center_z - 0.5],  # Goal position 1
                [center_x + 1.0, 0.0, center_z + 1.0],  # Goal position 2  
            ]
            goal_positions.append(goals)
        
        return goal_positions
    
    
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
        """
        Render 2D top-down POMDP observation using faithful DrStrategy renderer.
        
        Uses the DrStrategyRenderer which exactly copies the original OpenGL logic
        and parameters from the DrStrategy MiniWorld implementation.
        
        Returns:
            RGB observation as CHW array (3, observation_size, observation_size)
        """
        return self.dr_renderer.render_pomdp_view(
            agent_pos=self.agent_pos,
            agent_dir=self.agent_dir,
            rooms=self.rooms,
            boxes=self.boxes,
            connections=self.connections,
            goal_pos=self._goal_position,
            door_size=self.door_size
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset agent to room 0 center (matching original placed_room=0)
        self.current_room = 0
        self.agent_pos = np.array([2.5, 2.5])  # Matching original place_agent pos
        self.agent_dir = 0.0
        self.step_count = 0
        
        # Reset goal state
        self._goal_is_given = False
        self._goal_room = None
        self._goal_position = None
        
        # Generate observation
        obs = self._render_2d_observation()
        
        info = {
            'room': self.current_room,
            'position': self.agent_pos.copy(),
            'direction': self.agent_dir,
        }
        
        return obs, info
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        
        self.step_count += 1
        reward = 0.0
        
        # Execute action
        if self.continuous:
            # Continuous action space: [forward/backward, turn]
            forward_action = action[0]  # -1 to 1
            turn_action = action[1]     # -1 to 1
            
            # Move forward/backward
            if abs(forward_action) > 0.1:
                distance = forward_action * 2.0  # Scale movement
                self._move_forward(distance)
            
            # Turn left/right
            if abs(turn_action) > 0.1:
                turn_amount = turn_action * np.pi / 4  # Scale turning
                self.agent_dir += turn_amount
        else:
            # Discrete action space
            if action == 0:  # Stay
                pass
            elif action == 1:  # Move forward
                self._move_forward()
            elif action == 2:  # Turn left
                self.agent_dir += np.pi / 2
            elif action == 3:  # Turn right
                self.agent_dir -= np.pi / 2
            elif action == 4:  # Move forward + turn left
                self._move_forward()
                self.agent_dir += np.pi / 2
            elif action == 5:  # Move forward + turn right
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
            'pos': self.agent_pos.copy(),  # For compatibility
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
        """Set goal and return goal image (matching original interface)."""
        self._goal_is_given = True
        
        if room_idx is None:
            self._goal_room = np.random.randint(self.num_rooms)
        else:
            self._goal_room = room_idx
        
        if goal_idx is None:
            goal_idx = np.random.randint(len(self.goal_positions[self._goal_room]))
        
        self._goal_position = self.goal_positions[self._goal_room][goal_idx]
        self.room_goal_cnt[self._goal_room] += 1
        
        # Return goal visualization
        return self._render_2d_observation()
    
    def is_goal_achieved(self, pos: np.ndarray, goal_pos: Optional[np.ndarray] = None, 
                        threshold: float = 2.0, direction: Optional[np.ndarray] = None) -> bool:
        """Check if goal is achieved (matching original interface)."""
        if goal_pos is None:
            if self._goal_position is None:
                return False
            goal_pos = np.array([self._goal_position[0], self._goal_position[2]])
        
        # Calculate distance (using only x,z coordinates)
        pos_2d = pos if len(pos) == 2 else np.array([pos[0], pos[2]])
        goal_2d = goal_pos if len(goal_pos) == 2 else np.array([goal_pos[0], goal_pos[2]])
        
        distance = np.sqrt(np.sum((pos_2d - goal_2d)**2))
        
        achieved = distance < threshold
        if achieved and self._goal_is_given:
            self.room_goal_success[self._goal_room] += 1
            
        return achieved
    
    def get_room_goal_ratio(self) -> np.ndarray:
        """Get success ratio for each room (matching original interface)."""
        return np.array(self.room_goal_success) / (np.array(self.room_goal_cnt) + 1e-6)
    
    def render(self) -> np.ndarray:
        """Render environment (Gymnasium v0.29+ render API)."""
        obs = self._render_2d_observation()
        # Convert CHW back to HWC for display
        return np.transpose(obs, (1, 2, 0))
    
    def render_on_pos(self, pos):
        """
        Render observation from a specific position (matching original RoomNav.render_on_pos).
        
        Args:
            pos: Position as [x, y, z] or [x, z] (y component ignored)
            
        Returns:
            Rendered observation from that position as CHW array
        """
        if len(pos) == 3:
            # Convert from [x, y, z] to [x, z]
            render_pos = [pos[0], pos[2]]
        else:
            # Already [x, z] format
            render_pos = pos
            
        # Save current agent position
        current_pos = self.agent_pos.copy()
        
        # Temporarily move agent to render position
        self.agent_pos = render_pos
        
        # Get POMDP observation from this position  
        obs = self._render_2d_observation()
        
        # Restore original agent position
        self.agent_pos = current_pos
        
        # DrStrategy renderer already produces the correct size (64x64) and CHW format
        # No additional resizing needed since we're using the faithful renderer
        return obs


class SpiralNineRoomsDrStrategyEnv(NineRoomsDrStrategyEnv):
    """Nine Rooms environment with spiral connectivity (matching original SpiralNineRooms)."""
    
    def __init__(self, **kwargs):
        # Override connections with spiral pattern (from original SpiralNineRooms.__init__)
        super().__init__(**kwargs)
        self.connections = [(0,1), (0,3), (1,2), (2,5), (3,6), (4,5), (6,7), (7,8)]
        
        # Rebuild adjacency with new connections
        self.adjacency = {i: [] for i in range(self.num_rooms)}
        for room1, room2 in self.connections:
            self.adjacency[room1].append(room2)
            self.adjacency[room2].append(room1)