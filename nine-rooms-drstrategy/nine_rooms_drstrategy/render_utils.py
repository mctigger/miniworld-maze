"""
Rendering utilities that closely replicate the original DrStrategy MiniWorld rendering.

This module copies the exact rendering logic and parameters from the original
DrStrategy implementation to ensure visual fidelity.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io


class DrStrategyRenderer:
    """
    Renderer that replicates the original DrStrategy MiniWorld top-down POMDP view.
    
    Copies the exact rendering parameters and logic from:
    drstrategy/drstrategy_envs/drstrategy_envs/miniworld/miniworld.py:render_top_view()
    """
    
    def __init__(self, observation_size=64):
        self.observation_size = observation_size
        
        # Original DrStrategy POMDP parameters (from render_top_view POMDP=True)
        self.pomdp_radius = 2.5  # min_x = agent.pos[0] - 2.5, etc.
        
        # Room colors based on original MiniWorld textures
        # These should match the actual rendered colors from MiniWorld
        self.room_colors = {
            'beige': (0.96, 0.96, 0.86),           # Room 0
            'lightbeige': (0.98, 0.98, 0.90),     # Room 1  
            'lightgray': (0.83, 0.83, 0.83),      # Room 2
            'copperred': (0.72, 0.45, 0.20),      # Room 3
            'skyblue': (0.53, 0.81, 0.92),        # Room 4
            'lightcobaltgreen': (0.56, 0.93, 0.56), # Room 5
            'oakbrown': (0.59, 0.29, 0.00),       # Room 6
            'navyblue': (0.00, 0.00, 0.50),       # Room 7
            'cobaltgreen': (0.24, 0.70, 0.44),    # Room 8
        }
        
        # Box colors (from original entity.py COLORS)
        self.box_colors = {
            'red': np.array([1.0, 0.0, 0.0]),
            'green': np.array([0.0, 1.0, 0.0]),
            'blue': np.array([0.0, 0.0, 1.0]),
            'purple': np.array([0.44, 0.15, 0.76]),
            'yellow': np.array([1.00, 1.00, 0.00]),
            'light_yellow': np.array([0.5, 0.00, 0.39]),
            'color1': np.array([0.7, 0.9, 0.39]),
            'color2': np.array([0.15, 0.3, 0.39]),
            'color3': np.array([1.0, 0.5, 0.0]),
            'color4': np.array([1.0, 0.0, 0.5]),
            'color5': np.array([0.3, 0.7, 0.1]),
        }
        self.box_color_names = sorted(list(self.box_colors.keys()))
        
        # Background color (from original: glClearColor(*self.black, 1.0))
        self.background_color = 'black'
    
    def render_pomdp_view(self, agent_pos, agent_dir, rooms, boxes, connections=None, goal_pos=None, door_size=2.5):
        """
        Render POMDP view exactly like original DrStrategy.
        
        Replicates the logic from:
        miniworld.py:render_top_view() with POMDP=True
        """
        agent_x, agent_z = agent_pos[0], agent_pos[1]
        
        # Original POMDP extents calculation (copied from render_top_view)
        min_x = agent_x - self.pomdp_radius
        max_x = agent_x + self.pomdp_radius  
        min_z = agent_z - self.pomdp_radius
        max_z = agent_z + self.pomdp_radius
        
        # Original aspect ratio calculation (copied from render_top_view)
        width = max_x - min_x
        height = max_z - min_z
        aspect = width / height
        fb_aspect = 1.0  # Square frame buffer (64x64)
        
        # Adjust aspect extents to match frame buffer aspect (copied logic)
        if aspect > fb_aspect:
            # Want to add to denom, add to height
            new_h = width / fb_aspect
            h_diff = new_h - height
            min_z -= h_diff / 2
            max_z += h_diff / 2
        elif aspect < fb_aspect:
            # Want to add to num, add to width
            new_w = height * fb_aspect
            w_diff = new_w - width
            min_x -= w_diff / 2
            max_x += w_diff / 2
        
        # Create figure with exact view extents
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_z, max_z)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor(self.background_color)
        
        # Render rooms within POMDP view
        for room in rooms:
            x_min, x_max, z_min, z_max = room['bounds']
            
            # Check if room intersects with POMDP view
            if (x_max >= min_x and x_min <= max_x and 
                z_max >= min_z and z_min <= max_z):
                
                # Clip room bounds to view
                clipped_x_min = max(x_min, min_x)
                clipped_x_max = min(x_max, max_x)
                clipped_z_min = max(z_min, min_z)
                clipped_z_max = min(z_max, max_z)
                
                width = clipped_x_max - clipped_x_min
                height = clipped_z_max - clipped_z_min
                
                if width > 0 and height > 0:
                    # Get room color
                    texture = room['texture']
                    color = self.room_colors.get(texture, (0.8, 0.8, 0.8))
                    
                    # Draw room rectangle (only visible portion)
                    rect = patches.Rectangle(
                        (clipped_x_min, clipped_z_min), width, height,
                        linewidth=1, edgecolor='black', facecolor=color, alpha=0.9
                    )
                    ax.add_patch(rect)
        
        # Render doors/connections - show gaps between connected rooms
        if connections:
            for room1_idx, room2_idx in connections:
                if room1_idx < len(rooms) and room2_idx < len(rooms):
                    room1 = rooms[room1_idx]
                    room2 = rooms[room2_idx]
                    
                    # Get room bounds
                    x1_min, x1_max, z1_min, z1_max = room1['bounds']
                    x2_min, x2_max, z2_min, z2_max = room2['bounds']
                    
                    # Original door logic: connect_rooms creates gaps in walls
                    # Check if rooms are horizontally adjacent (same z, different x)
                    if abs(z1_min - z2_min) < 0.1:  # Same row
                        if abs(x1_max - x2_min) < 0.1:  # room1 left of room2
                            # Door in vertical wall between rooms
                            door_x = x1_max  # Wall position
                            room_center_z = (z1_min + z1_max) / 2
                            door_z_min = room_center_z - door_size/2
                            door_z_max = room_center_z + door_size/2
                            
                            # Draw door opening (black gap) if in view
                            if (min_x <= door_x <= max_x and 
                                door_z_max >= min_z and door_z_min <= max_z):
                                ax.plot([door_x, door_x], [door_z_min, door_z_max], 'k-', linewidth=4, alpha=1.0)
                    
                    # Check if rooms are vertically adjacent (same x, different z)  
                    elif abs(x1_min - x2_min) < 0.1:  # Same column
                        if abs(z1_max - z2_min) < 0.1:  # room1 above room2
                            # Door in horizontal wall between rooms
                            door_z = z1_max  # Wall position
                            room_center_x = (x1_min + x1_max) / 2
                            door_x_min = room_center_x - door_size/2
                            door_x_max = room_center_x + door_size/2
                            
                            # Draw door opening (black gap) if in view
                            if (min_z <= door_z <= max_z and 
                                door_x_max >= min_x and door_x_min <= max_x):
                                ax.plot([door_x_min, door_x_max], [door_z, door_z], 'k-', linewidth=4, alpha=1.0)
        
        # Render boxes within POMDP view - as squares like original
        for box in boxes:
            box_x, _, box_z = box['pos']
            
            # Check if box is within POMDP view
            if (min_x <= box_x <= max_x and min_z <= box_z <= max_z):
                color_rgb = self.box_colors[box['color']]
                box_size = box.get('size', 2.0)  # Default from original: 2*room_size/15
                
                # Draw box as square (matching original Box entity)
                box_rect = patches.Rectangle(
                    (box_x - box_size/2, box_z - box_size/2), 
                    box_size, box_size,
                    linewidth=1, edgecolor='black', facecolor=color_rgb, alpha=0.8
                )
                ax.add_patch(box_rect)
        
        # Render agent (copied from original agent rendering)
        # Agent direction arrow
        arrow_length = 0.8  # Smaller for POMDP view
        dx = arrow_length * np.cos(agent_dir)
        dz = arrow_length * np.sin(agent_dir)
        
        ax.arrow(agent_x, agent_z, dx, dz, 
                head_width=0.3, head_length=0.2, fc='red', ec='red', linewidth=2)
        
        # Agent position dot
        ax.plot(agent_x, agent_z, 'ro', markersize=4)
        
        # Render goal if provided and within view
        if goal_pos is not None:
            goal_x, _, goal_z = goal_pos
            if (min_x <= goal_x <= max_x and min_z <= goal_z <= max_z):
                ax.plot(goal_x, goal_z, 'b*', markersize=8, alpha=0.9)
        
        # Convert to image with exact size matching original
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, 
                   dpi=self.observation_size/6*100, facecolor=self.background_color)
        buf.seek(0)
        
        # Load and resize to exact observation size
        img = Image.open(buf)
        img = img.resize((self.observation_size, self.observation_size), Image.LANCZOS)
        img_array = np.array(img)[:, :, :3]  # Remove alpha channel
        
        plt.close(fig)
        buf.close()
        
        # Convert from HWC to CHW format (PyTorch style) to match original
        return np.transpose(img_array, (2, 0, 1))
    
    def render_goal_view(self, goal_pos, rooms, boxes, connections=None):
        """
        Render view from goal position (for goal visualization).
        
        Replicates the logic from RoomNav.render_on_pos()
        """
        # Use goal position as "agent" position for rendering
        goal_dir = 0.0  # Default direction for goal rendering
        
        return self.render_pomdp_view(
            agent_pos=[goal_pos[0], goal_pos[2]], 
            agent_dir=goal_dir,
            rooms=rooms,
            boxes=boxes, 
            connections=connections,
            goal_pos=goal_pos
        )