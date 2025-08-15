#!/usr/bin/env python3
"""
Simple keyboard navigation script for Nine Rooms DrStrategy Environment.

Controls:
- Arrow Keys: Move agent (Up=Forward, Down=Backward, Left=Turn Left, Right=Turn Right)  
- WASD: Alternative movement controls
- SPACE: Stay/do nothing
- ESC/Q: Quit
- R: Reset environment

Shows two views side by side:
1. Left panel: POMDP observation (agent's limited view)
2. Right panel: Full grid view (complete environment layout)
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import tempfile
import os
import sys
from nine_rooms_drstrategy import NineRoomsDrStrategyEnv

# Pygame constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 600
PANEL_WIDTH = WINDOW_WIDTH // 2
PANEL_HEIGHT = WINDOW_HEIGHT
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class KeyboardNavigator:
    """Interactive keyboard navigation for Nine Rooms environment."""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Nine Rooms DrStrategy - Keyboard Navigation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Initialize environment
        self.env = NineRoomsDrStrategyEnv()
        self.obs, self.info = self.env.reset()
        
        # Create temp directory for images
        self.temp_dir = tempfile.mkdtemp()
        
        # Action mapping
        self.action_map = {
            'stay': 0,
            'forward': 1, 
            'turn_left': 2,
            'turn_right': 3,
            'forward_left': 4,
            'forward_right': 5
        }
        
        self.running = True
        self.step_count = 0
        
    def __del__(self):
        """Cleanup temp directory."""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass
        
    def handle_events(self):
        """Handle pygame events and keyboard input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                action = None
                
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Reset environment
                    self.obs, self.info = self.env.reset()
                    self.step_count = 0
                    print("Environment reset!")
                elif event.key == pygame.K_SPACE:
                    action = self.action_map['stay']
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    action = self.action_map['forward']
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    # Backward = turn around + forward
                    action = self.action_map['forward']  # We'll handle backward differently
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    action = self.action_map['turn_left']
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    action = self.action_map['turn_right']
                
                # Execute action
                if action is not None:
                    # Special handling for backward movement
                    if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        # Turn around, move forward, turn back
                        self.obs, reward, terminated, truncated, self.info = self.env.step(self.action_map['turn_left'])
                        self.obs, reward, terminated, truncated, self.info = self.env.step(self.action_map['turn_left'])
                        self.obs, reward, terminated, truncated, self.info = self.env.step(self.action_map['forward'])
                        self.obs, reward, terminated, truncated, self.info = self.env.step(self.action_map['turn_left'])
                        self.obs, reward, terminated, truncated, self.info = self.env.step(self.action_map['turn_left'])
                    else:
                        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
                    
                    self.step_count += 1
                    self.print_status()
    
    def print_status(self):
        """Print current status to console."""
        pos = self.info['position']
        room = self.info['room']
        direction = self.info['direction']
        print(f"Step {self.step_count}: Room {room}, Pos [{pos[0]:.1f}, {pos[1]:.1f}], Dir {direction:.2f}")
    
    def save_pomdp_view(self):
        """Save POMDP observation as image file and return path."""
        # Convert CHW to HWC
        obs_hwc = np.transpose(self.obs, (1, 2, 0))
        
        # Save as image
        img = Image.fromarray(obs_hwc)
        img = img.resize((PANEL_WIDTH - 40, PANEL_WIDTH - 40))
        
        filepath = os.path.join(self.temp_dir, 'pomdp_view.png')
        img.save(filepath)
        return filepath
    
    def save_full_grid_view(self):
        """Save full grid view as image file and return path."""
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Set limits to show entire grid
        room_size = self.env.room_size
        grid_size = 3 * room_size
        padding = 2
        
        ax.set_xlim(-padding, grid_size + padding)
        ax.set_ylim(-padding, grid_size + padding)
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        ax.axis('off')
        
        # Draw all rooms
        for room in self.env.rooms:
            x_min, x_max, z_min, z_max = room['bounds']
            width = x_max - x_min
            height = z_max - z_min
            
            # Get room color
            texture = room['texture']
            color = self.env.dr_renderer.room_colors.get(texture, (0.8, 0.8, 0.8))
            
            # Draw room rectangle
            rect = patches.Rectangle(
                (x_min, z_min), width, height,
                linewidth=1, edgecolor='gray', facecolor=color, alpha=0.9
            )
            ax.add_patch(rect)
            
            # Add room number
            center_x, center_z = room['center']
            ax.text(center_x, center_z, str(room['idx']), 
                    ha='center', va='center', fontsize=10, fontweight='bold', 
                    color='black', alpha=0.8)
        
        # Draw connections
        for room1_idx, room2_idx in self.env.connections:
            room1 = self.env.rooms[room1_idx]
            room2 = self.env.rooms[room2_idx]
            x1, z1 = room1['center']
            x2, z2 = room2['center']
            ax.plot([x1, x2], [z1, z2], 'g-', linewidth=2, alpha=0.6)
        
        # Draw some boxes (sample)
        for i, box in enumerate(self.env.boxes[::9]):  # Every 9th box
            box_x, _, box_z = box['pos']
            color_rgb = self.env.dr_renderer.box_colors[box['color']]
            ax.plot(box_x, box_z, 'o', color=color_rgb, markersize=4, alpha=0.8)
        
        # Draw agent
        agent_x, agent_z = self.env.agent_pos
        arrow_length = 1.5
        dx = arrow_length * np.cos(self.env.agent_dir)
        dz = arrow_length * np.sin(self.env.agent_dir)
        
        ax.arrow(agent_x, agent_z, dx, dz, 
                head_width=0.8, head_length=0.5, fc='red', ec='red', linewidth=2)
        ax.plot(agent_x, agent_z, 'ro', markersize=8)
        
        # Save figure
        filepath = os.path.join(self.temp_dir, 'full_grid.png')
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0, 
                   dpi=100, facecolor='black')
        plt.close(fig)
        
        return filepath
    
    def draw_info_panel(self, surface, x, y, title, content):
        """Draw an information panel."""
        panel_rect = pygame.Rect(x, y, PANEL_WIDTH - 20, 80)
        pygame.draw.rect(surface, GRAY, panel_rect)
        pygame.draw.rect(surface, WHITE, panel_rect, 2)
        
        # Title
        title_text = self.font.render(title, True, WHITE)
        surface.blit(title_text, (x + 10, y + 5))
        
        # Content
        y_offset = 25
        for line in content:
            text = self.small_font.render(line, True, WHITE)
            surface.blit(text, (x + 10, y + y_offset))
            y_offset += 18
    
    def draw_controls_help(self, surface):
        """Draw controls help at the bottom."""
        help_text = [
            "Controls: Arrow Keys/WASD=Move, SPACE=Stay, R=Reset, ESC/Q=Quit",
            "Movement: ↑/W=Forward, ↓/S=Backward, ←/A=Turn Left, →/D=Turn Right"
        ]
        
        y = WINDOW_HEIGHT - 50
        for i, text in enumerate(help_text):
            rendered_text = self.small_font.render(text, True, WHITE)
            surface.blit(rendered_text, (10, y + i * 20))
    
    def run(self):
        """Main game loop."""
        print("Nine Rooms DrStrategy - Keyboard Navigation")
        print("="*50)
        print("Controls:")
        print("  Arrow Keys/WASD: Move agent")
        print("  SPACE: Stay")  
        print("  R: Reset environment")
        print("  ESC/Q: Quit")
        print("="*50)
        self.print_status()
        
        while self.running:
            self.handle_events()
            
            # Clear screen
            self.screen.fill(BLACK)
            
            try:
                # Left panel - POMDP observation
                pomdp_path = self.save_pomdp_view()
                pomdp_surface = pygame.image.load(pomdp_path)
                self.screen.blit(pomdp_surface, (20, 100))
                
                # Right panel - Full grid view  
                grid_path = self.save_full_grid_view()
                grid_surface = pygame.image.load(grid_path)
                grid_surface = pygame.transform.scale(grid_surface, (PANEL_WIDTH - 40, PANEL_HEIGHT - 120))
                self.screen.blit(grid_surface, (PANEL_WIDTH + 20, 100))
                
            except Exception as e:
                print(f"Error loading images: {e}")
                # Draw error message
                error_text = self.font.render("Error loading images", True, RED)
                self.screen.blit(error_text, (50, 200))
            
            # POMDP info panel
            pomdp_info = [
                f"Agent View (POMDP): 5x5 unit window",
                f"Room: {self.info['room']}",
                f"Position: [{self.info['position'][0]:.1f}, {self.info['position'][1]:.1f}]",
                f"Direction: {self.info['direction']:.2f} rad"
            ]
            self.draw_info_panel(self.screen, 10, 10, "Agent's Limited View", pomdp_info)
            
            # Full grid info panel
            grid_info = [
                f"Complete Environment (3x3 rooms)",
                f"Step count: {self.step_count}",
                f"Room colors: Original DrStrategy textures",
                f"Green lines: Room connections"
            ]
            self.draw_info_panel(self.screen, PANEL_WIDTH + 10, 10, "Full Environment View", grid_info)
            
            # Draw controls help
            self.draw_controls_help(self.screen)
            
            # Draw separator line
            pygame.draw.line(self.screen, WHITE, (PANEL_WIDTH, 0), (PANEL_WIDTH, WINDOW_HEIGHT), 2)
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        print("Navigation session ended.")

if __name__ == "__main__":
    try:
        navigator = KeyboardNavigator()
        navigator.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        pygame.quit()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        pygame.quit()
        sys.exit(1)