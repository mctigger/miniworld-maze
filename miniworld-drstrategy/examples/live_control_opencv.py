#!/usr/bin/env python3
"""
OpenCV-based live keyboard control for Nine Rooms environments.
This avoids OpenGL conflicts by using OpenCV for display instead of pygame.

Controls:
    W/A/S/D - Move forward/turn left/move backward/turn right
    SPACE   - Pick up object
    X       - Drop object
    E       - Toggle/activate object
    R       - Reset environment
    ESC/Q   - Quit
    1/2/3   - Switch between environment variants
"""

import argparse
import time
import sys
from typing import Optional, Dict, Any
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("❌ opencv-python is required for live control. Install with: pip install opencv-python")
    OPENCV_AVAILABLE = False
    sys.exit(1)

from miniworld_drstrategy import create_nine_rooms_env, ObservationLevel


class OpenCVLiveController:
    """Live environment controller using OpenCV for display."""
    
    def __init__(self, variant: str = "NineRooms", size: int = 256, obs_level: ObservationLevel = ObservationLevel.FIRST_PERSON):
        """
        Initialize the OpenCV live controller.
        
        Args:
            variant: Environment variant to start with
            size: Observation image size
            obs_level: Observation level (FIRST_PERSON, TOP_DOWN_PARTIAL, TOP_DOWN_FULL)
        """
        self.size = size
        self.current_variant = variant
        self.obs_level = obs_level
        self.env = None
        self.current_obs = None
        self.running = True
        
        # Available variants
        self.variants = ["NineRooms", "SpiralNineRooms", "TwentyFiveRooms"]
        self.variant_index = self.variants.index(variant) if variant in self.variants else 0
        
        # Action mapping for keyboard
        self.action_map = {
            ord('w'): 2,       # move_forward
            ord('s'): 3,       # move_back  
            ord('a'): 0,       # turn_left
            ord('d'): 1,       # turn_right
            ord(' '): 4,       # pickup (space)
            ord('x'): 5,       # drop
            ord('e'): 6,       # toggle
        }
        
        # Stats
        self.step_count = 0
        self.episode_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Display settings
        self.display_width = 800
        self.display_height = 600
        self.info_height = 150  # Height reserved for info text
        
    def create_environment(self, variant: str) -> bool:
        """
        Create or recreate the environment.
        
        Args:
            variant: Environment variant to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.env:
                self.env.close()
                
            print(f"🔄 Creating {variant} environment...")
            self.env = create_nine_rooms_env(
                variant=variant, 
                size=self.size,
                obs_level=self.obs_level
            )
            
            # Test reset to ensure environment works
            obs, info = self.env.reset()
            self.current_obs = obs  # Store current observation
            
            self.current_variant = variant
            self.step_count = 0
            self.episode_count += 1
            
            print(f"✅ {variant} environment ready!")
            print(f"   Observation shape: {obs.shape}")
            print(f"   Action space: {self.env.action_space}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create {variant} environment: {e}")
            return False
    
    def get_observation_display(self) -> Optional[np.ndarray]:
        """
        Get current observation formatted for OpenCV display.
        
        Returns:
            BGR image array for OpenCV or None if failed
        """
        try:
            if self.current_obs is not None:
                # Convert from CHW to HWC
                obs_hwc = np.transpose(self.current_obs, (1, 2, 0))
                
                # Convert RGB to BGR for OpenCV
                obs_bgr = cv2.cvtColor(obs_hwc, cv2.COLOR_RGB2BGR)
                
                # Resize for display
                display_size = min(self.display_width, self.display_height - self.info_height)
                obs_resized = cv2.resize(obs_bgr, (display_size, display_size))
                
                return obs_resized
            else:
                # Create placeholder
                placeholder = np.zeros((self.size, self.size, 3), dtype=np.uint8)
                placeholder[::20, :] = [100, 100, 100]  # Grid pattern
                placeholder[:, ::20] = [100, 100, 100]
                
                # Convert to BGR and resize
                placeholder_bgr = cv2.cvtColor(placeholder, cv2.COLOR_RGB2BGR)
                display_size = min(self.display_width, self.display_height - self.info_height)
                placeholder_resized = cv2.resize(placeholder_bgr, (display_size, display_size))
                
                return placeholder_resized
                
        except Exception as e:
            print(f"⚠️  Display conversion failed: {e}")
            # Return error placeholder
            error_img = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.putText(error_img, "Display Error", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return error_img
    
    def create_info_panel(self, obs_img: np.ndarray) -> np.ndarray:
        """
        Create info panel with controls and stats.
        
        Args:
            obs_img: Observation image
            
        Returns:
            Combined image with info panel
        """
        # Create info panel
        info_panel = np.zeros((self.info_height, self.display_width, 3), dtype=np.uint8)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)  # White text
        thickness = 1
        
        y_pos = 25
        line_height = 25
        
        # Environment info
        cv2.putText(info_panel, f"Environment: {self.current_variant}", (10, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        
        # Observation level
        obs_level_name = self.obs_level.name if hasattr(self.obs_level, 'name') else str(self.obs_level)
        cv2.putText(info_panel, f"View: {obs_level_name}", (10, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        
        # Stats
        cv2.putText(info_panel, f"Episode: {self.episode_count} | Step: {self.step_count} | FPS: {self.current_fps}", 
                   (10, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        
        # Controls (smaller font)
        font_scale_small = 0.5
        controls = [
            "CONTROLS: W/A/S/D=Move/Turn, SPACE=Pickup, X=Drop, E=Toggle",
            "R=Reset, 1/2/3=Switch Env, ESC/Q=Quit"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(info_panel, control, (10, y_pos + i * 20), font, font_scale_small, (200, 200, 200), thickness)
        
        # Resize observation to fit beside info
        obs_height = self.display_height - self.info_height
        obs_resized = cv2.resize(obs_img, (obs_height, obs_height))
        
        # Create combined image
        combined_height = self.display_height
        combined_width = max(obs_height, self.display_width)
        combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Place observation
        combined_img[0:obs_height, 0:obs_height] = obs_resized
        
        # Place info panel below (adjust width to match combined image)
        info_width = min(info_panel.shape[1], combined_width)
        combined_img[obs_height:combined_height, 0:info_width] = info_panel[:, 0:info_width]
        
        return combined_img
    
    def handle_input(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Args:
            key: OpenCV key code
            
        Returns:
            True to continue, False to quit
        """
        try:
            if key == 27 or key == ord('q'):  # ESC or Q
                return False
                
            elif key == ord('r'):
                # Reset environment
                if self.env:
                    try:
                        obs, info = self.env.reset()
                        self.current_obs = obs
                        self.step_count = 0
                        self.episode_count += 1
                        print(f"🔄 Environment reset (Episode {self.episode_count})")
                    except Exception as e:
                        print(f"❌ Reset failed: {e}")
                        
            elif key in [ord('1'), ord('2'), ord('3')]:
                # Switch environment variant
                variant_map = {ord('1'): 0, ord('2'): 1, ord('3'): 2}
                new_index = variant_map[key]
                if new_index < len(self.variants):
                    new_variant = self.variants[new_index]
                    print(f"🔄 Switching to {new_variant}...")
                    if self.create_environment(new_variant):
                        self.variant_index = new_index
                        
            elif self.env and key in self.action_map:
                # Execute action
                action = self.action_map[key]
                try:
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    self.current_obs = obs
                    self.step_count += 1
                    
                    # Print action feedback
                    action_names = ['turn_left', 'turn_right', 'move_forward', 'move_back', 
                                  'pickup', 'drop', 'toggle']
                    if action < len(action_names):
                        print(f"🎯 {action_names[action]} | Reward: {reward:.2f} | Step: {self.step_count}")
                    
                    if terminated or truncated:
                        print(f"🏁 Episode ended! Reward: {reward:.2f}")
                        obs, info = self.env.reset()
                        self.current_obs = obs
                        self.step_count = 0
                        self.episode_count += 1
                        
                except Exception as e:
                    print(f"❌ Action failed: {e}")
                    
        except Exception as e:
            print(f"❌ Input handling error: {e}")
            
        return True
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def run(self):
        """Main run loop."""
        print("🎮 Nine Rooms OpenCV Live Controller")
        print("=" * 50)
        
        print("🔄 Step 1: Creating initial environment...")
        # Create initial environment
        if not self.create_environment(self.current_variant):
            print("❌ Failed to create initial environment")
            return
        
        print("✅ Step 1 completed: Environment created")
            
        print("\n📖 Controls:")
        print("   W/A/S/D - Move forward/turn left/move backward/turn right")
        print("   SPACE - Pick up object")
        print("   X - Drop object")
        print("   E - Toggle/activate object")
        print("   R - Reset environment")
        print("   1/2/3 - Switch between environment variants")
        print("   ESC/Q - Quit")
        print("\n👀 OpenCV window will open - click on it to focus for keyboard input!")
        print("🔍 If window doesn't appear, try running with: DISPLAY=:0 python ...")
        
        # Create window
        window_name = "Nine Rooms Live Control (OpenCV)"
        print(f"📺 Creating OpenCV window: {window_name}")
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        print(f"✅ Window created successfully")
        
        try:
            while self.running:
                # Get current observation
                obs_img = self.get_observation_display()
                
                if obs_img is not None:
                    # Create combined display with info
                    display_img = self.create_info_panel(obs_img)
                    
                    # Show image
                    cv2.imshow(window_name, display_img)
                
                # Handle input (1ms timeout for responsiveness)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if not self.handle_input(key):
                        break
                
                # Update FPS
                self.update_fps()
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            
        finally:
            if self.env:
                self.env.close()
            cv2.destroyAllWindows()
            print("\n👋 OpenCV live controller stopped")


def main():
    """Main function."""
    if not OPENCV_AVAILABLE:
        return
        
    parser = argparse.ArgumentParser(description='OpenCV live control for Nine Rooms environments')
    parser.add_argument('--variant', 
                        choices=['NineRooms', 'SpiralNineRooms', 'TwentyFiveRooms'],
                        default='NineRooms',
                        help='Environment variant to start with')
    parser.add_argument('--size', type=int, default=128,
                        help='Observation image size (default: 128)')
    parser.add_argument('--obs-level', 
                        choices=['FIRST_PERSON', 'TOP_DOWN_PARTIAL', 'TOP_DOWN_FULL'],
                        default='FIRST_PERSON',
                        help='Observation level (default: FIRST_PERSON)')
    
    args = parser.parse_args()
    
    # Convert string to enum
    obs_level_map = {
        'FIRST_PERSON': ObservationLevel.FIRST_PERSON,
        'TOP_DOWN_PARTIAL': ObservationLevel.TOP_DOWN_PARTIAL,
        'TOP_DOWN_FULL': ObservationLevel.TOP_DOWN_FULL
    }
    obs_level = obs_level_map[args.obs_level]
    
    controller = OpenCVLiveController(variant=args.variant, size=args.size, obs_level=obs_level)
    controller.run()


if __name__ == "__main__":
    main()