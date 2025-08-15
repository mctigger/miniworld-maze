#!/usr/bin/env python3
"""
Keyboard-controlled agent for Nine Rooms environment.
Uses command line input and saves partial observations as images.
"""

import sys
import numpy as np
from PIL import Image
import os

# Add path for our pure gymnasium implementation
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences')

from nine_rooms_fully_pure_gymnasium import NineRoomsFullyPureGymnasium


class KeyboardAgent:
    def __init__(self, save_images=True):
        self.env = NineRoomsFullyPureGymnasium(name="NineRooms", obs_level=1, continuous=False, size=64)
        self.obs = None
        self.info = None
        self.step_count = 0
        self.total_reward = 0
        self.save_images = save_images
        
        # Action mapping
        self.action_map = {
            'a': 0,  # turn_left
            'left': 0,  # turn_left
            'd': 1,  # turn_right  
            'right': 1,  # turn_right
            'w': 2,  # move_forward
            'up': 2,  # move_forward
            'forward': 2,  # move_forward
        }
        
        # Create output directory for images
        if self.save_images:
            os.makedirs('keyboard_agent_observations', exist_ok=True)
            
    def reset_env(self):
        """Reset the environment and display info."""
        self.obs, self.info = self.env.reset(seed=42)
        self.step_count = 0
        self.total_reward = 0
        self.save_observation("reset")
        
        # Get agent position
        base_env = self.env._env
        while hasattr(base_env, 'env') or hasattr(base_env, '_env'):
            if hasattr(base_env, 'env'):
                base_env = base_env.env
            elif hasattr(base_env, '_env'):
                base_env = base_env._env
            else:
                break
                
        agent_pos = base_env.agent.pos
        agent_dir = base_env.agent.dir
        
        print(f"\nEnvironment reset!")
        print(f"Agent position: ({agent_pos[0]:.1f}, {agent_pos[2]:.1f})")
        print(f"Agent direction: {agent_dir:.1f}°")
        
    def step_env(self, action):
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.info = info
        self.step_count += 1
        self.total_reward += reward
        
        # Get updated agent position
        base_env = self.env._env
        while hasattr(base_env, 'env') or hasattr(base_env, '_env'):
            if hasattr(base_env, 'env'):
                base_env = base_env.env
            elif hasattr(base_env, '_env'):
                base_env = base_env._env
            else:
                break
                
        agent_pos = base_env.agent.pos
        agent_dir = base_env.agent.dir
        
        # Save observation
        action_names = ['turn_left', 'turn_right', 'move_forward']
        action_name = action_names[action]
        self.save_observation(f"step_{self.step_count:03d}_{action_name}")
        
        # Print step info
        print(f"Step {self.step_count}: Action={action_name}, Reward={reward:.2f}, Total={self.total_reward:.2f}")
        print(f"Agent position: ({agent_pos[0]:.1f}, {agent_pos[2]:.1f}), Direction: {agent_dir:.1f}°")
        
        if self.save_images:
            print(f"Observation saved: keyboard_agent_observations/step_{self.step_count:03d}_{action_name}.png")
        
        if terminated or truncated:
            print(f"\nEpisode finished! Total steps: {self.step_count}, Total reward: {self.total_reward:.2f}")
            return True
        return False
        
    def save_observation(self, prefix):
        """Save current observation as an image."""
        if self.obs is None or not self.save_images:
            return
            
        # Convert CHW to HWC for PIL
        obs_hwc = np.transpose(self.obs, (1, 2, 0))
        
        # Save image
        filename = f"keyboard_agent_observations/{prefix}.png"
        Image.fromarray(obs_hwc).save(filename)
        
    def print_help(self):
        """Print help information."""
        print("\n" + "="*60)
        print("NINE ROOMS KEYBOARD CONTROL")
        print("="*60)
        print("Controls:")
        print("  w, up, forward     - Move Forward")
        print("  a, left           - Turn Left")
        print("  d, right          - Turn Right")
        print("  r, reset          - Reset Environment")
        print("  h, help           - Show this help")
        print("  q, quit, exit     - Quit")
        print("="*60)
        print("The agent starts in the top-left room and can navigate")
        print("through all 9 rooms connected by doorways.")
        if self.save_images:
            print("Observations are saved as images in 'keyboard_agent_observations/'")
        print("="*60)
        
    def run(self):
        """Main game loop with keyboard input."""
        print("Starting Nine Rooms Environment with Keyboard Control...")
        
        # Initialize environment
        self.reset_env()
        self.print_help()
        
        print(f"\nReady! Enter commands (type 'help' for controls):")
        
        try:
            while True:
                # Get user input
                try:
                    user_input = input("> ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nQuitting...")
                    break
                    
                if not user_input:
                    continue
                    
                # Handle commands
                if user_input in ['q', 'quit', 'exit']:
                    print("Goodbye!")
                    break
                elif user_input in ['h', 'help']:
                    self.print_help()
                elif user_input in ['r', 'reset']:
                    self.reset_env()
                elif user_input in self.action_map:
                    action = self.action_map[user_input]
                    done = self.step_env(action)
                    if done:
                        print("Episode finished. Type 'r' to reset or 'q' to quit.")
                else:
                    print(f"Unknown command: '{user_input}'. Type 'help' for available commands.")
                    
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Check if user wants to disable image saving
    import argparse
    parser = argparse.ArgumentParser(description='Keyboard control for Nine Rooms environment')
    parser.add_argument('--no-images', action='store_true', 
                       help='Disable saving observation images')
    args = parser.parse_args()
    
    agent = KeyboardAgent(save_images=not args.no_images)
    agent.run()