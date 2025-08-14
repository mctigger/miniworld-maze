#!/usr/bin/env python3
"""
Memory Maze Keyboard Navigation Script

Navigate through the 3D Memory Maze environment using arrow keys.
Based on the GUI implementation but simplified for easy keyboard control.
"""

import argparse
import os
import sys
import traceback
from collections import defaultdict

import numpy as np
import pygame
import pygame.freetype
from PIL import Image


def setup_mujoco_backend(backend: str) -> None:
    """Set MUJOCO_GL environment variable."""
    if backend != "auto":
        os.environ["MUJOCO_GL"] = backend
        print(f"Set MUJOCO_GL={backend}")
    elif "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "glfw"  # Use windowed rendering for interactive mode
        print("Set MUJOCO_GL=glfw (windowed rendering)")
    else:
        print(f"Using existing MUJOCO_GL={os.environ['MUJOCO_GL']}")


def get_keymap():
    """Define keyboard to action mapping for Memory Maze."""
    return {
        tuple(): 0,  # No action (stop)
        (pygame.K_UP,): 1,  # Move forward
        (pygame.K_LEFT,): 2,  # Turn left
        (pygame.K_RIGHT,): 3,  # Turn right
        (pygame.K_UP, pygame.K_LEFT): 4,  # Move forward and turn left
        (pygame.K_UP, pygame.K_RIGHT): 5,  # Move forward and turn right
    }


def find_7x7_env():
    """Find a suitable 7x7 Memory Maze environment."""
    import gymnasium as gym
    
    # Try different 7x7 variants in order of preference
    preferred_envs = [
        "MemoryMaze-four-rooms-7x7-fixed-layout-v0",
        "MemoryMaze-four-rooms-7x7-fixed-layout-Top-v0",
        "MemoryMaze-four-rooms-7x7-fixed-layout-random-goals-v0",
    ]
    
    for env_id in preferred_envs:
        try:
            gym.make(env_id)
            return env_id
        except gym.error.UnregisteredEnv:
            continue
    
    # Fallback to any 7x7 environment
    registry = gym.envs.registry
    for env_id in registry.keys():
        if "MemoryMaze" in env_id and "7x7" in env_id:
            try:
                gym.make(env_id)
                return env_id
            except gym.error.UnregisteredEnv:
                continue
    
    raise RuntimeError("No 7x7 MemoryMaze environments found")


def obs_to_text(obs, steps, return_, reward):
    """Convert observation to text display."""
    lines = []
    lines.append("=== MEMORY MAZE NAVIGATION ===")
    lines.append("")
    lines.append(f"Step:     {steps:>8}")
    lines.append(f"Return:   {return_:>8.2f}")
    lines.append(f"Reward:   {reward:>8.2f}")
    lines.append("")
    
    if isinstance(obs, dict):
        lines.append("=== OBSERVATIONS ===")
        for key, value in obs.items():
            if key != 'image':  # Skip image data
                if isinstance(value, np.ndarray):
                    if value.size <= 10:  # Only show small arrays
                        lines.append(f"{key}: {value}")
                    else:
                        lines.append(f"{key}: {value.shape} {value.dtype}")
                else:
                    lines.append(f"{key}: {value}")
    
    return lines


def keymap_to_text():
    """Generate help text for keyboard controls."""
    lines = []
    lines.append("=== CONTROLS ===")
    lines.append("")
    lines.append("↑         Move Forward")
    lines.append("←         Turn Left") 
    lines.append("→         Turn Right")
    lines.append("↑ + ←     Forward + Left")
    lines.append("↑ + →     Forward + Right")
    lines.append("")
    lines.append("SPACE     Pause/Resume")
    lines.append("BACKSPACE Reset Episode")
    lines.append("TAB       Speed Up")
    lines.append("ESC       Quit")
    lines.append("")
    lines.append("=== GOAL ===")
    lines.append("Navigate to the target!")
    return lines


def run_navigation(env_id: str, render_size: tuple, fps: int, backend: str, camera_resolution: int = None):
    """Run the keyboard navigation interface."""
    
    # Set up backend
    setup_mujoco_backend(backend)
    
    # Import after setting MUJOCO_GL
    import gymnasium as gym
    import memory_maze  # This triggers environment registration
    
    # Use flexible environment with custom resolution if specified
    if camera_resolution:
        if env_id == "MemoryMaze-four-rooms-7x7-fixed-layout-v0":
            env_id = "MemoryMaze-four-rooms-7x7-fixed-layout-flexible-v0"
            print(f"Using flexible environment with camera resolution: {camera_resolution}")
            env = gym.make(env_id, camera_resolution=camera_resolution)
        else:
            print(f"Camera resolution specified ({camera_resolution}) but environment {env_id} doesn't support flexible resolution")
            env = gym.make(env_id)
    else:
        print(f"Creating environment: {env_id}")
        env = gym.make(env_id)
    
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    keymap = get_keymap()
    
    # Game state
    steps = 0
    return_ = 0.0
    episode = 0
    last_reward = 0.0
    
    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
    
    # Initialize pygame
    pygame.init()
    window_size = (render_size[0] + 400, render_size[1])  # Extra space for text
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Memory Maze - Keyboard Navigation")
    clock = pygame.time.Clock()
    font = pygame.freetype.SysFont('Courier', 14)
    
    running = True
    paused = False
    speedup = False
    
    print("\n=== Navigation Started ===")
    print("Use arrow keys to navigate!")
    print("Press ESC to quit, SPACE to pause, BACKSPACE to reset")
    
    try:
        while running:
            # === RENDERING ===
            screen.fill((32, 32, 32))
            
            # Render main observation image
            if isinstance(obs, dict):
                if 'image' in obs:
                    image = obs['image']
                else:
                    # Find first image-like observation
                    image = None
                    for key, value in obs.items():
                        if isinstance(value, np.ndarray) and value.ndim == 3:
                            image = value
                            break
                    if image is None:
                        raise RuntimeError("No image observation found")
            else:
                image = obs
            
            # Convert and resize image
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize(render_size, resample=Image.NEAREST)
            image_array = np.array(pil_image)
            
            # Create pygame surface and blit
            surface = pygame.surfarray.make_surface(image_array.transpose((1, 0, 2)))
            screen.blit(surface, (0, 0))
            
            # Render text information
            text_lines = obs_to_text(obs, steps, return_, last_reward)
            y = 10
            for line in text_lines:
                if line.strip():  # Skip empty lines for rendering
                    text_surface, rect = font.render(line, (255, 255, 255))
                    screen.blit(text_surface, (render_size[0] + 10, y))
                y += 18
            
            # Render controls help
            help_lines = keymap_to_text()
            y += 20
            for line in help_lines:
                if line.strip():
                    color = (200, 255, 200) if line.startswith("===") else (255, 255, 255)
                    text_surface, rect = font.render(line, color)
                    screen.blit(text_surface, (render_size[0] + 10, y))
                y += 18
            
            # Show pause indicator
            if paused:
                pause_text, rect = font.render("*** PAUSED ***", (255, 255, 0))
                screen.blit(pause_text, (render_size[0] + 10, render_size[1] - 30))
            
            pygame.display.flip()
            clock.tick(fps if not speedup else 60)
            
            # === INPUT HANDLING ===
            pygame.event.pump()
            keys_down = defaultdict(bool)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    keys_down[event.key] = True
            
            keys_hold = pygame.key.get_pressed()
            
            # Determine action from keymap
            action = keymap[tuple()]  # Default: no action
            for keys, act in keymap.items():
                if all(keys_hold[key] or keys_down[key] for key in keys):
                    action = act
            
            # Handle special keys
            force_reset = False
            speedup = keys_hold[pygame.K_TAB]
            
            if keys_down[pygame.K_ESCAPE]:
                running = False
            if keys_down[pygame.K_SPACE]:
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif action != keymap[tuple()]:
                paused = False  # Unpause on movement
            if keys_down[pygame.K_BACKSPACE]:
                force_reset = True
                print("FORCED RESET")
            
            if paused:
                continue
            
            # === ENVIRONMENT STEP ===
            step_result = env.step(action)
            
            # Handle both old and new gym API
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                terminated = done
                truncated = False
            else:
                obs, reward, terminated, truncated, info = step_result
            
            steps += 1
            return_ += reward
            last_reward = reward
            
            # Print rewards
            if reward > 0:
                print(f"REWARD: {reward:.3f} at step {steps}")
            
            # Handle episode termination
            if terminated or truncated or force_reset:
                print(f"\n=== Episode {episode + 1} Complete ===")
                print(f"Steps: {steps}, Return: {return_:.3f}")
                if terminated:
                    print("Episode terminated (goal reached?)")
                elif truncated:
                    print("Episode truncated (timeout?)")
                
                # Reset environment
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    obs, info = reset_result
                else:
                    obs = reset_result
                
                steps = 0
                return_ = 0.0
                last_reward = 0.0
                episode += 1
                print("=== New Episode Started ===\n")
    
    finally:
        pygame.quit()
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Navigate Memory Maze with keyboard")
    parser.add_argument("--env-id", default="auto",
                       help="Environment ID (default: auto-detect 7x7)")
    parser.add_argument("--size", type=int, nargs=2, default=[1024, 1024],
                       help="Render size in pixels (default: 1024x1024)")
    parser.add_argument("--fps", type=int, default=10,
                       help="Frames per second (default: 10)")
    parser.add_argument("--backend", default="auto",
                       choices=["auto", "glfw", "egl", "osmesa"],
                       help="MUJOCO_GL backend (default: auto)")
    parser.add_argument("--camera-resolution", type=int, 
                       help="Camera resolution for environment rendering (e.g., 64, 128, 256, 512)")
    
    args = parser.parse_args()
    
    try:
        # Find environment
        if args.env_id == "auto":
            env_id = find_7x7_env()
            print(f"Auto-selected environment: {env_id}")
        else:
            env_id = args.env_id
        
        # Run navigation
        run_navigation(env_id, tuple(args.size), args.fps, args.backend, args.camera_resolution)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()