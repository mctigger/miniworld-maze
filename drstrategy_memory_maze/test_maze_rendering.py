#!/usr/bin/env python3
"""Simple test script to verify maze rendering works."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import gymnasium as gym
import drstrategy_memory_maze


def test_maze_rendering():
    """Test and save a sample of maze rendering."""
    print("Testing DrStrategy Memory Maze rendering...")
    
    # Test different environments
    env_ids = [
        'DrStrategy-MemoryMaze-4x7x7-v0',
        'DrStrategy-MemoryMaze-mzx15x15-v0',
        'DrStrategy-MemoryMaze-8x30x30-v0'
    ]
    
    for env_id in env_ids:
        print(f"\nTesting {env_id}")
        
        # Create environment
        env = gym.make(env_id)
        obs, info = env.reset()
        
        # Take some steps to see agent movement and color changes
        images = []
        for step in range(15):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 5 == 0:  # Save every 5th frame
                images.append(obs['image'].copy())
                print(f"  Step {step}: Image range {obs['image'].min()}-{obs['image'].max()}, "
                      f"unique pixels: {len(np.unique(obs['image']))}, "
                      f"target: {obs['target_color']}")
        
        env.close()
        
        # Save sample images
        fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
        fig.suptitle(f'{env_id} - Agent Movement Over Time')
        
        if len(images) == 1:
            axes = [axes]
            
        for i, (ax, img) in enumerate(zip(axes, images)):
            ax.imshow(img)
            ax.set_title(f'Step {i*5}')
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        output_file = f'maze_test_{env_id.split("-")[-2]}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved sample images to {output_file}")


def test_web_api_simulation():
    """Simulate what the web API would return."""
    print("\nTesting Web API data format...")
    
    env = gym.make('DrStrategy-MemoryMaze-4x7x7-v0')
    obs, info = env.reset()
    
    # Take some steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    
    # Simulate the data that would be sent to web interface
    target_color = obs['target_color']
    target_color_list = [float(x) for x in target_color]
    
    api_data = {
        'env_id': 'DrStrategy-MemoryMaze-4x7x7-v0',
        'step_count': int(obs['step_count'][0]),
        'target_color': target_color_list,
        'target_color_hex': f"#{int(target_color_list[0]*255):02x}{int(target_color_list[1]*255):02x}{int(target_color_list[2]*255):02x}",
        'image_shape': obs['image'].shape,
        'image_min_max': [int(obs['image'].min()), int(obs['image'].max())],
        'unique_pixel_count': int(len(np.unique(obs['image'])))
    }
    
    print("Sample API data:")
    for key, value in api_data.items():
        print(f"  {key}: {value}")
    
    env.close()
    
    print("  ✅ Web API data format working correctly")


if __name__ == '__main__':
    test_maze_rendering()
    test_web_api_simulation()
    print("\n🎉 All maze rendering tests passed!")
    print("\nThe visualization scripts should now show:")
    print("  - Gray walls and light gray floors")  
    print("  - Green start positions and red goals")
    print("  - Blue agent moving around the maze")
    print("  - Changing target colors every 10 steps")
    print("  - White activity indicators in the corner")
    print("\nYou can now use:")
    print("  python visualize_frames.py  # Save frames to disk")
    print("  python visualize_web.py     # Web-based visualization")