#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def save_gif(images, path, fps=5):
    """
    Save a sequence of images as a GIF
    """
    import imageio
    imageio.mimsave(path, images, fps=fps)

def create_video(images, path, fps=30):
    """
    Save a sequence of images as a video
    """
    import cv2
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for img in images:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)
    
    out.release()

def plot_training_curve(rewards, path=None):
    """
    Plot training rewards over time
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.grid(True)
    
    if path:
        plt.savefig(path)
    else:
        plt.show()

def random_agent_demo(env, num_episodes=5):
    """
    Demo of random agent in environment
    """
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1}: {steps} steps, reward: {total_reward:.2f}")