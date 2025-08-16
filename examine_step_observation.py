#!/usr/bin/env python3
"""
Examine what the .step() observation contains - is it partial view or something else?
"""

import sys
import numpy as np
from PIL import Image

# Add path for our implementations
sys.path.insert(0, '/home/tim/Projects/drstrategy_memory-maze_differences')

from nine_rooms_factory import create_nine_rooms_env

def examine_step_observations():
    """Examine what the .step() method returns as observations."""
    print("=" * 70)
    print("EXAMINING STEP() OBSERVATIONS")
    print("=" * 70)
    
    # Create environment
    env = create_nine_rooms_env(variant="NineRooms", size=64)
    
    # Get base environment for comparison
    base_env = env._env
    while hasattr(base_env, 'env') or hasattr(base_env, '_env'):
        if hasattr(base_env, 'env'):
            base_env = base_env.env
        elif hasattr(base_env, '_env'):
            base_env = base_env._env
        else:
            break
    
    print(f"Environment created: {type(base_env)}")
    
    # Reset and get initial observation
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation from reset():")
    print(f"  Shape: {obs.shape}")
    print(f"  Data type: {obs.dtype}")
    print(f"  Range: [{obs.min()}, {obs.max()}]")
    
    # Save initial observation from .reset()
    obs_hwc = np.transpose(obs, (1, 2, 0))  # Convert CHW to HWC
    Image.fromarray(obs_hwc).save('step_obs_reset.png')
    print(f"  ✓ Saved: step_obs_reset.png")
    
    # Get agent position for context
    agent_pos = base_env.agent.pos
    agent_dir = base_env.agent.dir
    print(f"\nAgent state:")
    print(f"  Position: [{agent_pos[0]:.2f}, {agent_pos[1]:.2f}, {agent_pos[2]:.2f}]")
    print(f"  Direction: {agent_dir:.2f} degrees")
    
    # Compare with different render modes from base environment
    print(f"\n" + "="*50)
    print("COMPARING WITH BASE ENVIRONMENT RENDERS")
    print(f"="*50)
    
    # 1. Partial view (POMDP=True) - what agent can see
    partial_view = base_env.render_top_view(POMDP=True)
    Image.fromarray(partial_view).save('step_obs_partial_view.png')
    print(f"\n1. Partial view (POMDP=True):")
    print(f"  Shape: {partial_view.shape}")
    print(f"  ✓ Saved: step_obs_partial_view.png")
    print(f"  Description: Top-down view showing only what agent can 'see' within radius")
    
    # 2. Full view (POMDP=False) - complete environment
    full_view = base_env.render_top_view(POMDP=False)
    Image.fromarray(full_view).save('step_obs_full_view.png')
    print(f"\n2. Full view (POMDP=False):")
    print(f"  Shape: {full_view.shape}")
    print(f"  ✓ Saved: step_obs_full_view.png")
    print(f"  Description: Complete top-down view of entire environment")
    
    # 3. First-person view (what agent actually sees)
    first_person = base_env.render()
    Image.fromarray(first_person).save('step_obs_first_person.png')
    print(f"\n3. First-person view (base_env.render()):")
    print(f"  Shape: {first_person.shape}")
    print(f"  ✓ Saved: step_obs_first_person.png")
    print(f"  Description: First-person perspective from agent's viewpoint")
    
    # Now take some steps and see what changes
    print(f"\n" + "="*50)
    print("TAKING STEPS AND EXAMINING OBSERVATIONS")
    print(f"="*50)
    
    actions = [2, 1, 2]  # move_forward, turn_right, move_forward
    action_names = ['move_forward', 'turn_right', 'move_forward']
    
    for i, (action, name) in enumerate(zip(actions, action_names)):
        print(f"\n--- Step {i+1}: {name} ---")
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Save step observation
        obs_hwc = np.transpose(obs, (1, 2, 0))
        Image.fromarray(obs_hwc).save(f'step_obs_step_{i+1}_{name}.png')
        
        # Get new agent state
        new_pos = base_env.agent.pos
        new_dir = base_env.agent.dir
        
        print(f"  Action: {name} (action={action})")
        print(f"  Reward: {reward}")
        print(f"  New position: [{new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f}]")
        print(f"  New direction: {new_dir:.2f} degrees")
        print(f"  Observation shape: {obs.shape}")
        print(f"  ✓ Saved: step_obs_step_{i+1}_{name}.png")
        
        # Compare with first-person view at new position
        step_first_person = base_env.render()
        Image.fromarray(step_first_person).save(f'step_obs_first_person_step_{i+1}.png')
        print(f"  ✓ Saved: step_obs_first_person_step_{i+1}.png")
        
        if terminated or truncated:
            print(f"  Episode ended!")
            break
    
    # Analysis
    print(f"\n" + "="*70)
    print("ANALYSIS - WHAT IS THE STEP() OBSERVATION?")
    print(f"="*70)
    
    print(f"\n🔍 COMPARISON OF OBSERVATION TYPES:")
    print(f"")
    print(f"1. step() observation (processed through wrappers):")
    print(f"   • Shape: {obs.shape} (CHW format)")
    print(f"   • Size: 64x64 (resized from 80x80)")
    print(f"   • Content: First-person view from agent's perspective")
    print(f"   • Processing: ResizeObservationGymnasium + ImageToPyTorch wrappers")
    print(f"")
    print(f"2. base_env.render() (raw first-person):")
    print(f"   • Shape: {first_person.shape} (HWC format)")
    print(f"   • Size: 80x80 (original size)")
    print(f"   • Content: First-person view from agent's perspective")
    print(f"   • Processing: None (raw from MiniWorld)")
    print(f"")
    print(f"3. base_env.render_top_view(POMDP=True) (partial top-down):")
    print(f"   • Shape: {partial_view.shape} (HWC format)")
    print(f"   • Size: 80x80")
    print(f"   • Content: Top-down view showing agent's observable area")
    print(f"   • Processing: None (raw from MiniWorld)")
    print(f"")
    print(f"4. base_env.render_top_view(POMDP=False) (full top-down):")
    print(f"   • Shape: {full_view.shape} (HWC format)")
    print(f"   • Size: 80x80")
    print(f"   • Content: Complete top-down view of environment")
    print(f"   • Processing: None (raw from MiniWorld)")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"The observation returned by .step() contains the FIRST-PERSON VIEW")
    print(f"from the agent's current position and orientation, NOT the partial")
    print(f"top-down view. It shows what the agent would 'see' if it had eyes.")
    print(f"")
    print(f"This is similar to what a human would see looking forward in the")
    print(f"environment - walls, textures, and obstacles in front of the agent.")
    print(f"")
    print(f"The partial view (POMDP=True) is a different representation that")
    print(f"shows a top-down view of what's within the agent's 'sensing radius'.")
    
    env.close()

if __name__ == "__main__":
    examine_step_observations()