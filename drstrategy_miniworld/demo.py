#!/usr/bin/env python3
"""
Demo script showing how to use DrStrategy Miniworld environments.

This script demonstrates:
1. Basic environment creation and usage
2. Different environment types
3. Environment parameters

Usage:
    python demo.py
"""

import numpy as np
from drstrategy_miniworld.envs import (
    OneRoom, TwoRoomsVer1, ThreeRooms, 
    PickupObjs, RoomObjs, 
    SimToRealGoto, SimToRealPush
)

def demo_environment(env_class, env_name, **kwargs):
    """Demo a specific environment."""
    print(f"\n{'='*50}")
    print(f"🏠 Testing {env_name}")
    print(f"{'='*50}")
    
    # Create environment
    env = env_class(**kwargs)
    print(f"✓ Environment created with parameters: {kwargs}")
    print(f"  Action space: {env.action_space}")
    
    # Reset environment
    obs, info = env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Info keys: {list(info.keys()) if info else 'None'}")
    
    # Run a few steps
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Get action name if available
        action_name = ""
        if hasattr(env, 'actions'):
            try:
                action_name = f" ({env.actions(action).name})"
            except:
                pass
        
        print(f"  Step {step+1}: action={action}{action_name}, reward={reward}")
        
        # Show environment-specific info
        if hasattr(env, 'num_picked_up'):
            print(f"    Objects picked up: {env.num_picked_up}")
        if hasattr(env, 'agent') and hasattr(env.agent, 'carrying'):
            print(f"    Carrying object: {bool(env.agent.carrying)}")
        
        if terminated or truncated:
            print(f"    Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    print(f"  Total reward: {total_reward}")
    env.close()

def main():
    """Run demonstrations of all environments."""
    print("🎮 DrStrategy Miniworld Environment Demo")
    print("This demo shows all available environments with random actions")
    
    # Navigation environments
    demo_environment(OneRoom, "OneRoom (Single room navigation)", room_size=6)
    demo_environment(TwoRoomsVer1, "TwoRoomsVer1 (Two connected rooms)", room_size=6)
    demo_environment(ThreeRooms, "ThreeRooms (Linear three rooms)", room_size=6)
    
    # Object manipulation
    demo_environment(PickupObjs, "PickupObjs (Multi-object pickup)", size=8, num_objs=3)
    demo_environment(RoomObjs, "RoomObjs (Room with various objects)", size=6)
    
    # Sim-to-real transfer
    demo_environment(SimToRealGoto, "SimToRealGoto (Navigate to red box)")
    demo_environment(SimToRealPush, "SimToRealPush (Push red box to yellow box)")
    
    print(f"\n{'='*50}")
    print("🎉 Demo completed!")
    print("💡 For live visualization, run:")
    print("   python visualize_web.py --env OneRoom")
    print("   python visualize_web.py --env PickupObjs --num-objs 5")
    print("   python visualize_web.py --env SimToRealGoto")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()