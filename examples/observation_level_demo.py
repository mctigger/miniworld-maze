#!/usr/bin/env python3
"""
Demonstration of the new ObservationLevel Enum for better code readability.

This example shows how to use the descriptive enum values instead of magic numbers.
"""

from PIL import Image
import gymnasium as gym

from miniworld_maze import ObservationLevel
import miniworld_maze  # noqa: F401


def main():
    print("🎮 ObservationLevel Enum Demonstration")
    print("=" * 50)

    # Show available observation levels
    print("Available observation levels:")
    for level in ObservationLevel:
        print(f"  {level.name}: {level.description}")

    print("\n" + "=" * 50)
    print("Creating environments with different observation levels...")

    # Test each observation level
    for obs_level in ObservationLevel:
        print(f"\n🔍 Testing {obs_level.name}...")

        # Create environment with specific observation level
        env = gym.make(
            "NineRooms-v0",
            obs_level=obs_level,  # Using descriptive enum!
            obs_width=64,
            obs_height=64,
        )

        # Reset and get observation
        obs, info = env.reset(seed=42)
        obs_array = obs["observation"] if isinstance(obs, dict) else obs
        print(f"   ✅ Observation shape: {obs_array.shape}")

        # Save sample observation (already in HWC format)
        obs_hwc = obs_array
        filename = f"obs_level_{obs_level.name.lower()}.png"
        Image.fromarray(obs_hwc).save(filename)
        print(f"   💾 Saved sample to: {filename}")

        env.close()

    print("\n" + "=" * 50)
    print("🎉 Demonstration complete!")
    print("Compare the files to see the different observation types.")

    # Show the difference between old and new API
    print("\n📝 API Comparison:")
    print("OLD (magic numbers):    obs_level=1")
    print("NEW (descriptive enum): obs_level=ObservationLevel.TOP_DOWN_PARTIAL")
    print("")
    print("Benefits of the new Enum approach:")
    print("  ✅ Self-documenting code")
    print("  ✅ IDE autocompletion")
    print("  ✅ Type safety")
    print("  ✅ Descriptive error messages")
    print("  ✅ Backward compatible with integers")


if __name__ == "__main__":
    main()
