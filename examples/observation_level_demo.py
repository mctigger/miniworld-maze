#!/usr/bin/env python3
"""
Demonstration of the new ObservationLevel Enum for better code readability.

This example shows how to use the descriptive enum values instead of magic numbers.
"""

import numpy as np
from miniworld_drstrategy import ObservationLevel, create_nine_rooms_env
from PIL import Image


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
        env = create_nine_rooms_env(
            variant="NineRooms",
            obs_level=obs_level,  # Using descriptive enum!
            size=64,
        )

        # Reset and get observation
        obs, info = env.reset(seed=42)
        print(f"   ✅ Observation shape: {obs.shape}")

        # Save sample observation
        obs_hwc = np.transpose(obs, (1, 2, 0))
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
