"""
MiniWorld DrStrategy - Multi-Room Maze Environment Package

This package contains complete implementations of multi-room maze environment
variants used in the DrStrategy paper, along with tools for generating observations.

Available environment variants:
- NineRooms: Classic 3x3 grid of rooms (9 rooms total)
- SpiralNineRooms: 3x3 grid with spiral connections (9 rooms total)
- TwentyFiveRooms: Large 5x5 grid with 40 connections (25 rooms total)

Main modules:
- environments: Environment implementations
- wrappers: Gymnasium wrappers for PyTorch compatibility
- tools: Observation generation and utilities
"""

import os
import warnings

# Set PYGLET_HEADLESS=1 by default if not already set
if "PYGLET_HEADLESS" not in os.environ:
    os.environ["PYGLET_HEADLESS"] = "1"
    warnings.warn(
        "Automatically set PYGLET_HEADLESS=1 for headless rendering. "
        "Set PYGLET_HEADLESS=0 before importing miniworld_maze to override this behavior.",
        UserWarning,
        stacklevel=2,
    )

from .core import ObservationLevel
from .environments.nine_rooms import NineRooms
from .environments.spiral_nine_rooms import SpiralNineRooms
from .environments.twenty_five_rooms import TwentyFiveRooms
from .environments.spiral_twenty_five_rooms import SpiralTwentyFiveRooms

__all__ = [
    "NineRooms",
    "SpiralNineRooms",
    "TwentyFiveRooms",
    "SpiralTwentyFiveRooms",
    "ObservationLevel",
]
