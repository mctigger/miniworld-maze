"""
MiniWorld DrStrategy - Nine Rooms Environment Package

This package contains the complete implementation of the Nine Rooms environment
variants used in the DrStrategy paper, along with tools for generating observations.

Available environment variants:
- NineRooms: Classic 3x3 grid of rooms
- SpiralNineRooms: 3x3 grid with spiral connections
- TwentyFiveRooms: Large 5x5 grid with 40 connections

Main modules:
- environments: Environment implementations
- wrappers: Gymnasium wrappers for PyTorch compatibility
- tools: Observation generation and utilities
"""

from .environments.factory import create_nine_rooms_env, NineRoomsEnvironmentWrapper
from .environments.nine_rooms import NineRooms
from .environments.spiral_nine_rooms import SpiralNineRooms
from .environments.twenty_five_rooms import TwentyFiveRooms

__version__ = "1.0.0"
__all__ = [
    "create_nine_rooms_env",
    "NineRoomsEnvironmentWrapper", 
    "NineRooms",
    "SpiralNineRooms", 
    "TwentyFiveRooms"
]