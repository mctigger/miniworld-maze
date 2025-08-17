"""
MiniWorld DrStrategy - Nine Rooms Environment Package

This package contains the complete implementation of the Nine Rooms environment
variants used in the DrStrategy paper, along with tools for generating observations.

Available environment variants:
- NineRooms: Classic 3x3 grid of rooms
- SpiralNineRooms: 3x3 grid with spiral connections
- TwentyFiveRooms: Large 5x5 grid with 40 connections

Main modules:
- miniworld_gymnasium: Core environment implementation
- nine_rooms_factory: Factory for creating environment instances
- generate_observations: Tool for generating comprehensive observation datasets
"""

from .nine_rooms_factory import create_nine_rooms_env, NineRoomsEnvironmentWrapper

__version__ = "1.0.0"
__all__ = [
    "create_nine_rooms_env", 
    "NineRoomsEnvironmentWrapper"
]