"""
Nine Rooms DrStrategy Environment

This package provides a faithful port of the Nine Rooms environment from the 
original DrStrategy implementation, adapted for modern Gymnasium and providing
2D top-down observations without MiniWorld dependencies.
"""

from gymnasium.envs.registration import register
from .nine_rooms_env import NineRoomsDrStrategyEnv, SpiralNineRoomsDrStrategyEnv

# Register environments
register(
    id='NineRoomsDrStrategy-v0',
    entry_point='nine_rooms_drstrategy.nine_rooms_env:NineRoomsDrStrategyEnv',
    max_episode_steps=1000,
)

register(
    id='SpiralNineRoomsDrStrategy-v0', 
    entry_point='nine_rooms_drstrategy.nine_rooms_env:SpiralNineRoomsDrStrategyEnv',
    max_episode_steps=1000,
)

__version__ = "1.0.0"
__all__ = ["NineRoomsDrStrategyEnv", "SpiralNineRoomsDrStrategyEnv"]