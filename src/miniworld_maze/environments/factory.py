"""Gymnasium environment registrations for Nine Rooms environment variants."""

from gymnasium.envs.registration import register
from ..core import ObservationLevel
from ..core.constants import FACTORY_DOOR_SIZE, FACTORY_ROOM_SIZE

# Register environment variants with factory defaults matching the original wrapper
register(
    id="NineRooms-v0",
    entry_point="miniworld_maze.environments.nine_rooms:NineRooms",
    max_episode_steps=1000,
    kwargs={
        "room_size": FACTORY_ROOM_SIZE,
        "door_size": FACTORY_DOOR_SIZE,
        "obs_level": ObservationLevel.TOP_DOWN_PARTIAL,
        "agent_mode": None,  # becomes "empty" by default
    },
)

register(
    id="SpiralNineRooms-v0", 
    entry_point="miniworld_maze.environments.spiral_nine_rooms:SpiralNineRooms",
    max_episode_steps=1000,
    kwargs={
        "room_size": FACTORY_ROOM_SIZE,
        "door_size": FACTORY_DOOR_SIZE,
        "obs_level": ObservationLevel.TOP_DOWN_PARTIAL,
        "agent_mode": None,
    },
)

register(
    id="TwentyFiveRooms-v0",
    entry_point="miniworld_maze.environments.twenty_five_rooms:TwentyFiveRooms", 
    max_episode_steps=1000,
    kwargs={
        "room_size": FACTORY_ROOM_SIZE,
        "door_size": FACTORY_DOOR_SIZE,
        "obs_level": ObservationLevel.TOP_DOWN_PARTIAL,
        "agent_mode": None,
    },
)
