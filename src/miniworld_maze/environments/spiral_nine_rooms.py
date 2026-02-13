"""SpiralNineRooms environment implementation."""

from ..core import ObservationLevel
from ..core.constants import TextureThemes
from .base_grid_rooms import GridRoomsEnvironment
from .nine_rooms import NineRooms


class SpiralNineRooms(GridRoomsEnvironment):
    """
    Traverse the 9 rooms in spiral

    -------------
    | 0 | 1 | 2 |
    -------------
    | 3 | 4 | 5 |
    -------------
    | 6 | 7 | 8 |
    -------------
    """

    def __init__(
        self,
        connections=None,
        textures=None,
        placed_room=None,
        obs_level=ObservationLevel.TOP_DOWN_PARTIAL,
        continuous=False,
        room_size=15,
        door_size=2.5,
        agent_mode=None,
        obs_width=80,
        obs_height=80,
        goal_threshold=0.1,
        **kwargs,
    ):
        # Default configuration for SpiralNineRooms
        default_connections = [
            (0, 1),
            (0, 3),
            (1, 2),
            (2, 5),
            (3, 6),
            (4, 5),
            (6, 7),
            (7, 8),
        ]
        default_textures = TextureThemes.SPIRAL_NINE_ROOMS

        # Initialize goal positions for each room (2 goals per room)
        # Using the same formula as NineRooms to match original drstrategy
        goal_positions = NineRooms._generate_nine_rooms_goal_positions(3, room_size)

        super().__init__(
            grid_size=3,
            connections=connections or default_connections,
            textures=textures or default_textures,
            goal_positions=goal_positions,
            placed_room=placed_room,
            obs_level=obs_level,
            continuous=continuous,
            room_size=room_size,
            door_size=door_size,
            agent_mode=agent_mode,
            obs_width=obs_width,
            obs_height=obs_height,
            goal_threshold=goal_threshold,
            **kwargs,
        )
