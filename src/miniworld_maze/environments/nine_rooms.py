"""NineRooms environment implementation."""

from typing import List, Union

from ..core import ObservationLevel
from ..core.constants import TextureThemes
from .base_grid_rooms import GridRoomsEnvironment


class NineRooms(GridRoomsEnvironment):
    """
    Traverse the 9 rooms

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
        move_scale: float = 1.0,
        room_size=15,
        door_size=2.5,
        agent_mode=None,
        obs_width=80,
        obs_height=80,
        goal_threshold=0.1,
        **kwargs,
    ):
        # Default configuration for NineRooms
        default_connections = [
            (0, 1),
            (0, 3),
            (1, 2),
            (1, 4),
            (2, 5),
            (3, 4),
            (3, 6),
            (4, 5),
            (4, 7),
            (5, 8),
            (6, 7),
            (7, 8),
        ]
        default_textures = TextureThemes.NINE_ROOMS

        # Initialize goal positions for each room (2 goals per room)
        # Using the original drstrategy formula
        goal_positions = self._generate_nine_rooms_goal_positions(3, room_size)

        super().__init__(
            grid_size=3,
            connections=connections or default_connections,
            textures=textures or default_textures,
            goal_positions=goal_positions,
            placed_room=placed_room,
            obs_level=obs_level,
            continuous=continuous,
            move_scale=move_scale,
            room_size=room_size,
            door_size=door_size,
            agent_mode=agent_mode,
            obs_width=obs_width,
            obs_height=obs_height,
            goal_threshold=goal_threshold,
            **kwargs,
        )

    @staticmethod
    def _generate_nine_rooms_goal_positions(
        grid_size: int, room_size: Union[int, float]
    ) -> List[List[List[float]]]:
        """
        Generate goal positions matching the original drstrategy implementation.

        Original formula from drstrategy/envs.py:
        - Goal 1: [room_size*(j + 0.5) - 0.5, 0.0, room_size*(i + 0.5) - 0.5]
        - Goal 2: [room_size*(j + 0.3) - 0.5, 0.0, room_size*(i + 0.7) - 0.5]

        The offset is a fixed -0.5 value as per the original implementation,
        regardless of room_size.

        Args:
            grid_size: Size of the grid (e.g., 3 for 3x3)
            room_size: Size of each room

        Returns:
            List of goal positions for each room
        """
        goal_positions = []
        # Fixed offset matching original drstrategy implementation
        offset = 0.5

        for i in range(grid_size):  # rows
            for j in range(grid_size):  # columns
                goal_positions.append(
                    [
                        # Goal 1: near center with small offset
                        [
                            room_size * (j + 0.5) - offset,
                            0.0,
                            room_size * (i + 0.5) - offset,
                        ],
                        # Goal 2: asymmetric position (0.3, 0.7 of room)
                        [
                            room_size * (j + 0.3) - offset,
                            0.0,
                            room_size * (i + 0.7) - offset,
                        ],
                    ]
                )
        return goal_positions
