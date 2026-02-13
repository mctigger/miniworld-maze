"""TwentyFiveRooms environment implementation."""

from typing import List, Union

from ..core import ObservationLevel
from ..core.constants import TextureThemes
from .base_grid_rooms import GridRoomsEnvironment


class TwentyFiveRooms(GridRoomsEnvironment):
    """
    Traverse the 25 rooms

    ---------------------
    | 0 | 1 | 2 | 3 | 4 |
    ---------------------
    | 5 | 6 | 7 | 8 | 9 |
    ---------------------
    |10 |11 |12 |13 |14 |
    ---------------------
    |15 |16 |17 |18 |19 |
    ---------------------
    |20 |21 |22 |23 |24 |
    ---------------------
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
        # Default configuration for TwentyFiveRooms
        default_connections = [
            (0, 1),
            (0, 5),
            (1, 2),
            (1, 6),
            (2, 3),
            (2, 7),
            (3, 4),
            (3, 8),
            (4, 9),
            (5, 6),
            (5, 10),
            (6, 7),
            (6, 11),
            (7, 8),
            (7, 12),
            (8, 9),
            (8, 13),
            (9, 14),
            (10, 11),
            (10, 15),
            (11, 12),
            (11, 16),
            (12, 13),
            (12, 17),
            (13, 14),
            (13, 18),
            (14, 19),
            (15, 16),
            (15, 20),
            (16, 17),
            (16, 21),
            (17, 18),
            (17, 22),
            (18, 19),
            (18, 23),
            (19, 24),
            (20, 21),
            (21, 22),
            (22, 23),
            (23, 24),
        ]
        default_textures = TextureThemes.TWENTY_FIVE_ROOMS

        # Initialize goal positions for each room (1 goal per room)
        # Using the original drstrategy formula with -0.5 offset
        goal_positions = self._generate_twenty_five_rooms_goal_positions(5, room_size)

        super().__init__(
            grid_size=5,
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

    @staticmethod
    def _generate_twenty_five_rooms_goal_positions(
        grid_size: int, room_size: Union[int, float]
    ) -> List[List[List[float]]]:
        """
        Generate goal positions matching the original drstrategy implementation.

        Original formula from drstrategy/envs.py:
        - Goal: [room_size*(j + 0.5) - 0.5, 0.0, room_size*(i + 0.5) - 0.5]

        The offset is a fixed -0.5 value as per the original implementation,
        regardless of room_size.

        Args:
            grid_size: Size of the grid (e.g., 5 for 5x5)
            room_size: Size of each room

        Returns:
            List of goal positions for each room (1 goal per room)
        """
        goal_positions = []
        # Fixed offset matching original drstrategy implementation
        offset = 0.5

        for i in range(grid_size):  # rows
            for j in range(grid_size):  # columns
                goal_positions.append(
                    [
                        # Single goal near center with small offset
                        [
                            room_size * (j + 0.5) - offset,
                            0.0,
                            room_size * (i + 0.5) - offset,
                        ],
                    ]
                )
        return goal_positions
