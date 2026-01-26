"""FourRooms environment implementation."""

from typing import List, Union

from ..core import ObservationLevel
from ..core.constants import TextureThemes
from .base_grid_rooms import GridRoomsEnvironment


class FourRooms(GridRoomsEnvironment):
    """
    Traverse the 4 rooms

    ---------
    | 0 | 1 |
    ---------
    | 2 | 3 |
    ---------
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
        **kwargs,
    ):
        # Default configuration for FourRooms
        default_connections = [
            (0, 1),  # Top row
            (0, 2),  # Left column
            (1, 3),  # Right column
            (2, 3),  # Bottom row
        ]
        default_textures = TextureThemes.FOUR_ROOMS

        # Initialize goal positions for each room (1 goal per room)
        # Using the original drstrategy formula with -0.5 offset
        goal_positions = self._generate_four_rooms_goal_positions(2, room_size)

        super().__init__(
            grid_size=2,
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
            **kwargs,
        )

    @staticmethod
    def _generate_four_rooms_goal_positions(
        grid_size: int, room_size: Union[int, float]
    ) -> List[List[List[float]]]:
        """
        Generate goal positions matching the original drstrategy implementation.

        Original formula from drstrategy/envs.py:
        - Goal: [room_size*(j + 0.5) - 0.5, 0.0, room_size*(i + 0.5) - 0.5]

        The offset is a fixed -0.5 value as per the original implementation,
        regardless of room_size.

        Args:
            grid_size: Size of the grid (2 for 2x2)
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
