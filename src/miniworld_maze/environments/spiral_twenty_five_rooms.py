"""SpiralTwentyFiveRooms environment implementation."""

from ..core import ObservationLevel
from ..core.constants import (
    ROOM_BOUNDARY_MARGIN,
    TextureThemes,
)
from .base_grid_rooms import GridRoomsEnvironment
from .twenty_five_rooms import TwentyFiveRooms


class SpiralTwentyFiveRooms(GridRoomsEnvironment):
    """
    Traverse the 25 rooms in spiral pattern

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
        move_scale: float = 1.0,
        room_size=15,
        door_size=2.5,
        agent_mode=None,
        obs_width=80,
        obs_height=80,
        goal_threshold=0.1,
        **kwargs,
    ):
        # Default configuration for SpiralTwentyFiveRooms (spiral connections)
        default_connections = [
            (12, 13),
            (8, 13),
            (7, 8),
            (6, 7),
            (6, 11),
            (11, 16),
            (16, 17),
            (17, 18),
            (18, 19),
            (14, 19),
            (9, 14),
            (4, 9),
            (3, 4),
            (2, 3),
            (1, 2),
            (0, 1),
            (0, 5),
            (5, 10),
            (10, 15),
            (15, 20),
            (20, 21),
            (21, 22),
            (22, 23),
            (23, 24),
        ]
        default_textures = TextureThemes.SPIRAL_TWENTY_FIVE_ROOMS

        # Store room_size for use in _generate_world_layout
        self._spiral_room_size = room_size

        # Initialize goal positions for each room (1 goal per room)
        # Using the same formula as TwentyFiveRooms
        goal_positions = TwentyFiveRooms._generate_twenty_five_rooms_goal_positions(
            5, room_size
        )

        super().__init__(
            grid_size=5,
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

    def _generate_world_layout(self, pos=None):
        """
        Generate world layout with custom agent start position.

        SpiralTwentyFiveRooms uses a unique start position in room 4:
        Original formula: [room_size*4 + room_size - 2.5, 0, 2.5]
        With room_size=15: [72.5, 0, 2.5]
        """
        rooms = []

        # Create rooms in grid layout
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rooms.append(
                    self.add_rect_room(
                        min_x=self.room_size * j,
                        max_x=self.room_size * (j + (1 - ROOM_BOUNDARY_MARGIN)),
                        min_z=self.room_size * i,
                        max_z=self.room_size * (i + (1 - ROOM_BOUNDARY_MARGIN)),
                        floor_tex=self.textures[self.grid_size * i + j],
                    )
                )

        # Connect rooms based on connection list
        for connection in self.connections:
            if rooms[connection[0]].mid_x == rooms[connection[1]].mid_x:
                self.connect_rooms(
                    rooms[connection[0]],
                    rooms[connection[1]],
                    min_x=rooms[connection[0]].mid_x - self.door_size,
                    max_x=rooms[connection[0]].mid_x + self.door_size,
                )
            else:
                self.connect_rooms(
                    rooms[connection[0]],
                    rooms[connection[1]],
                    min_z=rooms[connection[0]].mid_z - self.door_size,
                    max_z=rooms[connection[0]].mid_z + self.door_size,
                )

        # Place agent with unique start position for SpiralTwentyFiveRooms
        # Original formula: [room_size*4 + room_size - 2.5, 0, 2.5]
        if pos is None:
            spiral_start_x = self._spiral_room_size * 4 + self._spiral_room_size - 2.5
            spiral_start_z = 2.5
            self.place_agent(pos=[spiral_start_x, 0, spiral_start_z])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        # Place box entities in each room in a 3x3 grid pattern
        self._place_room_boxes()
