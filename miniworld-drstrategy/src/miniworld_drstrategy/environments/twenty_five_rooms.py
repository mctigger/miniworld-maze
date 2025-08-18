"""TwentyFiveRooms environment implementation."""

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

    def __init__(self, connections=None, textures=None, placed_room=None, 
                 obs_level=1, continuous=False, room_size=5, door_size=2,
                 agent_mode=None, **kwargs):
        
        super().__init__(
            variant='twenty_five_rooms',
            connections=connections,
            textures=textures,
            placed_room=placed_room,
            obs_level=obs_level,
            continuous=continuous,
            room_size=room_size,
            door_size=door_size,
            agent_mode=agent_mode,
            **kwargs
        )