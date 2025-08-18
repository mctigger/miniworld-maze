"""NineRooms environment implementation."""

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

    def __init__(self, connections=None, textures=None, placed_room=None, 
                 obs_level=1, continuous=False, room_size=5, door_size=2,
                 agent_mode=None, **kwargs):
        
        super().__init__(
            variant='nine_rooms',
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