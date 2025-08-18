"""Base Grid Rooms environment implementation."""

from gymnasium import spaces
from ..core import CustomMiniWorldEnv, Box, COLORS


class GridRoomsEnvironment(CustomMiniWorldEnv):
    """
    Base class for grid-based room environments.
    
    Supports different grid sizes and connection patterns.
    """
    
    # Default configurations for different variants
    VARIANT_CONFIGS = {
        'nine_rooms': {
            'grid_size': 3,
            'connections': [(0,1), (0,3), (1,2), (1,4), (2,5), (3,4), (3,6), (4,5), (4,7), (5,8), (6,7), (7,8)],
            'textures': ['beige','lightbeige', 'lightgray',
                        'copperred', 'skyblue', 'lightcobaltgreen',
                        'oakbrown', 'navyblue', 'cobaltgreen']
        },
        'spiral_nine_rooms': {
            'grid_size': 3,
            'connections': [(0,1), (0,3), (1,2), (2,5), (3,6), (4,5), (6,7), (7,8)],
            'textures': ['beige','lightbeige', 'lightgray',
                        'copperred', 'skyblue', 'lightcobaltgreen',
                        'oakbrown', 'navyblue', 'cobaltgreen']
        },
        'twenty_five_rooms': {
            'grid_size': 5,
            'connections': [(0,1), (0,5), (1,2), (1,6), (2,3), (2,7), (3,4), (3,8), (4,9),
                           (5,6), (5,10), (6,7), (6,11), (7,8), (7,12), (8,9), (8,13), (9,14),
                           (10,11), (10,15), (11,12), (11,16), (12,13), (12,17), (13,14), (13,18), (14,19),
                           (15,16), (15,20), (16,17), (16,21), (17,18), (17,22), (18,19), (18,23), (19,24),
                           (20,21), (21,22), (22,23), (23,24)],
            'textures': ['crimson','beanpaste', 'cobaltgreen', 'lightnavyblue', 'skyblue', 
                        'lightcobaltgreen','oakbrown', 'copperred', 'lightgray', 'lime',
                        'turquoise', 'violet', 'beige', 'morningglory', 'silver',
                        'magenta','sunnyyellow', 'blueberry', 'lightbeige', 'seablue',
                        'lemongrass', 'orchid', 'redbean', 'orange', 'realblueberry']
        }
    }

    def __init__(self, variant=None, grid_size=None, connections=None, textures=None, 
                 placed_room=None, obs_level=1, continuous=False, room_size=5, door_size=2,
                 agent_mode=None, **kwargs):
        
        # If variant is specified, use its default configuration
        if variant is not None:
            if variant not in self.VARIANT_CONFIGS:
                raise ValueError(f"Unknown variant '{variant}'. Available variants: {list(self.VARIANT_CONFIGS.keys())}")
            
            config = self.VARIANT_CONFIGS[variant]
            self.grid_size = grid_size or config['grid_size']
            default_connections = config['connections']
            default_textures = config['textures']
        else:
            # Manual configuration
            if grid_size is None:
                raise ValueError("Either 'variant' or 'grid_size' must be specified")
            self.grid_size = grid_size
            default_connections = None
            default_textures = None

        # Calculate total number of rooms
        self.total_rooms = self.grid_size * self.grid_size
        
        # Set connections
        if connections is None:
            if default_connections is None:
                raise ValueError("Connections must be specified when not using a predefined variant")
            self.connections = default_connections
        else:
            assert len(connections) > 0, "Connection between rooms should be more than 1"
            self.connections = connections
        
        # Set textures
        if textures is None:
            if default_textures is None:
                raise ValueError("Textures must be specified when not using a predefined variant")
            self.textures = default_textures
        else:
            assert len(textures) == self.total_rooms, f"Textures for floor should be same as the number of the rooms ({self.total_rooms})"
            self.textures = textures
        
        # Set placed room
        if placed_room is None:
            self.placed_room = 0
        else:
            assert 0 <= placed_room < self.total_rooms, f"placing point should be in 0~{self.total_rooms-1}"
            self.placed_room = placed_room
        
        # Set agent mode
        if agent_mode is None:
            self.agent_mode = 'empty'
        else:
            assert agent_mode in ['triangle', 'circle', 'empty'], "configuration cannot be done"
            self.agent_mode = agent_mode

        self.room_size = room_size
        self.door_size = door_size
        
        super().__init__(
            obs_level=obs_level,
            max_episode_steps=1000,
            continuous=continuous,
            agent_mode=self.agent_mode,
            **kwargs
        )

        if not self.continuous:
            self.action_space = spaces.Discrete(self.actions.move_forward+1)
    
    def _gen_world(self, pos=None):
        rooms = []
        
        # Create rooms in grid layout
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rooms.append(self.add_rect_room(
                    min_x=self.room_size*j,
                    max_x=self.room_size*(j+0.95), 
                    min_z=self.room_size*i,
                    max_z=self.room_size*(i+0.95), 
                    floor_tex=self.textures[self.grid_size*i+j]
                ))
        
        # Connect rooms based on connection list
        for connection in self.connections:
            if rooms[connection[0]].mid_x == rooms[connection[1]].mid_x:
                self.connect_rooms(
                    rooms[connection[0]], rooms[connection[1]], 
                    min_x=rooms[connection[0]].mid_x-self.door_size,
                    max_x=rooms[connection[0]].mid_x+self.door_size
                )
            else:
                self.connect_rooms(
                    rooms[connection[0]], rooms[connection[1]], 
                    min_z=rooms[connection[0]].mid_z-self.door_size,
                    max_z=rooms[connection[0]].mid_z+self.door_size
                )
        
        # Place agent
        if pos is None:
            self.place_agent(pos=[2.5, 0, 2.5])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        # Place box entities in each room
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                _start_x = self.room_size*j
                _start_y = self.room_size*i
                for k in range(9):
                    self.place_entity(
                        ent=Box(
                            list(COLORS.keys())[(k+1+(i+1)*(j+1))%9],
                            transparentable=True,
                            size=2*self.room_size/15
                        ),
                        pos=[
                            _start_x + (self.room_size/3)*(k%3) + 0.16*self.room_size,
                            0,
                            _start_y + (self.room_size/3)*(k//3) + 0.16*self.room_size
                        ],
                        dir=0
                    )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info