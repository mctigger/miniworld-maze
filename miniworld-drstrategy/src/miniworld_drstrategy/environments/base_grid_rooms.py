"""Base Grid Rooms environment implementation."""

from gymnasium import spaces
from ..core import CustomMiniWorldEnv, Box, COLORS


class GridRoomsEnvironment(CustomMiniWorldEnv):
    """
    Base class for grid-based room environments.
    
    Supports different grid sizes and connection patterns.
    Subclasses pass their specific configurations directly to __init__.
    """

    def __init__(self, grid_size, connections, textures, placed_room=None, 
                 obs_level=1, continuous=False, room_size=5, door_size=2,
                 agent_mode=None, **kwargs):
        """
        Initialize a grid-based room environment.
        
        Args:
            grid_size: Size of the grid (e.g., 3 for 3x3 grid)
            connections: List of (room1, room2) tuples for connections
            textures: List of texture names for each room
            placed_room: Initial room index (defaults to 0)
            obs_level: Observation level (defaults to 1)
            continuous: Whether to use continuous actions (defaults to False)
            room_size: Size of each room in environment units (defaults to 5)
            door_size: Size of doors between rooms (defaults to 2)
            agent_mode: Agent rendering mode ('triangle', 'circle', 'empty')
            **kwargs: Additional arguments passed to parent class
        """
        
        # Set grid configuration
        self.grid_size = grid_size
        self.total_rooms = self.grid_size * self.grid_size
        
        # Validate and set connections
        assert len(connections) > 0, "Connection between rooms should be more than 1"
        self.connections = connections
        
        # Validate and set textures
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