"""SpiralNineRooms environment implementation."""

from gymnasium import spaces
from ..core import CustomMiniWorldEnv, Box, COLORS


class SpiralNineRooms(CustomMiniWorldEnv):
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

    def __init__(self, connections: list=None, textures: list=None, placed_room=None, 
                 obs_level: int=1, continuous=False, room_size=5, door_size=2,
                 agent_mode: str=None, **kwargs):

        if connections is None:
            self.connections = [(0,1), (0,3), (1,2), (2,5), (3,6), (4,5), (6,7), (7,8)]
        else:
            assert len(connections) > 0, "Connection between rooms should be more than 1"
            self.connections = connections
        
        if textures is None:
            self.textures = ['beige','lightbeige', 'lightgray',
                             'copperred', 'skyblue', 'lightcobaltgreen',
                             'oakbrown', 'navyblue', 'cobaltgreen']
        else:
            assert len(textures) == 9, "Textures for floor should be same as the number of the rooms"
            self.textures = textures
        
        if placed_room is None:
            self.placed_room = 0
        else:
            assert ((placed_room > -1) or (placed_room < 9)), "placing point should be in 0~8"
            self.placed_room = placed_room
        
        if agent_mode is None:
            self.agent_mode='empty'
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
        for i in range(3):
            for j in range(3):
                rooms.append(self.add_rect_room(min_x=self.room_size*j,max_x=self.room_size*(j+0.95), 
                                                min_z=self.room_size*i, max_z=self.room_size*(i+0.95), 
                                                floor_tex=self.textures[3*i+j]))
        
        for connection in self.connections:
            if rooms[connection[0]].mid_x == rooms[connection[1]].mid_x:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_x=rooms[connection[0]].mid_x-self.door_size, max_x=rooms[connection[0]].mid_x+self.door_size)
            else:
                self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                                   min_z=rooms[connection[0]].mid_z-self.door_size, max_z=rooms[connection[0]].mid_z+self.door_size)
        
        if pos is None:
            self.place_agent(pos=[2.5, 0, 2.5])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        for i in range(3):
            for j in range(3):
                _start_x = self.room_size*j
                _start_y = self.room_size*i
                for k in range(9):
                    self.place_entity(
                        ent=Box(list(COLORS.keys())[(k+1+(i+1)*(j+1))%9], transparentable=True, size=2*self.room_size/15),
                        pos=[_start_x+ (self.room_size/3)*(k%3)+0.16*self.room_size, 0, _start_y + (self.room_size/3)*(k//3)+0.16*self.room_size],
                        dir=0
                    )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info