import random
import numpy as np
import math

from gymnasium import spaces, utils
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box

# Define common colors (using Farama-compatible colors)
COLORS = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']

class OneRoom(MiniWorldEnv, utils.EzPickle):
    """
    Traverse a room
    -----
    | 0 |
    -----
    """

    def __init__(self, connections=None, textures=None, placed_room=None, obs_level=1, 
                 continuous=False, room_size=5, door_size=2, agent_mode=None, **kwargs):

        if connections is None:
            self.connections = []
        else:
            assert len(connections) == 0, "Connection should be exactly 0"
            self.connections = connections
        
        if textures is None:
            self.textures = ['white']
        else:
            assert len(textures) == 1, "Textures for floor should be same as the number of the rooms"
            self.textures = textures
        
        if placed_room is None:
            self.placed_room = 0
        else:
            assert (placed_room == 0), "placing point should be in 0"
            self.placed_room = placed_room

        if agent_mode is None:
            self.agent_mode = 'empty'
        else:
            assert agent_mode in ['triangle', 'circle', 'empty'], "configuration cannot be done"
            self.agent_mode = agent_mode

        self.room_size = room_size
        self.door_size = door_size

        MiniWorldEnv.__init__(
            self,
            max_episode_steps=1000,
            **kwargs
        )
        utils.EzPickle.__init__(self, connections=connections, textures=textures, 
                               placed_room=placed_room, obs_level=obs_level,
                               continuous=continuous, room_size=room_size, 
                               door_size=door_size, agent_mode=agent_mode, **kwargs)

        if not continuous:
            self.action_space = spaces.Discrete(self.actions.move_forward+1)
    
    def _gen_world(self, pos=None):
        rooms = []
        room = self.add_rect_room(
            min_x=0, max_x=self.room_size*0.95, 
            min_z=0, max_z=self.room_size*0.95, 
            floor_tex=self.textures[0]
        )
        rooms.append(room)
        
        if pos is None:
            self.place_agent(pos=[2.5, 0, 2.5])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        for k in range(9):
            color_name = COLORS[k % len(COLORS)]
            self.place_entity(
                Box(color=color_name, size=2*self.room_size/15),
                pos=[(self.room_size/3)*(k%3)+0.16*self.room_size, 0, 
                     (self.room_size/3)*(k//3)+0.16*self.room_size]
            )

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        return obs, reward, termination, truncation, info


class TwoRoomsVer1(MiniWorldEnv, utils.EzPickle):
    """
    Traverse the 2 rooms
    ---------
    | 0 | 1 |
    ---------
    """

    def __init__(self, connections=None, textures=None, placed_room=None, obs_level=1, 
                 continuous=False, room_size=5, door_size=2, agent_mode=None, **kwargs):

        if connections is None:
            self.connections = [(0,1)]
        else:
            assert len(connections) > 0, "Connection between rooms should be more than 1"
            self.connections = connections
        
        if textures is None:
            self.textures = ['wood', 'white']
        else:
            assert len(textures) == 2, "Textures for floor should be same as the number of the rooms"
            self.textures = textures
        
        if placed_room is None:
            self.placed_room = 0
        else:
            assert 0 <= placed_room < 2, "placing point should be in 0~1"
            self.placed_room = placed_room
        
        if agent_mode is None:
            self.agent_mode = 'empty'
        else:
            assert agent_mode in ['triangle', 'circle', 'empty'], "configuration cannot be done"
            self.agent_mode = agent_mode

        self.room_size = room_size
        self.door_size = door_size

        MiniWorldEnv.__init__(
            self,
            max_episode_steps=1000,
            **kwargs
        )
        utils.EzPickle.__init__(self, connections=connections, textures=textures, 
                               placed_room=placed_room, obs_level=obs_level,
                               continuous=continuous, room_size=room_size, 
                               door_size=door_size, agent_mode=agent_mode, **kwargs)

        if not continuous:
            self.action_space = spaces.Discrete(self.actions.move_forward+1)
    
    def _gen_world(self, pos=None):
        rooms = []
        for j in range(2):
            room = self.add_rect_room(
                min_x=self.room_size*j, max_x=self.room_size*(j+0.95), 
                min_z=0, max_z=self.room_size*0.95, 
                floor_tex=self.textures[j]
            )
            rooms.append(room)
        
        for connection in self.connections:
            self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                              min_z=rooms[connection[0]].mid_z-self.door_size, 
                              max_z=rooms[connection[0]].mid_z+self.door_size)
        
        if pos is None:
            self.place_agent(pos=[2.5, 0, 2.5])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        for j in range(2):
            start_x = self.room_size*j
            for k in range(9):
                color_name = COLORS[(k+1+(j+1)) % len(COLORS)]
                self.place_entity(
                    Box(color=color_name, size=2*self.room_size/15),
                    pos=[start_x + (self.room_size/3)*(k%3)+0.16*self.room_size, 0, 
                         (self.room_size/3)*(k//3)+0.16*self.room_size]
                )

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        return obs, reward, termination, truncation, info


class ThreeRooms(MiniWorldEnv, utils.EzPickle):
    """
    Traverse the 3 rooms
    -------------
    | 0 | 1 | 2 |
    -------------
    """
    
    def __init__(self, connections=None, textures=None, placed_room=None, obs_level=1, 
                 continuous=False, room_size=5, door_size=2, agent_mode=None, **kwargs):

        if connections is None:
            self.connections = [(0,1), (1,2)]
        else:
            assert len(connections) > 0, "Connection between rooms should be more than 1"
            self.connections = connections
        
        if textures is None:
            self.textures = ['wood', 'white', 'concrete']
        else:
            assert len(textures) == 3, "Textures for floor should be same as the number of the rooms"
            self.textures = textures
        
        if placed_room is None:
            self.placed_room = 0
        else:
            assert 0 <= placed_room < 3, "placing point should be in 0~2"
            self.placed_room = placed_room
        
        if agent_mode is None:
            self.agent_mode = 'empty'
        else:
            assert agent_mode in ['triangle', 'circle', 'empty'], "configuration cannot be done"
            self.agent_mode = agent_mode

        self.room_size = room_size
        self.door_size = door_size

        MiniWorldEnv.__init__(
            self,
            max_episode_steps=1000,
            **kwargs
        )
        utils.EzPickle.__init__(self, connections=connections, textures=textures, 
                               placed_room=placed_room, obs_level=obs_level,
                               continuous=continuous, room_size=room_size, 
                               door_size=door_size, agent_mode=agent_mode, **kwargs)

        if not continuous:
            self.action_space = spaces.Discrete(self.actions.move_forward+1)
    
    def _gen_world(self, pos=None):
        rooms = []
        for j in range(3):
            room = self.add_rect_room(
                min_x=self.room_size*j, max_x=self.room_size*(j+0.95), 
                min_z=0, max_z=self.room_size*0.95, 
                floor_tex=self.textures[j]
            )
            rooms.append(room)
        
        for connection in self.connections:
            self.connect_rooms(rooms[connection[0]], rooms[connection[1]], 
                              min_z=rooms[connection[0]].mid_z-self.door_size, 
                              max_z=rooms[connection[0]].mid_z+self.door_size)
        
        if pos is None:
            self.place_agent(pos=[2.5, 0, 2.5])
        else:
            self.place_agent(pos=[pos[0], 0, pos[1]])

        for j in range(3):
            start_x = self.room_size*j
            for k in range(9):
                color_name = COLORS[(k+1+(j+1)) % len(COLORS)]
                self.place_entity(
                    Box(color=color_name, size=2*self.room_size/15),
                    pos=[start_x + (self.room_size/3)*(k%3)+0.16*self.room_size, 0, 
                         (self.room_size/3)*(k//3)+0.16*self.room_size]
                )

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        return obs, reward, termination, truncation, info