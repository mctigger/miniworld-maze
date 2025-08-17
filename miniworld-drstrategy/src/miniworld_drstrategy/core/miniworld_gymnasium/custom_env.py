"""Custom MiniWorld environment class with enhanced features."""

import math
from enum import IntEnum
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pyglet
from pyglet.gl import *
from ctypes import POINTER

from .random import RandGen
from .opengl import *
from .objmesh import *
from .entity import *
from .math import *
from .params import DEFAULT_PARAMS
from .room import Room
from .base_env import MiniWorldEnv


class CustomMiniWorldEnv(gym.Env):
    """Base class for MiniWorld environments. Implements the procedural
    world generation and simulation logic.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left or right by a small amount
        turn_left = 0
        turn_right = 1

        # Move forward or back by a small amount
        move_forward = 2
        move_back = 3

        # Pick up or drop an object being carried
        pickup = 4
        drop = 5

        # Toggle/activate an object
        toggle = 6

        # Done completing task
        done = 7

    def __init__(
        self,
        obs_level=3,
        continuous=False,
        agent_mode='cirlce',
        max_episode_steps=1500,
        obs_width=80,
        obs_height=80,
        window_width=800,
        window_height=600,
        params=DEFAULT_PARAMS,
        domain_rand=False
    ):
        # Observation level
        # 1 for 2D POMDP, 2 for 2D MDP, 3 for 3D POMDP
        self.obs_level = obs_level
        # Agent mode
        # triangle, circle and empty
        self.agent_mode = agent_mode

        # continuous or not?
        assert isinstance(continuous, bool), "the feature \"continuous\" should be boolean"
        self.continuous = continuous

        
        if not self.continuous:

            # Action enumeration for this environment
            self.actions = MiniWorldEnv.Actions
            
            # Actions are discrete integer values
            self.action_space = spaces.Discrete(len(self.actions))
        
        else:

            # Actions are continous, speed and the difference of direction

            self.action_space = spaces.Box(
                low=-1,
                high=1,
                shape=(2,),
                dtype=np.float32
            )

        # Observations are RGB images with pixels in [0, 255]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_height, obs_width, 3),
            dtype=np.uint8
        )

        self.reward_range = (-math.inf, math.inf)

        # Maximum number of steps per episode
        self.max_episode_steps = max_episode_steps

        # Simulation parameters, used for domain randomization
        self.params = params

        # Domain randomization enable/disable flag
        self.domain_rand = domain_rand

        # Window for displaying the environment to humans
        self.window = None

        # Invisible window to render into (shadow OpenGL context)
        self.shadow_window = pyglet.window.Window(width=1, height=1, visible=False)

        # Enable depth testing and backface culling
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

        # Frame buffer used to render observations
        self.obs_fb = FrameBuffer(obs_width, obs_height, 8)

        # Frame buffer used for human visualization
        self.vis_fb = FrameBuffer(window_width, window_height, 16)

        self.topdown_fb = None

        # Compute the observation display size
        self.obs_disp_width = 256
        self.obs_disp_height = obs_height * (self.obs_disp_width / obs_width)

        # For displaying text
        self.text_label = pyglet.text.Label(
            font_name="Arial",
            font_size=14,
            multiline=True,
            width=400,
            x = window_width + 5,
            y = window_height - (self.obs_disp_height + 19)
        )

        # Initialize the state
        self.seed()
        self.reset()

    def close(self):
        pass

    def seed(self, seed=None):
        self.rand = RandGen(seed)
        return [seed]

    def reset(self, seed=None, options=None, pos=None):
        """Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """

        # Handle seed for Gymnasium compatibility
        if seed is not None:
            self.seed(seed)

        # Step count since episode start
        self.step_count = 0

        # Create the agent
        self.agent = Agent(mode=self.agent_mode)

        # List of entities contained
        self.entities = []

        # List of rooms in the world
        self.rooms = []

        # Wall segments for collision detection
        # Shape is (N, 2, 3)
        self.wall_segs = []

        # Generate the world
        self._gen_world(pos)

        # Check if domain randomization is enabled or not
        rand = self.rand if self.domain_rand else None

        # Randomize elements of the world (domain randomization)
        self.params.sample_many(rand, self, [
            'black',
            'light_pos',
            'light_color',
            'light_ambient'
        ])

        # Get the max forward step distance
        self.max_forward_step = self.params.get_max('forward_step')

        # Randomize parameters of the entities
        for ent in self.entities:
            ent.randomize(self.params, rand)

        # Compute the min and max x, z extents of the whole floorplan
        
        self.min_x = min([r.min_x for r in self.rooms])
        self.max_x = max([r.max_x for r in self.rooms])
        self.min_z = min([r.min_z for r in self.rooms])
        self.max_z = max([r.max_z for r in self.rooms])

        if self.topdown_fb is None:
            self.topdown_fb = FrameBuffer(20*(int(self.max_x - self.min_x)+1), 20*(int(self.max_z - self.min_z)+1), 8)

        # Generate static data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # Generate the first camera image
        obs = None
        if self.obs_level == 1:
            if self.agent_mode == 'empty':
                obs = self.render_top_view(POMDP=True, render_ag=False)
            else:
                obs = self.render_top_view(POMDP=True)

        elif self.obs_level == 2:
            obs = self.render_top_view(POMDP=False)

        elif self.obs_level == 3:
            obs = self.render_obs()

        else:
            assert ((self.obs_level < 1) or (self.obs_level > 3)), "obs_level should be between 1 and 3"
            import sys
            sys.exit(0)

        # Return first observation with info dict for Gymnasium compatibility
        return obs, {}

    def _get_carry_pos(self, agent_pos, ent):
        """Compute the position at which to place an object being carried"""

        dist = self.agent.radius + ent.radius + self.max_forward_step
        pos = agent_pos + self.agent.dir_vec * 1.05 * dist

        # Adjust the Y-position so the object is visible while being carried
        y_pos = max(self.agent.cam_height - ent.height - 0.3, 0)
        pos = pos + Y_VEC * y_pos

        return pos

    def move_agent(self, fwd_dist, fwd_drift):
        """Move the agent forward"""

        next_pos = (
            self.agent.pos +
            self.agent.dir_vec * fwd_dist +
            self.agent.right_vec * fwd_drift
        )

        if self.intersect(self.agent, next_pos, self.agent.radius):
            return False

        carrying = self.agent.carrying
        if carrying:
            next_carrying_pos = self._get_carry_pos(next_pos, carrying)

            if self.intersect(carrying, next_carrying_pos, carrying.radius):
                return False

            carrying.pos = next_carrying_pos

        self.agent.pos = next_pos

        return True

    def turn_agent(self, turn_angle):
        """Turn the agent left or right"""

        turn_angle *= (math.pi / 180)
        orig_dir = self.agent.dir

        self.agent.dir += turn_angle

        if self.intersect(self.agent, self.agent.pos, self.agent.radius):
            self.agent.dir -= turn_angle
            return False

        carrying = self.agent.carrying

        
        if carrying:
            pos = self._get_carry_pos(self.agent.pos, carrying)

            if self.intersect(carrying, pos, carrying.radius):
                self.agent.dir = orig_dir
                return False

            carrying.pos = pos
            carrying.dir = self.agent.dir

        return True
    
    def turn_and_move_agent(self, fwd_dist, turn_angle):
        orig_dir = self.agent.dir
        self.agent.dir += turn_angle * (math.pi / 180)

        next_pos = (
            self.agent.pos +
            self.agent.dir_vec * fwd_dist
        )
        if self.intersect(self.agent, next_pos, self.agent.radius):
            self.agent.dir = orig_dir
            return False
        else:
            self.agent.pos = next_pos
            

            return True
    
    def pos_agent(self, fwd_dist, angle):
        self.agent.dir = angle * (math.pi / 180)
        next_pos = (
            self.agent.pos + 
            self.agent.dir_vec * fwd_dist
        )
        if self.intersect(self.agent, next_pos, self.agent.radius):
            return False
        else:
            self.agent.pos = next_pos

    def step(self, action):
        """Perform one action and update the simulation"""

        self.step_count += 1

        if self.continuous:
            if self.agent.mode == 'circle':
                self.pos_agent(action[0], 180*action[1])
            else:
                self.turn_and_move_agent(action[0], 15*action[1])
                    
            
            
        else:
            rand = self.rand if self.domain_rand else None
            fwd_step = self.params.sample(rand, 'forward_step')
            fwd_drift = self.params.sample(rand, 'forward_drift')
            turn_step = self.params.sample(rand, 'turn_step')
            
            if action == self.actions.move_forward:
                self.move_agent(fwd_step, fwd_drift)
                
            elif action == self.actions.move_back:
                self.move_agent(-fwd_step, fwd_drift)
            
            elif action == self.actions.turn_left:
                self.turn_agent(turn_step)
            
            elif action == self.actions.turn_right:
                self.turn_agent(-turn_step)
            
            # Pick up an object
            elif action == self.actions.pickup:
                # Position at which we will test for an intersection
                test_pos = self.agent.pos + self.agent.dir_vec * 1.5 * self.agent.radius
                ent = self.intersect(self.agent, test_pos, 1.2 * self.agent.radius)
                if not self.agent.carrying:
                    if isinstance(ent, Entity):
                        if not ent.is_static:
                            self.agent.carrying = ent

            # Drop an object being carried
            elif action == self.actions.drop:
                if self.agent.carrying:
                    self.agent.carrying.pos[1] = 0
                    self.agent.carrying = None
            
            # If we are carrying an object, update its position as we move
            if self.agent.carrying:
                ent_pos = self._get_carry_pos(self.agent.pos, self.agent.carrying)
                self.agent.carrying.pos = ent_pos
                self.agent.carrying.dir = self.agent.dir

        # Generate the current camera image
        obs = None
        topdown = None
        if self.obs_level == 1:
            if self.agent_mode == 'empty':
                obs = self.render_top_view(POMDP=True, render_ag=False)
            else:
                obs = self.render_top_view(POMDP=True)
                
            topdown = self.render_top_view(POMDP=False, frame_buffer=self.topdown_fb)


        elif self.obs_level == 2:
            obs = self.render_top_view(POMDP=False)

        elif self.obs_level == 3:
            obs = self.render_obs()
            topdown = self.render_top_view(POMDP=False, frame_buffer=self.topdown_fb)

        else:
            assert ((self.obs_level < 1) or (self.obs_level > 3)), "obs_level should be between 1 and 3"
            import sys
            sys.exit(0)

        # If the maximum time step count is reached
        if self.step_count >= self.max_episode_steps:
            terminated = True
            reward = 0
            if self.obs_level != 2:
                return obs, reward, terminated, False, {"pos": self.agent.pos, "mdp_view": topdown}
            else:
                return obs, reward, terminated, False, {"pos": self.agent.pos, "mdp_view": obs}

        reward = 0
        terminated = False

        if self.obs_level != 2:
            return obs, reward, terminated, False, {"pos": self.agent.pos, "mdp_view": topdown}
        else:
            return obs, reward, terminated, False, {"pos": self.agent.pos, "mdp_view": obs}

    def add_rect_room(
        self,
        min_x,
        max_x,
        min_z,
        max_z,
        **kwargs
    ):
        """Create a rectangular room"""

        # 2D outline coordinates of the room,
        # listed in counter-clockwise order when viewed from the top
        outline = np.array([
            # East wall
            [max_x, max_z],
            # North wall
            [max_x, min_z],
            # West wall
            [min_x, min_z],
            # South wall
            [min_x, max_z],
        ])

        return self.add_room(outline=outline, **kwargs)

    def add_room(self, **kwargs):
        """Create a new room"""

        assert len(self.wall_segs) == 0, "cannot add rooms after static data is generated"

        room = Room(**kwargs)
        self.rooms.append(room)

        return room

    def connect_rooms(
        self,
        room_a,
        room_b,
        min_x=None,
        max_x=None,
        min_z=None,
        max_z=None,
        max_y=None
    ):
        """Connect two rooms along facing edges"""

        def find_facing_edges():
            for idx_a in range(room_a.num_walls):
                norm_a = room_a.edge_norms[idx_a]

                for idx_b in range(room_b.num_walls):
                    norm_b = room_b.edge_norms[idx_b]

                    # Reject edges that are not facing each other
                    if np.dot(norm_a, norm_b) > -0.9:
                        continue

                    dir = room_b.outline[idx_b] - room_a.outline[idx_a]

                    # Reject edges that are not touching
                    if np.dot(norm_a, dir) > 0.05:
                        continue

                    return idx_a, idx_b

            return None, None

        idx_a, idx_b = find_facing_edges()
        assert idx_a != None, "matching edges not found in connect_rooms"

        start_a, end_a = room_a.add_portal(
            edge=idx_a,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z,
            max_y=max_y
        )

        start_b, end_b = room_b.add_portal(
            edge=idx_b,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z,
            max_y=max_y
        )

        a = room_a.outline[idx_a] + room_a.edge_dirs[idx_a] * start_a
        b = room_a.outline[idx_a] + room_a.edge_dirs[idx_a] * end_a
        c = room_b.outline[idx_b] + room_b.edge_dirs[idx_b] * start_b
        d = room_b.outline[idx_b] + room_b.edge_dirs[idx_b] * end_b

        # If the portals are directly connected, stop
        if np.linalg.norm(a - d) < 0.001:
            return

        len_a = np.linalg.norm(b - a)
        len_b = np.linalg.norm(d - c)

        # Room outline points must be specified in counter-clockwise order
        outline = np.stack([c, b, a, d])
        outline = np.stack([outline[:, 0], outline[:, 2]], axis=1)

        max_y = max_y if max_y != None else room_a.wall_height

        room = Room(
            outline,
            wall_height=max_y,
            wall_tex=room_a.wall_tex_name,
            floor_tex=room_a.floor_tex_name,
            ceil_tex=room_a.ceil_tex_name,
            no_ceiling=room_a.no_ceiling,
        )

        self.rooms.append(room)

        room.add_portal(1, start_pos=0, end_pos=len_a)
        room.add_portal(3, start_pos=0, end_pos=len_b)

    def place_entity(
        self,
        ent,
        room=None,
        pos=None,
        dir=None,
        min_x=None,
        max_x=None,
        min_z=None,
        max_z=None
    ):
        """Place an entity/object in the world.
        Find a position that doesn't intersect with any other object.
        """

        assert len(self.rooms) > 0, "create rooms before calling place_entity"
        assert ent.radius != None, "entity must have physical size defined"

        # Generate collision detection data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # If an exact position if specified
        if pos is not None:
            ent.dir = dir if dir != None else self.rand.float(-math.pi, math.pi)
            ent.pos = pos
            self.entities.append(ent)
            return ent

        # Keep retrying until we find a suitable position
        while True:
            # Pick a room, sample rooms proportionally to floor surface area
            r = room if room else self.rand.choice(self.rooms, probs=self.room_probs)

            # Choose a random point within the square bounding box of the room
            lx = r.min_x if min_x == None else min_x
            hx = r.max_x if max_x == None else max_x
            lz = r.min_z if min_z == None else min_z
            hz = r.max_z if max_z == None else max_z
            
            pos = self.rand.float(
                low =[lx + ent.radius, 0, lz + ent.radius],
                high=[hx - ent.radius, 0, hz - ent.radius]
            )

            # Make sure the position is within the room's outline
            if not r.point_inside(pos):
                continue

            # Pick a direction
            d = dir if dir != None else self.rand.float(-math.pi, math.pi)

            ent.pos = pos
            ent.dir = d

            # Make sure the position doesn't intersect with any walls
            if self.intersect(ent, pos, ent.radius):
                continue

            break

        self.entities.append(ent)

        return ent

    def place_agent(
        self,
        room=None,
        pos=None,
        dir=None,
        min_x=None,
        max_x=None,
        min_z=None,
        max_z=None
    ):
        """Place the agent in the environment at a random position
        and orientation
        """

        return self.place_entity(
            self.agent,
            room=room,
            pos=pos,
            dir=dir,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z
        )

    def intersect(self, ent, pos, radius):
        """Check if an entity intersects with the world"""

        # Ignore the Y position
        px, _, pz = pos
        pos = np.array([px, 0, pz])

        # Check for intersection with walls
        if intersect_circle_segs(pos, radius, self.wall_segs):
            return True

        # Check for entity intersection
        for ent2 in self.entities:
            # Entities can't intersect with themselves
            if ent2 is ent:
                continue

            px, _, pz = ent2.pos
            pos2 = np.array([px, 0, pz])
            
            d = 0
            if (ent.trable  or ent2.trable):
                d = 10000000
            else:
                d = np.linalg.norm(pos2 - pos)
            if d < radius + ent2.radius:
                return ent2

        return None

    def near(self, ent0, ent1=None):
        """Test if the two entities are near each other.
        Used for "go to" or "put next" type tasks
        """

        if ent1 == None:
            ent1 = self.agent

        dist = np.linalg.norm(ent0.pos - ent1.pos)
        return dist < ent0.radius + ent1.radius + 1.1 * self.max_forward_step

    def _load_tex(self, tex_name):
        """Load a texture, with or without domain randomization"""

        rand = self.rand if self.params.sample(self.rand, 'tex_rand') else None
        return Texture.get(tex_name, rand)

    def _gen_static_data(self):
        """Generate static data needed for rendering and collision detection"""

        # Generate the static data for each room
        for room in self.rooms:
            room._gen_static_data(
                self.params,
                self.rand if self.domain_rand else None
            )

        # Concatenate the wall segments
        self.wall_segs = np.concatenate([r.wall_segs for r in self.rooms])

        # Room selection probabilities
        self.room_probs = np.array([r.area for r in self.rooms], dtype=float)
        self.room_probs /= np.sum(self.room_probs)

    def _gen_world(self, pos=None):
        """Generate the world. Derived classes must implement this method."""

        raise NotImplementedError

    def _reward(self):
        """Default sparse reward computation"""

        return 1.0 - 0.2 * (self.step_count / self.max_episode_steps)

    def _render_static(self):
        """Render the static elements of the scene into a display list.
        Called once at the beginning of each episode.
        """

        # TODO: manage this automatically
        # glIsList
        glDeleteLists(1, 1);
        glNewList(1, GL_COMPILE);

        # Light position
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat*4)(*self.light_pos + [1]))

        # Background/minimum light level
        glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat*4)(*self.light_ambient))

        # Diffuse light color
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat*4)(*self.light_color))

        #glLightf(GL_LIGHT0, GL_SPOT_CUTOFF, 180)
        #glLightf(GL_LIGHT0, GL_SPOT_EXPONENT, 0)
        #glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0)
        #glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0)
        #glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        glShadeModel(GL_SMOOTH)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Render the rooms
        glEnable(GL_TEXTURE_2D)
        for room in self.rooms:
            room._render()

        # Render the static entities
        for ent in self.entities:
            if ent.is_static:
                ent.render()

        glEndList()

    def _render_world(
        self,
        frame_buffer,
        render_agent
    ):
        """Render the world from a given camera position into a frame buffer,
        and produce a numpy image array as output.
        """

        # Call the display list for the static parts of the environment
        glCallList(1)

        # TODO: keep the non-static entities in a different list for efficiency?
        # Render the non-static entities
        for ent in self.entities:
            if not ent.is_static and ent is not self.agent:
                ent.render()
                #ent.draw_bound()

        if render_agent:
            self.agent.render()

        # Resolve the rendered image into a numpy array
        img = frame_buffer.resolve()

        return img

    def render_top_view(self, frame_buffer=None, POMDP=False, render_ag=True):
        """Render a top view of the whole map (from above)"""
        assert isinstance(POMDP, bool), "POMDP parameter should be the type of boolean"

        if frame_buffer == None:
            frame_buffer = self.obs_fb

        # Switch to the default OpenGL context
        # This is necessary on Linux Nvidia drivers
        self.shadow_window.switch_to()

        # Bind the frame buffer before rendering into it
        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*self.black, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Scene extents to render
        min_x = None
        max_x = None
        min_z = None
        max_z = None

        if POMDP:
            min_x = self.agent.pos[0] - 2.5
            max_x = self.agent.pos[0] + 2.5
            min_z = self.agent.pos[2] - 2.5
            max_z = self.agent.pos[2] + 2.5
        else:
            min_x = self.min_x - 1
            max_x = self.max_x + 1
            min_z = self.min_z - 1
            max_z = self.max_z + 1
    
    
        width = max_x - min_x
        height = max_z - min_z
        aspect = width / height
        fb_aspect = frame_buffer.width / frame_buffer.height

        # Adjust the aspect extents to match the frame buffer aspect
        if aspect > fb_aspect:
            # Want to add to denom, add to height
            new_h = width / fb_aspect
            h_diff = new_h - height
            min_z -= h_diff / 2
            max_z += h_diff / 2
        elif aspect < fb_aspect:
            # Want to add to num, add to width
            new_w = height * fb_aspect
            w_diff = new_w - width
            min_x -= w_diff / 2
            max_x += w_diff / 2

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(
            min_x,
            max_x,
            -max_z,
            -min_z,
            -100, 100.0
        )

        # Setup the camera
        # Y maps to +Z, Z maps to +Y
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        m = [
            1, 0, 0, 0,
            0, 0, 1, 0,
            0, -1, 0, 0,
            0, 0, 0, 1,
        ]
        glLoadMatrixf((GLfloat * len(m))(*m))

        return self._render_world(
            frame_buffer,
            render_agent=render_ag
        )

    def render_obs(self, frame_buffer=None):
        """Render an observation from the point of view of the agent"""

        if frame_buffer == None:
            frame_buffer = self.obs_fb

        # Switch to the default OpenGL context
        # This is necessary on Linux Nvidia drivers
        self.shadow_window.switch_to()

        # Bind the frame buffer before rendering into it
        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*self.black, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            self.agent.cam_fov_y,
            frame_buffer.width / float(frame_buffer.height),
            0.04,
            100.0
        )

        # Setup the camera
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            # Eye position
            *self.agent.cam_pos,
            # Target
            *(self.agent.cam_pos + self.agent.cam_dir),
            # Up vector
            0, 1.0, 0.0
        )

        return self._render_world(
            frame_buffer,
            render_agent=False
        )

    def render_depth(self, frame_buffer=None):
        """Produce a depth map
        Values are floating-point, map shape is (H,W,1)
        Distances are in meters from the observer
        """

        if frame_buffer == None:
            frame_buffer = self.obs_fb

        # Render the world
        self.render_obs(frame_buffer)

        return frame_buffer.get_depth_map(0.04, 100.0)

    def get_visible_ents(self):
        """Get a list of visible entities.
        Uses OpenGL occlusion queries to approximate visibility.
        :return: set of objects visible to the agent
        """

        # Allocate the occlusion query ids
        num_ents = len(self.entities)
        query_ids = (GLuint * num_ents)()
        glGenQueries(num_ents, query_ids)

        # Switch to the default OpenGL context
        # This is necessary on Linux Nvidia drivers
        self.shadow_window.switch_to()

        # Use the small observation frame buffer
        frame_buffer = self.obs_fb

        # Bind the frame buffer before rendering into it
        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*self.black, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            self.agent.cam_fov_y,
            frame_buffer.width / float(frame_buffer.height),
            0.04,
            100.0
        )

        # Setup the cameravisible objects
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            # Eye position
            *self.agent.cam_pos,
            # Target
            *(self.agent.cam_pos + self.agent.cam_dir),
            # Up vector
            0, 1.0, 0.0
        )

        # Render the rooms, without texturing
        glDisable(GL_TEXTURE_2D)
        for room in self.rooms:
            room._render()

        # For each entity
        for ent_idx, ent in enumerate(self.entities):
            if ent is self.agent:
                continue

            glBeginQuery(GL_ANY_SAMPLES_PASSED, query_ids[ent_idx])
            pos = ent.pos

            #glColor3f(1, 0, 0)
            drawBox(
                x_min=pos[0] - 0.1,
                x_max=pos[0] + 0.1,
                y_min=pos[1],
                y_max=pos[1] + 0.2,
                z_min=pos[2] - 0.1,
                z_max=pos[2] + 0.1
            )

            glEndQuery(GL_ANY_SAMPLES_PASSED)

        vis_objs = set()

        # Get query results
        for ent_idx, ent in enumerate(self.entities):
            if ent is self.agent:
                continue

            visible = (GLuint*1)(1)
            glGetQueryObjectuiv(query_ids[ent_idx], GL_QUERY_RESULT, visible);

            if visible[0] != 0:
                vis_objs.add(ent)

        # Free the occlusion query ids
        glDeleteQueries(1, query_ids)

        #img = frame_buffer.resolve()
        #return img

        return vis_objs

    def render(self, mode='human', close=False, view='agent'):
        """Render the environment for human viewing"""

        if close:
            if self.window:
                self.window.close()
            return

        # Render the human-view image
        assert view in ['agent', 'top']
        if view == 'agent':
            img = self.render_obs(self.vis_fb)
        else:
            if self.obs_level == 1:
                img = self.render_top_view(self.vis_fb, POMDP=True)
            else:
                img = self.render_top_view(self.vis_fb, POMDP=False)
        img_width = img.shape[1]
        img_height = img.shape[0]

        if mode == 'rgb_array':
            return img

        # Render the agent's view
        obs = self.render_obs()
        obs_width = obs.shape[1]
        obs_height = obs.shape[0]

        window_width = img_width + self.obs_disp_width
        window_height = img_height

        if self.window is None:
            config = pyglet.gl.Config(double_buffer=True)
            self.window = pyglet.window.Window(
                width=window_width,
                height=window_height,
                resizable=False,
                config=config
            )

        self.window.clear()
        self.window.switch_to()

        # Bind the default frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        # Clear the color and depth buffers
        glClearColor(0, 0, 0, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        # Setup orghogonal projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glOrtho(0, window_width, 0, window_height, 0, 10)

        # Draw the human render to the rendering window
        img_flip = np.ascontiguousarray(np.flip(img, axis=0))
        img_data = pyglet.image.ImageData(
            img_width,
            img_height,
            'RGB',
            img_flip.ctypes.data_as(POINTER(GLubyte)),
            pitch=img_width * 3,
        )
        img_data.blit(
            0,
            0,
            0,
            width=img_width,
            height=img_height
        )

        # Draw the observation
        obs = np.ascontiguousarray(np.flip(obs, axis=0))
        obs_data = pyglet.image.ImageData(
            obs_width,
            obs_height,
            'RGB',
            obs.ctypes.data_as(POINTER(GLubyte)),
            pitch=obs_width * 3,
        )
        obs_data.blit(
            img_width,
            img_height - self.obs_disp_height,
            0,
            width=self.obs_disp_width,
            height=self.obs_disp_height
        )

        # Draw the text label in the window
        self.text_label.text = "pos: (%.2f, %.2f, %.2f)\nangle: %d\nsteps: %d" % (
            *self.agent.pos,
            int(self.agent.dir * 180 / math.pi) % 360,
            self.step_count
        )
        self.text_label.draw()

        # Force execution of queued commands
        glFlush()

        # If we are not running the Pyglet event loop,
        # we have to manually flip the buffers and dispatch events
        if mode == 'human':
            self.window.flip()
            self.window.dispatch_events()

        return img