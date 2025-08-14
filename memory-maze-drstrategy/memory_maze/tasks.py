import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import string
from dm_control import composer
from dm_control.locomotion.arenas import labmaze_textures
from dm_control.locomotion.arenas.labmaze_textures import labmaze_assets
from dm_control.locomotion.arenas import covering
from dm_control import mjcf

from memory_maze.maze import *
from memory_maze.oracle import DrawMinimapWrapper, PathToTargetWrapper
from memory_maze.wrappers import *

# Slow control (4Hz), so that agent without HRL has a chance.
# Native control would be ~20Hz, so this corresponds roughly to action_repeat=5.
DEFAULT_CONTROL_FREQ = 4.0


# DrStrategy Custom Texture Classes (from original DrStrategy implementation)

class CWallTexture(labmaze_textures.WallTextures):
    def _build(self, color=[0.8, 0.8, 0.8], model='labmaze_style_01'):
        labmaze_textures = labmaze_assets.get_wall_texture_paths('style_01')
        texture_path = labmaze_textures['blue']
        self._mjcf_root = mjcf.RootElement(model=model)
        im_frame = Image.open(texture_path)
        im_frame = im_frame.convert('RGB')
        np_frame = np.array(im_frame)
        new_color = np.array(color)
        non_black_mask = (np_frame[:, :, :3] > [70, 70, 70]).any(axis=-1)
        np_frame[non_black_mask] = new_color
        image = Image.fromarray(np_frame)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            image.save(temp_file.name)
            valid_name = temp_file.name.split('/')[-1]
            self._textures= [self._mjcf_root.asset.add(
                'texture', type='2d', name=f'wall_{valid_name}',
                file=temp_file.name)]


class CFixedFloorTexture2(labmaze_textures.FloorTextures):
    """Selects a single texture instead of a collection to sample from."""
    
    def _build(self, style, colors):
        self._mjcf_root = mjcf.RootElement(model='labmaze_' + style)
        self._textures = []
        for i,color in enumerate(colors):
            self._textures.append(self._mjcf_root.asset.add(
                'texture', type='2d', name='floor'+str(i), builtin='flat',
                rgb1=color, width=100, height=100))
    
    def append(self, texture):
        self._textures.extend(texture._textures)


# DrStrategy Custom Arena Class with proper texture variations
class CMazeWithTargetsArenaFixedLayout(MazeWithTargetsArenaFixedLayout):
    def _block_variations(self):
        nblocks = 3
        _DEFAULT_FLOOR_CHAR = '.'
        floor_chars = _DEFAULT_FLOOR_CHAR + string.ascii_uppercase
        n, m = self._maze.variations_layer.shape[:2]
        mblocks = m * nblocks // n
        if mblocks <= 1:
            mblocks = 1
        elif 10 % mblocks == 0:
            mblocks -= 1
        ivar = 0
        for i in range(nblocks):
            for j in range(mblocks):
                i_from = i * n // nblocks
                i_to = (i + 1) * n // nblocks
                j_from = j * m // mblocks
                j_to = (j + 1) * m // mblocks
                self._change_block_char(i_from, i_to, j_from, j_to, floor_chars[ivar])
                ivar = (ivar + 1) % 10

    def _change_block_char(self, i1, i2, j1, j2, char):
        grid = self._maze.variations_layer
        i, j = np.where(grid[i1:i2, j1:j2] == '.')
        grid[i + i1, j + j1] = char
    
    def regenerate(self, random_state):
        """Generates a new maze layout. Patch of MazeWithTargets.regenerate() which uses random_state."""        
        self._maze.regenerate()
        self._find_spawn_and_target_positions()

        if self._text_maze_regenerated_hook:
            self._text_maze_regenerated_hook()

        # Remove old texturing planes.
        for geom_name in self._texturing_geom_names:
            del self._mjcf_root.worldbody.geom[geom_name]
        self._texturing_geom_names = []

        # Remove old texturing materials.
        for material_name in self._texturing_material_names:
            del self._mjcf_root.asset.material[material_name]
        self._texturing_material_names = []

        # Remove old actual-wall geoms.
        self._maze_body.geom.clear()
        
        # Fixed random state for consistent wall textures
        random_state_fixed_wall = np.random.RandomState(2)

        self._current_wall_texture = {
            # Use first texture instead of random choice for consistency
            wall_char: wall_textures[0]
            for wall_char, wall_textures in self._wall_textures.items()
        }
        for wall_char in self._wall_textures:
            self._make_wall_geoms(wall_char)
        self._make_floor_variations()

    def _make_floor_variations(self, build_tile_geoms_fn=None):
        """Fork of mazes.MazeWithTargets._make_floor_variations().
        Makes the room floors different if possible, instead of sampling randomly.
        """
        # Apply block variations first
        self._block_variations()
        
        # Call parent class method to handle the floor variations properly
        super()._make_floor_variations(build_tile_geoms_fn)


def memory_maze_single_room_3x3(**kwargs):
    return _memory_maze(
        maze_size=3,
        n_targets=3,
        time_limit=float('inf'),
        max_rooms=1,
        room_min_size=3,
        room_max_size=3,
        targets_per_room=3,
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


def memory_maze_two_rooms_3x7(**kwargs):
    return _memory_maze(
        maze_size=(7, 3),
        n_targets=6,
        time_limit=float('inf'),
        max_rooms=2,
        room_min_size=3,
        room_max_size=3,
        targets_per_room=3,
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


TWO_ROOMS_3x7_LAYOUT = """
*********
*   *   *
*   *   *
*       *
*********
"""[1:]


TWO_ROOMS_3x7_LAYOUT_RANDOM_GOALS = """
*********
*PGG*GGP*
*PPG*GPP*
*PP   PP*
*********
"""[1:]


def memory_maze_two_rooms_3x7_fixed_layout(**kwargs):
    return _memory_maze_fixed_layout(
        entity_layer=TWO_ROOMS_3x7_LAYOUT,
        n_targets=6,
        time_limit=float('inf'),
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


def memory_maze_two_rooms_3x7_fixed_layout_random_goals(**kwargs):
    return _memory_maze_fixed_layout(
        entity_layer=TWO_ROOMS_3x7_LAYOUT_RANDOM_GOALS,
        n_targets=3,
        time_limit=float('inf'),
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


# Original DrStrategy complex maze layout
CMAZE_7x7_LAYOUT = """
*********
*P  *G  *
*** *** *
*G*     *
* *** ***
*     *G*
* ***   *
* G*G   *
*********
"""[1:]

# Simple four rooms layout (original memory-maze-drstrategy)
FOUR_ROOMS_7x7_LAYOUT = """
*********
*G      *
* G *G  *
*   *  G*
* ***** *
*   *G  *
*G  *   *
*       *
*********
"""[1:]

# Original DrStrategy 15x15 complex maze layout
CMAZE_15x15_LAYOUT = """
*****************
***P      *     *
*** *  G* *   * *
*       * *  G* *
*   * *** ***** *
*         *     *
* ***   * * *** *
*G      *   *   *
*** *** * * *  G*
*** *     * *   *
*** * *   * * * *
*     *G      * *
* *** * * *  G* *
*  G      *   * *
*   *   * *** * *
*   *  G*       *
*****************
"""[1:]


FOUR_ROOMS_7x7_LAYOUT_RANDOM_GOALS = """
*********
*PP   PP*
*PPG*GPP*
* GG*GG *
* ***** *
* GG*GG *
*PPG*GPP*
*PP   PP*
*********
"""[1:]


TEN_ROOMS_7x19_LAYOUT_RANDOM_GOALS = """
*********************
*PP   PP*PP   PP*GGP*
*PPG*GPG*GPG*GPG*GPP*
* GG*PP   PP*PP   P *
* ***************** *
* GG*PP   PP*PP   P *
*PPG*GPG*GPG*GPG*GPP*
*PP   PP*PP   PP*GGP*
*********************
"""[1:]


TWENTY_ROOMS_7x39_LAYOUT_RANDOM_GOALS = """
*****************************************
*PP   PP*PP   PP*PP   PP*PP   PP*PP   PP*
*PPG*GPG*GPG*GPG*GPG*GPG*GPG*GPG*GPG*GPP*
* GG*PP   PP*PP   PP*PP   PP*PP   PP*GG *
* ************************************* *
* GG*PP   PP*PP   PP*PP   PP*PP   PP*GG *
*PPG*GPG*GPG*GPG*GPG*GPG*GPG*GPG*GPG*GPP*
*PP   PP*PP   PP*PP   PP*PP   PP*PP   PP*
*****************************************
"""[1:]


def memory_maze_four_rooms_7x7_fixed_layout(**kwargs):
    return _memory_maze_fixed_layout(
        entity_layer=FOUR_ROOMS_7x7_LAYOUT,
        n_targets=6,
        time_limit=float('inf'),
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


def memory_maze_cmaze_7x7_fixed_layout(**kwargs):
    """DrStrategy original 7x7 complex maze layout with original textures"""
    # Set up DrStrategy original texture configuration
    cmap = plt.get_cmap('tab20')
    
    # Wall textures using custom CWallTexture with tab20 colors
    wall_textures = {'*': CWallTexture([0.8, 0.8, 0.8])}
    for index in range(10):
        wall_textures[str(index)] = CWallTexture([int(i*255) for i in cmap(index*2)[:3]])
    for index, idx1 in enumerate([i for i in range(1, 20, 2)]):
        wall_textures[str(index+10)] = CWallTexture([int(i*255) for i in cmap(idx1*2)[:3]])
    
    # Floor textures using custom CFixedFloorTexture2 with tab20 colors
    floor_colors = []
    for index in range(10):
        floor_colors.append([int(i*255) for i in cmap(index*2)[:3]])
    for index, idx1 in enumerate([i for i in range(1, 20, 2)]):
        floor_colors.append([int(i*255) for i in cmap(idx1*2)[:3]])
    floor_textures = CFixedFloorTexture2('style_01', floor_colors)
    
    return _memory_maze_fixed_layout(
        entity_layer=CMAZE_7x7_LAYOUT,
        n_targets=6,
        time_limit=float('inf'),
        target_color_in_image=False,
        seed=42,
        wall_textures=wall_textures,
        floor_textures=floor_textures,
        **kwargs,
    )


def memory_maze_cmaze_15x15_fixed_layout(**kwargs):
    """DrStrategy original 15x15 complex maze layout with original textures"""
    # Set up DrStrategy original texture configuration
    cmap = plt.get_cmap('tab20')
    
    # Wall textures using custom CWallTexture with tab20 colors
    wall_textures = {'*': CWallTexture([0.8, 0.8, 0.8])}
    for index in range(10):
        wall_textures[str(index)] = CWallTexture([int(i*255) for i in cmap(index*2)[:3]])
    for index, idx1 in enumerate([i for i in range(1, 20, 2)]):
        wall_textures[str(index+10)] = CWallTexture([int(i*255) for i in cmap(idx1*2)[:3]])
    
    # Floor textures using custom CFixedFloorTexture2 with tab20 colors
    floor_colors = []
    for index in range(10):
        floor_colors.append([int(i*255) for i in cmap(index*2)[:3]])
    for index, idx1 in enumerate([i for i in range(1, 20, 2)]):
        floor_colors.append([int(i*255) for i in cmap(idx1*2)[:3]])
    floor_textures = CFixedFloorTexture2('style_01', floor_colors)
    
    return _memory_maze_fixed_layout(
        entity_layer=CMAZE_15x15_LAYOUT,
        n_targets=15,
        time_limit=float('inf'),
        target_color_in_image=False,
        seed=42,
        wall_textures=wall_textures,
        floor_textures=floor_textures,
        **kwargs,
    )


def memory_maze_four_rooms_7x7_fixed_layout_random_goals(**kwargs):
    return _memory_maze_fixed_layout(
        entity_layer=FOUR_ROOMS_7x7_LAYOUT_RANDOM_GOALS,
        n_targets=6,
        time_limit=float('inf'),
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


def memory_maze_ten_rooms_7x19_fixed_layout_random_goals(**kwargs):
    return _memory_maze_fixed_layout(
        entity_layer=TEN_ROOMS_7x19_LAYOUT_RANDOM_GOALS,
        n_targets=15,
        time_limit=float('inf'),
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


def memory_maze_twenty_rooms_7x39_fixed_layout_random_goals(**kwargs):
    return _memory_maze_fixed_layout(
        entity_layer=TWENTY_ROOMS_7x39_LAYOUT_RANDOM_GOALS,
        n_targets=30,
        time_limit=float('inf'),
        target_color_in_image=False,
        seed=42,
        **kwargs,
    )


def memory_maze_9x9(**kwargs):
    """
    Maze based on DMLab30-explore_goal_locations_small
    {
        mazeHeight = 11,  # with outer walls
        mazeWidth = 11,
        roomCount = 4,
        roomMaxSize = 5,
        roomMinSize = 3,
    }
    """
    return _memory_maze(9, 3, 250, **kwargs)


def memory_maze_11x11(**kwargs):
    return _memory_maze(11, 4, 500, **kwargs)


def memory_maze_13x13(**kwargs):
    return _memory_maze(13, 5, 750, **kwargs)


def memory_maze_15x15(**kwargs):
    """
    Maze based on DMLab30-explore_goal_locations_large
    {
        mazeHeight = 17,  # with outer walls
        mazeWidth = 17,
        roomCount = 9,
        roomMaxSize = 3,
        roomMaxSize = 3,
    }
    """
    return _memory_maze(15, 6, 1000, max_rooms=9, room_max_size=3, **kwargs)


def _memory_maze(
    maze_size,  # measured without exterior walls
    n_targets,
    time_limit,
    max_rooms=6,
    room_min_size=3,
    room_max_size=5,
    targets_per_room=1,
    control_freq=DEFAULT_CONTROL_FREQ,
    discrete_actions=True,
    image_only_obs=False,
    target_color_in_image=True,
    global_observables=False,
    top_camera=False,
    good_visibility=False,
    show_path=False,
    camera_resolution=64,
    seed=None,
):
    random_state = np.random.RandomState(seed)
    walker = RollingBallWithFriction(camera_height=0.3, add_ears=top_camera)
    if isinstance(maze_size, int):
        maze_size = (maze_size, maze_size)
    arena = MazeWithTargetsArena(
        x_cells=maze_size[0] + 2,  # inner size => outer size
        y_cells=maze_size[1] + 2,
        xy_scale=2.0,
        z_height=1.5 if not good_visibility else 0.4,
        max_rooms=max_rooms,
        room_min_size=room_min_size,
        room_max_size=room_max_size,
        spawns_per_room=1,
        targets_per_room=targets_per_room,
        floor_textures=FixedFloorTexture('style_01', ['blue', 'blue_bright']),
        wall_textures=dict({
            '*': FixedWallTexture('style_01', 'yellow'),  # default wall
        }, **{str(i): labmaze_textures.WallTextures('style_01') for i in range(10)}  # variations
        ),
        skybox_texture=None,
        random_seed=random_state.randint(2147483648),
    )

    task = MemoryMazeTask(
        walker=walker,
        maze_arena=arena,
        n_targets=n_targets,
        target_radius=0.6,
        target_height_above_ground=0.5 if good_visibility else -0.6,
        enable_global_task_observables=True,  # Always add to underlying env, but not always expose in RemapObservationWrapper
        control_timestep=1.0 / control_freq,
        camera_resolution=camera_resolution,
    )

    if top_camera:
        task.observables['top_camera'].enabled = True

    env = composer.Environment(
        time_limit=time_limit - 1e-3,  # subtract epsilon to make sure ep_length=time_limit*fps
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)

    obs_mapping = {
        'image': 'walker/egocentric_camera' if not top_camera else 'top_camera',
        'target_color': 'target_color',
    }
    if global_observables:
        env = TargetsPositionWrapper(env, task._maze_arena.xy_scale, task._maze_arena.maze.width, task._maze_arena.maze.height)
        env = AgentPositionWrapper(env, task._maze_arena.xy_scale, task._maze_arena.maze.width, task._maze_arena.maze.height)
        env = MazeLayoutWrapper(env)
        obs_mapping = dict(obs_mapping, **{
            'agent_pos': 'agent_pos',
            'agent_dir': 'agent_dir',
            'targets_vec': 'targets_vec',
            'targets_pos': 'targets_pos',
            'target_vec': 'target_vec',
            'target_pos': 'target_pos',
            'maze_layout': 'maze_layout',
        })

    env = RemapObservationWrapper(env, obs_mapping)

    if target_color_in_image:
        env = TargetColorAsBorderWrapper(env)

    if show_path:
        env = PathToTargetWrapper(env)
        env = DrawMinimapWrapper(env)

    if image_only_obs:
        # assert target_color_in_image, 'Image-only observation only makes sense with target_color_in_image'
        env = ImageOnlyObservationWrapper(env)

    if discrete_actions:
        env = DiscreteActionSetWrapper(env, [
            np.array([0.0, 0.0]),  # noop
            np.array([-1.0, 0.0]),  # forward
            np.array([0.0, -1.0]),  # left
            np.array([0.0, +1.0]),  # right
            np.array([-1.0, -1.0]),  # forward + left
            np.array([-1.0, +1.0]),  # forward + right
        ])

    return env


def _memory_maze_fixed_layout(
    entity_layer,
    n_targets,
    time_limit,
    control_freq=DEFAULT_CONTROL_FREQ,
    discrete_actions=True,
    image_only_obs=False,
    target_color_in_image=True,
    global_observables=False,
    top_camera=False,
    good_visibility=False,
    show_path=False,
    camera_resolution=64,
    seed=None,
    allow_same_color_targets=True,
    wall_textures=None,
    floor_textures=None,
):
    random_state = np.random.RandomState(seed)
    walker = RollingBallWithFriction(camera_height=0.3, add_ears=top_camera)
    # Use provided textures or default ones
    if floor_textures is None:
        floor_textures = FixedFloorTexture('style_01', ['blue', 'blue_bright'])
    if wall_textures is None:
        wall_textures = dict({
            '*': FixedWallTexture('style_01', 'yellow'),  # default wall
        }, **{str(i): labmaze_textures.WallTextures('style_01') for i in range(10)}  # variations
        )
    
    arena = CMazeWithTargetsArenaFixedLayout(
        entity_layer=entity_layer,
        num_objects=n_targets,
        xy_scale=2.0,
        z_height=1.5 if not good_visibility else 0.4,
        floor_textures=floor_textures,
        wall_textures=wall_textures,
        skybox_texture=None,
        random_state=random_state,
    )

    task = MemoryMazeTask(
        walker=walker,
        maze_arena=arena,
        n_targets=n_targets,
        target_radius=0.6,
        target_height_above_ground=0.5 if good_visibility else -0.6,
        enable_global_task_observables=True,  # Always add to underlying env, but not always expose in RemapObservationWrapper
        control_timestep=1.0 / control_freq,
        camera_resolution=camera_resolution,
        allow_same_color_targets=allow_same_color_targets,
    )

    if top_camera:
        task.observables['top_camera'].enabled = True

    env = composer.Environment(
        time_limit=time_limit - 1e-3,  # subtract epsilon to make sure ep_length=time_limit*fps
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)

    obs_mapping = {
        'image': 'walker/egocentric_camera' if not top_camera else 'top_camera',
        'target_color': 'target_color',
    }
    if global_observables:
        env = TargetsPositionWrapper(env, task._maze_arena.xy_scale, task._maze_arena.maze.width, task._maze_arena.maze.height)
        env = AgentPositionWrapper(env, task._maze_arena.xy_scale, task._maze_arena.maze.width, task._maze_arena.maze.height)
        env = MazeLayoutWrapper(env)
        obs_mapping = dict(obs_mapping, **{
            'agent_pos': 'agent_pos',
            'agent_dir': 'agent_dir',
            'targets_vec': 'targets_vec',
            'targets_pos': 'targets_pos',
            'target_vec': 'target_vec',
            'target_pos': 'target_pos',
            'maze_layout': 'maze_layout',
        })

    env = RemapObservationWrapper(env, obs_mapping)

    if target_color_in_image:
        env = TargetColorAsBorderWrapper(env)

    if show_path:
        env = PathToTargetWrapper(env)
        env = DrawMinimapWrapper(env)

    if image_only_obs:
        # assert target_color_in_image, 'Image-only observation only makes sense with target_color_in_image'
        env = ImageOnlyObservationWrapper(env)

    if discrete_actions:
        env = DiscreteActionSetWrapper(env, [
            np.array([0.0, 0.0]),  # noop
            np.array([-1.0, 0.0]),  # forward
            np.array([0.0, -1.0]),  # left
            np.array([0.0, +1.0]),  # right
            np.array([-1.0, -1.0]),  # forward + left
            np.array([-1.0, +1.0]),  # forward + right
        ])

    return env
