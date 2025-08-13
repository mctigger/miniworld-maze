"""
Mock labmaze module to replace the real labmaze dependency.
This provides minimal stubs for the interfaces used by memory-maze.
"""

import os
import tempfile
from collections import namedtuple

# Mock maze generation classes
class RandomMaze:
    """Mock replacement for labmaze.RandomMaze"""
    def __init__(self, *args, **kwargs):
        pass
    
    def regenerate(self):
        pass
    
    @property
    def entity_layer(self):
        # Return a simple 7x7 maze layout
        return [
            ".........",
            ".G     G.",
            ". G * G .",
            ".   *   .",
            ". ***** .",
            ".   *G  .",
            ".G  *   .",
            ".       .",
            ".........",
        ]
    
    @property
    def variations_layer(self):
        # Return a simple variations layer
        return [
            ".........",
            ".0     1.",
            ". 0 * 1 .",
            ".   *   .",
            ". ***** .",
            ".   *2  .",
            ".3  *   .",
            ".       .",
            ".........",
        ]
    
    @property
    def width(self):
        return 9
    
    @property
    def height(self):
        return 9

# Mock assets module
class MockAssets:
    """Mock replacement for labmaze.assets"""
    
    @staticmethod
    def get_wall_texture_paths(style):
        """Mock wall texture paths"""
        return {
            'blue': '/tmp/mock_blue_wall.png',
            'red': '/tmp/mock_red_wall.png',
            'green': '/tmp/mock_green_wall.png',
            'yellow': '/tmp/mock_yellow_wall.png',
            'purple': '/tmp/mock_purple_wall.png',
            'orange': '/tmp/mock_orange_wall.png',
        }
    
    @staticmethod
    def get_floor_texture_paths(style):
        """Mock floor texture paths"""
        return {
            'blue': '/tmp/mock_blue_floor.png',
            'orange': '/tmp/mock_orange_floor.png',
            'green': '/tmp/mock_green_floor.png',
            'red': '/tmp/mock_red_floor.png',
        }
    
    @staticmethod
    def get_sky_texture_paths(style):
        """Mock sky texture paths"""
        SkyTextures = namedtuple('SkyTextures', ['left', 'right', 'up', 'down', 'back', 'front'])
        return SkyTextures(
            left='/tmp/mock_sky_left.png',
            right='/tmp/mock_sky_right.png',
            up='/tmp/mock_sky_up.png',
            down='/tmp/mock_sky_down.png',
            back='/tmp/mock_sky_back.png',
            front='/tmp/mock_sky_front.png'
        )

# Create mock texture files
def _create_mock_texture_files():
    """Create empty mock texture files"""
    from PIL import Image
    
    # Create a simple 64x64 colored image for each texture
    colors = {
        'blue': (0, 0, 255),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'yellow': (255, 255, 0),
        'purple': (255, 0, 255),
        'orange': (255, 165, 0),
    }
    
    for color_name, rgb in colors.items():
        # Wall textures
        img = Image.new('RGB', (64, 64), rgb)
        img.save(f'/tmp/mock_{color_name}_wall.png')
        
        # Floor textures
        darker_rgb = tuple(max(0, c - 50) for c in rgb)
        img_floor = Image.new('RGB', (64, 64), darker_rgb)
        img_floor.save(f'/tmp/mock_{color_name}_floor.png')
    
    # Sky textures
    sky_img = Image.new('RGB', (64, 64), (135, 206, 235))  # Sky blue
    for direction in ['left', 'right', 'up', 'down', 'back', 'front']:
        sky_img.save(f'/tmp/mock_sky_{direction}.png')

# Create the mock texture files when module is imported
try:
    _create_mock_texture_files()
except ImportError:
    # PIL not available, skip texture creation
    pass

# Mock defaults module
class MockDefaults:
    """Mock replacement for labmaze.defaults"""
    MAX_ROOMS = 100
    ROOM_MIN_SIZE = 3
    ROOM_MAX_SIZE = 5
    MAX_VARIATIONS = 26
    SPAWN_COUNT = 3
    OBJECT_COUNT = 3
    WALL_CHAR = '*'
    FLOOR_CHAR = '.'
    SPAWN_CHAR = 'P'
    OBJECT_CHAR = 'G'

# Module-level exports to match labmaze interface
assets = MockAssets()
defaults = MockDefaults()