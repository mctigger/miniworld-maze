# Import all classes from the new modular structure
from .texture_utils import gen_texcs_wall, gen_texcs_floor, TEX_DENSITY
from .room import Room, DEFAULT_WALL_HEIGHT
from .base_env import MiniWorldEnv
from .custom_env import CustomMiniWorldEnv