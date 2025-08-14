"""
DrStrategy Memory Maze environments with Gymnasium compatibility.

This package provides simplified access to DrStrategy Memory Maze environments
using the modern Gymnasium API. It supports various maze layouts for testing
agent memory and navigation capabilities.
"""

from .envs import MemoryMaze, Enhanced3DMemoryMaze, make_env, make
from .maze_layouts import get_layout, list_layouts, validate_layout, LAYOUTS

# Package metadata
__version__ = "1.0.0"

# Register gymnasium environments  
import gymnasium as gym
from gymnasium.envs.registration import register

# Environment configurations  
# Each environment maps to a specific maze layout with default settings
_ENV_CONFIGS = [
    ('DrStrategy-MemoryMaze-4x7x7-v0', '4x7x7', 'FourRooms7x7'),
    ('DrStrategy-MemoryMaze-4x15x15-v0', '4x15x15', 'FourRooms15x15'), 
    ('DrStrategy-MemoryMaze-8x30x30-v0', '8x30x30', 'EightRooms30x30'),
    ('DrStrategy-MemoryMaze-mzx7x7-v0', 'mzx7x7', 'Maze7x7'),
    ('DrStrategy-MemoryMaze-mzx15x15-v0', 'mzx15x15', 'Maze15x15')
]

# Register environments with gymnasium
for env_id, task, layout_name in _ENV_CONFIGS:
    register(
        id=env_id,
        entry_point='drstrategy_memory_maze:make_env',
        kwargs={
            'task': task,
            'discrete_actions': True  # Default to discrete actions
        },
        max_episode_steps=LAYOUTS[layout_name].max_steps
    )

# Make key components available at package level
__all__ = [
    'MemoryMaze',
    'Enhanced3DMemoryMaze',
    'make_env', 
    'make',
    'get_layout',
    'list_layouts',
    'validate_layout', 
    'LAYOUTS',
    '__version__'
]