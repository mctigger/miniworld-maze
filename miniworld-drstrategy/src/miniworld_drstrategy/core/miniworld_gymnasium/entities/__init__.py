"""Entity classes for MiniWorld environments."""

from .base_entity import Entity, MeshEnt, COLORS, COLOR_NAMES
from .objects import Box, Key, Ball
from .agent import Agent

__all__ = [
    "Entity", "MeshEnt", "COLORS", "COLOR_NAMES",
    "Box", "Key", "Ball", 
    "Agent"
]