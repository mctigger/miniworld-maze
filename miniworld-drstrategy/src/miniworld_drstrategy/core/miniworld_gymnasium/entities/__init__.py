"""Entity classes for MiniWorld environments."""

from .base_entity import Entity, MeshEnt, COLORS, COLOR_NAMES
from .objects import Box, Key, Ball
from .ui_entities import ImageFrame, TextFrame
from .agent import Agent

__all__ = [
    "Entity", "MeshEnt", "COLORS", "COLOR_NAMES",
    "Box", "Key", "Ball", 
    "ImageFrame", "TextFrame",
    "Agent"
]