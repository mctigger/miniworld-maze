"""Maze layout definitions and metadata for DrStrategy Memory Maze environments.

This module contains the text-based maze layouts and their associated metadata
like room boundaries, goal positions, and step limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Callable, Dict


@dataclass(frozen=True)
class MazeLayout:
    """Immutable maze layout configuration.
    
    Attributes:
        layout: Text representation of the maze using ASCII characters
        max_steps: Maximum episode length before truncation
        len_x: Width of the maze in cells
        len_y: Height of the maze in cells  
        rooms: Room boundaries as [min_x, max_x, min_y, max_y] coordinates
        invert_origin: Function to convert coordinate systems
    """
    layout: str
    max_steps: int
    len_x: int
    len_y: int
    rooms: List[List[float]]
    invert_origin: Callable[[List[float]], List[float]]


# Maze layout string constants
# These define the visual structure of each maze using ASCII characters:
# '*' = wall, ' ' = empty space, 'P' = player start, 'G' = goal location

FOUR_ROOMS_7X7 = """
*********
*P      *
* G * G *
*   *   *
* ***** *
*   *   *
* G * G *
*       *
*********
"""[1:]  # Remove leading newline

FOUR_ROOMS_15X15 = """
*****************
*P      *       *
*       *       *
*               *
*   G       G   *
*               *
*       *       *
*       *       *
***   *****   ***
*       *       *
*       *       *
*               *
*   G       G   *
*               *
*       *       *
*       *       *
*****************
"""[1:]

EIGHT_ROOMS_30X30 = """
*****************
*P      *       *
*       *       *
*               *
*   G       G   *
*               *
*       *       *
*       *       *
***   *****   ***
*       *       *
*       *       *
*               *
*   G       G   *
*               *
*       *       *
*       *       *
***   *****   ***
*       *       *
*       *       *
*               *
*   G       G   *
*               *
*       *       *
*       *       *
***   *****   ***
*       *       *
*       *       *
*               *
*   G       G   *
*               *
*       *       *
*       *       *
*****************
"""[1:]

MAZE_7X7 = """
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

MAZE_15X15 = """
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


# Layout metadata
LAYOUTS = {
    'FourRooms7x7': MazeLayout(
        layout=FOUR_ROOMS_7X7,
        max_steps=500,
        len_x=7,
        len_y=7,
        rooms=[[-1.5, 4.5, -1.5, 4.5], [4.5, 8.5, -1.5, 4.5], 
               [-1.5, 4.5, 4.5, 8.5], [4.5, 8.5, 4.5, 8.5]],
        invert_origin=lambda p: [p[0], -p[1] + 7]
    ),
    
    'FourRooms15x15': MazeLayout(
        layout=FOUR_ROOMS_15X15,
        max_steps=1000,
        len_x=15,
        len_y=15,
        rooms=[[-1.5, 8.5, -1.5, 8.5], [8.5, 16.5, -1.5, 8.5],
               [-1.5, 8.5, 8.5, 16.5], [8.5, 16.5, 8.5, 16.5]],
        invert_origin=lambda p: [p[0], -p[1] + 15]
    ),
    
    'EightRooms30x30': MazeLayout(
        layout=EIGHT_ROOMS_30X30,
        max_steps=2000,
        len_x=31,
        len_y=31,
        rooms=[[0, 7.5, 0, 8], [7.5, 15, 0, 8],
               [0, 7.5, 8, 15.5], [7.5, 15, 8, 15.5],
               [0, 7.5, 15.5, 23.5], [7.5, 15, 15.5, 23.5],
               [0, 7.5, 23.5, 31], [7.5, 15, 23.5, 31]],
        invert_origin=lambda p: [p[0], -p[1] + 31]
    ),
    
    'Maze7x7': MazeLayout(
        layout=MAZE_7X7,
        max_steps=500,
        len_x=7,
        len_y=7,
        rooms=[[-1.5, 4.5, -1.5, 4.5], [4.5, 8.5, -1.5, 4.5],
               [-1.5, 4.5, 4.5, 8.5], [4.5, 8.5, 4.5, 8.5]],
        invert_origin=lambda p: [p[0], -p[1] + 7]
    ),
    
    'Maze15x15': MazeLayout(
        layout=MAZE_15X15,
        max_steps=1000,
        len_x=15,
        len_y=15,
        rooms=[[-1.5, 8.5, -1.5, 8.5], [8.5, 16.5, -1.5, 8.5],
               [-1.5, 8.5, 8.5, 16.5], [8.5, 16.5, 8.5, 16.5]],
        invert_origin=lambda p: [p[0], -p[1] + 15]
    ),
}


def get_layout(name: str) -> MazeLayout:
    """Get maze layout configuration by name.
    
    Args:
        name: Layout name (e.g., 'FourRooms7x7', 'Maze15x15')
        
    Returns:
        MazeLayout configuration object
        
    Raises:
        ValueError: If layout name is not recognized
    """
    if name not in LAYOUTS:
        available = sorted(LAYOUTS.keys())
        raise ValueError(f"Unknown layout '{name}'. Available layouts: {available}")
    return LAYOUTS[name]


def list_layouts() -> List[str]:
    """Get list of all available maze layout names.
    
    Returns:
        Sorted list of layout names
    """
    return sorted(LAYOUTS.keys())


def validate_layout(layout: str) -> bool:
    """Validate that a maze layout string is properly formatted.
    
    Args:
        layout: Maze layout string to validate
        
    Returns:
        True if layout is valid, False otherwise
    """
    if not layout or not isinstance(layout, str):
        return False
        
    lines = layout.strip().split('\n')
    if len(lines) < 3:  # Minimum viable maze size
        return False
        
    # Check that all lines have same length
    first_line_len = len(lines[0])
    return all(len(line) == first_line_len for line in lines)