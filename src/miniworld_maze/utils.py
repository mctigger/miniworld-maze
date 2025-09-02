"""Utility functions for miniworld_maze package."""

from typing import Tuple, Union
import numpy as np


def environment_to_pixel_coords(
    env_pos: np.ndarray,
    env_min: np.ndarray,
    env_max: np.ndarray,
    image_size: Union[int, Tuple[int, int]],
) -> Tuple[int, int]:
    """
    Convert environment coordinates to pixel coordinates.

    Args:
        env_pos: Environment position as ndarray [x, z] or [x, y]
        env_min: Environment minimum bounds as ndarray [min_x, min_z] or [min_x, min_y]
        env_max: Environment maximum bounds as ndarray [max_x, max_z] or [max_x, max_y]
        image_size: Image size (width, height) or single size for square image

    Returns:
        Tuple of (pixel_x, pixel_y) coordinates
    """
    env_x, env_y = env_pos[:2]
    env_min_x, env_min_y = env_min[:2]
    env_max_x, env_max_y = env_max[:2]

    if isinstance(image_size, int):
        width = height = image_size
    else:
        width, height = image_size

    # Normalize to [0, 1] range and scale to pixel coordinates
    pixel_x = int((env_x - env_min_x) / (env_max_x - env_min_x) * width)
    pixel_y = int((env_y - env_min_y) / (env_max_y - env_min_y) * height)

    return pixel_x, pixel_y


def pixel_to_environment_coords(
    pixel_pos: np.ndarray,
    env_min: np.ndarray,
    env_max: np.ndarray,
    image_size: Union[int, Tuple[int, int]],
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to environment coordinates.

    Args:
        pixel_pos: Pixel position as ndarray [x, y]
        env_min: Environment minimum bounds as ndarray [min_x, min_z] or [min_x, min_y]
        env_max: Environment maximum bounds as ndarray [max_x, max_z] or [max_x, max_y]
        image_size: Image size (width, height) or single size for square image

    Returns:
        Tuple of (env_x, env_y) coordinates
    """
    pixel_x, pixel_y = pixel_pos[:2]
    env_min_x, env_min_y = env_min[:2]
    env_max_x, env_max_y = env_max[:2]

    if isinstance(image_size, int):
        width = height = image_size
    else:
        width, height = image_size

    # Convert to normalized [0, 1] range and scale to environment coordinates
    env_x = pixel_x / width * (env_max_x - env_min_x) + env_min_x
    env_y = pixel_y / height * (env_max_y - env_min_y) + env_min_y

    return env_x, env_y


def clamp_to_bounds(
    value: Union[int, float, np.ndarray],
    min_val: Union[int, float, np.ndarray],
    max_val: Union[int, float, np.ndarray],
) -> Union[int, float, np.ndarray]:
    """
    Clamp a value or array of values to specified bounds.

    Args:
        value: Value(s) to clamp (scalar or ndarray)
        min_val: Minimum bound(s) (scalar or ndarray)
        max_val: Maximum bound(s) (scalar or ndarray)

    Returns:
        Clamped value(s)
    """
    if isinstance(value, np.ndarray):
        return np.clip(value, min_val, max_val)
    else:
        return max(min_val, min(max_val, value))


def clamp_pixel_coords(
    pixel_x: int, pixel_y: int, image_size: Union[int, Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Clamp pixel coordinates to image bounds.

    Args:
        pixel_x: X pixel coordinate
        pixel_y: Y pixel coordinate
        image_size: Image size (width, height) or single size for square image

    Returns:
        Tuple of clamped (pixel_x, pixel_y) coordinates
    """
    if isinstance(image_size, int):
        width = height = image_size
    else:
        width, height = image_size

    clamped_x = max(0, min(width - 1, pixel_x))
    clamped_y = max(0, min(height - 1, pixel_y))

    return clamped_x, clamped_y


def normalize_coordinates(
    coords: np.ndarray,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
) -> Tuple[float, float]:
    """
    Normalize coordinates to [0, 1] range based on given bounds.

    Args:
        coords: Coordinates to normalize as ndarray
        min_bounds: Minimum bounds as ndarray
        max_bounds: Maximum bounds as ndarray

    Returns:
        Normalized coordinates as (x, y) tuple
    """
    x, y = coords[:2]
    min_x, min_y = min_bounds[:2]
    max_x, max_y = max_bounds[:2]

    norm_x = (x - min_x) / (max_x - min_x)
    norm_y = (y - min_y) / (max_y - min_y)

    return norm_x, norm_y


def denormalize_coordinates(
    normalized_coords: np.ndarray,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
) -> Tuple[float, float]:
    """
    Convert normalized [0, 1] coordinates back to original coordinate space.

    Args:
        normalized_coords: Normalized coordinates in [0, 1] range as ndarray
        min_bounds: Original minimum bounds as ndarray
        max_bounds: Original maximum bounds as ndarray

    Returns:
        Denormalized coordinates as (x, y) tuple
    """
    norm_x, norm_y = normalized_coords[:2]
    min_x, min_y = min_bounds[:2]
    max_x, max_y = max_bounds[:2]

    x = norm_x * (max_x - min_x) + min_x
    y = norm_y * (max_y - min_y) + min_y

    return x, y


def get_environment_bounds(env) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Extract environment bounds from an environment object.

    Args:
        env: Environment object (may be wrapped)

    Returns:
        Tuple of ((min_x, min_z), (max_x, max_z))
    """
    # Unwrap environment to access base environment
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    min_bounds = (base_env.min_x, base_env.min_z)
    max_bounds = (base_env.max_x, base_env.max_z)

    return min_bounds, max_bounds


def calculate_view_size_from_bounds(
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
) -> Tuple[float, float]:
    """
    Calculate view size (width, height) from coordinate bounds.

    Args:
        min_bounds: Minimum bounds as ndarray (min_x, min_y)
        max_bounds: Maximum bounds as ndarray (max_x, max_y)

    Returns:
        Tuple of (width, height)
    """
    min_x, min_y = min_bounds[:2]
    max_x, max_y = max_bounds[:2]

    width = max_x - min_x
    height = max_y - min_y

    return width, height


def scale_coordinates(
    coords: np.ndarray,
    scale_factor: Union[float, Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Scale coordinates by a given factor.

    Args:
        coords: Coordinates to scale as ndarray
        scale_factor: Scale factor (uniform) or (scale_x, scale_y)

    Returns:
        Scaled coordinates as (x, y) tuple
    """
    x, y = coords[:2]

    if isinstance(scale_factor, (tuple, list)):
        scale_x, scale_y = scale_factor
    else:
        scale_x = scale_y = scale_factor

    return x * scale_x, y * scale_y


def distance_2d(
    pos1: np.ndarray,
    pos2: np.ndarray,
) -> float:
    """
    Calculate 2D Euclidean distance between two positions.

    Args:
        pos1: First position as ndarray (x, y)
        pos2: Second position as ndarray (x, y)

    Returns:
        Euclidean distance
    """
    x1, y1 = pos1[:2]
    x2, y2 = pos2[:2]

    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def lerp_2d(
    pos1: np.ndarray,
    pos2: np.ndarray,
    t: float,
) -> Tuple[float, float]:
    """
    Linear interpolation between two 2D positions.

    Args:
        pos1: Start position as ndarray (x, y)
        pos2: End position as ndarray (x, y)
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated position as (x, y) tuple
    """
    x1, y1 = pos1[:2]
    x2, y2 = pos2[:2]

    t = clamp_to_bounds(t, 0.0, 1.0)

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return x, y
