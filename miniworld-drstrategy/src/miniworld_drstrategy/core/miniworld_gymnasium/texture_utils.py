"""Texture utility functions for MiniWorld environments."""

import numpy as np

# Texture size/density in texels/meter
TEX_DENSITY = 512


def gen_texcs_wall(tex, min_x, min_y, width, height):
    """Generate texture coordinates for a wall quad"""
    xc = (TEX_DENSITY / tex.width)
    yc = (TEX_DENSITY / tex.height)

    min_u = (min_x) * xc
    max_u = (min_x + width) * xc
    min_v = (min_y) * yc
    max_v = (min_y + height) * yc

    return np.array(
        [
            [min_u, min_v],
            [min_u, max_v],
            [max_u, max_v],
            [max_u, min_v],
        ],
        dtype=np.float32
    )


def gen_texcs_floor(tex, poss):
    """Generate texture coordinates for the floor or ceiling
    
    This is done by mapping x,z positions directly to texture coordinates
    """
    texc_mul = np.array(
        [
            TEX_DENSITY / tex.width,
            TEX_DENSITY / tex.height
        ],
        dtype=float
    )

    coords = np.stack([poss[:,0], poss[:,2]], axis=1) * texc_mul

    return coords