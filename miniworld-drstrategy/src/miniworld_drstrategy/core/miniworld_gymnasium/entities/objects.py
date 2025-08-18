"""Object entities for MiniWorld environments."""

import math
import numpy as np
from pyglet.gl import *
from .base_entity import Entity, MeshEnt, COLORS, COLOR_NAMES
from ..opengl import drawBox


class Box(Entity):
    """Colored box object"""

    def __init__(self, color, size=0.8, transparentable=False, static=False):
        super().__init__()
        self.trable = transparentable
        self.static_flag = static
        if type(size) is int or type(size) is float:
            size = np.array([size, size, size])
        size = np.array(size)
        sx, sy, sz = size

        self.color = color
        self.size = size

        self.radius = math.sqrt(sx*sx + sz*sz)/2
        self.height = sy

    def randomize(self, params, rng):
        self.color_vec = COLORS[self.color] + params.sample(rng, 'obj_color_bias')
        self.color_vec = np.clip(self.color_vec, 0, 1)

    def render(self):
        """Draw the object"""
        sx, sy, sz = self.size

        glDisable(GL_TEXTURE_2D)
        glColor3f(*self.color_vec)

        glPushMatrix()
        glTranslatef(*self.pos)
        glRotatef(self.dir * (180/math.pi), 0, 1, 0)

        drawBox(
            x_min=-sx/2,
            x_max=+sx/2,
            y_min=0,
            y_max=sy,
            z_min=-sz/2,
            z_max=+sz/2
        )

        glPopMatrix()

    @property
    def is_static(self):
        """Return whether this box is static (cannot move)"""
        return self.static_flag


class Key(MeshEnt):
    """Key the agent can pick up, carry, and use to open doors"""

    def __init__(self, color):
        assert color in COLOR_NAMES
        super().__init__(
            mesh_name='key_{}'.format(color),
            height=0.35,
            static=False
        )


class Ball(MeshEnt):
    """Ball (sphere) the agent can pick up and carry"""

    def __init__(self, color, size=0.6):
        assert color in COLOR_NAMES
        super().__init__(
            mesh_name='ball_{}'.format(color),
            height=size,
            static=False
        )