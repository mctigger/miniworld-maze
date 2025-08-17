"""UI entity classes for MiniWorld environments."""

import math
from pyglet.gl import *
from .base_entity import Entity
from ..opengl import Texture


class ImageFrame(Entity):
    """Frame to display an image on a wall
    Note: the position is in the middle of the frame, on the wall
    """

    def __init__(self, pos, dir, tex_name, width, depth=0.05):
        super().__init__()

        self.pos = pos
        self.dir = dir

        # Load the image to be displayed
        self.tex = Texture.get(tex_name)

        self.width = width
        self.depth = depth
        self.height = (float(self.tex.height) / self.tex.width) * self.width

    @property
    def is_static(self):
        return True

    def render(self):
        """Draw the object"""
        x, y, z = self.pos

        # sx is depth
        # Frame points towards +sx
        sx = self.depth
        hz = self.width / 2
        hy = self.height / 2

        glPushMatrix()
        glTranslatef(*self.pos)
        glRotatef(self.dir * (180/math.pi), 0, 1, 0)

        # Bind texture for front
        glColor3f(1, 1, 1)
        glEnable(GL_TEXTURE_2D)
        self.tex.bind()

        # Front face, showing image
        glBegin(GL_QUADS)
        glNormal3f(1, 0, 0)
        glTexCoord2f(1, 1)
        glVertex3f(sx, +hy, -hz)
        glTexCoord2f(0, 1)
        glVertex3f(sx, +hy, +hz)
        glTexCoord2f(0, 0)
        glVertex3f(sx, -hy, +hz)
        glTexCoord2f(1, 0)
        glVertex3f(sx, -hy, -hz)
        glEnd()

        # Black frame/border
        glDisable(GL_TEXTURE_2D)
        glColor3f(0, 0, 0)

        glBegin(GL_QUADS)

        # Left
        glNormal3f(0, 0, -1)
        glVertex3f(0  , +hy, -hz)
        glVertex3f(+sx, +hy, -hz)
        glVertex3f(+sx, -hy, -hz)
        glVertex3f(0  , -hy, -hz)

        # Right
        glNormal3f(0, 0, 1)
        glVertex3f(+sx, +hy, +hz)
        glVertex3f(0  , +hy, +hz)
        glVertex3f(0  , -hy, +hz)
        glVertex3f(+sx, -hy, +hz)

        # Top
        glNormal3f(0, 1, 0)
        glVertex3f(+sx, +hy, +hz)
        glVertex3f(+sx, +hy, -hz)
        glVertex3f(0  , +hy, -hz)
        glVertex3f(0  , +hy, +hz)

        # Bottom
        glNormal3f(0, -1, 0)
        glVertex3f(+sx, -hy, -hz)
        glVertex3f(+sx, -hy, +hz)
        glVertex3f(0  , -hy, +hz)
        glVertex3f(0  , -hy, -hz)

        glEnd()

        glPopMatrix()


class TextFrame(Entity):
    """Frame to display text or numbers on a wall
    Note: the position is in the middle of the frame, on the wall
    """

    def __init__(self, pos, dir, str, height=0.15, depth=0.05):
        super().__init__()

        self.pos = pos
        self.dir = dir

        self.str = str

        self.depth = depth
        self.height = height
        self.width = len(str) * height

    @property
    def is_static(self):
        return True

    def randomize(self, params, rng):
        self.texs = []
        for ch in self.str:
            try:
                if ch == ' ':
                    self.texs.append(None)
                else:
                    tex_name = f'chars/ch_0x{ord(ch)}'
                    self.texs.append(Texture.get(tex_name, rng))
            except:
                raise 'only alphanumerical characters supported in TextFrame'

    def render(self):
        """Draw the object"""
        x, y, z = self.pos

        # sx is depth
        # Frame points towards +sx
        sx = 0.05
        hz = self.width / 2
        hy = self.height / 2

        glPushMatrix()
        glTranslatef(*self.pos)
        glRotatef(self.dir * (180/math.pi), 0, 1, 0)

        # Bind texture for front
        glColor3f(1, 1, 1)

        # For each character
        for idx, ch in enumerate(self.str):
            tex = self.texs[idx]
            if tex:
                glEnable(GL_TEXTURE_2D)
                self.texs[idx].bind()
            else:
                glDisable(GL_TEXTURE_2D)

            char_width = self.height
            z_0 = hz - char_width * (idx+1)
            z_1 = z_0 + char_width

            # Front face, showing image
            glBegin(GL_QUADS)
            glNormal3f(1, 0, 0)
            glTexCoord2f(1, 1)
            glVertex3f(sx, +hy, z_0)
            glTexCoord2f(0, 1)
            glVertex3f(sx, +hy, z_1)
            glTexCoord2f(0, 0)
            glVertex3f(sx, -hy, z_1)
            glTexCoord2f(1, 0)
            glVertex3f(sx, -hy, z_0)
            glEnd()

        # Black frame/border
        glDisable(GL_TEXTURE_2D)
        glColor3f(0, 0, 0)

        glBegin(GL_QUADS)

        # Left
        glNormal3f(0, 0, -1)
        glVertex3f(0  , +hy, -hz)
        glVertex3f(+sx, +hy, -hz)
        glVertex3f(+sx, -hy, -hz)
        glVertex3f(0  , -hy, -hz)

        # Right
        glNormal3f(0, 0, 1)
        glVertex3f(+sx, +hy, +hz)
        glVertex3f(0  , +hy, +hz)
        glVertex3f(0  , -hy, +hz)
        glVertex3f(+sx, -hy, +hz)

        # Top
        glNormal3f(0, 1, 0)
        glVertex3f(+sx, +hy, +hz)
        glVertex3f(+sx, +hy, -hz)
        glVertex3f(0  , +hy, -hz)
        glVertex3f(0  , +hy, +hz)

        # Bottom
        glNormal3f(0, -1, 0)
        glVertex3f(+sx, -hy, -hz)
        glVertex3f(+sx, -hy, +hz)
        glVertex3f(0  , -hy, +hz)
        glVertex3f(0  , -hy, -hz)

        glEnd()

        glPopMatrix()