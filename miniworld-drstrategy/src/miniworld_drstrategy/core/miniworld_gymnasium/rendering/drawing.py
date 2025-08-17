"""Drawing utility functions for MiniWorld environments."""

from pyglet.gl import *


def drawAxes(len=0.1):
    """
    Draw X/Y/Z axes in red/green/blue colors
    """

    glBegin(GL_LINES)

    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(len, 0, 0)

    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, len, 0)

    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, len)

    glEnd()


def drawBox(
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max
):
    """
    Draw a 3D box
    """

    glBegin(GL_QUADS)

    glNormal3f(0, 0, 1)
    glVertex3f(x_max, y_max, z_max)
    glVertex3f(x_min, y_max, z_max)
    glVertex3f(x_min, y_min, z_max)
    glVertex3f(x_max, y_min, z_max)

    glNormal3f(0, 0, -1)
    glVertex3f(x_min, y_max, z_min)
    glVertex3f(x_max, y_max, z_min)
    glVertex3f(x_max, y_min, z_min)
    glVertex3f(x_min, y_min, z_min)

    glNormal3f(-1, 0, 0)
    glVertex3f(x_min, y_max, z_max)
    glVertex3f(x_min, y_max, z_min)
    glVertex3f(x_min, y_min, z_min)
    glVertex3f(x_min, y_min, z_max)

    glNormal3f(1, 0, 0)
    glVertex3f(x_max, y_max, z_min)
    glVertex3f(x_max, y_max, z_max)
    glVertex3f(x_max, y_min, z_max)
    glVertex3f(x_max, y_min, z_min)

    glNormal3f(0, 1, 0)
    glVertex3f(x_max, y_max, z_max)
    glVertex3f(x_max, y_max, z_min)
    glVertex3f(x_min, y_max, z_min)
    glVertex3f(x_min, y_max, z_max)

    glNormal3f(0, -1, 0)
    glVertex3f(x_max, y_min, z_min)
    glVertex3f(x_max, y_min, z_max)
    glVertex3f(x_min, y_min, z_max)
    glVertex3f(x_min, y_min, z_min)

    glEnd()