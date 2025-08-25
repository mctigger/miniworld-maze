# Release Notes

## v1.0.0 - Initial Release

### Overview
First release of miniworld-maze, a Python package providing Gymnasium-compatible environments for multi-room maze navigation tasks. This implementation is based on environments from the DrStrategy paper and designed for reinforcement learning research.

### Features
- **Multiple Environment Types**
  - 9-room grid maze
  - 25-room grid maze  
  - Spiral 9-room maze configuration
  - Extensible base classes for custom environments

- **Rendering Modes**
  - First-person perspective rendering
  - Top-down view rendering
  - Configurable observation levels and image transforms

- **Gymnasium Integration**
  - Full compatibility with Gymnasium API
  - Standard action and observation spaces
  - Built-in environment wrappers

- **Advanced Features**
  - Partial observability support
  - OpenGL-based 3D rendering engine
  - Rich texture library with 100+ textures
  - Portrait gallery for visual landmarks
  - Configurable maze layouts and room connections

### Technical Specifications
- **Python Support**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Dependencies**: Gymnasium, NumPy, OpenCV, Pillow, PyOpenGL, Pyglet
- **Optional**: MuJoCo support for enhanced physics

### Installation
```bash
pip install miniworld-maze
```

### Quick Start
```python
import gymnasium as gym
import miniworld_maze

env = gym.make('MiniWorld-NineRooms-v0')
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

### Development Tools
- **Code Quality**: Black formatting, isort imports, Ruff linting
- **Build System**: Modern Python packaging with uv
- **Examples**: Comprehensive example scripts for common use cases

### Known Issues
- Requires OpenGL support for rendering
- Some textures may not display correctly on older graphics drivers

### Contributors
- Tim Joseph (tim@mctigger.com)

### License
See LICENSE file for details.

---

For documentation, examples, and issue reporting, visit: https://github.com/mctigger/miniworld-maze