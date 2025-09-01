# MiniWorld DrStrategy - Multi-Room Maze Environment

A refactored implementation of Dr. Strategy's MiniWorld-based maze environments with updated dependencies and modern Python packaging. Based on the now-deprecated [MiniWorld](https://github.com/Farama-Foundation/Miniworld) project and the original [DrStrategy implementation](https://github.com/ahn-ml/drstrategy).

## Environment Observations

### Environment Views
Full environment layout and render-on-position views:

| Full Environment | Partial Top-Down Observations | Partial First-Person Observations |
|---|---|---|
| ![Full View Clean](assets/images/full_view_clean.png) | ![Top Middle TD](assets/images/render_on_pos_1_top_middle_room_topdown.png) ![Center TD](assets/images/render_on_pos_3_environment_center_topdown.png) | ![Top Middle FP](assets/images/render_on_pos_1_top_middle_room_firstperson.png) ![Center FP](assets/images/render_on_pos_3_environment_center_firstperson.png) |

## Installation

```bash
pip install miniworld-maze
```

## Usage

### Basic Usage

```python
from miniworld_maze import create_drstrategy_env

# Create environment
env = create_drstrategy_env(variant="NineRooms", size=64)
obs, info = env.reset()

# Take actions
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

### Headless Environments

When running in headless environments (servers, CI/CD, Docker containers) or when encountering X11/OpenGL context issues, you need to enable headless rendering:

```bash
# Set environment variable before running Python
export PYGLET_HEADLESS=1
python your_script.py
```

Or in your Python code (must be set before importing the library):

```python
import os
os.environ['PYGLET_HEADLESS'] = '1'

import miniworld_maze
# ... rest of your code
```

This configures the underlying pyglet library to use EGL rendering instead of X11, allowing the environments to run without a display server.

## Environment Variants

- **NineRooms**: 3×3 grid layout
- **SpiralNineRooms**: Spiral connection pattern  
- **TwentyFiveRooms**: 5×5 grid layout


## License

MIT License - see LICENSE file for details.