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

### Registered Environments

This package registers the following gymnasium environments:

| Environment ID | Description | Rooms | Max Steps |
|---|---|---|---|
| `NineRooms-v0` | Standard 3×3 grid with adjacent room connections | 9 | 1000 |
| `SpiralNineRooms-v0` | 3×3 grid with spiral connection pattern | 9 | 1000 |
| `TwentyFiveRooms-v0` | Large 5×5 grid with complex navigation | 25 | 1000 |

All environments use `TOP_DOWN_PARTIAL` observation level and factory default room/door sizes by default.

### Basic Usage

See `examples/basic_usage.py` for a complete working example:

```python
#!/usr/bin/env python3
"""
Basic usage example for miniworld-maze environments.

This is a minimal example showing how to create and interact with the environment.
"""

import gymnasium as gym
import miniworld_maze  # noqa: F401


def main():
    # Create environment using gymnasium registry
    env = gym.make("NineRooms-v0", obs_width=64, obs_height=64)
    obs, info = env.reset()

    # obs is a dictionary containing:
    # - 'observation': (64, 64, 3) RGB image array
    # - 'desired_goal': (64, 64, 3) RGB image of the goal state
    # - 'achieved_goal': (64, 64, 3) RGB image of the current state

    # Take a few random actions
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step + 1}: reward={reward:.3f}, terminated={terminated}")

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("Environment closed successfully!")


if __name__ == "__main__":
    main()
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

### Available Environments

The package provides three main environment variants, each with different room layouts and connection patterns:

#### 1. NineRooms (3×3 Grid)
```
-------------
| 0 | 1 | 2 |
-------------
| 3 | 4 | 5 |
-------------
| 6 | 7 | 8 |
-------------
```
A standard 3×3 grid where adjacent rooms are connected. The agent can navigate between rooms through doorways, with connections forming a fully connected grid pattern.

#### 2. SpiralNineRooms (3×3 Spiral Pattern)
```
-------------
| 0 | 1 | 2 |
-------------
| 3 | 4 | 5 |
-------------
| 6 | 7 | 8 |
-------------
```
Same room layout as NineRooms but with a spiral connection pattern. Only specific room pairs are connected, creating a more challenging navigation task with fewer available paths.

#### 3. TwentyFiveRooms (5×5 Grid)
```
---------------------
| 0 | 1 | 2 | 3 | 4 |
---------------------
| 5 | 6 | 7 | 8 | 9 |
---------------------
|10 |11 |12 |13 |14 |
---------------------
|15 |16 |17 |18 |19 |
---------------------
|20 |21 |22 |23 |24 |
---------------------
```
A larger 5×5 grid environment with 25 rooms, providing more complex navigation challenges and longer episode lengths.

### Observation Types

Each environment supports three different observation modes:

- **`TOP_DOWN_PARTIAL`** (default): Agent-centered partial top-down view with limited visibility range (POMDP)
- **`TOP_DOWN_FULL`**: Complete top-down view showing the entire environment
- **`FIRST_PERSON`**: 3D first-person perspective view from the agent's current position

### Action Space

- **Discrete Actions** (default): 7 discrete actions (turn left/right, move forward/backward, strafe left/right, no-op)
- **Continuous Actions**: Continuous control with `continuous=True` parameter

### Environment Configuration

All environments can be customized with the following parameters:

```python
import gymnasium as gym
from miniworld_maze import ObservationLevel
import miniworld_maze  # noqa: F401

env = gym.make(
    "NineRooms-v0",                        # Environment variant
    obs_level=ObservationLevel.TOP_DOWN_PARTIAL,  # Observation type
    obs_width=64,                          # Observation image width
    obs_height=64,                         # Observation image height
    room_size=5,                           # Size of each room in environment units
    door_size=2,                           # Size of doors between rooms  
    agent_mode="empty",                    # Agent rendering: "empty", "circle", "triangle"
)
```

### Observation Format

The environment returns observations in dictionary format:

```python
obs = {
    'observation': np.ndarray,    # (64, 64, 3) RGB image of current view
    'desired_goal': np.ndarray,   # (64, 64, 3) RGB image of goal location
    'achieved_goal': np.ndarray,  # (64, 64, 3) RGB image of current state
}
```

### Reward Structure

- **Goal reaching**: Positive reward when agent reaches the goal location
- **Step penalty**: Small negative reward per step to encourage efficiency
- **Episode termination**: When goal is reached or maximum steps exceeded


## License

MIT License - see LICENSE file for details.