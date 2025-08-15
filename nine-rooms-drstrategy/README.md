# Nine Rooms DrStrategy Environment

This package provides a faithful implementation of the Nine Rooms 2D top-down navigation environment from the DrStrategy paper, ported from the original MiniWorld-based implementation.

## Features

- **2D Top-Down View**: Provides partial observations of the environment with a top-down perspective
- **Nine Room Layout**: 3x3 grid of connected rooms with distinctive textures
- **Room Navigation**: Agent can navigate between rooms through doorways
- **Goal-Based Tasks**: Supports goal-directed navigation with room-specific objectives
- **Gymnasium Compatible**: Modern Gymnasium API (v0.27+) compatibility
- **Lightweight**: No MiniWorld dependencies, pure Python implementation

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
import nine_rooms_drstrategy

# Create nine rooms environment
env = gym.make('NineRoomsDrStrategy-v0')

# Run a simple episode
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

## Environment Details

### Action Space
- **Type**: Discrete(6)
- **Actions**:
  - 0: Stay in place
  - 1: Move forward
  - 2: Turn left
  - 3: Turn right  
  - 4: Move forward + turn left
  - 5: Move forward + turn right

### Observation Space
- **Type**: Box(0, 255, (3, 64, 64), uint8)
- **Format**: CHW (Channel-Height-Width) RGB image
- **Description**: 2D top-down view of the environment showing the agent's current location and nearby rooms

### Room Layout

The nine rooms are arranged in a 3x3 grid:

```
-------------
| 0 | 1 | 2 |
-------------  
| 3 | 4 | 5 |
-------------
| 6 | 7 | 8 |
-------------
```

Each room has a unique floor texture and is connected to adjacent rooms through doorways.

### Goal System

- Use `env.get_goal()` to set a random goal
- Use `env.is_goal_achieved(pos)` to check if the agent has reached the goal
- Goals can be set to specific rooms with `env.get_goal(room_idx=X)`

## Environments

- `NineRoomsDrStrategy-v0`: Standard nine rooms with full connectivity
- `SpiralNineRoomsDrStrategy-v0`: Nine rooms with spiral connectivity pattern

## Compatibility

This implementation maintains compatibility with the original DrStrategy paper's environment while providing:
- Modern Gymnasium API
- Efficient 2D rendering 
- No external graphics dependencies
- Easy integration with RL frameworks

## Citation

If you use this environment in your research, please cite the original DrStrategy paper:

```bibtex
@inproceedings{drstrategy2023,
    title={DrStrategy: Learning to Navigate Complex Environments with Strategic Representations},
    author={...},
    booktitle={...},
    year={2023}
}
```