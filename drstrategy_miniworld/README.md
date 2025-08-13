# DrStrategy Miniworld Extensions

This package extends the Farama-Foundation Miniworld environment with additional environments and utilities specifically designed for DrStrategy research.

## Overview

Instead of copying the entire Farama Miniworld codebase, this package follows a dependency pattern where:

1. **Base Framework**: Uses Farama-Foundation/Miniworld as the core dependency
2. **Custom Extensions**: Implements only the unique environments and features from DrStrategy
3. **API Compatibility**: Adapts the old gym API to the new gymnasium API

## Installation

```bash
pip install -e .
```

This will automatically install Farama Miniworld as a dependency.

## Key Differences from Original DrStrategy Miniworld

### API Updates
- **Gymnasium API**: Updated from old gym to new gymnasium API
- **Return Values**: Step function now returns `(obs, reward, terminated, truncated, info)` instead of `(obs, reward, done, info)`
- **Reset Function**: Now returns `(obs, info)` instead of just `obs`

### Custom Environments

#### Object Manipulation
- **PickupObjs**: Multi-object pickup environment with configurable object types and counts
- **RoomObjs**: Single room with various objects for visual navigation tasks

#### Navigation Tasks  
- **OneRoom**: Single room navigation with visual landmarks
- **TwoRoomsVer1**: Two connected rooms with different floor textures
- **ThreeRooms**: Linear three-room navigation task

#### Sim-to-Real Transfer
- **SimToRealGoto**: Robot navigation to target with domain randomization
- **SimToRealPush**: Object pushing task designed for real robot deployment
- **RemoteBot**: Remote control interface for real robot interaction

## Usage

### Python API

```python
import drstrategy_miniworld
from drstrategy_miniworld.envs import OneRoom, PickupObjs

# Create an environment directly
env = OneRoom(room_size=8)

# Standard gymnasium interface
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Live Visualization

The package includes a web-based visualization tool for live viewing of environments:

```bash
# Basic usage - visualize OneRoom environment
python visualize_web.py --env OneRoom

# Visualize PickupObjs with 5 objects in a 10x10 room
python visualize_web.py --env PickupObjs --size 10 --num-objs 5

# Run on different port for external access
python visualize_web.py --env ThreeRooms --host 0.0.0.0 --port 8080

# Visualize sim-to-real environments
python visualize_web.py --env SimToRealGoto
```

The visualization server provides:
- **Real-time 3D view**: Live agent perspective with 60x80 RGB images
- **Episode statistics**: Reward history, step counts, action distribution
- **Interactive controls**: Pause/resume, restart episode
- **Environment info**: Object pickup status, carrying state, custom metrics
- **Action analysis**: Distribution of recent actions with action names

Open your browser to `http://localhost:8080` to view the visualization.

## Architecture Benefits

1. **Reduced Maintenance**: No need to maintain a fork of the entire Miniworld codebase
2. **Automatic Updates**: Benefits from Farama Miniworld improvements and bug fixes
3. **Clean Separation**: Clear distinction between base functionality and custom extensions
4. **Smaller Package**: Only includes the unique DrStrategy components

## Dependencies

- `miniworld>=2.1.0` (Farama-Foundation)
- `gymnasium>=0.29.0`
- `numpy>=1.22.0`
- `pyglet>=1.5.27,<2.0`

## Development

The package structure follows Python best practices:

```
drstrategy_miniworld/
├── drstrategy_miniworld/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── pickupobjs.py      # Multi-object pickup tasks
│   │   ├── remotebot.py       # Real robot interface  
│   │   ├── roomnav.py         # Navigation environments (1-3 rooms)
│   │   ├── roomobjs.py        # Single room with objects
│   │   ├── simtorealgoto.py   # Sim-to-real navigation
│   │   └── simtorealpush.py   # Sim-to-real pushing
│   └── experiments/
│       ├── __init__.py
│       └── utils.py           # Research utilities
├── visualize_web.py           # Live web visualization
├── setup.py
└── README.md
```