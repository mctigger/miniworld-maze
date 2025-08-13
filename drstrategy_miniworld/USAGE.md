# DrStrategy Miniworld Usage Guide

## Quick Start

```bash
# 1. Activate environment
source /path/to/venv_local/bin/activate

# 2. Install package
pip install -e .

# 3. Run live visualization
python visualize_web.py --env OneRoom

# 4. Open browser to http://localhost:8080
```

## Available Environments

### Navigation Environments
- **OneRoom**: Single room with colored objects as landmarks
- **TwoRoomsVer1**: Two connected rooms with different floor textures  
- **ThreeRooms**: Linear sequence of three connected rooms

### Object Manipulation
- **PickupObjs**: Multi-object pickup task with configurable object count
- **RoomObjs**: Single room with boxes, balls, and keys to explore

### Sim-to-Real Transfer
- **SimToRealGoto**: Navigate to red box with domain randomization
- **SimToRealPush**: Push red box toward yellow box
- **RemoteBot**: Interface for controlling real robots via ZMQ

## Environment Parameters

```python
# Room navigation environments
OneRoom(room_size=8, door_size=2)
TwoRoomsVer1(room_size=6, door_size=2)  
ThreeRooms(room_size=6, door_size=2)

# Object manipulation
PickupObjs(size=10, num_objs=5)
RoomObjs(size=8)

# Sim-to-real (no parameters needed)
SimToRealGoto()
SimToRealPush()
```

## Visualization Options

```bash
# Basic environments
python visualize_web.py --env OneRoom --room-size 8
python visualize_web.py --env TwoRoomsVer1 --room-size 6
python visualize_web.py --env ThreeRooms --room-size 6

# Object environments  
python visualize_web.py --env PickupObjs --size 12 --num-objs 7
python visualize_web.py --env RoomObjs --size 10

# Sim-to-real environments
python visualize_web.py --env SimToRealGoto
python visualize_web.py --env SimToRealPush

# Network options
python visualize_web.py --env OneRoom --host 0.0.0.0 --port 8080
```

## Visualization Features

The web interface provides:

### Real-time Monitoring
- **Agent View**: Live 60x80 RGB images from agent perspective
- **Episode Stats**: Current episode, step count, total reward
- **Action Distribution**: Histogram of recent actions with names
- **Reward History**: Visual timeline of last 100 rewards

### Interactive Controls
- **Pause/Resume**: Stop/start the random agent
- **Restart Episode**: Reset environment to start new episode
- **Live Updates**: 10 FPS refresh rate optimized for 3D environments

### Environment Information
- **Object Status**: Shows objects picked up, carrying state
- **Parameters**: Displays environment configuration
- **Navigation**: Room layout and agent position context

## Programming Interface

```python
import drstrategy_miniworld
from drstrategy_miniworld.envs import OneRoom, PickupObjs

# Create environment
env = OneRoom(room_size=8)

# Standard Gymnasium interface
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# Check action space
print(f"Actions: {env.action_space}")
if hasattr(env, 'actions'):
    for i in range(env.action_space.n):
        print(f"  {i}: {env.actions(i).name}")

# Environment-specific features
if hasattr(env, 'num_picked_up'):
    print(f"Objects picked up: {env.num_picked_up}")
```

## Troubleshooting

### Common Issues

1. **Texture not found errors**: 
   - Ensure Farama Miniworld is properly installed
   - Check texture names match available textures

2. **Portal outside wall extents**:
   - Use larger room_size parameters (minimum 6)
   - Check door_size is smaller than room_size

3. **Import errors**:
   - Verify package is installed: `pip install -e .`
   - Check virtual environment is activated

4. **Visualization server issues**:
   - Ensure port is not already in use
   - Try different port: `--port 8081`
   - Check firewall settings for external access

### Performance Tips

- Use `--host 0.0.0.0` for remote access
- Larger environments may run slower (reduce room_size if needed)
- Visualization runs at 10 FPS for optimal 3D rendering
- Close environments with `env.close()` when done