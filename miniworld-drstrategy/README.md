# MiniWorld DrStrategy - Nine Rooms Environment Package

This self-contained package provides complete implementations of the Nine Rooms environment variants from the DrStrategy paper, along with comprehensive observation generation tools.

## Directory Structure

```
miniworld-drstrategy/
├── __init__.py                    # Package initialization and exports
├── generate_observations.py      # Main observation generation script
├── nine_rooms_factory.py         # Environment factory and wrappers
├── README.md                      # This documentation
└── miniworld_gymnasium/          # Core environment implementation
    ├── __init__.py
    ├── entity.py                  # 3D entities and objects
    ├── envs/
    │   ├── __init__.py
    │   └── roomnav.py            # NineRooms, SpiralNineRooms, TwentyFiveRooms
    ├── math.py                    # Mathematical utilities
    ├── miniworld.py               # Base environment class
    ├── objmesh.py                 # 3D mesh handling
    ├── opengl.py                  # OpenGL rendering (includes FrameBuffer)
    ├── params.py                  # Environment parameters
    ├── random.py                  # Random utilities
    ├── textures/                  # Wall and floor textures (essential subset)
    ├── utils.py                   # General utilities
    └── wrappers.py               # Environment wrappers
```

## Environment Variants

1. **NineRooms**: Classic 3×3 grid with 12 connections
2. **SpiralNineRooms**: 3×3 grid with 8 spiral connections  
3. **TwentyFiveRooms**: Large 5×5 grid with 40 connections

## Usage

### Generate Observations

```bash
# From within the miniworld-drstrategy directory:
python generate_observations.py NineRooms
python generate_observations.py SpiralNineRooms  
python generate_observations.py TwentyFiveRooms --high-res-full

# Optional arguments:
--output-dir DIRECTORY     # Custom output directory
--high-res-full           # Generate 512×512 full environment views
```

### Use as Python Package

```python
from nine_rooms_factory import create_nine_rooms_env

# Create environment
env = create_nine_rooms_env(variant="NineRooms", size=64)
obs, info = env.reset()

# Take actions
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# Render from specific position
view = env.render_on_pos([15.0, 0.0, 15.0])
```

## Generated Observations

Each variant generates 15 images:

- **3 Full environment views**: Complete environment layout
- **4 Partial observations**: POMDP views from different positions
- **4 Gymnasium observations**: Processed observations through wrappers
- **4 Render-on-pos examples**: Views from strategic positions

## Dependencies

- gymnasium
- numpy
- PIL (Pillow)
- cv2 (OpenCV)
- OpenGL rendering libraries
- MuJoCo (for physics simulation)

## Environment Variables

```bash
# For headless rendering
export MUJOCO_GL=osmesa

# For DrStrategy package compatibility (if needed)
export PYTHONPATH="/path/to/drstrategy:/path/to/drstrategy/drstrategy"
```

## Features

✅ **Self-contained**: All dependencies included  
✅ **Zero external path dependencies**: Works from any location  
✅ **Modern Gymnasium API**: Full compatibility with gym 0.26+  
✅ **High-resolution rendering**: Optional 512×512 layout views  
✅ **Clean architecture**: Modular, well-documented code  
✅ **Multiple variants**: All three DrStrategy environment types  

## Paper Compliance

This implementation maintains high fidelity to the original DrStrategy paper specifications:
- 64×64×3 RGB observations
- 1000-step episode limits  
- Discrete action space (turn_left, turn_right, move_forward)
- Proper room connectivity and texturing
- POMDP partial observability