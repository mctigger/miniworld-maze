# MiniWorld DrStrategy - Nine Rooms Environment Package

A well-structured Python package providing complete implementations of the Nine Rooms environment variants from the DrStrategy paper, along with comprehensive observation generation tools.

## Package Structure

```
miniworld-drstrategy/
├── pyproject.toml                 # Modern Python packaging configuration
├── README.md                      # This documentation
└── src/
    └── miniworld_drstrategy/      # Main package
        ├── __init__.py            # Package exports
        ├── core/                  # Core MiniWorld implementation
        │   ├── __init__.py
        │   └── miniworld_gymnasium/   # 3D rendering engine
        │       ├── entity.py          # 3D entities and objects
        │       ├── miniworld.py       # Base environment class
        │       ├── opengl.py          # OpenGL rendering & FrameBuffer
        │       ├── textures/          # Wall and floor textures
        │       └── ...                # Math, utils, wrappers
        ├── environments/          # Environment implementations
        │   ├── __init__.py
        │   ├── factory.py         # Environment factory and wrapper
        │   ├── nine_rooms.py      # NineRooms implementation
        │   ├── spiral_nine_rooms.py  # SpiralNineRooms implementation
        │   └── twenty_five_rooms.py  # TwentyFiveRooms implementation
        ├── wrappers/              # Gymnasium wrappers
        │   ├── __init__.py
        │   └── image_transforms.py   # PyTorch compatibility wrappers
        └── tools/                 # Command-line tools
            ├── __init__.py
            └── generate_observations.py  # Observation generator
```

## Environment Variants

1. **NineRooms**: Classic 3×3 grid with 12 connections
2. **SpiralNineRooms**: 3×3 grid with 8 spiral connections  
3. **TwentyFiveRooms**: Large 5×5 grid with 40 connections

## Installation

### Development Installation

```bash
# From the package root directory
pip install -e .

# With optional dependencies
pip install -e ".[dev,mujoco]"
```

### Production Installation

```bash
pip install miniworld-drstrategy
```

## Usage

### Command Line Interface

```bash
# Generate observations for any variant
generate-observations NineRooms
generate-observations SpiralNineRooms  
generate-observations TwentyFiveRooms --high-res-full

# From source (development)
python -m miniworld_drstrategy.tools.generate_observations NineRooms

# Optional arguments:
--output-dir DIRECTORY     # Custom output directory
--high-res-full           # Generate 512×512 full environment views
```

### Python API

```python
# Import the package
from miniworld_drstrategy import create_nine_rooms_env

# Create environment
env = create_nine_rooms_env(variant="NineRooms", size=64)
obs, info = env.reset()

# Take actions
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# Render from specific position
view = env.render_on_pos([15.0, 0.0, 15.0])

# Close environment
env.close()
```

### Advanced Usage

```python
# Import specific components
from miniworld_drstrategy.environments import NineRooms, SpiralNineRooms
from miniworld_drstrategy.wrappers import ImageToPyTorch
from miniworld_drstrategy.tools import generate_observations

# Create environment directly
env = NineRooms(room_size=15, door_size=2.5)

# Generate observations programmatically
output_dir = generate_observations("NineRooms", high_res_full_views=True)
```

## Generated Observations

Each variant generates 15 images:

- **3 Full environment views**: Complete environment layout (optional 512×512)
- **4 Partial observations**: POMDP views from different positions
- **4 Gymnasium observations**: Processed observations through wrappers
- **4 Render-on-pos examples**: Views from strategic positions

## Dependencies

### Core Dependencies
- `gymnasium>=0.26.0` - Modern RL environment interface
- `numpy>=1.20.0` - Numerical computing
- `opencv-python>=4.5.0` - Image processing
- `Pillow>=8.0.0` - Image I/O
- `PyOpenGL>=3.1.0` - OpenGL rendering

### Optional Dependencies
- `mujoco>=2.3.0` - Physics simulation (install with `pip install miniworld-drstrategy[mujoco]`)
- Development tools (install with `pip install miniworld-drstrategy[dev]`)

## Environment Variables

```bash
# For headless rendering (recommended)
export MUJOCO_GL=osmesa

# For DrStrategy package compatibility (if needed)
export PYTHONPATH="/path/to/drstrategy:/path/to/drstrategy/drstrategy"
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
isort src/
```

### Building Package

```bash
python -m build
```

## Features

✅ **Modern Python packaging**: Uses `pyproject.toml` and `src/` layout  
✅ **Well-structured codebase**: Logical module separation  
✅ **Command-line tools**: Easy-to-use CLI interface  
✅ **Type hints**: Full type annotation support  
✅ **Zero external dependencies**: Self-contained implementation  
✅ **Modern Gymnasium API**: Full compatibility with gym 0.26+  
✅ **High-resolution rendering**: Optional 512×512 layout views  
✅ **Multiple variants**: All three DrStrategy environment types  

## Paper Compliance

This implementation maintains high fidelity to the original DrStrategy paper specifications:
- 64×64×3 RGB observations
- 1000-step episode limits  
- Discrete action space (turn_left, turn_right, move_forward)
- Proper room connectivity and texturing
- POMDP partial observability

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite the original DrStrategy paper:

```bibtex
@article{drstrategy2023,
  title={DrStrategy: Model-Based Reinforcement Learning with Generalized State Space Models},
  author={...},
  journal={...},
  year={2023}
}
```