# Claude Configuration

## Project Overview
MiniWorld DrStrategy package - Nine Rooms environment implementations for reinforcement learning research.

## Allowed Commands

The following commands are pre-approved for execution:

- `pytest` - Python testing framework
- `python` - Python interpreter 
- `pip` - Package installer
- `grep` - Text search utility
- `ls` - List directory contents
- `cd` - Change directory
- `source` - Load environment variables and activate virtual environments
- `PYTHONPATH` - Set Python module search path environment variable
- `MUJOCO_GL` - MuJoCo rendering backend environment variable
- `black` - Python code formatter
- `isort` - Import sorter
- `flake8` - Python linter

## Package Structure

Located at: `miniworld-drstrategy/`
- **Purpose**: Nine Rooms environments (NineRooms, SpiralNineRooms, TwentyFiveRooms)
- **Structure**: Modern `src/` layout with `pyproject.toml`
- **Python Version**: 3.8+ (supports 3.8, 3.9, 3.10, 3.11, 3.12)

### Key Directories
```
miniworld-drstrategy/
├── pyproject.toml                 # Modern Python packaging
├── README.md                      # Documentation
├── CLAUDE.md                      # This configuration file
├── examples/                      # Usage examples and demos
│   ├── benchmark_rendering.py     # Performance benchmarking
│   ├── generate_observations.py   # Observation generation example
│   └── observation_level_demo.py  # Observation level demonstration
└── src/
    └── miniworld_drstrategy/      # Main package
        ├── core/                  # 3D rendering engine
        │   ├── constants.py       # Environment constants
        │   ├── observation_types.py # Observation level enums
        │   └── miniworld_gymnasium/ # Core 3D engine
        ├── environments/          # Environment implementations
        ├── tools/                 # CLI tools & observation generator
        └── wrappers/              # Gymnasium wrappers
```

## Dependencies

### Core Dependencies
- `gymnasium>=1.0.0` - RL environment interface
- `numpy>=1.20.0` - Numerical computing
- `opencv-python>=4.5.0` - Computer vision
- `Pillow>=8.0.0` - Image processing
- `PyOpenGL>=3.1.0` - OpenGL rendering

### Development Dependencies
- `pytest>=6.0` - Testing framework
- `black` - Code formatter
- `isort` - Import sorter
- `flake8` - Linter

### Optional Dependencies
- `mujoco>=2.3.0` - Physics engine (optional)

## Usage Examples

### Command Line Interface
```bash
# Navigate to package directory
cd miniworld-drstrategy/

# Install package in development mode
pip install -e .

# Generate observations (using installed script)
generate-observations NineRooms
generate-observations SpiralNineRooms  
generate-observations TwentyFiveRooms --high-res-full

# Alternative: Run examples directly
MUJOCO_GL=egl python examples/generate_observations.py NineRooms --output-dir test_output
MUJOCO_GL=egl python examples/benchmark_rendering.py --all-variants
MUJOCO_GL=egl python examples/observation_level_demo.py

# Run tests
pytest

# Format code
black src/
isort src/

# Lint code
flake8 src/
```

### Python API Usage
```python
from miniworld_drstrategy import create_nine_rooms_env, ObservationLevel

# Create environment with different variants
env = create_nine_rooms_env(variant='NineRooms', size=64)
obs, info = env.reset()
print(f'Observation shape: {obs.shape}')

# Use different observation levels
env2 = create_nine_rooms_env(
    variant='SpiralNineRooms', 
    obs_level=ObservationLevel.FIRST_PERSON, 
    size=128
)

# Test render_on_pos functionality
render_obs = env.render_on_pos([15.0, 0.0, 15.0])
print(f'Render observation shape: {render_obs.shape}')

env.close()
env2.close()
```

## Development Workflow

1. **Code Style**: Uses Black formatter with 88 character line length
2. **Import Sorting**: Uses isort with Black profile
3. **Testing**: Uses pytest with coverage reporting
4. **Linting**: Uses flake8 for code quality

## Environment Variables

When working with this package, you may need:
```bash
export MUJOCO_GL=egl     # For headless rendering (recommended)
export MUJOCO_GL=osmesa  # Alternative headless rendering
export MUJOCO_GL=glfw    # For desktop rendering with window
```

## Available Environments

### Variants
- **NineRooms**: Standard 3x3 grid layout
- **SpiralNineRooms**: Spiral-shaped room connections
- **TwentyFiveRooms**: Larger 5x5 grid layout

### Observation Levels
- **TOP_DOWN_PARTIAL**: Partial top-down view (default)
- **TOP_DOWN_FULL**: Complete top-down view
- **FIRST_PERSON**: Agent's first-person perspective

### Supported Sizes
- 64x64 (default, fastest)
- 128x128 (balanced quality/performance)
- 256x256 (high quality, slower)

## Important Instructions

- ALWAYS prefer editing existing files over creating new ones
- Follow the existing code style and patterns
- Use pytest for testing
- Run formatters (black, isort) and linter (flake8) before commits
- NEVER create documentation files unless explicitly requested