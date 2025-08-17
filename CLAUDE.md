# Claude Configuration

## Allowed Commands

The following commands are pre-approved for execution without requiring user confirmation:

- `pytest` - Python testing framework
- `python` - Python interpreter 
- `grep` - Text search utility
- `ls` - List directory contents
- `source` - Load environment variables and activate virtual environments
- `PYTHONPATH` - Set Python module search path environment variable

## Python Virtual Environments

### Local Environment (Recommended)
Located at: `venv_local/`
- **Python Version**: 3.12.7
- **Purpose**: Complete DrStrategy Memory-Maze environment setup
- **Activation**: `source venv_local/bin/activate`

#### Installed Packages:
- **labmaze==1.0.6**: DeepMind Lab maze assets (prebuilt wheels)
- **dm_control==1.0.31**: DeepMind Control Suite
- **mujoco==3.3.5**: MuJoCo physics engine
- **memory-maze==1.0.2**: DrStrategy custom memory-maze (editable install)
- **matplotlib==3.10.5**: Plotting and visualization
- **gym==0.26.2**: OpenAI Gym (legacy)
- **numpy==2.3.2**: Numerical computing
- **scipy==1.16.1**: Scientific computing
- Plus all required dependencies (PyOpenGL, etc.)

### Deep Installation (Alternative)
Located at: `drstrategy/drstrategy_envs/drstrategy_envs/memory-maze/venv312/`
- **Python Version**: 3.12.7
- **Purpose**: Originally used for deep package structure testing
- **Activation**: `source drstrategy/drstrategy_envs/drstrategy_envs/memory-maze/venv312/bin/activate`

## MiniWorld DrStrategy Package

### Package Location
Located at: `miniworld-drstrategy/`
- **Purpose**: Well-structured Python package for Nine Rooms environments
- **Structure**: Modern `src/` layout with `pyproject.toml`
- **Environments**: NineRooms, SpiralNineRooms, TwentyFiveRooms

### Usage Examples

#### Command Line Interface
```bash
# Navigate to package directory
cd miniworld-drstrategy/src

# Set environment variables
export PYTHONPATH="/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy:/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy"
export MUJOCO_GL=osmesa

# Generate observations for any variant
python -m miniworld_drstrategy.tools.generate_observations NineRooms
python -m miniworld_drstrategy.tools.generate_observations SpiralNineRooms
python -m miniworld_drstrategy.tools.generate_observations TwentyFiveRooms --high-res-full

# Alternative: Use installed package (if pip installed)
generate-observations NineRooms --output-dir custom_output
```

#### Python API Usage
```bash
# From miniworld-drstrategy/src directory
cd miniworld-drstrategy/src
export PYTHONPATH="/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy:/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy"
export MUJOCO_GL=osmesa

python3 -c "
from miniworld_drstrategy import create_nine_rooms_env
env = create_nine_rooms_env(variant='NineRooms', size=64)
obs, info = env.reset()
print(f'Observation shape: {obs.shape}')
env.close()
"
```

### Package Structure
```
miniworld-drstrategy/
├── pyproject.toml                 # Modern Python packaging
├── README.md                      # Comprehensive documentation
└── src/
    └── miniworld_drstrategy/      # Main package
        ├── __init__.py            # Package exports
        ├── core/                  # Core 3D rendering engine
        ├── environments/          # Environment implementations
        ├── wrappers/              # Gymnasium wrappers
        └── tools/                 # CLI tools & observation generator
```

## Legacy Usage Examples

### Quick Start with Local Environment
```bash
# Activate local environment
source venv_local/bin/activate

# Set Python paths
export PYTHONPATH="/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy:/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy"

# Run environment info (no graphics required)
python run_memory_maze.py --env mzx7x7 --info

# Run comprehensive test
python test_full_working_env.py
```

## Important Instructions Reminders

Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.