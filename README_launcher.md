# DrStrategy Memory-Maze Environment Launcher

A clean, refactored script to launch and interact with DrStrategy Memory-Maze environments.

## Quick Start

```bash
# Activate the Python 3.12 environment
source drstrategy/drstrategy_envs/drstrategy_envs/memory-maze/venv312/bin/activate

# Set the PYTHONPATH
export PYTHONPATH="/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy:/home/tim/Projects/drstrategy_memory-maze_differences/drstrategy/drstrategy"

# View environment information (no graphics required)
python run_memory_maze.py --env mzx7x7 --info

# Run an environment (requires graphics support)
python run_memory_maze.py --env mzx7x7 --steps 50
```

## Environment Variants

| Variant   | Description              | Layout        | Max Steps |
|-----------|-------------------------|---------------|-----------|
| `mzx7x7`  | 7x7 maze navigation     | Maze7x7       | 500       |
| `4x7x7`   | 7x7 four-room navigation| FourRooms7x7  | 500       |
| `mzx15x15`| 15x15 maze navigation   | Maze15x15     | 1000      |
| `4x15x15` | 15x15 four-room navigation| FourRooms15x15| 1000    |
| `8x30x30` | 30x30 eight-room navigation| EightRooms30x30| 2000   |

## Usage Examples

```bash
# Show help
python run_memory_maze.py --help

# View environment info (works without graphics)
python run_memory_maze.py --env mzx7x7 --info
python run_memory_maze.py --env 4x15x15 --info

# Run environment with visual features
python run_memory_maze.py --env mzx7x7 --steps 100 --render

# Interactive mode (step-by-step)
python run_memory_maze.py --env 4x7x7 --steps 20 --interactive

# Run with custom time limits
python run_memory_maze.py --env mzx15x15 --steps 50 --time-limit 200
```

## Command Line Options

- `--env`: Environment variant to run (required)
- `--steps`: Number of steps to run (default: 100) 
- `--time-limit`: Episode time limit (default: 100)
- `--render`: Enable visual features (wall patterns, textures, etc.)
- `--interactive`: Interactive mode - wait for Enter between steps
- `--info`: Show detailed environment information only
- `--headless`: Run without graphics (limited support)

## Graphics Requirements

The memory-maze environments require OpenGL support for visual observations. If you encounter graphics errors:

1. Ensure you have proper OpenGL support installed
2. Use X11 forwarding if running remotely: `ssh -X`
3. Install mesa-utils: `sudo apt-get install mesa-utils`
4. Use `--info` flag to explore environments without graphics

## Environment Setup

The script automatically:
- Sets up Python paths for DrStrategy imports
- Configures memory-maze environment parameters
- Handles action space conversion (numpy arrays)
- Provides detailed error messages and suggestions

## Action Space

All environments use `Discrete(6)` action space:
- 0: Move forward
- 1: Move backward  
- 2: Turn left
- 3: Turn right
- 4: Strafe left
- 5: Strafe right

## Observation Space

Observations include:
- `image`: RGB images (64x64x3)
- `position`: Agent 3D position 
- `direction`: Agent orientation
- `target_pos`: Target location
- `maze_layout`: Environment layout info

## Based On

This launcher is based on the successful integration of:
- **labmaze**: Real DeepMind Lab assets and textures
- **dm_control + MuJoCo**: Full physics simulation
- **DrStrategy**: Custom memory-maze implementation
- **Python 3.12**: With prebuilt wheels compatibility