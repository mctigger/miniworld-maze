# DrStrategy Memory Maze

A modern, clean implementation of DrStrategy Memory Maze environments with full Gymnasium compatibility.

## Features

- **🎯 Simple & Clean**: Minimal, well-documented codebase following best practices
- **🏃 Gymnasium Ready**: Full compatibility with latest Farama Foundations Gymnasium (≥0.29.0)
- **🏢 Multiple Mazes**: Five different maze layouts for varied memory challenges
- **🎮 Flexible Actions**: Support for both discrete and continuous action spaces
- **🔧 Modern Python**: Type hints, dataclasses, and Python 3.9+ features
- **✅ Well Tested**: Comprehensive test suite with pytest

## Installation

### Development Installation
```bash
# Clone and install in development mode
git clone <repository-url>
cd drstrategy_memory_maze
pip install -e .
```

### With Development Tools
```bash
# Install with testing and linting tools
pip install -e ".[dev]"
```

## Quick Start

```python
import gymnasium as gym
import drstrategy_memory_maze

# Create environment
env = gym.make('DrStrategy-MemoryMaze-4x7x7-v0')

# Use it like any Gym environment
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## Available Environments

| Environment ID | Description | Layout | Max Steps |
|---------------|-------------|---------|-----------|
| `DrStrategy-MemoryMaze-4x7x7-v0` | Four rooms, small | 7×7 | 500 |
| `DrStrategy-MemoryMaze-4x15x15-v0` | Four rooms, large | 15×15 | 1000 |
| `DrStrategy-MemoryMaze-8x30x30-v0` | Eight rooms, extra large | 30×30 | 2000 |
| `DrStrategy-MemoryMaze-mzx7x7-v0` | Custom maze, small | 7×7 | 500 |
| `DrStrategy-MemoryMaze-mzx15x15-v0` | Custom maze, large | 15×15 | 1000 |

## Direct Environment Creation

```python
from drstrategy_memory_maze import MemoryMaze

# Create environment directly with more control
env = MemoryMaze(
    task='4x7x7',           # Maze layout
    discrete_actions=True   # Action type
)
```

## Architecture

The package consists of just two main modules:

- **`maze_layouts.py`**: Defines maze layouts and metadata
- **`envs.py`**: Main environment class wrapping memory_maze

This design follows modern Python best practices:
- Type hints throughout for better IDE support and error detection
- Immutable dataclasses for configuration
- Comprehensive error handling and validation
- Full test coverage with pytest
- Clean separation of concerns between layout definitions and environment logic

## Dependencies

### Core Dependencies
- **memory-maze**: Our refactored memory-maze package
- **gymnasium ≥0.29.0**: Latest Farama Foundations Gymnasium
- **dm_control ≥1.0.14**: DeepMind Control Suite with latest features
- **mujoco ≥3.0.0**: Latest MuJoCo physics engine
- **numpy ≥1.21.0**: Modern NumPy with typing support

### Development Dependencies
- **pytest ≥7.0.0**: Testing framework
- **mypy ≥1.0.0**: Static type checking
- **black ≥22.0.0**: Code formatting
- **ruff ≥0.1.0**: Fast Python linter

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=drstrategy_memory_maze

# Run specific test file
pytest tests/test_envs.py
```

### Code Quality
```bash
# Format code
black drstrategy_memory_maze/

# Lint code
ruff drstrategy_memory_maze/

# Type checking
mypy drstrategy_memory_maze/
```

### Package Structure
```
drstrategy_memory_maze/
├── drstrategy_memory_maze/
│   ├── __init__.py          # Package initialization and registration
│   ├── envs.py             # Main MemoryMaze environment class
│   └── maze_layouts.py     # Maze layout definitions and utilities
├── tests/                   # Comprehensive test suite
├── pyproject.toml          # Modern Python packaging configuration
└── py.typed                # Type information marker
```

## License

MIT License