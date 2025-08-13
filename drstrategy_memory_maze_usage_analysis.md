# DrStrategy Memory-Maze Environment Usage Analysis

This document analyzes how the memory-maze environment is used within the DrStrategy framework, including configurations, training setups, and evaluation protocols.

## Environment Configuration System

### 1. Domain and Task Naming Convention

DrStrategy uses a hierarchical naming system for memory-maze environments:

**Domain**: `rnavmemorymaze3D` (3D navigation in memory maze)
**Task Format**: `rnavmemorymaze3Ddifffloortextureskynowalltexture_{maze_type}`

Where `{maze_type}` includes:
- `mzx7x7` - Maze layout 7x7
- `mzx15x15` - Maze layout 15x15
- `4x7x7` - Four rooms 7x7 
- `4x15x15` - Four rooms 15x15
- `8x30x30` - Eight rooms 30x30

### 2. Environment Feature Flags

The domain string encodes multiple configuration flags:
- `difffloortexture` - Enable different floor textures per room
- `sky` - Enable skybox textures
- `nowalltexture` - Disable wall pattern textures
- `highwalls` - Override with high walls

Example: `rnavmemorymaze3Ddifffloortextureskynowalltexture_mzx7x7`
- Different floor textures: ✓
- Sky textures: ✓
- Wall textures: ✗
- High walls: ✗

## Supported Environment Variants

### 1. Maze Layouts
- **`mzx7x7`**: 7x7 maze with complex pathway structure
- **`mzx15x15`**: 15x15 maze with increased complexity

### 2. Room-Based Layouts
- **`4x7x7`**: Four rooms in 7x7 grid (FourRooms7x7)
- **`4x15x15`**: Four rooms in 15x15 grid (FourRooms15x15)  
- **`8x30x30`**: Eight rooms in 30x30 grid (EightRooms30x30)

## Training Configuration

### 1. Base Configuration (`memorymaze_pixels.yaml`)
```yaml
obs_type: pixels
action_repeat: 1
encoder: 
  mlp_keys: '$^'
  cnn_keys: 'observation'
  norm: none
  cnn_depth: 48
  cnn_kernels: [4, 4, 4, 4]
  mlp_layers: [400, 400, 400, 400]
decoder:
  mlp_keys: '$^'
  cnn_keys: 'observation'
  norm: none
  cnn_depth: 48
  cnn_kernels: [5, 5, 6, 6]
  mlp_layers: [400, 400, 400, 400]
replay.capacity: 1e6
pos_decoder_shapes: 
  keys: ['position', 'direction']
  values: [3, 2]
```

### 2. Training Parameters

#### For `mzx7x7` (7x7 Maze):
- **Training frames**: 2,000,010
- **Seed frames**: 1,000
- **Landmark dimension**: 64
- **Code dimension**: 16
- **Reset behavior**: Every 500 frames
- **SE max step**: 100
- **Time limit**: Variable (500 steps default)

#### For `mzx15x15` (15x15 Maze):
- **Training frames**: 4,000,010
- **Seed frames**: 1,000
- **Landmark dimension**: 128
- **Code dimension**: 16
- **Reset behavior**: Every 1,000 frames
- **SE max step**: 200
- **Time limit**: Variable (1,000 steps default)

### 3. Agent Configuration
- **Agent**: `p2eDrStrategy` (Plan2Explore with DrStrategy)
- **Skilled exploration**: Enabled
- **Landmark-to-landmark**: Disabled
- **Achiever training**: Enabled
- **Non-episodic**: Enabled
- **Achiever sampling**: Trajectory-based

## Environment Implementation Details

### 1. Custom MemoryMaze Class (`envs.py:623-719`)

Key features:
- **Rendering**: Uses 'osmesa' for headless GPU rendering
- **Layout selection**: Automatic based on task name
- **Observables**: Global observables enabled by default
- **Action space**: Support for both discrete and continuous actions
- **Room tracking**: Automatic computation of room boundaries
- **Goal tracking**: Room-specific goal success ratios

### 2. Observation Space
```python
# Standard observations include:
- 'image': RGB camera observation
- 'agent_pos': 3D position [x, y, z]
- 'agent_dir': 2D direction vector [dx, dy]
- 'target_pos': Target position
- 'target_color': RGB target color
- 'maze_layout': Binary maze layout
- 'top_view': Top-down camera view (when enabled)
```

### 3. Data Specifications
For memory-maze tasks, the replay buffer uses:
```python
data_specs = (
    observation_spec(),    # Environment observations
    action_spec(),        # Action space
    Array((1,), float32, 'reward'),     # Scalar reward
    Array((1,), float32, 'discount'),   # Discount factor
    Array((3,), float32, 'position'),   # 3D position
    Array((2,), float32, 'direction'),  # 2D direction
)
```

## Data Generation Pipeline

### 1. Offline Dataset Generation
Multiple scripts in `drstrategy_envs/memory-maze/data/` for generating training data:

- **Single room**: `single_room_*` scripts
- **Two rooms**: `two_rooms_*` scripts  
- **Four rooms**: `four_rooms_*` scripts
- **Multi-room**: `ten_rooms_*`, `twenty_rooms_*` scripts

### 2. Data Generation Features
- **Room boundary detection**: Automatic room classification
- **Traversal counting**: Two-traversal requirements
- **Coordinate reset**: Position-based resets
- **Probing data**: Specialized evaluation datasets

## Evaluation Protocol

### 1. Evaluation Scripts
- `eval_drstrategy_mzx7x7.sh`: 7x7 maze evaluation
- `eval_drstrategy_mzx15x15.sh`: 15x15 maze evaluation

### 2. Evaluation Configuration
Uses same base configuration as training with additional evaluation-specific parameters for model loading and testing.

## Integration with DrStrategy Framework

### 1. Hierarchical RL Components
- **Skilled Explorer**: Landmark-based exploration with configurable max steps
- **Achiever**: Goal-conditioned policy training
- **Landmark System**: Compressed spatial representations

### 2. Memory and Navigation Features
- **Position tracking**: 3D coordinates with direction vectors
- **Room-aware navigation**: Automatic room boundary computation
- **Goal pose management**: Predefined goal positions per layout
- **Success tracking**: Room-specific achievement ratios

### 3. Visual Enhancements
- **Custom textures**: Programmable wall and floor colors
- **Top-down views**: Bird's eye perspective for analysis
- **Multiple camera modes**: First-person and overhead cameras
- **Landmark visualization**: Enhanced visual feedback

## Research Applications

### 1. Long-term Memory Evaluation
- Extended episode lengths (500-1000+ steps)
- Non-episodic training for continual learning
- Multiple room configurations for complexity scaling

### 2. Hierarchical Navigation
- Landmark-based spatial representations
- Multi-room traversal requirements  
- Goal-conditioned policy learning

### 3. Visual Representation Learning
- Pixel-based observations with CNN processing
- Texture variations for generalization
- Position and direction encoding

## Summary

DrStrategy uses the memory-maze environment as a testbed for hierarchical reinforcement learning with the following key characteristics:

1. **Five main environment variants** (mzx7x7, mzx15x15, 4x7x7, 4x15x15, 8x30x30)
2. **Configurable visual features** (textures, skyboxes, wall patterns)
3. **Hierarchical RL integration** with landmark-based exploration
4. **Comprehensive data generation pipeline** for offline learning
5. **Extensive evaluation protocols** with room-aware metrics
6. **Non-episodic training** for continual learning scenarios

The environment serves as a complex navigation and memory task where agents must learn to efficiently traverse multi-room mazes while maintaining spatial memory over extended time horizons.