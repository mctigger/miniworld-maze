# DrStrategy Memory-Maze Modifications Analysis

This document analyzes the differences between the original `memory-maze` package by jurgisp and the modified version used in `drstrategy/drstrategy_envs/drstrategy_envs/memory-maze/`.

## Overview

The DrStrategy team has created a customized fork of the Memory Maze environment with significant extensions for their reinforcement learning research. The modifications include new environment variants, custom maze layouts, enhanced visualization features, and additional observables.

## Key Differences

### 1. Version Information
- **Original**: Version 1.0.3
- **DrStrategy**: Version 1.0.2 (downgraded)

### 2. New Files Added

#### 2.1 Custom Task Implementation (`custom_task.py`)
- **Size**: 1,042 lines (major addition)
- **Purpose**: Comprehensive custom environment implementation
- **Key Features**:
  - Custom texture classes (`WallNoTexture`, `CWallTexture`, `CFixedFloorTexture`)
  - Custom colormap with 10+ predefined colors
  - Enhanced maze arena (`CMazeWithTargetsArenaFixedLayout`)
  - Custom task class (`CMemoryMazeTask`)
  - Multiple predefined maze layouts (7x7, 15x15, 30x30)
  - Layout classes for different room configurations

#### 2.2 Data Generation Scripts (`data/` directory)
- Multiple Python scripts for generating training data
- Scripts for different room configurations (single, two, four, ten, twenty rooms)
- Batch generation scripts with shell commands
- Probing and evaluation data generation

### 3. Enhanced Environment Registration (`__init__.py`)

#### 3.1 Additional Environment Variants
The DrStrategy version registers **108 additional environment variants** compared to the original's **60 variants**:

- `MemoryMaze-single-room-3x3-v0`
- `MemoryMaze-two-rooms-3x7-v0` 
- `MemoryMaze-two-rooms-3x7-fixed-layout-v0`
- `MemoryMaze-four-rooms-7x7-fixed-layout-v0`
- `MemoryMaze-four-rooms-7x7-fixed-layout-random-goals-v0`
- `MemoryMaze-twenty-rooms-7x39-fixed-layout-random-goals-v0`

Each variant includes both standard and top-camera versions.

### 4. Task Extensions (`tasks.py`)

#### 4.1 New Task Functions
- `memory_maze_single_room_3x3()`
- `memory_maze_two_rooms_3x7()`
- `memory_maze_two_rooms_3x7_fixed_layout()`
- `memory_maze_four_rooms_7x7_fixed_layout()`
- `memory_maze_four_rooms_7x7_fixed_layout_random_goals()`
- `memory_maze_twenty_rooms_7x39_fixed_layout_random_goals()`

#### 4.2 Predefined Layouts
Multiple ASCII maze layouts:
- `TWO_ROOMS_3x7_LAYOUT`
- `TWO_ROOMS_3x7_LAYOUT_RANDOM_GOALS`
- `FOUR_ROOMS_7x7_LAYOUT`
- `FOUR_ROOMS_7x7_LAYOUT_RANDOM_GOALS`
- `TWENTY_ROOMS_7x39_LAYOUT`

### 5. Helper Function Extensions (`helpers.py`)
- Added `to_onehot()` function for one-hot encoding
- Imported from pydreamer preprocessing module

### 6. GUI Modifications (`gui/run_gui.py`)
- Removed Linux-specific rendering logic
- Simplified MUJOCO_GL configuration
- Changed from platform-specific to uniform `glfw` rendering

### 7. Removed Features
- **6-color variants**: The original's six-color maze variants (`6CL-v0`, `6CL-Top-v0`, `6CL-ExtraObs-v0`) are not present in the DrStrategy version

## Technical Enhancements

### 1. Visual Customizations
- Custom color palettes using matplotlib's 'tab20' colormap
- Enhanced texture systems with programmable wall and floor colors
- Support for different floor textures per room
- Configurable skybox textures

### 2. Layout Control
- Fixed maze layouts vs. procedurally generated
- Precise room boundary definitions
- Custom spawn and goal positioning
- Support for various room configurations (1-20+ rooms)

### 3. Observational Enhancements
- Top-down camera views with configurable resolutions (256x256, 480x480)
- Extra observables for research analysis
- Enhanced global observables support

### 4. Research-Oriented Features
- Data generation pipeline for offline RL research
- Batch processing scripts for large-scale experiments
- Probing and evaluation-specific environment variants
- Integration with DrStrategy's training pipeline

## Use Case Focus

### Original Memory-Maze
- General-purpose memory evaluation environment
- Standard RL benchmarking
- Academic research baseline
- Broad compatibility

### DrStrategy Version
- Specialized for hierarchical RL research
- Custom data generation for specific experiments
- Enhanced visualization for analysis
- Integration with DrStrategy training framework
- Focus on multi-room navigation tasks

## Summary

The DrStrategy modification represents a substantial expansion of the original Memory-Maze environment, adding approximately **1,000+ lines of custom code** and **48 additional environment variants**. The modifications are research-focused, emphasizing:

1. **Custom maze layouts** with precise control over room structures
2. **Enhanced visual customization** with programmable textures and colors
3. **Data generation pipeline** for offline RL research
4. **Additional environment variants** for systematic evaluation
5. **Integration capabilities** with the broader DrStrategy framework

The changes maintain backward compatibility with the core Memory-Maze API while extending functionality significantly for specialized reinforcement learning research applications.