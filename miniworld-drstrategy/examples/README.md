# Examples

This directory contains example scripts demonstrating the usage and performance of the Nine Rooms environments.

## Scripts

### live_control_sequence.py ⭐ **RECOMMENDED**

Interactive keyboard control that generates image sequences. Perfect for exploring environments step-by-step.

**Usage:**
```bash
# Start interactive session with first-person view
python examples/live_control_sequence.py --variant NineRooms --size 128

# Use top-down view for better spatial understanding
python examples/live_control_sequence.py --variant NineRooms --size 128 --obs-level TOP_DOWN_PARTIAL

# Custom output directory
python examples/live_control_sequence.py --variant TwentyFiveRooms --output-dir my_exploration
```

**Controls:**
- `w/a/s/d` - Move forward/turn left/move backward/turn right
- `space` - Pick up object, `x` - Drop object, `e` - Toggle/activate
- `r` - Reset environment, `1/2/3` - Switch environment variants
- `q` - Quit

**Features:**
- Each action generates a PNG image with overlay information
- Works in any environment (headless or with display)
- No OpenGL conflicts - completely reliable
- Perfect for step-by-step environment exploration

### live_control_opencv.py

Real-time OpenCV-based live control with immediate visual feedback.

**Usage:**
```bash
# Real-time window with keyboard control
python examples/live_control_opencv.py --variant NineRooms --size 128
```

**Requirements:**
- Display environment (not headless)
- Click OpenCV window to focus for keyboard input
- Same controls as sequence version but with immediate feedback

### generate_observations.py

Generate comprehensive observation datasets for all environment variants.

**Usage:**
```bash
# Generate observations for NineRooms (default 64x64)
python examples/generate_observations.py NineRooms

# Generate observations for SpiralNineRooms with custom output directory
python examples/generate_observations.py SpiralNineRooms --output-dir my_observations

# Generate high-resolution full environment views (512x512)
python examples/generate_observations.py TwentyFiveRooms --high-res-full
```

**Generated observations:**
- Full environment views (with/without agent)
- Partial observations from different positions
- Standard Gymnasium observations after actions
- render_on_pos examples from strategic locations

### benchmark_rendering.py

Benchmark rendering performance across different observation sizes and environment variants.

**Usage:**
```bash
# Quick benchmark of NineRooms with common sizes
python examples/benchmark_rendering.py --quick --variant NineRooms

# Comprehensive benchmark of all variants and sizes
python examples/benchmark_rendering.py --all-variants --sizes 32 64 128 256 --steps 100

# Custom benchmark with specific parameters
python examples/benchmark_rendering.py --variant SpiralNineRooms --sizes 64 128 256 --steps 50
```

**Features:**
- FPS measurement across different observation sizes
- Performance comparison between environment variants
- Memory usage analysis
- Detailed timing statistics (min/max/avg/std)
- Scaling analysis (how performance scales with resolution)

### observation_level_demo.py

Demonstrates different observation levels and their visual differences.

**Usage:**
```bash
python examples/observation_level_demo.py
```

**Features:**
- Shows first-person, top-down partial, and top-down full views
- Saves comparison images for each observation level
- Demonstrates the impact of different observation settings

## Key Benefits Demonstrated

### Interactive Control
- **Real-time Exploration**: Navigate environments with keyboard controls
- **Multiple View Modes**: Switch between first-person and top-down perspectives
- **No OpenGL Conflicts**: Reliable operation in any environment
- **Image Sequence Generation**: Perfect for analysis and documentation

### Direct Rendering Performance
- **No Resize Overhead**: Environments render directly at requested size
- **Excellent Scaling**: 256x256 images only ~1.2x slower than 32x32 despite 64x more pixels
- **High Frame Rates**: 60-80 FPS for standard sizes (64x64, 128x128)
- **Flexible Resolutions**: Any custom size supported without performance penalty

### Environment Variants
- **NineRooms**: ~73 FPS average (fastest, simplest 3x3 grid)
- **SpiralNineRooms**: ~62 FPS average (moderate complexity)  
- **TwentyFiveRooms**: ~26 FPS average (most complex, 5x5 grid with 25 rooms)

## Usage Tips

1. **For Interactive Exploration**: Use `live_control_sequence.py` to navigate and explore
2. **For Real-time Control**: Use `live_control_opencv.py` if you have a display
3. **For Research**: Use higher resolutions (256x256+) for detailed analysis
4. **For Training**: Use standard sizes (64x64, 128x128) for optimal speed
5. **For Visualization**: Use `generate_observations.py` to create comprehensive datasets
6. **For Optimization**: Use `benchmark_rendering.py` to find optimal settings