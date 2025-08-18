# Examples

This directory contains example scripts demonstrating the usage and performance of the Nine Rooms environments.

## Scripts

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

## Key Benefits Demonstrated

### Direct Rendering Performance
- **No Resize Overhead**: Environments render directly at requested size
- **Excellent Scaling**: 256x256 images only ~1.2x slower than 32x32 despite 64x more pixels
- **High Frame Rates**: 60-80 FPS for standard sizes (64x64, 128x128)
- **Flexible Resolutions**: Any custom size supported without performance penalty

### Environment Variants
- **NineRooms**: ~73 FPS average (fastest, simplest 3x3 grid)
- **SpiralNineRooms**: ~62 FPS average (moderate complexity)  
- **TwentyFiveRooms**: ~26 FPS average (most complex, 5x5 grid with 25 rooms)

### Performance Insights
The benchmarks show that the direct rendering approach provides:
- Consistent performance across all observation sizes
- Linear scaling with environment complexity (room count)
- Minimal memory overhead
- No CPU-intensive resizing operations

## Usage Tips

1. **For Research**: Use higher resolutions (256x256+) for detailed analysis
2. **For Training**: Use standard sizes (64x64, 128x128) for optimal speed
3. **For Visualization**: Use generate_observations.py to create datasets
4. **For Optimization**: Use benchmark_rendering.py to find optimal settings