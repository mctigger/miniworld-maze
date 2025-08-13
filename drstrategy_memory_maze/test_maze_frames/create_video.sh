#!/bin/bash
# Script to convert frames to video using ffmpeg

# Create video at 20 FPS
ffmpeg -r 20 -i frame_%06d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -r 20 -pix_fmt yuv420p DrStrategy_MemoryMaze_4x7x7_v0_visualization.mp4

echo "Video created: DrStrategy_MemoryMaze_4x7x7_v0_visualization.mp4"

# Alternative: Create GIF (smaller file, lower quality)
ffmpeg -r 5 -i frame_%06d.png -vf "scale=800:-1,palettegen" palette.png
ffmpeg -r 5 -i frame_%06d.png -i palette.png -lavfi "scale=800:-1,paletteuse" DrStrategy_MemoryMaze_4x7x7_v0_visualization.gif
rm palette.png

echo "GIF created: DrStrategy_MemoryMaze_4x7x7_v0_visualization.gif"
