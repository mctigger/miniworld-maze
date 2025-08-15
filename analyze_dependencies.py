#!/usr/bin/env python3
"""
Analyze what files are actually needed for our Nine Rooms implementation.
"""

import sys
import ast
import os

def analyze_imports(file_path):
    """Analyze imports in a Python file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
                for alias in node.names:
                    if alias.name != '*':
                        imports.append(f"{node.module}.{alias.name}" if node.module else alias.name)
        
        return imports
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return []

def trace_dependencies():
    """Trace dependencies from our main implementation."""
    miniworld_path = "/home/tim/Projects/drstrategy_memory-maze_differences/nine_rooms_pure_gymnasium_env/miniworld_gymnasium"
    
    # Start with core files we know we need
    needed_files = set()
    
    # Core files
    core_files = [
        "envs/roomnav.py",  # Nine Rooms environment
        "miniworld.py",     # Base environment
        "entity.py",        # Box and other entities
        "params.py",        # Parameters
        "random.py",        # Random utilities
        "opengl.py",        # OpenGL rendering
        "objmesh.py",       # Object meshes
        "math.py",          # Math utilities
        "__init__.py",      # Package init
        "envs/__init__.py", # Envs init
    ]
    
    print("Analyzing dependencies...")
    
    for file in core_files:
        full_path = os.path.join(miniworld_path, file)
        if os.path.exists(full_path):
            imports = analyze_imports(full_path)
            needed_files.add(file)
            print(f"\n{file}:")
            for imp in imports:
                print(f"  - {imp}")
        else:
            print(f"File not found: {full_path}")
    
    # Check what's actually in envs directory
    envs_dir = os.path.join(miniworld_path, "envs")
    if os.path.exists(envs_dir):
        env_files = [f for f in os.listdir(envs_dir) if f.endswith('.py')]
        print(f"\nEnvironment files available: {env_files}")
        
        # We only need roomnav.py for Nine Rooms
        needed_env_files = ['__init__.py', 'roomnav.py']
        print(f"Environment files we actually need: {needed_env_files}")
    
    # Check textures and meshes needed
    textures_path = os.path.join(miniworld_path, "textures")
    meshes_path = os.path.join(miniworld_path, "meshes")
    
    print(f"\nChecking asset directories...")
    print(f"Textures directory exists: {os.path.exists(textures_path)}")
    print(f"Meshes directory exists: {os.path.exists(meshes_path)}")
    
    # Determine minimal asset set
    essential_assets = {
        'textures': [
            'beige_1.png', 'lightbeige_1.png', 'lightgray_1.png',
            'copperred_1.png', 'skyblue_1.png', 'lightcobaltgreen_1.png',
            'oakbrown_1.png', 'navyblue_1.png', 'cobaltgreen_1.png'
        ],
        'meshes': [
            # Only need basic box rendering, no complex meshes for Nine Rooms
        ]
    }
    
    print(f"\nEssential assets needed:")
    for asset_type, files in essential_assets.items():
        print(f"{asset_type}: {files}")
    
    return needed_files, essential_assets

if __name__ == "__main__":
    needed_files, essential_assets = trace_dependencies()