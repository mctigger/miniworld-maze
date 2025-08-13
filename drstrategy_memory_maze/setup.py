from setuptools import setup, find_packages
import pathlib

__version__ = "1.0.0"

setup(
    name="drstrategy-memory-maze",
    version=__version__,
    author="Claude Code",
    author_email="noreply@anthropic.com",
    url="https://github.com/anthropics/claude-code",
    description="DrStrategy Memory Maze environments with Gymnasium compatibility",
    long_description=pathlib.Path('README.md').read_text() if pathlib.Path('README.md').exists() else "DrStrategy Memory Maze environments with Gymnasium compatibility",
    long_description_content_type='text/markdown',
    zip_safe=False,
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        'memory-maze',      # Our refactored memory-maze package
        'gymnasium>=0.29.0', # Latest Farama Foundations Gymnasium
        'dm_control>=1.0.14', # DeepMind Control Suite with latest MuJoCo
        'mujoco>=3.0.0',    # Latest MuJoCo physics engine
        'numpy>=1.21.0',    # Modern NumPy with typing support
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)