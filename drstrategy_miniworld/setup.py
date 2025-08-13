from setuptools import setup, find_packages

setup(
    name="drstrategy-miniworld",
    version="1.0.0",
    description="DrStrategy extensions for Farama-Foundation Miniworld",
    packages=find_packages(),
    install_requires=[
        "miniworld>=2.1.0",
        "gymnasium>=0.29.0",
        "numpy>=1.22.0",
        "pyglet>=1.5.27,<2.0",
    ],
    python_requires=">=3.8",
)