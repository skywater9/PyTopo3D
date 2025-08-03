#!/usr/bin/env python3
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytopo3d",
    version="0.1.0",
    author="Jihoon Kim, Namwoo Kang",
    author_email="jihoonkim888@example.com",  # Replace with actual email
    description="3D SIMP Topology Optimization Framework for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jihoonkim888/PyTopo3D",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "scikit-image",
        "networkx",
        "pillow",
        "plotly",
        "tqdm",
        "trimesh",
        "psutil",
        "pypardiso",
        "pyyaml"
    ],
    extras_require={
        "dev": [
            "jupyter",
            "jupyterlab",
        ],
        "gpu": [
            "cupy-cuda12x",  # Replace with appropriate CUDA version as needed
        ],
    },
    include_package_data=True,
)
