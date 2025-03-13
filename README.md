# PyTopo3D: 3D SIMP Topology Optimization Framework for Python

![Optimized design with boundary conditions](assets/optimized_design_with_boundary_conditions.png)


A comprehensive Python implementation of 3D Topology Optimization based on SIMP (Solid Isotropic Material with Penalization) method. Unlike traditional MATLAB implementations, PyTopo3D brings the power of 3D SIMP-based optimization to the Python ecosystem with support for obstacle regions.

## Overview

This code performs 3D structural topology optimization using the SIMP (Solid Isotropic Material with Penalization) method. It is designed to be efficient by utilizing:

- Parallel solver (PyPardiso if available, otherwise SciPy's spsolve)
- Precomputed assembly mapping for fast matrix assembly
- Minimal plotting overhead
- Support for obstacle regions where no material can be placed
- Flexible obstacle configuration via JSON files

## Installation

1. Clone this repository:
```bash
git clone https://github.com/jihoonkim888/PyTopo3D.git
cd PyTopo3D
```

2. Create and activate the conda environment:
```bash
# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate pytopo3d

# Alternatively, use the provided activation script
source ./activate_pytopo3d.sh
```

For better performance, it's recommended to have PyPardiso installed, which requires:
- A compatible BLAS/LAPACK implementation
- Intel MKL (included with PyPardiso)

## Project Structure

```
PyTopo3D/
├── README.md                  # This file
├── requirements.txt           # Required dependencies
├── main.py                    # Command-line interface
├── run_with_obstacles.sh      # Script to run with obstacle configuration
├── examples/                  # Example files
│   ├── obstacles/             # Example obstacle configurations
├── pytopo3d/                  # Main package
│   ├── __init__.py
│   ├── core/                  # Core optimization functionality
│   │   ├── __init__.py
│   │   ├── optimizer.py       # Main top3d function
│   │   ├── compliance.py      # Element compliance calculation
│   │   ├── utils/             # Utility functions
│   │   └── obstacles.py       # Obstacle generation utilities
│   └── visualization/         # Visualization utilities
│       └── display.py         # 3D visualization functions
└── examples/                  # Example scripts
    ├── __init__.py
    ├── obstacle_example.py    # Example with single obstacle
    ├── multi_obstacle_example.py  # Example with multiple obstacles
    └── obstacles_config.json  # Example obstacle configuration file
```

## Usage

### Command-line Interface

To run a basic optimization:

```bash
python main.py --nelx 60 --nely 20 --nelz 10 --volfrac 0.3 --penal 3.0 --rmin 3.0
```

To include a default obstacle (cube in the middle):

```bash
python main.py --obstacle
```

To use a custom obstacle configuration from a JSON file:

```bash
python main.py --obstacle-config examples/obstacles_config.json
```

For full options:

```bash
python main.py --help
```

### As a Python Package

```python
import numpy as np
from pytopo3d.core.optimizer import top3d

# Define parameters
nelx, nely, nelz = 60, 20, 10
volfrac = 0.3
penal = 3.0
rmin = 3.0
disp_thres = 0.5

# Optional: Create an obstacle mask
obstacle_mask = np.zeros((nely, nelx, nelz), dtype=bool)
obstacle_mask[5:15, 20:40, 3:7] = True  # Example obstacle

# Run optimization
result = top3d(nelx, nely, nelz, volfrac, penal, rmin, disp_thres, obstacle_mask)

# Save result
np.save("optimized_design.npy", result)
```

### Using Obstacle Configuration Files

You can define complex obstacle configurations using JSON files:

```python
from pytopo3d.utils.obstacles import parse_obstacle_config_file

# Load obstacles from config file
shape = (nely, nelx, nelz)
obstacle_mask = parse_obstacle_config_file("path/to/config.json", shape)

# Use the mask in optimization
result = top3d(nelx, nely, nelz, volfrac, penal, rmin, disp_thres, obstacle_mask=obstacle_mask)
```

## Exporting Results as STL Files

You can export the final optimization result as an STL file for 3D printing or further analysis in CAD software.

### Command Line Arguments

```bash
python main.py --nelx 60 --nely 20 --nelz 10 \
               --volfrac 0.3 --penal 3.0 --rmin 3.0 \
               --export-stl \
               [--stl-level 0.5] \
               [--smooth-stl] \
               [--smooth-iterations 5]
```

- `--export-stl`: Flag to enable STL export of the final optimization result
- `--stl-level`: Contour level for the marching cubes algorithm (default: 0.5)
- `--smooth-stl`: Flag to apply Laplacian smoothing to the mesh (default: True)
- `--smooth-iterations`: Number of iterations for mesh smoothing (default: 5)

### How It Works

1. The optimized design is saved as a voxel representation (.npy file)
2. The voxel data is converted to a triangulated mesh using the marching cubes algorithm
3. Optional mesh smoothing is applied to improve the quality of the mesh
4. The mesh is exported as an STL file in the experiment results directory

This feature allows you to directly use the optimization results in CAD software or for 3D printing.

## Obstacle Configuration Format

The obstacle configuration file is a JSON file with the following structure:

```json
{
  "obstacles": [
    {
      "type": "cube",
      "center": [0.5, 0.5, 0.2],  // x, y, z as fractions [0-1]
      "size": 0.15                // single value for a cube
    },
    {
      "type": "sphere",
      "center": [0.25, 0.25, 0.6],
      "radius": 0.1
    },
    {
      "type": "cylinder",
      "center": [0.75, 0.5, 0.5],
      "radius": 0.08,
      "height": 0.7,
      "axis": 2                  // 0=x, 1=y, 2=z
    },
    {
      "type": "cube",
      "center": [0.25, 0.75, 0.5],
      "size": [0.15, 0.05, 0.3]  // [x, y, z] for a cuboid
    }
  ]
}
```

Supported obstacle types:
- `cube`: A cube or cuboid. Use `size` as a single value for a cube, or as `[x, y, z]` for a cuboid.
- `sphere`: A sphere. Use `radius` to set the size.
- `cylinder`: A cylinder. Use `radius`, `height`, and `axis` (0=x, 1=y, 2=z) to configure.

All positions are specified as fractions [0-1] of the domain size, making it easy to reuse configurations across different mesh resolutions.

## Examples

The `examples/` directory contains:

- `obstacle_example.py`: A basic example with a rectangular obstacle in the middle of the design domain.
- `multi_obstacle_example.py`: An example that uses a JSON configuration file to define multiple obstacles.
- `obstacles_config.json`: An example configuration file with multiple obstacle types.

To run the examples:

```bash
python -m examples.obstacle_example
python -m examples.multi_obstacle_example
```

Or use the provided script:

```bash
./run_with_obstacles.sh
```

## Configuration

The main optimization parameters are:

- `nelx`, `nely`, `nelz`: Number of elements in x, y, z directions
- `volfrac`: Volume fraction constraint (0.0-1.0)
- `penal`: Penalization power for SIMP method (typically 3.0)
- `rmin`: Filter radius for sensitivity filtering
- `disp_thres`: Display threshold for 3D visualization (elements with density > disp_thres are shown)
- `obstacle_mask`: Boolean 3D array marking regions where no material can be placed

## Acknowledgements

This code is adapted from [Liu & Tovar's MATLAB code](https://www.top3d.app/) for 3D topology optimization.

> K. Liu and A. Tovar, "An efficient 3D topology optimization code written in Matlab", Struct Multidisc Optim, 50(6): 1175-1196, 2014, doi:10.1007/s00158-014-1107-x
