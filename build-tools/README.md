# Build Tools

This directory contains files related to the Python package build process for PyTopo3D.

## Files

- **setup.py**: Legacy setup script for setuptools
- **pyproject.toml**: Modern package specification following PEP 517/518
- **MANIFEST.in**: Specifies additional files to include in package distribution

## Usage

For developers working on the package:

```bash
# Navigate to the project root (not this directory)
cd ..

# Install in development mode
pip install -e .

# Build the package
python -m build

# Create distribution packages
python -m build --sdist --wheel
```

For users, the package can be installed directly from PyPI:
```bash
pip install pytopo3d
``` 