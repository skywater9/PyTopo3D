"""
Utilities for creating obstacle masks for topology optimization.

This module provides functions to create various obstacle shapes and to parse
obstacle configuration files.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

def create_cube_obstacle(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    size: Union[float, Tuple[float, float, float]]
) -> np.ndarray:
    """
    Create a cube or cuboid obstacle mask.
    
    Parameters
    ----------
    shape : tuple of int
        Shape of the design domain (nely, nelx, nelz).
    center : tuple of float
        Center coordinates of the obstacle as fractions [0-1] of domain size (x, y, z).
    size : float or tuple of float
        Size of the obstacle as fraction of domain size. If a single value, creates a cube.
        If three values, creates a cuboid with (x, y, z) dimensions.
        
    Returns
    -------
    np.ndarray
        Boolean mask with True values where the obstacle is located.
    """
    nely, nelx, nelz = shape
    
    # Convert center from fraction to indices
    cx = int(center[0] * nelx)
    cy = int(center[1] * nely)
    cz = int(center[2] * nelz)
    
    # Convert size from fraction to number of elements
    if isinstance(size, (int, float)):
        half_x = int(size * nelx / 2)
        half_y = int(size * nely / 2)
        half_z = int(size * nelz / 2)
    else:
        half_x = int(size[0] * nelx / 2)
        half_y = int(size[1] * nely / 2)
        half_z = int(size[2] * nelz / 2)
    
    # Calculate bounds with clipping to prevent out-of-bounds indices
    x_lo, x_hi = max(0, cx - half_x), min(nelx, cx + half_x)
    y_lo, y_hi = max(0, cy - half_y), min(nely, cy + half_y)
    z_lo, z_hi = max(0, cz - half_z), min(nelz, cz + half_z)
    
    # Create the mask with correct dimensions (nely, nelx, nelz)
    mask = np.zeros((nely, nelx, nelz), dtype=bool)
    mask[y_lo:y_hi, x_lo:x_hi, z_lo:z_hi] = True
    
    return mask

def create_sphere_obstacle(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    radius: float
) -> np.ndarray:
    """
    Create a spherical obstacle mask.
    
    Parameters
    ----------
    shape : tuple of int
        Shape of the design domain (nely, nelx, nelz).
    center : tuple of float
        Center coordinates of the obstacle as fractions [0-1] of domain size (x, y, z).
    radius : float
        Radius of the sphere as a fraction of the smallest domain dimension.
        
    Returns
    -------
    np.ndarray
        Boolean mask with True values where the obstacle is located.
    """
    nely, nelx, nelz = shape
    
    # Convert center from fraction to indices
    cx = int(center[0] * nelx)
    cy = int(center[1] * nely)
    cz = int(center[2] * nelz)
    
    # Convert radius from fraction to number of elements
    r = int(radius * min(nelx, nely, nelz))
    
    # Create coordinates relative to center
    y_indices, x_indices, z_indices = np.ogrid[:nely, :nelx, :nelz]
    y_distance = y_indices - cy
    x_distance = x_indices - cx
    z_distance = z_indices - cz
    
    # Calculate squared distance from center
    dist_squared = x_distance**2 + y_distance**2 + z_distance**2
    
    # Create the mask
    mask = np.zeros((nely, nelx, nelz), dtype=bool)
    mask[dist_squared <= r**2] = True
    
    return mask

def create_cylinder_obstacle(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    radius: float,
    height: float,
    axis: int = 2
) -> np.ndarray:
    """
    Create a cylindrical obstacle mask.
    
    Parameters
    ----------
    shape : tuple of int
        Shape of the design domain (nely, nelx, nelz).
    center : tuple of float
        Center coordinates of the obstacle as fractions [0-1] of domain size (x, y, z).
    radius : float
        Radius of the cylinder as a fraction of the smallest domain dimension in the
        plane perpendicular to the cylinder axis.
    height : float
        Height of the cylinder as a fraction of the domain dimension along the cylinder axis.
    axis : int, optional
        Axis along which the cylinder extends (0=x, 1=y, 2=z). Default is 2 (z-axis).
        
    Returns
    -------
    np.ndarray
        Boolean mask with True values where the obstacle is located.
    """
    nely, nelx, nelz = shape
    dimensions = [nelx, nely, nelz]
    
    # Convert center from fraction to indices
    cx = int(center[0] * nelx)
    cy = int(center[1] * nely)
    cz = int(center[2] * nelz)
    center_idx = [cx, cy, cz]
    
    # Determine perpendicular dimensions based on axis
    perp_dims = [i for i in range(3) if i != axis]
    perp_shape = [dimensions[i] for i in perp_dims]
    
    # Convert radius from fraction to number of elements
    r = int(radius * min(perp_shape))
    
    # Convert height from fraction to number of elements
    h = int(height * dimensions[axis])
    half_h = h // 2
    
    # Calculate bounds for the axis direction with clipping
    axis_lo = max(0, center_idx[axis] - half_h)
    axis_hi = min(dimensions[axis], center_idx[axis] + half_h)
    
    # Create the mask with correct dimensions
    mask = np.zeros((nely, nelx, nelz), dtype=bool)
    
    # Create coordinates relative to center
    y_indices, x_indices, z_indices = np.ogrid[:nely, :nelx, :nelz]
    y_distance = y_indices - cy
    x_distance = x_indices - cx
    z_distance = z_indices - cz
    
    if axis == 0:  # x-axis
        # Calculate distance in yz-plane
        dist_squared = y_distance**2 + z_distance**2
        # Apply mask for elements within radius and within height bounds
        x_in_bounds = (x_indices >= axis_lo) & (x_indices < axis_hi)
        mask[(dist_squared <= r**2) & x_in_bounds] = True
        
    elif axis == 1:  # y-axis
        # Calculate distance in xz-plane
        dist_squared = x_distance**2 + z_distance**2
        # Apply mask for elements within radius and within height bounds
        y_in_bounds = (y_indices >= axis_lo) & (y_indices < axis_hi)
        mask[(dist_squared <= r**2) & y_in_bounds] = True
        
    else:  # z-axis (default)
        # Calculate distance in xy-plane
        dist_squared = x_distance**2 + y_distance**2
        # Apply mask for elements within radius and within height bounds
        z_in_bounds = (z_indices >= axis_lo) & (z_indices < axis_hi)
        mask[(dist_squared <= r**2) & z_in_bounds] = True
    
    return mask

def create_obstacle_from_config(
    shape: Tuple[int, int, int],
    config: Dict
) -> np.ndarray:
    """
    Create an obstacle mask from a configuration dictionary.
    
    Parameters
    ----------
    shape : tuple of int
        Shape of the design domain (nely, nelx, nelz).
    config : dict
        Configuration dictionary describing the obstacle.
        Must contain 'type' and other required parameters for that type.
        
    Returns
    -------
    np.ndarray
        Boolean mask with True values where the obstacle is located.
    """
    obstacle_type = config.get('type', '').lower()
    
    if obstacle_type == 'cube':
        center = config.get('center', [0.5, 0.5, 0.5])
        size = config.get('size', 0.2)
        return create_cube_obstacle(shape, center, size)
    
    elif obstacle_type == 'sphere':
        center = config.get('center', [0.5, 0.5, 0.5])
        radius = config.get('radius', 0.2)
        return create_sphere_obstacle(shape, center, radius)
    
    elif obstacle_type == 'cylinder':
        center = config.get('center', [0.5, 0.5, 0.5])
        radius = config.get('radius', 0.2)
        height = config.get('height', 0.5)
        axis = config.get('axis', 2)
        return create_cylinder_obstacle(shape, center, radius, height, axis)
    
    else:
        raise ValueError(f"Unknown obstacle type: {obstacle_type}")

def parse_obstacle_config_file(
    config_file: str,
    shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Parse a JSON configuration file and create an obstacle mask.
    
    Parameters
    ----------
    config_file : str
        Path to the JSON configuration file.
    shape : tuple of int
        Shape of the design domain (nely, nelx, nelz).
        
    Returns
    -------
    np.ndarray
        Combined boolean mask with True values where any obstacle is located.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Create a mask with all False (no obstacles)
    combined_mask = np.zeros(shape, dtype=bool)
    
    # Process each obstacle in the config
    obstacles = config.get('obstacles', [])
    for obstacle_config in obstacles:
        mask = create_obstacle_from_config(shape, obstacle_config)
        # Combine with OR operation
        combined_mask = np.logical_or(combined_mask, mask)
    
    return combined_mask 