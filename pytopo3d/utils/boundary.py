"""
Boundary condition utilities for 3D topology optimization.

This module contains functions for handling boundary conditions such as loads and constraints.
"""

import numpy as np
from typing import Tuple

def calculate_boundary_positions(nelx: int, nely: int, nelz: int) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Calculate the positions of loads and constraints for visualization.
    
    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
        
    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
        (load_positions, constraint_positions), where each position is a tuple of (x, y, z) arrays.
    """
    # Calculate load positions
    il, jl, kl = np.meshgrid([nelx], [0], np.arange(nelz + 1), indexing="ij")
    load_x = il.ravel()
    load_y = nely - jl.ravel()  # Converted to visualization coordinates
    load_z = kl.ravel()
    
    # Calculate constraint positions
    iif, jf, kf = np.meshgrid([0], np.arange(nely + 1), np.arange(nelz + 1), indexing="ij")
    constraint_x = iif.ravel()
    constraint_y = nely - jf.ravel()  # Converted to visualization coordinates
    constraint_z = kf.ravel()
    
    return (load_x, load_y, load_z), (constraint_x, constraint_y, constraint_z)

def create_boundary_arrays(nelx: int, nely: int, nelz: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create density arrays for loads and constraints visualization.
    
    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (loads_array, constraints_array), each with shape (nely, nelx, nelz)
        where positions have value 1.0 and the rest have value 0.0
    """
    # Create empty arrays for loads and constraints
    loads_array = np.zeros((nely, nelx, nelz))
    constraints_array = np.zeros((nely, nelx, nelz))
    
    # Set loads at the right face (x=nelx-1) at the bottom (y=nely-1) on all Z levels
    loads_array[nely-1, nelx-1, :] = 1.0
    
    # Set constraints at the left face (x=0) on all Y and Z levels
    constraints_array[:, 0, :] = 1.0
    
    return loads_array, constraints_array 