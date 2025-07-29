"""
Boundary condition utilities for 3D topology optimization.

This module contains functions for handling boundary conditions such as loads and constraints.
"""

from typing import Tuple

import numpy as np


def calculate_boundary_positions(
    nelx: int, nely: int, nelz: int
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
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
    iif, jf, kf = np.meshgrid(
        [0], np.arange(nely + 1), np.arange(nelz + 1), indexing="ij"
    )
    constraint_x = iif.ravel()
    constraint_y = nely - jf.ravel()  # Converted to visualization coordinates
    constraint_z = kf.ravel()

    return (load_x, load_y, load_z), (constraint_x, constraint_y, constraint_z)


def create_boundary_arrays(
    nelx: int, nely: int, nelz: int
) -> Tuple[np.ndarray, np.ndarray]:
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

    # DEPRECATED - Hardcoded default BCs, use create_bc_visualization_arrays instead
    # Set loads at the right face (x=nelx-1) at the bottom (y=nely-1) on all Z levels
    # loads_array[nely-1, nelx-1, :] = 1.0

    # Set constraints at the left face (x=0) on all Y and Z levels
    # constraints_array[:, 0, :] = 1.0
    print(
        "Warning: create_boundary_arrays is deprecated. Use create_bc_visualization_arrays."
    )
    return loads_array, constraints_array


def create_bc_visualization_arrays(
    nelx: int, nely: int, nelz: int, ndof: int, F: np.ndarray, fixeddof0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates visualization arrays for loads and constraints based on actual FEA boundary conditions.

    Maps nodal forces and fixed DOFs to the element grid for visualization.
    Marks all elements adjacent to a node with applied BCs.

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
    ndof : int
        Total number of degrees of freedom.
    F : np.ndarray
        Global force vector (shape: ndof).
    fixeddof0 : np.ndarray
        Array of 0-based fixed degree of freedom indices.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (loads_array, constraints_array), each with shape (nely, nelx, nelz).
    """
    loads_array = np.zeros((nely, nelx, nelz), dtype=float)
    constraints_array = np.zeros((nely, nelx, nelz), dtype=float)

    # --- Map Fixed DOFs to Element Constraints ---
    fixed_nid0 = np.unique(fixeddof0 // 3)
    nelyp1 = nely + 1
    nelxp1_nelyp1 = (nelx + 1) * (nely + 1)

    for nid in fixed_nid0:
        # Inverse calculation for Fortran node index -> (ix, iy, iz)
        iz = nid // nelxp1_nelyp1
        rem = nid % nelxp1_nelyp1
        ix = rem // nelyp1
        iy = rem % nelyp1

        # Mark adjacent elements (within grid bounds)
        for elz in range(max(0, iz - 1), min(nelz, iz + 1)):
            for elx in range(max(0, ix - 1), min(nelx, ix + 1)):
                for ely in range(max(0, iy - 1), min(nely, iy + 1)):
                    if 0 <= elx < nelx and 0 <= ely < nely and 0 <= elz < nelz:
                        constraints_array[ely, elx, elz] = 1.0

    # --- Map Forces to Element Loads ---
    loaded_dof0 = np.where(F != 0)[0]
    loaded_nid0 = np.unique(loaded_dof0 // 3)

    for nid in loaded_nid0:
        # Inverse calculation for Fortran node index -> (ix, iy, iz)
        iz = nid // nelxp1_nelyp1
        rem = nid % nelxp1_nelyp1
        ix = rem // nelyp1
        iy = rem % nelyp1

        # Mark adjacent elements (within grid bounds)
        for elz in range(max(0, iz - 1), min(nelz, iz + 1)):
            for elx in range(max(0, ix - 1), min(nelx, ix + 1)):
                for ely in range(max(0, iy - 1), min(nely, iy + 1)):
                    if 0 <= elx < nelx and 0 <= ely < nely and 0 <= elz < nelz:
                        loads_array[ely, elx, elz] = 1.0

    return loads_array, constraints_array

def create_bc_visualization_arrays_from_masks(
    nelx: int, nely: int, nelz: int, ndof: int, force_field: np.ndarray, support_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
        """
    Creates visualization arrays for loads and constraints based on user input.

    Maps force fields and support masks to the element grid for visualization.

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
    ndof : int
        Total number of degrees of freedom.
    force_field : np.ndarray
        force field (shape: ndof).
    support_mask : np.ndarray
        constrained elements (shape: ndof).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (loads_array, constraints_array), each with shape (nely, nelx, nelz).
    """
        loads_array = np.zeros((nely, nelx, nelz), dtype=float)
        constraints_array = np.zeros((nely, nelx, nelz), dtype=float)

        loads_array[np.any(force_field != 0, axis=3)] = 1.0
        constraints_array[support_mask != 0] = 1.0

        return loads_array, constraints_array