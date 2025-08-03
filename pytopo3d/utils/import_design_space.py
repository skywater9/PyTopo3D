"""
Functions for importing STL files and converting them to voxel representations
for topology optimization.

This module enables the use of STL geometry to define the design space.
"""

import os

import numpy as np
import trimesh


def import_stl(stl_file: str) -> trimesh.Trimesh:
    """
    Import an STL file and return a trimesh object.

    Parameters
    ----------
    stl_file : str
        Path to the STL file.

    Returns
    -------
    trimesh.Trimesh
        Loaded mesh from the STL file.
    """
    if not os.path.exists(stl_file):
        raise FileNotFoundError(f"STL file not found: {stl_file}")

    try:
        mesh = trimesh.load(stl_file)
        return mesh
    except Exception as e:
        raise ImportError(f"Failed to load STL file: {e}")


def voxelize_mesh(mesh: trimesh.Trimesh, pitch: float = 1.0) -> np.ndarray:
    """
    Convert a mesh to a voxel representation.
    Resolution is determined entirely by the mesh and the pitch value.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to voxelize.
    pitch : float
        The distance between voxel centers (smaller values create finer detail).

    Returns
    -------
    np.ndarray
        Boolean array where True values represent the interior of the mesh.
    """
    # Get and print the mesh extents
    mesh_extents = mesh.extents  # [x_extent, y_extent, z_extent]
    print(f"Mesh extents: {mesh_extents}")
    print(f"Mesh bounds: {mesh.bounds}")

    # Create a voxel grid with specified pitch
    voxel_grid = trimesh.voxel.creation.voxelize(
        mesh=mesh,
        pitch=pitch,
        method="subdivide",
    )

    # Fill the interior of the voxel grid to ensure a solid volume
    # The fill() method operates in-place on voxel_grid.
    voxel_grid.fill()

    # Print voxel grid information
    voxel_matrix = voxel_grid.matrix
    print(f"Voxel grid shape: {voxel_matrix.shape}")
    print(f"Voxel grid resolution determined by mesh and pitch: {voxel_matrix.shape}")

    return voxel_matrix


def stl_to_design_space(
    stl_file: str,
    target_nelx: int = None,
    invert: bool = False,
):
    """
    Convert an STL file to a design space mask.

    Parameters
    ----------
    stl_file : str
        Path to the STL file.
    target_nelx : int 
        Desired number of voxels in the x-direction.
    invert : bool, optional
        If True, invert the mask.

    Returns
    -------
    tuple
        - np.ndarray: Boolean array representing the design space.
    """
    # Import the STL file
    mesh = import_stl(stl_file)

    # Voxelize the mesh with specified number of elements in X
    bounds = mesh.bounds
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]

    length_x = x_max - x_min
    pitch = length_x / (target_nelx-1)
    voxels = voxelize_mesh(mesh, pitch)

    # Invert if requested
    if invert:
        voxels = ~voxels

    return voxels