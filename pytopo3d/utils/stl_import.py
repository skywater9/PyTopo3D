"""
Functions for importing STL files and converting them to voxel representations
for topology optimization.

This module enables the use of STL geometry to define the design space.
"""

import os
from typing import Tuple

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


def voxelize_mesh(
    mesh: trimesh.Trimesh, resolution: Tuple[int, int, int]
) -> np.ndarray:
    """
    Convert a mesh to a voxel representation while preserving the original proportions.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to voxelize.
    resolution : tuple of int
        The desired voxel grid resolution (nely, nelx, nelz).

    Returns
    -------
    np.ndarray
        Boolean array where True values represent the interior of the mesh.
        The array has shape (nely, nelx, nelz) to match the topology optimization format.
    """
    nely, nelx, nelz = resolution

    # Get the original mesh extents
    mesh_extents = mesh.extents  # [x_extent, y_extent, z_extent]

    # Create a voxel grid with uniform pitch
    voxel_grid = trimesh.voxel.creation.voxelize(
        mesh=mesh,
        pitch=1.0,  # We'll scale afterward
        method="ray",
    )

    # Get the raw voxel data
    original_voxels = voxel_grid.matrix
    original_shape = np.array(original_voxels.shape)  # Ensure it's a numpy array

    # Calculate the original aspect ratio
    # Convert to our coordinate system: original_shape is [y, x, z]
    original_aspect_ratio = np.array(
        [
            original_shape[0] / max(original_shape),  # y
            original_shape[1] / max(original_shape),  # x
            original_shape[2] / max(original_shape),  # z
        ]
    )

    # Calculate target aspect ratio
    target_shape = np.array([nely, nelx, nelz])
    target_aspect_ratio = np.array(
        [
            nely / max(target_shape),
            nelx / max(target_shape),
            nelz / max(target_shape),
        ]
    )

    # Create a new voxel array with the desired resolution
    scaled_voxels = np.zeros((nely, nelx, nelz), dtype=bool)

    # Calculate scaling factor that preserves proportions
    # We'll use the minimum dimension to ensure the mesh fits
    ratio_adjustment = target_aspect_ratio / original_aspect_ratio
    adjusted_original_shape = original_shape * ratio_adjustment
    scale_factors = target_shape / adjusted_original_shape
    uniform_scale = min(scale_factors)

    # Calculate the effective shape after uniform scaling
    effective_shape = np.round(
        original_shape * uniform_scale * ratio_adjustment
    ).astype(int)

    # Make sure effective shape doesn't exceed target shape
    effective_shape = np.minimum(effective_shape, target_shape)

    # Calculate offsets to center the model in the target grid
    offsets = ((target_shape - effective_shape) / 2).astype(int)

    # Map the original voxels to the new grid while maintaining proportions
    for i in range(nely):
        for j in range(nelx):
            for k in range(nelz):
                # Check if this voxel is within the effective shape (centered)
                if (
                    i >= offsets[0]
                    and i < offsets[0] + effective_shape[0]
                    and j >= offsets[1]
                    and j < offsets[1] + effective_shape[1]
                    and k >= offsets[2]
                    and k < offsets[2] + effective_shape[2]
                ):
                    # Map back to the original voxel space
                    # We need to account for both scaling and aspect ratio adjustment
                    orig_i = int(
                        (i - offsets[0]) / (uniform_scale * ratio_adjustment[0])
                    )
                    orig_j = int(
                        (j - offsets[1]) / (uniform_scale * ratio_adjustment[1])
                    )
                    orig_k = int(
                        (k - offsets[2]) / (uniform_scale * ratio_adjustment[2])
                    )

                    # Prevent out of bounds
                    orig_i = min(max(orig_i, 0), original_shape[0] - 1)
                    orig_j = min(max(orig_j, 0), original_shape[1] - 1)
                    orig_k = min(max(orig_k, 0), original_shape[2] - 1)

                    scaled_voxels[i, j, k] = original_voxels[orig_i, orig_j, orig_k]

    return scaled_voxels


def stl_to_design_space(
    stl_file: str, resolution: Tuple[int, int, int], invert: bool = False
) -> np.ndarray:
    """
    Convert an STL file to a design space mask.

    The function preserves the original aspect ratio (proportions) of the mesh,
    scaling it uniformly to fit within the target resolution without distortion.
    The voxelized mesh is centered within the design space.

    Parameters
    ----------
    stl_file : str
        Path to the STL file.
    resolution : tuple of int
        The desired voxel grid resolution (nely, nelx, nelz).
    invert : bool, optional
        If True, invert the mask (i.e., True becomes False and vice versa).
        This is useful when the STL represents a void space rather than the design space.

    Returns
    -------
    np.ndarray
        Boolean array where True values represent the design space.
        The array has shape (nely, nelx, nelz) to match the topology optimization format.
    """
    # Import the STL file
    mesh = import_stl(stl_file)

    # Voxelize the mesh
    voxels = voxelize_mesh(mesh, resolution)

    # Invert if requested
    if invert:
        voxels = ~voxels

    return voxels
