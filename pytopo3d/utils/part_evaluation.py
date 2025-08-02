"""
Utilities for evaluating the optimized part to pass as results

This module provides functions to predict the mechanical properties of the final result
"""

import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def get_avg_displacement_vector(
    U,
    x1: int, x2: int,
    y1: int, y2: int,
    z1: int, z2: int,
    nelx: int, nely: int, nelz: int,
) -> np.ndarray:
    """
    Compute average displacement vector (ux, uy, uz) over a given subregion.

    Parameters
    ----------
    U : (ndof,) ndarray
        Global displacement vector (NumPy or CuPy).
    x1, x2, y1, y2, z1, z2 : int
        Index bounds for the element box to evaluate.
    nelx, nely, nelz : int
        Number of elements in each dimension.

    Returns
    -------
    avg_disp : (3,) np.ndarray
        Average displacement vector over the selected region.
    """
    is_gpu = HAS_CUPY and isinstance(U, cp.ndarray)
    xp = cp if is_gpu else np

    nnx = nelx + 1
    nny = nely + 1
    nnz = nelz + 1

    ux_list = []
    uy_list = []
    uz_list = []

    for z in range(z1, z2 + 1):
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                node = x + y * nnx + z * nnx * nny
                ux_list.append(U[3 * node + 0])
                uy_list.append(U[3 * node + 1])
                uz_list.append(U[3 * node + 2])

    ux = xp.array(ux_list)
    uy = xp.array(uy_list)
    uz = xp.array(uz_list)

    # Convert to NumPy safely
    if is_gpu:
        avg_disp = np.array([float(ux.mean().get()), float(uy.mean().get()), float(uz.mean().get())])
    else:
        avg_disp = np.array([ux.mean(), uy.mean(), uz.mean()])

    return avg_disp


def estimate_failure_force(
    force_vector: np.ndarray, 
    ce: np.ndarray, 
    yield_stress: float, 
    method: str = "max"
) -> float:
    """
    Estimate the applied force that would cause failure based on max strain energy.

    Parameters:
        force_vector : (ndof,) array – original force vector applied to structure
        ce           : (nely, nelx, nelz) array – elementwise compliance energy density
        yield_stress : float – material yield stress (in Pascals)
        method       : str – "max" or "mean" (which energy to use for scaling)

    Returns:
        estimated_failure_force : float – estimated failure load (same unit as input force)
    """
    if method == "max":
        ce_val = np.max(ce)
    elif method == "mean":
        ce_val = np.mean(ce)
    else:
        raise ValueError("method must be 'max' or 'mean'")

    # Compute magnitude of original applied force
    F_applied = np.linalg.norm(force_vector)

    # Energy density ~ stress² → scale force by stress / sqrt(energy)
    estimated_failure_force = F_applied * yield_stress / np.sqrt(ce_val)

    return estimated_failure_force