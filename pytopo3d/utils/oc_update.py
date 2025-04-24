"""
Optimality Criteria (OC) update scheme for 3D topology optimization.

This module contains the function for updating design variables
using the optimality criteria method.
"""

from typing import Tuple

import numpy as np
import scipy.sparse as sp

# Check if CuPy is available for GPU acceleration
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def optimality_criteria_update(
    x: np.ndarray,
    dc: np.ndarray,
    dv: np.ndarray,
    volfrac: float,
    H: "sp.csr_matrix",
    Hs: np.ndarray,
    nele: int,
    obstacle_mask: np.ndarray,
    design_nele: int,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Performs the optimality criteria (OC) update with bisection on the Lagrange
    multiplier. The volume constraint is enforced only on the design domain
    (excluding obstacle elements).

    Parameters
    ----------
    x : ndarray
        Current design variables.
    dc : ndarray
        Sensitivity of compliance.
    dv : ndarray
        Sensitivity of volume.
    volfrac : float
        Target volume fraction.
    H : scipy.sparse.csr_matrix or cupyx.scipy.sparse.csr_matrix
        Filter matrix.
    Hs : ndarray or cupy.ndarray
        Filter normalization factors.
    nele : int
        Total number of elements.
    obstacle_mask : ndarray of bool
        Mask indicating obstacle elements.
    design_nele : int
        Number of elements in design domain (not obstacles).
    use_gpu : bool, optional
        Whether to use GPU acceleration if available. Default is False.

    Returns
    -------
    tuple
        (updated design variables, maximum change)
    """
    # Check if GPU acceleration should be used
    gpu_available = use_gpu and HAS_CUPY

    l1, l2 = 1e-9, 1e9
    move = 0.2
    xnew = x.copy()

    # All design cells (not obstacle)
    design_cells = ~obstacle_mask

    # Transfer to GPU if using GPU acceleration
    if gpu_available:
        import cupyx.scipy.sparse as cusp

        # Transfer all arrays to GPU
        x_gpu = cp.asarray(x)
        dc_gpu = cp.asarray(dc)
        dv_gpu = cp.asarray(dv)
        obstacle_mask_gpu = cp.asarray(obstacle_mask)
        design_cells_gpu = cp.asarray(design_cells)

        # Convert filter to GPU if it's not already
        if not isinstance(H, cusp.csr_matrix):
            H_gpu = cusp.csr_matrix(
                (cp.asarray(H.data), cp.asarray(H.indices), cp.asarray(H.indptr)),
                shape=H.shape,
            )
        else:
            H_gpu = H

        # Convert Hs to GPU if it's not already
        if not isinstance(Hs, cp.ndarray):
            Hs_gpu = cp.asarray(Hs)
        else:
            Hs_gpu = Hs

        xnew_gpu = x_gpu.copy()

    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)

        if gpu_available:
            # GPU-accelerated OC update
            update_term = -dc_gpu / (dv_gpu * lmid)

            # Handle potential issues: dv near zero or negative term inside sqrt
            update_term[dv_gpu < 1e-9] = 0.0  # Avoid division by zero or tiny dv
            update_term[update_term < 0] = 0.0  # Avoid sqrt of negative numbers

            # Now calculate sqrt and update x
            x_candidate_gpu = x_gpu * cp.sqrt(update_term)

            # Clipping and forcing obstacles to zero
            x_candidate_gpu = cp.clip(
                x_candidate_gpu,
                cp.maximum(0.0, x_gpu - move),
                cp.minimum(1.0, x_gpu + move),
            )
            x_candidate_gpu[obstacle_mask_gpu] = 0.0

            # Filter the candidate
            xPhysCandidate_flat_gpu = H_gpu @ x_candidate_gpu.ravel(order="F") / Hs_gpu
            xPhysCandidate_gpu = xPhysCandidate_flat_gpu.reshape(x_gpu.shape, order="F")
            xPhysCandidate_gpu[obstacle_mask_gpu] = 0.0

            # Check volume constraint but only over design domain
            if cp.sum(xPhysCandidate_gpu[design_cells_gpu]) > volfrac * design_nele:
                l1 = lmid
            else:
                l2 = lmid

            xnew_gpu = x_candidate_gpu
        else:
            # CPU version
            # Standard OC update - Calculate with checks for numerical stability
            # Calculate the term inside the square root
            update_term = -dc / (dv * lmid)

            # Handle potential issues: dv near zero or negative term inside sqrt
            update_term[dv < 1e-9] = 0.0  # Avoid division by zero or tiny dv
            update_term[update_term < 0] = 0.0  # Avoid sqrt of negative numbers

            # Now calculate sqrt and update x (problematic terms result in factor 0)
            x_candidate = x * np.sqrt(update_term)

            # Original clipping and forcing obstacles to zero:
            x_candidate = np.clip(
                x_candidate,
                np.maximum(0.0, x - move),
                np.minimum(1.0, x + move),
            )
            x_candidate[obstacle_mask] = 0.0

            # Filter the candidate
            xPhysCandidate = (H * x_candidate.ravel(order="F")) / Hs
            xPhysCandidate = xPhysCandidate.reshape(x.shape, order="F")
            xPhysCandidate[obstacle_mask] = 0.0

            # Check volume constraint but only over design domain
            if xPhysCandidate[design_cells].sum() > volfrac * design_nele:
                l1 = lmid
            else:
                l2 = lmid
            xnew = x_candidate

    # Transfer result back to CPU if using GPU
    if gpu_available:
        xnew = cp.asnumpy(xnew_gpu)
        change = float(cp.max(cp.abs(xnew_gpu - x_gpu)))
    else:
        change = np.abs(xnew - x).max()

    return xnew, change
