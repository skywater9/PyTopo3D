"""
Optimality Criteria (OC) update scheme for 3D topology optimization.

This module contains the function for updating design variables
using the optimality criteria method.
"""

import numpy as np

def optimality_criteria_update(x, dc, dv, volfrac, H, Hs, nele, obstacle_mask, design_nele):
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
    H : scipy.sparse.csr_matrix
        Filter matrix.
    Hs : ndarray
        Filter normalization factors.
    nele : int
        Total number of elements.
    obstacle_mask : ndarray of bool
        Mask indicating obstacle elements.
    design_nele : int
        Number of elements in design domain (not obstacles).
        
    Returns
    -------
    tuple
        (updated design variables, maximum change)
    """
    l1, l2 = 0.0, 1e9
    move = 0.2
    xnew = x.copy()

    # Flatten the mask for convenience if needed
    obs_flat = obstacle_mask.ravel(order="F")

    # All design cells (not obstacle)
    design_cells = ~obstacle_mask

    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)

        # Standard OC update
        x_candidate = x * np.sqrt(-dc / dv / lmid)
        x_candidate = np.clip(
            x_candidate,
            np.maximum(0.0, x - move),
            np.minimum(1.0, x + move),
        )

        # Force obstacle region to remain at 0 in the candidate
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

    change = np.abs(xnew - x).max()
    return xnew, change