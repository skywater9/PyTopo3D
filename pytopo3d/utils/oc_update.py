"""
Optimality Criteria (OC) update scheme for 3D topology optimization.

This module contains the function for updating design variables
using the optimality criteria method.
"""

from typing import Tuple, Union

import numpy as np
import scipy.sparse as sp

# Import filtering function
from pytopo3d.utils.filter import apply_filter, HAS_CUPY

# Check if CuPy is available for GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def optimality_criteria_update(
    x: Union[np.ndarray, "cp.ndarray"],
    dc: Union[np.ndarray, "cp.ndarray"],
    dv: Union[np.ndarray, "cp.ndarray"],
    volfrac: float,
    H: Union[sp.csr_matrix, "cusp.csr_matrix"],
    Hs: Union[np.ndarray, "cp.ndarray"],
    nele: int,
    obstacle_mask: Union[np.ndarray, "cp.ndarray"],
    protected_zone_mask: Union[np.ndarray, "cp.ndarray"],  # <<< MODIFIED: ensure passed in
    design_nele: int,
    use_gpu: bool = False
) -> Tuple[Union[np.ndarray, "cp.ndarray"], float]:
    
    inputs_on_gpu = HAS_CUPY and any(
        isinstance(arr, cp.ndarray) 
        for arr in [x, dc, dv, obstacle_mask, protected_zone_mask, Hs]
    ) or isinstance(H, cusp.csr_matrix)
    
    use_gpu_for_calc = (use_gpu and HAS_CUPY) or inputs_on_gpu
    l1, l2 = 1e-9, 1e9
    move = 0.2
    
    if use_gpu_for_calc:
        # Transfer arrays to GPU
        x_gpu = x if isinstance(x, cp.ndarray) else cp.asarray(x)
        dc_gpu = dc if isinstance(dc, cp.ndarray) else cp.asarray(dc)
        dv_gpu = dv if isinstance(dv, cp.ndarray) else cp.asarray(dv)
        obstacle_gpu = obstacle_mask if isinstance(obstacle_mask, cp.ndarray) else cp.asarray(obstacle_mask)
        protected_gpu = protected_zone_mask if isinstance(protected_zone_mask, cp.ndarray) else cp.asarray(protected_zone_mask)  # <<< MODIFIED
        free_mask_gpu = (~obstacle_gpu) & (~protected_gpu)  # <<< MODIFIED: free elements mask

        H_gpu = H if isinstance(H, cusp.csr_matrix) else cusp.csr_matrix(
            (cp.asarray(H.data), cp.asarray(H.indices), cp.asarray(H.indptr)), shape=H.shape
        )
        Hs_gpu = Hs if isinstance(Hs, cp.ndarray) else cp.asarray(Hs)
        xnew_gpu = x_gpu.copy()

        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            update_term = -dc_gpu / (dv_gpu * lmid)
            update_term[dv_gpu < 1e-9] = 0.0
            update_term[update_term < 0] = 0.0

            x_candidate_gpu = x_gpu * cp.sqrt(update_term)
            x_candidate_gpu = cp.clip(x_candidate_gpu, cp.maximum(0.0, x_gpu - move), cp.minimum(1.0, x_gpu + move))
            x_candidate_gpu[obstacle_gpu] = 0.0
            x_candidate_gpu[protected_gpu] = 1.0  # <<< MODIFIED: enforce protected

            xPhysCandidate_gpu = apply_filter(H_gpu, x_candidate_gpu, Hs_gpu, x_gpu.shape, use_gpu=True)
            xPhysCandidate_gpu[obstacle_gpu] = 0.0
            xPhysCandidate_gpu[protected_gpu] = 1.0  # <<< MODIFIED

            vol_constraint = cp.sum(xPhysCandidate_gpu[free_mask_gpu])  # <<< MODIFIED: volume on free only
            if vol_constraint > volfrac * design_nele:
                l1 = lmid
            else:
                l2 = lmid
            
            xnew_gpu = x_candidate_gpu

        change = float(cp.max(cp.abs(xnew_gpu[free_mask_gpu] - x_gpu[free_mask_gpu])))  # <<< MODIFIED: change on free only
        return xnew_gpu, change

    else:
        # CPU
        xnew = x.copy()
        free_mask = (~obstacle_mask) & (~protected_zone_mask)  # <<< MODIFIED

        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            update_term = -dc / (dv * lmid)
            update_term[dv < 1e-9] = 0.0
            update_term[update_term < 0] = 0.0

            x_candidate = x * np.sqrt(update_term)
            x_candidate = np.clip(x_candidate, np.maximum(0.0, x - move), np.minimum(1.0, x + move))
            x_candidate[obstacle_mask] = 0.0
            x_candidate[protected_zone_mask] = 1.0  # <<< MODIFIED

            xPhysCandidate = apply_filter(H, x_candidate, Hs, x.shape, use_gpu=False)
            xPhysCandidate[obstacle_mask] = 0.0
            xPhysCandidate[protected_zone_mask] = 1.0  # <<< MODIFIED

            if xPhysCandidate[free_mask].sum() > volfrac * design_nele:  # <<< MODIFIED
                l1 = lmid
            else:
                l2 = lmid

            xnew = x_candidate

        change = float(np.max(np.abs(xnew[free_mask] - x[free_mask])))  # <<< MODIFIED
        return xnew, change

