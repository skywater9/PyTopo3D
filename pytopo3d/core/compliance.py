"""
Element compliance calculation for 3D topology optimization.

This module contains functions for computing element-wise compliance.
"""

from typing import Union

import numpy as np

# Check if CuPy is available for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def element_compliance(
    U: Union[np.ndarray, "cp.ndarray"], 
    edofMat: Union[np.ndarray, "cp.ndarray"], 
    KE: Union[np.ndarray, "cp.ndarray"]
) -> Union[np.ndarray, "cp.ndarray"]:
    """
    Compute element-wise compliance:
      ce[e] = Ue @ KE @ Ue^T  for each element e.
    
    Parameters
    ----------
    U : ndarray or cupy.ndarray
        Global displacement vector.
    edofMat : ndarray or cupy.ndarray
        Element DOF mapping matrix.
    KE : ndarray or cupy.ndarray
        Element stiffness matrix.
        
    Returns
    -------
    ndarray or cupy.ndarray
        A 1D array of length nele with element compliance values.
        Returns same type as input arrays.
    """
    # Check if inputs are on GPU
    inputs_on_gpu = HAS_CUPY and any(
        isinstance(arr, cp.ndarray) for arr in [U, edofMat, KE]
    )
    
    if inputs_on_gpu:
        # Ensure all arrays are on GPU
        if not isinstance(U, cp.ndarray):
            U_gpu = cp.asarray(U)
        else:
            U_gpu = U
            
        if not isinstance(edofMat, cp.ndarray):
            edofMat_gpu = cp.asarray(edofMat)
        else:
            edofMat_gpu = edofMat
            
        if not isinstance(KE, cp.ndarray):
            KE_gpu = cp.asarray(KE)
        else:
            KE_gpu = KE
            
        nele = edofMat_gpu.shape[0]
        # edofMat is 1-based indexing, so subtract 1 for 0-based U
        Ue = U_gpu[edofMat_gpu.astype(int) - 1]  # shape (nele, 24)
        ce = cp.sum((Ue @ KE_gpu) * Ue, axis=1)
        return ce
    else:
        # CPU version (unchanged)
        nele = edofMat.shape[0]
        # edofMat is 1-based indexing, so subtract 1 for 0-based U
        Ue = U[edofMat.astype(int) - 1]  # shape (nele, 24)
        ce = np.sum((Ue @ KE) * Ue, axis=1)
        return ce