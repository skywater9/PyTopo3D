"""
Element compliance calculation for 3D topology optimization.

This module contains functions for computing element-wise compliance.
"""

import numpy as np

def element_compliance(U, edofMat, KE):
    """
    Compute element-wise compliance:
      ce[e] = Ue @ KE @ Ue^T  for each element e.
    
    Parameters
    ----------
    U : ndarray
        Global displacement vector.
    edofMat : ndarray
        Element DOF mapping matrix.
    KE : ndarray
        Element stiffness matrix.
        
    Returns
    -------
    ndarray
        A 1D array of length nele with element compliance values.
    """
    nele = edofMat.shape[0]
    # edofMat is 1-based indexing, so subtract 1 for 0-based U
    Ue = U[edofMat.astype(int) - 1]  # shape (nele, 24)
    ce = np.sum((Ue @ KE) * Ue, axis=1)
    return ce