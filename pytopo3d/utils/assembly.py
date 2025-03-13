"""
Assembly utilities for 3D topology optimization.

This module contains helper functions for assembling the force vector,
boundary conditions, and element DOF matrices.
"""

import numpy as np
import scipy.sparse as sp

def build_force_vector(nelx, nely, nelz, ndof):
    """
    Build the force vector.
    
    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
    ndof : int
        Total number of degrees of freedom.
        
    Returns
    -------
    ndarray
        Force vector with applied loads.
    """
    il, jl, kl = np.meshgrid([nelx], [0], np.arange(nelz + 1), indexing="ij")
    loadnid = kl * (nelx + 1) * (nely + 1) + il * (nely + 1) + (nely + 1 - jl)
    loaddof = 3 * loadnid.ravel() - 1
    F = np.zeros(ndof)
    F[loaddof.astype(int) - 1] = -1.0
    return F

def build_supports(nelx, nely, nelz, ndof):
    """
    Build support constraints (fixed DOFs).
    
    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
    ndof : int
        Total number of degrees of freedom.
        
    Returns
    -------
    tuple
        (free DOFs, fixed DOFs) as zero-indexed arrays.
    """
    iif, jf, kf = np.meshgrid(
        [0], np.arange(nely + 1), np.arange(nelz + 1), indexing="ij"
    )
    fixednid = kf * (nelx + 1) * (nely + 1) + iif * (nely + 1) + (nely + 1 - jf)
    fixeddof = np.concatenate(
        [
            3 * fixednid.ravel(),
            3 * fixednid.ravel() - 1,
            3 * fixednid.ravel() - 2,
        ]
    )
    all_dofs = np.arange(1, ndof + 1)
    freedofs = np.setdiff1d(all_dofs, fixeddof)
    freedofs0 = freedofs - 1
    fixeddof0 = fixeddof - 1
    return freedofs0, fixeddof0

def build_edof(nelx, nely, nelz):
    """
    Build element DOF mapping and global assembly indices.
    
    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
        
    Returns
    -------
    tuple
        (edofMat, iK, jK) for assembly of global stiffness matrix.
    """
    nodegrd = np.arange(1, (nely + 1) * (nelx + 1) + 1).reshape(
        (nely + 1, nelx + 1), order="F"
    )
    nodeids = nodegrd[:-1, :-1].ravel(order="F")
    nodeidz = np.arange(0, nelz * (nely + 1) * (nelx + 1), (nely + 1) * (nelx + 1))
    nodeids3d = (nodeids.reshape(-1, 1) + nodeidz.reshape(1, -1)).ravel(order="F")
    edofVec = 3 * nodeids3d + 1  # 1-based
    offset0 = [0, 1, 2]
    offset1 = (3 * nely + np.array([3, 4, 5, 0, 1, 2])).tolist()
    offset2 = [-3, -2, -1]
    sub_part = np.concatenate(
        [
            np.array([0, 1, 2]),
            3 * nely + np.array([3, 4, 5, 0, 1, 2]),
            np.array([-3, -2, -1]),
        ]
    )
    offset3 = (3 * (nely + 1) * (nelx + 1) + sub_part).tolist()
    offset_full = np.array(offset0 + offset1 + offset2 + offset3, dtype=int)

    edofMat = np.tile(edofVec.reshape(-1, 1), (1, 24)) + np.tile(
        offset_full, (len(edofVec), 1)
    )

    iK = np.kron(edofMat, np.ones((24, 1))).ravel()
    jK = np.kron(edofMat, np.ones((1, 24))).ravel()
    return edofMat, iK, jK