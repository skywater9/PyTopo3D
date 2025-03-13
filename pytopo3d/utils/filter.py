"""
Filter utilities for 3D topology optimization.

This module contains functions for building spatial density filters.
"""

import numpy as np
import scipy.sparse as sp

def build_filter(nelx, nely, nelz, rmin):
    """
    Build the density filter matrix.
    
    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
    rmin : float
        Filter radius.
        
    Returns
    -------
    tuple
        (H, Hs) - filter matrix and column sums for normalization.
    """
    nele = nelx * nely * nelz
    rminCeil = int(np.ceil(rmin))
    iH_list = []
    jH_list = []
    sH_list = []
    for k1 in range(1, nelz + 1):
        for i1 in range(1, nelx + 1):
            for j1 in range(1, nely + 1):
                e1 = (k1 - 1) * nelx * nely + (i1 - 1) * nely + (j1 - 1)
                for k2 in range(
                    max(k1 - rminCeil + 1, 1), min(k1 + rminCeil - 1, nelz) + 1
                ):
                    for i2 in range(
                        max(i1 - rminCeil + 1, 1), min(i1 + rminCeil - 1, nelx) + 1
                    ):
                        for j2 in range(
                            max(j1 - rminCeil + 1, 1), min(j1 + rminCeil - 1, nely) + 1
                        ):
                            e2 = (k2 - 1) * nelx * nely + (i2 - 1) * nely + (j2 - 1)
                            dist = np.sqrt(
                                (i1 - i2) ** 2 + (j1 - j2) ** 2 + (k1 - k2) ** 2
                            )
                            val = rmin - dist
                            if val > 0.0:
                                iH_list.append(e1)
                                jH_list.append(e2)
                                sH_list.append(val)
    iH_arr = np.array(iH_list, dtype=int)
    jH_arr = np.array(jH_list, dtype=int)
    sH_arr = np.array(sH_list, dtype=float)
    H = sp.coo_matrix((sH_arr, (iH_arr, jH_arr)), shape=(nele, nele)).tocsr()
    Hs = np.array(H.sum(axis=1)).flatten()
    return H, Hs