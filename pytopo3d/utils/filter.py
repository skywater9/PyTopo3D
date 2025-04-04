"""
Filter utilities for 3D topology optimization.

This module contains functions for building spatial density filters.
"""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree  # Import cKDTree


def build_filter(nelx: int, nely: int, nelz: int, rmin: float) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Build the density filter matrix using KD-tree for neighbor search
    but replicating the original integer-based distance calculation for accuracy.

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
    rmin : float
        Filter radius.

    Returns
    -------
    tuple
        (H, Hs) - filter matrix (CSR format) and row sums for normalization.
    """
    nele = nelx * nely * nelz

    # 1. Create coordinates for element centers (still needed for KD-Tree)
    x_coords = np.arange(nelx) + 0.5
    y_coords = np.arange(nely) + 0.5
    z_coords = np.arange(nelz) + 0.5
    # Use 'ij' indexing consistent with element index calculation (k changes fastest)
    zz, xx, yy = np.meshgrid(z_coords, x_coords, y_coords, indexing="ij")
    centers = np.vstack((xx.ravel(order="F"), yy.ravel(order="F"), zz.ravel(order="F"))).T

    # 2. Build KD-Tree to find potential neighbors efficiently
    tree = cKDTree(centers)
    # Find indices of all points within distance rmin + epsilon (to be safe)
    # We will apply the exact distance check later. Add small epsilon for safety.
    neighbor_indices_list = tree.query_ball_point(centers, rmin + 1e-9)

    # 3. Prepare sparse matrix data using original distance logic
    iH_list = []
    jH_list = []
    sH_list = []

    # Pre-calculate mapping from 0-based element index to 1-based (i, j, k) coords
    # This reverses e = (k-1)*nx*ny + (i-1)*ny + (j-1)
    e_indices = np.arange(nele)
    k1_all = (e_indices // (nelx * nely)) + 1
    i1_all = ((e_indices % (nelx * nely)) // nely) + 1
    j1_all = (e_indices % nely) + 1

    # Iterate through each element (source of the filter influence)
    for e1 in range(nele):
        # Get 1-based coords for e1
        k1, i1, j1 = k1_all[e1], i1_all[e1], j1_all[e1]

        # Get indices of potential neighbors found by KD-Tree
        potential_neighbors = neighbor_indices_list[e1]

        if not potential_neighbors:
            continue

        # Get 1-based coords for potential neighbors
        k2_neighbors = k1_all[potential_neighbors]
        i2_neighbors = i1_all[potential_neighbors]
        j2_neighbors = j1_all[potential_neighbors]

        # Calculate distances using original integer-based formula
        dist_sq = (i1 - i2_neighbors)**2 + (j1 - j2_neighbors)**2 + (k1 - k2_neighbors)**2
        # Avoid sqrt of zero if e1 == e2
        dist = np.sqrt(np.maximum(dist_sq, 1e-12)) # Add tiny epsilon before sqrt

        # Calculate filter weights (linear decay)
        weights = rmin - dist

        # Filter using the original threshold condition (val > 0.0)
        valid_mask = weights > 1e-9 # Use small epsilon for float comparison robustness
        
        # Get the actual neighbors (e2 indices) and their weights
        valid_e2_indices = np.array(potential_neighbors)[valid_mask]
        valid_weights = weights[valid_mask]

        # Append to lists for COO format
        iH_list.extend([e1] * len(valid_e2_indices))
        jH_list.extend(valid_e2_indices)
        sH_list.extend(valid_weights)

    # 4. Create sparse matrix H
    H = sp.coo_matrix(
        (np.array(sH_list), (np.array(iH_list), np.array(jH_list))),
        shape=(nele, nele),
    ).tocsr()

    # 5. Calculate row sums Hs for normalization
    Hs = np.array(H.sum(axis=1)).flatten()
    # Handle cases where Hs might be zero (e.g., isolated elements with rmin=0)
    Hs[Hs == 0] = 1.0 

    return H, Hs

# -------- Original Slow Implementation (for reference) --------
# def build_filter_slow(nelx, nely, nelz, rmin):
#     nele = nelx * nely * nelz
#     rminCeil = int(np.ceil(rmin))
#     iH_list = []
#     jH_list = []
#     sH_list = []
#     for k1 in range(1, nelz + 1):
#         for i1 in range(1, nelx + 1):
#             for j1 in range(1, nely + 1):
#                 e1 = (k1 - 1) * nelx * nely + (i1 - 1) * nely + (j1 - 1)
#                 for k2 in range(
#                     max(k1 - rminCeil + 1, 1), min(k1 + rminCeil - 1, nelz) + 1
#                 ):
#                     for i2 in range(
#                         max(i1 - rminCeil + 1, 1), min(i1 + rminCeil - 1, nelx) + 1
#                     ):
#                         for j2 in range(
#                             max(j1 - rminCeil + 1, 1), min(j1 + rminCeil - 1, nely) + 1
#                         ):
#                             e2 = (k2 - 1) * nelx * nely + (i2 - 1) * nely + (j2 - 1)
#                             dist = np.sqrt(
#                                 (i1 - i2) ** 2 + (j1 - j2) ** 2 + (k1 - k2) ** 2
#                             )
#                             val = rmin - dist
#                             if val > 0.0:
#                                 iH_list.append(e1)
#                                 jH_list.append(e2)
#                                 sH_list.append(val)
#     iH_arr = np.array(iH_list, dtype=int)
#     jH_arr = np.array(jH_list, dtype=int)
#     sH_arr = np.array(sH_list, dtype=float)
#     H = sp.coo_matrix((sH_arr, (iH_arr, jH_arr)), shape=(nele, nele)).tocsr()
#     Hs = np.array(H.sum(axis=1)).flatten()
#     return H, Hs