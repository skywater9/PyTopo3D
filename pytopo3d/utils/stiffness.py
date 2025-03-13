"""
Stiffness matrix generation for 3D topology optimization.

This module contains the function to generate the element stiffness matrix
for a hexahedral (H8) element.
"""

import numpy as np

def lk_H8(nu):
    """
    Generate the 24x24 hexahedral element stiffness matrix (H8).
    
    Parameters
    ----------
    nu : float
        Poisson's ratio.
        
    Returns
    -------
    ndarray
        24x24 element stiffness matrix.
    """
    A = np.array(
        [
            [32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
            [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12],
        ],
        dtype=float,
    )
    kvec = (1.0 / 144.0) * A.T @ np.array([1.0, nu])
    k = kvec.ravel()  # 14 entries

    K1 = np.array(
        [
            [k[0], k[1], k[1], k[2], k[4], k[4]],
            [k[1], k[0], k[1], k[3], k[5], k[6]],
            [k[1], k[1], k[0], k[3], k[6], k[5]],
            [k[2], k[3], k[3], k[0], k[7], k[7]],
            [k[4], k[5], k[6], k[7], k[0], k[1]],
            [k[4], k[6], k[5], k[7], k[1], k[0]],
        ]
    )
    K2 = np.array(
        [
            [k[8], k[7], k[11], k[5], k[3], k[6]],
            [k[7], k[8], k[11], k[4], k[2], k[4]],
            [k[9], k[9], k[12], k[6], k[3], k[5]],
            [k[5], k[4], k[10], k[8], k[1], k[9]],
            [k[3], k[2], k[4], k[1], k[8], k[11]],
            [k[10], k[3], k[5], k[11], k[9], k[12]],
        ]
    )
    K3 = np.array(
        [
            [k[5], k[6], k[3], k[8], k[11], k[7]],
            [k[6], k[5], k[3], k[9], k[12], k[9]],
            [k[4], k[4], k[2], k[7], k[11], k[8]],
            [k[8], k[9], k[1], k[5], k[10], k[4]],
            [k[11], k[12], k[9], k[10], k[5], k[3]],
            [k[1], k[11], k[8], k[3], k[4], k[2]],
        ]
    )
    K4 = np.array(
        [
            [k[13], k[10], k[10], k[12], k[9], k[9]],
            [k[10], k[13], k[10], k[11], k[8], k[7]],
            [k[10], k[10], k[13], k[11], k[7], k[8]],
            [k[12], k[11], k[11], k[13], k[6], k[6]],
            [k[9], k[8], k[7], k[6], k[13], k[10]],
            [k[9], k[7], k[8], k[6], k[10], k[13]],
        ]
    )
    K5 = np.array(
        [
            [k[0], k[1], k[7], k[2], k[4], k[3]],
            [k[1], k[0], k[7], k[3], k[5], k[10]],
            [k[7], k[7], k[0], k[4], k[10], k[5]],
            [k[2], k[3], k[4], k[0], k[7], k[1]],
            [k[4], k[5], k[10], k[7], k[0], k[7]],
            [k[3], k[10], k[5], k[1], k[7], k[0]],
        ]
    )
    K6 = np.array(
        [
            [k[13], k[10], k[6], k[12], k[9], k[11]],
            [k[10], k[13], k[6], k[11], k[8], k[1]],
            [k[6], k[6], k[13], k[9], k[1], k[8]],
            [k[12], k[11], k[9], k[13], k[6], k[10]],
            [k[9], k[8], k[1], k[6], k[13], k[6]],
            [k[11], k[1], k[8], k[10], k[6], k[13]],
        ]
    )

    KE_block = np.vstack(
        [
            np.hstack([K1, K2, K3, K4]),
            np.hstack([K2.T, K5, K6, K3.T]),
            np.hstack([K3.T, K6, K5.T, K2.T]),
            np.hstack([K4, K3, K2, K1]),
        ]
    )
    factor = 1.0 / ((nu + 1.0) * (1.0 - 2.0 * nu))
    return factor * KE_block