"""
Utilities for evaluating the optimized part to pass as results

This module provides functions to predict the mechanical properties of the final result
"""

import numpy as np
from typing import Tuple

from pytopo3d.utils.stiffness import make_C_matrix

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

    ux = np.mean(ux_list)
    uy = np.mean(uy_list)
    uz = np.mean(uz_list)

    avg_disp = np.array([ux, uy, uz])

    return avg_disp


def compute_element_stress(
    u_e: np.ndarray, 
    B: np.ndarray, 
    C: np.ndarray
) -> np.ndarray:
    """
    Compute the stress vector (Voigt notation) for one element.

    Parameters:
        u_e : (24,) element nodal displacement vector
        B   : (6, 24) strain-displacement matrix
        C   : (6, 6) material stiffness matrix (orthotropic or anisotropic)

    Returns:
        sigma : (6,) stress vector [sx, sy, sz, tyz, tzx, txy]
    """
    strain = B @ u_e
    stress = C @ strain
    return stress


def estimate_failure_force(
    force_vector: np.ndarray,
    stress_tensors: np.ndarray,
    sigma_x_yield: float,
    sigma_y_yield: float,
    sigma_z_yield: float,
    tau_xy_yield: float,
    tau_yz_yield: float,
    tau_zx_yield: float,
) -> float:
    """
    Estimate failure force using independent yield strengths for normal and shear stresses.

    Parameters:
        force_vector : (ndof,) applied force vector
        stress_tensors : (nele, 6) stress tensors for each element in order
                         [σ_x, σ_y, σ_z, τ_yz, τ_zx, τ_xy]
        sigma_x_yield, sigma_y_yield, sigma_z_yield : yield stresses in MPa for normal stresses
        tau_xy_yield, tau_yz_yield, tau_zx_yield : yield stresses in MPa for shear stresses

    Returns:
        Estimated failure force in same units as force_vector norm
    """

    ratios = []
    for sigma in stress_tensors:
        sx, sy, sz, tyz, tzx, txy = sigma
        f = ((sx / sigma_x_yield)**2 +
             (sy / sigma_y_yield)**2 +
             (sz / sigma_z_yield)**2 +
             (txy / tau_xy_yield)**2 +
             (tyz / tau_yz_yield)**2 +
             (tzx / tau_zx_yield)**2)
        ratios.append(np.sqrt(f))

    max_ratio = max(ratios)
    F_applied = np.linalg.norm(force_vector)
    F_fail = F_applied / max_ratio
    return F_fail


def build_element_stress_tensors(
    U: np.ndarray,
    edofMat: np.ndarray,
    B_matrices: np.ndarray,
    C: np.ndarray
) -> np.ndarray:
    """
    Assemble stress tensors for all elements.

    Parameters:
        U : (ndof,) global displacement vector
        edofMat : (nele, 24) DOF mapping per element
        B_matrices : (nele, 6, 24) precomputed B matrices per element
        C : (6, 6) orthotropic material stiffness matrix

    Returns:
        stress_tensors : (nele, 6) stress vector per element
    """
    nele = edofMat.shape[0]
    stress_tensors = np.zeros((nele, 6))
    for e in range(nele):
        u_e = U[edofMat[e] - 1]
        B = B_matrices[e]
        stress_tensors[e] = compute_element_stress(u_e, B, C)
    return stress_tensors


def compute_B_matrix_H8(dNdx: np.ndarray) -> np.ndarray:
    """
    Compute B matrix for 8-node hexahedral (H8) element.

    Parameters:
        dNdx : (8, 3) derivatives of shape functions w.r.t x, y, z

    Returns:
        B : (6, 24) strain-displacement matrix
    """
    B = np.zeros((6, 24))
    for i in range(8):
        xi, yi, zi = dNdx[i]
        B[0, 3*i + 0] = xi
        B[1, 3*i + 1] = yi
        B[2, 3*i + 2] = zi
        B[3, 3*i + 1] = zi
        B[3, 3*i + 2] = yi
        B[4, 3*i + 0] = zi
        B[4, 3*i + 2] = xi
        B[5, 3*i + 0] = yi
        B[5, 3*i + 1] = xi
    return B


def dNdx_hex8_center(elem_size: float = 1.0) -> np.ndarray:
    """
    Compute shape function derivatives at the center of a regular H8 element.
    Assumes a cube with uniform element size.

    Returns:
        dNdx : (8, 3) derivatives of shape functions w.r.t physical x, y, z
    """
    a = elem_size / 2.0
    dN_dxi = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ]) * 0.125

    # For a regular cube, the Jacobian is diagonal and constant:
    J = np.eye(3) * a
    dNdx = dN_dxi @ np.linalg.inv(J)
    return dNdx


def generate_B_matrices(nelx: int, nely: int, nelz: int, elem_size: float = 1.0) -> np.ndarray:
    """
    Generate B matrices for a regular grid of H8 elements with unit size.

    Parameters:
        nelx, nely, nelz : number of elements in x, y, z
        elem_size : cube size (assumed uniform)

    Returns:
        B_matrices : (nele, 6, 24)
    """
    nele = nelx * nely * nelz
    dNdx_ref = dNdx_hex8_center(elem_size)
    B = compute_B_matrix_H8(dNdx_ref)
    return np.repeat(B[None, :, :], nele, axis=0)