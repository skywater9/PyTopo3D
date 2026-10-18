"""
Stiffness matrix generation for 3D topology optimization.

This module contains the function to generate the element stiffness matrix
for a hexahedral (H8) element.
"""

import numpy as np

from pytopo3d.utils.assembly import H8_NODE_OFFSETS


H8_GAUSS_POINTS = np.array(
    [
        [-1 / np.sqrt(3), -1 / np.sqrt(3), -1 / np.sqrt(3)],
        [1 / np.sqrt(3), -1 / np.sqrt(3), -1 / np.sqrt(3)],
        [-1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)],
        [1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)],
        [-1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)],
        [1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)],
        [-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
        [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
    ],
    dtype=float,
)


def h8_shape_function_gradients(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Return H8 shape-function gradients in natural coordinates."""
    node_signs = 2.0 * np.asarray(H8_NODE_OFFSETS, dtype=float) - 1.0
    gradients = np.zeros((8, 3), dtype=float)
    for index, (xi_sign, eta_sign, zeta_sign) in enumerate(node_signs):
        gradients[index, 0] = (
            0.125
            * xi_sign
            * (1.0 + eta_sign * eta)
            * (1.0 + zeta_sign * zeta)
        )
        gradients[index, 1] = (
            0.125
            * eta_sign
            * (1.0 + xi_sign * xi)
            * (1.0 + zeta_sign * zeta)
        )
        gradients[index, 2] = (
            0.125
            * zeta_sign
            * (1.0 + xi_sign * xi)
            * (1.0 + eta_sign * eta)
        )
    return gradients


def h8_strain_displacement_matrix(shape_gradients: np.ndarray) -> np.ndarray:
    """Build the engineering-strain B matrix from physical H8 gradients."""
    shape_gradients = np.asarray(shape_gradients, dtype=float)
    if shape_gradients.shape != (8, 3):
        raise ValueError(
            f"shape_gradients must have shape (8, 3), got {shape_gradients.shape}"
        )

    matrix = np.zeros((6, 24), dtype=float)
    for index, (dN_dx, dN_dy, dN_dz) in enumerate(shape_gradients):
        column = 3 * index
        matrix[:, column : column + 3] = np.array(
            [
                [dN_dx, 0.0, 0.0],
                [0.0, dN_dy, 0.0],
                [0.0, 0.0, dN_dz],
                [dN_dy, dN_dx, 0.0],
                [0.0, dN_dz, dN_dy],
                [dN_dz, 0.0, dN_dx],
            ]
        )
    return matrix


def h8_gauss_integration_data(elem_size: float = 1.0):
    """Return B matrices, Jacobian determinants, and weights at 8 Gauss points."""
    if not np.isfinite(elem_size) or elem_size <= 0.0:
        raise ValueError(f"elem_size must be finite and positive, got {elem_size}")

    node_signs = 2.0 * np.asarray(H8_NODE_OFFSETS, dtype=float) - 1.0
    coordinates = node_signs * (0.5 * elem_size)
    matrices = np.empty((8, 6, 24), dtype=float)
    determinants = np.empty(8, dtype=float)
    weights = np.ones(8, dtype=float)

    for index, (xi, eta, zeta) in enumerate(H8_GAUSS_POINTS):
        natural_gradients = h8_shape_function_gradients(xi, eta, zeta)
        jacobian = natural_gradients.T @ coordinates
        determinant = np.linalg.det(jacobian)
        if determinant <= 0.0:
            raise ValueError("Negative or zero Jacobian determinant.")
        physical_gradients = np.linalg.solve(
            jacobian.T, natural_gradients.T
        ).T
        matrices[index] = h8_strain_displacement_matrix(physical_gradients)
        determinants[index] = determinant

    return matrices, determinants, weights

def lk_H8(
    E_x: float = 1,
    E_y: float = None,
    E_z: float = None,
    G_xy: float = 0.4,
    G_yz: float = None,
    G_zx: float = None,
    nu_xy: float = 0.3,
    nu_yz: float = None,
    nu_zx: float = None,
    elem_size: float = 1.0
) -> np.ndarray:
    """
    Generate the 24x24 element stiffness matrix for a fully anisotropic material
    using 8-point Gauss integration.

    Parameters
    ----------
    E_x, E_y, E_z : float
        Young's moduli along x, y, z axes.
    nu_xy, nu_yz, nu_zx : float
        Poisson's ratios.
    G_xy, G_yz, G_zx : float
        Shear moduli.
    elem_size : float
        Side length of each H8 element in meters.

    Returns
    -------
    KE : ndarray
        24x24 element stiffness matrix.
    """
    C = make_C_matrix(
        E_x,
        E_y,
        E_z,
        G_xy,
        G_yz,
        G_zx,
        nu_xy,
        nu_yz,
        nu_zx,
        normalize=False
    )
    assert C.shape == (6, 6), "Elasticity tensor must be 6x6 in Voigt notation."

    matrices, determinants, weights = h8_gauss_integration_data(elem_size)
    KE = np.zeros((24, 24))
    for matrix, determinant, weight in zip(matrices, determinants, weights):
        KE += matrix.T @ C @ matrix * determinant * weight

    return KE


def make_C_matrix(
    E_x: float,
    E_y: float = None,
    E_z: float = None,
    G_xy: float = 0.4,
    G_yz: float = None,
    G_zx: float = None,
    nu_xy: float = 0.3,
    nu_yz: float = None,
    nu_zx: float = None,
    normalize: bool = False
) -> np.ndarray:
    """
    Generate a 6x6 stiffness matrix C in Voigt notation for 3D anisotropic materials.

    Parameters
    ----------
    E_x, E_y, E_z : float
        Young's moduli along x, y, z axes.
    nu_xy, nu_yz, nu_zx : float
        Poisson's ratios.
    G_xy, G_yz, G_zx : float
        Shear moduli.
    normalize : bool
        If True, normalizes so that E_x = 1.0.

    Returns
    -------
    C : ndarray
        6x6 stiffness matrix in Voigt notation.
    """

    if E_y is None:
        E_y = E_x
    if E_z is None:
        E_z = E_x
    if G_yz is None:
        G_yz = G_xy
    if G_zx is None:
        G_zx = G_xy
    if nu_yz is None:
        nu_yz = nu_xy
    if nu_zx is None:
        nu_zx = nu_xy

    S = np.zeros((6, 6))

    S[0, 0] = 1 / E_x
    S[1, 1] = 1 / E_y
    S[2, 2] = 1 / E_z

    S[0, 1] = S[1, 0] = -nu_xy / E_x
    S[1, 2] = S[2, 1] = -nu_yz / E_y
    S[0, 2] = S[2, 0] = -nu_zx / E_z

    # This order must match make_B: [xx, yy, zz, xy, yz, zx].
    S[3, 3] = 1 / G_xy
    S[4, 4] = 1 / G_yz
    S[5, 5] = 1 / G_zx

    C = np.linalg.inv(S)

    if normalize and E_x != 0:
        return C / E_x
    return C
