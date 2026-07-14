"""
Stiffness matrix generation for 3D topology optimization.

This module contains the function to generate the element stiffness matrix
for a hexahedral (H8) element.
"""

import numpy as np

from pytopo3d.utils.assembly import H8_NODE_OFFSETS

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

    # Gauss points and weights for 2-point Gauss quadrature
    gpts = np.array([[-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)],
                     [ 1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)],
                     [-1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)],
                     [ 1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)],
                     [-1/np.sqrt(3), -1/np.sqrt(3),  1/np.sqrt(3)],
                     [ 1/np.sqrt(3), -1/np.sqrt(3),  1/np.sqrt(3)],
                     [-1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)],
                     [ 1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)]])
    weights = np.ones(8)

    # Natural-coordinate signs in the same local-corner order used by
    # build_edof.  Keep these +/-1 signs separate from physical coordinates;
    # the H8 shape functions are defined on [-1, 1]^3.
    node_signs = 2.0 * np.asarray(H8_NODE_OFFSETS, dtype=float) - 1.0

    # Shape function derivatives with respect to ξ, η, ζ
    def shape_fn_grad(ξ, η, ζ):
        dN = np.zeros((8, 3))
        for i, (xi_sign, eta_sign, zeta_sign) in enumerate(node_signs):
            dN[i, 0] = (
                0.125
                * xi_sign
                * (1.0 + eta_sign * η)
                * (1.0 + zeta_sign * ζ)
            )
            dN[i, 1] = (
                0.125
                * eta_sign
                * (1.0 + xi_sign * ξ)
                * (1.0 + zeta_sign * ζ)
            )
            dN[i, 2] = (
                0.125
                * zeta_sign
                * (1.0 + xi_sign * ξ)
                * (1.0 + eta_sign * η)
            )
        return dN

    # B-matrix constructor
    def make_B(dNdx):
        B = np.zeros((6, 24))
        for i in range(8):
            i3 = i * 3
            dNxi, dNyi, dNzi = dNdx[i]
            B[:, i3:i3+3] = np.array([
                [dNxi,     0,     0],
                [    0, dNyi,     0],
                [    0,     0, dNzi],
                [dNyi, dNxi,     0],
                [    0, dNzi, dNyi],
                [dNzi,     0, dNxi]
            ])
        return B

    # Physical coordinates of a cube centered at the origin.
    coords = node_signs * (0.5 * elem_size)

    KE = np.zeros((24, 24))
    for w, (ξ, η, ζ) in zip(weights, gpts):
        # Compute shape function gradients in physical coordinates
        dN_dxi = shape_fn_grad(ξ, η, ζ)    # 8x3
        J = dN_dxi.T @ coords              # 3x3 Jacobian
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError("Negative or zero Jacobian determinant.")

        dN_dx = np.linalg.solve(J.T, dN_dxi.T).T  # 8x3 shape fn grads in x,y,z
        B = make_B(dN_dx)
        KE += B.T @ C @ B * detJ * w

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
