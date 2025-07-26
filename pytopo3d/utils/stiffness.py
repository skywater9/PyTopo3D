"""
Stiffness matrix generation for 3D topology optimization.

This module contains the function to generate the element stiffness matrix
for a hexahedral (H8) element.
"""

import numpy as np

def lk_H8(
    E_x: float = 1.0,
    E_y: float = 1.0,
    E_z: float = 1.0,
    nu_xy: float = 0.3,
    nu_yz: float = 0.3,
    nu_zx: float = 0.3,
    G_xy: float = 0.4,
    G_yz: float = 0.4,
    G_zx: float = 0.4,
    material_type: str = "isotropic",
    normalize: bool = True
) -> np.ndarray:
    """
    Generate the 24x24 element stiffness matrix for a fully anisotropic material
    using 8-point Gauss integration.

    Parameters
    ----------
    C : ndarray
        6x6 anisotropic stiffness matrix in Voigt notation.

    Returns
    -------
    KE : ndarray
        24x24 element stiffness matrix.
    """
    C = make_C_matrix(
        E_x,
        E_y,
        E_z,
        nu_xy,
        nu_yz,
        nu_zx,
        G_xy,
        G_yz,
        G_zx,
        material_type,
        normalize
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

    # Node positions in reference element
    node_locs = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1]
    ]) * 0.5

    # Shape function derivatives with respect to ξ, η, ζ
    def shape_fn_grad(ξ, η, ζ):
        dN = np.zeros((8, 3))
        for i, (xi, eta, zeta) in enumerate(node_locs):
            dN[i, 0] = 0.125 * (1 + eta * η) * (1 + zeta * ζ) * xi / abs(xi)
            dN[i, 1] = 0.125 * (1 + xi * ξ) * (1 + zeta * ζ) * eta / abs(eta)
            dN[i, 2] = 0.125 * (1 + xi * ξ) * (1 + eta * η) * zeta / abs(zeta)
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

    # Hex element with unit dimensions
    coords = node_locs

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
    E_y: float,
    E_z: float,
    nu_xy: float,
    nu_yz: float,
    nu_zx: float,
    G_xy: float,
    G_yz: float,
    G_zx: float,
    material_type: str,
    normalize: bool
) -> np.ndarray:
    """
    Generate a 6x6 stiffness matrix C in Voigt notation for 3D anisotropic materials.

    Parameters
    ----------
    material_type : str
        'isotropic', 'orthotropic', or 'transversely_isotropic'.
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
    if material_type == "isotropic":
        # Variables: E_x, nu_xy
        E = E_x
        nu = nu_xy
        G = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = G

        C = np.array([
            [lam + 2*mu, lam,        lam,        0,     0,     0],
            [lam,        lam + 2*mu, lam,        0,     0,     0],
            [lam,        lam,        lam + 2*mu, 0,     0,     0],
            [0,          0,          0,          mu,    0,     0],
            [0,          0,          0,          0,     mu,    0],
            [0,          0,          0,          0,     0,     mu]
        ])

    elif material_type == "transversely_isotropic":
        # Variables: E_x, E_z, nu_xy, nu_zx, G_xy, G_zx
        # Assumes transverse isotropy about Z-axis
        S = np.zeros((6, 6))

        S[0, 0] = S[1, 1] = 1 / E_x
        S[2, 2] = 1 / E_z
        S[0, 1] = S[1, 0] = -nu_xy / E_x
        S[0, 2] = S[2, 0] = -nu_zx / E_z
        S[1, 2] = S[2, 1] = -nu_zx / E_z
        S[3, 3] = S[4, 4] = 1 / G_xy
        S[5, 5] = 1 / G_zx

        C = np.linalg.inv(S)

    elif material_type == "orthotropic":
        # Variables: E_x, E_z, E_y, nu_xy, nu_yz, nu_zx, G_xy, G_yz, G_zx
        S = np.zeros((6, 6))

        S[0, 0] = 1 / E_x
        S[1, 1] = 1 / E_y
        S[2, 2] = 1 / E_z

        S[0, 1] = S[1, 0] = -nu_xy / E_x
        S[1, 2] = S[2, 1] = -nu_yz / E_y
        S[0, 2] = S[2, 0] = -nu_zx / E_z

        S[3, 3] = 1 / G_yz
        S[4, 4] = 1 / G_zx
        S[5, 5] = 1 / G_xy

        C = np.linalg.inv(S)

    else:
        raise ValueError("Invalid material_type. Choose 'isotropic', 'orthotropic', or 'transversely_isotropic'.")

    if normalize and E_x != 0:
        return C / E_x
    return C