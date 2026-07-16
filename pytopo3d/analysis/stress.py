"""H8 Gauss-point strain and full-density stress recovery."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from pytopo3d.utils.stiffness import h8_gauss_integration_data


def _validated_element_displacements(
    displacement: np.ndarray,
    edof_matrix: np.ndarray,
) -> np.ndarray:
    displacement = np.asarray(displacement, dtype=float)
    edof_matrix = np.asarray(edof_matrix)

    if displacement.ndim != 1:
        raise ValueError(
            f"displacement must be one-dimensional, got shape {displacement.shape}"
        )
    if edof_matrix.ndim != 2 or edof_matrix.shape[1] != 24:
        raise ValueError(
            f"edof_matrix must have shape (number_of_elements, 24), "
            f"got {edof_matrix.shape}"
        )
    if not np.issubdtype(edof_matrix.dtype, np.integer):
        if not np.all(np.equal(edof_matrix, np.floor(edof_matrix))):
            raise ValueError("edof_matrix must contain integer 1-based DOF indices")
        edof_matrix = edof_matrix.astype(np.int64)
    if edof_matrix.size and (
        np.min(edof_matrix) < 1 or np.max(edof_matrix) > displacement.size
    ):
        raise ValueError(
            "edof_matrix contains a DOF outside the 1-based displacement range"
        )
    if not np.all(np.isfinite(displacement)):
        raise ValueError("displacement must contain only finite values")

    return displacement[edof_matrix.astype(np.int64, copy=False) - 1]


def recover_gauss_strain(
    displacement: np.ndarray,
    edof_matrix: np.ndarray,
    *,
    elem_size: float = 1.0,
) -> np.ndarray:
    """Recover engineering strain at all eight H8 Gauss points.

    The result has shape ``(number_of_elements, 8, 6)`` and uses
    ``[epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_yz, gamma_zx]``.
    ``edof_matrix`` follows the solver's existing 1-based DOF convention.
    """
    element_displacements = _validated_element_displacements(
        displacement, edof_matrix
    )
    b_matrices, _, _ = h8_gauss_integration_data(elem_size)
    return np.einsum(
        "gij,ej->egi",
        b_matrices,
        element_displacements,
        optimize=True,
    )


def recover_gauss_stress(
    displacement: np.ndarray,
    edof_matrix: np.ndarray,
    constitutive_matrix: np.ndarray,
    *,
    elem_size: float = 1.0,
    return_strain: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Recover full-density global stress at every H8 Gauss point.

    Stress uses ``[sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_zx]``.
    No SIMP density scaling or gray-element relaxation is applied in this
    solid-element recovery function.
    """
    constitutive_matrix = np.asarray(constitutive_matrix, dtype=float)
    if constitutive_matrix.shape != (6, 6):
        raise ValueError(
            f"constitutive_matrix must have shape (6, 6), "
            f"got {constitutive_matrix.shape}"
        )
    if not np.all(np.isfinite(constitutive_matrix)):
        raise ValueError("constitutive_matrix must contain only finite values")

    strain = recover_gauss_strain(
        displacement,
        edof_matrix,
        elem_size=elem_size,
    )
    stress = np.einsum(
        "ij,egj->egi",
        constitutive_matrix,
        strain,
        optimize=True,
    )
    if return_strain:
        return stress, strain
    return stress

