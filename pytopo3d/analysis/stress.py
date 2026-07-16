"""H8 Gauss-point strain and full-density stress recovery."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from pytopo3d.utils.stiffness import h8_gauss_integration_data


STRESS_VOIGT_ORDER = (
    "sigma_xx",
    "sigma_yy",
    "sigma_zz",
    "tau_xy",
    "tau_yz",
    "tau_zx",
)


def stress_voigt_to_tensor(stress_voigt: np.ndarray) -> np.ndarray:
    """Convert ``[..., 6]`` stress Voigt vectors to symmetric tensors."""
    stress_voigt = np.asarray(stress_voigt, dtype=float)
    if stress_voigt.ndim < 1 or stress_voigt.shape[-1] != 6:
        raise ValueError(
            f"stress_voigt must have final dimension 6, got {stress_voigt.shape}"
        )
    if not np.all(np.isfinite(stress_voigt)):
        raise ValueError("stress_voigt must contain only finite values")

    tensor = np.zeros(stress_voigt.shape[:-1] + (3, 3), dtype=float)
    tensor[..., 0, 0] = stress_voigt[..., 0]
    tensor[..., 1, 1] = stress_voigt[..., 1]
    tensor[..., 2, 2] = stress_voigt[..., 2]
    tensor[..., 0, 1] = tensor[..., 1, 0] = stress_voigt[..., 3]
    tensor[..., 1, 2] = tensor[..., 2, 1] = stress_voigt[..., 4]
    tensor[..., 2, 0] = tensor[..., 0, 2] = stress_voigt[..., 5]
    return tensor


def stress_tensor_to_voigt(stress_tensor: np.ndarray) -> np.ndarray:
    """Convert symmetric ``[..., 3, 3]`` stress tensors to Voigt vectors."""
    stress_tensor = np.asarray(stress_tensor, dtype=float)
    if stress_tensor.ndim < 2 or stress_tensor.shape[-2:] != (3, 3):
        raise ValueError(
            f"stress_tensor must have final dimensions (3, 3), got "
            f"{stress_tensor.shape}"
        )
    if not np.all(np.isfinite(stress_tensor)):
        raise ValueError("stress_tensor must contain only finite values")
    if not np.allclose(
        stress_tensor,
        np.swapaxes(stress_tensor, -1, -2),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("stress_tensor must be symmetric")

    return np.stack(
        (
            stress_tensor[..., 0, 0],
            stress_tensor[..., 1, 1],
            stress_tensor[..., 2, 2],
            stress_tensor[..., 0, 1],
            stress_tensor[..., 1, 2],
            stress_tensor[..., 2, 0],
        ),
        axis=-1,
    )


def validate_orientation_matrix(orientation_matrix: np.ndarray) -> np.ndarray:
    """Validate an orthogonal material-to-global orientation matrix."""
    orientation_matrix = np.asarray(orientation_matrix, dtype=float)
    if orientation_matrix.shape != (3, 3):
        raise ValueError(
            f"orientation_matrix must have shape (3, 3), got "
            f"{orientation_matrix.shape}"
        )
    if not np.all(np.isfinite(orientation_matrix)):
        raise ValueError("orientation_matrix must contain only finite values")
    if not np.allclose(
        orientation_matrix.T @ orientation_matrix,
        np.eye(3),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("orientation_matrix must be orthogonal")
    determinant = np.linalg.det(orientation_matrix)
    if not np.isclose(abs(determinant), 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("orientation_matrix determinant must have magnitude 1")
    return orientation_matrix


def rotate_stress_to_material(
    stress_global: np.ndarray,
    orientation_matrix: np.ndarray,
) -> np.ndarray:
    """Rotate global stress into material coordinates using ``R.T @ s @ R``.

    ``orientation_matrix`` maps material basis vectors into global coordinates.
    The input and output use :data:`STRESS_VOIGT_ORDER`, and arbitrary leading
    dimensions (including element and Gauss-point axes) are preserved.
    """
    orientation_matrix = validate_orientation_matrix(orientation_matrix)
    global_tensor = stress_voigt_to_tensor(stress_global)
    material_tensor = np.einsum(
        "ia,...ij,jb->...ab",
        orientation_matrix,
        global_tensor,
        orientation_matrix,
        optimize=True,
    )
    return stress_tensor_to_voigt(material_tensor)


def stress_rotation_matrix_to_material(
    orientation_matrix: np.ndarray,
) -> np.ndarray:
    """Return the 6x6 linear global-to-material stress transformation."""
    basis = np.eye(6)
    return rotate_stress_to_material(basis, orientation_matrix).T


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
