"""Orthotropic material-failure evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from pytopo3d.utils.config_loader import (
    MaterialStrength,
    validate_material_strength,
)


FAILURE_MODE_LABELS = np.array(
    [
        "X tension",
        "X compression",
        "Y tension",
        "Y compression",
        "Z tension",
        "Z compression",
        "XY shear",
        "YZ shear",
        "ZX shear",
    ],
    dtype="<U13",
)


@dataclass(frozen=True)
class MaximumStressResult:
    """Maximum-stress ratios and controlling mode at arbitrary sample points."""

    failure_components: np.ndarray
    failure_index: np.ndarray
    critical_mode_index: np.ndarray
    critical_mode: np.ndarray


@dataclass(frozen=True)
class GaussFailureResult:
    """Maximum-stress results reduced from Gauss points to H8 elements."""

    failure_components_gauss: np.ndarray
    failure_index_gauss: np.ndarray
    failure_index_element: np.ndarray
    critical_mode_index_gauss: np.ndarray
    critical_failure_mode_gauss: np.ndarray
    critical_gauss_point_element: np.ndarray
    critical_failure_mode_element: np.ndarray


@dataclass(frozen=True)
class CriticalFailure:
    """One exact critical location and its controlling failure mode."""

    failure_index: float
    element: int
    gauss_point: int
    mode_index: int
    mode: str


def _coerce_strength(
    strength: MaterialStrength | Mapping[str, object],
) -> MaterialStrength:
    if isinstance(strength, MaterialStrength):
        return strength
    if isinstance(strength, Mapping):
        return validate_material_strength(strength)
    raise TypeError("strength must be MaterialStrength or a strength mapping")


def maximum_stress_components(
    stress_material: np.ndarray,
    strength: MaterialStrength | Mapping[str, object],
) -> np.ndarray:
    """Return six directional maximum-stress ratios.

    The final input dimension follows
    ``[sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_zx]``. The output has
    the same shape and contains one sign-selected normal ratio for X/Y/Z plus
    three absolute shear ratios.
    """
    stress_material = np.asarray(stress_material, dtype=float)
    if stress_material.ndim < 1 or stress_material.shape[-1] != 6:
        raise ValueError(
            f"stress_material must have final dimension 6, got "
            f"{stress_material.shape}"
        )
    if not np.all(np.isfinite(stress_material)):
        raise ValueError("stress_material must contain only finite values")

    strength = _coerce_strength(strength)
    tensile = np.array([strength.X_t, strength.Y_t, strength.Z_t])
    compressive = np.array([strength.X_c, strength.Y_c, strength.Z_c])
    shear = np.array([strength.S_xy, strength.S_yz, strength.S_zx])

    normal_stress = stress_material[..., :3]
    components = np.empty_like(stress_material, dtype=float)
    components[..., :3] = np.where(
        normal_stress >= 0.0,
        normal_stress / tensile,
        -normal_stress / compressive,
    )
    components[..., 3:] = np.abs(stress_material[..., 3:]) / shear
    return components


def evaluate_maximum_stress(
    stress_material: np.ndarray,
    strength: MaterialStrength | Mapping[str, object],
) -> MaximumStressResult:
    """Evaluate the maximum-stress index and controlling directional mode."""
    stress_material = np.asarray(stress_material, dtype=float)
    components = maximum_stress_components(stress_material, strength)
    critical_component = np.argmax(components, axis=-1)
    failure_index = np.take_along_axis(
        components,
        critical_component[..., None],
        axis=-1,
    )[..., 0]

    mode_by_component = np.empty(components.shape, dtype=np.int8)
    for normal_component in range(3):
        mode_by_component[..., normal_component] = np.where(
            stress_material[..., normal_component] >= 0.0,
            2 * normal_component,
            2 * normal_component + 1,
        )
    mode_by_component[..., 3] = 6
    mode_by_component[..., 4] = 7
    mode_by_component[..., 5] = 8
    critical_mode_index = np.take_along_axis(
        mode_by_component,
        critical_component[..., None],
        axis=-1,
    )[..., 0]

    return MaximumStressResult(
        failure_components=components,
        failure_index=failure_index,
        critical_mode_index=critical_mode_index,
        critical_mode=FAILURE_MODE_LABELS[critical_mode_index],
    )


def evaluate_gauss_maximum_stress(
    stress_material_gauss: np.ndarray,
    strength: MaterialStrength | Mapping[str, object],
) -> GaussFailureResult:
    """Evaluate and reduce maximum-stress failure over eight H8 Gauss points."""
    stress_material_gauss = np.asarray(stress_material_gauss, dtype=float)
    if stress_material_gauss.ndim != 3 or stress_material_gauss.shape[1:] != (
        8,
        6,
    ):
        raise ValueError(
            "stress_material_gauss must have shape "
            f"(number_of_elements, 8, 6), got {stress_material_gauss.shape}"
        )

    point_result = evaluate_maximum_stress(stress_material_gauss, strength)
    critical_gauss_point = np.argmax(point_result.failure_index, axis=1)
    element_indices = np.arange(stress_material_gauss.shape[0])
    failure_index_element = point_result.failure_index[
        element_indices, critical_gauss_point
    ]
    critical_failure_mode_element = point_result.critical_mode[
        element_indices, critical_gauss_point
    ]

    return GaussFailureResult(
        failure_components_gauss=point_result.failure_components,
        failure_index_gauss=point_result.failure_index,
        failure_index_element=failure_index_element,
        critical_mode_index_gauss=point_result.critical_mode_index,
        critical_failure_mode_gauss=point_result.critical_mode,
        critical_gauss_point_element=critical_gauss_point,
        critical_failure_mode_element=critical_failure_mode_element,
    )


def critical_failure_location(
    result: GaussFailureResult,
    *,
    eligible_elements: np.ndarray | None = None,
) -> CriticalFailure:
    """Return the exact maximum over all eligible elements and Gauss points."""
    number_of_elements = result.failure_index_gauss.shape[0]
    if eligible_elements is None:
        eligible_elements = np.ones(number_of_elements, dtype=bool)
    else:
        eligible_elements = np.asarray(eligible_elements, dtype=bool)
        if eligible_elements.shape != (number_of_elements,):
            raise ValueError(
                f"eligible_elements must have shape ({number_of_elements},), "
                f"got {eligible_elements.shape}"
            )
    if not np.any(eligible_elements):
        raise ValueError("at least one element must be eligible for failure evaluation")

    eligible_failure = np.where(
        eligible_elements[:, None],
        result.failure_index_gauss,
        -np.inf,
    )
    flat_location = int(np.argmax(eligible_failure))
    element, gauss_point = np.unravel_index(
        flat_location, result.failure_index_gauss.shape
    )
    mode_index = int(result.critical_mode_index_gauss[element, gauss_point])
    return CriticalFailure(
        failure_index=float(result.failure_index_gauss[element, gauss_point]),
        element=int(element),
        gauss_point=int(gauss_point),
        mode_index=mode_index,
        mode=str(FAILURE_MODE_LABELS[mode_index]),
    )


def predicted_failure_load(
    reference_load: float,
    maximum_failure_index: float,
) -> float:
    """Scale a positive linear-elastic reference load to first initiation."""
    reference_load = float(reference_load)
    maximum_failure_index = float(maximum_failure_index)
    if not np.isfinite(reference_load) or reference_load <= 0.0:
        raise ValueError(
            f"reference_load must be finite and positive, got {reference_load}"
        )
    if not np.isfinite(maximum_failure_index) or maximum_failure_index < 0.0:
        raise ValueError(
            "maximum_failure_index must be finite and nonnegative, got "
            f"{maximum_failure_index}"
        )
    if maximum_failure_index == 0.0:
        return float("inf")
    return reference_load / maximum_failure_index

