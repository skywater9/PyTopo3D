"""Projected and binary failure post-processing for completed topology runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from pytopo3d.analysis.failure import (
    CriticalFailure,
    GaussFailureResult,
    critical_failure_location,
    evaluate_gauss_maximum_stress,
    predicted_failure_load,
)
from pytopo3d.analysis.stress import (
    recover_gauss_stress,
    rotate_stress_to_material,
)
from pytopo3d.core.optimizer import evaluate_fixed_geometry_metrics
from pytopo3d.utils.config_loader import MaterialStrength
from pytopo3d.utils.stiffness import make_C_matrix


logger = logging.getLogger(__name__)

_BINARY_FAILURE_RECOMMENDATIONS = (
    "lower the optimization failure limit below one (for example 0.9)",
    "increase the smooth aggregate exponent",
    "recalibrate the aggregate correction factor",
    "add regional failure constraint groups",
    "strengthen the projection continuation schedule",
    "increase minimum feature-size control",
)


@dataclass(frozen=True)
class FailureFieldResult:
    """Stress/failure fields and exact regional critical locations."""

    density_shape: tuple[int, int, int]
    stress_global_gauss: np.ndarray
    stress_material_gauss: np.ndarray
    gauss_failure: GaussFailureResult
    critical_all_elements: Optional[CriticalFailure]
    critical_design_region: Optional[CriticalFailure]
    critical_fixture_region: Optional[CriticalFailure]
    all_element_count: int
    design_element_count: int
    fixture_element_count: int
    reference_load: float

    def summary(self, representation: str) -> Dict[str, Any]:
        """Return JSON-friendly exact failure metrics for one representation."""
        representation = representation.strip().lower()
        if not representation:
            raise ValueError("representation must not be empty")

        def predicted(critical):
            if critical is None or self.reference_load <= 0.0:
                return None
            value = float(
                predicted_failure_load(
                    self.reference_load,
                    critical.failure_index,
                )
            )
            return value if np.isfinite(value) else None

        def predicted_status(critical):
            if critical is None:
                return "not_evaluable_no_eligible_elements"
            if self.reference_load <= 0.0:
                return "not_evaluable_no_reference_load"
            if critical.failure_index == 0.0:
                return "unbounded_zero_failure_index"
            return "finite"

        def coordinates(critical):
            if critical is None:
                return None
            y_index, x_index, z_index = np.unravel_index(
                critical.element,
                self.density_shape,
                order="F",
            )
            return [int(y_index), int(x_index), int(z_index)]

        critical_all = self.critical_all_elements
        critical_design = self.critical_design_region
        critical_fixture = self.critical_fixture_region

        critical_region = "none"
        if critical_all is not None:
            if (
                critical_fixture is not None
                and critical_all.element == critical_fixture.element
            ):
                critical_region = "fixture_or_load"
            elif (
                critical_design is not None
                and critical_all.element == critical_design.element
            ):
                critical_region = "design"
            else:
                critical_region = "unclassified"

        metrics: Dict[str, Any] = {
            f"failure_index_max_{representation}": (
                None if critical_all is None else critical_all.failure_index
            ),
            f"predicted_failure_load_{representation}": predicted(critical_all),
            f"critical_element_{representation}": (
                None if critical_all is None else critical_all.element
            ),
            f"critical_element_yxz_{representation}": coordinates(critical_all),
            f"critical_gauss_point_{representation}": (
                None if critical_all is None else critical_all.gauss_point
            ),
            f"critical_mode_{representation}": (
                None if critical_all is None else critical_all.mode
            ),
            f"critical_region_{representation}": critical_region,
            f"critical_region_is_fixture_or_load_{representation}": (
                critical_region == "fixture_or_load"
            ),
            f"max_failure_index_all_elements_{representation}": (
                None if critical_all is None else critical_all.failure_index
            ),
            f"max_failure_index_design_region_{representation}": (
                None if critical_design is None else critical_design.failure_index
            ),
            f"predicted_failure_load_design_region_{representation}": predicted(
                critical_design
            ),
            f"predicted_failure_load_status_design_region_{representation}": (
                predicted_status(critical_design)
            ),
            f"critical_element_design_region_{representation}": (
                None if critical_design is None else critical_design.element
            ),
            f"critical_gauss_point_design_region_{representation}": (
                None if critical_design is None else critical_design.gauss_point
            ),
            f"critical_mode_design_region_{representation}": (
                None if critical_design is None else critical_design.mode
            ),
            f"max_failure_index_fixture_region_{representation}": (
                None if critical_fixture is None else critical_fixture.failure_index
            ),
            f"predicted_failure_load_fixture_region_{representation}": predicted(
                critical_fixture
            ),
            f"predicted_failure_load_status_fixture_region_{representation}": (
                predicted_status(critical_fixture)
            ),
            f"critical_element_fixture_region_{representation}": (
                None if critical_fixture is None else critical_fixture.element
            ),
            f"critical_gauss_point_fixture_region_{representation}": (
                None if critical_fixture is None else critical_fixture.gauss_point
            ),
            f"critical_mode_fixture_region_{representation}": (
                None if critical_fixture is None else critical_fixture.mode
            ),
            f"predicted_failure_load_status_{representation}": predicted_status(
                critical_all
            ),
            f"failure_all_element_count_{representation}": self.all_element_count,
            f"failure_design_element_count_{representation}": (
                self.design_element_count
            ),
            f"failure_fixture_element_count_{representation}": (
                self.fixture_element_count
            ),
            f"failure_reference_load_{representation}": self.reference_load,
            f"failure_stress_model_{representation}": "full_density_unrelaxed",
        }
        return metrics


@dataclass(frozen=True)
class FailureRepresentationResults:
    """Independent projected and binary solves plus their failure fields."""

    projected_response: Dict[str, Any]
    binary_response: Dict[str, Any]
    projected: FailureFieldResult
    binary: FailureFieldResult
    binary_density: np.ndarray
    metrics: Dict[str, Any]
    saved_files: Dict[str, str]


def build_failure_region_masks(
    shape: tuple[int, int, int],
    *,
    obstacle_mask: Optional[np.ndarray] = None,
    protected_zone_mask: Optional[np.ndarray] = None,
    force_field: Optional[np.ndarray] = None,
    support_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return non-obstacle and idealized fixture/load element masks."""
    nely, nelx, nelz = shape
    obstacle = (
        np.zeros(shape, dtype=bool)
        if obstacle_mask is None
        else np.asarray(obstacle_mask, dtype=bool)
    )
    if obstacle.shape != shape:
        raise ValueError(f"obstacle_mask has shape {obstacle.shape}, expected {shape}")

    fixture = np.zeros(shape, dtype=bool)
    if protected_zone_mask is not None:
        protected = np.asarray(protected_zone_mask, dtype=bool)
        if protected.shape != shape:
            raise ValueError(
                f"protected_zone_mask has shape {protected.shape}, expected {shape}"
            )
        fixture |= protected

    if force_field is None:
        fixture[:, nelx - 1, 0] = True
    else:
        force_field = np.asarray(force_field)
        if force_field.shape != shape + (3,):
            raise ValueError(
                f"force_field has shape {force_field.shape}, expected {shape + (3,)}"
            )
        fixture |= np.any(force_field != 0.0, axis=-1)

    if support_mask is None:
        fixture[:, 0, :] = True
    else:
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != shape:
            raise ValueError(
                f"support_mask has shape {support.shape}, expected {shape}"
            )
        if np.any(support):
            fixture |= support
        else:
            fixture[:, 0, :] = True

    non_obstacle = ~obstacle
    fixture &= non_obstacle
    return non_obstacle, fixture


def evaluate_failure_fields(
    *,
    density_shape: tuple[int, int, int],
    displacement: np.ndarray,
    edof_matrix: np.ndarray,
    constitutive_matrix_global: np.ndarray,
    orientation_matrix: np.ndarray,
    strength: MaterialStrength,
    elem_size: float,
    all_elements: np.ndarray,
    design_elements: np.ndarray,
    fixture_elements: np.ndarray,
    reference_load: float,
) -> FailureFieldResult:
    """Recover unrelaxed full-density stress and exact regional failure."""
    expected_shape = density_shape
    masks = []
    for name, value in (
        ("all_elements", all_elements),
        ("design_elements", design_elements),
        ("fixture_elements", fixture_elements),
    ):
        mask = np.asarray(value, dtype=bool)
        if mask.shape != expected_shape:
            raise ValueError(f"{name} has shape {mask.shape}, expected {expected_shape}")
        masks.append(mask)
    all_elements, design_elements, fixture_elements = masks

    if edof_matrix.shape[0] != int(np.prod(density_shape)):
        raise ValueError(
            "edof_matrix element count does not match density_shape in Fortran order"
        )

    stress_global = recover_gauss_stress(
        displacement,
        edof_matrix,
        constitutive_matrix_global,
        elem_size=elem_size,
    )
    stress_material = rotate_stress_to_material(
        stress_global,
        orientation_matrix,
    )
    gauss_failure = evaluate_gauss_maximum_stress(stress_material, strength)

    def critical(mask):
        flat_mask = mask.ravel(order="F")
        if not np.any(flat_mask):
            return None
        return critical_failure_location(
            gauss_failure,
            eligible_elements=flat_mask,
        )

    return FailureFieldResult(
        density_shape=density_shape,
        stress_global_gauss=stress_global,
        stress_material_gauss=stress_material,
        gauss_failure=gauss_failure,
        critical_all_elements=critical(all_elements),
        critical_design_region=critical(design_elements),
        critical_fixture_region=critical(fixture_elements),
        all_element_count=int(np.sum(all_elements)),
        design_element_count=int(np.sum(design_elements)),
        fixture_element_count=int(np.sum(fixture_elements)),
        reference_load=float(reference_load),
    )


def save_failure_fields(
    results_manager,
    result: FailureFieldResult,
    representation: str,
) -> Dict[str, str]:
    """Save all detailed stress/failure arrays with a representation suffix."""
    representation = representation.strip().lower()
    arrays = {
        f"stress_global_gauss_{representation}.npy": result.stress_global_gauss,
        f"stress_material_gauss_{representation}.npy": result.stress_material_gauss,
        f"failure_components_gauss_{representation}.npy": (
            result.gauss_failure.failure_components_gauss
        ),
        f"failure_index_gauss_{representation}.npy": (
            result.gauss_failure.failure_index_gauss
        ),
        f"failure_index_element_{representation}.npy": (
            result.gauss_failure.failure_index_element
        ),
        f"critical_failure_mode_{representation}.npy": (
            result.gauss_failure.critical_failure_mode_gauss
        ),
    }
    return {
        filename: results_manager.save_result(array, filename)
        for filename, array in arrays.items()
    }


def evaluate_failure_representations(
    *,
    x_projected: np.ndarray,
    binary_threshold: float,
    penal: float,
    material_params: Sequence[float],
    strength: MaterialStrength,
    orientation_matrix: np.ndarray,
    elem_size: float,
    force_field: Optional[np.ndarray] = None,
    support_mask: Optional[np.ndarray] = None,
    obstacle_mask: Optional[np.ndarray] = None,
    protected_zone_mask: Optional[np.ndarray] = None,
    smooth_failure_aggregate: Optional[float] = None,
    smooth_failure_limit: Optional[float] = None,
    failure_feasibility_tolerance: float = 1.0e-9,
    smooth_failure_relative_tolerance: float = 1.0e-3,
    use_gpu: bool = False,
    results_manager=None,
) -> FailureRepresentationResults:
    """Perform independent projected/binary solves and exact failure recovery."""
    x_projected = np.asarray(x_projected, dtype=float)
    if x_projected.ndim != 3:
        raise ValueError(f"x_projected must be 3-D, got {x_projected.shape}")
    if not np.all(np.isfinite(x_projected)):
        raise ValueError("x_projected must contain only finite values")
    if np.any((x_projected < 0.0) | (x_projected > 1.0)):
        raise ValueError("x_projected values must lie in [0, 1]")
    if not np.isfinite(binary_threshold) or not 0.0 <= binary_threshold <= 1.0:
        raise ValueError(
            f"binary_threshold must be finite and in [0, 1], got {binary_threshold}"
        )
    failure_feasibility_tolerance = float(failure_feasibility_tolerance)
    if (
        not np.isfinite(failure_feasibility_tolerance)
        or failure_feasibility_tolerance < 0.0
    ):
        raise ValueError(
            "failure_feasibility_tolerance must be finite and nonnegative"
        )
    smooth_failure_relative_tolerance = float(smooth_failure_relative_tolerance)
    if (
        not np.isfinite(smooth_failure_relative_tolerance)
        or smooth_failure_relative_tolerance < 0.0
    ):
        raise ValueError(
            "smooth_failure_relative_tolerance must be finite and nonnegative"
        )
    if (smooth_failure_aggregate is None) != (smooth_failure_limit is None):
        raise ValueError(
            "smooth_failure_aggregate and smooth_failure_limit must be supplied together"
        )
    if smooth_failure_aggregate is not None:
        smooth_failure_aggregate = float(smooth_failure_aggregate)
        smooth_failure_limit = float(smooth_failure_limit)
        if (
            not np.isfinite(smooth_failure_aggregate)
            or smooth_failure_aggregate < 0.0
        ):
            raise ValueError(
                "smooth_failure_aggregate must be finite and nonnegative"
            )
        if not np.isfinite(smooth_failure_limit) or smooth_failure_limit <= 0.0:
            raise ValueError("smooth_failure_limit must be finite and positive")

    shape = x_projected.shape
    non_obstacle, fixture = build_failure_region_masks(
        shape,
        obstacle_mask=obstacle_mask,
        protected_zone_mask=protected_zone_mask,
        force_field=force_field,
        support_mask=support_mask,
    )

    protected = (
        np.zeros(shape, dtype=bool)
        if protected_zone_mask is None
        else np.asarray(protected_zone_mask, dtype=bool)
    )
    if protected.shape != shape:
        raise ValueError(
            f"protected_zone_mask has shape {protected.shape}, expected {shape}"
        )
    projected_density = np.array(x_projected, dtype=float, copy=True)
    projected_density[protected] = 1.0
    projected_density[~non_obstacle] = 0.0

    projected_response = evaluate_fixed_geometry_metrics(
        xPhys=projected_density,
        penal=penal,
        material_params=tuple(material_params),
        elem_size=elem_size,
        force_field=force_field,
        support_mask=support_mask,
        obstacle_mask=obstacle_mask,
        protected_zone_mask=protected_zone_mask,
        use_gpu=use_gpu,
        return_displacement=True,
    )

    binary_density = (projected_density >= binary_threshold).astype(float)
    # Only explicitly protected zones are prescribed solid.  The fixture mask
    # also labels idealized support/load boundaries for reporting; silently
    # solidifying those elements would no longer be a true thresholded solve.
    binary_density[protected] = 1.0
    binary_density[~non_obstacle] = 0.0
    binary_response = evaluate_fixed_geometry_metrics(
        xPhys=binary_density,
        penal=penal,
        material_params=tuple(material_params),
        elem_size=elem_size,
        force_field=force_field,
        support_mask=support_mask,
        obstacle_mask=obstacle_mask,
        protected_zone_mask=protected_zone_mask,
        use_gpu=use_gpu,
        return_displacement=True,
    )

    constitutive_global = make_C_matrix(*material_params)
    projected_all = non_obstacle
    projected_design = non_obstacle & ~fixture
    projected_fixture = non_obstacle & fixture
    binary_all = non_obstacle & (binary_density == 1.0)
    binary_design = binary_all & ~fixture
    binary_fixture = binary_all & fixture

    projected = evaluate_failure_fields(
        density_shape=shape,
        displacement=projected_response["displacement"],
        edof_matrix=projected_response["edof_matrix"],
        constitutive_matrix_global=constitutive_global,
        orientation_matrix=orientation_matrix,
        strength=strength,
        elem_size=elem_size,
        all_elements=projected_all,
        design_elements=projected_design,
        fixture_elements=projected_fixture,
        reference_load=projected_response["F_total"],
    )
    binary = evaluate_failure_fields(
        density_shape=shape,
        displacement=binary_response["displacement"],
        edof_matrix=binary_response["edof_matrix"],
        constitutive_matrix_global=constitutive_global,
        orientation_matrix=orientation_matrix,
        strength=strength,
        elem_size=elem_size,
        all_elements=binary_all,
        design_elements=binary_design,
        fixture_elements=binary_fixture,
        reference_load=binary_response["F_total"],
    )

    design_region = non_obstacle & ~fixture
    element_volume = float(elem_size) ** 3
    reference_element_count = int(np.sum(non_obstacle))
    design_element_count = int(np.sum(design_region))

    def density_metrics(density, response, representation):
        total_density = float(np.sum(density[non_obstacle]))
        design_density = float(np.sum(density[design_region]))
        return {
            f"compliance_{representation}": float(response["compliance"]),
            f"predicted_stiffness_{representation}": (
                None if response["k_avg"] is None else float(response["k_avg"])
            ),
            f"volume_fraction_{representation}": (
                None
                if reference_element_count == 0
                else total_density / reference_element_count
            ),
            f"design_region_volume_fraction_{representation}": (
                None
                if design_element_count == 0
                else design_density / design_element_count
            ),
            f"material_volume_m3_{representation}": total_density * element_volume,
            f"reference_volume_m3_{representation}": (
                reference_element_count * element_volume
            ),
        }

    projected_failure = (
        None
        if projected.critical_all_elements is None
        else float(projected.critical_all_elements.failure_index)
    )
    binary_failure = (
        None
        if binary.critical_all_elements is None
        else float(binary.critical_all_elements.failure_index)
    )

    def verification_metrics(failure_index, representation):
        feasible = (
            None
            if failure_index is None
            else bool(failure_index <= 1.0 + failure_feasibility_tolerance)
        )
        return {
            f"failure_strength_limit_{representation}": 1.0,
            f"failure_strength_margin_{representation}": (
                None if failure_index is None else 1.0 - failure_index
            ),
            f"failure_strength_feasible_{representation}": feasible,
            f"failure_verification_status_{representation}": (
                "not_evaluable_no_solid_elements"
                if feasible is None
                else ("passed" if feasible else "failed")
            ),
        }

    smooth_relative_residual = (
        None
        if smooth_failure_aggregate is None
        else smooth_failure_aggregate / smooth_failure_limit - 1.0
    )
    smooth_feasible = (
        None
        if smooth_relative_residual is None
        else bool(smooth_relative_residual <= smooth_failure_relative_tolerance)
    )
    smooth_binary_mismatch = (
        None
        if smooth_feasible is None or binary_failure is None
        else bool(
            smooth_feasible
            and binary_failure > 1.0 + failure_feasibility_tolerance
        )
    )
    binary_feasible = (
        None
        if binary_failure is None
        else bool(binary_failure <= 1.0 + failure_feasibility_tolerance)
    )

    metrics = {
        **projected.summary("projected"),
        **binary.summary("binary"),
        **density_metrics(projected_density, projected_response, "projected"),
        **density_metrics(binary_density, binary_response, "binary"),
        **verification_metrics(projected_failure, "projected"),
        **verification_metrics(binary_failure, "binary"),
        "failure_binary_threshold": float(binary_threshold),
        "failure_feasibility_tolerance": failure_feasibility_tolerance,
        "smooth_failure_relative_tolerance": smooth_failure_relative_tolerance,
        "volume_units": "m^3",
        "failure_criterion": strength.criterion,
        "failure_strength_units": strength.units,
        "failure_strength": strength.as_dict(),
        "smooth_failure_aggregate": smooth_failure_aggregate,
        "smooth_failure_limit": smooth_failure_limit,
        "smooth_failure_relative_residual": smooth_relative_residual,
        "smooth_failure_feasible": smooth_feasible,
        "smooth_to_binary_failure_mismatch": smooth_binary_mismatch,
        "failure_verification_recommendations": (
            list(_BINARY_FAILURE_RECOMMENDATIONS)
            if smooth_binary_mismatch
            else []
        ),
        "stage10_internal_verification_passed": binary_feasible is True,
        "stage10_internal_verification_status": (
            "not_evaluable_no_solid_elements"
            if binary_feasible is None
            else ("passed" if binary_feasible else "failed_binary_strength")
        ),
        # Neither fixture singularity assessment nor ANSYS agreement can be
        # established by repeating the same internal voxel solve.
        "critical_region_artifact_validation_status": "external_review_not_run",
        "ansys_validation_status": "not_run",
    }
    if smooth_binary_mismatch:
        logger.warning(
            "Smooth failure aggregate G=%.6e at limit %.6e is feasible within "
            "relative tolerance (residual %.6e <= %.6e), but exact binary "
            "FI=%.6e exceeds one; see failure_verification_recommendations.",
            smooth_failure_aggregate,
            smooth_failure_limit,
            smooth_relative_residual,
            smooth_failure_relative_tolerance,
            binary_failure,
        )
    elif binary_feasible is False:
        logger.warning(
            "Exact binary verification failed: FI=%.6e exceeds one.",
            binary_failure,
        )
    saved_files: Dict[str, str] = {}
    if results_manager is not None:
        saved_files.update(save_failure_fields(results_manager, projected, "projected"))
        saved_files.update(save_failure_fields(results_manager, binary, "binary"))
        saved_files["failure_binary_density.npy"] = results_manager.save_result(
            binary_density,
            "failure_binary_density.npy",
        )
        metrics["failure_result_files"] = {
            filename: str(path) for filename, path in saved_files.items()
        }

    return FailureRepresentationResults(
        projected_response=projected_response,
        binary_response=binary_response,
        projected=projected,
        binary=binary,
        binary_density=binary_density,
        metrics=metrics,
        saved_files=saved_files,
    )
