"""Adjoint sensitivity of the relaxed, aggregated failure measure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pytopo3d.analysis.failure import (
    GaussFailureResult,
    evaluate_gauss_maximum_stress,
    maximum_stress_gradient,
)
from pytopo3d.analysis.failure_aggregation import (
    FailureAggregateResult,
    aggregate_gauss_failure,
    corrected_pnorm_gradient,
)
from pytopo3d.analysis.stress import (
    recover_gauss_stress,
    relax_gauss_stress,
    rotate_stress_to_material,
    stress_rotation_matrix_to_material,
)
from pytopo3d.utils.config_loader import MaterialStrength
from pytopo3d.utils.filter import apply_density_filter_chain_rule
from pytopo3d.utils.stiffness import h8_gauss_integration_data


@dataclass(frozen=True)
class FailurePartials:
    """Failure value and its partial derivatives at a fixed displacement."""

    aggregate_result: FailureAggregateResult
    gauss_failure: GaussFailureResult
    solid_stress_material_gauss: np.ndarray
    relaxed_stress_material_gauss: np.ndarray
    aggregate_failure_index_derivative: np.ndarray
    displacement_derivative: np.ndarray
    explicit_density_derivative: np.ndarray
    active_gauss_point: np.ndarray
    active_component: np.ndarray


@dataclass(frozen=True)
class FailureSensitivityResult:
    """Complete physical-density derivative from one adjoint solve."""

    partials: FailurePartials
    adjoint: np.ndarray
    adjoint_density_derivative: np.ndarray
    physical_density_derivative: np.ndarray
    adjoint_relative_residual: float
    adjoint_solve_count: int


@dataclass(frozen=True)
class AdjointResult:
    """Global adjoint vector and reduced-system solve diagnostics."""

    adjoint: np.ndarray
    relative_residual: float
    solve_count: int


def _density_vector(density: np.ndarray, number_of_elements: int) -> np.ndarray:
    density = np.asarray(density, dtype=float).ravel(order="F")
    if density.shape != (number_of_elements,):
        raise ValueError(
            f"density must contain {number_of_elements} element values, "
            f"got {density.size}"
        )
    if not np.all(np.isfinite(density)) or np.any(density < 0.0) or np.any(
        density > 1.0
    ):
        raise ValueError("density must contain only finite values in [0, 1]")
    return density


def _relaxation_derivative(
    density: np.ndarray,
    exponent: float,
    eligible: np.ndarray,
) -> np.ndarray:
    """Return ``q*rho**(q-1)`` under an explicit zero-density policy."""
    exponent = float(exponent)
    if not np.isfinite(exponent) or exponent < 0.0:
        raise ValueError(
            f"relaxation_exponent must be finite and nonnegative, got {exponent}"
        )
    derivative = np.zeros_like(density)
    if exponent == 0.0:
        return derivative
    if exponent < 1.0 and np.any(eligible & (density <= 0.0)):
        indices = np.flatnonzero(eligible & (density <= 0.0))
        preview = ", ".join(str(int(index)) for index in indices[:8])
        raise ValueError(
            "eligible physical densities must be strictly positive when "
            f"0 < relaxation_exponent < 1; zero at element(s) {preview}"
        )
    derivative[eligible] = exponent * np.power(
        density[eligible], exponent - 1.0
    )
    return derivative


def evaluate_failure_partials(
    displacement: np.ndarray,
    edof_matrix: np.ndarray,
    constitutive_matrix: np.ndarray,
    orientation_matrix: np.ndarray,
    density: np.ndarray,
    strength: MaterialStrength | Mapping[str, object],
    *,
    elem_size: float = 1.0,
    relaxation_exponent: float = 0.5,
    aggregate_exponent: float = 8.0,
    correction_factor: float = 1.0,
    element_weights: Optional[np.ndarray] = None,
    eligible_elements: Optional[np.ndarray] = None,
) -> FailurePartials:
    """Evaluate ``G`` plus ``partial G/partial u`` and explicit ``partial G/partial rho``.

    The element and Gauss-point maxima use a deterministic active-set
    derivative. Finite-difference checks must therefore stay away from ties.
    Fixed voids must be excluded through ``eligible_elements``; for ``q<1`` an
    eligible zero density is rejected rather than silently clipping only the
    derivative.
    """
    displacement = np.asarray(displacement, dtype=float)
    edof_matrix = np.asarray(edof_matrix)
    number_of_elements = edof_matrix.shape[0] if edof_matrix.ndim == 2 else 0
    density_vector = _density_vector(density, number_of_elements)

    solid_stress_global = recover_gauss_stress(
        displacement,
        edof_matrix,
        constitutive_matrix,
        elem_size=elem_size,
    )
    solid_stress_material = rotate_stress_to_material(
        solid_stress_global,
        orientation_matrix,
    )
    relaxed_stress_material = relax_gauss_stress(
        solid_stress_material,
        density_vector,
        exponent=relaxation_exponent,
    )
    gauss_failure = evaluate_gauss_maximum_stress(
        relaxed_stress_material,
        strength,
    )
    if element_weights is not None:
        element_weights = np.asarray(element_weights, dtype=float).ravel(order="F")
    if eligible_elements is not None:
        eligible_elements = np.asarray(eligible_elements, dtype=bool).ravel(order="F")
    aggregate_result = aggregate_gauss_failure(
        gauss_failure.failure_index_gauss,
        exponent=aggregate_exponent,
        correction_factor=correction_factor,
        element_weights=element_weights,
        eligible_elements=eligible_elements,
    )
    aggregate_failure_derivative = corrected_pnorm_gradient(aggregate_result)
    eligible = aggregate_result.eligible_weights > 0.0
    relaxation_derivative = _relaxation_derivative(
        density_vector,
        relaxation_exponent,
        eligible,
    )

    active_gauss = gauss_failure.critical_gauss_point_element
    element_indices = np.arange(number_of_elements)
    active_relaxed_stress = relaxed_stress_material[
        element_indices, active_gauss
    ]
    active_component = np.argmax(
        gauss_failure.failure_components_gauss[element_indices, active_gauss],
        axis=1,
    )
    active_stress_gradient = maximum_stress_gradient(
        active_relaxed_stress,
        strength,
    )

    b_matrices, _, _ = h8_gauss_integration_data(elem_size)
    global_stress_operators = np.einsum(
        "ij,gjk->gik",
        np.asarray(constitutive_matrix, dtype=float),
        b_matrices,
        optimize=True,
    )
    material_rotation = stress_rotation_matrix_to_material(orientation_matrix)
    material_stress_operators = np.einsum(
        "ij,gjk->gik",
        material_rotation,
        global_stress_operators,
        optimize=True,
    )
    active_operators = material_stress_operators[active_gauss]
    active_operator_gradient = np.einsum(
        "ei,eij->ej",
        active_stress_gradient,
        active_operators,
        optimize=True,
    )
    relaxation_scale = np.power(density_vector, float(relaxation_exponent))
    local_displacement_derivative = (
        aggregate_failure_derivative[:, None]
        * relaxation_scale[:, None]
        * active_operator_gradient
    )

    displacement_derivative = np.zeros_like(displacement, dtype=float)
    np.add.at(
        displacement_derivative,
        edof_matrix.astype(np.int64, copy=False).ravel() - 1,
        local_displacement_derivative.ravel(),
    )

    active_solid_stress = solid_stress_material[element_indices, active_gauss]
    active_failure_per_solid_scale = np.einsum(
        "ei,ei->e",
        active_stress_gradient,
        active_solid_stress,
        optimize=True,
    )
    explicit_density_derivative = (
        aggregate_failure_derivative
        * relaxation_derivative
        * active_failure_per_solid_scale
    )

    return FailurePartials(
        aggregate_result=aggregate_result,
        gauss_failure=gauss_failure,
        solid_stress_material_gauss=solid_stress_material,
        relaxed_stress_material_gauss=relaxed_stress_material,
        aggregate_failure_index_derivative=aggregate_failure_derivative,
        displacement_derivative=displacement_derivative,
        explicit_density_derivative=explicit_density_derivative,
        active_gauss_point=active_gauss.copy(),
        active_component=active_component,
    )


def solve_failure_adjoint(
    reduced_stiffness_matrix,
    displacement_derivative: np.ndarray,
    free_dofs: np.ndarray,
    *,
    linear_solver: Optional[Callable[[object, np.ndarray], np.ndarray]] = None,
) -> AdjointResult:
    """Solve one reduced adjoint system and scatter it to global DOFs.

    ``reduced_stiffness_matrix`` must be the same symmetric ``Kff`` used by the
    primal solve. Keeping this function separate lets optimizer integrations
    reuse an existing factorization or GPU sparse system.
    """
    displacement_derivative = np.asarray(displacement_derivative, dtype=float)
    if displacement_derivative.ndim != 1 or not np.all(
        np.isfinite(displacement_derivative)
    ):
        raise ValueError("displacement_derivative must be a finite vector")
    ndof = displacement_derivative.size
    free_dofs = np.asarray(free_dofs)
    if free_dofs.ndim != 1 or not np.issubdtype(free_dofs.dtype, np.integer):
        raise ValueError("free_dofs must be a one-dimensional integer array")
    if free_dofs.size == 0:
        raise ValueError("free_dofs must contain at least one degree of freedom")
    if np.min(free_dofs) < 0 or np.max(free_dofs) >= ndof:
        raise ValueError("free_dofs contains an index outside displacement_derivative")
    if np.unique(free_dofs).size != free_dofs.size:
        raise ValueError("free_dofs must not contain duplicates")
    expected_shape = (free_dofs.size, free_dofs.size)
    if getattr(reduced_stiffness_matrix, "shape", None) != expected_shape:
        raise ValueError(
            f"reduced_stiffness_matrix must have shape {expected_shape}, got "
            f"{getattr(reduced_stiffness_matrix, 'shape', None)}"
        )
    if sp.issparse(reduced_stiffness_matrix):
        if not np.all(np.isfinite(reduced_stiffness_matrix.data)):
            raise ValueError("reduced_stiffness_matrix must contain only finite values")
    else:
        reduced_stiffness_matrix = np.asarray(
            reduced_stiffness_matrix, dtype=float
        )
        if not np.all(np.isfinite(reduced_stiffness_matrix)):
            raise ValueError("reduced_stiffness_matrix must contain only finite values")

    right_hand_side = displacement_derivative[free_dofs]
    right_hand_side_norm = float(np.linalg.norm(right_hand_side))
    if right_hand_side_norm == 0.0:
        return AdjointResult(
            adjoint=np.zeros(ndof, dtype=float),
            relative_residual=0.0,
            solve_count=0,
        )
    if linear_solver is None:
        if sp.issparse(reduced_stiffness_matrix):
            adjoint_free = spsolve(reduced_stiffness_matrix, right_hand_side)
        else:
            adjoint_free = np.linalg.solve(
                reduced_stiffness_matrix, right_hand_side
            )
    else:
        adjoint_free = linear_solver(reduced_stiffness_matrix, right_hand_side)
    adjoint_free = np.asarray(adjoint_free, dtype=float)
    if adjoint_free.shape != right_hand_side.shape or not np.all(
        np.isfinite(adjoint_free)
    ):
        raise ValueError("adjoint solver returned a nonfinite or incorrectly shaped vector")
    residual = reduced_stiffness_matrix @ adjoint_free - right_hand_side
    relative_residual = float(np.linalg.norm(residual) / right_hand_side_norm)
    if not np.isfinite(relative_residual):
        raise ValueError("adjoint solve produced a nonfinite residual")

    adjoint = np.zeros(ndof, dtype=float)
    adjoint[free_dofs] = adjoint_free
    return AdjointResult(
        adjoint=adjoint,
        relative_residual=relative_residual,
        solve_count=1,
    )


def combine_failure_density_gradient(
    explicit_density_derivative: np.ndarray,
    adjoint: np.ndarray,
    displacement: np.ndarray,
    edof_matrix: np.ndarray,
    element_stiffness: np.ndarray,
    density: np.ndarray,
    *,
    simp_penal: float,
    stiffness_solid_scale: float = 1.0,
    stiffness_void_scale: float = 1.0e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine explicit relaxation and implicit stiffness derivatives."""
    displacement = np.asarray(displacement, dtype=float)
    adjoint = np.asarray(adjoint, dtype=float)
    if displacement.ndim != 1 or adjoint.shape != displacement.shape:
        raise ValueError("displacement and adjoint must be equally shaped vectors")
    if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(adjoint)):
        raise ValueError("displacement and adjoint must contain only finite values")
    edof_matrix = np.asarray(edof_matrix)
    if edof_matrix.ndim != 2 or edof_matrix.shape[1] != 24:
        raise ValueError("edof_matrix must have shape (number_of_elements, 24)")
    if not np.issubdtype(edof_matrix.dtype, np.integer):
        if not np.all(np.equal(edof_matrix, np.floor(edof_matrix))):
            raise ValueError("edof_matrix must contain integer 1-based DOF indices")
    density_vector = _density_vector(density, edof_matrix.shape[0])
    explicit_density_derivative = np.asarray(
        explicit_density_derivative, dtype=float
    ).ravel(order="F")
    if explicit_density_derivative.shape != density_vector.shape or not np.all(
        np.isfinite(explicit_density_derivative)
    ):
        raise ValueError(
            "explicit_density_derivative must be a finite element vector"
        )

    element_stiffness = np.asarray(element_stiffness, dtype=float)
    if element_stiffness.shape != (24, 24) or not np.all(
        np.isfinite(element_stiffness)
    ):
        raise ValueError("element_stiffness must be a finite (24, 24) matrix")
    simp_penal = float(simp_penal)
    if not np.isfinite(simp_penal) or simp_penal <= 0.0:
        raise ValueError(f"simp_penal must be finite and positive, got {simp_penal}")
    if simp_penal < 1.0 and np.any(density_vector <= 0.0):
        raise ValueError("physical densities must be positive when simp_penal < 1")
    stiffness_solid_scale = float(stiffness_solid_scale)
    stiffness_void_scale = float(stiffness_void_scale)
    if (
        not np.isfinite(stiffness_solid_scale)
        or not np.isfinite(stiffness_void_scale)
        or stiffness_solid_scale <= stiffness_void_scale
        or stiffness_void_scale < 0.0
    ):
        raise ValueError("stiffness scales must be finite with solid > void >= 0")

    integer_edof = edof_matrix.astype(np.int64, copy=False)
    if integer_edof.size and (
        np.min(integer_edof) < 1 or np.max(integer_edof) > displacement.size
    ):
        raise ValueError("edof_matrix contains a DOF outside displacement")
    element_displacement = displacement[integer_edof - 1]
    element_adjoint = adjoint[integer_edof - 1]
    adjoint_energy = np.einsum(
        "ei,ij,ej->e",
        element_adjoint,
        element_stiffness,
        element_displacement,
        optimize=True,
    )
    stiffness_derivative = (
        simp_penal
        * (stiffness_solid_scale - stiffness_void_scale)
        * np.power(density_vector, simp_penal - 1.0)
    )
    adjoint_density_derivative = -stiffness_derivative * adjoint_energy
    return (
        adjoint_density_derivative,
        explicit_density_derivative + adjoint_density_derivative,
    )


def evaluate_failure_sensitivity(
    displacement: np.ndarray,
    stiffness_matrix,
    free_dofs: np.ndarray,
    edof_matrix: np.ndarray,
    element_stiffness: np.ndarray,
    constitutive_matrix: np.ndarray,
    orientation_matrix: np.ndarray,
    density: np.ndarray,
    strength: MaterialStrength | Mapping[str, object],
    *,
    simp_penal: float,
    stiffness_solid_scale: float = 1.0,
    stiffness_void_scale: float = 1.0e-9,
    elem_size: float = 1.0,
    relaxation_exponent: float = 0.5,
    aggregate_exponent: float = 8.0,
    correction_factor: float = 1.0,
    element_weights: Optional[np.ndarray] = None,
    eligible_elements: Optional[np.ndarray] = None,
    linear_solver: Optional[Callable[[object, np.ndarray], np.ndarray]] = None,
) -> FailureSensitivityResult:
    """Return the complete ``dG/d(rho_physical)`` using one adjoint solve."""
    partials = evaluate_failure_partials(
        displacement,
        edof_matrix,
        constitutive_matrix,
        orientation_matrix,
        density,
        strength,
        elem_size=elem_size,
        relaxation_exponent=relaxation_exponent,
        aggregate_exponent=aggregate_exponent,
        correction_factor=correction_factor,
        element_weights=element_weights,
        eligible_elements=eligible_elements,
    )
    displacement = np.asarray(displacement, dtype=float)
    ndof = displacement.size
    free_dofs = np.asarray(free_dofs)
    if free_dofs.ndim != 1 or not np.issubdtype(free_dofs.dtype, np.integer):
        raise ValueError("free_dofs must be a one-dimensional integer array")
    if free_dofs.size == 0 or np.min(free_dofs) < 0 or np.max(free_dofs) >= ndof:
        raise ValueError("free_dofs must contain valid displacement indices")
    if np.unique(free_dofs).size != free_dofs.size:
        raise ValueError("free_dofs must not contain duplicates")
    if getattr(stiffness_matrix, "shape", None) != (ndof, ndof):
        raise ValueError(
            f"stiffness_matrix must have shape ({ndof}, {ndof}), got "
            f"{getattr(stiffness_matrix, 'shape', None)}"
        )
    if sp.issparse(stiffness_matrix):
        if not np.all(np.isfinite(stiffness_matrix.data)):
            raise ValueError("stiffness_matrix must contain only finite values")
        adjoint_matrix = stiffness_matrix[free_dofs, :][:, free_dofs].tocsr()
    else:
        stiffness_matrix = np.asarray(stiffness_matrix, dtype=float)
        if not np.all(np.isfinite(stiffness_matrix)):
            raise ValueError("stiffness_matrix must contain only finite values")
        adjoint_matrix = stiffness_matrix[np.ix_(free_dofs, free_dofs)]

    adjoint_result = solve_failure_adjoint(
        adjoint_matrix,
        partials.displacement_derivative,
        free_dofs,
        linear_solver=linear_solver,
    )
    adjoint_density_derivative, physical_density_derivative = (
        combine_failure_density_gradient(
            partials.explicit_density_derivative,
            adjoint_result.adjoint,
            displacement,
            edof_matrix,
            element_stiffness,
            density,
            simp_penal=simp_penal,
            stiffness_solid_scale=stiffness_solid_scale,
            stiffness_void_scale=stiffness_void_scale,
        )
    )

    return FailureSensitivityResult(
        partials=partials,
        adjoint=adjoint_result.adjoint,
        adjoint_density_derivative=adjoint_density_derivative,
        physical_density_derivative=physical_density_derivative,
        adjoint_relative_residual=adjoint_result.relative_residual,
        adjoint_solve_count=adjoint_result.solve_count,
    )


def map_failure_gradient_to_design(
    physical_density_derivative: np.ndarray,
    projection_derivative: np.ndarray,
    filter_matrix,
    filter_row_sums: np.ndarray,
    *,
    fixed_design_elements: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply projection/filter transpose and mask fixed design variables."""
    physical_density_derivative = np.asarray(physical_density_derivative, dtype=float)
    projection_derivative = np.asarray(projection_derivative, dtype=float)
    if physical_density_derivative.shape != projection_derivative.shape:
        raise ValueError(
            "physical_density_derivative and projection_derivative must have "
            f"the same shape, got {physical_density_derivative.shape} and "
            f"{projection_derivative.shape}"
        )
    if not np.all(np.isfinite(physical_density_derivative)) or not np.all(
        np.isfinite(projection_derivative)
    ):
        raise ValueError("density and projection derivatives must be finite")
    filter_row_sums = np.asarray(filter_row_sums, dtype=float)
    expected_row_sums_shape = (physical_density_derivative.size,)
    if filter_row_sums.shape != expected_row_sums_shape:
        raise ValueError(
            f"filter_row_sums has shape {filter_row_sums.shape}, expected "
            f"{expected_row_sums_shape}"
        )
    if not np.all(np.isfinite(filter_row_sums)) or np.any(filter_row_sums <= 0.0):
        raise ValueError("filter_row_sums must be finite and strictly positive")
    design_derivative = apply_density_filter_chain_rule(
        physical_density_derivative,
        projection_derivative,
        filter_matrix,
        filter_row_sums,
    )
    if fixed_design_elements is not None:
        fixed_design_elements = np.asarray(fixed_design_elements, dtype=bool)
        if fixed_design_elements.shape != design_derivative.shape:
            raise ValueError(
                f"fixed_design_elements has shape {fixed_design_elements.shape}, "
                f"expected {design_derivative.shape}"
            )
        design_derivative = np.where(fixed_design_elements, 0.0, design_derivative)
    if not np.all(np.isfinite(design_derivative)):
        raise ValueError("filter chain produced a nonfinite design derivative")
    return design_derivative
