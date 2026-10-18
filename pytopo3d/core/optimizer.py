"""
Main optimizer for 3-D topology optimization – fixed CSR scatter version
(24 April 2025).

Changes relative to the previous refactor
-----------------------------------------
* Keep the 576 × nele COO coordinates (with duplicates) **and** build a
  mapping `dup2uniq` → CSR.data so we can scatter-add each iteration.
* Still reuse the CSR structure – no allocation inside the loop.
* Compatible with the CG + Jacobi GPU solver provided in solver.py.
"""

from __future__ import annotations

import time
from typing import Any, Dict, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from pytopo3d.analysis.failure import (
    critical_failure_location,
    predicted_failure_load,
)
from pytopo3d.analysis.failure_aggregation import calibrate_pnorm_correction
from pytopo3d.analysis.failure_sensitivity import (
    combine_failure_density_gradient,
    evaluate_failure_partials,
    solve_failure_adjoint,
)
from pytopo3d.analysis.stress import validate_orientation_matrix
from pytopo3d.core.compliance import element_compliance
from pytopo3d.utils.assembly import build_edof, build_force_field, build_force_vector, build_support_mask, build_supports
from pytopo3d.utils.filter import (
    HAS_CUPY,
    apply_density_filter_chain_rule,
    build_filter,
    build_physical_density,
)
from pytopo3d.utils.logger import get_logger
from pytopo3d.utils.mma_update import mma_update
from pytopo3d.utils.oc_update import optimality_criteria_update_projected
from pytopo3d.utils.solver import get_solver
from pytopo3d.utils.stiffness import lk_H8, make_C_matrix
from pytopo3d.visualization.display import display_3D

logger = get_logger(__name__)

if HAS_CUPY:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    logger.info("CuPy is available for GPU acceleration")


# ──────────────────────────────────────────────────────────────────────────
def _make_scatter_map(i_full, j_full, ndof):
    """Memory– and speed-optimised scatter map builder."""
    linear = i_full.astype(np.int64) * ndof + j_full        # 1-D key
    uniq_lin, dup2uniq = np.unique(linear, return_inverse=True)
    i_uniq = uniq_lin // ndof
    j_uniq = uniq_lin % ndof
    return i_uniq, j_uniq, dup2uniq


def _mma_nonlinear_kkt_components(
    x_free: np.ndarray,
    normalized_objective_gradient: np.ndarray,
    constraint_values: np.ndarray,
    constraint_gradients: np.ndarray,
    multipliers: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> Dict[str, float]:
    """Measure KKT residuals for the current nonlinear topology problem."""
    x_free = np.asarray(x_free, dtype=float)
    objective_gradient = np.asarray(normalized_objective_gradient, dtype=float)
    constraint_values = np.asarray(constraint_values, dtype=float)
    constraint_gradients = np.asarray(constraint_gradients, dtype=float)
    multipliers = np.asarray(multipliers, dtype=float)
    lagrange_constraint_gradient = constraint_gradients.T @ multipliers
    lagrangian_gradient = objective_gradient + lagrange_constraint_gradient
    gradient_scale = max(
        float(np.linalg.norm(objective_gradient, ord=np.inf)),
        float(np.linalg.norm(lagrange_constraint_gradient, ord=np.inf)),
        1.0e-12,
    )
    projected_step = x_free - np.clip(
        x_free - lagrangian_gradient / gradient_scale,
        lower_bound,
        upper_bound,
    )
    stationarity = float(np.linalg.norm(projected_step, ord=np.inf))
    primal = max(0.0, float(np.max(constraint_values)))
    dual = max(0.0, float(np.max(-multipliers)))
    complementarity = float(
        np.linalg.norm(multipliers * constraint_values, ord=np.inf)
        / (1.0 + np.linalg.norm(multipliers, ord=np.inf))
    )
    return {
        "stationarity": stationarity,
        "primal_violation": primal,
        "dual_violation": dual,
        "complementarity": complementarity,
        "residual": max(stationarity, primal, dual, complementarity),
    }


def _expand_continuation_schedule(
    values,
    fallback: float,
    number_of_stages: int,
    *,
    name: str,
    minimum_exclusive: float,
    direction: str,
) -> tuple[float, ...]:
    """Validate a scalar/broadcast continuation schedule."""
    if values is None:
        schedule = (float(fallback),) * number_of_stages
    else:
        try:
            schedule = tuple(float(value) for value in values)
        except TypeError:
            schedule = (float(values),)
        if len(schedule) == 1:
            schedule = schedule * number_of_stages
        elif len(schedule) != number_of_stages:
            raise ValueError(
                f"{name} must contain one value or {number_of_stages} values "
                f"to match beta_schedule; got {len(schedule)}"
            )
    if any(
        not np.isfinite(value) or value <= minimum_exclusive
        for value in schedule
    ):
        raise ValueError(
            f"all {name} values must be finite and > {minimum_exclusive:g}; "
            f"got {schedule}"
        )
    differences = np.diff(schedule)
    if direction == "nonincreasing" and np.any(differences > 1.0e-12):
        raise ValueError(f"{name} must be nonincreasing; got {schedule}")
    if direction == "nondecreasing" and np.any(differences < -1.0e-12):
        raise ValueError(f"{name} must be nondecreasing; got {schedule}")
    return schedule


def evaluate_fixed_geometry_metrics(
    xPhys: np.ndarray,
    penal: float,
    material_params: Optional[Tuple[float, ...]] = None,
    elem_size: float = 0.01,
    force_field: Optional[np.ndarray] = None,
    support_mask: Optional[np.ndarray] = None,
    obstacle_mask: Optional[np.ndarray] = None,
    protected_zone_mask: Optional[np.ndarray] = None,
    use_gpu: bool = False,
    return_displacement: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate fixed-geometry response metrics under given material/BC settings.

    This performs a single FE solve with no OC update and returns:
    - compliance: objective definition used in optimization
    - u{x,y,z}_avg_load_patch: average displacement on loaded-node DOFs
    - k_avg_{x,y,z}: directional equivalent stiffness F_dir / abs(u_dir)
    - k_avg: equivalent stiffness on dominant loading direction (legacy)
    - F_total and F_total_{x,y,z}: absolute load magnitudes

    When ``return_displacement`` is true, the result additionally contains the
    NumPy displacement vector and existing 1-based element DOF matrix needed by
    Gauss-point recovery. The default response contract remains unchanged.
    """
    nely, nelx, nelz = xPhys.shape
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

    obstacle_mask = (
        np.zeros((nely, nelx, nelz), dtype=bool)
        if obstacle_mask is None
        else obstacle_mask
    )
    protected_zone_mask = (
        np.zeros((nely, nelx, nelz), dtype=bool)
        if protected_zone_mask is None
        else protected_zone_mask
    )

    # Ensure fixed regions remain valid for evaluation.
    x_eval = np.array(xPhys, dtype=float, copy=True)
    x_eval[protected_zone_mask] = 1.0
    x_eval[obstacle_mask] = 0.0

    E0, Emin = 1.0, 1e-9

    F = build_force_vector(nelx, nely, nelz, ndof, force_field)
    loaded_dofs = np.where(np.abs(F) > 0.0)[0]
    loaded_nodes = np.unique(loaded_dofs // 3)
    loaded_x_dofs = 3 * loaded_nodes
    loaded_y_dofs = 3 * loaded_nodes + 1
    loaded_z_dofs = 3 * loaded_nodes + 2
    F_total_x = float(np.sum(np.abs(F[loaded_x_dofs]))) if loaded_x_dofs.size else 0.0
    F_total_y = float(np.sum(np.abs(F[loaded_y_dofs]))) if loaded_y_dofs.size else 0.0
    F_total_z = float(np.sum(np.abs(F[loaded_z_dofs]))) if loaded_z_dofs.size else 0.0
    F_total = float(np.sum(np.abs(F[loaded_dofs]))) if loaded_dofs.size else 0.0
    freedofs0, _ = build_supports(nelx, nely, nelz, ndof, support_mask)

    if material_params is None:
        KE = lk_H8(elem_size=elem_size)
    else:
        KE = lk_H8(*material_params, elem_size=elem_size)

    edofMat, iK, jK = build_edof(nelx, nely, nelz)
    iK0, jK0 = iK - 1, jK - 1

    solver_func, _ = get_solver(use_gpu)

    i_unique, j_unique, dup2uniq = _make_scatter_map(iK0, jK0, ndof)

    if HAS_CUPY and use_gpu:
        KE_gpu = cp.asarray(KE)
        stiff_gpu = Emin + (cp.asarray(x_eval).ravel(order="F") ** penal) * (E0 - Emin)
        elem_vals_gpu = cp.kron(stiff_gpu, KE_gpu.ravel())

        K_gpu = cusp.csr_matrix(
            (cp.zeros(len(i_unique)), (cp.asarray(i_unique), cp.asarray(j_unique))),
            shape=(ndof, ndof),
        )
        cp.add.at(K_gpu.data, cp.asarray(dup2uniq), elem_vals_gpu)

        F_gpu = cp.asarray(F)
        freedofs0_gpu = cp.asarray(freedofs0)

        Kff_gpu = K_gpu[freedofs0_gpu, :][:, freedofs0_gpu]
        Uf_gpu = solver_func(Kff_gpu, F_gpu[freedofs0_gpu])
        U_gpu = cp.zeros(ndof)
        U_gpu[freedofs0_gpu] = Uf_gpu

        ce_flat_gpu = element_compliance(U_gpu, cp.asarray(edofMat), KE_gpu)
        ce_gpu = ce_flat_gpu.reshape(nely, nelx, nelz, order="F")
        c_gpu = cp.sum((Emin + cp.asarray(x_eval) ** penal * (E0 - Emin)) * ce_gpu)
        compliance = float(c_gpu.item())

        ux_avg_load_patch: Optional[float] = None
        uy_avg_load_patch: Optional[float] = None
        uz_avg_load_patch: Optional[float] = None
        if loaded_x_dofs.size:
            loaded_x_dofs_gpu = cp.asarray(loaded_x_dofs)
            ux_avg_load_patch = float(cp.mean(U_gpu[loaded_x_dofs_gpu]).item())
        if loaded_y_dofs.size:
            loaded_y_dofs_gpu = cp.asarray(loaded_y_dofs)
            uy_avg_load_patch = float(cp.mean(U_gpu[loaded_y_dofs_gpu]).item())
        if loaded_z_dofs.size:
            loaded_z_dofs_gpu = cp.asarray(loaded_z_dofs)
            uz_avg_load_patch = float(cp.mean(U_gpu[loaded_z_dofs_gpu]).item())

        k_avg_x: Optional[float] = None
        if ux_avg_load_patch is not None and ux_avg_load_patch != 0.0 and F_total_x > 0.0:
            k_avg_x = float(F_total_x / abs(ux_avg_load_patch))

        k_avg_y: Optional[float] = None
        if uy_avg_load_patch is not None and uy_avg_load_patch != 0.0 and F_total > 0.0:
            k_avg_y = float(F_total_y / abs(uy_avg_load_patch)) if F_total_y > 0.0 else None

        k_avg_z: Optional[float] = None
        if uz_avg_load_patch is not None and uz_avg_load_patch != 0.0 and F_total_z > 0.0:
            k_avg_z = float(F_total_z / abs(uz_avg_load_patch))

        dominant_dir = max(
            (("x", F_total_x), ("y", F_total_y), ("z", F_total_z)),
            key=lambda item: item[1],
        )[0]
        k_avg: Optional[float] = {"x": k_avg_x, "y": k_avg_y, "z": k_avg_z}[dominant_dir]

        response: Dict[str, Any] = {
            "compliance": compliance,
            "ux_avg_load_patch": ux_avg_load_patch,
            "uy_avg_load_patch": uy_avg_load_patch,
            "uz_avg_load_patch": uz_avg_load_patch,
            "k_avg_x": k_avg_x,
            "k_avg_y": k_avg_y,
            "k_avg_z": k_avg_z,
            "k_avg": k_avg,
            "F_total": F_total,
            "F_total_x": F_total_x,
            "F_total_y": F_total_y,
            "F_total_z": F_total_z,
            "loaded_x_dof_count": int(loaded_x_dofs.size),
            "loaded_y_dof_count": int(loaded_y_dofs.size),
            "loaded_z_dof_count": int(loaded_z_dofs.size),
        }
        if return_displacement:
            response["displacement"] = cp.asnumpy(U_gpu)
            response["edof_matrix"] = edofMat.copy()
        return response

    stiff = Emin + (x_eval.ravel(order="F") ** penal) * (E0 - Emin)
    elem_vals = np.kron(stiff, KE.ravel())

    K = sp.csr_matrix((np.zeros(len(i_unique)), (i_unique, j_unique)), shape=(ndof, ndof))
    np.add.at(K.data, dup2uniq, elem_vals)

    Kff = K[freedofs0, :][:, freedofs0]
    Uf = solver_func(Kff, F[freedofs0])
    U = np.zeros(ndof)
    U[freedofs0] = Uf

    ce_flat = element_compliance(U, edofMat, KE)
    ce = ce_flat.reshape(nely, nelx, nelz, order="F")
    c = ((Emin + x_eval ** penal * (E0 - Emin)) * ce).sum()

    ux_avg_load_patch: Optional[float] = None
    uy_avg_load_patch: Optional[float] = None
    uz_avg_load_patch: Optional[float] = None
    if loaded_x_dofs.size:
        ux_avg_load_patch = float(np.mean(U[loaded_x_dofs]))
    if loaded_y_dofs.size:
        uy_avg_load_patch = float(np.mean(U[loaded_y_dofs]))
    if loaded_z_dofs.size:
        uz_avg_load_patch = float(np.mean(U[loaded_z_dofs]))

    k_avg_x: Optional[float] = None
    if ux_avg_load_patch is not None and ux_avg_load_patch != 0.0 and F_total_x > 0.0:
        k_avg_x = float(F_total_x / abs(ux_avg_load_patch))

    k_avg_y: Optional[float] = None
    if uy_avg_load_patch is not None and uy_avg_load_patch != 0.0 and F_total_y > 0.0:
        k_avg_y = float(F_total_y / abs(uy_avg_load_patch))

    k_avg_z: Optional[float] = None
    if uz_avg_load_patch is not None and uz_avg_load_patch != 0.0 and F_total_z > 0.0:
        k_avg_z = float(F_total_z / abs(uz_avg_load_patch))

    dominant_dir = max(
        (("x", F_total_x), ("y", F_total_y), ("z", F_total_z)),
        key=lambda item: item[1],
    )[0]
    k_avg: Optional[float] = {"x": k_avg_x, "y": k_avg_y, "z": k_avg_z}[dominant_dir]

    response = {
        "compliance": float(c),
        "ux_avg_load_patch": ux_avg_load_patch,
        "uy_avg_load_patch": uy_avg_load_patch,
        "uz_avg_load_patch": uz_avg_load_patch,
        "k_avg_x": k_avg_x,
        "k_avg_y": k_avg_y,
        "k_avg_z": k_avg_z,
        "k_avg": k_avg,
        "F_total": F_total,
        "F_total_x": F_total_x,
        "F_total_y": F_total_y,
        "F_total_z": F_total_z,
        "loaded_x_dof_count": int(loaded_x_dofs.size),
        "loaded_y_dof_count": int(loaded_y_dofs.size),
        "loaded_z_dof_count": int(loaded_z_dofs.size),
    }
    if return_displacement:
        response["displacement"] = U.copy()
        response["edof_matrix"] = edofMat.copy()
    return response


def evaluate_fixed_geometry_compliance(
    xPhys: np.ndarray,
    penal: float,
    material_params: Optional[Tuple[float, ...]] = None,
    elem_size: float = 0.01,
    force_field: Optional[np.ndarray] = None,
    support_mask: Optional[np.ndarray] = None,
    obstacle_mask: Optional[np.ndarray] = None,
    protected_zone_mask: Optional[np.ndarray] = None,
    use_gpu: bool = False,
) -> float:
    """Backward-compatible wrapper returning only fixed-geometry compliance."""
    return float(
        evaluate_fixed_geometry_metrics(
            xPhys=xPhys,
            penal=penal,
            material_params=material_params,
            elem_size=elem_size,
            force_field=force_field,
            support_mask=support_mask,
            obstacle_mask=obstacle_mask,
            protected_zone_mask=protected_zone_mask,
            use_gpu=use_gpu,
        )["compliance"]
    )


# ──────────────────────────────────────────────────────────────────────────
def top3d(
    nelx: int,
    nely: int,
    nelz: int,
    volfrac: float,
    penal: float,
    rmin: float,
    disp_thres: float,
    material_params: Optional[Sequence[float]] = None,
    elem_size: float = 0.01,
    obstacle_mask: Optional[np.ndarray] = None,
    force_field: Optional[np.ndarray] = None,
    support_mask: Optional[np.ndarray] = None,
    tolx: float = 0.01,
    maxloop: int = 2000,
    save_history: bool = False,
    history_frequency: int = 10,
    use_gpu: bool = False,
    protected_zone_mask: Optional[np.ndarray] = None,
    beta_schedule: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    projection_eta: float = 0.5,
    move: float = 0.2,
    diagnostics_out: Optional[MutableMapping[str, Any]] = None,
    optimization_mode: str = "compliance",
    optimizer: str = "oc",
    material_strength: Optional[Any] = None,
    material_orientation: Optional[np.ndarray] = None,
    failure_limit: float = 1.0,
    failure_aggregate_exponent: float = 8.0,
    failure_relaxation_exponent: float = 0.5,
    mma_move: float = 0.05,
    mma_min_density: float = 1.0e-3,
    failure_limit_schedule: Optional[Sequence[float]] = None,
    failure_aggregate_exponent_schedule: Optional[Sequence[float]] = None,
    initial_design: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]], Optional[np.ndarray], float]:
    """Run density-filtered SIMP optimization with Heaviside continuation.

    maxloop is the maximum number of nonlinear evaluations for each beta
    stage. MMA reserves the final budgeted evaluation for reporting, so every
    returned constrained design has current objective and constraint values.
    The returned array keeps the legacy xPhys contract but now contains the
    projected physical density used consistently by FEA and volume control.
    """
    if min(nelx, nely, nelz) < 1:
        raise ValueError("nelx, nely, and nelz must all be positive")
    expected_shape = (nely, nelx, nelz)
    if not 0.0 < volfrac <= 1.0:
        raise ValueError(f"volfrac must be in (0, 1], got {volfrac}")
    if not np.isfinite(penal) or penal <= 0:
        raise ValueError(f"penal must be positive, got {penal}")
    if not np.isfinite(maxloop) or maxloop < 1:
        raise ValueError(f"maxloop must be at least 1, got {maxloop}")
    if not np.isfinite(tolx) or tolx <= 0:
        raise ValueError(f"tolx must be positive, got {tolx}")
    if not np.isfinite(history_frequency) or history_frequency < 1:
        raise ValueError(
            f"history_frequency must be at least 1, got {history_frequency}"
        )
    if not np.isfinite(move) or move <= 0:
        raise ValueError(f"move must be positive, got {move}")
    optimization_mode = str(optimization_mode).strip().lower()
    optimizer = str(optimizer).strip().lower()
    valid_modes = {"compliance", "compliance_failure_constrained"}
    if optimization_mode not in valid_modes:
        raise ValueError(
            f"optimization_mode must be one of {sorted(valid_modes)}, "
            f"got {optimization_mode!r}"
        )
    if optimizer not in {"oc", "mma"}:
        raise ValueError(f"optimizer must be 'oc' or 'mma', got {optimizer!r}")
    failure_constrained = optimization_mode == "compliance_failure_constrained"
    if failure_constrained and optimizer != "mma":
        raise ValueError("compliance_failure_constrained mode requires optimizer='mma'")
    if failure_constrained and material_strength is None:
        raise ValueError(
            "compliance_failure_constrained mode requires validated material strength"
        )
    if failure_constrained and material_params is None:
        raise ValueError(
            "compliance_failure_constrained mode requires material_params"
        )
    failure_limit = float(failure_limit)
    if not np.isfinite(failure_limit) or failure_limit <= 0.0:
        raise ValueError(f"failure_limit must be finite and positive, got {failure_limit}")
    failure_aggregate_exponent = float(failure_aggregate_exponent)
    if (
        not np.isfinite(failure_aggregate_exponent)
        or failure_aggregate_exponent <= 1.0
    ):
        raise ValueError(
            "failure_aggregate_exponent must be finite and > 1, got "
            f"{failure_aggregate_exponent}"
        )
    failure_relaxation_exponent = float(failure_relaxation_exponent)
    if (
        not np.isfinite(failure_relaxation_exponent)
        or failure_relaxation_exponent < 0.0
    ):
        raise ValueError(
            "failure_relaxation_exponent must be finite and nonnegative, got "
            f"{failure_relaxation_exponent}"
        )
    mma_move = float(mma_move)
    if not np.isfinite(mma_move) or mma_move <= 0.0:
        raise ValueError(f"mma_move must be finite and positive, got {mma_move}")
    mma_min_density = float(mma_min_density)
    if not np.isfinite(mma_min_density) or not 0.0 <= mma_min_density < 1.0:
        raise ValueError(
            f"mma_min_density must be finite and in [0, 1), got {mma_min_density}"
        )
    if (
        failure_constrained
        and 0.0 < failure_relaxation_exponent < 1.0
        and mma_min_density <= 0.0
    ):
        raise ValueError(
            "failure-constrained MMA with 0 < q < 1 requires a positive "
            "mma_min_density"
        )
    if not 0.0 < projection_eta < 1.0:
        raise ValueError(
            f"projection_eta must be between 0 and 1, got {projection_eta}"
        )

    beta_schedule = tuple(float(beta) for beta in beta_schedule)
    if not beta_schedule:
        raise ValueError("beta_schedule must contain at least one value")
    if any(not np.isfinite(beta) or beta <= 0.0 for beta in beta_schedule):
        raise ValueError(
            f"all beta_schedule values must be finite and positive, got {beta_schedule}"
        )
    if np.any(np.diff(beta_schedule) < -1.0e-12):
        raise ValueError(f"beta_schedule must be nondecreasing, got {beta_schedule}")
    if not failure_constrained and (
        failure_limit_schedule is not None
        or failure_aggregate_exponent_schedule is not None
    ):
        raise ValueError(
            "failure continuation schedules require "
            "optimization_mode='compliance_failure_constrained'"
        )
    if failure_constrained:
        stage_failure_limits = _expand_continuation_schedule(
            failure_limit_schedule,
            failure_limit,
            len(beta_schedule),
            name="failure_limit_schedule",
            minimum_exclusive=0.0,
            direction="nonincreasing",
        )
        stage_failure_exponents = _expand_continuation_schedule(
            failure_aggregate_exponent_schedule,
            failure_aggregate_exponent,
            len(beta_schedule),
            name="failure_aggregate_exponent_schedule",
            minimum_exclusive=1.0,
            direction="nondecreasing",
        )
    else:
        stage_failure_limits = (failure_limit,) * len(beta_schedule)
        stage_failure_exponents = (failure_aggregate_exponent,) * len(
            beta_schedule
        )

    obstacle_mask = (
        np.zeros(expected_shape, dtype=bool)
        if obstacle_mask is None
        else np.asarray(obstacle_mask, dtype=bool).copy()
    )
    protected_zone_mask = (
        np.zeros(expected_shape, dtype=bool)
        if protected_zone_mask is None
        else np.asarray(protected_zone_mask, dtype=bool).copy()
    )
    if obstacle_mask.shape != expected_shape:
        raise ValueError(
            f"obstacle_mask has shape {obstacle_mask.shape}, expected {expected_shape}"
        )
    if protected_zone_mask.shape != expected_shape:
        raise ValueError(
            "protected_zone_mask has shape "
            f"{protected_zone_mask.shape}, expected {expected_shape}"
        )

    # Load/support attachment elements were historically made solid only when
    # saving the STL. Treat them as protected solids up front so every consumer
    # sees the same physical field.
    if force_field is None:
        # Default load nodes lie on x=nelx, z=0; protect their adjacent
        # element strip just as an explicit force-field region is protected.
        protected_zone_mask[:, -1, 0] = True
    else:
        expected_force_shape = expected_shape + (3,)
        if force_field.shape != expected_force_shape:
            raise ValueError(
                f"force_field has shape {force_field.shape}, "
                f"expected {expected_force_shape}"
            )
        protected_zone_mask |= np.any(force_field != 0.0, axis=-1)
    if support_mask is None:
        # Default supports fix the x=0 node face.
        protected_zone_mask[:, 0, :] = True
    else:
        if support_mask.shape != expected_shape:
            raise ValueError(
                f"support_mask has shape {support_mask.shape}, "
                f"expected {expected_shape}"
            )
        if np.any(support_mask):
            protected_zone_mask |= np.asarray(support_mask, dtype=bool)
        else:
            protected_zone_mask[:, 0, :] = True

    overlap = obstacle_mask & protected_zone_mask
    if np.any(overlap):
        logger.warning(
            "Protected-solid and protected-void masks overlap at %d elements; "
            "protected void takes precedence.",
            int(np.sum(overlap)),
        )
        protected_zone_mask[overlap] = False

    free_mask = ~(obstacle_mask | protected_zone_mask)
    design_nele = int(np.sum(free_mask))
    if design_nele == 0:
        raise ValueError("optimization requires at least one free design element")
    initial_design_array = None
    if initial_design is not None:
        initial_design_array = np.asarray(initial_design, dtype=float).copy()
        if initial_design_array.shape != expected_shape:
            raise ValueError(
                f"initial_design has shape {initial_design_array.shape}, "
                f"expected {expected_shape}"
            )
        if not np.all(np.isfinite(initial_design_array)):
            raise ValueError("initial_design must contain only finite values")
        if np.any(initial_design_array < 0.0) or np.any(
            initial_design_array > 1.0
        ):
            raise ValueError("initial_design values must lie in [0, 1]")
        free_lower_bound = mma_min_density if failure_constrained else 0.0
        if np.any(initial_design_array[free_mask] < free_lower_bound):
            raise ValueError(
                "initial_design free values must be at least "
                f"{free_lower_bound:g} for the selected optimization mode"
            )
    target_free_volume = float(volfrac * design_nele)
    nele = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    logger.debug("free design elements: %d/%d", design_nele, nele)

    gpu = HAS_CUPY and use_gpu
    if use_gpu and not HAS_CUPY:
        logger.warning("GPU requested, but CuPy not found – falling back to CPU.")
    elif gpu:
        logger.info("Using GPU acceleration with CuPy.")

    history = (
        {
            "density_history": [],
            "iteration_history": [],
            "compliance_history": [],
            "beta_history": [],
        }
        if save_history
        else None
    )
    if history is not None and optimizer == "mma":
        history.update(
            {
                "volume_constraint_history": [],
                "mma_kkt_residual_history": [],
                "mma_subproblem_kkt_residual_history": [],
                "mma_record_kind_history": [],
                "mma_restored_from_iteration_history": [],
            }
        )
        if failure_constrained:
            history.update(
                {
                    "failure_aggregate_history": [],
                    "failure_exact_max_history": [],
                    "failure_constraint_history": [],
                    "failure_constraint_violation_history": [],
                    "failure_limit_history": [],
                    "failure_aggregate_exponent_history": [],
                    "critical_failure_mode_history": [],
                    "critical_element_history": [],
                    "predicted_failure_load_history": [],
                }
            )

    F = build_force_vector(nelx, nely, nelz, ndof, force_field)
    freedofs0, _ = build_supports(nelx, nely, nelz, ndof, support_mask)
    if material_params is None:
        KE = lk_H8(elem_size=elem_size)
    else:
        KE = lk_H8(*material_params, elem_size=elem_size)

    failure_constitutive = None
    failure_orientation = None
    failure_eligible = None
    failure_reference_load = None
    if failure_constrained:
        failure_constitutive = make_C_matrix(*material_params)
        failure_orientation = validate_orientation_matrix(
            np.eye(3) if material_orientation is None else material_orientation
        )
        # Boundary/load attachment elements are fixed fixtures and are kept in
        # the exact final diagnostic, but excluded from the optimization
        # aggregate so support singularities do not dominate it.
        failure_eligible = free_mask.ravel(order="F")
        failure_reference_load = float(np.sum(np.abs(F)))
        if failure_reference_load <= 0.0:
            raise ValueError("failure-constrained optimization requires nonzero load")

    edofMat, iK, jK = build_edof(nelx, nely, nelz)
    iK0, jK0 = iK - 1, jK - 1
    solver_func, solver_name = get_solver(use_gpu)
    logger.info("Linear solver: %s", solver_name)

    H, Hs = build_filter(nelx, nely, nelz, rmin)
    i_unique, j_unique, dup2uniq = _make_scatter_map(iK0, jK0, ndof)

    if gpu:
        xp = cp
        H_work = cusp.csr_matrix(
            (cp.asarray(H.data), cp.asarray(H.indices), cp.asarray(H.indptr)),
            shape=H.shape,
        )
        Hs_work = cp.asarray(Hs)
        KE_work = cp.asarray(KE)
        F_work = cp.asarray(F)
        freedofs_work = cp.asarray(freedofs0)
        edof_work = cp.asarray(edofMat)
        protected_solid_work = cp.asarray(protected_zone_mask)
        protected_void_work = cp.asarray(obstacle_mask)
        free_work = cp.asarray(free_mask)
        dup2uniq_work = cp.asarray(dup2uniq)
        K = cusp.csr_matrix(
            (
                cp.zeros(len(i_unique)),
                (cp.asarray(i_unique), cp.asarray(j_unique)),
            ),
            shape=(ndof, ndof),
        )
    else:
        xp = np
        H_work = H
        Hs_work = Hs
        KE_work = KE
        F_work = F
        freedofs_work = freedofs0
        edof_work = edofMat
        protected_solid_work = protected_zone_mask
        protected_void_work = obstacle_mask
        free_work = free_mask
        dup2uniq_work = dup2uniq
        K = sp.csr_matrix(
            (np.zeros(len(i_unique)), (i_unique, j_unique)),
            shape=(ndof, ndof),
        )

    initial_free_density = (
        max(volfrac, mma_min_density) if failure_constrained else volfrac
    )
    x = (
        xp.full(expected_shape, initial_free_density, dtype=float)
        if initial_design_array is None
        else xp.asarray(initial_design_array)
    )
    x[protected_solid_work] = 1.0
    x[protected_void_work] = 0.0

    rho_filtered, rho_physical, projection_derivative = build_physical_density(
        x,
        H=H_work,
        Hs=Hs_work,
        beta=beta_schedule[0],
        eta=projection_eta,
        protected_solid=protected_solid_work,
        protected_void=protected_void_work,
        xp=xp,
    )

    def to_numpy(array):
        if gpu:
            return cp.asnumpy(array)
        return np.asarray(array).copy()

    def append_mma_history_record(record, density):
        """Append one evaluated/restored MMA design to aligned histories."""
        history["density_history"].append(to_numpy(density))
        history["iteration_history"].append(record["iteration"])
        history["compliance_history"].append(record["compliance"])
        history["beta_history"].append(record["beta"])
        history["volume_constraint_history"].append(
            record["volume_constraint"]
        )
        history["mma_kkt_residual_history"].append(
            record["mma_kkt_residual"]
        )
        history["mma_subproblem_kkt_residual_history"].append(
            record.get("mma_subproblem_kkt_residual")
        )
        history["mma_record_kind_history"].append(
            "restoration"
            if record.get("continuation_restoration", False)
            else "evaluation"
        )
        history["mma_restored_from_iteration_history"].append(
            record.get("restored_from_iteration")
        )
        if failure_constrained:
            history["failure_aggregate_history"].append(
                record["failure_aggregate"]
            )
            history["failure_exact_max_history"].append(
                record["failure_exact_max"]
            )
            history["failure_constraint_history"].append(
                record["failure_constraint"]
            )
            history["failure_constraint_violation_history"].append(
                record["failure_constraint_violation"]
            )
            history["failure_limit_history"].append(record["failure_limit"])
            history["failure_aggregate_exponent_history"].append(
                record["failure_aggregate_exponent"]
            )
            history["critical_failure_mode_history"].append(
                record["critical_failure_mode"]
            )
            history["critical_element_history"].append(
                record["critical_element"]
            )
            history["predicted_failure_load_history"].append(
                record["predicted_failure_load"]
            )

    def design_checksum(design):
        """Compact deterministic checksum used to audit stage warm starts."""
        free_values = np.asarray(design).ravel(order="F")[
            free_mask.ravel(order="F")
        ]
        weights = np.arange(1, free_values.size + 1, dtype=float)
        return float(np.dot(free_values, weights) / np.sum(weights))

    def evaluate_failure_iteration(
        reduced_stiffness,
        displacement,
        physical_density,
        projection_derivative,
        correction_factor,
        aggregate_exponent,
    ):
        """Evaluate one failure aggregate and its full design derivative."""
        displacement_numpy = to_numpy(displacement)
        density_numpy = to_numpy(physical_density)

        def build_partials(frozen_correction):
            return evaluate_failure_partials(
                displacement_numpy,
                edofMat,
                failure_constitutive,
                failure_orientation,
                density_numpy,
                material_strength,
                elem_size=elem_size,
                relaxation_exponent=failure_relaxation_exponent,
                aggregate_exponent=aggregate_exponent,
                correction_factor=frozen_correction,
                eligible_elements=failure_eligible,
            )

        if correction_factor is None:
            uncorrected = build_partials(1.0)
            correction_factor = calibrate_pnorm_correction(
                uncorrected.aggregate_result.element_failure_index,
                exponent=aggregate_exponent,
                eligible=failure_eligible,
            )
        partials = build_partials(correction_factor)

        if gpu:
            right_hand_side = cp.asarray(
                partials.displacement_derivative[freedofs0]
            )
            right_hand_side_norm = float(cp.linalg.norm(right_hand_side).item())
            adjoint = cp.zeros(ndof)
            if right_hand_side_norm == 0.0:
                adjoint_solve_count = 0
                adjoint_relative_residual = 0.0
            else:
                adjoint_free = solver_func(reduced_stiffness, right_hand_side)
                adjoint[freedofs_work] = adjoint_free
                residual = reduced_stiffness @ adjoint_free - right_hand_side
                adjoint_relative_residual = float(
                    (cp.linalg.norm(residual) / right_hand_side_norm).item()
                )
                adjoint_solve_count = 1

            element_displacement = displacement[edof_work.astype(int) - 1]
            element_adjoint = adjoint[edof_work.astype(int) - 1]
            adjoint_energy = cp.sum(
                (element_adjoint @ KE_work) * element_displacement,
                axis=1,
            )
            density_flat = physical_density.ravel(order="F")
            stiffness_derivative = (
                penal
                * (1.0 - 1.0e-9)
                * density_flat ** (penal - 1.0)
            )
            adjoint_density_derivative = -stiffness_derivative * adjoint_energy
            physical_density_derivative = (
                cp.asarray(partials.explicit_density_derivative)
                + adjoint_density_derivative
            )
        else:
            adjoint_result = solve_failure_adjoint(
                reduced_stiffness,
                partials.displacement_derivative,
                freedofs0,
                linear_solver=solver_func,
            )
            _, physical_density_derivative = combine_failure_density_gradient(
                partials.explicit_density_derivative,
                adjoint_result.adjoint,
                displacement_numpy,
                edofMat,
                KE,
                density_numpy,
                simp_penal=penal,
            )
            adjoint_solve_count = adjoint_result.solve_count
            adjoint_relative_residual = adjoint_result.relative_residual

        physical_density_derivative = physical_density_derivative.reshape(
            expected_shape,
            order="F",
        )
        design_derivative = apply_density_filter_chain_rule(
            physical_density_derivative,
            projection_derivative,
            H_work,
            Hs_work,
        )
        design_derivative = xp.where(free_work, design_derivative, 0.0)

        critical = critical_failure_location(
            partials.gauss_failure,
            eligible_elements=failure_eligible,
        )
        predicted_load = predicted_failure_load(
            failure_reference_load,
            partials.aggregate_result.exact_max,
        )
        predicted_load = (
            float(predicted_load) if np.isfinite(predicted_load) else None
        )
        return {
            "partials": partials,
            "design_derivative": design_derivative,
            "correction_factor": float(correction_factor),
            "critical": critical,
            "predicted_failure_load": predicted_load,
            "adjoint_solve_count": int(adjoint_solve_count),
            "adjoint_relative_residual": float(adjoint_relative_residual),
        }

    def achievable_free_volume_bounds(beta):
        lower_design = xp.zeros_like(x)
        if failure_constrained:
            lower_design[free_work] = mma_min_density
        lower_design[protected_solid_work] = 1.0
        lower_design[protected_void_work] = 0.0
        upper_design = xp.ones_like(x)
        upper_design[protected_void_work] = 0.0

        _, lower_density, _ = build_physical_density(
            lower_design,
            H=H_work,
            Hs=Hs_work,
            beta=beta,
            eta=projection_eta,
            protected_solid=protected_solid_work,
            protected_void=protected_void_work,
            xp=xp,
        )
        _, upper_density, _ = build_physical_density(
            upper_design,
            H=H_work,
            Hs=Hs_work,
            beta=beta,
            eta=projection_eta,
            protected_solid=protected_solid_work,
            protected_void=protected_void_work,
            xp=xp,
        )
        return (
            float(xp.mean(lower_density[free_work]).item()),
            float(xp.mean(upper_density[free_work]).item()),
        )

    total_loop = 0
    c = float("nan")
    stage_summaries = []
    mma_iteration_records = []
    mma_objective_reference = None
    mma_termination_status = None
    mma_optimization_feasible = None
    last_mma_constraints = None
    last_mma_kkt = None
    last_failure_iteration = None
    mma_abort = False
    last_executed_beta = beta_schedule[0]
    last_executed_failure_limit = stage_failure_limits[0]
    last_executed_failure_exponent = stage_failure_exponents[0]

    for stage_index, (
        beta,
        stage_failure_limit,
        stage_failure_exponent,
    ) in enumerate(
        zip(beta_schedule, stage_failure_limits, stage_failure_exponents)
    ):
        last_executed_beta = beta
        last_executed_failure_limit = stage_failure_limit
        last_executed_failure_exponent = stage_failure_exponent
        logger.info("Starting projection stage: beta=%g", beta)
        # MMA may use its entire evaluation budget before taking an update
        # (notably maxloop=1), and a failed first subproblem must still produce
        # finite diagnostics. Convergence also requires a populated MMA state,
        # so zero here cannot cause premature convergence.
        change = float("inf") if optimizer == "oc" else 0.0
        stage_loop = 0
        c_prev = float("nan")
        mma_state = None
        mma_convergence_count = 0
        mma_stage_finished = False
        mma_stage_converged = False
        mma_subproblem_failed = False
        failure_correction_factor = None
        stage_last_constraint_violation = float("inf")
        stage_last_subproblem_kkt = float("nan")
        stage_start_design = to_numpy(x)
        stage_best_feasible_snapshot = None
        stage_least_violation_snapshot = None
        stage_restored_iteration = None

        rho_filtered, rho_physical, projection_derivative = build_physical_density(
            x,
            H=H_work,
            Hs=Hs_work,
            beta=beta,
            eta=projection_eta,
            protected_solid=protected_solid_work,
            protected_void=protected_void_work,
            xp=xp,
        )
        minimum_volume_fraction, maximum_volume_fraction = (
            achievable_free_volume_bounds(beta)
        )
        target_within_achievable_range = (
            minimum_volume_fraction - 1.0e-10
            <= volfrac
            <= maximum_volume_fraction + 1.0e-10
        )
        if volfrac < minimum_volume_fraction - 1.0e-10:
            raise ValueError(
                "projected free-volume target is infeasible at "
                f"beta={beta:g}: target={volfrac:.6f}, minimum achievable="
                f"{minimum_volume_fraction:.6f}. Protected-solid density "
                "spreads into the free region through the density filter."
            )
        if volfrac > maximum_volume_fraction + 1.0e-10:
            logger.warning(
                "Projected free-volume target %.6f exceeds the maximum "
                "achievable %.6f at beta=%g; the volume upper bound is "
                "inactive at the all-solid free design.",
                volfrac,
                maximum_volume_fraction,
                beta,
            )

        while stage_loop < maxloop and (
            change > tolx if optimizer == "oc" else not mma_stage_finished
        ):
            stage_loop += 1
            total_loop += 1
            t0 = time.time()
            rho_used_for_fea = rho_physical

            stiffness_scale = 1.0e-9 + (1.0 - 1.0e-9) * (
                rho_physical.ravel(order="F") ** penal
            )
            element_values = xp.kron(stiffness_scale, KE_work.ravel())
            K.data.fill(0.0)
            xp.add.at(K.data, dup2uniq_work, element_values)

            Kff = K[freedofs_work, :][:, freedofs_work]
            Uf = solver_func(Kff, F_work[freedofs_work])
            U = xp.zeros(ndof)
            U[freedofs_work] = Uf

            element_energy_flat = element_compliance(U, edof_work, KE_work)
            element_energy = element_energy_flat.reshape(
                expected_shape,
                order="F",
            )
            stiffness_scale_3d = (
                1.0e-9
                + (1.0 - 1.0e-9) * rho_physical**penal
            )
            c = float(xp.sum(stiffness_scale_3d * element_energy).item())

            dc_drho = (
                -penal
                * (1.0 - 1.0e-9)
                * rho_physical ** (penal - 1.0)
                * element_energy
            )
            dv_drho = free_work.astype(float)
            dc_dx = apply_density_filter_chain_rule(
                dc_drho,
                projection_derivative,
                H_work,
                Hs_work,
            )
            dv_dx = apply_density_filter_chain_rule(
                dv_drho,
                projection_derivative,
                H_work,
                Hs_work,
            )

            failure_iteration = None
            mma_diagnostics = None
            mma_iteration_record = None
            if optimizer == "oc":
                x_new, rho_new = optimality_criteria_update_projected(
                    x=x,
                    dc_dx=dc_dx,
                    dv_dx=dv_dx,
                    move=move,
                    target_free_volume=target_free_volume,
                    H=H_work,
                    Hs=Hs_work,
                    beta=beta,
                    eta=projection_eta,
                    free_mask=free_work,
                    protected_solid=protected_solid_work,
                    protected_void=protected_void_work,
                    xp=xp,
                )
                change = float(
                    xp.max(xp.abs(x_new[free_work] - x[free_work])).item()
                )
                x = x_new
                rho_physical = rho_new
                current_volume_fraction = float(
                    xp.mean(rho_physical[free_work]).item()
                )
                gray_fraction = float(
                    xp.mean(
                        (
                            (rho_physical[free_work] > 0.05)
                            & (rho_physical[free_work] < 0.95)
                        ).astype(float)
                    ).item()
                )
            else:
                current_volume_fraction = float(
                    xp.mean(rho_used_for_fea[free_work]).item()
                )
                gray_fraction = float(
                    xp.mean(
                        (
                            (rho_used_for_fea[free_work] > 0.05)
                            & (rho_used_for_fea[free_work] < 0.95)
                        ).astype(float)
                    ).item()
                )
                volume_constraint = current_volume_fraction / volfrac - 1.0
                constraint_values = [volume_constraint]
                constraint_gradient_fields = [dv_dx / target_free_volume]

                if failure_constrained:
                    failure_iteration = evaluate_failure_iteration(
                        Kff,
                        U,
                        rho_used_for_fea,
                        projection_derivative,
                        failure_correction_factor,
                        stage_failure_exponent,
                    )
                    failure_correction_factor = failure_iteration[
                        "correction_factor"
                    ]
                    failure_aggregate = failure_iteration[
                        "partials"
                    ].aggregate_result.aggregate
                    failure_constraint = (
                        failure_aggregate / stage_failure_limit - 1.0
                    )
                    constraint_values.append(failure_constraint)
                    constraint_gradient_fields.append(
                        failure_iteration["design_derivative"]
                        / stage_failure_limit
                    )
                    last_failure_iteration = failure_iteration

                constraint_values = np.asarray(constraint_values, dtype=float)
                constraint_gradients = np.stack(
                    [
                        to_numpy(field).ravel(order="F")[
                            free_mask.ravel(order="F")
                        ]
                        for field in constraint_gradient_fields
                    ]
                )
                current_design_numpy = to_numpy(x)
                x_free = current_design_numpy.ravel(order="F")[
                    free_mask.ravel(order="F")
                ]
                objective_gradient_free = to_numpy(dc_dx).ravel(order="F")[
                    free_mask.ravel(order="F")
                ]
                current_objective_reference = (
                    mma_objective_reference
                    if mma_objective_reference is not None
                    else max(abs(c), 1.0e-12)
                )
                current_multipliers = (
                    np.zeros(constraint_values.size)
                    if mma_state is None
                    else mma_state.dual_multipliers
                )
                mma_kkt = _mma_nonlinear_kkt_components(
                    x_free,
                    objective_gradient_free / current_objective_reference,
                    constraint_values,
                    constraint_gradients,
                    current_multipliers,
                    mma_min_density if failure_constrained else 0.0,
                    1.0,
                )
                last_mma_constraints = constraint_values.copy()
                last_mma_kkt = mma_kkt
                stage_last_constraint_violation = max(
                    0.0, float(np.max(constraint_values))
                )

                convergence_ready = (
                    mma_state is not None
                    and change <= tolx
                    and stage_last_constraint_violation <= 1.0e-3
                    and mma_kkt["residual"] <= 5.0e-3
                )
                mma_convergence_count = (
                    mma_convergence_count + 1 if convergence_ready else 0
                )
                mma_stage_converged = mma_convergence_count >= 3
                stop_before_update = mma_stage_converged or stage_loop >= maxloop

                mma_iteration_record = {
                    "iteration": int(total_loop - 1),
                    "continuation_stage": int(stage_index),
                    "beta": float(beta),
                    "failure_limit": (
                        float(stage_failure_limit)
                        if failure_constrained
                        else None
                    ),
                    "failure_aggregate_exponent": (
                        float(stage_failure_exponent)
                        if failure_constrained
                        else None
                    ),
                    "compliance": float(c),
                    "physical_density_fraction": current_volume_fraction,
                    "volume_constraint": float(volume_constraint),
                    "max_constraint_violation": stage_last_constraint_violation,
                    "mma_kkt_residual": float(mma_kkt["residual"]),
                    "mma_kkt_components": dict(mma_kkt),
                    "multipliers_for_current_design": current_multipliers.tolist(),
                }
                if failure_iteration is not None:
                    partials = failure_iteration["partials"]
                    critical = failure_iteration["critical"]
                    mma_iteration_record.update(
                        {
                            "failure_aggregate": float(
                                partials.aggregate_result.aggregate
                            ),
                            "failure_exact_max": float(
                                partials.aggregate_result.exact_max
                            ),
                            "failure_constraint": float(failure_constraint),
                            "failure_constraint_violation": max(
                                0.0, float(failure_constraint)
                            ),
                            "critical_failure_mode": critical.mode,
                            "critical_element": int(critical.element),
                            "critical_gauss_point": int(critical.gauss_point),
                            "predicted_failure_load": failure_iteration[
                                "predicted_failure_load"
                            ],
                            "failure_correction_factor": float(
                                failure_correction_factor
                            ),
                            "failure_adjoint_relative_residual": float(
                                failure_iteration["adjoint_relative_residual"]
                            ),
                            "failure_adjoint_solve_count": int(
                                failure_iteration["adjoint_solve_count"]
                            ),
                        }
                    )

                snapshot = (
                    current_design_numpy.copy(),
                    dict(mma_iteration_record),
                )
                if (
                    stage_least_violation_snapshot is None
                    or stage_last_constraint_violation
                    < stage_least_violation_snapshot[1][
                        "max_constraint_violation"
                    ]
                    - 1.0e-12
                    or (
                        np.isclose(
                            stage_last_constraint_violation,
                            stage_least_violation_snapshot[1][
                                "max_constraint_violation"
                            ],
                        )
                        and c
                        < stage_least_violation_snapshot[1]["compliance"]
                    )
                ):
                    stage_least_violation_snapshot = snapshot
                if stage_last_constraint_violation <= 1.0e-3 and (
                    stage_best_feasible_snapshot is None
                    or c < stage_best_feasible_snapshot[1]["compliance"]
                ):
                    stage_best_feasible_snapshot = snapshot

                if stop_before_update:
                    x_new = x.copy()
                    rho_new = rho_physical.copy()
                    mma_stage_finished = True
                else:
                    previous_mma_state = mma_state
                    subproblem_attempts = []
                    mma_result = None
                    mma_result_move_limit = None
                    for move_factor in (1.0, 0.5, 0.25):
                        attempted_move = mma_move * move_factor
                        try:
                            candidate_result = mma_update(
                                x_free,
                                c,
                                objective_gradient_free,
                                constraint_values,
                                constraint_gradients,
                                lower_bounds=(
                                    mma_min_density
                                    if failure_constrained
                                    else 0.0
                                ),
                                upper_bounds=1.0,
                                move_limit=attempted_move,
                                state=previous_mma_state,
                                objective_reference=mma_objective_reference,
                            )
                        except (
                            RuntimeError,
                            FloatingPointError,
                            np.linalg.LinAlgError,
                        ) as exc:
                            subproblem_attempts.append(
                                {
                                    "move_limit": float(attempted_move),
                                    "kkt_residual": None,
                                    "success": False,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                }
                            )
                            continue
                        subproblem_attempts.append(
                            {
                                "move_limit": float(attempted_move),
                                "kkt_residual": float(
                                    candidate_result.diagnostics
                                    .subproblem_kkt_residual
                                ),
                                "success": bool(
                                    candidate_result.diagnostics.dual_success
                                ),
                            }
                        )
                        mma_result = candidate_result
                        mma_result_move_limit = attempted_move
                        if candidate_result.diagnostics.dual_success:
                            break
                    if mma_result is None:
                        mma_iteration_record.update(
                            {
                                "mma_subproblem_kkt_residual": None,
                                "mma_subproblem_slack": None,
                                "mma_dual_status": None,
                                "mma_dual_success": False,
                                "mma_effective_move_limit": (
                                    subproblem_attempts[-1]["move_limit"]
                                ),
                                "mma_subproblem_attempts": subproblem_attempts,
                            }
                        )
                        mma_subproblem_failed = True
                        mma_stage_finished = True
                        mma_abort = True
                        mma_termination_status = "subproblem_failed"
                        x_new = x.copy()
                        rho_new = rho_physical.copy()
                    else:
                        mma_state = mma_result.state
                        mma_objective_reference = mma_state.objective_reference
                        mma_diagnostics = mma_result.diagnostics
                        stage_last_subproblem_kkt = (
                            mma_diagnostics.subproblem_kkt_residual
                        )
                        mma_iteration_record.update(
                            {
                                "mma_subproblem_kkt_residual": float(
                                    mma_diagnostics.subproblem_kkt_residual
                                ),
                                "mma_subproblem_slack": (
                                    mma_diagnostics.slack_variables.tolist()
                                ),
                                "mma_dual_status": int(
                                    mma_diagnostics.dual_status
                                ),
                                "mma_dual_success": bool(
                                    mma_diagnostics.dual_success
                                ),
                                "mma_effective_move_limit": (
                                    float(mma_result_move_limit)
                                ),
                                "mma_subproblem_attempts": subproblem_attempts,
                            }
                        )
                        if not mma_diagnostics.dual_success:
                            mma_subproblem_failed = True
                            mma_stage_finished = True
                            mma_abort = True
                            mma_termination_status = "subproblem_failed"
                            x_new = x.copy()
                            rho_new = rho_physical.copy()
                        else:
                            x_new = x.copy()
                            updated_free = (
                                cp.asarray(mma_result.x_new)
                                if gpu
                                else mma_result.x_new
                            )
                            x_new[free_work] = updated_free
                            x_new[protected_solid_work] = 1.0
                            x_new[protected_void_work] = 0.0
                            _, rho_new, _ = build_physical_density(
                                x_new,
                                H=H_work,
                                Hs=Hs_work,
                                beta=beta,
                                eta=projection_eta,
                                protected_solid=protected_solid_work,
                                protected_void=protected_void_work,
                                xp=xp,
                            )
                            change = float(
                                xp.max(
                                    xp.abs(x_new[free_work] - x[free_work])
                                ).item()
                            )
                            x = x_new
                            rho_physical = rho_new

                mma_iteration_records.append(mma_iteration_record)

            c_delta = c - c_prev if np.isfinite(c_prev) else float("nan")
            c_prev = c
            elapsed = time.time() - t0

            if save_history and (
                (total_loop - 1) % history_frequency == 0
                or change <= tolx
                or stage_loop == maxloop
                or mma_stage_finished
            ):
                if optimizer == "mma":
                    append_mma_history_record(
                        mma_iteration_record,
                        rho_used_for_fea,
                    )
                else:
                    history["density_history"].append(
                        to_numpy(rho_used_for_fea)
                    )
                    history["iteration_history"].append(total_loop - 1)
                    history["compliance_history"].append(c)
                    history["beta_history"].append(beta)

            if optimizer == "oc":
                logger.info(
                    "beta=%4.1f iter=%4d total=%4d C=%.6e dC=%.3e "
                    "vol=%.5f gray=%.5f change=%.5e time=%.2fs",
                    beta,
                    stage_loop,
                    total_loop,
                    c,
                    c_delta,
                    current_volume_fraction,
                    gray_fraction,
                    change,
                    elapsed,
                )
            else:
                failure_log = ""
                if failure_constrained:
                    failure_log = (
                        f" gF={mma_iteration_record['failure_constraint']:.3e}"
                        f" FI={mma_iteration_record['failure_exact_max']:.3e}"
                    )
                logger.info(
                    "beta=%4.1f iter=%4d total=%4d MMA C=%.6e dC=%.3e "
                    "vol=%.5f gV=%.3e%s KKT=%.3e change=%.5e time=%.2fs",
                    beta,
                    stage_loop,
                    total_loop,
                    c,
                    c_delta,
                    current_volume_fraction,
                    mma_iteration_record["volume_constraint"],
                    failure_log,
                    mma_iteration_record["mma_kkt_residual"],
                    change,
                    elapsed,
                )

            rho_filtered, rho_physical, projection_derivative = (
                build_physical_density(
                    x,
                    H=H_work,
                    Hs=Hs_work,
                    beta=beta,
                    eta=projection_eta,
                    protected_solid=protected_solid_work,
                    protected_void=protected_void_work,
                    xp=xp,
                )
            )

        stage_continuation_feasible = None
        stage_restoration_linf = 0.0
        stage_subproblem_failure_recovered = False
        if optimizer == "mma":
            stage_continuation_feasible = (
                stage_best_feasible_snapshot is not None
            )
            selected_snapshot = None
            if not mma_stage_converged:
                selected_snapshot = (
                    stage_best_feasible_snapshot
                    if stage_best_feasible_snapshot is not None
                    else stage_least_violation_snapshot
                )
            if selected_snapshot is not None:
                selected_design, selected_record = selected_snapshot
                current_design = to_numpy(x)
                stage_restoration_linf = float(
                    np.max(
                        np.abs(
                            selected_design[free_mask]
                            - current_design[free_mask]
                        )
                    )
                )
                selected_is_current = (
                    mma_iteration_records
                    and selected_record["iteration"]
                    == mma_iteration_records[-1]["iteration"]
                    and stage_restoration_linf <= 1.0e-14
                )
                if not selected_is_current:
                    x = (
                        cp.asarray(selected_design)
                        if gpu
                        else selected_design.copy()
                    )
                    x[protected_solid_work] = 1.0
                    x[protected_void_work] = 0.0
                    rho_filtered, rho_physical, projection_derivative = (
                        build_physical_density(
                            x,
                            H=H_work,
                            Hs=Hs_work,
                            beta=beta,
                            eta=projection_eta,
                            protected_solid=protected_solid_work,
                            protected_void=protected_void_work,
                            xp=xp,
                        )
                    )
                    restored_record = dict(selected_record)
                    restored_record.update(
                        {
                            "iteration": int(total_loop - 1),
                            "continuation_restoration": True,
                            "restored_from_iteration": int(
                                selected_record["iteration"]
                            ),
                        }
                    )
                    mma_iteration_records.append(restored_record)
                    stage_restored_iteration = int(
                        selected_record["iteration"]
                    )
                    if save_history:
                        append_mma_history_record(
                            restored_record,
                            rho_physical,
                        )
                last_mma_constraints = np.asarray(
                    [
                        selected_record["volume_constraint"],
                        *(
                            [selected_record["failure_constraint"]]
                            if failure_constrained
                            else []
                        ),
                    ],
                    dtype=float,
                )
                last_mma_kkt = dict(selected_record["mma_kkt_components"])
                stage_last_constraint_violation = float(
                    selected_record["max_constraint_violation"]
                )
                c = float(selected_record["compliance"])
            if (
                mma_abort
                and mma_termination_status == "subproblem_failed"
                and stage_continuation_feasible
            ):
                # The rejected subproblem never changed x. A previously
                # evaluated feasible design is therefore a safe continuation
                # point after resetting the stage-local MMA state.
                stage_subproblem_failure_recovered = True
                mma_abort = False
                mma_termination_status = None
            if not stage_continuation_feasible and not mma_abort:
                mma_termination_status = "continuation_stage_infeasible"
                mma_abort = True

        stage_volume_fraction = float(
            xp.mean(rho_physical[free_work]).item()
        )
        stage_converged = (
            change <= tolx if optimizer == "oc" else mma_stage_converged
        )
        if not stage_converged:
            if optimizer == "oc":
                logger.warning(
                    "Projection stage beta=%g reached maxloop=%d without "
                    "converging: final change %.6e exceeds tolx %.6e.",
                    beta,
                    maxloop,
                    change,
                    tolx,
                )
            else:
                logger.warning(
                    "MMA stage beta=%g did not converge: change=%.3e, "
                    "violation=%.3e, KKT=%s, subproblem_failed=%s.",
                    beta,
                    change,
                    stage_last_constraint_violation,
                    (
                        "n/a"
                        if last_mma_kkt is None
                        else f"{last_mma_kkt['residual']:.3e}"
                    ),
                    mma_subproblem_failed,
                )
        stage_gray_fraction = float(
            xp.mean(
                (
                    (rho_physical[free_work] > 0.05)
                    & (rho_physical[free_work] < 0.95)
                ).astype(float)
            ).item()
        )
        stage_mma_records = [
            record
            for record in mma_iteration_records
            if record.get("continuation_stage") == stage_index
        ]
        stage_failure_constraints = [
            record["failure_constraint"]
            for record in stage_mma_records
            if "failure_constraint" in record
        ]
        failure_tail = stage_failure_constraints[-10:]
        failure_tail_peak_to_peak = (
            float(max(failure_tail) - min(failure_tail))
            if failure_tail
            else None
        )
        maximum_failure_constraint_jump = (
            float(np.max(np.abs(np.diff(stage_failure_constraints))))
            if len(stage_failure_constraints) > 1
            else 0.0 if stage_failure_constraints else None
        )
        stage_summaries.append(
            {
                "beta": float(beta),
                "continuation_stage": int(stage_index),
                "failure_limit": (
                    float(stage_failure_limit) if failure_constrained else None
                ),
                "failure_aggregate_exponent": (
                    float(stage_failure_exponent)
                    if failure_constrained
                    else None
                ),
                "stage_start_design_checksum": design_checksum(
                    stage_start_design
                ),
                "stage_end_design_checksum": design_checksum(to_numpy(x)),
                "iterations": int(stage_loop),
                "converged": bool(stage_converged),
                "physical_density_fraction": stage_volume_fraction,
                "volume_fraction_error": stage_volume_fraction - float(volfrac),
                "gray_fraction_005_095": stage_gray_fraction,
                "final_change": float(change),
                "initial_constraint_violation": (
                    float(stage_mma_records[0]["max_constraint_violation"])
                    if stage_mma_records
                    else None
                ),
                "failure_constraint_tail_peak_to_peak": (
                    failure_tail_peak_to_peak
                ),
                "maximum_failure_constraint_jump": (
                    maximum_failure_constraint_jump
                ),
                "minimum_achievable_physical_fraction": minimum_volume_fraction,
                "maximum_achievable_physical_fraction": maximum_volume_fraction,
                "target_within_achievable_range": target_within_achievable_range,
                **(
                    {
                        "max_constraint_violation": float(
                            stage_last_constraint_violation
                        ),
                        "mma_kkt_residual": (
                            None
                            if last_mma_kkt is None
                            else float(last_mma_kkt["residual"])
                        ),
                        "mma_subproblem_kkt_residual": (
                            None
                            if not np.isfinite(stage_last_subproblem_kkt)
                            else float(stage_last_subproblem_kkt)
                        ),
                        "subproblem_failed": bool(mma_subproblem_failed),
                        "subproblem_failure_recovered": bool(
                            stage_subproblem_failure_recovered
                        ),
                        "continuation_feasible": bool(
                            stage_continuation_feasible
                        ),
                        "best_feasible_iteration": (
                            None
                            if stage_best_feasible_snapshot is None
                            else int(
                                stage_best_feasible_snapshot[1]["iteration"]
                            )
                        ),
                        "least_violation_iteration": (
                            None
                            if stage_least_violation_snapshot is None
                            else int(
                                stage_least_violation_snapshot[1]["iteration"]
                            )
                        ),
                        "restored_iteration": stage_restored_iteration,
                        "restoration_linf": float(stage_restoration_linf),
                        "design_change_from_stage_start": float(
                            np.max(
                                np.abs(
                                    to_numpy(x)[free_mask]
                                    - stage_start_design[free_mask]
                                )
                            )
                        ),
                    }
                    if optimizer == "mma"
                    else {}
                ),
            }
        )
        if mma_abort:
            break

    # A failed MMA subproblem stops continuation. Final reporting must use the
    # projection stage that actually produced ``x`` rather than a later,
    # unvisited beta from the requested schedule.
    final_beta = last_executed_beta
    rho_filtered, rho_physical, _ = build_physical_density(
        x,
        H=H_work,
        Hs=Hs_work,
        beta=final_beta,
        eta=projection_eta,
        protected_solid=protected_solid_work,
        protected_void=protected_void_work,
        xp=xp,
    )
    final_x = to_numpy(x)
    final_rho_filtered = to_numpy(rho_filtered)
    final_xPhys = to_numpy(rho_physical)

    final_compliance = evaluate_fixed_geometry_compliance(
        xPhys=final_xPhys,
        penal=penal,
        material_params=material_params,
        elem_size=elem_size,
        force_field=force_field,
        support_mask=support_mask,
        obstacle_mask=obstacle_mask,
        protected_zone_mask=protected_zone_mask,
        use_gpu=use_gpu,
    )
    if save_history and optimizer == "oc":
        history["density_history"].append(final_xPhys.copy())
        history["iteration_history"].append(total_loop)
        history["compliance_history"].append(float(final_compliance))
        history["beta_history"].append(float(final_beta))

    rho_binary = (final_xPhys >= 0.5).astype(float)
    rho_binary[protected_zone_mask] = 1.0
    rho_binary[obstacle_mask] = 0.0
    binary_compliance = evaluate_fixed_geometry_compliance(
        xPhys=rho_binary,
        penal=penal,
        material_params=material_params,
        elem_size=elem_size,
        force_field=force_field,
        support_mask=support_mask,
        obstacle_mask=obstacle_mask,
        protected_zone_mask=protected_zone_mask,
        use_gpu=use_gpu,
    )
    binary_compliance_delta = (
        (binary_compliance - final_compliance) / final_compliance
        if final_compliance != 0.0
        else None
    )

    free_values = final_xPhys[free_mask]
    gray_mask = (free_values > 0.05) & (free_values < 0.95)
    projection_metrics = {
        "optimization_mode": optimization_mode,
        "optimizer": optimizer,
        "initial_design_supplied": initial_design_array is not None,
        "projection_enabled": True,
        "projection_beta": float(final_beta),
        "projection_beta_schedule": [float(beta) for beta in beta_schedule],
        "failure_limit_schedule": (
            [float(value) for value in stage_failure_limits]
            if failure_constrained
            else None
        ),
        "failure_aggregate_exponent_schedule": (
            [float(value) for value in stage_failure_exponents]
            if failure_constrained
            else None
        ),
        "projection_eta": float(projection_eta),
        "penal": float(penal),
        "design_variable_fraction": float(np.mean(final_x[free_mask])),
        "filtered_density_fraction": float(
            np.mean(final_rho_filtered[free_mask])
        ),
        "physical_density_fraction": float(np.mean(free_values)),
        "gray_fraction_005_095": float(np.mean(gray_mask.astype(float))),
        "projected_compliance": float(final_compliance),
        "binary_compliance": float(binary_compliance),
        "binary_compliance_delta": (
            None
            if binary_compliance_delta is None
            else float(binary_compliance_delta)
        ),
        "binary_threshold": 0.5,
        "target_free_volume_fraction": float(volfrac),
        "physical_volume_fraction_error": float(np.mean(free_values) - volfrac),
        "projection_total_iterations": int(total_loop),
        "projection_stage_summaries": stage_summaries,
        "projection_converged": all(
            stage["converged"] for stage in stage_summaries
        ),
        "projection_target_within_achievable_range": all(
            stage["target_within_achievable_range"] for stage in stage_summaries
        ),
        "continuation_stages_requested": len(beta_schedule),
        "continuation_stages_completed": len(stage_summaries),
        "continuation_completed": (
            len(stage_summaries) == len(beta_schedule)
            and (
                optimizer != "mma"
                or all(
                    stage.get("continuation_feasible", False)
                    for stage in stage_summaries
                )
            )
        ),
    }
    if optimizer == "mma":
        final_stage_index = len(stage_summaries) - 1
        final_stage_records = [
            record
            for record in mma_iteration_records
            if record.get("continuation_stage") == final_stage_index
        ]
        feasible_records = [
            record
            for record in final_stage_records
            if record["max_constraint_violation"] <= 1.0e-3
        ]
        best_feasible_record = (
            min(feasible_records, key=lambda record: record["compliance"])
            if feasible_records
            else None
        )
        least_violation_record = (
            min(
                final_stage_records,
                key=lambda record: record["max_constraint_violation"],
            )
            if final_stage_records
            else None
        )
        final_violation = (
            float("inf")
            if last_mma_constraints is None
            else max(0.0, float(np.max(last_mma_constraints)))
        )
        mma_optimization_feasible = final_violation <= 1.0e-3
        if mma_termination_status is None:
            if stage_summaries and all(
                stage["converged"] for stage in stage_summaries
            ):
                mma_termination_status = "converged"
            elif mma_optimization_feasible:
                mma_termination_status = "feasible_not_converged"
            else:
                mma_termination_status = "stalled_infeasible_or_not_found"
        projection_metrics.update(
            {
                "termination_status": mma_termination_status,
                "optimization_feasible": bool(mma_optimization_feasible),
                "max_constraint_violation": final_violation,
                "volume_constraint": (
                    None
                    if last_mma_constraints is None
                    else float(last_mma_constraints[0])
                ),
                "mma_objective_reference": (
                    None
                    if mma_objective_reference is None
                    else float(mma_objective_reference)
                ),
                "mma_move_limit": float(mma_move),
                "mma_min_density": (
                    float(mma_min_density) if failure_constrained else 0.0
                ),
                "mma_kkt_residual": (
                    None
                    if last_mma_kkt is None
                    else float(last_mma_kkt["residual"])
                ),
                "mma_kkt_components": last_mma_kkt,
                "mma_iteration_history": mma_iteration_records,
                "best_feasible_iteration": (
                    None
                    if best_feasible_record is None
                    else int(best_feasible_record["iteration"])
                ),
                "least_violation_iteration": (
                    None
                    if least_violation_record is None
                    else int(least_violation_record["iteration"])
                ),
                "failure_gpu_path": (
                    "hybrid_cpu_partials_gpu_adjoint"
                    if failure_constrained and gpu
                    else (
                        "cpu" if failure_constrained else None
                    )
                ),
            }
        )
        if failure_constrained and last_failure_iteration is not None:
            final_failure_record = next(
                (
                    record
                    for record in reversed(mma_iteration_records)
                    if "failure_aggregate" in record
                ),
                None,
            )
            if final_failure_record is not None:
                projection_metrics.update(
                    {
                        "failure_limit": float(last_executed_failure_limit),
                        "failure_aggregate_exponent": float(
                            last_executed_failure_exponent
                        ),
                        "failure_relaxation_exponent": float(
                            failure_relaxation_exponent
                        ),
                        "failure_aggregate": final_failure_record[
                            "failure_aggregate"
                        ],
                        "failure_exact_max": final_failure_record[
                            "failure_exact_max"
                        ],
                        "failure_constraint": final_failure_record[
                            "failure_constraint"
                        ],
                        "failure_constraint_violation": final_failure_record[
                            "failure_constraint_violation"
                        ],
                        "critical_failure_mode": final_failure_record[
                            "critical_failure_mode"
                        ],
                        "critical_element": final_failure_record[
                            "critical_element"
                        ],
                        "critical_gauss_point": final_failure_record[
                            "critical_gauss_point"
                        ],
                        "predicted_failure_load": final_failure_record[
                            "predicted_failure_load"
                        ],
                        "failure_correction_factor": final_failure_record[
                            "failure_correction_factor"
                        ],
                        "failure_adjoint_relative_residual": (
                            final_failure_record[
                                "failure_adjoint_relative_residual"
                            ]
                        ),
                    }
                )
    logger.info(
        "Final projection: beta=%.1f eta=%.3f vol=%.5f gray=%.5f "
        "C_projected=%.6e C_binary=%.6e binary_delta=%s",
        final_beta,
        projection_eta,
        projection_metrics["physical_density_fraction"],
        projection_metrics["gray_fraction_005_095"],
        final_compliance,
        binary_compliance,
        (
            "n/a"
            if binary_compliance_delta is None
            else f"{binary_compliance_delta:.6e}"
        ),
    )
    if diagnostics_out is not None:
        diagnostics_out.clear()
        diagnostics_out.update(projection_metrics)

    failure_force = (
        projection_metrics.get("predicted_failure_load")
        if failure_constrained
        else None
    )
    return final_xPhys, history, final_compliance, failure_force
