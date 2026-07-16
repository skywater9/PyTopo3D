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

from pytopo3d.core.compliance import element_compliance
from pytopo3d.utils.assembly import build_edof, build_force_field, build_force_vector, build_support_mask, build_supports
from pytopo3d.utils.filter import (
    HAS_CUPY,
    apply_density_filter_chain_rule,
    build_filter,
    build_physical_density,
)
from pytopo3d.utils.logger import get_logger
from pytopo3d.utils.oc_update import optimality_criteria_update_projected
from pytopo3d.utils.solver import get_solver
from pytopo3d.utils.stiffness import lk_H8
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
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]], Optional[np.ndarray], float]:
    """Run density-filtered SIMP optimization with Heaviside continuation.

    maxloop is the maximum number of iterations for each beta stage. The
    returned array keeps the legacy xPhys contract but now contains the
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

    F = build_force_vector(nelx, nely, nelz, ndof, force_field)
    freedofs0, _ = build_supports(nelx, nely, nelz, ndof, support_mask)
    if material_params is None:
        KE = lk_H8(elem_size=elem_size)
    else:
        KE = lk_H8(*material_params, elem_size=elem_size)

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

    x = xp.full(expected_shape, volfrac, dtype=float)
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

    def achievable_free_volume_bounds(beta):
        lower_design = xp.zeros_like(x)
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

    for beta in beta_schedule:
        logger.info("Starting projection stage: beta=%g", beta)
        change = float("inf")
        stage_loop = 0
        c_prev = float("nan")

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

        while change > tolx and stage_loop < maxloop:
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
            change = float(xp.max(xp.abs(x_new[free_work] - x[free_work])).item())
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

            c_delta = c - c_prev if np.isfinite(c_prev) else float("nan")
            c_prev = c
            elapsed = time.time() - t0

            if save_history and (
                (total_loop - 1) % history_frequency == 0
                or change <= tolx
                or stage_loop == maxloop
            ):
                history["density_history"].append(to_numpy(rho_used_for_fea))
                history["iteration_history"].append(total_loop - 1)
                history["compliance_history"].append(c)
                history["beta_history"].append(beta)

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

        stage_volume_fraction = float(
            xp.mean(rho_physical[free_work]).item()
        )
        stage_converged = change <= tolx
        if not stage_converged:
            logger.warning(
                "Projection stage beta=%g reached maxloop=%d without "
                "converging: final change %.6e exceeds tolx %.6e.",
                beta,
                maxloop,
                change,
                tolx,
            )
        stage_gray_fraction = float(
            xp.mean(
                (
                    (rho_physical[free_work] > 0.05)
                    & (rho_physical[free_work] < 0.95)
                ).astype(float)
            ).item()
        )
        stage_summaries.append(
            {
                "beta": float(beta),
                "iterations": int(stage_loop),
                "converged": bool(stage_converged),
                "physical_density_fraction": stage_volume_fraction,
                "volume_fraction_error": stage_volume_fraction - float(volfrac),
                "gray_fraction_005_095": stage_gray_fraction,
                "final_change": float(change),
                "minimum_achievable_physical_fraction": minimum_volume_fraction,
                "maximum_achievable_physical_fraction": maximum_volume_fraction,
                "target_within_achievable_range": target_within_achievable_range,
            }
        )

    final_beta = beta_schedule[-1]
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
    if save_history:
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
        "projection_enabled": True,
        "projection_beta": float(final_beta),
        "projection_beta_schedule": [float(beta) for beta in beta_schedule],
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
    }
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

    failure_force = None
    return final_xPhys, history, final_compliance, failure_force
