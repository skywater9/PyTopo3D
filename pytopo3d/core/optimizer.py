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
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from pytopo3d.core.compliance import element_compliance
from pytopo3d.utils.assembly import build_edof, build_force_field, build_force_vector, build_support_mask, build_supports
from pytopo3d.utils.filter import HAS_CUPY, apply_filter, build_filter
from pytopo3d.utils.logger import get_logger
from pytopo3d.utils.oc_update import optimality_criteria_update
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
) -> Dict[str, Optional[float]]:
    """
    Evaluate fixed-geometry response metrics under given material/BC settings.

    This performs a single FE solve with no OC update and returns:
    - compliance: objective definition used in optimization
    - u{x,y,z}_avg_load_patch: average displacement on loaded-node DOFs
    - k_avg_{x,y,z}: directional equivalent stiffness F_dir / abs(u_dir)
    - k_avg: equivalent stiffness on dominant loading direction (legacy)
    - F_total and F_total_{x,y,z}: absolute load magnitudes
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
    x_eval[obstacle_mask] = 0.0
    x_eval[protected_zone_mask] = 1.0

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

        return {
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

    return {
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
    material_params: Optional[[float]] = None,
    elem_size: float = 0.01, # 1 cm 
    obstacle_mask: Optional[np.ndarray] = None,
    force_field: Optional[np.ndarray] = None,
    support_mask: Optional[np.ndarray] = None,
    tolx: float = 0.01,
    maxloop: int = 2000,
    save_history: bool = False,
    history_frequency: int = 10,
    use_gpu: bool = False,
    protected_zone_mask: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]], Optional[np.ndarray], float]:
    # ─────────────────────── setup
    gpu = HAS_CUPY and use_gpu
    if use_gpu and not HAS_CUPY:
        logger.warning("GPU requested, but CuPy not found – falling back to CPU.")
    elif gpu:
        logger.info("Using GPU acceleration with CuPy.")

    E0, Emin, nu = 1.0, 1e-9, 0.3
    nele = nelx * nely * nelz
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

    design_nele = nele - obstacle_mask.sum() - protected_zone_mask.sum()
    logger.debug(f"design elements: {design_nele}/{nele}")

    # History dict
    history = (
        {"density_history": [], "iteration_history": [], "compliance_history": []}
        if save_history
        else None
    )


    # Force / supports
    F = build_force_vector(nelx, nely, nelz, ndof, force_field)
    freedofs0, _ = build_supports(nelx, nely, nelz, ndof, support_mask)

    # Element stiffness
    if material_params is None:
        KE = lk_H8(elem_size=elem_size)
    else:
        KE = lk_H8(
            *material_params,
            elem_size=elem_size
        )
        
    edofMat, iK, jK = build_edof(nelx, nely, nelz)
    iK0, jK0 = iK - 1, jK - 1

    # Solver
    solver_func, solver_name = get_solver(use_gpu)
    logger.info(f"Linear solver: {solver_name}")

    # GPU transfers
    if gpu:
        KE_gpu = cp.asarray(KE)
        iK0_gpu, jK0_gpu = cp.asarray(iK0), cp.asarray(jK0)
        F_gpu = cp.asarray(F)
        freedofs0_gpu = cp.asarray(freedofs0)
        obstacle_gpu = cp.asarray(obstacle_mask)
        protected_gpu = cp.asarray(protected_zone_mask)
        U_gpu = cp.zeros(ndof)

    # Filter
    H, Hs = build_filter(nelx, nely, nelz, rmin)
    if gpu:
        H_gpu = cusp.csr_matrix(
            (cp.asarray(H.data), cp.asarray(H.indices), cp.asarray(H.indptr)),
            shape=H.shape,
        )
        Hs_gpu = cp.asarray(Hs)

    # Initial design
    if gpu:
        x_gpu = cp.full((nely, nelx, nelz), volfrac)
        x_gpu[protected_gpu] = 1.0
        x_gpu[obstacle_gpu] = 0.0
        xPhys_gpu = apply_filter(H_gpu, x_gpu, Hs_gpu, x_gpu.shape, use_gpu=True)
        xPhys_gpu[obstacle_gpu] = 0.0
    else:
        x = np.full((nely, nelx, nelz), volfrac)
        x[protected_zone_mask] = 1.0
        x[obstacle_mask] = 0.0
        xPhys = (H * x.ravel(order="F") / Hs).reshape((nely, nelx, nelz), order="F")
        xPhys[obstacle_mask] = 0.0

    # ─────────────────────── sparsity pattern + scatter map
    logger.debug("Building global stiffness pattern & scatter map")
    i_unique, j_unique, dup2uniq = _make_scatter_map(iK0, jK0, ndof)

    if gpu:
        dup2uniq_gpu = cp.asarray(dup2uniq)
        K_gpu = cusp.csr_matrix(
            (cp.zeros(len(i_unique)), (cp.asarray(i_unique), cp.asarray(j_unique))),
            shape=(ndof, ndof),
        )
    else:
        K = sp.csr_matrix((np.zeros(len(i_unique)), (i_unique, j_unique)), shape=(ndof, ndof))

    # ─────────────────────── main loop
    loop, change, c_prev = 0, 1.0, np.inf
    c = np.nan
    if save_history:
        history_frequency = history_frequency
        if gpu:
            history["density_history"].append(cp.asnumpy(xPhys_gpu))
        else:
            history["density_history"].append(xPhys.copy())
        history["iteration_history"].append(0)
        history["compliance_history"].append(0.0)

    while change > tolx and loop < maxloop:
        loop += 1
        t0 = time.time()

        # ================================================= GPU
        if gpu:
            # Element-wise stiffness coefficients
            stiff_gpu = Emin + (xPhys_gpu.ravel(order="F") ** penal) * (E0 - Emin)
            elem_vals_gpu = cp.kron(stiff_gpu, KE_gpu.ravel())  # 576×nele

            # Scatter-add into CSR.data
            K_gpu.data.fill(0.0)
            cp.add.at(K_gpu.data, dup2uniq_gpu, elem_vals_gpu)

            # Solve
            Kff_gpu = K_gpu[freedofs0_gpu, :][:, freedofs0_gpu]
            Uf_gpu = solver_func(Kff_gpu, F_gpu[freedofs0_gpu])
            U_gpu.fill(0)
            U_gpu[freedofs0_gpu] = Uf_gpu

            # Compliance & sensitivities
            ce_flat_gpu = element_compliance(U_gpu, cp.asarray(edofMat), KE_gpu)
            ce_gpu = ce_flat_gpu.reshape(nely, nelx, nelz, order="F")
            c = cp.sum((Emin + xPhys_gpu ** penal * (E0 - Emin)) * ce_gpu).item()

            dc_gpu = -penal * (E0 - Emin) * xPhys_gpu ** (penal - 1) * ce_gpu
            dv_gpu = cp.ones_like(xPhys_gpu)

            dc_gpu = apply_filter(H_gpu, dc_gpu, Hs_gpu, xPhys_gpu.shape, use_gpu=True)
            dv_gpu = apply_filter(H_gpu, dv_gpu, Hs_gpu, xPhys_gpu.shape, use_gpu=True)
            dc_gpu[obstacle_gpu | protected_gpu] = dv_gpu[obstacle_gpu | protected_gpu] = 0.0
            dv_gpu += 1e-9

            xnew_gpu, change = optimality_criteria_update(
                x_gpu,
                dc_gpu,
                dv_gpu,
                volfrac,
                H_gpu,
                Hs_gpu,
                nele,
                obstacle_gpu,
                protected_gpu,
                design_nele,
                use_gpu=True,
            )
            xnew_gpu[protected_gpu] = 1.0
            xnew_gpu[obstacle_gpu] = 0.0
            xPhys_gpu = apply_filter(H_gpu, xnew_gpu, Hs_gpu, xnew_gpu.shape, use_gpu=True)
            xnew_gpu[protected_gpu] = 1.0
            xPhys_gpu[obstacle_gpu] = 0.0
            x_gpu = xnew_gpu

            free_mask = (~obstacle_gpu) & (~protected_gpu)
            current_vol = cp.mean(xPhys_gpu[~(obstacle_gpu | protected_gpu)]).item()

        # ================================================= CPU
        else:
            stiff = Emin + (xPhys.ravel(order="F") ** penal) * (E0 - Emin)
            elem_vals = np.kron(stiff, KE.ravel())
            K.data[:] = 0.0
            np.add.at(K.data, dup2uniq, elem_vals)

            Kff = K[freedofs0, :][:, freedofs0]
            Uf = solver_func(Kff, F[freedofs0])
            U = np.zeros(ndof)
            U[freedofs0] = Uf

            ce_flat = element_compliance(U, edofMat, KE)
            ce = ce_flat.reshape(nely, nelx, nelz, order="F")
            c = ((Emin + xPhys ** penal * (E0 - Emin)) * ce).sum()

            dc = -penal * (E0 - Emin) * xPhys ** (penal - 1) * ce
            dv = np.ones_like(xPhys)
            dc = (H * (dc.ravel(order="F") / Hs)).reshape((nely, nelx, nelz), order="F")
            dv = (H * (dv.ravel(order="F") / Hs)).reshape((nely, nelx, nelz), order="F")
            dc[obstacle_mask | protected_zone_mask] = dv[obstacle_mask | protected_zone_mask] = 0.0
            dv += 1e-9

            xnew, change = optimality_criteria_update(
                x,
                dc,
                dv,
                volfrac,
                H,
                Hs,
                nele,
                obstacle_mask,
                protected_zone_mask,
                design_nele,
                use_gpu=False,
            )
            xnew[protected_zone_mask] = 1.0
            xnew[obstacle_mask] = 0.0
            xPhys = (H * xnew.ravel(order="F") / Hs).reshape((nely, nelx, nelz), order="F")
            xnew[protected_zone_mask] = 1.0
            xPhys[obstacle_mask] = 0.0
            x = xnew

            free_mask = (~obstacle_mask) & (~protected_zone_mask)
            current_vol = xPhys[free_mask].mean()

        # ------------------------------------------------ logging / history
        c_delta, c_prev = c - c_prev, c
        iter_t = time.time() - t0

        if save_history and (loop % history_frequency == 0 or change <= tolx):
            history["density_history"].append(
                cp.asnumpy(xPhys_gpu) if gpu else xPhys.copy()
            )
            history["iteration_history"].append(loop)
            history["compliance_history"].append(c)

        logger.info(
            f"Iter {loop:4d}: Obj={c:11.4e}, ΔObj={c_delta:11.4e}, "
            f"Vol={current_vol:6.3f}, change={change:6.3f}, "
            f"time={iter_t:5.2f}s"
        )

    # ─────────────────────── final output
    failure_force = None

    if gpu:
        final_xPhys = cp.asnumpy(xPhys_gpu)
    else:
        final_xPhys = xPhys

    # Re-evaluate compliance on the returned final geometry so the reported
    # final objective matches the actual exported/returned design state.
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
        return final_xPhys, history, final_compliance, failure_force
    return final_xPhys, None, final_compliance, failure_force