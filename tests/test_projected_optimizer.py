import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pytopo3d.core.compliance import element_compliance
from pytopo3d.core.optimizer import evaluate_fixed_geometry_compliance, top3d
from pytopo3d.utils.assembly import (
    build_edof,
    build_force_field,
    build_force_vector,
    build_support_mask,
    build_supports,
)
from pytopo3d.utils.filter import (
    apply_density_filter_chain_rule,
    build_filter,
    build_physical_density,
)
from pytopo3d.utils.stiffness import lk_H8
from pytopo3d.utils.boundary import create_bc_visualization_arrays_from_masks


def _solve_projected_design(x, H, Hs, beta, eta, nelx, nely, nelz):
    shape = (nely, nelx, nelz)
    fixed = np.zeros(shape, dtype=bool)
    _, rho_physical, projection_derivative = build_physical_density(
        x,
        H,
        Hs,
        beta=beta,
        eta=eta,
        protected_solid=fixed,
        protected_void=fixed,
        xp=np,
    )

    KE = lk_H8(elem_size=1.0)
    edof_mat, iK, jK = build_edof(nelx, nely, nelz)
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    F = build_force_vector(nelx, nely, nelz, ndof)
    free_dofs, _ = build_supports(nelx, nely, nelz, ndof)

    stiffness_scale = 1.0e-9 + (1.0 - 1.0e-9) * (
        rho_physical.ravel(order="F") ** 3.0
    )
    element_values = np.kron(stiffness_scale, KE.ravel())
    K = sp.coo_matrix(
        (element_values, (iK - 1, jK - 1)),
        shape=(ndof, ndof),
    ).tocsr()
    U = np.zeros(ndof)
    U[free_dofs] = spla.spsolve(
        K[free_dofs, :][:, free_dofs],
        F[free_dofs],
    )

    element_energy = element_compliance(U, edof_mat, KE).reshape(
        shape,
        order="F",
    )
    compliance = float(
        np.sum(
            (1.0e-9 + (1.0 - 1.0e-9) * rho_physical**3.0)
            * element_energy
        )
    )
    dc_drho = (
        -3.0
        * (1.0 - 1.0e-9)
        * rho_physical**2.0
        * element_energy
    )
    dc_dx = apply_density_filter_chain_rule(
        dc_drho,
        projection_derivative,
        H,
        Hs,
    )
    return compliance, dc_dx


def test_full_projected_compliance_gradient_matches_finite_difference():
    nelx, nely, nelz = 2, 2, 2
    shape = (nely, nelx, nelz)
    H, Hs = build_filter(nelx, nely, nelz, rmin=1.5)
    x = np.linspace(0.35, 0.65, np.prod(shape)).reshape(shape, order="F")
    beta = 4.0
    eta = 0.5

    _, analytical = _solve_projected_design(
        x, H, Hs, beta, eta, nelx, nely, nelz
    )
    epsilon = 1.0e-6
    for index in (0, 3, 7):
        upper = x.ravel(order="F").copy()
        lower = x.ravel(order="F").copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        upper_value, _ = _solve_projected_design(
            upper.reshape(shape, order="F"),
            H,
            Hs,
            beta,
            eta,
            nelx,
            nely,
            nelz,
        )
        lower_value, _ = _solve_projected_design(
            lower.reshape(shape, order="F"),
            H,
            Hs,
            beta,
            eta,
            nelx,
            nely,
            nelz,
        )
        finite_difference = (upper_value - lower_value) / (2.0 * epsilon)
        assert analytical.ravel(order="F")[index] == pytest.approx(
            finite_difference,
            rel=2.0e-4,
            abs=1.0e-5,
        )


def test_symmetric_tensile_update_preserves_end_to_end_symmetry():
    nelx, nely, nelz = 4, 12, 2
    force_field = build_force_field(
        nelx,
        nely,
        nelz,
        0,
        nelx,
        nely - 1,
        nely,
        0,
        nelz,
        0.0,
        1.0,
        0.0,
    )
    support_mask = build_support_mask(
        nelx,
        nely,
        nelz,
        0,
        nelx,
        0,
        1,
        0,
        nelz,
    )
    protected = np.zeros((nely, nelx, nelz), dtype=bool)
    protected[:3] = True
    protected[-3:] = True

    rho_physical, _, _, _ = top3d(
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        volfrac=0.5,
        penal=3.0,
        rmin=2.0,
        disp_thres=0.5,
        force_field=force_field,
        support_mask=support_mask,
        protected_zone_mask=protected,
        tolx=1.0e-12,
        maxloop=1,
        beta_schedule=(1.0,),
    )

    free_density = rho_physical[3:-3]
    np.testing.assert_allclose(
        free_density,
        free_density[::-1],
        rtol=0.0,
        atol=1.0e-3,
    )


def test_continuation_returns_one_physical_field_and_records_metrics():
    nelx = nely = nelz = 2
    shape = (nely, nelx, nelz)
    support_mask = np.zeros(shape, dtype=bool)
    support_mask[:, 0, :] = True
    force_field = np.zeros(shape + (3,))
    force_field[0, 1, 0, 0] = 1.0
    obstacle_mask = np.zeros(shape, dtype=bool)
    obstacle_mask[1, 1, 1] = True
    diagnostics = {}

    rho_physical, history, compliance, _ = top3d(
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        volfrac=0.5,
        penal=3.0,
        rmin=1.0,
        disp_thres=0.5,
        force_field=force_field,
        support_mask=support_mask,
        obstacle_mask=obstacle_mask,
        tolx=1.0e-12,
        maxloop=1,
        save_history=True,
        history_frequency=1,
        beta_schedule=(1.0, 2.0),
        diagnostics_out=diagnostics,
    )

    protected_by_boundary = support_mask | np.any(force_field != 0.0, axis=-1)
    assert np.all(rho_physical[protected_by_boundary] == 1.0)
    assert rho_physical[obstacle_mask].item() == 0.0
    np.testing.assert_allclose(history["density_history"][-1], rho_physical)
    assert history["iteration_history"] == [0, 1, 2]
    assert history["beta_history"] == [1.0, 2.0, 2.0]
    assert history["compliance_history"][-1] == pytest.approx(compliance)
    for density, stored_compliance in zip(
        history["density_history"], history["compliance_history"]
    ):
        recomputed = evaluate_fixed_geometry_compliance(
            xPhys=density,
            penal=3.0,
            force_field=force_field,
            support_mask=support_mask,
            obstacle_mask=obstacle_mask,
        )
        assert stored_compliance == pytest.approx(recomputed, rel=1.0e-9)

    assert diagnostics["projection_enabled"] is True
    assert diagnostics["projection_beta_schedule"] == [1.0, 2.0]
    assert diagnostics["projection_beta"] == 2.0
    assert diagnostics["projection_total_iterations"] == 2
    assert diagnostics["projection_converged"] is False
    for stage in diagnostics["projection_stage_summaries"]:
        assert stage["converged"] is False
        assert stage["volume_fraction_error"] == pytest.approx(0.0, abs=1.0e-7)
    assert diagnostics["physical_density_fraction"] == pytest.approx(
        0.5,
        abs=1.0e-7,
    )
    assert diagnostics["projected_compliance"] == pytest.approx(compliance)
    expected_delta = (
        diagnostics["binary_compliance"] - diagnostics["projected_compliance"]
    ) / diagnostics["projected_compliance"]
    assert diagnostics["binary_compliance_delta"] == pytest.approx(expected_delta)


def test_default_boundary_attachments_are_visualized_and_protected():
    nelx, nely, nelz = 3, 2, 2
    empty_support = np.zeros((nely, nelx, nelz), dtype=bool)
    loads, constraints = create_bc_visualization_arrays_from_masks(
        nelx,
        nely,
        nelz,
        3 * (nelx + 1) * (nely + 1) * (nelz + 1),
        None,
        empty_support,
    )
    assert np.all(loads[:, -1, 0] == 1.0)
    assert np.all(constraints[:, 0, :] == 1.0)

    diagnostics = {}
    rho_physical, _, _, _ = top3d(
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        volfrac=0.5,
        penal=3.0,
        rmin=1.5,
        disp_thres=0.5,
        support_mask=empty_support,
        maxloop=1,
        beta_schedule=(1.0,),
        diagnostics_out=diagnostics,
    )
    assert np.all(rho_physical[:, 0, :] == 1.0)
    assert np.all(rho_physical[:, -1, 0] == 1.0)
    stage = diagnostics["projection_stage_summaries"][0]
    assert stage["minimum_achievable_physical_fraction"] > 0.0
    assert stage["target_within_achievable_range"] is True
    assert diagnostics["physical_density_fraction"] == pytest.approx(
        0.5,
        abs=1.0e-7,
    )


def test_infeasible_projected_volume_target_is_reported_before_fea():
    with pytest.raises(ValueError, match="minimum achievable"):
        top3d(
            nelx=2,
            nely=2,
            nelz=2,
            volfrac=0.3,
            penal=3.0,
            rmin=1.5,
            disp_thres=0.5,
            maxloop=1,
            beta_schedule=(1.0,),
        )
