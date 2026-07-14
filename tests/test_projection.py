import numpy as np
import pytest
import scipy.sparse as sp

from pytopo3d.utils.filter import (
    apply_density_filter_chain_rule,
    build_filter,
    build_physical_density,
    heaviside_projection,
)
from pytopo3d.utils.oc_update import optimality_criteria_update_projected


def test_projection_derivative_matches_central_difference():
    rho = np.linspace(0.01, 0.99, 50)
    beta = 8.0
    eta = 0.5
    epsilon = 1.0e-7

    _, derivative = heaviside_projection(rho, beta=beta, eta=eta, xp=np)
    upper, _ = heaviside_projection(rho + epsilon, beta, eta, np)
    lower, _ = heaviside_projection(rho - epsilon, beta, eta, np)
    finite_difference = (upper - lower) / (2.0 * epsilon)

    np.testing.assert_allclose(
        derivative,
        finite_difference,
        rtol=1.0e-5,
        atol=1.0e-7,
    )


@pytest.mark.parametrize(
    ("beta", "eta"),
    [
        (0.0, 0.5),
        (-1.0, 0.5),
        (np.nan, 0.5),
        (np.inf, 0.5),
        (1.0, 0.0),
        (1.0, 1.0),
    ],
)
def test_projection_rejects_invalid_parameters(beta, eta):
    with pytest.raises(ValueError):
        heaviside_projection(np.array([0.5]), beta=beta, eta=eta)


def test_physical_density_enforces_fixed_regions_and_fortran_order():
    shape = (2, 2, 1)
    x = np.array([[[0.1], [0.3]], [[0.2], [0.4]]])
    # The identity makes the expected filtered vector exactly x.ravel(F).
    H = sp.eye(x.size, format="csr")
    Hs = np.ones(x.size)
    protected_solid = np.zeros(shape, dtype=bool)
    protected_void = np.zeros(shape, dtype=bool)
    protected_solid[0, 1, 0] = True
    protected_void[1, 0, 0] = True

    rho_filtered, rho_physical, derivative = build_physical_density(
        x,
        H,
        Hs,
        beta=4.0,
        eta=0.5,
        protected_solid=protected_solid,
        protected_void=protected_void,
        xp=np,
    )

    np.testing.assert_array_equal(
        rho_filtered.ravel(order="F"),
        x.ravel(order="F"),
    )
    assert rho_physical[protected_solid].item() == 1.0
    assert rho_physical[protected_void].item() == 0.0
    assert derivative[protected_solid].item() == 0.0
    assert derivative[protected_void].item() == 0.0


def test_protected_design_values_filter_neighbors_but_clamp_physical_response():
    shape = (1, 3, 1)
    x = np.array([[[1.0], [0.2], [0.0]]])
    protected_solid = np.zeros(shape, dtype=bool)
    protected_void = np.zeros(shape, dtype=bool)
    protected_solid[0, 0, 0] = True
    protected_void[0, 2, 0] = True

    H = sp.csr_matrix(
        np.array(
            [
                [2.0, 1.0, 0.0],
                [1.0, 2.0, 1.0],
                [0.0, 1.0, 2.0],
            ]
        )
    )
    Hs = np.asarray(H.sum(axis=1)).ravel()

    rho_filtered, rho_physical, derivative = build_physical_density(
        x,
        H,
        Hs,
        beta=4.0,
        eta=0.5,
        protected_solid=protected_solid,
        protected_void=protected_void,
        xp=np,
    )

    assert rho_filtered[0, 1, 0] == pytest.approx((1.0 + 2.0 * 0.2 + 0.0) / 4.0)
    assert rho_physical[protected_solid].item() == 1.0
    assert rho_physical[protected_void].item() == 0.0
    assert derivative[protected_solid].item() == 0.0
    assert derivative[protected_void].item() == 0.0
    assert derivative[0, 1, 0] > 0.0


def test_density_filter_chain_rule_uses_transpose():
    shape = (2, 2, 1)
    x = np.array([[[0.25], [0.45]], [[0.35], [0.65]]])
    # Deliberately nonsymmetric: an H-vs-H.T mistake cannot pass this test.
    H = sp.csr_matrix(
        np.array(
            [
                [2.0, 1.0, 0.0, 0.0],
                [0.2, 2.0, 0.8, 0.0],
                [0.0, 0.4, 2.0, 0.6],
                [0.5, 0.0, 0.0, 2.0],
            ]
        )
    )
    Hs = np.asarray(H.sum(axis=1)).ravel()
    fixed = np.zeros(shape, dtype=bool)
    weights = np.array([[[1.0], [0.4]], [[-0.2], [1.7]]])

    def objective(candidate):
        _, rho, _ = build_physical_density(
            candidate,
            H,
            Hs,
            beta=3.0,
            eta=0.5,
            protected_solid=fixed,
            protected_void=fixed,
            xp=np,
        )
        return float(np.sum(weights * rho))

    _, _, projection_derivative = build_physical_density(
        x,
        H,
        Hs,
        beta=3.0,
        eta=0.5,
        protected_solid=fixed,
        protected_void=fixed,
        xp=np,
    )
    analytical = apply_density_filter_chain_rule(
        weights,
        projection_derivative,
        H,
        Hs,
    )

    epsilon = 1.0e-7
    finite_difference = np.empty(x.size)
    for index in range(x.size):
        upper = x.copy().ravel(order="F")
        lower = x.copy().ravel(order="F")
        upper[index] += epsilon
        lower[index] -= epsilon
        finite_difference[index] = (
            objective(upper.reshape(shape, order="F"))
            - objective(lower.reshape(shape, order="F"))
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        analytical.ravel(order="F"),
        finite_difference,
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_filter_matches_fortran_order_reference_on_rectangular_grid():
    nelx, nely, nelz = 3, 2, 4
    rmin = 1.6
    H, Hs = build_filter(nelx, nely, nelz, rmin)

    coordinates = np.array(
        [
            (ix, iy, iz)
            for iz in range(nelz)
            for ix in range(nelx)
            for iy in range(nely)
        ],
        dtype=float,
    )
    distances = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :],
        axis=2,
    )
    expected = np.maximum(0.0, rmin - distances)
    expected[expected <= 1.0e-9] = 0.0

    np.testing.assert_allclose(H.toarray(), expected)
    np.testing.assert_allclose(Hs, expected.sum(axis=1))
    np.testing.assert_allclose(H.diagonal(), rmin)


@pytest.mark.parametrize("sensitivity_scale", (1.0, 1.0e-18, 1.0e18))
def test_projected_oc_preserves_free_volume_and_fixed_regions(
    sensitivity_scale,
):
    shape = (1, 6, 1)
    x = np.full(shape, 0.5)
    protected_solid = np.zeros(shape, dtype=bool)
    protected_void = np.zeros(shape, dtype=bool)
    protected_solid[0, 0, 0] = True
    protected_void[0, -1, 0] = True
    free_mask = ~(protected_solid | protected_void)
    x[protected_solid] = 1.0
    x[protected_void] = 0.0
    H = sp.eye(x.size, format="csr")
    Hs = np.ones(x.size)

    _, _, projection_derivative = build_physical_density(
        x,
        H,
        Hs,
        beta=4.0,
        eta=0.5,
        protected_solid=protected_solid,
        protected_void=protected_void,
        xp=np,
    )
    dc_dx = apply_density_filter_chain_rule(
        -sensitivity_scale * np.ones(shape),
        projection_derivative,
        H,
        Hs,
    )
    dv_dx = apply_density_filter_chain_rule(
        free_mask.astype(float), projection_derivative, H, Hs
    )
    target_fraction = 0.4

    x_new, rho_new = optimality_criteria_update_projected(
        x=x,
        dc_dx=dc_dx,
        dv_dx=dv_dx,
        move=0.2,
        target_free_volume=target_fraction * int(np.sum(free_mask)),
        H=H,
        Hs=Hs,
        beta=4.0,
        eta=0.5,
        free_mask=free_mask,
        protected_solid=protected_solid,
        protected_void=protected_void,
        xp=np,
    )

    assert x_new[protected_solid].item() == 1.0
    assert x_new[protected_void].item() == 0.0
    assert rho_new[protected_solid].item() == 1.0
    assert rho_new[protected_void].item() == 0.0
    assert np.mean(rho_new[free_mask]) == pytest.approx(
        target_fraction,
        abs=1.0e-7,
    )
