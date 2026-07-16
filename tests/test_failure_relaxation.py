import numpy as np
import pytest

from pytopo3d.analysis.failure import (
    evaluate_gauss_maximum_stress,
    evaluate_relaxed_gauss_maximum_stress,
)
from pytopo3d.analysis.stress import relax_gauss_stress
from pytopo3d.utils.config_loader import MaterialStrength


STRENGTH = MaterialStrength(
    X_t=100.0,
    X_c=80.0,
    Y_t=50.0,
    Y_c=40.0,
    Z_t=25.0,
    Z_c=20.0,
    S_xy=30.0,
    S_yz=15.0,
    S_zx=10.0,
)


def _stress_field(number_of_elements=3):
    stress = np.zeros((number_of_elements, 8, 6))
    stress[:, :, 0] = 60.0
    stress[:, :, 4] = -3.0
    return stress


def test_full_density_relaxation_preserves_solid_stress_and_failure():
    solid_stress = _stress_field(2)
    relaxed_stress = relax_gauss_stress(solid_stress, np.ones(2))
    relaxed_failure = evaluate_relaxed_gauss_maximum_stress(
        solid_stress,
        np.ones(2),
        STRENGTH,
    )
    exact_failure = evaluate_gauss_maximum_stress(solid_stress, STRENGTH)

    np.testing.assert_array_equal(relaxed_stress, solid_stress)
    np.testing.assert_array_equal(
        relaxed_failure.failure_index_gauss,
        exact_failure.failure_index_gauss,
    )


def test_default_q_scales_gray_stress_and_failure_homogeneously():
    solid_stress = _stress_field(1)
    density = np.array([0.25])

    relaxed_stress = relax_gauss_stress(solid_stress, density)
    relaxed_failure = evaluate_relaxed_gauss_maximum_stress(
        solid_stress,
        density,
        STRENGTH,
    )
    exact_failure = evaluate_gauss_maximum_stress(solid_stress, STRENGTH)

    np.testing.assert_allclose(relaxed_stress, 0.5 * solid_stress)
    np.testing.assert_allclose(
        relaxed_failure.failure_index_gauss,
        0.5 * exact_failure.failure_index_gauss,
    )


def test_gray_connection_is_not_suppressed_by_simp_stiffness_exponent():
    solid_stress = _stress_field(1)
    density = np.array([0.1])

    q_relaxed = relax_gauss_stress(solid_stress, density, exponent=0.5)
    simp_scaled = density[0] ** 3.0 * solid_stress

    assert np.max(np.abs(q_relaxed)) > 300.0 * np.max(np.abs(simp_scaled))
    assert np.max(np.abs(q_relaxed)) < np.max(np.abs(solid_stress))


def test_near_void_is_relaxed_and_q_sweep_is_monotone_and_bounded():
    solid_stress = _stress_field(3)
    density = np.array([1.0e-12, 0.25, 1.0])
    maxima = []
    for exponent in (0.25, 0.5, 1.0):
        result = evaluate_relaxed_gauss_maximum_stress(
            solid_stress,
            density,
            STRENGTH,
            relaxation_exponent=exponent,
        )
        maxima.append(result.failure_index_element)

    assert maxima[0][0] < 0.01 * maxima[0][2]
    np.testing.assert_array_less(maxima[2][1], maxima[1][1])
    np.testing.assert_array_less(maxima[1][1], maxima[0][1])
    assert maxima[0][2] == maxima[1][2] == maxima[2][2]


def test_density_grid_uses_fortran_element_order():
    solid_stress = _stress_field(4)
    density_grid = np.array([[[0.01], [0.09]], [[0.04], [0.16]]])
    relaxed = relax_gauss_stress(solid_stress, density_grid, exponent=0.5)

    expected_scale = np.sqrt(density_grid.ravel(order="F"))
    np.testing.assert_allclose(relaxed[:, 0, 0], 60.0 * expected_scale)


@pytest.mark.parametrize(
    ("stress", "density", "exponent", "message"),
    [
        (np.zeros((1, 7, 6)), np.ones(1), 0.5, "shape"),
        (_stress_field(2), np.ones(1), 0.5, "2 element values"),
        (_stress_field(1), np.array([-0.1]), 0.5, "[0, 1]"),
        (_stress_field(1), np.array([1.1]), 0.5, "[0, 1]"),
        (_stress_field(1), np.ones(1), -0.1, "nonnegative"),
    ],
)
def test_relaxation_rejects_invalid_inputs(stress, density, exponent, message):
    with pytest.raises(ValueError, match=message):
        relax_gauss_stress(stress, density, exponent=exponent)
