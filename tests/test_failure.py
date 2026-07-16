import numpy as np
import pytest

from pytopo3d.analysis.failure import (
    FAILURE_MODE_LABELS,
    critical_failure_location,
    evaluate_gauss_maximum_stress,
    evaluate_maximum_stress,
    maximum_stress_components,
    predicted_failure_load,
)
from pytopo3d.utils.config_loader import MaterialStrength


STRENGTH = MaterialStrength(
    X_t=10.0,
    X_c=20.0,
    Y_t=30.0,
    Y_c=40.0,
    Z_t=50.0,
    Z_c=60.0,
    S_xy=70.0,
    S_yz=80.0,
    S_zx=90.0,
)


@pytest.mark.parametrize(
    ("component", "stress", "mode"),
    [
        (0, 10.0, "X tension"),
        (0, -20.0, "X compression"),
        (1, 30.0, "Y tension"),
        (1, -40.0, "Y compression"),
        (2, 50.0, "Z tension"),
        (2, -60.0, "Z compression"),
        (3, 70.0, "XY shear"),
        (3, -70.0, "XY shear"),
        (4, 80.0, "YZ shear"),
        (4, -80.0, "YZ shear"),
        (5, 90.0, "ZX shear"),
        (5, -90.0, "ZX shear"),
    ],
)
def test_each_strength_produces_unit_index_and_correct_mode(component, stress, mode):
    stress_state = np.zeros(6)
    stress_state[component] = stress

    result = evaluate_maximum_stress(stress_state, STRENGTH)

    assert result.failure_index == pytest.approx(1.0)
    assert result.critical_mode == mode


def test_tension_and_compression_select_distinct_strengths():
    tension = evaluate_maximum_stress(np.array([5.0, 0, 0, 0, 0, 0]), STRENGTH)
    compression = evaluate_maximum_stress(
        np.array([-5.0, 0, 0, 0, 0, 0]), STRENGTH
    )

    assert tension.failure_index == pytest.approx(0.5)
    assert compression.failure_index == pytest.approx(0.25)
    assert tension.critical_mode == "X tension"
    assert compression.critical_mode == "X compression"


def test_doubling_stress_doubles_components_and_failure_index():
    stress = np.array([2.0, -8.0, 15.0, -14.0, 24.0, -36.0])
    result = evaluate_maximum_stress(stress, STRENGTH)
    doubled = evaluate_maximum_stress(2.0 * stress, STRENGTH)

    np.testing.assert_allclose(
        doubled.failure_components, 2.0 * result.failure_components
    )
    assert doubled.failure_index == pytest.approx(2.0 * result.failure_index)
    assert doubled.critical_mode == result.critical_mode


def test_gauss_failure_reduces_to_element_maximum_and_mode():
    stress = np.zeros((2, 8, 6))
    stress[0, 3, 1] = -20.0
    stress[0, 5, 3] = 35.0
    stress[1, 2, 5] = -90.0

    result = evaluate_gauss_maximum_stress(stress, STRENGTH)

    assert result.failure_components_gauss.shape == (2, 8, 6)
    assert result.failure_index_gauss.shape == (2, 8)
    np.testing.assert_allclose(result.failure_index_element, (0.5, 1.0))
    np.testing.assert_array_equal(result.critical_gauss_point_element, (3, 2))
    np.testing.assert_array_equal(
        result.critical_failure_mode_element,
        ("Y compression", "ZX shear"),
    )


def test_critical_location_respects_element_eligibility():
    stress = np.zeros((3, 8, 6))
    stress[0, 6, 0] = 20.0
    stress[1, 4, 4] = 40.0
    stress[2, 2, 2] = -45.0
    result = evaluate_gauss_maximum_stress(stress, STRENGTH)

    all_elements = critical_failure_location(result)
    design_region = critical_failure_location(
        result, eligible_elements=np.array([False, True, True])
    )

    assert all_elements.failure_index == pytest.approx(2.0)
    assert (all_elements.element, all_elements.gauss_point, all_elements.mode) == (
        0,
        6,
        "X tension",
    )
    assert design_region.failure_index == pytest.approx(0.75)
    assert (
        design_region.element,
        design_region.gauss_point,
        design_region.mode,
    ) == (2, 2, "Z compression")


def test_predicted_failure_load_is_reference_load_independent():
    assert predicted_failure_load(200.0, 0.4) == pytest.approx(500.0)
    assert predicted_failure_load(100.0, 0.2) == pytest.approx(500.0)


def test_zero_failure_index_has_unbounded_linear_prediction():
    assert predicted_failure_load(100.0, 0.0) == np.inf


def test_zero_stress_uses_deterministic_x_tension_mode():
    result = evaluate_maximum_stress(np.zeros(6), STRENGTH)

    assert result.failure_index == 0.0
    assert result.critical_mode == FAILURE_MODE_LABELS[0]


def test_maximum_stress_accepts_and_validates_strength_mapping():
    components = maximum_stress_components(
        np.zeros(6),
        STRENGTH.as_dict(),
    )
    np.testing.assert_array_equal(components, np.zeros(6))


@pytest.mark.parametrize(
    ("function", "value", "message"),
    [
        (maximum_stress_components, np.zeros(5), "dimension 6"),
        (evaluate_gauss_maximum_stress, np.zeros((1, 7, 6)), "shape"),
    ],
)
def test_failure_evaluation_rejects_invalid_stress_shape(function, value, message):
    with pytest.raises(ValueError, match=message):
        function(value, STRENGTH)


@pytest.mark.parametrize(
    ("reference_load", "failure_index"),
    [(0.0, 1.0), (-1.0, 1.0), (np.inf, 1.0), (1.0, -1.0), (1.0, np.nan)],
)
def test_predicted_failure_load_rejects_invalid_inputs(reference_load, failure_index):
    with pytest.raises(ValueError):
        predicted_failure_load(reference_load, failure_index)

