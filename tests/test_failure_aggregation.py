import numpy as np
import pytest

from pytopo3d.analysis.failure_aggregation import (
    aggregate_gauss_failure,
    calibrate_pnorm_correction,
    corrected_pnorm_aggregate,
    corrected_pnorm_gradient,
    pnorm_aggregate,
)


def test_raw_pnorm_and_normalized_pmean_match_definitions():
    values = np.array([0.2, 0.4, 0.8])
    exponent = 4.0

    raw = pnorm_aggregate(values, exponent=exponent)
    normalized = pnorm_aggregate(values, exponent=exponent, normalized=True)

    assert raw == pytest.approx(np.sum(values**exponent) ** (1.0 / exponent))
    assert normalized == pytest.approx(
        np.mean(values**exponent) ** (1.0 / exponent)
    )


def test_normalized_pmean_does_not_grow_for_repeated_constant_samples():
    for sample_count in (1, 8, 1000):
        values = np.full(sample_count, 0.73)
        assert pnorm_aggregate(
            values, exponent=8.0, normalized=True
        ) == pytest.approx(0.73)


def test_increasing_exponent_moves_pmean_toward_exact_maximum():
    values = np.array([0.1, 0.3, 0.6, 1.0])
    aggregates = [
        pnorm_aggregate(values, exponent=exponent, normalized=True)
        for exponent in (2.0, 4.0, 8.0, 16.0)
    ]

    assert aggregates == sorted(aggregates)
    assert all(value < np.max(values) for value in aggregates)
    assert abs(aggregates[-1] - 1.0) < abs(aggregates[0] - 1.0)


def test_calibrated_correction_matches_reference_but_is_frozen_afterward():
    reference = np.array([0.2, 0.4, 0.7, 0.9])
    correction = calibrate_pnorm_correction(reference, exponent=8.0)
    reference_result = corrected_pnorm_aggregate(
        reference,
        exponent=8.0,
        correction_factor=correction,
    )
    perturbed = reference * np.array([1.0, 1.0, 1.0, 1.05])
    perturbed_result = corrected_pnorm_aggregate(
        perturbed,
        exponent=8.0,
        correction_factor=correction,
    )

    assert reference_result.aggregate == pytest.approx(reference_result.exact_max)
    assert perturbed_result.aggregate != pytest.approx(perturbed_result.exact_max)
    assert perturbed_result.aggregate == pytest.approx(
        correction
        * np.mean(perturbed**8.0) ** (1.0 / 8.0)
    )


def test_element_mask_excludes_fixture_spike_without_hiding_separate_exact_data():
    values = np.array([5.0, 0.6, 0.7, 0.8])
    design_eligible = np.array([False, True, True, True])

    result = corrected_pnorm_aggregate(
        values,
        exponent=8.0,
        eligible=design_eligible,
    )

    assert result.exact_max == pytest.approx(0.8)
    assert result.sample_count == 3
    assert result.eligible_weights[0] == 0.0
    assert values.max() == 5.0


def test_aggregate_is_stable_when_two_critical_elements_switch():
    delta = 1.0e-6
    left_critical = np.array([1.0 + delta, 1.0 - delta, 0.4])
    right_critical = np.array([1.0 - delta, 1.0 + delta, 0.4])

    left = corrected_pnorm_aggregate(left_critical, exponent=8.0)
    right = corrected_pnorm_aggregate(right_critical, exponent=8.0)

    assert np.argmax(left_critical) != np.argmax(right_critical)
    assert left.aggregate == pytest.approx(right.aggregate, rel=1e-14)


def test_gauss_failure_uses_maximum_point_per_element_before_global_aggregate():
    failure_gauss = np.zeros((3, 8))
    failure_gauss[0, 4] = 0.2
    failure_gauss[1, 7] = 0.9
    failure_gauss[2, 2] = 0.5

    result = aggregate_gauss_failure(failure_gauss, exponent=8.0)

    np.testing.assert_array_equal(
        result.element_failure_index,
        np.array([0.2, 0.9, 0.5]),
    )
    assert result.exact_max == pytest.approx(0.9)


def test_aggregate_scales_linearly_with_failure_index():
    values = np.array([0.2, 0.3, 0.8])
    base = corrected_pnorm_aggregate(
        values, exponent=8.0, correction_factor=1.1
    )
    scaled = corrected_pnorm_aggregate(
        3.0 * values, exponent=8.0, correction_factor=1.1
    )

    assert scaled.aggregate == pytest.approx(3.0 * base.aggregate)
    assert scaled.exact_max == pytest.approx(3.0 * base.exact_max)


def test_zero_field_has_zero_aggregate_and_unit_calibration():
    values = np.zeros(5)

    assert calibrate_pnorm_correction(values, exponent=8.0) == 1.0
    result = corrected_pnorm_aggregate(values, exponent=8.0)
    assert result.aggregate == 0.0
    assert result.exact_max == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"values": np.array([[1.0]])}, "one-dimensional"),
        ({"values": np.array([-1.0])}, "nonnegative"),
        ({"values": np.ones(2), "exponent": 1.0}, "exponent"),
        ({"values": np.ones(2), "correction_factor": 0.0}, "correction_factor"),
        (
            {"values": np.ones(2), "eligible": np.zeros(2, dtype=bool)},
            "positive eligible",
        ),
    ],
)
def test_invalid_aggregate_inputs_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        corrected_pnorm_aggregate(**kwargs)


def test_large_exponent_is_numerically_stable():
    values = np.array([1.0e100, 0.9e100, 0.5e100])

    result = corrected_pnorm_aggregate(values, exponent=128.0)

    assert np.isfinite(result.aggregate)
    assert result.aggregate <= result.exact_max


def test_corrected_normalized_pmean_gradient_matches_finite_difference():
    values = np.array([0.13, 0.41, 0.72, 0.9])
    weights = np.array([0.0, 2.0, 0.5, 1.5])
    correction = 1.17
    result = corrected_pnorm_aggregate(
        values,
        exponent=8.0,
        correction_factor=correction,
        weights=weights,
    )
    analytical = corrected_pnorm_gradient(result)
    finite_difference = np.zeros_like(values)
    step = 1.0e-6
    for index in range(values.size):
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        plus_value = corrected_pnorm_aggregate(
            plus,
            exponent=8.0,
            correction_factor=correction,
            weights=weights,
        ).aggregate
        minus_value = corrected_pnorm_aggregate(
            minus,
            exponent=8.0,
            correction_factor=correction,
            weights=weights,
        ).aggregate
        finite_difference[index] = (plus_value - minus_value) / (2.0 * step)

    np.testing.assert_allclose(analytical, finite_difference, rtol=2e-9, atol=1e-11)


def test_zero_aggregate_uses_zero_subgradient():
    result = corrected_pnorm_aggregate(np.zeros(4), exponent=8.0)
    np.testing.assert_array_equal(corrected_pnorm_gradient(result), np.zeros(4))


def test_pmean_gradient_is_stable_for_tiny_positive_weight():
    result = corrected_pnorm_aggregate(
        np.array([1.0, 1.0e-5]),
        exponent=128.0,
        weights=np.array([1.0e-320, 1.0]),
    )
    gradient = corrected_pnorm_gradient(result)

    assert np.all(np.isfinite(gradient))
    assert np.all(gradient >= 0.0)
