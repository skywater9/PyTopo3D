import numpy as np
import pytest

from pytopo3d.utils.mma_update import mma_update


def test_mma_model_is_exactly_tangent_at_current_design():
    x = np.array([0.3, 0.7])
    objective = 12.0
    objective_gradient = np.array([-4.0, 2.0])
    constraints = np.array([0.2, -0.4])
    constraint_gradients = np.array([[1.5, -0.5], [-2.0, 3.0]])

    result = mma_update(
        x,
        objective,
        objective_gradient,
        constraints,
        constraint_gradients,
        objective_reference=objective,
    )
    diagnostics = result.diagnostics

    assert diagnostics.model_objective_at_current == pytest.approx(1.0)
    np.testing.assert_allclose(
        diagnostics.model_objective_gradient_at_current,
        objective_gradient / objective,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        diagnostics.model_constraint_values_at_current,
        constraints,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        diagnostics.model_constraint_gradients_at_current,
        constraint_gradients,
        rtol=0.0,
        atol=1.0e-14,
    )


def test_update_respects_global_and_move_bounds():
    x = np.array([0.05, 0.5, 0.95])
    result = mma_update(
        x,
        1.0,
        np.array([-1.0, 1.0, -1.0]),
        np.array([-1.0]),
        np.zeros((1, 3)),
        move_limit=0.1,
    )

    assert np.all(result.x_new >= 0.0)
    assert np.all(result.x_new <= 1.0)
    assert np.all(np.abs(result.x_new - x) <= 0.1 + 1.0e-14)


def test_same_direction_expands_and_reversal_contracts_asymptotes():
    def update(value, state):
        return mma_update(
            np.array([value]),
            1.0,
            np.array([-1.0]),
            np.empty(0),
            np.empty((0, 1)),
            state=state,
        )

    first = update(0.4, None)
    second = update(0.5, first.state)
    third = update(0.6, second.state)
    fourth = update(0.55, third.state)
    expanded = third.state.upper_asymptotes[0] - 0.6
    contracted = fourth.state.upper_asymptotes[0] - 0.55

    assert expanded > second.state.upper_asymptotes[0] - 0.5
    assert contracted < expanded


def test_two_constraint_toy_problem_converges_with_active_constraints():
    x = np.array([0.6, 0.6])
    state = None
    for _ in range(100):
        objective = (x[0] - 0.5) ** 2 + (x[1] - 0.4) ** 2
        objective_gradient = 2.0 * (x - np.array([0.5, 0.4]))
        constraints = np.array([1.1 - x.sum(), x[0] - 0.3])
        constraint_gradients = np.array([[-1.0, -1.0], [1.0, 0.0]])
        result = mma_update(
            x,
            objective,
            objective_gradient,
            constraints,
            constraint_gradients,
            move_limit=0.1,
            state=state,
            objective_reference=0.1 if state is None else None,
        )
        if np.max(np.abs(result.x_new - x)) < 1.0e-8:
            x = result.x_new
            state = result.state
            break
        x = result.x_new
        state = result.state

    constraints = np.array([1.1 - x.sum(), x[0] - 0.3])
    assert np.max(constraints) < 2.0e-4
    np.testing.assert_allclose(x, np.array([0.3, 0.8]), atol=2.0e-3)
    assert np.all(state.dual_multipliers > 1.0e-4)


def test_loose_constraint_has_negligible_multiplier():
    result = mma_update(
        np.array([0.5]),
        1.0,
        np.array([-1.0]),
        np.array([-10.0]),
        np.array([[0.0]]),
    )
    assert result.state.dual_multipliers[0] < 1.0e-8


def test_fixed_reference_makes_objective_scaling_invariant():
    arguments = {
        "x": np.array([0.35, 0.65]),
        "constraint_values": np.array([0.1]),
        "constraint_gradients": np.array([[1.0, 1.0]]),
    }
    base = mma_update(
        objective_value=2.0,
        objective_gradient=np.array([-3.0, 4.0]),
        objective_reference=2.0,
        **arguments,
    )
    scaled = mma_update(
        objective_value=2.0e12,
        objective_gradient=np.array([-3.0e12, 4.0e12]),
        objective_reference=2.0e12,
        **arguments,
    )
    np.testing.assert_allclose(base.x_new, scaled.x_new, rtol=0.0, atol=1.0e-12)


def test_infeasible_move_box_activates_elastic_slack():
    result = mma_update(
        np.array([0.5]),
        1.0,
        np.array([0.0]),
        np.array([1.0]),
        np.array([[0.0]]),
        move_limit=0.01,
    )

    assert result.diagnostics.slack_variables[0] > 0.5
    assert not result.diagnostics.feasible
    assert result.diagnostics.max_approximate_constraint_violation > 0.5


def test_successful_scipy_status_does_not_override_bad_kkt_residual():
    result = mma_update(
        np.array([0.60437793, 0.04336467, 0.43087486]),
        166.894905,
        np.array([-0.1756182, -0.03089292, 0.12709397]),
        np.array([1.40573471, -0.97320811]),
        np.array(
            [
                [3591.774, 20750.843, -10681.233],
                [-40216.523, 28898.607, -82078.468],
            ]
        ),
        move_limit=0.05,
    )

    assert result.diagnostics.dual_status == 0
    assert result.diagnostics.subproblem_kkt_residual > 0.1
    assert result.diagnostics.dual_success is False


def test_large_vector_update_has_linear_sized_state():
    size = 20_000
    result = mma_update(
        np.full(size, 0.5),
        1.0,
        -np.ones(size),
        np.array([0.0]),
        np.ones((1, size)) / size,
        move_limit=0.05,
    )

    assert result.x_new.shape == (size,)
    assert result.state.lower_asymptotes.shape == (size,)
    assert result.state.upper_asymptotes.shape == (size,)
    assert result.diagnostics.model_constraint_gradients_at_current.shape == (1, size)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"x": np.empty(0)}, "nonempty"),
        ({"objective_gradient": np.ones(3)}, "match x"),
        ({"constraint_gradients": np.ones((2, 2))}, "shape"),
        ({"lower_bounds": 1.0, "upper_bounds": 0.0}, "strictly greater"),
    ],
)
def test_invalid_inputs_are_rejected(kwargs, message):
    arguments = {
        "x": np.array([0.4, 0.6]),
        "objective_value": 1.0,
        "objective_gradient": np.ones(2),
        "constraint_values": np.array([0.0]),
        "constraint_gradients": np.ones((1, 2)),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        mma_update(**arguments)
