"""Moving-asymptote updates for one or more inequality constraints.

This is the usual ``a_i = 0`` topology-optimization reduction of Svanberg's
MMA subproblem. It operates only on the free one-dimensional design vector;
callers remain responsible for scattering fixed solid/void entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import OptimizeResult, minimize


@dataclass(frozen=True)
class MMAState:
    """Persistent moving-asymptote and dual warm-start state."""

    iteration: int
    lower_asymptotes: np.ndarray
    upper_asymptotes: np.ndarray
    previous_x: np.ndarray
    previous_previous_x: np.ndarray
    objective_reference: float
    dual_multipliers: np.ndarray


@dataclass(frozen=True)
class MMADiagnostics:
    """Inner convex-subproblem solution and KKT diagnostics."""

    dual_success: bool
    dual_status: int
    dual_message: str
    dual_iterations: int
    dual_function_evaluations: int
    dual_objective: float
    normalized_approximate_objective: float
    approximate_constraint_values: np.ndarray
    slack_variables: np.ndarray
    max_approximate_constraint_violation: float
    max_elastic_constraint_violation: float
    complementarity_residual: float
    projected_dual_gradient_residual: float
    stationarity_residual: float
    subproblem_kkt_residual: float
    feasible: bool
    lower_subproblem_bounds: np.ndarray
    upper_subproblem_bounds: np.ndarray
    dual_multipliers: np.ndarray
    model_objective_at_current: float
    model_objective_gradient_at_current: np.ndarray
    model_constraint_values_at_current: np.ndarray
    model_constraint_gradients_at_current: np.ndarray


@dataclass(frozen=True)
class MMAUpdateResult:
    """Updated free variables, persistent state, and subproblem diagnostics."""

    x_new: np.ndarray
    state: MMAState
    diagnostics: MMADiagnostics


def _finite_vector(value, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(length, float(array))
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape ({length},)")
    return array


def _mma_model(
    value: float,
    gradient: np.ndarray,
    x: np.ndarray,
    low: np.ndarray,
    upp: np.ndarray,
    variable_range: np.ndarray,
    gradient_regularization: float,
    curvature_regularization: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    regularization = (
        gradient_regularization * np.abs(gradient)
        + curvature_regularization / variable_range
    )
    upper_distance = upp - x
    lower_distance = x - low
    p = (np.maximum(gradient, 0.0) + regularization) * upper_distance**2
    q = (np.maximum(-gradient, 0.0) + regularization) * lower_distance**2
    constant = float(
        value
        - np.sum(p / upper_distance + q / lower_distance)
    )
    return p, q, constant


def _model_value(
    x: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    constant,
    low: np.ndarray,
    upp: np.ndarray,
):
    if p.ndim == 1:
        return float(constant + np.sum(p / (upp - x) + q / (x - low)))
    return np.asarray(
        constant
        + np.sum(p / (upp - x)[None, :] + q / (x - low)[None, :], axis=1),
        dtype=float,
    )


def _model_gradient_at(
    x: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    low: np.ndarray,
    upp: np.ndarray,
):
    if p.ndim == 1:
        return p / (upp - x) ** 2 - q / (x - low) ** 2
    return p / (upp - x)[None, :] ** 2 - q / (x - low)[None, :] ** 2


def _updated_asymptotes(
    x: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    state: Optional[MMAState],
    initial_asymptote: float,
    asymptote_increase: float,
    asymptote_decrease: float,
    minimum_asymptote: float,
    maximum_asymptote: float,
) -> tuple[np.ndarray, np.ndarray]:
    variable_range = upper_bounds - lower_bounds
    if state is None or state.iteration < 2:
        low = x - initial_asymptote * variable_range
        upp = x + initial_asymptote * variable_range
    else:
        if any(
            array.shape != x.shape
            for array in (
                state.lower_asymptotes,
                state.upper_asymptotes,
                state.previous_x,
                state.previous_previous_x,
            )
        ):
            raise ValueError("MMA state size does not match the current design")
        trend = (x - state.previous_x) * (
            state.previous_x - state.previous_previous_x
        )
        factor = np.ones_like(x)
        factor[trend > 0.0] = asymptote_increase
        factor[trend < 0.0] = asymptote_decrease
        low = x - factor * (state.previous_x - state.lower_asymptotes)
        upp = x + factor * (state.upper_asymptotes - state.previous_x)

    minimum_distance = minimum_asymptote * variable_range
    maximum_distance = maximum_asymptote * variable_range
    low = np.maximum(x - maximum_distance, np.minimum(x - minimum_distance, low))
    upp = np.minimum(x + maximum_distance, np.maximum(x + minimum_distance, upp))
    return low, upp


def mma_update(
    x: np.ndarray,
    objective_value: float,
    objective_gradient: np.ndarray,
    constraint_values: np.ndarray,
    constraint_gradients: np.ndarray,
    *,
    lower_bounds=0.0,
    upper_bounds=1.0,
    move_limit=0.2,
    state: Optional[MMAState] = None,
    objective_reference: Optional[float] = None,
    initial_asymptote: float = 0.5,
    asymptote_increase: float = 1.2,
    asymptote_decrease: float = 0.7,
    minimum_asymptote: float = 0.01,
    maximum_asymptote: float = 10.0,
    asymptote_bound_fraction: float = 0.1,
    gradient_regularization: float = 1.0e-3,
    curvature_regularization: float = 1.0e-5,
    slack_penalty=1000.0,
    slack_quadratic=1.0,
    dual_upper_bound: float = 1.0e12,
    dual_max_iterations: int = 500,
    dual_tolerance: float = 1.0e-9,
    feasibility_tolerance: float = 1.0e-7,
) -> MMAUpdateResult:
    """Solve one elastic MMA subproblem for inequalities ``g(x) <= 0``."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a nonempty finite one-dimensional vector")
    number_of_variables = x.size
    lower_bounds = _finite_vector(lower_bounds, number_of_variables, "lower_bounds")
    upper_bounds = _finite_vector(upper_bounds, number_of_variables, "upper_bounds")
    if np.any(upper_bounds <= lower_bounds):
        raise ValueError("upper_bounds must be strictly greater than lower_bounds")
    if np.any(x < lower_bounds) or np.any(x > upper_bounds):
        raise ValueError("x must lie inside the global bounds")
    move_limit = _finite_vector(move_limit, number_of_variables, "move_limit")
    if np.any(move_limit <= 0.0):
        raise ValueError("move_limit must be strictly positive")

    objective_value = float(objective_value)
    if not np.isfinite(objective_value):
        raise ValueError("objective_value must be finite")
    objective_gradient = np.asarray(objective_gradient, dtype=float)
    if objective_gradient.shape != x.shape or not np.all(
        np.isfinite(objective_gradient)
    ):
        raise ValueError("objective_gradient must be finite and match x")

    constraint_values = np.asarray(constraint_values, dtype=float)
    if constraint_values.ndim == 0:
        constraint_values = constraint_values.reshape(1)
    if constraint_values.ndim != 1 or not np.all(np.isfinite(constraint_values)):
        raise ValueError("constraint_values must be a finite one-dimensional vector")
    number_of_constraints = constraint_values.size
    constraint_gradients = np.asarray(constraint_gradients, dtype=float)
    if number_of_constraints == 1 and constraint_gradients.ndim == 1:
        constraint_gradients = constraint_gradients.reshape(1, -1)
    expected_gradient_shape = (number_of_constraints, number_of_variables)
    if constraint_gradients.shape != expected_gradient_shape or not np.all(
        np.isfinite(constraint_gradients)
    ):
        raise ValueError(
            "constraint_gradients must be finite with shape "
            f"{expected_gradient_shape}"
        )

    positive_parameters = {
        "initial_asymptote": initial_asymptote,
        "asymptote_increase": asymptote_increase,
        "asymptote_decrease": asymptote_decrease,
        "minimum_asymptote": minimum_asymptote,
        "maximum_asymptote": maximum_asymptote,
        "asymptote_bound_fraction": asymptote_bound_fraction,
        "gradient_regularization": gradient_regularization,
        "curvature_regularization": curvature_regularization,
        "dual_upper_bound": dual_upper_bound,
        "dual_tolerance": dual_tolerance,
        "feasibility_tolerance": feasibility_tolerance,
    }
    for name, value in positive_parameters.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if not 0.0 < asymptote_bound_fraction < 0.5:
        raise ValueError("asymptote_bound_fraction must be between 0 and 0.5")
    if asymptote_increase <= 1.0:
        raise ValueError("asymptote_increase must exceed one")
    if asymptote_decrease >= 1.0:
        raise ValueError("asymptote_decrease must be below one")
    if maximum_asymptote < minimum_asymptote:
        raise ValueError("maximum_asymptote must not be below minimum_asymptote")
    if int(dual_max_iterations) < 1:
        raise ValueError("dual_max_iterations must be at least one")

    slack_penalty = _finite_vector(
        slack_penalty, number_of_constraints, "slack_penalty"
    )
    slack_quadratic = _finite_vector(
        slack_quadratic, number_of_constraints, "slack_quadratic"
    )
    if np.any(slack_penalty <= 0.0) or np.any(slack_quadratic <= 0.0):
        raise ValueError("slack penalties must be strictly positive")

    if state is not None:
        if not isinstance(state, MMAState):
            raise TypeError("state must be MMAState or None")
        if state.dual_multipliers.shape != (number_of_constraints,):
            raise ValueError("MMA state constraint count does not match")
        persisted_reference = float(state.objective_reference)
        if objective_reference is not None and not np.isclose(
            float(objective_reference), persisted_reference
        ):
            raise ValueError("objective_reference conflicts with the MMA state")
        objective_reference = persisted_reference
    elif objective_reference is None:
        objective_reference = max(abs(objective_value), 1.0e-12)
    else:
        objective_reference = float(objective_reference)
    if not np.isfinite(objective_reference) or objective_reference <= 0.0:
        raise ValueError("objective_reference must be finite and positive")

    normalized_objective = objective_value / objective_reference
    normalized_objective_gradient = objective_gradient / objective_reference
    variable_range = upper_bounds - lower_bounds
    low, upp = _updated_asymptotes(
        x,
        lower_bounds,
        upper_bounds,
        state,
        initial_asymptote,
        asymptote_increase,
        asymptote_decrease,
        minimum_asymptote,
        maximum_asymptote,
    )
    alpha = np.maximum.reduce(
        (
            lower_bounds,
            x - move_limit,
            low + asymptote_bound_fraction * (x - low),
        )
    )
    beta = np.minimum.reduce(
        (
            upper_bounds,
            x + move_limit,
            upp - asymptote_bound_fraction * (upp - x),
        )
    )
    if np.any(alpha > beta):
        raise ValueError("MMA subproblem bounds are inconsistent")

    p0, q0, r0 = _mma_model(
        normalized_objective,
        normalized_objective_gradient,
        x,
        low,
        upp,
        variable_range,
        gradient_regularization,
        curvature_regularization,
    )
    p = np.empty(expected_gradient_shape)
    q = np.empty(expected_gradient_shape)
    r = np.empty(number_of_constraints)
    for index in range(number_of_constraints):
        p[index], q[index], r[index] = _mma_model(
            constraint_values[index],
            constraint_gradients[index],
            x,
            low,
            upp,
            variable_range,
            gradient_regularization,
            curvature_regularization,
        )

    def dual_terms(multipliers: np.ndarray):
        combined_p = p0 + multipliers @ p
        combined_q = q0 + multipliers @ q
        sqrt_p = np.sqrt(combined_p)
        sqrt_q = np.sqrt(combined_q)
        candidate = (sqrt_p * low + sqrt_q * upp) / (sqrt_p + sqrt_q)
        candidate = np.clip(candidate, alpha, beta)
        slack = np.maximum(0.0, (multipliers - slack_penalty) / slack_quadratic)
        approximate_constraints = _model_value(candidate, p, q, r, low, upp)
        dual_value = float(
            r0
            + multipliers @ r
            + np.sum(
                combined_p / (upp - candidate)
                + combined_q / (candidate - low)
            )
            - 0.5 * np.sum(slack_quadratic * slack**2)
        )
        dual_gradient = approximate_constraints - slack
        return candidate, slack, approximate_constraints, dual_value, dual_gradient

    def negative_dual(multipliers: np.ndarray):
        *_, dual_value, dual_gradient = dual_terms(multipliers)
        return -dual_value, -dual_gradient

    if number_of_constraints:
        initial_multipliers = (
            np.zeros(number_of_constraints)
            if state is None
            else np.clip(state.dual_multipliers, 0.0, dual_upper_bound)
        )
        dual_result: OptimizeResult = minimize(
            negative_dual,
            initial_multipliers,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, dual_upper_bound)] * number_of_constraints,
            options={
                "maxiter": int(dual_max_iterations),
                "ftol": dual_tolerance,
                "gtol": dual_tolerance,
                "maxls": 50,
            },
        )
        if not np.all(np.isfinite(dual_result.x)):
            raise RuntimeError("MMA dual solve returned nonfinite multipliers")
        multipliers = np.asarray(dual_result.x, dtype=float)
        x_new, slack, approximate_constraints, dual_value, dual_gradient = dual_terms(
            multipliers
        )
    else:
        multipliers = np.empty(0)
        x_new, slack, approximate_constraints, dual_value, dual_gradient = dual_terms(
            multipliers
        )
        dual_result = OptimizeResult(
            success=True,
            status=0,
            message="No inequality constraints",
            nit=0,
            nfev=1,
        )

    combined_p = p0 + multipliers @ p
    combined_q = q0 + multipliers @ q
    lagrangian_gradient = (
        combined_p / (upp - x_new) ** 2
        - combined_q / (x_new - low) ** 2
    )
    lower_active = np.isclose(x_new, alpha, rtol=0.0, atol=1.0e-10)
    upper_active = np.isclose(x_new, beta, rtol=0.0, atol=1.0e-10)
    stationarity_vector = np.abs(lagrangian_gradient)
    stationarity_vector[lower_active] = np.maximum(
        -lagrangian_gradient[lower_active], 0.0
    )
    stationarity_vector[upper_active] = np.maximum(
        lagrangian_gradient[upper_active], 0.0
    )
    stationarity_residual = float(
        np.max(stationarity_vector)
        / (1.0 + np.max(np.abs(lagrangian_gradient)))
    )
    elastic_constraints = approximate_constraints - slack
    max_hard_violation = (
        max(0.0, float(np.max(approximate_constraints)))
        if number_of_constraints
        else 0.0
    )
    max_elastic_violation = (
        max(0.0, float(np.max(elastic_constraints)))
        if number_of_constraints
        else 0.0
    )
    complementarity = (
        float(
            np.max(np.abs(multipliers * elastic_constraints) / (1.0 + multipliers))
        )
        if number_of_constraints
        else 0.0
    )
    projected_dual = np.where(
        multipliers > 1.0e-10,
        np.abs(dual_gradient),
        np.maximum(dual_gradient, 0.0),
    )
    projected_dual_residual = (
        float(np.max(projected_dual)) if number_of_constraints else 0.0
    )
    subproblem_kkt = max(
        max_elastic_violation,
        complementarity,
        projected_dual_residual,
        stationarity_residual,
    )
    # A 1e-4 inner KKT tolerance is conservative relative to the outer
    # topology tolerances while avoiding false failures from L-BFGS-B's
    # finite precision on elastic subproblems. Residuals above this threshold
    # are rejected regardless of SciPy's status flag.
    acceptance_tolerance = max(1.0e-4, 100.0 * dual_tolerance)
    # L-BFGS-B can report convergence from a small relative objective change
    # even when its projected gradient (and therefore the subproblem KKT
    # residual) is still large. Never accept an update on the library status
    # alone; the residual is the optimizer-facing contract.
    dual_success = bool(subproblem_kkt <= acceptance_tolerance)

    previous_x = x.copy()
    previous_previous_x = (
        x.copy() if state is None else state.previous_x.copy()
    )
    new_state = MMAState(
        iteration=1 if state is None else state.iteration + 1,
        lower_asymptotes=low.copy(),
        upper_asymptotes=upp.copy(),
        previous_x=previous_x,
        previous_previous_x=previous_previous_x,
        objective_reference=float(objective_reference),
        dual_multipliers=multipliers.copy(),
    )
    diagnostics = MMADiagnostics(
        dual_success=dual_success,
        dual_status=int(dual_result.status),
        dual_message=str(dual_result.message),
        dual_iterations=int(getattr(dual_result, "nit", 0)),
        dual_function_evaluations=int(getattr(dual_result, "nfev", 0)),
        dual_objective=float(dual_value),
        normalized_approximate_objective=_model_value(
            x_new, p0, q0, r0, low, upp
        ),
        approximate_constraint_values=np.asarray(
            approximate_constraints, dtype=float
        ).copy(),
        slack_variables=np.asarray(slack, dtype=float).copy(),
        max_approximate_constraint_violation=max_hard_violation,
        max_elastic_constraint_violation=max_elastic_violation,
        complementarity_residual=complementarity,
        projected_dual_gradient_residual=projected_dual_residual,
        stationarity_residual=stationarity_residual,
        subproblem_kkt_residual=float(subproblem_kkt),
        feasible=bool(max_hard_violation <= feasibility_tolerance),
        lower_subproblem_bounds=alpha.copy(),
        upper_subproblem_bounds=beta.copy(),
        dual_multipliers=multipliers.copy(),
        model_objective_at_current=_model_value(x, p0, q0, r0, low, upp),
        model_objective_gradient_at_current=_model_gradient_at(
            x, p0, q0, low, upp
        ),
        model_constraint_values_at_current=np.asarray(
            _model_value(x, p, q, r, low, upp), dtype=float
        ).copy(),
        model_constraint_gradients_at_current=np.asarray(
            _model_gradient_at(x, p, q, low, upp), dtype=float
        ).copy(),
    )
    return MMAUpdateResult(
        x_new=np.asarray(x_new, dtype=float).copy(),
        state=new_state,
        diagnostics=diagnostics,
    )
