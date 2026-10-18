import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pytopo3d.analysis.failure_sensitivity import (
    evaluate_failure_partials,
    evaluate_failure_sensitivity,
    map_failure_gradient_to_design,
    solve_failure_adjoint,
)
from pytopo3d.utils.assembly import (
    build_edof,
    build_force_vector,
    build_supports,
)
from pytopo3d.utils.config_loader import (
    apply_material_orientation,
    get_material_params,
    get_material_strength,
    material_orientation_matrix,
)
from pytopo3d.utils.filter import build_filter, build_physical_density
from pytopo3d.utils.stiffness import lk_H8, make_C_matrix


def _gradient_problem():
    nelx, nely, nelz = 8, 4, 1
    shape = (nely, nelx, nelz)
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    elem_size = 0.01
    penal = 3.0
    orientation = "yzx"
    material_params = apply_material_orientation(
        get_material_params("orthotropic_validation"),
        orientation,
    )
    element_stiffness = lk_H8(*material_params, elem_size=elem_size)
    constitutive_matrix = make_C_matrix(*material_params)
    orientation_matrix = material_orientation_matrix(orientation)
    strength = get_material_strength("orthotropic_validation")

    edof_matrix, rows, columns = build_edof(nelx, nely, nelz)
    free_dofs, _ = build_supports(nelx, nely, nelz, ndof)
    force_field = np.zeros(shape + (3,))
    force_field[1, -1, 0] = np.array([30.0, -17.0, -43.0])
    force = build_force_vector(nelx, nely, nelz, ndof, force_field)

    protected_solid = np.zeros(shape, dtype=bool)
    protected_solid[:, 0, :] = True
    protected_solid[1, -1, 0] = True
    protected_void = np.zeros(shape, dtype=bool)
    protected_void[3, 3, 0] = True
    fixed = protected_solid | protected_void
    eligible = (~protected_void).ravel(order="F")

    element_index = np.arange(np.prod(shape)).reshape(shape, order="F")
    x = 0.48 + 0.12 * np.sin(0.73 * element_index + 0.2)
    x[protected_solid] = 1.0
    x[protected_void] = 0.0
    filter_matrix, filter_row_sums = build_filter(nelx, nely, nelz, rmin=1.6)

    return {
        "shape": shape,
        "ndof": ndof,
        "elem_size": elem_size,
        "penal": penal,
        "element_stiffness": element_stiffness,
        "constitutive_matrix": constitutive_matrix,
        "orientation_matrix": orientation_matrix,
        "strength": strength,
        "edof_matrix": edof_matrix,
        "rows": rows - 1,
        "columns": columns - 1,
        "free_dofs": free_dofs,
        "force": force,
        "protected_solid": protected_solid,
        "protected_void": protected_void,
        "fixed": fixed,
        "eligible": eligible,
        "x": x,
        "filter_matrix": filter_matrix,
        "filter_row_sums": filter_row_sums,
    }


def _evaluate_design(problem, x, *, relaxation_exponent=0.5):
    _, density, projection_derivative = build_physical_density(
        x,
        H=problem["filter_matrix"],
        Hs=problem["filter_row_sums"],
        beta=2.0,
        eta=0.5,
        protected_solid=problem["protected_solid"],
        protected_void=problem["protected_void"],
    )
    density_vector = density.ravel(order="F")
    stiffness_scale = 1.0e-9 + (1.0 - 1.0e-9) * density_vector ** problem[
        "penal"
    ]
    element_values = np.kron(
        stiffness_scale,
        problem["element_stiffness"].ravel(),
    )
    stiffness = sp.coo_matrix(
        (element_values, (problem["rows"], problem["columns"])),
        shape=(problem["ndof"], problem["ndof"]),
    ).tocsr()
    displacement = np.zeros(problem["ndof"])
    free = problem["free_dofs"]
    displacement[free] = spsolve(
        stiffness[free, :][:, free],
        problem["force"][free],
    )
    partials = evaluate_failure_partials(
        displacement,
        problem["edof_matrix"],
        problem["constitutive_matrix"],
        problem["orientation_matrix"],
        density,
        problem["strength"],
        elem_size=problem["elem_size"],
        relaxation_exponent=relaxation_exponent,
        aggregate_exponent=8.0,
        correction_factor=1.13,
        eligible_elements=problem["eligible"],
    )
    return stiffness, displacement, density, projection_derivative, partials


def test_complete_adjoint_gradient_matches_filtered_projected_finite_difference():
    problem = _gradient_problem()
    x = problem["x"]
    stiffness, displacement, density, projection_derivative, base_partials = (
        _evaluate_design(problem, x)
    )
    solve_calls = []

    def counting_solver(matrix, right_hand_side):
        solve_calls.append((matrix.shape, right_hand_side.shape))
        return spsolve(matrix, right_hand_side)

    sensitivity = evaluate_failure_sensitivity(
        displacement,
        stiffness,
        problem["free_dofs"],
        problem["edof_matrix"],
        problem["element_stiffness"],
        problem["constitutive_matrix"],
        problem["orientation_matrix"],
        density,
        problem["strength"],
        simp_penal=problem["penal"],
        elem_size=problem["elem_size"],
        relaxation_exponent=0.5,
        aggregate_exponent=8.0,
        correction_factor=1.13,
        eligible_elements=problem["eligible"],
        linear_solver=counting_solver,
    )
    physical_gradient = sensitivity.physical_density_derivative.reshape(
        problem["shape"], order="F"
    )
    analytical = map_failure_gradient_to_design(
        physical_gradient,
        projection_derivative,
        problem["filter_matrix"],
        problem["filter_row_sums"],
        fixed_design_elements=problem["fixed"],
    ).ravel(order="F")

    assert sensitivity.adjoint_solve_count == 1
    assert sensitivity.adjoint_relative_residual < 1.0e-10
    assert len(solve_calls) == 1
    np.testing.assert_array_equal(
        analytical[problem["fixed"].ravel(order="F")],
        0.0,
    )

    free_indices = np.flatnonzero(~problem["fixed"].ravel(order="F"))
    selected = free_indices[[2, len(free_indices) // 2, -3]]
    errors_by_step = {step: [] for step in (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)}
    x_flat = x.ravel(order="F")
    important = (
        base_partials.aggregate_failure_index_derivative
        > 1.0e-8
        * np.max(base_partials.aggregate_failure_index_derivative)
    )

    for index in selected:
        for step, errors in errors_by_step.items():
            plus = x_flat.copy()
            minus = x_flat.copy()
            plus[index] += step
            minus[index] -= step
            plus_state = _evaluate_design(
                problem,
                plus.reshape(problem["shape"], order="F"),
            )[-1]
            minus_state = _evaluate_design(
                problem,
                minus.reshape(problem["shape"], order="F"),
            )[-1]
            np.testing.assert_array_equal(
                plus_state.active_gauss_point[important],
                base_partials.active_gauss_point[important],
            )
            np.testing.assert_array_equal(
                minus_state.active_gauss_point[important],
                base_partials.active_gauss_point[important],
            )
            np.testing.assert_array_equal(
                plus_state.active_component[important],
                base_partials.active_component[important],
            )
            np.testing.assert_array_equal(
                minus_state.active_component[important],
                base_partials.active_component[important],
            )
            finite_difference = (
                plus_state.aggregate_result.aggregate
                - minus_state.aggregate_result.aggregate
            ) / (2.0 * step)
            assert np.sign(finite_difference) == np.sign(analytical[index])
            relative_error = abs(finite_difference - analytical[index]) / max(
                abs(finite_difference),
                abs(analytical[index]),
                1.0e-12,
            )
            errors.append(relative_error)

    assert max(errors_by_step[1.0e-3]) < 2.0e-3
    assert max(errors_by_step[1.0e-4]) < 2.0e-4
    assert max(errors_by_step[1.0e-5]) < 2.0e-4
    assert max(errors_by_step[1.0e-6]) < 2.0e-3


def test_q_below_one_rejects_eligible_zero_but_allows_excluded_void():
    problem = _gradient_problem()
    _, displacement, density, _, _ = _evaluate_design(problem, problem["x"])
    zero_element = np.flatnonzero(problem["protected_void"].ravel(order="F"))[0]

    allowed = evaluate_failure_partials(
        displacement,
        problem["edof_matrix"],
        problem["constitutive_matrix"],
        problem["orientation_matrix"],
        density,
        problem["strength"],
        elem_size=problem["elem_size"],
        relaxation_exponent=0.5,
        eligible_elements=problem["eligible"],
    )
    assert allowed.explicit_density_derivative[zero_element] == 0.0

    eligible_with_void = problem["eligible"].copy()
    eligible_with_void[zero_element] = True
    with pytest.raises(ValueError, match="strictly positive"):
        evaluate_failure_partials(
            displacement,
            problem["edof_matrix"],
            problem["constitutive_matrix"],
            problem["orientation_matrix"],
            density,
            problem["strength"],
            elem_size=problem["elem_size"],
            relaxation_exponent=0.5,
            eligible_elements=eligible_with_void,
        )


def test_q_zero_disables_relaxation_and_has_no_explicit_density_term():
    problem = _gradient_problem()
    _, displacement, density, _, partials = _evaluate_design(
        problem,
        problem["x"],
        relaxation_exponent=0.0,
    )

    np.testing.assert_array_equal(
        partials.relaxed_stress_material_gauss,
        partials.solid_stress_material_gauss,
    )
    np.testing.assert_array_equal(partials.explicit_density_derivative, 0.0)


def test_zero_failure_rhs_uses_zero_adjoint_without_a_linear_solve():
    solve_calls = []

    def unexpected_solver(matrix, right_hand_side):
        solve_calls.append(1)
        return np.linalg.solve(matrix, right_hand_side)

    result = solve_failure_adjoint(
        np.eye(2),
        np.zeros(4),
        np.array([1, 3]),
        linear_solver=unexpected_solver,
    )

    np.testing.assert_array_equal(result.adjoint, np.zeros(4))
    assert result.relative_residual == 0.0
    assert result.solve_count == 0
    assert solve_calls == []
