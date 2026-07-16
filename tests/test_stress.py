import numpy as np
import pytest

from pytopo3d.analysis.stress import recover_gauss_stress
from pytopo3d.utils.assembly import H8_NODE_OFFSETS, build_edof
from pytopo3d.utils.stiffness import lk_H8, make_C_matrix


def _global_displacement_from_local(local_displacement):
    edof_matrix, _, _ = build_edof(1, 1, 1)
    displacement = np.zeros(24)
    displacement[edof_matrix[0] - 1] = np.asarray(local_displacement).ravel()
    return displacement, edof_matrix


@pytest.mark.parametrize(
    ("displacement_axis", "coordinate_axis", "stress_component"),
    [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
        (0, 1, 3),
        (1, 2, 4),
        (2, 0, 5),
    ],
)
def test_affine_states_recover_each_voigt_stress_component(
    displacement_axis,
    coordinate_axis,
    stress_component,
):
    elem_size = 0.7
    coordinates = np.asarray(H8_NODE_OFFSETS, dtype=float) * elem_size
    local_displacement = np.zeros((8, 3))
    local_displacement[:, displacement_axis] = coordinates[:, coordinate_axis]
    displacement, edof_matrix = _global_displacement_from_local(local_displacement)
    constitutive = np.diag([11.0, 12.0, 13.0, 14.0, 15.0, 16.0])

    stress, strain = recover_gauss_stress(
        displacement,
        edof_matrix,
        constitutive,
        elem_size=elem_size,
        return_strain=True,
    )

    expected_strain = np.zeros(6)
    expected_strain[stress_component] = 1.0
    expected_stress = constitutive @ expected_strain
    expected_strain_gauss = np.broadcast_to(expected_strain, (1, 8, 6))
    expected_stress_gauss = np.broadcast_to(expected_stress, (1, 8, 6))
    assert stress.shape == (1, 8, 6)
    np.testing.assert_allclose(strain, expected_strain_gauss, atol=1e-14)
    np.testing.assert_allclose(stress, expected_stress_gauss, atol=1e-13)


def _solve_single_element_bar(total_force):
    elem_size = 0.2
    area = elem_size**2
    material_params = (
        2.0e9,
        2.0e9,
        2.0e9,
        1.0e9,
        1.0e9,
        1.0e9,
        0.0,
        0.0,
        0.0,
    )
    stiffness = lk_H8(*material_params, elem_size=elem_size)
    constitutive = make_C_matrix(*material_params)
    edof_matrix, _, _ = build_edof(1, 1, 1)
    element_dofs = edof_matrix[0] - 1

    global_stiffness = np.zeros((24, 24))
    global_stiffness[np.ix_(element_dofs, element_dofs)] += stiffness
    force = np.zeros(24)

    fixed_dofs = []
    loaded_x_dofs = []
    for local_node, (x_offset, _, _) in enumerate(H8_NODE_OFFSETS):
        node_dofs = element_dofs[3 * local_node : 3 * local_node + 3]
        if x_offset == 0:
            fixed_dofs.extend(node_dofs)
        else:
            loaded_x_dofs.append(node_dofs[0])
    force[loaded_x_dofs] = total_force / len(loaded_x_dofs)

    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    free_dofs = np.setdiff1d(np.arange(24), fixed_dofs)
    displacement = np.zeros(24)
    displacement[free_dofs] = np.linalg.solve(
        global_stiffness[np.ix_(free_dofs, free_dofs)],
        force[free_dofs],
    )
    stress = recover_gauss_stress(
        displacement,
        edof_matrix,
        constitutive,
        elem_size=elem_size,
    )
    return stress, total_force / area


def test_rectangular_bar_stress_matches_force_over_area():
    stress, expected_axial_stress = _solve_single_element_bar(2_000.0)

    np.testing.assert_allclose(
        stress[0, :, 0], expected_axial_stress, rtol=1e-12, atol=1e-8
    )
    np.testing.assert_allclose(stress[0, :, 1:], 0.0, atol=1e-8)


def test_recovered_stress_scales_linearly_with_force():
    stress, _ = _solve_single_element_bar(1_000.0)
    doubled_stress, _ = _solve_single_element_bar(2_000.0)

    np.testing.assert_allclose(doubled_stress, 2.0 * stress, rtol=1e-12, atol=1e-8)


@pytest.mark.parametrize(
    ("displacement", "edof", "message"),
    [
        (np.zeros((2, 2)), np.ones((1, 24), dtype=int), "one-dimensional"),
        (np.zeros(24), np.ones((1, 23), dtype=int), "shape"),
        (np.zeros(24), np.zeros((1, 24), dtype=int), "1-based"),
    ],
)
def test_stress_recovery_rejects_invalid_displacement_mapping(
    displacement, edof, message
):
    with pytest.raises(ValueError, match=message):
        recover_gauss_stress(displacement, edof, np.eye(6))
