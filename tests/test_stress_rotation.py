import numpy as np
import pytest

from pytopo3d.analysis.stress import (
    rotate_stress_to_material,
    stress_rotation_matrix_to_material,
    stress_tensor_to_voigt,
    stress_voigt_to_tensor,
)
from pytopo3d.utils.config_loader import (
    apply_material_orientation,
    material_orientation_matrix,
)
from pytopo3d.utils.stiffness import make_C_matrix


def _rotation_about_z(angle_degrees):
    angle = np.deg2rad(angle_degrees)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _engineering_strain_voigt_to_tensor(strain):
    tensor = stress_voigt_to_tensor(strain)
    tensor[..., 0, 1] *= 0.5
    tensor[..., 1, 0] *= 0.5
    tensor[..., 1, 2] *= 0.5
    tensor[..., 2, 1] *= 0.5
    tensor[..., 2, 0] *= 0.5
    tensor[..., 0, 2] *= 0.5
    return tensor


def _engineering_strain_tensor_to_voigt(tensor):
    strain = stress_tensor_to_voigt(tensor)
    strain[..., 3:] *= 2.0
    return strain


def test_stress_voigt_tensor_round_trip_preserves_order_and_shape():
    stress = np.arange(2 * 8 * 6, dtype=float).reshape(2, 8, 6)

    tensor = stress_voigt_to_tensor(stress)
    recovered = stress_tensor_to_voigt(tensor)

    assert tensor.shape == (2, 8, 3, 3)
    np.testing.assert_array_equal(recovered, stress)
    np.testing.assert_array_equal(
        tensor[0, 0],
        np.array(
            [
                [stress[0, 0, 0], stress[0, 0, 3], stress[0, 0, 5]],
                [stress[0, 0, 3], stress[0, 0, 1], stress[0, 0, 4]],
                [stress[0, 0, 5], stress[0, 0, 4], stress[0, 0, 2]],
            ]
        ),
    )


def test_zero_rotation_leaves_gauss_stress_unchanged():
    stress = np.arange(3 * 8 * 6, dtype=float).reshape(3, 8, 6)

    rotated = rotate_stress_to_material(stress, np.eye(3))

    np.testing.assert_allclose(rotated, stress, atol=1e-13)


def test_ninety_degree_rotation_maps_global_x_tension_to_material_y():
    stress_global = np.array([90.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    stress_material = rotate_stress_to_material(
        stress_global, _rotation_about_z(90.0)
    )

    np.testing.assert_allclose(
        stress_material,
        np.array([0.0, 90.0, 0.0, 0.0, 0.0, 0.0]),
        atol=1e-13,
    )


def test_forty_five_degree_rotation_matches_analytical_transformation():
    stress_global = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    stress_material = rotate_stress_to_material(
        stress_global, _rotation_about_z(45.0)
    )

    np.testing.assert_allclose(
        stress_material,
        np.array([50.0, 50.0, 0.0, -50.0, 0.0, 0.0]),
        atol=1e-12,
    )


def test_axis_permutation_uses_material_to_global_convention():
    stress_global = np.array([60.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rotation = material_orientation_matrix("zxy")

    stress_material = rotate_stress_to_material(stress_global, rotation)

    # zxy maps material y to global x.
    np.testing.assert_allclose(
        stress_material,
        np.array([0.0, 60.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_rotation_matrix_matches_direct_tensor_rotation():
    rotation = _rotation_about_z(33.0)
    transformation = stress_rotation_matrix_to_material(rotation)
    stress = np.array([11.0, 7.0, 3.0, -2.0, 5.0, 4.0])

    np.testing.assert_allclose(
        transformation @ stress,
        rotate_stress_to_material(stress, rotation),
        atol=1e-13,
    )


@pytest.mark.parametrize(
    "orientation", ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
)
def test_stiffness_and_strength_rotation_conventions_match(orientation):
    material_params = (
        10.0,
        20.0,
        30.0,
        4.0,
        5.0,
        6.0,
        0.10,
        0.12,
        0.08,
    )
    rotation = material_orientation_matrix(orientation)
    constitutive_material = make_C_matrix(*material_params)
    constitutive_global = make_C_matrix(
        *apply_material_orientation(material_params, orientation)
    )

    for strain_global in np.eye(6):
        strain_global_tensor = _engineering_strain_voigt_to_tensor(strain_global)
        strain_material_tensor = (
            rotation.T @ strain_global_tensor @ rotation
        )
        strain_material = _engineering_strain_tensor_to_voigt(
            strain_material_tensor
        )
        expected_material_stress = constitutive_material @ strain_material
        global_stress = constitutive_global @ strain_global
        actual_material_stress = rotate_stress_to_material(
            global_stress, rotation
        )
        np.testing.assert_allclose(
            actual_material_stress,
            expected_material_stress,
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "invalid",
    [
        np.eye(2),
        np.ones((3, 3)),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, np.nan]]),
    ],
)
def test_invalid_orientation_matrix_is_rejected(invalid):
    with pytest.raises(ValueError, match="orientation_matrix"):
        rotate_stress_to_material(np.zeros(6), invalid)


def test_nonsymmetric_stress_tensor_is_rejected():
    nonsymmetric = np.array(
        [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]]
    )

    with pytest.raises(ValueError, match="symmetric"):
        stress_tensor_to_voigt(nonsymmetric)
