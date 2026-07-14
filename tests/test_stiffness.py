from itertools import product

import numpy as np
import pytest

from pytopo3d.utils.config_loader import apply_material_orientation, parse_material_orientation_xyz
from pytopo3d.utils.assembly import H8_NODE_OFFSETS, build_edof
from pytopo3d.utils.stiffness import lk_H8, make_C_matrix


ORTHOTROPIC_PARAMS = (
    10.0,  # E_x
    20.0,  # E_y
    30.0,  # E_z
    4.0,   # G_xy
    5.0,   # G_yz
    6.0,   # G_zx
    0.10,  # nu_xy
    0.12,  # nu_yz
    0.08,  # nu_zx
)


def _element_coordinates(elem_size=1.0):
    return np.asarray(H8_NODE_OFFSETS, dtype=float) * elem_size


def _reflection_transform(axis):
    coordinates = _element_coordinates()
    reflected = coordinates.copy()
    reflected[:, axis] = 1.0 - reflected[:, axis]
    reflected_nodes = np.array(
        [
            np.flatnonzero(np.all(coordinates == coordinate, axis=1)).item()
            for coordinate in reflected
        ]
    )

    transform = np.zeros((24, 24))
    component_reflection = np.eye(3)
    component_reflection[axis, axis] = -1.0
    for original_node, reflected_node in enumerate(reflected_nodes):
        transform[
            3 * reflected_node : 3 * reflected_node + 3,
            3 * original_node : 3 * original_node + 3,
        ] = component_reflection
    return transform


def test_make_C_matrix_uses_default_isotropic_values_when_optional_params_missing():
    C = make_C_matrix(1.0, None, None, 0.4, None, None, 0.3, None, None, normalize=False)
    assert C.shape == (6, 6)
    assert np.isfinite(C).all()


def test_lk_H8_accepts_default_material_parameters():
    KE = lk_H8(elem_size=0.01)
    assert KE.shape == (24, 24)
    assert np.isfinite(KE).all()


def test_build_edof_uses_the_authoritative_h8_corner_order():
    edof, _, _ = build_edof(1, 1, 1)
    actual_node_ids = (edof[0, ::3] + 2) // 3
    expected_node_ids = np.array([1, 2, 4, 3, 5, 6, 8, 7])
    np.testing.assert_array_equal(actual_node_ids, expected_node_ids)


@pytest.mark.parametrize("axis", range(3))
def test_lk_H8_is_invariant_under_physical_reflection(axis):
    stiffness = lk_H8(*ORTHOTROPIC_PARAMS)
    transform = _reflection_transform(axis)
    np.testing.assert_allclose(
        transform.T @ stiffness @ transform,
        stiffness,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("axis", range(3))
def test_lk_H8_reproduces_affine_normal_strain_energy(axis):
    elem_size = 0.7
    stiffness = lk_H8(*ORTHOTROPIC_PARAMS, elem_size=elem_size)
    constitutive = make_C_matrix(*ORTHOTROPIC_PARAMS)
    coordinates = _element_coordinates(elem_size)
    displacement = np.zeros((8, 3))
    displacement[:, axis] = coordinates[:, axis]

    actual_energy = displacement.ravel() @ stiffness @ displacement.ravel()
    expected_energy = elem_size**3 * constitutive[axis, axis]
    assert actual_energy == pytest.approx(expected_energy, rel=1.0e-12)


@pytest.mark.parametrize(
    ("displacement_axis", "coordinate_axis", "shear_modulus"),
    (
        (0, 1, ORTHOTROPIC_PARAMS[3]),  # gamma_xy -> G_xy
        (1, 2, ORTHOTROPIC_PARAMS[4]),  # gamma_yz -> G_yz
        (2, 0, ORTHOTROPIC_PARAMS[5]),  # gamma_zx -> G_zx
    ),
)
def test_lk_H8_reproduces_affine_shear_energy(
    displacement_axis,
    coordinate_axis,
    shear_modulus,
):
    elem_size = 0.7
    stiffness = lk_H8(*ORTHOTROPIC_PARAMS, elem_size=elem_size)
    coordinates = _element_coordinates(elem_size)
    displacement = np.zeros((8, 3))
    displacement[:, displacement_axis] = coordinates[:, coordinate_axis]

    actual_energy = displacement.ravel() @ stiffness @ displacement.ravel()
    expected_energy = elem_size**3 * shear_modulus
    assert actual_energy == pytest.approx(expected_energy, rel=1.0e-12)


def test_lk_H8_uses_full_natural_coordinate_shape_factors():
    """Check a non-affine nodal mode against the standard H8 formula."""
    elem_size = 0.7
    stiffness = lk_H8(*ORTHOTROPIC_PARAMS, elem_size=elem_size)
    constitutive = make_C_matrix(*ORTHOTROPIC_PARAMS)
    node_sign = 2.0 * np.asarray(H8_NODE_OFFSETS[0], dtype=float) - 1.0
    gauss_coordinates = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

    expected_diagonal = 0.0
    for xi, eta, zeta in product(gauss_coordinates, repeat=3):
        natural_gradient = 0.125 * np.array(
            [
                node_sign[0]
                * (1.0 + node_sign[1] * eta)
                * (1.0 + node_sign[2] * zeta),
                node_sign[1]
                * (1.0 + node_sign[0] * xi)
                * (1.0 + node_sign[2] * zeta),
                node_sign[2]
                * (1.0 + node_sign[0] * xi)
                * (1.0 + node_sign[1] * eta),
            ]
        )
        physical_gradient = natural_gradient * (2.0 / elem_size)
        strain_from_node_0_x = np.array(
            [
                physical_gradient[0],
                0.0,
                0.0,
                physical_gradient[1],
                0.0,
                physical_gradient[2],
            ]
        )
        expected_diagonal += (
            strain_from_node_0_x
            @ constitutive
            @ strain_from_node_0_x
            * (elem_size / 2.0) ** 3
        )

    assert stiffness[0, 0] == pytest.approx(expected_diagonal, rel=1.0e-12)


def test_parse_material_orientation_xyz_accepts_valid_mapping():
    assert parse_material_orientation_xyz("ZxY") == "zxy"


def test_parse_material_orientation_xyz_rejects_invalid_mapping():
    with pytest.raises(ValueError):
        parse_material_orientation_xyz("abz")


def test_apply_material_orientation_remaps_principal_and_grouped_terms():
    # (E_x, E_y, E_z, G_xy, G_yz, G_zx, nu_xy, nu_yz, nu_zx)
    material_params = (10.0, 20.0, 30.0, 100.0, 200.0, 300.0, 0.10, 0.20, 0.30)
    remapped = apply_material_orientation(material_params, "zxy")

    # zxy means material x->global z, material y->global x, material z->global y
    assert remapped == (20.0, 30.0, 10.0, 200.0, 300.0, 100.0, 0.20, 0.30, 0.10)
