import copy

import numpy as np
import pytest

from pytopo3d.utils.config_loader import (
    apply_material_orientation,
    get_material_strength,
    material_orientation_matrix,
    validate_material_strength,
)


VALID_STRENGTH = {
    "X_t": 10.0,
    "X_c": 11.0,
    "Y_t": 12.0,
    "Y_c": 13.0,
    "Z_t": 14.0,
    "Z_c": 15.0,
    "S_xy": 16.0,
    "S_yz": 17.0,
    "S_zx": 18.0,
    "criterion": "maximum_stress",
    "units": "Pa",
}


def test_strength_aware_preset_loads_positive_pa_values():
    strength = get_material_strength("orthotropic_validation")

    assert strength.units == "Pa"
    assert strength.criterion == "maximum_stress"
    assert all(
        value > 0.0
        for key, value in strength.as_dict().items()
        if key not in {"criterion", "units"}
    )


def test_stiffness_only_preset_rejects_failure_evaluation():
    with pytest.raises(ValueError, match="no strength data"):
        get_material_strength("pla_anisotropic")


@pytest.mark.parametrize(
    "field",
    ["X_t", "X_c", "Y_t", "Y_c", "Z_t", "Z_c", "S_xy", "S_yz", "S_zx"],
)
@pytest.mark.parametrize("invalid", [None, 0.0, -1.0, np.inf, np.nan, "bad", True])
def test_invalid_strength_allowables_are_rejected(field, invalid):
    strength = copy.deepcopy(VALID_STRENGTH)
    strength[field] = invalid

    with pytest.raises(ValueError, match=field):
        validate_material_strength(strength, material_name="invalid")


def test_missing_strength_allowable_is_rejected():
    strength = copy.deepcopy(VALID_STRENGTH)
    del strength["S_zx"]

    with pytest.raises(ValueError, match="S_zx"):
        validate_material_strength(strength, material_name="incomplete")


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("criterion", "tsai_wu", "unsupported"),
        ("units", "MPa", "must be Pa"),
    ],
)
def test_invalid_strength_metadata_is_rejected(key, value, message):
    strength = copy.deepcopy(VALID_STRENGTH)
    strength[key] = value

    with pytest.raises(ValueError, match=message):
        validate_material_strength(strength, material_name="invalid")


def test_orientation_matrix_maps_material_basis_vectors_to_global_axes():
    rotation = material_orientation_matrix("zxy")

    np.testing.assert_array_equal(
        rotation,
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    )
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3))


def test_reversed_orientation_maps_shear_planes_correctly():
    material_params = (
        10.0,
        20.0,
        30.0,
        100.0,
        200.0,
        300.0,
        0.10,
        0.20,
        0.30,
    )

    mapped = apply_material_orientation(material_params, "xzy")

    # material x->global x, y->global z, z->global y, hence global
    # xy/yz/zx planes correspond to material xz/zy/yx respectively.
    np.testing.assert_allclose(mapped[:3], (10.0, 30.0, 20.0))
    np.testing.assert_allclose(mapped[3:6], (300.0, 200.0, 100.0))


@pytest.mark.parametrize(
    "orientation", ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
)
def test_all_axis_permutations_preserve_normal_compliance(orientation):
    params = (
        10.0,
        20.0,
        30.0,
        100.0,
        200.0,
        300.0,
        0.10,
        0.20,
        0.30,
    )
    mapped = apply_material_orientation(params, orientation)
    rotation = material_orientation_matrix(orientation)

    def normal_compliance(values):
        E_x, E_y, E_z, _, _, _, nu_xy, nu_yz, nu_zx = values
        return np.array(
            [
                [1 / E_x, -nu_xy / E_x, -nu_zx / E_z],
                [-nu_xy / E_x, 1 / E_y, -nu_yz / E_y],
                [-nu_zx / E_z, -nu_yz / E_y, 1 / E_z],
            ]
        )

    expected = rotation @ normal_compliance(params) @ rotation.T
    np.testing.assert_allclose(normal_compliance(mapped), expected)
