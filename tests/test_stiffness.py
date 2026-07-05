import numpy as np
import pytest

from pytopo3d.utils.config_loader import apply_material_orientation, parse_material_orientation_xyz
from pytopo3d.utils.stiffness import lk_H8, make_C_matrix


def test_make_C_matrix_uses_default_isotropic_values_when_optional_params_missing():
    C = make_C_matrix(1.0, None, None, 0.4, None, None, 0.3, None, None, normalize=False)
    assert C.shape == (6, 6)
    assert np.isfinite(C).all()


def test_lk_H8_accepts_default_material_parameters():
    KE = lk_H8(elem_size=0.01)
    assert KE.shape == (24, 24)
    assert np.isfinite(KE).all()


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
