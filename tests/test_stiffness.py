import numpy as np

from pytopo3d.utils.stiffness import lk_H8, make_C_matrix


def test_make_C_matrix_uses_default_isotropic_values_when_optional_params_missing():
    C = make_C_matrix(1.0, None, None, 0.4, None, None, 0.3, None, None, normalize=False)
    assert C.shape == (6, 6)
    assert np.isfinite(C).all()


def test_lk_H8_accepts_default_material_parameters():
    KE = lk_H8(elem_size=0.01)
    assert KE.shape == (24, 24)
    assert np.isfinite(KE).all()
