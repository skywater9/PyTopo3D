from pathlib import Path

import numpy as np

from pytopo3d.analysis.postprocessing import (
    build_failure_region_masks,
    evaluate_failure_representations,
)
from pytopo3d.core.optimizer import evaluate_fixed_geometry_metrics
from pytopo3d.utils.config_loader import (
    get_material_params,
    get_material_strength,
    material_orientation_matrix,
)
from pytopo3d.utils.results_manager import ResultsManager


def _small_projected_bar():
    density = np.array([[[1.0], [0.8], [0.7], [1.0]]])
    material_params = get_material_params("orthotropic_validation")
    return density, material_params


def test_fixed_geometry_response_option_returns_recovery_inputs_without_default_leak():
    density, material_params = _small_projected_bar()

    default_metrics = evaluate_fixed_geometry_metrics(
        xPhys=density,
        penal=3.0,
        material_params=material_params,
        elem_size=0.01,
    )
    response = evaluate_fixed_geometry_metrics(
        xPhys=density,
        penal=3.0,
        material_params=material_params,
        elem_size=0.01,
        return_displacement=True,
    )

    assert "displacement" not in default_metrics
    assert "edof_matrix" not in default_metrics
    assert response["displacement"].shape == (3 * 5 * 2 * 2,)
    assert response["edof_matrix"].shape == (4, 24)
    assert response["compliance"] == default_metrics["compliance"]


def test_failure_region_masks_keep_fixture_values_visible_but_out_of_design_region():
    shape = (2, 5, 2)
    obstacle = np.zeros(shape, dtype=bool)
    obstacle[0, 2, 1] = True
    non_obstacle, fixture = build_failure_region_masks(
        shape,
        obstacle_mask=obstacle,
    )

    assert not non_obstacle[0, 2, 1]
    assert np.all(fixture[:, 0, :])
    assert np.all(fixture[:, -1, 0])
    assert not fixture[0, 2, 1]
    assert np.any(non_obstacle & ~fixture)


def test_projected_and_binary_failure_use_independent_solves_and_exact_fields(tmp_path):
    density, material_params = _small_projected_bar()
    results_manager = ResultsManager(
        base_dir=str(tmp_path),
        experiment_name="failure_postprocess",
    )

    result = evaluate_failure_representations(
        x_projected=density,
        binary_threshold=0.5,
        penal=3.0,
        material_params=material_params,
        strength=get_material_strength("orthotropic_validation"),
        orientation_matrix=material_orientation_matrix("xyz"),
        elem_size=0.01,
        results_manager=results_manager,
    )

    assert result.projected.stress_global_gauss.shape == (4, 8, 6)
    assert result.projected.stress_material_gauss.shape == (4, 8, 6)
    assert result.projected.gauss_failure.failure_components_gauss.shape == (
        4,
        8,
        6,
    )
    assert result.projected.gauss_failure.failure_index_gauss.shape == (4, 8)
    assert result.projected.gauss_failure.failure_index_element.shape == (4,)
    assert result.binary.gauss_failure.failure_index_gauss.shape == (4, 8)
    assert (
        result.metrics["failure_stress_model_binary"]
        == "full_density_unrelaxed"
    )
    assert (
        result.metrics["failure_stress_model_projected"]
        == "full_density_unrelaxed"
    )
    assert result.projected_response["compliance"] != result.binary_response[
        "compliance"
    ]

    for representation in ("projected", "binary"):
        assert result.metrics[f"failure_index_max_{representation}"] >= 0.0
        assert result.metrics[f"predicted_failure_load_{representation}"] > 0.0
        assert result.metrics[f"critical_element_{representation}"] is not None
        assert result.metrics[f"critical_gauss_point_{representation}"] in range(8)
        assert result.metrics[f"critical_mode_{representation}"]
        assert (
            result.metrics[
                f"max_failure_index_all_elements_{representation}"
            ]
            >= 0.0
        )
        assert (
            result.metrics[
                f"max_failure_index_design_region_{representation}"
            ]
            >= 0.0
        )
        assert (
            result.metrics[f"max_failure_index_fixture_region_{representation}"]
            >= 0.0
        )

    expected_files = {
        f"{stem}_{representation}.npy"
        for representation in ("projected", "binary")
        for stem in (
            "stress_global_gauss",
            "stress_material_gauss",
            "failure_components_gauss",
            "failure_index_gauss",
            "failure_index_element",
            "critical_failure_mode",
        )
    } | {"failure_binary_density.npy"}
    assert set(result.saved_files) == expected_files
    for filename in expected_files:
        assert Path(result.saved_files[filename]).is_file()

    saved_failure_index = np.load(
        result.saved_files["failure_index_gauss_binary.npy"]
    )
    np.testing.assert_array_equal(
        saved_failure_index,
        result.binary.gauss_failure.failure_index_gauss,
    )


def test_binary_failure_only_includes_solid_elements():
    density, material_params = _small_projected_bar()
    density[0, 1, 0] = 0.2

    result = evaluate_failure_representations(
        x_projected=density,
        binary_threshold=0.5,
        penal=3.0,
        material_params=material_params,
        strength=get_material_strength("orthotropic_validation"),
        orientation_matrix=np.eye(3),
        elem_size=0.01,
    )

    assert result.binary_density[0, 1, 0] == 0.0
    assert result.binary.all_element_count == 3
    assert result.projected.all_element_count == 4
