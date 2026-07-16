import json

import numpy as np

from pytopo3d.cli.parser import parse_args
from pytopo3d.runners import experiment
from pytopo3d.utils.export import voxel_to_stl
from pytopo3d.utils.metrics import collect_metrics
from pytopo3d.utils.results_manager import ResultsManager


def test_projection_cli_defaults_and_per_stage_alias():
    defaults = parse_args([])
    assert defaults.beta_schedule == [1.0, 2.0, 4.0, 8.0]
    assert defaults.projection_eta == 0.5
    assert defaults.move_limit == 0.2
    assert defaults.optimization_mode == "compliance"
    assert defaults.optimizer == "oc"
    assert defaults.mma_move_limit == 0.05
    assert defaults.mma_min_density == 1.0e-3
    assert defaults.failure_limit == 1.0
    assert defaults.failure_aggregate_exponent == 8.0
    assert defaults.failure_relaxation_exponent == 0.5

    custom = parse_args(
        [
            "--beta-schedule",
            "1",
            "3",
            "--projection-eta",
            "0.4",
            "--max-iterations-per-stage",
            "7",
        ]
    )
    assert custom.beta_schedule == [1.0, 3.0]
    assert custom.projection_eta == 0.4
    assert custom.maxloop == 7


def test_project_yxz_calibration_block_exports_at_physical_scale():
    # Project arrays are stored (nely, nelx, nelz), while mesh coordinates are
    # XYZ. This is the requested 4 x 50 x 160 element, h=0.5 mm calibration.
    rho_physical = np.ones((50, 4, 160), dtype=float)
    elem_size = 0.0005

    mesh = voxel_to_stl(
        rho_physical,
        level=0.5,
        smooth_mesh=True,
        smooth_iterations=5,
        fix_mesh=True,
        upscale_factor=3,
        elem_size=elem_size,
        array_order="yxz",
    )

    expected_extents = np.array([0.002, 0.025, 0.080])
    # One refined voxel is the stated surface-resolution tolerance.
    np.testing.assert_allclose(
        mesh.extents,
        expected_extents,
        atol=elem_size / 3.0,
        rtol=0.0,
    )
    assert mesh.is_watertight
    assert mesh.is_volume
    assert mesh.volume > 0.0


def test_export_runner_forwards_the_saved_physical_field_without_projection(
    tmp_path,
    monkeypatch,
):
    rho_physical = np.linspace(0.0, 1.0, 24).reshape((3, 2, 4))
    protected_void = np.zeros_like(rho_physical, dtype=bool)
    protected_void[0, 0, 0] = True
    rho_physical[protected_void] = 0.0
    result_path = tmp_path / "optimized_design.npy"
    np.save(result_path, rho_physical)
    captured = {}

    def capture_export(**kwargs):
        captured.update(kwargs)
        return kwargs["output_file"]

    monkeypatch.setattr(experiment, "voxel_to_stl", capture_export)
    results_mgr = type("ResultsStub", (), {"experiment_dir": str(tmp_path)})()

    exported = experiment.export_result_to_stl(
        export_stl=True,
        combined_obstacle_mask=protected_void,
        results_mgr=results_mgr,
        result_path=str(result_path),
        elem_size=0.0005,
    )

    assert exported is True
    np.testing.assert_array_equal(captured["input_file"], rho_physical)
    assert captured["array_order"] == "yxz"
    assert captured["elem_size"] == 0.0005
    assert captured["level"] == 0.5


def test_projection_metrics_serialize_through_results_manager(tmp_path):
    projection_metrics = {
        "projection_enabled": True,
        "projection_beta": 8.0,
        "projection_beta_schedule": [1.0, 2.0, 4.0, 8.0],
        "projection_eta": 0.5,
        "physical_density_fraction": 0.3,
        "gray_fraction_005_095": 0.02,
        "projected_compliance": 12.0,
        "binary_compliance": 12.6,
        "binary_compliance_delta": 0.05,
    }
    metrics = collect_metrics(
        terminal_input="test",
        nelx=2,
        nely=2,
        nelz=2,
        volfrac=0.3,
        penal=3.0,
        rmin=1.5,
        disp_thres=0.5,
        optimization_metrics=projection_metrics,
    )
    manager = ResultsManager(
        base_dir=str(tmp_path),
        experiment_name="projection_metrics",
    )
    metrics_path = manager.update_metrics(metrics)

    with open(metrics_path, encoding="utf-8") as metrics_file:
        saved = json.load(metrics_file)
    for key, value in projection_metrics.items():
        assert saved[key] == value
