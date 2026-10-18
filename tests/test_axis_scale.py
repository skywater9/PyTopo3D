"""Physical sizing across CLI, mechanics, saved results and manufacturing output."""

import importlib
import json
import logging
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import trimesh

from pytopo3d.analysis.postprocessing import evaluate_failure_representations
from pytopo3d.cli.parser import parse_args
from pytopo3d.core.optimizer import top3d
from pytopo3d.utils.config_loader import get_material_params, get_material_strength
from pytopo3d.utils.element_size import element_size_xyz, resolve_cli_element_size
from pytopo3d.utils.export import voxel_to_stl
from pytopo3d.utils.results_manager import ResultsManager
from pytopo3d.utils.stiffness import h8_gauss_integration_data, lk_H8
from pytopo3d.visualization.animation import create_frame
from pytopo3d.visualization.display import display_3D


@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("conflict", ["--elem-size", "--target-physical-x"])
@pytest.mark.parametrize("reverse", [False, True])
def test_axis_scale_rejects_explicit_conflicts(axis, conflict, reverse, capsys):
    groups = [[f"--{axis}-axis-scale", "1"], [conflict, "0.01"]]
    if reverse:
        groups.reverse()
    with pytest.raises(SystemExit) as error:
        parse_args(sum(groups, []))
    assert error.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--x-axis-scale", "--y-axis-scale", "--z-axis-scale", "--elem-size", "--target-physical-x"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_lengths(flag, value):
    with pytest.raises(SystemExit):
        parse_args([f"{flag}={value}"])


@pytest.mark.parametrize("flag", ["--nelx", "--nely", "--nelz", "--target-nelx"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_invalid_counts(flag, value):
    with pytest.raises(SystemExit):
        parse_args([flag, value])


def test_axis_lengths_resolve_after_final_counts_and_preserve_unspecified_axes():
    args = parse_args(["--nelx", "100", "--x-axis-scale", "1.0", "--z-axis-scale", "0.005", "--nelz", "1"])
    assert resolve_cli_element_size(args) == (0.01, 0.01, 0.005)
    args.nelx = 40  # Imported geometry can change counts after parsing.
    args.nely = 12
    args.nelz = 2
    assert resolve_cli_element_size(args) == (0.025, 0.01, 0.0025)
    assert resolve_cli_element_size(parse_args([])) == 0.01
    assert resolve_cli_element_size(parse_args(["--elem-size", "0.002"])) == 0.002
    assert resolve_cli_element_size(parse_args(["--nelx", "100", "--target-physical-x", "0.2"])) == 0.002


@pytest.mark.parametrize("size", [(1, 2), (1, 2, 3, 4), (1, 0, 1), (1, -1, 1), (1, np.inf, 1), (1, np.nan, 1), [[1, 2, 3]]])
def test_python_api_rejects_invalid_element_dimensions(size):
    with pytest.raises(ValueError, match="elem_size"):
        element_size_xyz(size)


def test_equal_xyz_edges_preserve_scalar_stiffness_and_integration():
    np.testing.assert_allclose(lk_H8(elem_size=0.01), lk_H8(elem_size=(0.01,) * 3), rtol=0, atol=0)
    for scalar, xyz in zip(h8_gauss_integration_data(0.01), h8_gauss_integration_data((0.01,) * 3)):
        np.testing.assert_array_equal(scalar, xyz)
    _, determinants, weights = h8_gauss_integration_data((0.01, 0.004, 0.002))
    assert np.sum(determinants * weights) == pytest.approx(0.01 * 0.004 * 0.002)


@pytest.mark.parametrize("mode,smooth", [("blocky", False), ("binary", False), ("density", False), ("density", True)])
@pytest.mark.parametrize("thin_axis", range(3))
@pytest.mark.parametrize("order", ["xyz", "yxz"])
def test_single_layer_exports_have_xyz_dimensions(mode, smooth, thin_axis, order):
    counts = [8, 6, 4]
    counts[thin_axis] = 1
    sizes = np.array([0.003, 0.002, 0.005])
    density = np.ones(counts)
    if order == "yxz":
        density = density.transpose(1, 0, 2)
    mesh = voxel_to_stl(density, export_mode=mode, elem_size=sizes, array_order=order, smooth_mesh=smooth)
    # Interpolated surfaces may move by at most one refined voxel per axis.
    tolerance_in_elements = 1 / 3 if mode == "density" else 1e-8
    np.testing.assert_allclose(mesh.extents / sizes, counts, rtol=0, atol=tolerance_in_elements)
    assert mesh.is_watertight
    assert mesh.is_volume
    if mode == "blocky":
        assert mesh.volume == pytest.approx(np.prod(np.array(counts) * sizes))


def test_rectangular_export_padding_uses_each_axis_length():
    density = np.ones((4, 3, 1))
    for mode in ("density", "binary", "blocky"):
        meshes = [voxel_to_stl(density, export_mode=mode, elem_size=(0.003, 0.002, 0.005), padding=p, smooth_mesh=False) for p in (1, 2)]
        np.testing.assert_allclose(meshes[0].bounds, meshes[1].bounds, atol=1e-8)


def test_rectangular_failure_postprocessing_reports_physical_volumes():
    density = np.array([[[1.0], [0.8], [0.3], [1.0]]])
    sizes = (0.01, 0.004, 0.002)
    result = evaluate_failure_representations(
        x_projected=density, binary_threshold=0.5, penal=3,
        material_params=get_material_params("orthotropic_validation"),
        strength=get_material_strength("orthotropic_validation"),
        orientation_matrix=np.eye(3), elem_size=sizes,
    )
    volume = np.prod(sizes)
    assert result.metrics["material_volume_m3_projected"] == pytest.approx(3.1 * volume)
    assert result.metrics["material_volume_m3_binary"] == pytest.approx(3 * volume)
    assert result.metrics["reference_volume_m3_projected"] == pytest.approx(4 * volume)


@pytest.mark.parametrize("mode", ["compliance", "compliance_failure_constrained"])
def test_single_layer_rectangular_optimization_with_in_plane_load(mode):
    force = np.zeros((2, 4, 1, 3))
    force[:, -1, :, 0] = 1.0
    diagnostics = {}
    density, _, compliance, _ = top3d(
        nelx=4, nely=2, nelz=1, volfrac=0.5, penal=3, rmin=1.5, disp_thres=0.5,
        elem_size=(0.01, 0.006, 0.004), force_field=force,
        material_params=get_material_params("orthotropic_validation"),
        material_strength=get_material_strength("orthotropic_validation"),
        material_orientation=np.eye(3), optimization_mode=mode,
        optimizer="mma" if mode != "compliance" else "oc",
        maxloop=6, beta_schedule=(1.0,), diagnostics_out=diagnostics,
    )
    assert density.shape == (2, 4, 1)
    assert np.all(np.isfinite(density))
    assert np.isfinite(compliance) and compliance > 0
    assert diagnostics["physical_density_fraction"] == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("animation", [False, True])
def test_visualizations_use_physical_coordinates_and_aspect(animation):
    density = np.ones((2, 4, 1))
    sizes = (0.01, 0.006, 0.004)
    if animation:
        fig = create_frame(density, elem_size=sizes)
    else:
        plt.figure()
        fig = display_3D(density, elem_size=sizes)
    try:
        ax = fig.axes[0]
        expected = [0.04, 0.012, 0.004]
        np.testing.assert_allclose([ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]], expected)
        np.testing.assert_allclose(ax.get_box_aspect() / ax.get_box_aspect()[0], np.array(expected) / expected[0])
        assert ax.get_xlabel() == "X (m)"
        fig.canvas.draw()
    finally:
        plt.close("all")


@pytest.mark.parametrize("module_name", ["main", "pytopo3d.cli.command"])
@pytest.mark.parametrize("skip", [False, True])
def test_entry_points_save_resolved_geometry_after_stl_import(module_name, skip, monkeypatch, tmp_path):
    entry = importlib.import_module(module_name)
    manager = ResultsManager(base_dir=str(tmp_path), experiment_name="axis_scale")
    monkeypatch.setattr(entry, "setup_experiment", lambda **kwargs: (logging.getLogger("axis_scale_test"), manager))
    # Exercise the real loader's XYZ -> YXZ path with different imported counts.
    geometry = importlib.import_module("pytopo3d.preprocessing.geometry")
    monkeypatch.setattr(geometry, "stl_to_design_space", lambda *args, **kwargs: np.ones((4, 2, 1), dtype=bool))
    monkeypatch.setattr(manager, "copy_file", lambda *args, **kwargs: None)
    argv = [
        "--nelx", "100", "--nely", "50", "--nelz", "3",
        "--design-space-stl", "domain.stl", "--target-nelx", "4",
        "--x-axis-scale", "0.04", "--y-axis-scale", "0.012", "--z-axis-scale", "0.004",
        "--material-preset", "orthotropic_validation", "--eval-material-presets", "orthotropic_validation",
        "--volfrac", "0.5", "--rmin", "1.5",
        "--maxloop", "2", "--beta-schedule", "1", "--export-stl", "--export-mode", "blocky",
    ]
    if skip:
        argv.append("--skip-optimization")
    else:
        argv.extend(["--create-animation", "--animation-frequency", "1"])
    if module_name == "main":
        monkeypatch.setattr(sys, "argv", ["main.py", *argv])
        code = entry.main()
    else:
        code = entry.main(argv)
    assert code == 0
    directory = tmp_path / "axis_scale"
    for filename in ("config.json", "metrics.json"):
        saved = json.loads((directory / filename).read_text())
        assert saved["x_axis_scale"] == 0.04
        assert saved["y_axis_scale"] == 0.012
        assert saved["z_axis_scale"] == 0.004
        assert saved["element_size_xyz_m"] == [0.01, 0.006, 0.004]
        assert saved["domain_size_xyz_m"] == [0.04, 0.012, 0.004]
    density = np.load(directory / "optimized_design.npy")
    assert density.shape == (2, 4, 1)
    if not skip:
        assert (directory / "visualizations" / "optimization_animation.gif").is_file()
    mesh = trimesh.load_mesh(directory / "optimized_design.stl")
    assert mesh.extents[2] == pytest.approx(0.004)
    if skip:
        np.testing.assert_allclose(mesh.extents, [0.04, 0.012, 0.004], atol=1e-9)
