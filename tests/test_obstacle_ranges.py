import json

import numpy as np
import pytest

from pytopo3d.cli.parser import parse_args
from pytopo3d.preprocessing import geometry
from pytopo3d.utils.metrics import collect_metrics
from pytopo3d.utils.obstacles import create_range_obstacle


def test_create_range_obstacle_uses_xyz_half_open_coordinates():
    mask = create_range_obstacle((5, 6, 4), (1, 4, 2, 5, 1, 3))

    expected = np.zeros((5, 6, 4), dtype=bool)
    expected[2:5, 1:4, 1:3] = True
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.parametrize(
    "coordinate_range, message",
    [
        ((-1, 2, 0, 1, 0, 1), "Obstacle x range"),
        ((1, 1, 0, 1, 0, 1), "Obstacle x range"),
        ((0, 2, 0, 6, 0, 1), "Obstacle y range"),
        ((0, 2, 0, 1, 0, 5), "Obstacle z range"),
        ((0, 2, 0, 1, 0, None), "z2 must be an integer"),
    ],
)
def test_create_range_obstacle_rejects_invalid_ranges(coordinate_range, message):
    with pytest.raises(ValueError, match=message):
        create_range_obstacle((5, 6, 4), coordinate_range)


def test_yaml_ranges_are_combined_with_json_obstacles(tmp_path, monkeypatch):
    json_path = tmp_path / "obstacles.json"
    json_path.write_text(
        json.dumps(
            {
                "obstacles": [
                    {"type": "cube", "center": [0.5, 0.5, 0.5], "size": 0.5}
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        geometry,
        "get_obstacle_ranges",
        lambda names: ((0, 2, 0, 1, 0, 1),),
    )

    _, obstacle_mask, combined_mask, nelx, nely, nelz = geometry.load_geometry_data(
        nelx=6,
        nely=4,
        nelz=4,
        obstacle_config=str(json_path),
        obstacle_ranges=["fixture_clearance"],
    )

    assert (nelx, nely, nelz) == (6, 4, 4)
    assert np.all(obstacle_mask[0:1, 0:2, 0:1])
    assert obstacle_mask[2, 3, 2]
    np.testing.assert_array_equal(combined_mask, obstacle_mask)


def test_yaml_ranges_use_xyz_coordinates_after_stl_axis_normalization(monkeypatch):
    monkeypatch.setattr(
        geometry,
        "stl_to_design_space",
        lambda *args, **kwargs: np.ones((6, 4, 2), dtype=bool),
    )
    monkeypatch.setattr(
        geometry,
        "get_obstacle_ranges",
        lambda names: ((4, 6, 1, 3, 0, 2),),
    )

    _, obstacle_mask, combined_mask, nelx, nely, nelz = geometry.load_geometry_data(
        nelx=1,
        nely=1,
        nelz=1,
        design_space_stl="domain.stl",
        obstacle_ranges=["clearance"],
    )

    assert (nelx, nely, nelz) == (6, 4, 2)
    expected = np.zeros((4, 6, 2), dtype=bool)
    expected[1:3, 4:6, 0:2] = True
    np.testing.assert_array_equal(obstacle_mask, expected)
    np.testing.assert_array_equal(combined_mask, expected)


def test_cli_accepts_multiple_obstacle_range_presets():
    args = parse_args(["--obstacle-ranges", "first", "second"])

    assert args.obstacle_ranges == ["first", "second"]


def test_metrics_report_yaml_obstacle_ranges():
    obstacle_mask = np.zeros((2, 3, 1), dtype=bool)
    obstacle_mask[0, :2, 0] = True

    metrics = collect_metrics(
        terminal_input="--obstacle-ranges channel",
        nelx=3,
        nely=2,
        nelz=1,
        volfrac=0.3,
        penal=3.0,
        rmin=1.5,
        disp_thres=0.5,
        obstacle_ranges=["channel"],
        obstacle_mask=obstacle_mask,
    )

    assert metrics["has_obstacles"] is True
    assert metrics["obstacle_ranges"] == ["channel"]
    assert metrics["obstacle_elements"] == 2
