from dataclasses import replace
import json

import numpy as np
import pytest

import pytopo3d.core.optimizer as optimizer_module
from pytopo3d.core.optimizer import top3d
from pytopo3d.utils.config_loader import (
    apply_material_orientation,
    get_material_params,
    get_material_strength,
    material_orientation_matrix,
)


def _strength_problem(orientation="xyz"):
    return {
        "material_params": apply_material_orientation(
            get_material_params("orthotropic_validation"),
            orientation,
        ),
        "material_strength": get_material_strength("orthotropic_validation"),
        "material_orientation": material_orientation_matrix(orientation),
    }


def _run_mixed_failure_problem(load_scale, *, save_history=False):
    nelx, nely, nelz = 8, 4, 1
    force_field = np.zeros((nely, nelx, nelz, 3))
    force_field[1, -1, 0] = load_scale * np.array([30.0, -17.0, -43.0])
    diagnostics = {}
    density, history, compliance, failure_load = top3d(
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        volfrac=0.5,
        penal=3.0,
        rmin=1.6,
        disp_thres=0.5,
        elem_size=0.01,
        force_field=force_field,
        tolx=2.0e-3,
        maxloop=100,
        save_history=save_history,
        history_frequency=9,
        beta_schedule=(2.0,),
        optimization_mode="compliance_failure_constrained",
        optimizer="mma",
        mma_move=0.015,
        diagnostics_out=diagnostics,
        **_strength_problem("yzx"),
    )
    return density, history, compliance, failure_load, diagnostics


def test_compliance_mma_matches_oc_scale_and_controls_physical_volume():
    common = {
        "nelx": 4,
        "nely": 2,
        "nelz": 1,
        "volfrac": 0.5,
        "penal": 3.0,
        "rmin": 1.5,
        "disp_thres": 0.5,
        "tolx": 2.0e-3,
        "maxloop": 80,
        "beta_schedule": (2.0,),
    }
    oc_density, _, oc_compliance, _ = top3d(**common)
    diagnostics = {}
    mma_density, _, mma_compliance, _ = top3d(
        **common,
        optimizer="mma",
        mma_move=0.05,
        diagnostics_out=diagnostics,
    )

    assert mma_compliance <= 1.25 * oc_compliance
    assert diagnostics["physical_density_fraction"] == pytest.approx(
        0.5, abs=2.0e-3
    )
    assert diagnostics["optimization_feasible"] is True
    assert diagnostics["volume_constraint"] <= 2.0e-3
    assert np.all((0.0 <= mma_density) & (mma_density <= 1.0))
    assert np.all(oc_density[:, 0, :] == 1.0)
    assert np.all(mma_density[:, 0, :] == 1.0)
    assert np.all(mma_density[:, -1, 0] == 1.0)


def test_higher_feasible_load_activates_failure_and_changes_geometry():
    low_density, _, _, _, low = _run_mixed_failure_problem(1.0)
    high_density, _, _, failure_load, high = _run_mixed_failure_problem(5.0)

    assert low["failure_aggregate"] < 0.3
    assert 0.9 <= high["failure_aggregate"] <= 1.001
    assert high["failure_constraint"] <= 1.0e-3
    assert high["optimization_feasible"] is True
    assert high["physical_density_fraction"] == pytest.approx(0.5, abs=2.0e-3)
    assert np.max(np.abs(high_density - low_density)) > 0.03
    assert failure_load == pytest.approx(high["predicted_failure_load"])
    assert high["predicted_failure_load"] > 1.1 * low["predicted_failure_load"]


def test_failure_mma_history_records_each_requested_diagnostic():
    _, history, _, _, diagnostics = _run_mixed_failure_problem(
        1.0, save_history=True
    )

    required = {
        "failure_aggregate_history",
        "failure_exact_max_history",
        "failure_constraint_history",
        "failure_constraint_violation_history",
        "critical_failure_mode_history",
        "critical_element_history",
        "predicted_failure_load_history",
        "mma_kkt_residual_history",
    }
    assert required <= history.keys()
    assert len({len(values) for values in history.values()}) == 1
    assert history["failure_aggregate_history"][-1] == pytest.approx(
        diagnostics["failure_aggregate"]
    )
    assert history["mma_subproblem_kkt_residual_history"][-1] is None
    assert history["critical_element_history"][-1] >= 0
    assert history["critical_failure_mode_history"][-1]


def test_grossly_infeasible_failure_problem_is_not_reported_as_success():
    force_field = np.zeros((2, 4, 1, 3))
    force_field[0, -1, 0, 2] = -1.0e6
    diagnostics = {}

    top3d(
        nelx=4,
        nely=2,
        nelz=1,
        volfrac=0.5,
        penal=3.0,
        rmin=1.5,
        disp_thres=0.5,
        elem_size=0.01,
        force_field=force_field,
        maxloop=8,
        beta_schedule=(2.0,),
        optimization_mode="compliance_failure_constrained",
        optimizer="mma",
        mma_move=0.02,
        diagnostics_out=diagnostics,
        **_strength_problem(),
    )

    assert diagnostics["optimization_feasible"] is False
    assert diagnostics["max_constraint_violation"] > 1.0
    assert diagnostics["failure_constraint"] > 1.0
    assert diagnostics["termination_status"] in {
        "stalled_infeasible_or_not_found",
        "subproblem_failed",
        "continuation_stage_infeasible",
    }
    assert diagnostics["least_violation_iteration"] is not None


def test_failure_mode_rejects_unsafe_optimizer_and_density_combinations():
    common = {
        "nelx": 4,
        "nely": 2,
        "nelz": 1,
        "volfrac": 0.5,
        "penal": 3.0,
        "rmin": 1.5,
        "disp_thres": 0.5,
        "maxloop": 1,
        "beta_schedule": (2.0,),
        "optimization_mode": "compliance_failure_constrained",
    }
    strength_problem = _strength_problem()

    with pytest.raises(ValueError, match="requires optimizer='mma'"):
        top3d(**common, optimizer="oc", **strength_problem)
    with pytest.raises(ValueError, match="requires validated material strength"):
        top3d(
            **common,
            optimizer="mma",
            material_params=strength_problem["material_params"],
        )
    with pytest.raises(ValueError, match="positive.*mma_min_density"):
        top3d(
            **common,
            optimizer="mma",
            mma_min_density=0.0,
            **strength_problem,
        )


def test_failed_mma_stage_reports_the_last_executed_projection(monkeypatch):
    real_mma_update = optimizer_module.mma_update

    def fail_after_solving(*args, **kwargs):
        result = real_mma_update(*args, **kwargs)
        return replace(
            result,
            diagnostics=replace(result.diagnostics, dual_success=False),
        )

    monkeypatch.setattr(optimizer_module, "mma_update", fail_after_solving)
    diagnostics = {}
    top3d(
        nelx=4,
        nely=2,
        nelz=1,
        volfrac=0.5,
        penal=3.0,
        rmin=1.5,
        disp_thres=0.5,
        maxloop=2,
        beta_schedule=(1.0, 8.0),
        optimizer="mma",
        diagnostics_out=diagnostics,
    )

    assert diagnostics["termination_status"] == "subproblem_failed"
    assert diagnostics["projection_beta"] == 1.0
    assert [
        stage["beta"] for stage in diagnostics["projection_stage_summaries"]
    ] == [1.0]
    assert diagnostics["projected_compliance"] == pytest.approx(
        diagnostics["mma_iteration_history"][-1]["compliance"], rel=1.0e-10
    )


def test_single_evaluation_mma_diagnostics_are_strict_json():
    diagnostics = {}
    top3d(
        nelx=4,
        nely=2,
        nelz=1,
        volfrac=0.5,
        penal=3.0,
        rmin=1.5,
        disp_thres=0.5,
        maxloop=1,
        beta_schedule=(2.0,),
        optimizer="mma",
        diagnostics_out=diagnostics,
    )

    assert diagnostics["projection_stage_summaries"][0]["final_change"] == 0.0
    json.dumps(diagnostics, allow_nan=False)


def test_mma_retries_a_numerical_exception_with_a_smaller_move(monkeypatch):
    real_mma_update = optimizer_module.mma_update
    calls = {"count": 0}

    def fail_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("synthetic dual failure")
        return real_mma_update(*args, **kwargs)

    monkeypatch.setattr(optimizer_module, "mma_update", fail_once)
    diagnostics = {}
    top3d(
        nelx=4,
        nely=2,
        nelz=1,
        volfrac=0.5,
        penal=3.0,
        rmin=1.5,
        disp_thres=0.5,
        maxloop=2,
        beta_schedule=(2.0,),
        optimizer="mma",
        mma_move=0.05,
        diagnostics_out=diagnostics,
    )

    attempts = diagnostics["mma_iteration_history"][0][
        "mma_subproblem_attempts"
    ]
    assert attempts[0]["error_type"] == "RuntimeError"
    assert attempts[0]["move_limit"] == pytest.approx(0.05)
    assert attempts[1]["success"] is True
    assert attempts[1]["move_limit"] == pytest.approx(0.025)


def test_all_mma_retry_exceptions_are_reported_without_escaping(monkeypatch):
    def always_fail(*args, **kwargs):
        raise RuntimeError("synthetic persistent dual failure")

    monkeypatch.setattr(optimizer_module, "mma_update", always_fail)
    diagnostics = {}
    top3d(
        nelx=4,
        nely=2,
        nelz=1,
        volfrac=0.5,
        penal=3.0,
        rmin=1.5,
        disp_thres=0.5,
        maxloop=2,
        beta_schedule=(2.0,),
        optimizer="mma",
        mma_move=0.05,
        diagnostics_out=diagnostics,
    )

    record = diagnostics["mma_iteration_history"][0]
    assert diagnostics["termination_status"] == "subproblem_failed"
    assert len(record["mma_subproblem_attempts"]) == 3
    assert record["mma_effective_move_limit"] == pytest.approx(0.0125)
    assert record["mma_subproblem_kkt_residual"] is None
    json.dumps(diagnostics, allow_nan=False)


def test_feasible_snapshot_recovers_a_failed_stage_and_continues(monkeypatch):
    real_mma_update = optimizer_module.mma_update
    calls = {"count": 0}

    def fail_one_complete_retry_set(*args, **kwargs):
        calls["count"] += 1
        result = real_mma_update(*args, **kwargs)
        if 5 <= calls["count"] <= 7:
            return replace(
                result,
                diagnostics=replace(result.diagnostics, dual_success=False),
            )
        return result

    monkeypatch.setattr(
        optimizer_module,
        "mma_update",
        fail_one_complete_retry_set,
    )
    diagnostics = {}
    top3d(
        nelx=4,
        nely=2,
        nelz=1,
        volfrac=0.5,
        penal=3.0,
        rmin=1.5,
        disp_thres=0.5,
        maxloop=10,
        beta_schedule=(1.0, 2.0),
        optimizer="mma",
        mma_move=0.05,
        diagnostics_out=diagnostics,
    )

    stages = diagnostics["projection_stage_summaries"]
    assert len(stages) == 2
    assert stages[0]["subproblem_failed"] is True
    assert stages[0]["subproblem_failure_recovered"] is True
    assert stages[0]["best_feasible_iteration"] is not None
    assert diagnostics["continuation_stages_completed"] == 2
