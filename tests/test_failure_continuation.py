import json

import numpy as np
import pytest

from pytopo3d.analysis.postprocessing import evaluate_failure_representations
from pytopo3d.core.optimizer import top3d
from pytopo3d.utils.config_loader import (
    apply_material_orientation,
    get_material_params,
    get_material_strength,
    material_orientation_matrix,
)


ORIENTATION = "yzx"
MATERIAL_PARAMS = apply_material_orientation(
    get_material_params("orthotropic_validation"),
    ORIENTATION,
)
STRENGTH = get_material_strength("orthotropic_validation")
ORIENTATION_MATRIX = material_orientation_matrix(ORIENTATION)


def _run_continuation(perturbation_sign):
    nelx, nely, nelz = 8, 4, 1
    force_field = np.zeros((nely, nelx, nelz, 3))
    force_field[1, -1, 0] = 4.0 * np.array([30.0, -17.0, -43.0])
    element_index = np.arange(nelx * nely * nelz).reshape(
        (nely, nelx, nelz), order="F"
    )
    initial_design = 0.5 + perturbation_sign * 0.005 * np.sin(
        0.7 * element_index + 0.2
    )
    diagnostics = {}
    density, history, compliance, _ = top3d(
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        volfrac=0.5,
        penal=3.0,
        rmin=1.6,
        disp_thres=0.5,
        material_params=MATERIAL_PARAMS,
        material_strength=STRENGTH,
        material_orientation=ORIENTATION_MATRIX,
        elem_size=0.01,
        force_field=force_field,
        tolx=3.0e-3,
        maxloop=80,
        save_history=True,
        history_frequency=20,
        beta_schedule=(1.0, 2.0, 4.0, 8.0),
        failure_limit_schedule=(1.5, 1.25, 1.1, 1.0),
        failure_aggregate_exponent_schedule=(4.0, 6.0, 8.0, 8.0),
        optimization_mode="compliance_failure_constrained",
        optimizer="mma",
        mma_move=0.012,
        initial_design=initial_design,
        diagnostics_out=diagnostics,
    )
    return density, history, compliance, diagnostics


@pytest.fixture(scope="module")
def continuation_runs():
    return _run_continuation(-1.0), _run_continuation(1.0)


@pytest.fixture(scope="module")
def binary_closeness_run():
    nelx, nely, nelz = 8, 4, 1
    force_field = np.zeros((nely, nelx, nelz, 3))
    force_field[1, -1, 0] = 20.0 * np.array([30.0, -17.0, 0.0])
    diagnostics = {}
    density, _, _, _ = top3d(
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        volfrac=0.5,
        penal=3.0,
        rmin=1.6,
        disp_thres=0.5,
        material_params=MATERIAL_PARAMS,
        material_strength=STRENGTH,
        material_orientation=ORIENTATION_MATRIX,
        elem_size=0.01,
        force_field=force_field,
        tolx=3.0e-3,
        maxloop=80,
        beta_schedule=(1.0, 2.0, 4.0, 8.0),
        failure_limit_schedule=(1.5, 1.25, 1.1, 1.0),
        failure_aggregate_exponent_schedule=(4.0, 6.0, 8.0, 8.0),
        optimization_mode="compliance_failure_constrained",
        optimizer="mma",
        mma_move=0.012,
        diagnostics_out=diagnostics,
    )
    verification = evaluate_failure_representations(
        x_projected=density,
        binary_threshold=0.5,
        penal=3.0,
        material_params=MATERIAL_PARAMS,
        strength=STRENGTH,
        orientation_matrix=ORIENTATION_MATRIX,
        elem_size=0.01,
        force_field=force_field,
        smooth_failure_aggregate=diagnostics["failure_aggregate"],
        smooth_failure_limit=diagnostics["failure_limit"],
    )
    return diagnostics, verification


def test_limit_p_and_projection_continuation_recovers_each_stage(
    continuation_runs,
):
    density, history, _, diagnostics = continuation_runs[0]
    stages = diagnostics["projection_stage_summaries"]

    assert diagnostics["continuation_completed"] is True
    assert diagnostics["continuation_stages_completed"] == 4
    assert diagnostics["failure_limit_schedule"] == [1.5, 1.25, 1.1, 1.0]
    assert diagnostics["failure_aggregate_exponent_schedule"] == [
        4.0,
        6.0,
        8.0,
        8.0,
    ]
    assert [stage["beta"] for stage in stages] == [1.0, 2.0, 4.0, 8.0]
    assert all(stage["continuation_feasible"] for stage in stages)
    assert diagnostics["optimization_feasible"] is True
    assert diagnostics["failure_constraint"] <= 1.0e-3
    assert 0.85 <= diagnostics["failure_aggregate"] <= 1.001
    assert abs(
        diagnostics["failure_exact_max"] - diagnostics["failure_aggregate"]
    ) < 0.2
    assert diagnostics["physical_density_fraction"] == pytest.approx(
        0.5, abs=3.0e-3
    )
    assert np.all((0.0 <= density) & (density <= 1.0))

    # The beta=8 transition is deliberately difficult in this fixture. It may
    # begin violated, but the violation must not be permanent or oscillatory.
    assert stages[-1]["initial_constraint_violation"] > 0.1
    assert stages[-1]["max_constraint_violation"] <= 1.0e-3
    assert all(
        stage["failure_constraint_tail_peak_to_peak"] < 0.35
        for stage in stages
    )

    for previous, following in zip(stages, stages[1:]):
        assert following["stage_start_design_checksum"] == pytest.approx(
            previous["stage_end_design_checksum"], abs=1.0e-14
        )

    assert len({len(values) for values in history.values()}) == 1
    assert set(history["failure_limit_history"]) == {1.0, 1.1, 1.25, 1.5}
    assert set(history["failure_aggregate_exponent_history"]) == {
        4.0,
        6.0,
        8.0,
    }
    assert "restoration" in history["mma_record_kind_history"]
    assert any(
        value is not None
        for value in history["mma_restored_from_iteration_history"]
    )
    json.dumps(diagnostics, allow_nan=False)


def test_exact_binary_failure_is_close_on_a_connected_load_case(
    binary_closeness_run,
):
    diagnostics, verification = binary_closeness_run
    binary_failure = verification.metrics["failure_index_max_binary"]
    smooth_failure = diagnostics["failure_aggregate"]

    assert diagnostics["continuation_completed"] is True
    assert diagnostics["optimization_feasible"] is True
    assert binary_failure <= 1.0
    assert verification.metrics["failure_strength_feasible_binary"] is True
    assert verification.metrics["smooth_failure_feasible"] is True
    assert verification.metrics["smooth_to_binary_failure_mismatch"] is False
    assert verification.metrics["stage10_internal_verification_passed"] is True
    assert abs(binary_failure - smooth_failure) / smooth_failure < 0.25
    assert verification.binary_response["compliance"] < (
        10.0 * verification.projected_response["compliance"]
    )


def test_small_initial_perturbations_give_similar_results(continuation_runs):
    negative, positive = continuation_runs
    negative_density, _, negative_compliance, negative_metrics = negative
    positive_density, _, positive_compliance, positive_metrics = positive

    relative_compliance_difference = abs(
        negative_compliance - positive_compliance
    ) / max(negative_compliance, positive_compliance)
    assert relative_compliance_difference < 0.08
    assert np.mean(np.abs(negative_density - positive_density)) < 0.04
    assert abs(
        negative_metrics["failure_aggregate"]
        - positive_metrics["failure_aggregate"]
    ) < 0.08
    assert negative_metrics["continuation_completed"] is True
    assert positive_metrics["continuation_completed"] is True


def test_continuation_schedules_and_initial_design_are_validated():
    common = {
        "nelx": 4,
        "nely": 2,
        "nelz": 1,
        "volfrac": 0.5,
        "penal": 3.0,
        "rmin": 1.5,
        "disp_thres": 0.5,
        "maxloop": 1,
        "beta_schedule": (1.0, 2.0),
        "optimization_mode": "compliance_failure_constrained",
        "optimizer": "mma",
        "material_params": MATERIAL_PARAMS,
        "material_strength": STRENGTH,
        "material_orientation": ORIENTATION_MATRIX,
    }
    with pytest.raises(ValueError, match="one value or 2 values"):
        top3d(**common, failure_limit_schedule=(1.5, 1.2, 1.0))
    with pytest.raises(ValueError, match="nonincreasing"):
        top3d(**common, failure_limit_schedule=(1.0, 1.2))
    with pytest.raises(ValueError, match="nondecreasing"):
        top3d(
            **common,
            failure_aggregate_exponent_schedule=(8.0, 4.0),
        )
    with pytest.raises(ValueError, match="beta_schedule must be nondecreasing"):
        top3d(**{**common, "beta_schedule": (2.0, 1.0)})
    with pytest.raises(ValueError, match="initial_design has shape"):
        top3d(**common, initial_design=np.full((2, 3, 1), 0.5))
    with pytest.raises(ValueError, match="at least"):
        top3d(**common, initial_design=np.zeros((2, 4, 1)))
    with pytest.raises(ValueError, match="schedules require"):
        top3d(
            nelx=4,
            nely=2,
            nelz=1,
            volfrac=0.5,
            penal=3.0,
            rmin=1.5,
            disp_thres=0.5,
            failure_limit_schedule=(1.5,),
        )
