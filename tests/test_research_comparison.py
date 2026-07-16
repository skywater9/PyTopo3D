import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pytopo3d.runners.research_comparison import (
    ComparisonInputs,
    ComparisonProtocol,
    _assembled_reference_load_N,
    attach_external_validation,
    binary_topology_metrics,
    build_design_variants,
    isotropize_orthotropic_material,
    load_ansys_results_json,
    load_comparison_protocol,
    load_experimental_measurements_csv,
    run_research_comparison,
    validate_ansys_results,
    validate_experimental_measurements,
)
from pytopo3d.utils.config_loader import (
    apply_material_orientation,
    get_material_params,
    get_material_strength,
    material_orientation_matrix,
)


ORIENTATION = "yzx"
MATERIAL_AXES = tuple(get_material_params("orthotropic_validation"))
ISOTROPIC = isotropize_orthotropic_material(MATERIAL_AXES)
ORTHOTROPIC_GLOBAL = tuple(
    apply_material_orientation(MATERIAL_AXES, ORIENTATION)
)
STRENGTH = get_material_strength("orthotropic_validation")
ROTATION = material_orientation_matrix(ORIENTATION)
COORDINATE_FRAME = "PyTopo3D global XYZ, origin at mesh minimum"


def _sha256(label):
    return hashlib.sha256(str(label).encode("utf-8")).hexdigest()


def _array_sha256(array):
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _protocol(comparison_id="abc_contract", **overrides):
    values = {
        "comparison_id": comparison_id,
        "load_case_name": "mixed_cantilever",
        "code_version_id": "stage11-test-contract-v1",
        "material_system_id": "orthotropic_validation",
        "material_data_provenance": "synthetic",
        "isotropic_optimizer_material_id": (
            "orthotropic_validation_axis_average_E_nu"
        ),
        "orthotropic_material_id": "orthotropic_validation",
        "nelx": 4,
        "nely": 2,
        "nelz": 1,
        "elem_size_m": 0.01,
        "volfrac": 0.5,
        "penal": 3.0,
        "rmin": 1.2,
        "print_orientation_xyz": ORIENTATION,
        "tolx": 0.05,
        "maxloop": 4,
        "beta_schedule": (1.0,),
        "failure_limit_schedule": (1.0,),
        "failure_aggregate_exponent_schedule": (4.0,),
        "mma_move": 0.02,
        "binary_volume_tolerance": 0.2,
        "minimum_experimental_replicates": 1,
    }
    values.update(overrides)
    return ComparisonProtocol(**values)


def _inputs(**overrides):
    shape = (2, 4, 1)
    force = np.zeros(shape + (3,))
    force[0, -1, 0] = [100.0, -50.0, 0.0]
    support = np.zeros(shape, dtype=bool)
    support[:, 0, :] = True
    values = {
        "orthotropic_material_params_material_axes": MATERIAL_AXES,
        "material_strength": STRENGTH,
        "material_orientation": ROTATION,
        "force_field": force,
        "support_mask": support,
        "obstacle_mask": np.zeros(shape, dtype=bool),
        "protected_zone_mask": np.zeros(shape, dtype=bool),
        "initial_design": np.full(shape, 0.5),
    }
    values.update(overrides)
    return ComparisonInputs(**values)


def _mock_functions():
    optimizer_calls = []
    evaluation_calls = []
    free_locations = [
        (0, 1, 0),
        (1, 1, 0),
        (0, 2, 0),
        (1, 2, 0),
        (1, 3, 0),
    ]
    projected_values = (
        (0.2, 0.3, 0.5, 0.7, 0.8),
        (0.8, 0.2, 0.7, 0.3, 0.5),
        (0.3, 0.8, 0.2, 0.7, 0.5),
    )
    occupied = ((0, 1), (0, 2), (3, 4))
    critical_locations = ((0, 1, 0), (0, 1, 0), (1, 2, 0))

    def optimize(**kwargs):
        index = len(optimizer_calls)
        optimizer_calls.append(kwargs)
        density = np.ones((2, 4, 1))
        for location, value in zip(free_locations, projected_values[index]):
            density[location] = value
        diagnostics = kwargs["diagnostics_out"]
        diagnostics.update(
            {
                "optimization_feasible": True,
                "continuation_completed": True,
                "continuation_stages_requested": 1,
                "continuation_stages_completed": 1,
                "projection_converged": True,
                "termination_status": "converged",
            }
        )
        if kwargs["optimization_mode"] == "compliance_failure_constrained":
            diagnostics.update(
                {
                    "failure_aggregate": 0.8,
                    "failure_limit": 1.0,
                    "failure_constraint": -0.2,
                }
            )
        return density, None, float(index + 1), None

    def evaluate(**kwargs):
        index = len(evaluation_calls)
        evaluation_calls.append(kwargs)
        binary = np.ones((2, 4, 1))
        for location in free_locations:
            binary[location] = 0.0
        for free_index in occupied[index]:
            binary[free_locations[free_index]] = 1.0
        failure_index = (1.4, 1.2, 0.8)[index]
        feasible = failure_index <= 1.0
        metrics = {
            "compliance_projected": float(20 + index),
            "compliance_binary": float(10 + index),
            "predicted_stiffness_binary": float(100 - 10 * index),
            "failure_index_max_binary": failure_index,
            "failure_reference_load_binary": 150.0,
            "predicted_failure_load_binary": 150.0 / failure_index,
            "critical_mode_binary": "Y tension",
            "critical_element_yxz_binary": list(critical_locations[index]),
            "critical_region_binary": "design",
            "failure_strength_feasible_binary": feasible,
            "stage10_internal_verification_passed": feasible,
            "volume_fraction_binary": float(np.mean(binary)),
            "material_volume_m3_binary": float(np.sum(binary) * 1.0e-6),
        }
        return SimpleNamespace(binary_density=binary, metrics=metrics)

    return optimize, evaluate, optimizer_calls, evaluation_calls


def _critical_xyz(protocol, design_record):
    y_index, x_index, z_index = design_record["common_evaluation"][
        "critical_element_yxz_binary"
    ]
    return [
        (x_index + 0.5) * protocol.elem_size_m,
        (y_index + 0.5) * protocol.elem_size_m,
        (z_index + 0.5) * protocol.elem_size_m,
    ]


def _experimental_record(protocol, report, design_id, specimen_id):
    design = report["designs"][design_id]
    metrics = design["common_evaluation"]
    fracture_x, fracture_y, fracture_z = _critical_xyz(protocol, design)
    artifact_hash = _sha256(f"manufactured-{design_id}")
    return {
        "case_id": protocol.comparison_id,
        "common_case_sha256": report["common_case_sha256"],
        "design_id": design_id,
        "specimen_id": specimen_id,
        "binary_topology_sha256": design["binary_topology_sha256"],
        "manufacturing_artifact_path": f"source/manufactured-{design_id}.stl",
        "manufacturing_artifact_sha256": artifact_hash,
        "raw_data_sha256": _sha256(f"raw-{specimen_id}"),
        "print_orientation_xyz": protocol.print_orientation_xyz,
        "stiffness_observable": protocol.stiffness_observable,
        "failure_force_observable": protocol.failure_force_observable,
        "experimental_stiffness_N_per_m": metrics[
            "predicted_stiffness_binary"
        ],
        "experimental_failure_force_N": metrics[
            "predicted_failure_load_binary"
        ],
        "fracture_location_x_m": fracture_x,
        "fracture_location_y_m": fracture_y,
        "fracture_location_z_m": fracture_z,
        "fracture_region": metrics["critical_region_binary"],
        "mass_kg": 0.01,
        "volume_m3": metrics["material_volume_m3_binary"],
        "raw_data_path": f"raw/{specimen_id}.csv",
        "notes": "test evidence",
    }


def _ansys_record(protocol, report, design_id):
    design = report["designs"][design_id]
    metrics = design["common_evaluation"]
    return {
        "comparison_id": protocol.comparison_id,
        "common_case_sha256": report["common_case_sha256"],
        "design_id": design_id,
        "binary_topology_sha256": design["binary_topology_sha256"],
        "geometry_artifact_path": f"source/ansys-geometry-{design_id}.stl",
        "geometry_artifact_sha256": _sha256(f"manufactured-{design_id}"),
        "result_artifact_path": f"source/ansys-result-{design_id}.json",
        "result_artifact_sha256": _sha256(f"ansys-result-{design_id}"),
        "solver": "ANSYS Mechanical",
        "solver_version": "2026 R1",
        "mesh_description": "independent quadratic mesh",
        "failure_force_observable": protocol.failure_force_observable,
        "failure_index": metrics["failure_index_max_binary"],
        "predicted_failure_force_N": metrics[
            "predicted_failure_load_binary"
        ],
        "critical_mode": metrics["critical_mode_binary"],
        "critical_region": metrics["critical_region_binary"],
        "critical_location_xyz_m": _critical_xyz(protocol, design),
        "coordinate_frame": COORDINATE_FRAME,
        "location_mapping_method": "nearest PyTopo3D element center",
        "location_tolerance_m": (
            protocol.critical_location_tolerance_elements
            * protocol.elem_size_m
        ),
    }


def _bound_evidence(protocol, report, artifact_root=None, replicates=1):
    experiments = [
        _experimental_record(
            protocol,
            report,
            design_id,
            f"{design_id}-{replicate + 1:02d}",
        )
        for design_id in "ABC"
        for replicate in range(replicates)
    ]
    ansys = [
        _ansys_record(protocol, report, design_id) for design_id in "ABC"
    ]
    if artifact_root is not None:
        artifact_root = Path(artifact_root)
        for record in experiments:
            for path_field, hash_field in (
                ("manufacturing_artifact_path", "manufacturing_artifact_sha256"),
                ("raw_data_path", "raw_data_sha256"),
            ):
                path = artifact_root / record[path_field]
                path.parent.mkdir(parents=True, exist_ok=True)
                identity = (
                    record["design_id"]
                    if path_field == "manufacturing_artifact_path"
                    else record["specimen_id"]
                )
                content = f"{path_field}:{identity}".encode("utf-8")
                path.write_bytes(content)
                record[hash_field] = hashlib.sha256(content).hexdigest()
        for record in ansys:
            for path_field, hash_field in (
                ("geometry_artifact_path", "geometry_artifact_sha256"),
                ("result_artifact_path", "result_artifact_sha256"),
            ):
                path = artifact_root / record[path_field]
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == ".json":
                    content = json.dumps(
                        {"artifact": path_field, "design_id": record["design_id"]}
                    ).encode("utf-8")
                else:
                    content = f"{path_field}:{record['design_id']}".encode("utf-8")
                path.write_bytes(content)
                record[hash_field] = hashlib.sha256(content).hexdigest()
    return experiments, ansys


def _assert_all_json_strict(root):
    for path in Path(root).rglob("*.json"):
        loaded = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(loaded, allow_nan=False)


def test_abc_dispatch_common_evaluation_hashes_and_pending_status(tmp_path):
    optimize, evaluate, optimizer_calls, evaluation_calls = _mock_functions()
    protocol = _protocol()

    result = run_research_comparison(
        protocol,
        _inputs(),
        tmp_path,
        optimization_function=optimize,
        evaluation_function=evaluate,
    )

    assert result.simulation_gate_passed is True
    assert result.validation_status == "simulation_complete_external_pending"
    assert [call["optimization_mode"] for call in optimizer_calls] == [
        "compliance",
        "compliance",
        "compliance_failure_constrained",
    ]
    assert all(call["optimizer"] == "mma" for call in optimizer_calls)
    assert optimizer_calls[0]["material_params"] == ISOTROPIC
    assert optimizer_calls[1]["material_params"] == ORTHOTROPIC_GLOBAL
    assert optimizer_calls[2]["material_params"] == ORTHOTROPIC_GLOBAL
    assert "failure_limit_schedule" not in optimizer_calls[0]
    assert "failure_limit_schedule" not in optimizer_calls[1]
    assert optimizer_calls[2]["failure_limit_schedule"] == (1.0,)
    assert optimizer_calls[2]["material_strength"] == STRENGTH

    for key in (
        "force_field",
        "support_mask",
        "obstacle_mask",
        "protected_zone_mask",
        "initial_design",
    ):
        for call in optimizer_calls[1:]:
            np.testing.assert_array_equal(call[key], optimizer_calls[0][key])
    assert len(evaluation_calls) == 3
    assert all(
        call["material_params"] == ORTHOTROPIC_GLOBAL
        for call in evaluation_calls
    )
    assert all(call["strength"] == STRENGTH for call in evaluation_calls)
    assert all(
        np.array_equal(call["orientation_matrix"], ROTATION)
        for call in evaluation_calls
    )
    assert len(
        {call["results_manager"].experiment_dir for call in evaluation_calls}
    ) == 3

    records = result.report["designs"]
    assert records["A"]["optimizer_compliance"] == 1.0
    assert records["A"]["common_evaluation"]["compliance_binary"] == 10.0
    assert len({record["common_case_sha256"] for record in records.values()}) == 1
    assert len({record["binary_topology_sha256"] for record in records.values()}) == 3
    assert result.report["research_questions"][
        "experimental_evidence_complete"
    ] is False

    root = Path(result.output_directory)
    for design_id in "ABC":
        binary_path = root / f"design_{design_id}/binary_density.npy"
        assert records[design_id]["binary_topology_sha256"] == _array_sha256(
            np.load(binary_path)
        )
    manifest = json.loads(
        (root / "inputs/manifest.json").read_text(encoding="utf-8")
    )
    for metadata in manifest["arrays"].values():
        assert hashlib.sha256((root / metadata["path"]).read_bytes()).hexdigest() == (
            metadata["file_sha256"]
        )
    with (root / "experimental_measurements_template.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert list(csv.DictReader(handle)) == []
    _assert_all_json_strict(root)


def test_same_system_isotropic_surrogate_is_derived_deterministically():
    expected_young = np.mean(MATERIAL_AXES[:3])
    expected_poisson = np.mean(MATERIAL_AXES[6:9])
    expected_shear = expected_young / (2.0 * (1.0 + expected_poisson))

    np.testing.assert_allclose(ISOTROPIC[:3], expected_young)
    np.testing.assert_allclose(ISOTROPIC[3:6], expected_shear)
    np.testing.assert_allclose(ISOTROPIC[6:9], expected_poisson)
    variants = build_design_variants(_protocol(), _inputs())
    assert variants[0].material_params == ISOTROPIC
    assert variants[1].material_params == ORTHOTROPIC_GLOBAL
    assert variants[2].material_params == ORTHOTROPIC_GLOBAL


def test_common_reference_load_uses_assembled_nodal_force():
    protocol = _protocol("assembled_load", nelx=2, nely=1, nelz=1)
    force_field = np.zeros((1, 2, 1, 3))
    force_field[0, 0, 0, 0] = 1.0
    force_field[0, 1, 0, 0] = -1.0

    assert np.sum(np.abs(force_field)) == 2.0
    assert _assembled_reference_load_N(protocol, force_field) == pytest.approx(1.0)


def test_binary_topology_metrics_exclude_fixtures_and_are_exact():
    mask = np.zeros((1, 6, 1), dtype=bool)
    mask[0, 1:5, 0] = True
    a = np.zeros_like(mask)
    b = np.zeros_like(mask)
    a[0, [1, 2], 0] = True
    b[0, [1, 3], 0] = True
    a[0, 0, 0] = True
    b[0, 5, 0] = True

    metrics = binary_topology_metrics(a, b, mask, elem_size_m=0.01)
    assert metrics["binary_jaccard_similarity"] == pytest.approx(1.0 / 3.0)
    assert metrics["binary_dice_similarity"] == pytest.approx(0.5)
    assert metrics["binary_xor_fraction"] == pytest.approx(0.5)

    b[0, 0, 0] = True
    b[0, 5, 0] = False
    assert binary_topology_metrics(a, b, mask, elem_size_m=0.01) == metrics


@pytest.mark.parametrize(
    ("index", "value", "message"),
    [
        (0, -1.0, "moduli must be positive"),
        (3, 0.0, "moduli must be positive"),
        (6, 0.5, "Poisson ratios"),
    ],
)
def test_material_axes_reject_nonphysical_constants(
    tmp_path, index, value, message
):
    values = list(MATERIAL_AXES)
    values[index] = value
    with pytest.raises(ValueError, match=message):
        run_research_comparison(
            _protocol("invalid_material"),
            _inputs(
                orthotropic_material_params_material_axes=tuple(values)
            ),
            tmp_path,
        )


def test_material_axes_reject_finite_but_non_positive_definite_constitutive(tmp_path):
    values = list(MATERIAL_AXES)
    values[6:] = [0.49, 0.49, 0.49]
    with pytest.raises(ValueError, match="positive definite"):
        run_research_comparison(
            _protocol("indefinite_material"),
            _inputs(
                orthotropic_material_params_material_axes=tuple(values)
            ),
            tmp_path,
        )


@pytest.mark.parametrize(
    "invalid_strength",
    [
        replace(STRENGTH, X_t=-1.0),
        replace(STRENGTH, criterion="tsai_wu"),
        replace(STRENGTH, units="MPa"),
    ],
)
def test_material_strength_is_revalidated_at_research_boundary(
    tmp_path, invalid_strength
):
    with pytest.raises(ValueError, match="strength|criterion|units"):
        run_research_comparison(
            _protocol("invalid_strength"),
            _inputs(material_strength=invalid_strength),
            tmp_path,
        )


@pytest.mark.parametrize(
    ("protocol_change", "input_change", "message"),
    [
        ({"projection_eta": 0.0}, {}, "projection_eta"),
        ({"mma_min_density": 1.0}, {}, "mma_min_density"),
        ({}, {"material_orientation": np.eye(3)}, "print_orientation_xyz"),
    ],
)
def test_protocol_and_inputs_reject_downstream_mismatches(
    tmp_path, protocol_change, input_change, message
):
    with pytest.raises(ValueError, match=message):
        run_research_comparison(
            _protocol("invalid_case", **protocol_change),
            _inputs(**input_change),
            tmp_path,
        )


def test_inputs_reject_obstacle_overlap_with_automatic_load_protection(tmp_path):
    inputs = _inputs()
    obstacle = np.array(inputs.obstacle_mask, copy=True)
    obstacle[0, -1, 0] = True
    with pytest.raises(ValueError, match="support, or load"):
        run_research_comparison(
            _protocol("overlap_case"),
            replace(inputs, obstacle_mask=obstacle),
            tmp_path,
        )


def test_missing_common_evaluation_metric_marks_simulation_invalid(tmp_path):
    optimize, evaluate, _, _ = _mock_functions()

    def missing_metric(**kwargs):
        result = evaluate(**kwargs)
        result.metrics.pop("predicted_stiffness_binary")
        return result

    result = run_research_comparison(
        _protocol("missing_metric"),
        _inputs(),
        tmp_path,
        optimization_function=optimize,
        evaluation_function=missing_metric,
    )

    assert result.simulation_gate_passed is False
    assert result.validation_status == "simulation_invalid"
    assert all(
        record["run_status"] == "evaluation_error"
        for record in result.report["designs"].values()
    )
    _assert_all_json_strict(result.output_directory)


def test_inconsistent_exact_failure_metrics_are_rejected(tmp_path):
    optimize, evaluate, _, evaluation_calls = _mock_functions()

    def inconsistent_failure(**kwargs):
        result = evaluate(**kwargs)
        if len(evaluation_calls) == 3:
            result.metrics["failure_index_max_binary"] = 1.2
        return result

    result = run_research_comparison(
        _protocol("inconsistent_failure"),
        _inputs(),
        tmp_path,
        optimization_function=optimize,
        evaluation_function=inconsistent_failure,
    )

    assert result.simulation_gate_passed is False
    assert result.report["designs"]["C"]["run_status"] == "evaluation_error"
    assert "inconsistent" in result.report["designs"]["C"]["error"][
        "message"
    ]


def test_uncompleted_continuation_cannot_pass_simulation_gate(tmp_path):
    optimize, evaluate, _, _ = _mock_functions()

    def unconverged(**kwargs):
        result = optimize(**kwargs)
        kwargs["diagnostics_out"].update(
            {
                "continuation_completed": False,
                "projection_converged": False,
                "termination_status": "feasible_not_converged",
            }
        )
        return result

    result = run_research_comparison(
        _protocol("uncompleted_continuation"),
        _inputs(),
        tmp_path,
        optimization_function=unconverged,
        evaluation_function=evaluate,
    )

    assert result.simulation_gate_passed is False
    failures = result.report["validation"]["simulation_failures"]
    assert any("continuation" in failure for failure in failures)
    assert any("projection" in failure for failure in failures)
    assert any("termination" in failure for failure in failures)


def test_external_schema_is_typed_bounded_hashed_and_observable_bound():
    protocol = _protocol(
        "external_schema", material_data_provenance="measured"
    )
    report = {
        "common_case_sha256": _sha256("case"),
        "designs": {
            "A": {
                "binary_topology_sha256": _sha256("topology-A"),
                "common_evaluation": {
                    "predicted_stiffness_binary": 100.0,
                    "predicted_failure_load_binary": 200.0,
                    "failure_index_max_binary": 0.75,
                    "critical_mode_binary": "Y tension",
                    "critical_region_binary": "design",
                    "critical_element_yxz_binary": [0, 1, 0],
                    "material_volume_m3_binary": 5.0e-6,
                },
            }
        },
    }
    experimental = _experimental_record(
        protocol, report, "A", "A-01"
    )
    normalized = validate_experimental_measurements(
        protocol, [experimental]
    )
    assert normalized[0]["raw_data_sha256"] == experimental["raw_data_sha256"]
    with pytest.raises(ValueError, match="specimen bounds"):
        validate_experimental_measurements(
            protocol,
            [{**experimental, "fracture_location_x_m": 1.0}],
        )
    with pytest.raises(ValueError, match="duplicate"):
        validate_experimental_measurements(
            protocol, [experimental, experimental]
        )
    with pytest.raises(ValueError, match="SHA-256"):
        validate_experimental_measurements(
            protocol, [{**experimental, "raw_data_sha256": "not-a-hash"}]
        )
    with pytest.raises(ValueError, match="observable"):
        validate_experimental_measurements(
            protocol,
            [{**experimental, "stiffness_observable": "another observable"}],
        )

    ansys = _ansys_record(protocol, report, "A")
    assert validate_ansys_results(protocol, [ansys])[0][
        "failure_index"
    ] == 0.75
    with pytest.raises(ValueError, match="out of bounds"):
        validate_ansys_results(
            protocol,
            [{**ansys, "critical_location_xyz_m": [1.0, 0.005, 0.005]}],
        )
    with pytest.raises(ValueError, match="coordinate_frame"):
        validate_ansys_results(
            protocol, [{**ansys, "coordinate_frame": "unknown frame"}]
        )


def test_external_attachment_does_not_rerun_and_round_trips(tmp_path):
    protocol = _protocol(
        "complete_external", material_data_provenance="measured"
    )
    optimize, evaluate, optimizer_calls, evaluation_calls = _mock_functions()
    base = run_research_comparison(
        protocol,
        _inputs(),
        tmp_path,
        optimization_function=optimize,
        evaluation_function=evaluate,
    )
    experiments, ansys = _bound_evidence(
        protocol, base.report, base.output_directory
    )

    attached = attach_external_validation(
        base.output_directory,
        experimental_measurements=experiments,
        ansys_results=ansys,
    )

    assert len(optimizer_calls) == 3
    assert len(evaluation_calls) == 3
    assert attached.validation_status == "validation_complete"
    assert attached.simulation_gate_passed is True
    assert attached.report["validation"]["experimental_status"] == "complete"
    assert attached.report["validation"]["ansys_status"] == "complete"
    assert attached.report["validation"]["external_validation_failures"] == []
    assert attached.report["research_questions"][
        "experimental_evidence_complete"
    ] is True
    assert attached.report["research_questions"][
        "experimental_strength_improvement_observed"
    ] is True

    root = Path(attached.output_directory)
    assert load_comparison_protocol(root / "protocol.json") == protocol
    saved_experiments = load_experimental_measurements_csv(
        root / "experimental_measurements.csv"
    )
    saved_ansys = load_ansys_results_json(root / "ansys_results.json")
    assert len(saved_experiments) == 3
    assert len(saved_ansys) == 3
    for record, path_field, hash_field in [
        *(
            (record, "manufacturing_artifact_path", "manufacturing_artifact_sha256")
            for record in saved_experiments
        ),
        *(
            (record, "raw_data_path", "raw_data_sha256")
            for record in saved_experiments
        ),
        *(
            (record, "geometry_artifact_path", "geometry_artifact_sha256")
            for record in saved_ansys
        ),
        *(
            (record, "result_artifact_path", "result_artifact_sha256")
            for record in saved_ansys
        ),
    ]:
        artifact_path = root / record[path_field]
        assert artifact_path.is_file()
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == record[
            hash_field
        ]
    _assert_all_json_strict(root)


def test_partial_external_attachment_stays_partial(tmp_path):
    protocol = _protocol(
        "partial_external", material_data_provenance="measured"
    )
    optimize, evaluate, _, _ = _mock_functions()
    base = run_research_comparison(
        protocol,
        _inputs(),
        tmp_path,
        optimization_function=optimize,
        evaluation_function=evaluate,
    )
    experiments, _ = _bound_evidence(
        protocol, base.report, base.output_directory
    )

    attached = attach_external_validation(
        base.output_directory,
        experimental_measurements=experiments,
    )

    assert attached.validation_status == "external_validation_partial"
    assert attached.report["validation"]["experimental_status"] == "complete"
    assert attached.report["validation"]["ansys_status"] == "not_run"
    assert "ansys_records_not_run" in attached.report["validation"][
        "external_validation_failures"
    ]


@pytest.mark.parametrize("source", ["ansys", "experimental"])
def test_external_disagreement_cannot_complete_validation(tmp_path, source):
    protocol = _protocol(
        f"disagreement_{source}",
        material_data_provenance="measured",
    )
    optimize, evaluate, _, _ = _mock_functions()
    base = run_research_comparison(
        protocol,
        _inputs(),
        tmp_path,
        optimization_function=optimize,
        evaluation_function=evaluate,
    )
    experiments, ansys = _bound_evidence(
        protocol, base.report, base.output_directory
    )
    if source == "ansys":
        ansys[-1]["predicted_failure_force_N"] *= (
            1.5 + protocol.ansys_relative_tolerance
        )
    else:
        experiments[-1]["experimental_failure_force_N"] *= (
            1.5 + protocol.experimental_relative_tolerance
        )

    attached = attach_external_validation(
        base.output_directory,
        experimental_measurements=experiments,
        ansys_results=ansys,
    )

    assert attached.validation_status == "external_validation_partial"
    failures = attached.report["validation"]["external_validation_failures"]
    assert any(source in failure and "agreement" in failure for failure in failures)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("common_case_sha256", "common[- ]case"),
        ("binary_topology_sha256", "topology"),
    ],
)
def test_external_attachment_rejects_wrong_bound_hash(
    tmp_path, field, message
):
    protocol = _protocol(
        f"wrong_{field}", material_data_provenance="measured"
    )
    optimize, evaluate, _, _ = _mock_functions()
    base = run_research_comparison(
        protocol,
        _inputs(),
        tmp_path,
        optimization_function=optimize,
        evaluation_function=evaluate,
    )
    experiments, _ = _bound_evidence(
        protocol, base.report, base.output_directory
    )
    experiments[0][field] = "0" * 64
    comparison_path = Path(base.output_directory) / "comparison.json"
    report_before = comparison_path.read_bytes()

    with pytest.raises(ValueError, match=message):
        attach_external_validation(
            base.output_directory,
            experimental_measurements=experiments,
        )
    assert comparison_path.read_bytes() == report_before


def test_external_attachment_rejects_unverified_artifact_digest(tmp_path):
    protocol = _protocol("bad_external_artifact", material_data_provenance="measured")
    optimize, evaluate, _, _ = _mock_functions()
    base = run_research_comparison(
        protocol,
        _inputs(),
        tmp_path,
        optimization_function=optimize,
        evaluation_function=evaluate,
    )
    experiments, ansys = _bound_evidence(
        protocol, base.report, base.output_directory
    )
    experiments[0]["raw_data_sha256"] = "0" * 64
    comparison_path = Path(base.output_directory) / "comparison.json"
    report_before = comparison_path.read_bytes()

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        attach_external_validation(
            base.output_directory,
            experimental_measurements=experiments,
            ansys_results=ansys,
        )
    assert comparison_path.read_bytes() == report_before


def test_attachment_uses_immutable_simulation_metrics_snapshot(tmp_path):
    protocol = _protocol("immutable_simulation_metrics")
    optimize, evaluate, _, _ = _mock_functions()
    base = run_research_comparison(
        protocol,
        _inputs(),
        tmp_path,
        optimization_function=optimize,
        evaluation_function=evaluate,
    )
    root = Path(base.output_directory)
    comparison_path = root / "comparison.json"
    tampered = json.loads(comparison_path.read_text(encoding="utf-8"))
    tampered["designs"]["A"]["common_evaluation"]["compliance_binary"] = 999.0
    tampered["validation"]["simulation_gate_passed"] = False
    comparison_path.write_text(json.dumps(tampered), encoding="utf-8")

    restored = attach_external_validation(root)

    assert restored.report["designs"]["A"]["common_evaluation"][
        "compliance_binary"
    ] == 10.0
    assert restored.simulation_gate_passed is True


def test_each_experimental_fracture_must_meet_location_tolerance(tmp_path):
    protocol = _protocol(
        "per_specimen_location",
        material_data_provenance="measured",
        minimum_experimental_replicates=3,
        critical_location_tolerance_elements=0.2,
    )
    optimize, evaluate, _, _ = _mock_functions()
    base = run_research_comparison(
        protocol,
        _inputs(),
        tmp_path,
        optimization_function=optimize,
        evaluation_function=evaluate,
    )
    experiments, ansys = _bound_evidence(
        protocol, base.report, base.output_directory, replicates=3
    )
    offsets = (-0.004, 0.008, -0.004)
    for record in experiments:
        replicate = int(record["specimen_id"].rsplit("-", 1)[1]) - 1
        record["fracture_location_x_m"] += offsets[replicate]

    attached = attach_external_validation(
        base.output_directory,
        experimental_measurements=experiments,
        ansys_results=ansys,
    )

    summary = attached.report["designs"]["A"]["experimental"]
    predicted_x = summary["predicted_critical_location_xyz_m"][0]
    assert summary["experimental_fracture_location_xyz_m"][0] == pytest.approx(
        predicted_x
    )
    assert summary["experimental_critical_location_match"] is False
    assert attached.validation_status == "external_validation_partial"


def test_research_claim_requires_fair_mass_and_preregistered_effect(tmp_path):
    protocol = _protocol(
        "fair_physical_comparison",
        material_data_provenance="measured",
        minimum_experimental_replicates=3,
    )
    optimize, evaluate, _, _ = _mock_functions()
    base = run_research_comparison(
        protocol,
        _inputs(),
        tmp_path,
        optimization_function=optimize,
        evaluation_function=evaluate,
    )
    experiments, ansys = _bound_evidence(
        protocol, base.report, base.output_directory, replicates=3
    )
    force_scales = (0.95, 1.0, 1.05)
    for record in experiments:
        replicate = int(record["specimen_id"].rsplit("-", 1)[1]) - 1
        record["experimental_failure_force_N"] *= force_scales[replicate]
        if record["design_id"] == "C":
            record["mass_kg"] *= 2.0

    attached = attach_external_validation(
        base.output_directory,
        experimental_measurements=experiments,
        ansys_results=ansys,
    )

    questions = attached.report["research_questions"]
    assert attached.validation_status == "validation_complete"
    assert questions["failure_force_improvement_ratio_gate_passed"] is True
    assert questions["standardized_effect_size_gate_passed"] is True
    assert questions["physical_comparison_fair"] is False
    assert questions["research_claim_supported"] is False


def test_external_attachment_rejects_tampered_saved_topology(tmp_path):
    protocol = _protocol("tampered_saved_topology")
    optimize, evaluate, _, _ = _mock_functions()
    base = run_research_comparison(
        protocol,
        _inputs(),
        tmp_path,
        optimization_function=optimize,
        evaluation_function=evaluate,
    )
    root = Path(base.output_directory)
    binary_path = root / base.report["designs"]["A"]["binary_density_path"]
    binary = np.load(binary_path, allow_pickle=False)
    binary[0, 1, 0] = 1.0 - binary[0, 1, 0]
    np.save(binary_path, binary)

    with pytest.raises(ValueError, match="topology hash mismatch"):
        attach_external_validation(root)


def test_real_abc_smoke_is_finite_and_external_pending(tmp_path):
    protocol = _protocol(
        "real_abc_smoke",
        volfrac=1.0,
        maxloop=6,
        binary_volume_tolerance=0.01,
    )
    inputs = _inputs(initial_design=np.ones((2, 4, 1)))

    result = run_research_comparison(protocol, inputs, tmp_path)

    assert result.simulation_gate_passed is True
    assert result.validation_status == "simulation_complete_external_pending"
    for design_id in "ABC":
        record = result.report["designs"][design_id]
        assert record["run_status"] == "complete"
        assert record["optimizer_feasible"] is True
        assert record["binary_free_volume_fraction"] == 1.0
        assert np.isfinite(record["common_evaluation"]["compliance_binary"])
    assert result.report["designs"]["C"]["common_evaluation"][
        "stage10_internal_verification_passed"
    ] is True
    json.dumps(result.report, allow_nan=False)
