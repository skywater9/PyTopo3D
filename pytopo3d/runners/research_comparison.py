"""Reproducible A/B/C research comparison and external-validation protocol.

The runner keeps optimization, common-material verification, and external
evidence separate.  A completed simulation is therefore never mislabeled as
an ANSYS or experimental validation when those records have not been supplied.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import yaml

from pytopo3d.analysis.postprocessing import evaluate_failure_representations
from pytopo3d.core.optimizer import top3d
from pytopo3d.utils.config_loader import (
    MaterialStrength,
    apply_material_orientation,
    material_orientation_matrix,
    validate_material_strength,
)
from pytopo3d.utils.assembly import build_force_vector
from pytopo3d.utils.results_manager import ResultsManager
from pytopo3d.utils.stiffness import make_C_matrix


DESIGN_IDS = ("A", "B", "C")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PROVENANCE_VALUES = {"measured", "literature", "synthetic"}
_KNOWN_TERMINATION_STATUSES = {"converged", "feasible_not_converged"}
DEFAULT_STIFFNESS_OBSERVABLE = (
    "dominant_axis_absolute_force_over_absolute_mean_loaded_patch_"
    "displacement_N_per_m"
)
DEFAULT_FAILURE_FORCE_OBSERVABLE = (
    "sum_absolute_loaded_dof_force_scaled_to_failure_index_one_N"
)
DEFAULT_COORDINATE_FRAME = "PyTopo3D global XYZ, origin at mesh minimum"

EXPERIMENTAL_COLUMNS = (
    "case_id",
    "common_case_sha256",
    "design_id",
    "specimen_id",
    "binary_topology_sha256",
    "manufacturing_artifact_path",
    "manufacturing_artifact_sha256",
    "raw_data_sha256",
    "print_orientation_xyz",
    "stiffness_observable",
    "failure_force_observable",
    "experimental_stiffness_N_per_m",
    "experimental_failure_force_N",
    "fracture_location_x_m",
    "fracture_location_y_m",
    "fracture_location_z_m",
    "fracture_region",
    "mass_kg",
    "volume_m3",
    "raw_data_path",
    "notes",
)

ANSYS_REQUIRED_FIELDS = (
    "comparison_id",
    "common_case_sha256",
    "design_id",
    "binary_topology_sha256",
    "geometry_artifact_path",
    "geometry_artifact_sha256",
    "result_artifact_path",
    "result_artifact_sha256",
    "solver",
    "solver_version",
    "mesh_description",
    "failure_force_observable",
    "failure_index",
    "predicted_failure_force_N",
    "critical_mode",
    "critical_region",
    "critical_location_xyz_m",
    "coordinate_frame",
    "location_mapping_method",
    "location_tolerance_m",
)


@dataclass(frozen=True)
class ComparisonProtocol:
    """Configuration shared by all three designs in one research comparison."""

    comparison_id: str
    load_case_name: str
    material_system_id: str
    material_data_provenance: str
    isotropic_optimizer_material_id: str
    orthotropic_material_id: str
    nelx: int
    nely: int
    nelz: int
    elem_size_m: float
    volfrac: float
    penal: float
    rmin: float
    print_orientation_xyz: str
    code_version_id: str
    schema_version: int = 1
    optimizer: str = "mma"
    disp_thres: float = 0.5
    tolx: float = 0.01
    maxloop: int = 2000
    beta_schedule: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    projection_eta: float = 0.5
    failure_limit_schedule: tuple[float, ...] = (1.5, 1.25, 1.1, 0.9)
    failure_aggregate_exponent_schedule: tuple[float, ...] = (4.0, 6.0, 8.0, 8.0)
    failure_relaxation_exponent: float = 0.5
    mma_move: float = 0.05
    mma_min_density: float = 1.0e-3
    binary_threshold: float = 0.5
    use_gpu: bool = False
    binary_volume_tolerance: float = 0.02
    minimum_experimental_replicates: int = 3
    bulk_density_kg_m3: Optional[float] = None
    topology_change_threshold: float = 0.02
    ansys_relative_tolerance: float = 0.25
    experimental_relative_tolerance: float = 0.35
    isotropic_surrogate_method: str = "axis_average_E_nu"
    stiffness_observable: str = DEFAULT_STIFFNESS_OBSERVABLE
    failure_force_observable: str = DEFAULT_FAILURE_FORCE_OBSERVABLE
    acceptable_termination_statuses: tuple[str, ...] = ("converged",)
    require_continuation_completed: bool = True
    require_projection_converged: bool = True
    critical_location_tolerance_elements: float = 1.75
    measured_volume_relative_tolerance: float = 0.05
    measured_mass_relative_tolerance: float = 0.05
    minimum_failure_force_improvement_ratio: float = 1.10
    minimum_standardized_effect_size: float = 0.80

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ComparisonProtocol":
        """Create a protocol from the documented nested YAML/JSON form."""
        if not isinstance(raw, Mapping):
            raise ValueError("comparison protocol must be a mapping")
        mesh = raw.get("mesh", {})
        optimization = raw.get("optimization", {})
        materials = raw.get("materials", {})
        validation = raw.get("validation", {})
        if not all(
            isinstance(section, Mapping)
            for section in (mesh, optimization, materials, validation)
        ):
            raise ValueError(
                "mesh, optimization, materials, and validation must be mappings"
            )
        return cls(
            schema_version=raw.get("schema_version", 1),
            comparison_id=raw.get("comparison_id", ""),
            load_case_name=raw.get("load_case_name", ""),
            code_version_id=raw.get("code_version_id", ""),
            material_system_id=raw.get("material_system_id", ""),
            material_data_provenance=raw.get("material_data_provenance", ""),
            isotropic_optimizer_material_id=materials.get(
                "isotropic_optimizer_material_id", ""
            ),
            orthotropic_material_id=materials.get("orthotropic_material_id", ""),
            nelx=mesh.get("nelx"),
            nely=mesh.get("nely"),
            nelz=mesh.get("nelz"),
            elem_size_m=mesh.get("elem_size_m"),
            volfrac=optimization.get("volfrac"),
            penal=optimization.get("penal"),
            rmin=optimization.get("rmin"),
            optimizer=optimization.get("optimizer", "mma"),
            disp_thres=optimization.get("disp_thres", 0.5),
            tolx=optimization.get("tolx", 0.01),
            maxloop=optimization.get("maxloop", 2000),
            beta_schedule=tuple(
                optimization.get("beta_schedule", (1.0, 2.0, 4.0, 8.0))
            ),
            projection_eta=optimization.get("projection_eta", 0.5),
            failure_limit_schedule=tuple(
                optimization.get(
                    "failure_limit_schedule", (1.5, 1.25, 1.1, 0.9)
                )
            ),
            failure_aggregate_exponent_schedule=tuple(
                optimization.get(
                    "failure_aggregate_exponent_schedule", (4.0, 6.0, 8.0, 8.0)
                )
            ),
            failure_relaxation_exponent=optimization.get(
                "failure_relaxation_exponent", 0.5
            ),
            mma_move=optimization.get("mma_move", 0.05),
            mma_min_density=optimization.get("mma_min_density", 1.0e-3),
            binary_threshold=optimization.get("binary_threshold", 0.5),
            use_gpu=optimization.get("use_gpu", False),
            acceptable_termination_statuses=tuple(
                optimization.get("acceptable_termination_statuses", ("converged",))
            ),
            require_continuation_completed=optimization.get(
                "require_continuation_completed", True
            ),
            require_projection_converged=optimization.get(
                "require_projection_converged", True
            ),
            print_orientation_xyz=materials.get("orientation_xyz", ""),
            bulk_density_kg_m3=materials.get("bulk_density_kg_m3"),
            binary_volume_tolerance=validation.get(
                "binary_volume_tolerance", 0.02
            ),
            minimum_experimental_replicates=validation.get(
                "minimum_experimental_replicates", 3
            ),
            topology_change_threshold=validation.get(
                "topology_change_threshold", 0.02
            ),
            ansys_relative_tolerance=validation.get(
                "ansys_relative_tolerance", 0.25
            ),
            experimental_relative_tolerance=validation.get(
                "experimental_relative_tolerance", 0.35
            ),
            critical_location_tolerance_elements=validation.get(
                "critical_location_tolerance_elements", 1.75
            ),
            measured_volume_relative_tolerance=validation.get(
                "measured_volume_relative_tolerance", 0.05
            ),
            measured_mass_relative_tolerance=validation.get(
                "measured_mass_relative_tolerance", 0.05
            ),
            minimum_failure_force_improvement_ratio=validation.get(
                "minimum_failure_force_improvement_ratio", 1.10
            ),
            minimum_standardized_effect_size=validation.get(
                "minimum_standardized_effect_size", 0.80
            ),
            isotropic_surrogate_method=materials.get(
                "isotropic_surrogate_method", "axis_average_E_nu"
            ),
            stiffness_observable=validation.get(
                "stiffness_observable", DEFAULT_STIFFNESS_OBSERVABLE
            ),
            failure_force_observable=validation.get(
                "failure_force_observable", DEFAULT_FAILURE_FORCE_OBSERVABLE
            ),
        )

    def as_mapping(self) -> Dict[str, Any]:
        """Return the stable nested representation written to protocol.json."""
        return {
            "schema_version": self.schema_version,
            "comparison_id": self.comparison_id,
            "load_case_name": self.load_case_name,
            "code_version_id": self.code_version_id,
            "material_system_id": self.material_system_id,
            "material_data_provenance": self.material_data_provenance,
            "mesh": {
                "nelx": self.nelx,
                "nely": self.nely,
                "nelz": self.nelz,
                "elem_size_m": self.elem_size_m,
            },
            "optimization": {
                "volfrac": self.volfrac,
                "penal": self.penal,
                "rmin": self.rmin,
                "optimizer": self.optimizer,
                "disp_thres": self.disp_thres,
                "tolx": self.tolx,
                "maxloop": self.maxloop,
                "beta_schedule": list(self.beta_schedule),
                "projection_eta": self.projection_eta,
                "failure_limit_schedule": list(self.failure_limit_schedule),
                "failure_aggregate_exponent_schedule": list(
                    self.failure_aggregate_exponent_schedule
                ),
                "failure_relaxation_exponent": self.failure_relaxation_exponent,
                "mma_move": self.mma_move,
                "mma_min_density": self.mma_min_density,
                "binary_threshold": self.binary_threshold,
                "use_gpu": self.use_gpu,
                "acceptable_termination_statuses": list(
                    self.acceptable_termination_statuses
                ),
                "require_continuation_completed": (
                    self.require_continuation_completed
                ),
                "require_projection_converged": self.require_projection_converged,
            },
            "materials": {
                "isotropic_optimizer_material_id": (
                    self.isotropic_optimizer_material_id
                ),
                "orthotropic_material_id": self.orthotropic_material_id,
                "isotropic_surrogate_method": self.isotropic_surrogate_method,
                "orientation_xyz": self.print_orientation_xyz,
                "bulk_density_kg_m3": self.bulk_density_kg_m3,
            },
            "validation": {
                "binary_volume_tolerance": self.binary_volume_tolerance,
                "minimum_experimental_replicates": (
                    self.minimum_experimental_replicates
                ),
                "topology_change_threshold": self.topology_change_threshold,
                "ansys_relative_tolerance": self.ansys_relative_tolerance,
                "experimental_relative_tolerance": (
                    self.experimental_relative_tolerance
                ),
                "critical_location_tolerance_elements": (
                    self.critical_location_tolerance_elements
                ),
                "measured_volume_relative_tolerance": (
                    self.measured_volume_relative_tolerance
                ),
                "measured_mass_relative_tolerance": (
                    self.measured_mass_relative_tolerance
                ),
                "minimum_failure_force_improvement_ratio": (
                    self.minimum_failure_force_improvement_ratio
                ),
                "minimum_standardized_effect_size": (
                    self.minimum_standardized_effect_size
                ),
                "stiffness_observable": self.stiffness_observable,
                "failure_force_observable": self.failure_force_observable,
            },
        }


@dataclass(frozen=True)
class ComparisonInputs:
    """Explicit common numerical inputs and the two optimizer materials."""

    orthotropic_material_params_material_axes: tuple[float, ...]
    material_strength: MaterialStrength
    material_orientation: np.ndarray
    force_field: np.ndarray
    support_mask: np.ndarray
    obstacle_mask: np.ndarray
    protected_zone_mask: np.ndarray
    initial_design: np.ndarray


@dataclass(frozen=True)
class DesignVariant:
    """One controlled optimization variant in the A/B/C experiment."""

    design_id: str
    label: str
    optimizer_material: str
    optimizer_material_id: str
    optimization_mode: str
    material_params: tuple[float, ...]
    uses_failure_constraint: bool


@dataclass(frozen=True)
class ResearchComparisonResult:
    """Paths and in-memory summary returned by the comparison runner."""

    output_directory: str
    validation_status: str
    simulation_gate_passed: bool
    report: Dict[str, Any]


def load_comparison_protocol(path: str | Path) -> ComparisonProtocol:
    """Load a YAML or JSON comparison protocol."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            raw = json.load(handle)
        else:
            raw = yaml.safe_load(handle)
    protocol = ComparisonProtocol.from_mapping(raw)
    _validate_protocol(protocol)
    return protocol


def load_experimental_measurements_csv(path: str | Path) -> list[Dict[str, Any]]:
    """Load specimen records; numerical normalization occurs during validation."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_ansys_results_json(path: str | Path) -> list[Dict[str, Any]]:
    """Load a list of independently produced ANSYS result records."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("ANSYS results JSON must contain a list of records")
    return [dict(record) for record in raw]


def build_design_variants(
    protocol: ComparisonProtocol,
    inputs: ComparisonInputs,
) -> tuple[DesignVariant, ...]:
    """Return the fixed A/B/C dispatch contract."""
    material_axes = _validate_material_params(
        "orthotropic_material_params_material_axes",
        inputs.orthotropic_material_params_material_axes,
    )
    isotropic = isotropize_orthotropic_material(material_axes)
    orthotropic_global = tuple(
        apply_material_orientation(material_axes, protocol.print_orientation_xyz)
    )
    return (
        DesignVariant(
            "A",
            "isotropic stiffness optimization",
            "isotropic",
            protocol.isotropic_optimizer_material_id,
            "compliance",
            isotropic,
            False,
        ),
        DesignVariant(
            "B",
            "anisotropic stiffness optimization",
            "orthotropic",
            protocol.orthotropic_material_id,
            "compliance",
            orthotropic_global,
            False,
        ),
        DesignVariant(
            "C",
            "anisotropic strength-constrained optimization",
            "orthotropic",
            protocol.orthotropic_material_id,
            "compliance_failure_constrained",
            orthotropic_global,
            True,
        ),
    )


def _finite_float(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if positive and normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if normalized != value or normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _validate_schedule(
    name: str,
    values: Sequence[float],
    stage_count: int,
    *,
    direction: str,
    minimum_exclusive: float,
) -> tuple[float, ...]:
    normalized = tuple(_finite_float(name, value) for value in values)
    if len(normalized) not in {1, stage_count}:
        raise ValueError(
            f"{name} must contain one value or {stage_count} beta-stage values"
        )
    if any(value <= minimum_exclusive for value in normalized):
        raise ValueError(f"all {name} values must be > {minimum_exclusive:g}")
    differences = np.diff(normalized)
    if direction == "nonincreasing" and np.any(differences > 1.0e-12):
        raise ValueError(f"{name} must be nonincreasing")
    if direction == "nondecreasing" and np.any(differences < -1.0e-12):
        raise ValueError(f"{name} must be nondecreasing")
    return normalized


def _validate_protocol(protocol: ComparisonProtocol) -> None:
    if protocol.schema_version != 1:
        raise ValueError(
            f"unsupported comparison schema_version {protocol.schema_version}; expected 1"
        )
    if not _SAFE_ID.fullmatch(str(protocol.comparison_id)):
        raise ValueError(
            "comparison_id must contain only letters, digits, '.', '_', and '-'"
        )
    for name, value in (
        ("load_case_name", protocol.load_case_name),
        ("material_system_id", protocol.material_system_id),
        ("code_version_id", protocol.code_version_id),
        (
            "isotropic_optimizer_material_id",
            protocol.isotropic_optimizer_material_id,
        ),
        ("orthotropic_material_id", protocol.orthotropic_material_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if protocol.stiffness_observable != DEFAULT_STIFFNESS_OBSERVABLE:
        raise ValueError(
            "stiffness_observable does not match the implemented solver observable"
        )
    if protocol.failure_force_observable != DEFAULT_FAILURE_FORCE_OBSERVABLE:
        raise ValueError(
            "failure_force_observable does not match the implemented solver observable"
        )
    if protocol.material_data_provenance not in _PROVENANCE_VALUES:
        raise ValueError(
            "material_data_provenance must be measured, literature, or synthetic"
        )
    if protocol.isotropic_surrogate_method != "axis_average_E_nu":
        raise ValueError(
            "isotropic_surrogate_method must be 'axis_average_E_nu'"
        )
    for name, value in (
        ("nelx", protocol.nelx),
        ("nely", protocol.nely),
        ("nelz", protocol.nelz),
        ("maxloop", protocol.maxloop),
        ("minimum_experimental_replicates", protocol.minimum_experimental_replicates),
    ):
        _positive_int(name, value)
    for name, value in (
        ("elem_size_m", protocol.elem_size_m),
        ("penal", protocol.penal),
        ("rmin", protocol.rmin),
        ("tolx", protocol.tolx),
        ("mma_move", protocol.mma_move),
        ("mma_min_density", protocol.mma_min_density),
    ):
        _finite_float(name, value, positive=True)
    if not 0.0 < _finite_float("volfrac", protocol.volfrac) <= 1.0:
        raise ValueError("volfrac must be in (0, 1]")
    if protocol.optimizer != "mma":
        raise ValueError("research comparisons require optimizer='mma' for A/B/C fairness")
    if not 0.0 <= _finite_float("disp_thres", protocol.disp_thres) <= 1.0:
        raise ValueError("disp_thres must be in [0, 1]")
    if not 0.0 < _finite_float("projection_eta", protocol.projection_eta) < 1.0:
        raise ValueError("projection_eta must be in (0, 1)")
    if not 0.0 <= _finite_float(
        "binary_threshold", protocol.binary_threshold
    ) <= 1.0:
        raise ValueError("binary_threshold must be in [0, 1]")
    for name, value in (
        ("binary_volume_tolerance", protocol.binary_volume_tolerance),
        ("topology_change_threshold", protocol.topology_change_threshold),
        ("ansys_relative_tolerance", protocol.ansys_relative_tolerance),
        (
            "experimental_relative_tolerance",
            protocol.experimental_relative_tolerance,
        ),
        (
            "measured_volume_relative_tolerance",
            protocol.measured_volume_relative_tolerance,
        ),
        (
            "measured_mass_relative_tolerance",
            protocol.measured_mass_relative_tolerance,
        ),
    ):
        normalized = _finite_float(name, value)
        if not 0.0 <= normalized <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    _finite_float(
        "critical_location_tolerance_elements",
        protocol.critical_location_tolerance_elements,
        positive=True,
    )
    improvement_ratio = _finite_float(
        "minimum_failure_force_improvement_ratio",
        protocol.minimum_failure_force_improvement_ratio,
        positive=True,
    )
    if improvement_ratio <= 1.0:
        raise ValueError("minimum_failure_force_improvement_ratio must be > 1")
    effect_size = _finite_float(
        "minimum_standardized_effect_size",
        protocol.minimum_standardized_effect_size,
    )
    if effect_size < 0.0:
        raise ValueError("minimum_standardized_effect_size must be nonnegative")
    if protocol.bulk_density_kg_m3 is not None:
        _finite_float(
            "bulk_density_kg_m3", protocol.bulk_density_kg_m3, positive=True
        )
    mma_min_density = _finite_float("mma_min_density", protocol.mma_min_density)
    if not 0.0 < mma_min_density < 1.0:
        raise ValueError("mma_min_density must be in (0, 1)")
    if not isinstance(protocol.use_gpu, bool):
        raise ValueError("use_gpu must be a boolean")
    for name, value in (
        ("require_continuation_completed", protocol.require_continuation_completed),
        ("require_projection_converged", protocol.require_projection_converged),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
    if not isinstance(protocol.acceptable_termination_statuses, tuple) or not (
        protocol.acceptable_termination_statuses
    ):
        raise ValueError("acceptable_termination_statuses must be a nonempty tuple")
    invalid_statuses = [
        status
        for status in protocol.acceptable_termination_statuses
        if not isinstance(status, str)
        or status.strip() not in _KNOWN_TERMINATION_STATUSES
    ]
    if invalid_statuses:
        raise ValueError(
            "acceptable_termination_statuses may contain only "
            + ", ".join(sorted(_KNOWN_TERMINATION_STATUSES))
        )
    if not isinstance(protocol.print_orientation_xyz, str) or sorted(
        protocol.print_orientation_xyz.lower()
    ) != ["x", "y", "z"]:
        raise ValueError("print_orientation_xyz must be a permutation of 'xyz'")

    beta_schedule = tuple(
        _finite_float("beta_schedule", value, positive=True)
        for value in protocol.beta_schedule
    )
    if not beta_schedule:
        raise ValueError("beta_schedule must not be empty")
    if np.any(np.diff(beta_schedule) < -1.0e-12):
        raise ValueError("beta_schedule must be nondecreasing")
    _validate_schedule(
        "failure_limit_schedule",
        protocol.failure_limit_schedule,
        len(beta_schedule),
        direction="nonincreasing",
        minimum_exclusive=0.0,
    )
    _validate_schedule(
        "failure_aggregate_exponent_schedule",
        protocol.failure_aggregate_exponent_schedule,
        len(beta_schedule),
        direction="nondecreasing",
        minimum_exclusive=1.0,
    )
    relaxation = _finite_float(
        "failure_relaxation_exponent", protocol.failure_relaxation_exponent
    )
    if relaxation < 0.0:
        raise ValueError("failure_relaxation_exponent must be nonnegative")


def _validate_material_params(name: str, values: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(_finite_float(name, value) for value in values)
    if len(normalized) != 9:
        raise ValueError(f"{name} must contain nine orthotropic constants")
    if any(value <= 0.0 for value in normalized[:6]):
        raise ValueError(f"{name} elastic and shear moduli must be positive")
    if any(not -1.0 < value < 0.5 for value in normalized[6:]):
        raise ValueError(f"{name} Poisson ratios must lie in (-1, 0.5)")
    try:
        constitutive = make_C_matrix(*normalized)
    except Exception as exc:
        raise ValueError(f"{name} does not define a valid elastic material") from exc
    if not np.all(np.isfinite(constitutive)):
        raise ValueError(f"{name} produced a nonfinite constitutive matrix")
    eigenvalues = np.linalg.eigvalsh(0.5 * (constitutive + constitutive.T))
    if np.min(eigenvalues) <= 0.0:
        raise ValueError(f"{name} constitutive matrix must be positive definite")
    return normalized


def isotropize_orthotropic_material(
    material_axis_params: Sequence[float],
) -> tuple[float, ...]:
    """Derive Design A's isotropic surrogate from the same material system.

    The preregistered rule averages the three Young's moduli and Poisson ratios,
    then enforces the isotropic identity ``G = E / (2 * (1 + nu))``.
    """
    params = _validate_material_params(
        "orthotropic_material_params_material_axes", material_axis_params
    )
    young = float(np.mean(params[:3]))
    poisson = float(np.mean(params[6:9]))
    shear = young / (2.0 * (1.0 + poisson))
    return _validate_material_params(
        "derived_isotropic_material_params",
        (young, young, young, shear, shear, shear, poisson, poisson, poisson),
    )


def _validated_strength(
    protocol: ComparisonProtocol,
    strength: MaterialStrength,
) -> MaterialStrength:
    if not isinstance(strength, MaterialStrength):
        raise ValueError("material_strength must be a validated MaterialStrength")
    return validate_material_strength(
        strength.as_dict(), material_name=protocol.orthotropic_material_id
    )


def _validate_inputs(
    protocol: ComparisonProtocol,
    inputs: ComparisonInputs,
) -> Dict[str, np.ndarray]:
    shape = (protocol.nely, protocol.nelx, protocol.nelz)
    material_axes = _validate_material_params(
        "orthotropic_material_params_material_axes",
        inputs.orthotropic_material_params_material_axes,
    )
    isotropic = isotropize_orthotropic_material(material_axes)
    orthotropic_global = tuple(
        apply_material_orientation(material_axes, protocol.print_orientation_xyz)
    )
    _validated_strength(protocol, inputs.material_strength)

    orientation = np.asarray(inputs.material_orientation, dtype=float)
    if orientation.shape != (3, 3) or not np.all(np.isfinite(orientation)):
        raise ValueError("material_orientation must be a finite 3x3 matrix")
    if not np.allclose(orientation.T @ orientation, np.eye(3), atol=1.0e-10):
        raise ValueError("material_orientation must be orthonormal")
    if not np.isclose(abs(np.linalg.det(orientation)), 1.0, atol=1.0e-10):
        raise ValueError("material_orientation determinant magnitude must be one")
    expected_orientation = material_orientation_matrix(
        protocol.print_orientation_xyz.lower()
    )
    if not np.allclose(orientation, expected_orientation, atol=1.0e-12):
        raise ValueError(
            "material_orientation must match print_orientation_xyz exactly"
        )

    force = np.asarray(inputs.force_field, dtype=float)
    if force.shape != shape + (3,) or not np.all(np.isfinite(force)):
        raise ValueError(f"force_field must be finite with shape {shape + (3,)}")
    if not np.any(force != 0.0):
        raise ValueError("force_field must contain at least one nonzero load")

    arrays: Dict[str, np.ndarray] = {
        "force_field": np.array(force, copy=True),
        "material_orientation": np.array(orientation, copy=True),
    }
    for name, raw in (
        ("support_mask", inputs.support_mask),
        ("obstacle_mask", inputs.obstacle_mask),
        ("protected_zone_mask", inputs.protected_zone_mask),
    ):
        array = np.asarray(raw, dtype=bool)
        if array.shape != shape:
            raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
        arrays[name] = np.array(array, copy=True)
    if not np.any(arrays["support_mask"]):
        raise ValueError("support_mask must contain at least one supported region")
    effective_protected = (
        arrays["protected_zone_mask"]
        | arrays["support_mask"]
        | np.any(arrays["force_field"] != 0.0, axis=-1)
    )
    if np.any(arrays["obstacle_mask"] & effective_protected):
        raise ValueError(
            "obstacle_mask must not overlap protected, support, or load regions"
        )

    initial = np.asarray(inputs.initial_design, dtype=float)
    if initial.shape != shape or not np.all(np.isfinite(initial)):
        raise ValueError(f"initial_design must be finite with shape {shape}")
    if np.any((initial < 0.0) | (initial > 1.0)):
        raise ValueError("initial_design values must lie in [0, 1]")
    free = ~(arrays["obstacle_mask"] | effective_protected)
    if not np.any(free):
        raise ValueError("comparison requires at least one free design element")
    if (
        protocol.failure_relaxation_exponent < 1.0
        and np.any(initial[free] < protocol.mma_min_density)
    ):
        raise ValueError(
            "initial_design free values must be at least mma_min_density when q < 1"
        )
    arrays["initial_design"] = np.array(initial, copy=True)
    arrays["free_mask"] = free
    arrays["effective_protected_mask"] = effective_protected
    arrays["isotropic_material_params"] = np.asarray(isotropic, dtype=float)
    arrays["orthotropic_material_params_material_axes"] = np.asarray(
        material_axes, dtype=float
    )
    arrays["orthotropic_material_params_global"] = np.asarray(
        orthotropic_global, dtype=float
    )
    return arrays


def _json_safe(value: Any, path: str = "root") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item(), path)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite value at {path}")
        return value
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), path)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def _write_strict_json(path: Path, payload: Any) -> None:
    safe = _json_safe(payload)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(safe, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _array_hash(array: np.ndarray) -> str:
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_design_id(value: Any, *, field: str = "design_id") -> str:
    design_id = str(value).strip().upper()
    if design_id not in DESIGN_IDS:
        raise ValueError(f"{field} must be one of A, B, or C")
    return design_id


def _validate_sha256(name: str, value: Any) -> str:
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return normalized


def validate_experimental_measurements(
    protocol: ComparisonProtocol,
    measurements: Optional[Sequence[Mapping[str, Any]]],
) -> list[Dict[str, Any]]:
    """Validate and normalize physical specimen records in SI units."""
    _validate_protocol(protocol)
    if measurements is None:
        return []
    normalized: list[Dict[str, Any]] = []
    specimen_ids: set[str] = set()
    for index, raw in enumerate(measurements):
        if not isinstance(raw, Mapping):
            raise ValueError(f"experimental measurement {index} must be a mapping")
        missing = [column for column in EXPERIMENTAL_COLUMNS if column not in raw]
        if missing:
            raise ValueError(
                f"experimental measurement {index} is missing: {', '.join(missing)}"
            )
        if str(raw["case_id"]).strip() != protocol.comparison_id:
            raise ValueError(
                f"experimental measurement {index} case_id must match comparison_id"
            )
        design_id = _normalize_design_id(raw["design_id"])
        specimen_id = str(raw["specimen_id"]).strip()
        if not specimen_id:
            raise ValueError("experimental specimen_id must be nonempty")
        if specimen_id in specimen_ids:
            raise ValueError(f"duplicate experimental specimen_id {specimen_id!r}")
        specimen_ids.add(specimen_id)
        orientation = str(raw["print_orientation_xyz"]).strip().lower()
        if orientation != protocol.print_orientation_xyz.lower():
            raise ValueError(
                f"experimental specimen {specimen_id!r} orientation does not match protocol"
            )
        record: Dict[str, Any] = {
            "case_id": protocol.comparison_id,
            "common_case_sha256": _validate_sha256(
                "common_case_sha256", raw["common_case_sha256"]
            ),
            "design_id": design_id,
            "specimen_id": specimen_id,
            "binary_topology_sha256": _validate_sha256(
                "binary_topology_sha256", raw["binary_topology_sha256"]
            ),
            "manufacturing_artifact_path": str(
                raw["manufacturing_artifact_path"]
            ).strip(),
            "manufacturing_artifact_sha256": _validate_sha256(
                "manufacturing_artifact_sha256",
                raw["manufacturing_artifact_sha256"],
            ),
            "raw_data_sha256": _validate_sha256(
                "raw_data_sha256", raw["raw_data_sha256"]
            ),
            "print_orientation_xyz": orientation,
        }
        if not record["manufacturing_artifact_path"]:
            raise ValueError(
                "experimental manufacturing_artifact_path must be nonempty"
            )
        for field, expected in (
            ("stiffness_observable", protocol.stiffness_observable),
            ("failure_force_observable", protocol.failure_force_observable),
        ):
            value = str(raw[field]).strip()
            if value != expected:
                raise ValueError(
                    f"experimental {field} must match the preregistered protocol"
                )
            record[field] = value
        for field in (
            "experimental_stiffness_N_per_m",
            "experimental_failure_force_N",
            "mass_kg",
            "volume_m3",
        ):
            record[field] = _finite_float(field, raw[field], positive=True)
        for field in (
            "fracture_location_x_m",
            "fracture_location_y_m",
            "fracture_location_z_m",
        ):
            record[field] = _finite_float(field, raw[field])
        x_max = protocol.nelx * protocol.elem_size_m
        y_max = protocol.nely * protocol.elem_size_m
        z_max = protocol.nelz * protocol.elem_size_m
        for field, upper in (
            ("fracture_location_x_m", x_max),
            ("fracture_location_y_m", y_max),
            ("fracture_location_z_m", z_max),
        ):
            if not 0.0 <= record[field] <= upper:
                raise ValueError(
                    f"experimental {field} must lie within the specimen bounds"
                )
        fracture_region = str(raw["fracture_region"]).strip()
        if not fracture_region:
            raise ValueError("experimental fracture_region must be nonempty")
        record["fracture_region"] = fracture_region
        raw_data_path = str(raw["raw_data_path"]).strip()
        if not raw_data_path:
            raise ValueError("experimental raw_data_path must be nonempty")
        record["raw_data_path"] = raw_data_path
        record["notes"] = str(raw["notes"]).strip() or None
        normalized.append(record)
    return normalized


def validate_ansys_results(
    protocol: ComparisonProtocol,
    results: Optional[Sequence[Mapping[str, Any]]],
) -> list[Dict[str, Any]]:
    """Validate independent ANSYS records without inventing missing values."""
    _validate_protocol(protocol)
    if results is None:
        return []
    normalized: list[Dict[str, Any]] = []
    seen_designs: set[str] = set()
    for index, raw in enumerate(results):
        if not isinstance(raw, Mapping):
            raise ValueError(f"ANSYS result {index} must be a mapping")
        missing = [field for field in ANSYS_REQUIRED_FIELDS if field not in raw]
        if missing:
            raise ValueError(f"ANSYS result {index} is missing: {', '.join(missing)}")
        design_id = _normalize_design_id(raw["design_id"])
        if str(raw["comparison_id"]).strip() != protocol.comparison_id:
            raise ValueError("ANSYS comparison_id must match the protocol")
        if design_id in seen_designs:
            raise ValueError(f"duplicate ANSYS design_id {design_id}")
        seen_designs.add(design_id)
        for field in (
            "solver",
            "solver_version",
            "mesh_description",
            "critical_mode",
            "location_mapping_method",
        ):
            if not str(raw[field]).strip():
                raise ValueError(f"ANSYS {field} must be nonempty")
        critical_region = str(raw["critical_region"]).strip()
        if not critical_region:
            raise ValueError("ANSYS critical_region must be nonempty")
        observable = str(raw["failure_force_observable"]).strip()
        if observable != protocol.failure_force_observable:
            raise ValueError(
                "ANSYS failure_force_observable must match the preregistered protocol"
            )
        coordinate_frame = str(raw["coordinate_frame"]).strip()
        if coordinate_frame != DEFAULT_COORDINATE_FRAME:
            raise ValueError(
                "ANSYS coordinate_frame must use the documented PyTopo3D global frame"
            )
        location_raw = raw["critical_location_xyz_m"]
        if not isinstance(location_raw, Sequence) or isinstance(location_raw, str):
            raise ValueError("ANSYS critical_location_xyz_m must contain three values")
        if len(location_raw) != 3:
            raise ValueError("ANSYS critical_location_xyz_m must contain three values")
        location = [
            _finite_float("ANSYS critical_location_xyz_m", coordinate)
            for coordinate in location_raw
        ]
        for coordinate, upper, axis in zip(
            location,
            (
                protocol.nelx * protocol.elem_size_m,
                protocol.nely * protocol.elem_size_m,
                protocol.nelz * protocol.elem_size_m,
            ),
            "xyz",
        ):
            if not 0.0 <= coordinate <= upper:
                raise ValueError(
                    f"ANSYS critical_location_xyz_m {axis} is out of bounds"
                )
        failure_index = _finite_float("ANSYS failure_index", raw["failure_index"])
        if failure_index < 0.0:
            raise ValueError("ANSYS failure_index must be nonnegative")
        normalized.append(
            {
                "comparison_id": protocol.comparison_id,
                "common_case_sha256": _validate_sha256(
                    "ANSYS common_case_sha256", raw["common_case_sha256"]
                ),
                "design_id": design_id,
                "binary_topology_sha256": _validate_sha256(
                    "ANSYS binary_topology_sha256",
                    raw["binary_topology_sha256"],
                ),
                "geometry_artifact_path": str(
                    raw["geometry_artifact_path"]
                ).strip(),
                "geometry_artifact_sha256": _validate_sha256(
                    "ANSYS geometry_artifact_sha256",
                    raw["geometry_artifact_sha256"],
                ),
                "result_artifact_path": str(raw["result_artifact_path"]).strip(),
                "result_artifact_sha256": _validate_sha256(
                    "ANSYS result_artifact_sha256",
                    raw["result_artifact_sha256"],
                ),
                "solver": str(raw["solver"]).strip(),
                "solver_version": str(raw["solver_version"]).strip(),
                "mesh_description": str(raw["mesh_description"]).strip(),
                "failure_force_observable": observable,
                "failure_index": failure_index,
                "predicted_failure_force_N": _finite_float(
                    "ANSYS predicted_failure_force_N",
                    raw["predicted_failure_force_N"],
                    positive=True,
                ),
                "critical_mode": str(raw["critical_mode"]).strip(),
                "critical_region": critical_region,
                "critical_location_xyz_m": location,
                "coordinate_frame": coordinate_frame,
                "location_mapping_method": str(
                    raw["location_mapping_method"]
                ).strip(),
                "location_tolerance_m": _finite_float(
                    "ANSYS location_tolerance_m",
                    raw["location_tolerance_m"],
                    positive=True,
                ),
            }
        )
        if not normalized[-1]["geometry_artifact_path"]:
            raise ValueError("ANSYS geometry_artifact_path must be nonempty")
        if not normalized[-1]["result_artifact_path"]:
            raise ValueError("ANSYS result_artifact_path must be nonempty")
    return normalized


def binary_topology_metrics(
    baseline: np.ndarray,
    candidate: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    elem_size_m: float,
) -> Dict[str, Any]:
    """Return topology disagreement measures on the preregistered free region."""
    baseline = np.asarray(baseline, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    eligible = np.asarray(eligible_mask, dtype=bool)
    if baseline.shape != candidate.shape or baseline.shape != eligible.shape:
        raise ValueError("binary topology arrays and eligible_mask must share a shape")
    if not np.any(eligible):
        raise ValueError("eligible_mask must contain at least one element")
    a = baseline & eligible
    b = candidate & eligible
    union = int(np.sum(a | b))
    intersection = int(np.sum(a & b))
    jaccard = 1.0 if union == 0 else intersection / union
    total_occupied = int(np.sum(a)) + int(np.sum(b))
    dice = 1.0 if total_occupied == 0 else 2.0 * intersection / total_occupied
    xor_fraction = float(np.mean((a ^ b)[eligible]))

    def center_of_mass(mask):
        locations_yxz = np.argwhere(mask)
        if locations_yxz.size == 0:
            return None
        mean_yxz = np.mean(locations_yxz + 0.5, axis=0)
        return np.array([mean_yxz[1], mean_yxz[0], mean_yxz[2]]) * elem_size_m

    center_a = center_of_mass(a)
    center_b = center_of_mass(b)
    center_shift = (
        None
        if center_a is None or center_b is None
        else float(np.linalg.norm(center_b - center_a))
    )
    return {
        "binary_jaccard_similarity": float(jaccard),
        "binary_dice_similarity": float(dice),
        "binary_xor_fraction": xor_fraction,
        "binary_center_of_mass_shift_m": center_shift,
    }


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return float(numerator / denominator)


def _relative_error(observed: Optional[float], reference: Optional[float]) -> Optional[float]:
    if observed is None or reference is None or reference == 0.0:
        return None
    return float((observed - reference) / reference)


def _validate_common_evaluation_metrics(
    metrics: Mapping[str, Any],
    binary_density: np.ndarray,
    protocol: ComparisonProtocol,
    *,
    obstacle_mask: np.ndarray,
    protected_zone_mask: np.ndarray,
    expected_reference_load_N: float,
) -> Dict[str, Any]:
    """Enforce the Stage 10 contract used for every scientific comparison."""
    expected_shape = (protocol.nely, protocol.nelx, protocol.nelz)
    binary_density = np.asarray(binary_density, dtype=float)
    obstacle = np.asarray(obstacle_mask, dtype=bool)
    protected = np.asarray(protected_zone_mask, dtype=bool)
    if (
        binary_density.shape != expected_shape
        or obstacle.shape != expected_shape
        or protected.shape != expected_shape
    ):
        raise ValueError("common evaluation topology and masks must match the protocol mesh")
    if not np.all(np.isin(binary_density, (0.0, 1.0))):
        raise ValueError("common evaluation binary_density must contain only zero and one")
    if np.any(binary_density[obstacle] != 0.0):
        raise ValueError("common evaluation must keep every obstacle element void")
    if np.any(binary_density[protected] != 1.0):
        raise ValueError("common evaluation must keep every explicit protected element solid")
    expected_reference_load_N = _finite_float(
        "expected_reference_load_N", expected_reference_load_N, positive=True
    )
    required_numeric = {
        "compliance_binary": True,
        "predicted_stiffness_binary": True,
        "failure_index_max_binary": False,
        "predicted_failure_load_binary": True,
        "failure_reference_load_binary": True,
        "material_volume_m3_binary": False,
        "volume_fraction_binary": False,
    }
    missing = [name for name in required_numeric if name not in metrics]
    missing.extend(
        name
        for name in (
            "critical_mode_binary",
            "critical_element_yxz_binary",
            "critical_region_binary",
            "failure_strength_feasible_binary",
            "stage10_internal_verification_passed",
        )
        if name not in metrics
    )
    if missing:
        raise ValueError(
            "common evaluation is missing required Stage 10 metrics: "
            + ", ".join(sorted(set(missing)))
        )
    normalized = dict(metrics)
    for name, positive in required_numeric.items():
        value = _finite_float(name, metrics[name], positive=positive)
        if not positive and value < 0.0:
            raise ValueError(f"{name} must be nonnegative")
        normalized[name] = value
    if normalized["volume_fraction_binary"] > 1.0:
        raise ValueError("volume_fraction_binary must lie in [0, 1]")
    for name in (
        "failure_strength_feasible_binary",
        "stage10_internal_verification_passed",
    ):
        if not isinstance(metrics[name], (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
        normalized[name] = bool(metrics[name])
    for name in ("critical_mode_binary", "critical_region_binary"):
        value = str(metrics[name]).strip()
        if not value:
            raise ValueError(f"{name} must be nonempty")
        normalized[name] = value
    location = metrics["critical_element_yxz_binary"]
    if not isinstance(location, Sequence) or isinstance(location, str) or len(location) != 3:
        raise ValueError("critical_element_yxz_binary must contain three indices")
    bounded_location = []
    for coordinate, upper, axis in zip(
        location,
        (protocol.nely, protocol.nelx, protocol.nelz),
        "yxz",
    ):
        if isinstance(coordinate, bool) or int(coordinate) != coordinate:
            raise ValueError("critical_element_yxz_binary entries must be integers")
        coordinate = int(coordinate)
        if not 0 <= coordinate < upper:
            raise ValueError(f"critical_element_yxz_binary {axis} is out of bounds")
        bounded_location.append(coordinate)
    normalized["critical_element_yxz_binary"] = bounded_location

    non_obstacle = ~obstacle
    expected_volume = float(
        np.sum(binary_density[non_obstacle]) * protocol.elem_size_m**3
    )
    if not np.isclose(
        normalized["material_volume_m3_binary"],
        expected_volume,
        rtol=1.0e-10,
        atol=1.0e-15,
    ):
        raise ValueError(
            "material_volume_m3_binary is inconsistent with the binary topology"
        )
    expected_volume_fraction = float(np.mean(binary_density[non_obstacle]))
    if not np.isclose(
        normalized["volume_fraction_binary"],
        expected_volume_fraction,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(
            "volume_fraction_binary is inconsistent with the binary topology"
        )
    if not np.isclose(
        normalized["failure_reference_load_binary"],
        expected_reference_load_N,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(
            "failure_reference_load_binary is inconsistent with the common load case"
        )
    expected_failure_load = (
        normalized["failure_reference_load_binary"]
        / normalized["failure_index_max_binary"]
        if normalized["failure_index_max_binary"] > 0.0
        else None
    )
    if expected_failure_load is None or not np.isclose(
        normalized["predicted_failure_load_binary"],
        expected_failure_load,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(
            "predicted_failure_load_binary is inconsistent with reference load and FI"
        )
    expected_feasible = normalized["failure_index_max_binary"] <= 1.0 + 1.0e-9
    if normalized["failure_strength_feasible_binary"] != expected_feasible:
        raise ValueError(
            "failure_strength_feasible_binary is inconsistent with exact binary FI"
        )
    if normalized["stage10_internal_verification_passed"] != expected_feasible:
        raise ValueError(
            "stage10_internal_verification_passed is inconsistent with exact binary FI"
        )
    return normalized


def _assembled_reference_load_N(
    protocol: ComparisonProtocol, force_field: np.ndarray
) -> float:
    """Return the exact L1 nodal load used by the finite-element solver."""
    ndof = 3 * (protocol.nelx + 1) * (protocol.nely + 1) * (protocol.nelz + 1)
    force_vector = build_force_vector(
        protocol.nelx,
        protocol.nely,
        protocol.nelz,
        ndof,
        np.asarray(force_field, dtype=float),
    )
    return _finite_float(
        "assembled reference load", np.sum(np.abs(force_vector)), positive=True
    )


def _mean_or_none(records: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
    if not records:
        return None
    return float(np.mean([float(record[field]) for record in records]))


def _predicted_critical_xyz(
    predicted: Mapping[str, Any], protocol: ComparisonProtocol
) -> Optional[list[float]]:
    predicted_yxz = predicted.get("critical_element_yxz_binary")
    if predicted_yxz is None:
        return None
    return [
        (float(predicted_yxz[1]) + 0.5) * protocol.elem_size_m,
        (float(predicted_yxz[0]) + 0.5) * protocol.elem_size_m,
        (float(predicted_yxz[2]) + 0.5) * protocol.elem_size_m,
    ]


def _within_relative_tolerance(
    relative_error: Optional[float], tolerance: float
) -> Optional[bool]:
    if relative_error is None:
        return None
    return bool(abs(relative_error) <= tolerance + 1.0e-12)


def _experimental_summary(
    design_id: str,
    records: Sequence[Mapping[str, Any]],
    predicted: Mapping[str, Any],
    protocol: ComparisonProtocol,
    *,
    predicted_mass_kg: Optional[float],
) -> Dict[str, Any]:
    selected = [record for record in records if record["design_id"] == design_id]
    replicate_count = len(selected)
    status = (
        "not_run"
        if replicate_count == 0
        else (
            "complete"
            if replicate_count >= protocol.minimum_experimental_replicates
            else "partial"
        )
    )
    stiffness = _mean_or_none(selected, "experimental_stiffness_N_per_m")
    failure_force = _mean_or_none(selected, "experimental_failure_force_N")
    mass = _mean_or_none(selected, "mass_kg")
    volume = _mean_or_none(selected, "volume_m3")
    fracture_location = (
        None
        if not selected
        else [
            _mean_or_none(selected, "fracture_location_x_m"),
            _mean_or_none(selected, "fracture_location_y_m"),
            _mean_or_none(selected, "fracture_location_z_m"),
        ]
    )
    predicted_xyz = _predicted_critical_xyz(predicted, protocol)
    fracture_distance = (
        None
        if fracture_location is None or predicted_xyz is None
        else float(
            np.linalg.norm(
                np.asarray(fracture_location) - np.asarray(predicted_xyz)
            )
        )
    )
    individual_fracture_distances = (
        []
        if predicted_xyz is None
        else [
            float(
                np.linalg.norm(
                    np.asarray(
                        [
                            record["fracture_location_x_m"],
                            record["fracture_location_y_m"],
                            record["fracture_location_z_m"],
                        ],
                        dtype=float,
                    )
                    - np.asarray(predicted_xyz, dtype=float)
                )
            )
            for record in selected
        ]
    )

    def sample_std(field):
        if len(selected) < 2:
            return None
        return float(np.std([record[field] for record in selected], ddof=1))

    stiffness_error = _relative_error(
        stiffness, predicted.get("predicted_stiffness_binary")
    )
    failure_force_error = _relative_error(
        failure_force, predicted.get("predicted_failure_load_binary")
    )
    fracture_regions = sorted({record["fracture_region"] for record in selected})
    predicted_region = str(predicted.get("critical_region_binary", "")).strip()
    region_match = (
        None
        if not selected or not predicted_region
        else all(
            record["fracture_region"].casefold() == predicted_region.casefold()
            for record in selected
        )
    )
    location_tolerance_m = (
        protocol.critical_location_tolerance_elements * protocol.elem_size_m
    )
    location_match = (
        None
        if not individual_fracture_distances
        else bool(
            all(
                distance <= location_tolerance_m + 1.0e-12
                for distance in individual_fracture_distances
            )
        )
    )
    stiffness_agreement = _within_relative_tolerance(
        stiffness_error, protocol.experimental_relative_tolerance
    )
    failure_force_agreement = _within_relative_tolerance(
        failure_force_error, protocol.experimental_relative_tolerance
    )
    agreement_checks = (
        stiffness_agreement,
        failure_force_agreement,
        region_match,
        location_match,
    )
    agreement_passed = (
        None
        if status != "complete"
        else bool(all(check is True for check in agreement_checks))
    )
    return {
        "experimental_status": status,
        "experimental_replicate_count": replicate_count,
        "experimental_stiffness_N_per_m": stiffness,
        "experimental_stiffness_std_N_per_m": sample_std(
            "experimental_stiffness_N_per_m"
        ),
        "experimental_failure_force_N": failure_force,
        "experimental_failure_force_std_N": sample_std(
            "experimental_failure_force_N"
        ),
        "experimental_fracture_location_xyz_m": fracture_location,
        "predicted_critical_location_xyz_m": predicted_xyz,
        "critical_to_fracture_distance_m": fracture_distance,
        "critical_to_fracture_distances_m": individual_fracture_distances,
        "maximum_critical_to_fracture_distance_m": (
            None
            if not individual_fracture_distances
            else max(individual_fracture_distances)
        ),
        "experimental_fracture_regions": fracture_regions,
        "measured_mass_kg": mass,
        "measured_volume_m3": volume,
        "measured_mass_relative_error": _relative_error(
            mass, predicted_mass_kg
        ),
        "measured_volume_relative_error": _relative_error(
            volume, predicted.get("material_volume_m3_binary")
        ),
        "experimental_stiffness_relative_error": stiffness_error,
        "experimental_failure_force_relative_error": failure_force_error,
        "experimental_stiffness_agreement": stiffness_agreement,
        "experimental_failure_force_agreement": failure_force_agreement,
        "experimental_critical_region_match": region_match,
        "experimental_critical_location_match": location_match,
        "experimental_critical_location_tolerance_m": location_tolerance_m,
        "experimental_agreement_passed": agreement_passed,
        "manufacturing_artifact_sha256": sorted(
            {record["manufacturing_artifact_sha256"] for record in selected}
        ),
        "manufacturing_artifact_paths": sorted(
            {record["manufacturing_artifact_path"] for record in selected}
        ),
        "raw_data_sha256": sorted(
            {record["raw_data_sha256"] for record in selected}
        ),
        "raw_data_paths": sorted(
            {record["raw_data_path"] for record in selected}
        ),
        "stiffness_observable": protocol.stiffness_observable,
        "failure_force_observable": protocol.failure_force_observable,
    }


def _ansys_summary(
    design_id: str,
    records: Sequence[Mapping[str, Any]],
    predicted: Mapping[str, Any],
    protocol: ComparisonProtocol,
) -> Dict[str, Any]:
    record = next(
        (record for record in records if record["design_id"] == design_id),
        None,
    )
    if record is None:
        return {
            "ansys_status": "not_run",
            "ansys_solver": None,
            "ansys_failure_index": None,
            "ansys_predicted_failure_force_N": None,
            "ansys_critical_mode": None,
            "ansys_critical_region": None,
            "ansys_critical_location_xyz_m": None,
            "ansys_failure_index_relative_error": None,
            "ansys_failure_force_relative_error": None,
            "ansys_critical_mode_match": None,
            "ansys_critical_region_match": None,
            "ansys_critical_location_match": None,
            "ansys_critical_location_distance_m": None,
            "ansys_agreement_passed": None,
            "geometry_artifact_sha256": None,
            "result_artifact_sha256": None,
        }
    predicted_xyz = _predicted_critical_xyz(predicted, protocol)
    location_distance = (
        None
        if predicted_xyz is None
        else float(
            np.linalg.norm(
                np.asarray(record["critical_location_xyz_m"], dtype=float)
                - np.asarray(predicted_xyz, dtype=float)
            )
        )
    )
    preregistered_location_tolerance = (
        protocol.critical_location_tolerance_elements * protocol.elem_size_m
    )
    location_tolerance = min(
        record["location_tolerance_m"], preregistered_location_tolerance
    )
    fi_error = _relative_error(
        record["failure_index"], predicted.get("failure_index_max_binary")
    )
    failure_force_error = _relative_error(
        record["predicted_failure_force_N"],
        predicted.get("predicted_failure_load_binary"),
    )
    fi_agreement = _within_relative_tolerance(
        fi_error, protocol.ansys_relative_tolerance
    )
    force_agreement = _within_relative_tolerance(
        failure_force_error, protocol.ansys_relative_tolerance
    )
    mode_match = record["critical_mode"].casefold() == str(
        predicted.get("critical_mode_binary", "")
    ).casefold()
    region_match = record["critical_region"].casefold() == str(
        predicted.get("critical_region_binary", "")
    ).casefold()
    location_match = (
        None
        if location_distance is None
        else bool(location_distance <= location_tolerance + 1.0e-12)
    )
    agreement_passed = bool(
        fi_agreement is True
        and force_agreement is True
        and mode_match
        and region_match
        and location_match is True
    )
    return {
        "ansys_status": "complete",
        "ansys_solver": f"{record['solver']} {record['solver_version']}",
        "ansys_mesh_description": record["mesh_description"],
        "ansys_failure_index": record["failure_index"],
        "ansys_predicted_failure_force_N": record["predicted_failure_force_N"],
        "ansys_critical_mode": record["critical_mode"],
        "ansys_critical_region": record["critical_region"],
        "ansys_critical_location_xyz_m": record["critical_location_xyz_m"],
        "ansys_coordinate_frame": record["coordinate_frame"],
        "ansys_location_mapping_method": record["location_mapping_method"],
        "ansys_location_tolerance_m": location_tolerance,
        "ansys_critical_location_distance_m": location_distance,
        "ansys_failure_index_relative_error": fi_error,
        "ansys_failure_force_relative_error": failure_force_error,
        "ansys_failure_index_agreement": fi_agreement,
        "ansys_failure_force_agreement": force_agreement,
        "ansys_critical_mode_match": mode_match,
        "ansys_critical_region_match": region_match,
        "ansys_critical_location_match": location_match,
        "ansys_agreement_passed": agreement_passed,
        "geometry_artifact_sha256": record["geometry_artifact_sha256"],
        "geometry_artifact_path": record["geometry_artifact_path"],
        "result_artifact_sha256": record["result_artifact_sha256"],
        "result_artifact_path": record["result_artifact_path"],
        "failure_force_observable": protocol.failure_force_observable,
    }


def _pairwise_comparison(
    baseline_id: str,
    candidate_id: str,
    design_records: Mapping[str, Mapping[str, Any]],
    projected_densities: Mapping[str, np.ndarray],
    binary_densities: Mapping[str, np.ndarray],
    free_mask: np.ndarray,
    protocol: ComparisonProtocol,
) -> Dict[str, Any]:
    baseline = design_records[baseline_id]
    candidate = design_records[candidate_id]
    baseline_metrics = baseline["common_evaluation"]
    candidate_metrics = candidate["common_evaluation"]
    topology = binary_topology_metrics(
        binary_densities[baseline_id],
        binary_densities[candidate_id],
        free_mask,
        elem_size_m=protocol.elem_size_m,
    )
    projected_l1 = float(
        np.mean(
            np.abs(
                projected_densities[candidate_id][free_mask]
                - projected_densities[baseline_id][free_mask]
            )
        )
    )
    return {
        "baseline_design": baseline_id,
        "candidate_design": candidate_id,
        "candidate_to_baseline_common_compliance_ratio": _safe_ratio(
            candidate_metrics.get("compliance_binary"),
            baseline_metrics.get("compliance_binary"),
        ),
        "candidate_to_baseline_predicted_stiffness_ratio": _safe_ratio(
            candidate_metrics.get("predicted_stiffness_binary"),
            baseline_metrics.get("predicted_stiffness_binary"),
        ),
        "candidate_to_baseline_predicted_failure_force_ratio": _safe_ratio(
            candidate_metrics.get("predicted_failure_load_binary"),
            baseline_metrics.get("predicted_failure_load_binary"),
        ),
        "binary_failure_index_change": (
            None
            if candidate_metrics.get("failure_index_max_binary") is None
            or baseline_metrics.get("failure_index_max_binary") is None
            else float(
                candidate_metrics["failure_index_max_binary"]
                - baseline_metrics["failure_index_max_binary"]
            )
        ),
        "projected_density_l1_free_region": projected_l1,
        **topology,
    }


def _strict_common_case_hash(
    protocol: ComparisonProtocol,
    array_hashes: Mapping[str, str],
    material_strength: MaterialStrength,
) -> str:
    payload = {
        "protocol": protocol.as_mapping(),
        "array_hashes": dict(array_hashes),
        "material_strength": material_strength.as_dict(),
    }
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_experimental_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENTAL_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column) for column in EXPERIMENTAL_COLUMNS})


def _write_design_csv(path: Path, records: Mapping[str, Mapping[str, Any]]) -> None:
    columns = (
        "design_id",
        "label",
        "optimizer_material",
        "optimizer_material_id",
        "optimization_mode",
        "optimizer_feasible",
        "optimizer_compliance",
        "projected_free_volume_fraction",
        "binary_free_volume_fraction",
        "common_binary_compliance",
        "common_binary_stiffness",
        "exact_binary_failure_index",
        "predicted_failure_force_N",
        "critical_mode",
        "critical_region",
        "binary_strength_feasible",
        "binary_material_volume_m3",
        "predicted_mass_kg",
        "experimental_replicate_count",
        "experimental_stiffness_N_per_m",
        "experimental_failure_force_N",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for design_id in DESIGN_IDS:
            record = records[design_id]
            metrics = record["common_evaluation"]
            experimental = record["experimental"]
            writer.writerow(
                {
                    "design_id": design_id,
                    "label": record["label"],
                    "optimizer_material": record["optimizer_material"],
                    "optimizer_material_id": record["optimizer_material_id"],
                    "optimization_mode": record["optimization_mode"],
                    "optimizer_feasible": record["optimizer_feasible"],
                    "optimizer_compliance": record["optimizer_compliance"],
                    "projected_free_volume_fraction": record[
                        "projected_free_volume_fraction"
                    ],
                    "binary_free_volume_fraction": record[
                        "binary_free_volume_fraction"
                    ],
                    "common_binary_compliance": metrics.get("compliance_binary"),
                    "common_binary_stiffness": metrics.get(
                        "predicted_stiffness_binary"
                    ),
                    "exact_binary_failure_index": metrics.get(
                        "failure_index_max_binary"
                    ),
                    "predicted_failure_force_N": metrics.get(
                        "predicted_failure_load_binary"
                    ),
                    "critical_mode": metrics.get("critical_mode_binary"),
                    "critical_region": metrics.get("critical_region_binary"),
                    "binary_strength_feasible": metrics.get(
                        "failure_strength_feasible_binary"
                    ),
                    "binary_material_volume_m3": record[
                        "binary_material_volume_m3"
                    ],
                    "predicted_mass_kg": record["predicted_mass_kg"],
                    "experimental_replicate_count": experimental[
                        "experimental_replicate_count"
                    ],
                    "experimental_stiffness_N_per_m": experimental[
                        "experimental_stiffness_N_per_m"
                    ],
                    "experimental_failure_force_N": experimental[
                        "experimental_failure_force_N"
                    ],
                }
            )


def _simulation_gate_failures(
    protocol: ComparisonProtocol,
    design_records: Mapping[str, Dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    for design_id in DESIGN_IDS:
        record = design_records[design_id]
        if record["run_status"] != "complete":
            failures.append(f"design_{design_id}_{record['run_status']}")
            continue
        diagnostics = record["optimizer_diagnostics"]
        continuation_completed = diagnostics.get("continuation_completed") is True
        projection_converged = diagnostics.get("projection_converged") is True
        termination_status = diagnostics.get("termination_status")
        stages_requested = diagnostics.get("continuation_stages_requested")
        stages_completed = diagnostics.get("continuation_stages_completed")
        expected_stages = len(protocol.beta_schedule)
        numerical_acceptance = {
            "optimizer_feasible": bool(record["optimizer_feasible"]),
            "continuation_completed": continuation_completed,
            "projection_converged": projection_converged,
            "termination_status": termination_status,
            "termination_status_accepted": (
                termination_status in protocol.acceptable_termination_statuses
            ),
            "continuation_stages_requested": stages_requested,
            "continuation_stages_completed": stages_completed,
            "expected_continuation_stages": expected_stages,
        }
        record["numerical_acceptance"] = numerical_acceptance
        if not record["optimizer_feasible"]:
            failures.append(f"design_{design_id}_optimizer_infeasible")
        if (
            protocol.require_continuation_completed
            and not continuation_completed
        ):
            failures.append(f"design_{design_id}_continuation_incomplete")
        if stages_requested != expected_stages or stages_completed != expected_stages:
            failures.append(f"design_{design_id}_continuation_stage_count_mismatch")
        if termination_status not in protocol.acceptable_termination_statuses:
            failures.append(f"design_{design_id}_termination_status_unaccepted")
        if protocol.require_projection_converged and not projection_converged:
            failures.append(f"design_{design_id}_projection_not_converged")
        if (
            abs(record["projected_free_volume_fraction"] - protocol.volfrac)
            > protocol.binary_volume_tolerance
        ):
            failures.append(
                f"design_{design_id}_projected_volume_outside_tolerance"
            )
        if (
            abs(record["binary_free_volume_fraction"] - protocol.volfrac)
            > protocol.binary_volume_tolerance
        ):
            failures.append(f"design_{design_id}_binary_volume_outside_tolerance")

    if all(
        design_records[design_id]["run_status"] == "complete"
        for design_id in DESIGN_IDS
    ):
        binary_volumes = [
            design_records[design_id]["binary_free_volume_fraction"]
            for design_id in DESIGN_IDS
        ]
        if (
            max(binary_volumes) - min(binary_volumes)
            > protocol.binary_volume_tolerance
        ):
            failures.append("binary_volume_mismatch_between_designs")
        c_metrics = design_records["C"]["common_evaluation"]
        if not c_metrics["stage10_internal_verification_passed"]:
            failures.append("design_C_exact_binary_strength_failed")
        if c_metrics["failure_index_max_binary"] > 1.0 + 1.0e-9:
            failures.append("design_C_exact_binary_failure_index_exceeds_one")
    return list(dict.fromkeys(failures))


def _bind_external_records(
    report: Mapping[str, Any],
    experiments: Sequence[Mapping[str, Any]],
    ansys: Sequence[Mapping[str, Any]],
) -> None:
    common_case_hash = report["common_case_sha256"]
    designs = report["designs"]
    for source_name, records in (("experimental", experiments), ("ANSYS", ansys)):
        for record in records:
            design_id = record["design_id"]
            if record["common_case_sha256"] != common_case_hash:
                raise ValueError(
                    f"{source_name} common-case hash does not match this comparison"
                )
            expected = designs[design_id].get("binary_topology_sha256")
            if expected is None or record["binary_topology_sha256"] != expected:
                raise ValueError(
                    f"{source_name} topology hash does not match Design {design_id}"
                )


def _ingest_external_artifacts(
    root: Path,
    experiments: Sequence[Mapping[str, Any]],
    ansys: Sequence[Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Verify every declared external digest and copy artifacts into the run."""
    normalized_experiments = [dict(record) for record in experiments]
    normalized_ansys = [dict(record) for record in ansys]
    copy_plan: list[tuple[Path, Path, Dict[str, Any], str, str]] = []

    def register(
        record: Dict[str, Any],
        path_field: str,
        hash_field: str,
        destination_parts: Sequence[str],
    ) -> None:
        source = Path(record[path_field]).expanduser()
        if not source.is_absolute():
            source = root / source
        source = source.resolve()
        if not source.is_file():
            raise ValueError(f"external artifact does not exist: {source}")
        expected_hash = record[hash_field]
        actual_hash = _file_hash(source)
        if actual_hash != expected_hash:
            raise ValueError(
                f"external artifact hash mismatch for {path_field}: {source}"
            )
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", source.name)
        artifact_root = (root / "external_artifacts").resolve()
        try:
            source.relative_to(artifact_root)
            destination = source
        except ValueError:
            destination = root.joinpath(
                "external_artifacts",
                *destination_parts,
                f"{actual_hash}_{safe_name}",
            ).resolve()
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise ValueError("external artifact destination escaped output root") from exc
        copy_plan.append(
            (source, destination, record, path_field, actual_hash)
        )

    for record in normalized_experiments:
        location = ("experimental", record["design_id"], record["specimen_id"])
        register(
            record,
            "manufacturing_artifact_path",
            "manufacturing_artifact_sha256",
            location,
        )
        register(record, "raw_data_path", "raw_data_sha256", location)
    for record in normalized_ansys:
        location = ("ansys", record["design_id"])
        register(
            record,
            "geometry_artifact_path",
            "geometry_artifact_sha256",
            location,
        )
        register(
            record,
            "result_artifact_path",
            "result_artifact_sha256",
            location,
        )

    # No filesystem changes occur until every source file and digest has passed.
    for source, destination, record, path_field, expected_hash in copy_plan:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination:
            shutil.copy2(source, destination)
        if _file_hash(destination) != expected_hash:
            raise ValueError(
                f"copied external artifact failed verification: {destination}"
            )
        record[path_field] = destination.relative_to(root).as_posix()
    return normalized_experiments, normalized_ansys


def _agreement_failure_names(
    design_records: Mapping[str, Mapping[str, Any]],
    *,
    experiments_complete: bool,
    ansys_complete: bool,
) -> list[str]:
    failures: list[str] = []
    if experiments_complete:
        checks = (
            ("experimental_stiffness_agreement", "stiffness"),
            ("experimental_failure_force_agreement", "failure_force"),
            ("experimental_critical_region_match", "critical_region"),
            ("experimental_critical_location_match", "critical_location"),
        )
        for design_id in DESIGN_IDS:
            summary = design_records[design_id]["experimental"]
            for field, label in checks:
                if summary[field] is not True:
                    failures.append(
                        f"design_{design_id}_experimental_{label}_disagreement"
                    )
    if ansys_complete:
        checks = (
            ("ansys_failure_index_agreement", "failure_index"),
            ("ansys_failure_force_agreement", "failure_force"),
            ("ansys_critical_mode_match", "critical_mode"),
            ("ansys_critical_region_match", "critical_region"),
            ("ansys_critical_location_match", "critical_location"),
        )
        for design_id in DESIGN_IDS:
            summary = design_records[design_id]["ansys"]
            for field, label in checks:
                if summary[field] is not True:
                    failures.append(f"design_{design_id}_ansys_{label}_disagreement")
    return failures


def _relative_range(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    mean_value = float(np.mean(values))
    if mean_value <= 0.0:
        return None
    return float((max(values) - min(values)) / mean_value)


def _standardized_failure_force_effect(
    experiments: Sequence[Mapping[str, Any]],
) -> Optional[float]:
    baseline = np.asarray(
        [
            record["experimental_failure_force_N"]
            for record in experiments
            if record["design_id"] == "B"
        ],
        dtype=float,
    )
    candidate = np.asarray(
        [
            record["experimental_failure_force_N"]
            for record in experiments
            if record["design_id"] == "C"
        ],
        dtype=float,
    )
    if baseline.size < 2 or candidate.size < 2:
        return None
    degrees_of_freedom = baseline.size + candidate.size - 2
    pooled_variance = (
        (baseline.size - 1) * float(np.var(baseline, ddof=1))
        + (candidate.size - 1) * float(np.var(candidate, ddof=1))
    ) / degrees_of_freedom
    if pooled_variance <= 0.0:
        return None
    return float(
        (float(np.mean(candidate)) - float(np.mean(baseline)))
        / math.sqrt(pooled_variance)
    )


def _physical_comparison_fairness(
    protocol: ComparisonProtocol,
    design_records: Mapping[str, Mapping[str, Any]],
    *,
    experiments_complete: bool,
) -> Dict[str, Any]:
    if not experiments_complete:
        return {
            "physical_comparison_fair": None,
            "measured_volume_relative_range": None,
            "measured_mass_relative_range": None,
            "measured_volume_matches_predicted": None,
            "measured_mass_matches_predicted": None,
        }
    summaries = [
        design_records[design_id]["experimental"] for design_id in DESIGN_IDS
    ]
    volumes = [summary["measured_volume_m3"] for summary in summaries]
    masses = [summary["measured_mass_kg"] for summary in summaries]
    volume_range = _relative_range(volumes)
    mass_range = _relative_range(masses)
    volume_matches_predicted = all(
        summary["measured_volume_relative_error"] is not None
        and abs(summary["measured_volume_relative_error"])
        <= protocol.measured_volume_relative_tolerance + 1.0e-12
        for summary in summaries
    )
    mass_matches_predicted = (
        None
        if protocol.bulk_density_kg_m3 is None
        else all(
            summary["measured_mass_relative_error"] is not None
            and abs(summary["measured_mass_relative_error"])
            <= protocol.measured_mass_relative_tolerance + 1.0e-12
            for summary in summaries
        )
    )
    volume_range_passed = bool(
        volume_range is not None
        and volume_range <= protocol.measured_volume_relative_tolerance + 1.0e-12
    )
    mass_range_passed = bool(
        mass_range is not None
        and mass_range <= protocol.measured_mass_relative_tolerance + 1.0e-12
    )
    fair = bool(
        volume_matches_predicted
        and volume_range_passed
        and mass_range_passed
        and (mass_matches_predicted in {None, True})
    )
    return {
        "physical_comparison_fair": fair,
        "measured_volume_relative_range": volume_range,
        "measured_mass_relative_range": mass_range,
        "measured_volume_matches_predicted": volume_matches_predicted,
        "measured_mass_matches_predicted": mass_matches_predicted,
        "measured_volume_relative_tolerance": (
            protocol.measured_volume_relative_tolerance
        ),
        "measured_mass_relative_tolerance": protocol.measured_mass_relative_tolerance,
    }


def _apply_external_evidence(
    root: Path,
    protocol: ComparisonProtocol,
    base_report: Mapping[str, Any],
    experiments: Sequence[Mapping[str, Any]],
    ansys: Sequence[Mapping[str, Any]],
) -> ResearchComparisonResult:
    """Bind, assess, and persist external evidence without rerunning simulation."""
    report = _json_safe(base_report)
    _bind_external_records(report, experiments, ansys)
    experiments, ansys = _ingest_external_artifacts(root, experiments, ansys)
    design_records = report["designs"]
    for design_id in DESIGN_IDS:
        record = design_records[design_id]
        record["experimental"] = _experimental_summary(
            design_id,
            experiments,
            record["common_evaluation"],
            protocol,
            predicted_mass_kg=record.get("predicted_mass_kg"),
        )
        record["ansys"] = _ansys_summary(
            design_id,
            ansys,
            record["common_evaluation"],
            protocol,
        )

    replicate_counts = {
        design_id: design_records[design_id]["experimental"][
            "experimental_replicate_count"
        ]
        for design_id in DESIGN_IDS
    }
    experiments_complete = all(
        count >= protocol.minimum_experimental_replicates
        for count in replicate_counts.values()
    )
    ansys_complete = {record["design_id"] for record in ansys} == set(DESIGN_IDS)
    experimental_source_status = (
        "not_run" if not experiments else ("complete" if experiments_complete else "partial")
    )
    ansys_source_status = (
        "not_run" if not ansys else ("complete" if ansys_complete else "partial")
    )
    agreement_failures = _agreement_failure_names(
        design_records,
        experiments_complete=experiments_complete,
        ansys_complete=ansys_complete,
    )
    experimental_agreement = (
        None
        if not experiments_complete
        else all(
            design_records[design_id]["experimental"][
                "experimental_agreement_passed"
            ]
            is True
            for design_id in DESIGN_IDS
        )
    )
    ansys_agreement = (
        None
        if not ansys_complete
        else all(
            design_records[design_id]["ansys"]["ansys_agreement_passed"] is True
            for design_id in DESIGN_IDS
        )
    )

    external_failures: list[str] = []
    if experimental_source_status != "complete":
        external_failures.append(
            "experimental_records_not_run"
            if experimental_source_status == "not_run"
            else "experimental_replicates_incomplete"
        )
    if ansys_source_status != "complete":
        external_failures.append(
            "ansys_records_not_run"
            if ansys_source_status == "not_run"
            else "ansys_records_incomplete"
        )
    if protocol.material_data_provenance != "measured":
        external_failures.append("material_properties_not_measured")
    external_failures.extend(agreement_failures)

    simulation_validation = report["validation"]
    simulation_gate_passed = bool(simulation_validation["simulation_gate_passed"])
    if not simulation_gate_passed:
        validation_status = "simulation_invalid"
    elif not external_failures:
        validation_status = "validation_complete"
    elif experiments or ansys:
        validation_status = "external_validation_partial"
    else:
        validation_status = "simulation_complete_external_pending"

    b_vs_c = report.get("pairwise_comparisons", {}).get(
        "B_vs_C_directional_strength_effect", {}
    )
    topology_changed = (
        None
        if not b_vs_c
        else bool(
            b_vs_c["binary_xor_fraction"] >= protocol.topology_change_threshold
        )
    )
    predicted_force_ratio = b_vs_c.get(
        "candidate_to_baseline_predicted_failure_force_ratio"
    )
    experimental_force_ratio = _safe_ratio(
        design_records["C"]["experimental"].get(
            "experimental_failure_force_N"
        ),
        design_records["B"]["experimental"].get(
            "experimental_failure_force_N"
        ),
    )
    experimental_improvement = (
        None
        if not experiments_complete or experimental_force_ratio is None
        else bool(experimental_force_ratio > 1.0)
    )
    fairness = _physical_comparison_fairness(
        protocol,
        design_records,
        experiments_complete=experiments_complete,
    )
    standardized_effect = _standardized_failure_force_effect(experiments)
    improvement_ratio_gate = (
        experimental_force_ratio is not None
        and experimental_force_ratio
        >= protocol.minimum_failure_force_improvement_ratio
    )
    standardized_effect_gate = (
        standardized_effect is not None
        and standardized_effect >= protocol.minimum_standardized_effect_size
    )
    research_claim_supported = bool(
        validation_status == "validation_complete"
        and topology_changed is True
        and fairness["physical_comparison_fair"] is True
        and improvement_ratio_gate
        and standardized_effect_gate
    )
    report["research_questions"] = {
        "strength_changed_binary_topology": topology_changed,
        "strength_improved_predicted_failure_force": (
            None
            if predicted_force_ratio is None
            else bool(predicted_force_ratio > 1.0)
        ),
        "experimental_strength_improvement_ratio_C_to_B": (
            experimental_force_ratio
        ),
        "experimental_evidence_complete": bool(
            experiments_complete
            and protocol.material_data_provenance == "measured"
        ),
        "experimental_model_agreement_passed": experimental_agreement,
        "ansys_model_agreement_passed": ansys_agreement,
        "experimental_strength_improvement_observed": experimental_improvement,
        "standardized_failure_force_effect_C_vs_B": standardized_effect,
        "preregistered_minimum_failure_force_improvement_ratio": (
            protocol.minimum_failure_force_improvement_ratio
        ),
        "preregistered_minimum_standardized_effect_size": (
            protocol.minimum_standardized_effect_size
        ),
        "failure_force_improvement_ratio_gate_passed": improvement_ratio_gate,
        "standardized_effect_size_gate_passed": standardized_effect_gate,
        **fairness,
        "research_claim_supported": research_claim_supported,
    }
    critical_region_status = (
        "external_agreement_passed"
        if experimental_agreement is True and ansys_agreement is True
        else (
            "external_disagreement"
            if experimental_agreement is False or ansys_agreement is False
            else "external_review_not_complete"
        )
    )
    validation = {
        **simulation_validation,
        "status": validation_status,
        "ansys_status": ansys_source_status,
        "experimental_status": experimental_source_status,
        "experimental_replicates_by_design": replicate_counts,
        "material_data_provenance": protocol.material_data_provenance,
        "experimental_agreement_passed": experimental_agreement,
        "ansys_agreement_passed": ansys_agreement,
        "external_validation_failures": external_failures,
        "critical_region_artifact_validation_status": critical_region_status,
    }
    report["validation"] = validation
    report["external_evidence"] = {
        "experimental_record_count": len(experiments),
        "ansys_record_count": len(ansys),
        "common_case_binding_required": True,
        "binary_topology_binding_required": True,
        "external_artifact_digest_verification_passed": (
            True if experiments or ansys else None
        ),
    }

    experimental_path = root / "experimental_measurements.csv"
    ansys_path = root / "ansys_results.json"
    if experiments:
        _write_experimental_csv(experimental_path, experiments)
    elif experimental_path.exists():
        experimental_path.unlink()
    if ansys:
        _write_strict_json(ansys_path, list(ansys))
    elif ansys_path.exists():
        ansys_path.unlink()
    for design_id in DESIGN_IDS:
        _write_strict_json(
            root / f"design_{design_id}" / "metrics.json",
            design_records[design_id],
        )
    _write_design_csv(root / "comparison.csv", design_records)
    _write_strict_json(root / "comparison.json", report)
    _write_strict_json(root / "validation_status.json", validation)
    return ResearchComparisonResult(
        output_directory=str(root),
        validation_status=validation_status,
        simulation_gate_passed=simulation_gate_passed,
        report=_json_safe(report),
    )


def run_research_comparison(
    protocol: ComparisonProtocol,
    inputs: ComparisonInputs,
    output_dir: str | Path,
    *,
    experimental_measurements: Optional[Sequence[Mapping[str, Any]]] = None,
    ansys_results: Optional[Sequence[Mapping[str, Any]]] = None,
    optimization_function: Optional[Callable[..., Any]] = None,
    evaluation_function: Optional[Callable[..., Any]] = None,
) -> ResearchComparisonResult:
    """Run and persist the controlled A/B/C research comparison.

    The three optimizations differ only in the registered material/failure
    treatment.  Every resulting topology is then re-solved with the same
    orthotropic stiffness, directional strengths, loads, and supports.
    """
    _validate_protocol(protocol)
    arrays = _validate_inputs(protocol, inputs)
    material_strength = _validated_strength(protocol, inputs.material_strength)
    experiments = validate_experimental_measurements(
        protocol, experimental_measurements
    )
    ansys = validate_ansys_results(protocol, ansys_results)
    optimization_function = optimization_function or top3d
    evaluation_function = evaluation_function or evaluate_failure_representations

    root = Path(output_dir).expanduser().resolve() / protocol.comparison_id
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"comparison output already exists and is not empty: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    input_dir = root / "inputs"
    input_dir.mkdir(exist_ok=True)

    _write_strict_json(root / "protocol.json", protocol.as_mapping())
    _write_experimental_csv(root / "experimental_measurements_template.csv", [])

    saved_input_names = (
        "force_field",
        "support_mask",
        "obstacle_mask",
        "protected_zone_mask",
        "effective_protected_mask",
        "free_mask",
        "initial_design",
        "material_orientation",
        "isotropic_material_params",
        "orthotropic_material_params_material_axes",
        "orthotropic_material_params_global",
    )
    input_manifest: Dict[str, Any] = {"arrays": {}}
    array_hashes: Dict[str, str] = {}
    for name in saved_input_names:
        array = arrays[name]
        path = input_dir / f"{name}.npy"
        np.save(path, array)
        array_hash = _array_hash(array)
        array_hashes[name] = array_hash
        input_manifest["arrays"][name] = {
            "path": path.relative_to(root).as_posix(),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "array_sha256": array_hash,
            "file_sha256": _file_hash(path),
        }
    common_case_hash = _strict_common_case_hash(
        protocol, array_hashes, material_strength
    )
    input_manifest.update(
        {
            "common_case_sha256": common_case_hash,
            "isotropic_material_params": arrays[
                "isotropic_material_params"
            ].tolist(),
            "orthotropic_material_params_material_axes": arrays[
                "orthotropic_material_params_material_axes"
            ].tolist(),
            "orthotropic_material_params_global": arrays[
                "orthotropic_material_params_global"
            ].tolist(),
            "material_strength": material_strength.as_dict(),
            "print_orientation_xyz": protocol.print_orientation_xyz,
        }
    )
    _write_strict_json(input_dir / "manifest.json", input_manifest)

    variants = build_design_variants(protocol, inputs)
    design_records: Dict[str, Dict[str, Any]] = {}
    projected_densities: Dict[str, np.ndarray] = {}
    binary_densities: Dict[str, np.ndarray] = {}

    for variant in variants:
        design_id = variant.design_id
        design_dir = root / f"design_{design_id}"
        manager = ResultsManager(
            base_dir=str(root),
            experiment_name=f"design_{design_id}",
            description=variant.label,
        )
        diagnostics: Dict[str, Any] = {}
        base_record: Dict[str, Any] = {
            "design_id": design_id,
            "label": variant.label,
            "optimizer_material": variant.optimizer_material,
            "optimizer_material_id": variant.optimizer_material_id,
            "optimization_mode": variant.optimization_mode,
            "optimizer": protocol.optimizer,
            "uses_failure_constraint": variant.uses_failure_constraint,
            "common_case_sha256": common_case_hash,
            "common_input_hashes": dict(array_hashes),
            "print_orientation_xyz": protocol.print_orientation_xyz,
            "target_free_volume_fraction": protocol.volfrac,
            "stiffness_observable": protocol.stiffness_observable,
            "failure_force_observable": protocol.failure_force_observable,
            "run_status": "not_run",
            "optimizer_feasible": False,
            "optimizer_compliance": None,
            "projected_free_volume_fraction": None,
            "binary_free_volume_fraction": None,
            "binary_material_volume_m3": None,
            "predicted_mass_kg": None,
            "projected_density_path": None,
            "binary_density_path": None,
            "projected_density_sha256": None,
            "binary_topology_sha256": None,
            "optimizer_diagnostics": {},
            "common_evaluation": {},
            "error": None,
        }
        optimization_kwargs: Dict[str, Any] = {
            "nelx": protocol.nelx,
            "nely": protocol.nely,
            "nelz": protocol.nelz,
            "volfrac": protocol.volfrac,
            "penal": protocol.penal,
            "rmin": protocol.rmin,
            "disp_thres": protocol.disp_thres,
            "material_params": variant.material_params,
            "elem_size": protocol.elem_size_m,
            "force_field": np.array(arrays["force_field"], copy=True),
            "support_mask": np.array(arrays["support_mask"], copy=True),
            "obstacle_mask": np.array(arrays["obstacle_mask"], copy=True),
            "protected_zone_mask": np.array(
                arrays["protected_zone_mask"], copy=True
            ),
            "tolx": protocol.tolx,
            "maxloop": protocol.maxloop,
            "save_history": False,
            "use_gpu": protocol.use_gpu,
            "beta_schedule": tuple(protocol.beta_schedule),
            "projection_eta": protocol.projection_eta,
            "diagnostics_out": diagnostics,
            "optimization_mode": variant.optimization_mode,
            "optimizer": protocol.optimizer,
            "mma_move": protocol.mma_move,
            "mma_min_density": protocol.mma_min_density,
            "initial_design": np.array(arrays["initial_design"], copy=True),
        }
        if variant.uses_failure_constraint:
            optimization_kwargs.update(
                {
                    "material_strength": material_strength,
                    "material_orientation": np.array(
                        arrays["material_orientation"], copy=True
                    ),
                    "failure_limit": float(protocol.failure_limit_schedule[-1]),
                    "failure_aggregate_exponent": float(
                        protocol.failure_aggregate_exponent_schedule[-1]
                    ),
                    "failure_relaxation_exponent": (
                        protocol.failure_relaxation_exponent
                    ),
                    "failure_limit_schedule": tuple(
                        protocol.failure_limit_schedule
                    ),
                    "failure_aggregate_exponent_schedule": tuple(
                        protocol.failure_aggregate_exponent_schedule
                    ),
                }
            )

        try:
            optimized = optimization_function(**optimization_kwargs)
            if not isinstance(optimized, tuple) or len(optimized) != 4:
                raise ValueError("optimization_function must return four values")
            projected_density, _, optimizer_compliance, _ = optimized
            projected_density = np.asarray(projected_density, dtype=float)
            expected_shape = (protocol.nely, protocol.nelx, protocol.nelz)
            if projected_density.shape != expected_shape:
                raise ValueError(
                    f"optimized density has shape {projected_density.shape}, "
                    f"expected {expected_shape}"
                )
            if not np.all(np.isfinite(projected_density)) or np.any(
                (projected_density < 0.0) | (projected_density > 1.0)
            ):
                raise ValueError("optimized density must be finite and in [0, 1]")
            optimizer_compliance = _finite_float(
                "optimizer_compliance", optimizer_compliance, positive=True
            )
        except Exception as exc:
            base_record.update(
                {
                    "run_status": "optimizer_error",
                    "optimizer_diagnostics": diagnostics,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
            design_records[design_id] = base_record
            _write_strict_json(design_dir / "metrics.json", base_record)
            continue

        projected_path = Path(
            manager.save_result(projected_density, "projected_density.npy")
        ).resolve()
        try:
            verification = evaluation_function(
                x_projected=projected_density,
                binary_threshold=protocol.binary_threshold,
                penal=protocol.penal,
                material_params=tuple(
                    arrays["orthotropic_material_params_global"].tolist()
                ),
                strength=material_strength,
                orientation_matrix=np.array(
                    arrays["material_orientation"], copy=True
                ),
                elem_size=protocol.elem_size_m,
                force_field=np.array(arrays["force_field"], copy=True),
                support_mask=np.array(arrays["support_mask"], copy=True),
                obstacle_mask=np.array(arrays["obstacle_mask"], copy=True),
                protected_zone_mask=np.array(
                    arrays["protected_zone_mask"], copy=True
                ),
                smooth_failure_aggregate=(
                    diagnostics.get("failure_aggregate")
                    if variant.uses_failure_constraint
                    else None
                ),
                smooth_failure_limit=(
                    diagnostics.get("failure_limit")
                    if variant.uses_failure_constraint
                    else None
                ),
                use_gpu=protocol.use_gpu,
                results_manager=manager,
            )
            binary_density = np.asarray(verification.binary_density, dtype=float)
            if binary_density.shape != projected_density.shape or not np.all(
                np.isin(binary_density, (0.0, 1.0))
            ):
                raise ValueError("evaluation returned an invalid binary density")
            common_metrics = _validate_common_evaluation_metrics(
                verification.metrics,
                binary_density,
                protocol,
                obstacle_mask=arrays["obstacle_mask"],
                protected_zone_mask=arrays["protected_zone_mask"],
                expected_reference_load_N=_assembled_reference_load_N(
                    protocol, arrays["force_field"]
                ),
            )
            _json_safe(common_metrics, f"design_{design_id}.common_evaluation")
        except Exception as exc:
            base_record.update(
                {
                    "run_status": "evaluation_error",
                    "optimizer_compliance": optimizer_compliance,
                    "optimizer_feasible": bool(
                        diagnostics.get("optimization_feasible", False)
                    ),
                    "optimizer_diagnostics": diagnostics,
                    "projected_density_path": projected_path.relative_to(
                        root
                    ).as_posix(),
                    "projected_density_sha256": _array_hash(projected_density),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
            design_records[design_id] = base_record
            _write_strict_json(design_dir / "metrics.json", base_record)
            continue

        binary_path = Path(manager.save_result(binary_density, "binary_density.npy")).resolve()
        projected_densities[design_id] = np.array(projected_density, copy=True)
        binary_densities[design_id] = np.array(binary_density, copy=True)
        free = arrays["free_mask"]
        projected_free_volume = float(np.mean(projected_density[free]))
        binary_free_volume = float(np.mean(binary_density[free]))
        binary_volume = common_metrics.get("material_volume_m3_binary")
        predicted_mass = (
            None
            if protocol.bulk_density_kg_m3 is None or binary_volume is None
            else float(binary_volume * protocol.bulk_density_kg_m3)
        )
        base_record.update(
            {
                "run_status": "complete",
                "optimizer_feasible": bool(
                    diagnostics.get("optimization_feasible", False)
                ),
                "optimizer_compliance": optimizer_compliance,
                "projected_free_volume_fraction": projected_free_volume,
                "binary_free_volume_fraction": binary_free_volume,
                "binary_material_volume_m3": binary_volume,
                "predicted_mass_kg": predicted_mass,
                "projected_density_path": projected_path.relative_to(
                    root
                ).as_posix(),
                "binary_density_path": binary_path.relative_to(root).as_posix(),
                "projected_density_sha256": _array_hash(projected_density),
                "binary_topology_sha256": _array_hash(binary_density),
                "optimizer_diagnostics": diagnostics,
                "common_evaluation": common_metrics,
            }
        )
        design_records[design_id] = base_record

    successful = set(projected_densities) == set(DESIGN_IDS)
    pairwise: Dict[str, Any] = {}
    if successful:
        pairwise = {
            "A_vs_B_anisotropic_stiffness_effect": _pairwise_comparison(
                "A",
                "B",
                design_records,
                projected_densities,
                binary_densities,
                arrays["free_mask"],
                protocol,
            ),
            "B_vs_C_directional_strength_effect": _pairwise_comparison(
                "B",
                "C",
                design_records,
                projected_densities,
                binary_densities,
                arrays["free_mask"],
                protocol,
            ),
        }

    simulation_failures = _simulation_gate_failures(protocol, design_records)
    simulation_gate_passed = not simulation_failures
    validation = {
        "status": (
            "simulation_complete_external_pending"
            if simulation_gate_passed
            else "simulation_invalid"
        ),
        "simulation_gate_passed": simulation_gate_passed,
        "simulation_failures": simulation_failures,
    }
    report: Dict[str, Any] = {
        "schema_version": protocol.schema_version,
        "comparison_id": protocol.comparison_id,
        "common_case_sha256": common_case_hash,
        "protocol": protocol.as_mapping(),
        "input_manifest_path": "inputs/manifest.json",
        "designs": design_records,
        "pairwise_comparisons": pairwise,
        "observables": {
            "stiffness": protocol.stiffness_observable,
            "failure_force": protocol.failure_force_observable,
            "critical_location_coordinate_frame": DEFAULT_COORDINATE_FRAME,
        },
        "research_questions": {},
        "validation": validation,
    }
    simulation_report_path = root / "simulation_report.json"
    _write_strict_json(simulation_report_path, report)
    report["simulation_report_path"] = "simulation_report.json"
    report["simulation_report_sha256"] = _file_hash(simulation_report_path)
    # Persist the simulation result before attempting to bind optional external
    # evidence.  A bad evidence file therefore cannot destroy an expensive run.
    result = _apply_external_evidence(root, protocol, report, [], [])
    if experiments or ansys:
        result = _apply_external_evidence(
            root, protocol, result.report, experiments, ansys
        )
    return result


def _resolve_saved_path(root: Path, raw_path: Any, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{label} must be a nonempty saved path")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the comparison directory") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    return resolved


def _load_json_mapping(path: Path, *, label: str) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _verify_saved_comparison(
    root: Path,
    protocol: ComparisonProtocol,
    report: Mapping[str, Any],
) -> Dict[str, Any]:
    """Verify saved common inputs and topologies before accepting new evidence."""
    if report.get("comparison_id") != protocol.comparison_id:
        raise ValueError("saved comparison_id does not match protocol.json")
    if report.get("protocol") != _json_safe(protocol.as_mapping()):
        raise ValueError("saved comparison protocol does not match protocol.json")
    simulation_report_path = _resolve_saved_path(
        root,
        report.get("simulation_report_path"),
        label="immutable simulation report",
    )
    expected_simulation_hash = _validate_sha256(
        "simulation_report_sha256", report.get("simulation_report_sha256")
    )
    if _file_hash(simulation_report_path) != expected_simulation_hash:
        raise ValueError("immutable simulation report hash mismatch")
    simulation_report = _load_json_mapping(
        simulation_report_path, label="immutable simulation report"
    )
    for field in ("comparison_id", "common_case_sha256", "protocol"):
        if report.get(field) != simulation_report.get(field):
            raise ValueError(
                f"saved comparison {field} differs from immutable simulation report"
            )
    simulation_report["simulation_report_path"] = (
        simulation_report_path.relative_to(root).as_posix()
    )
    simulation_report["simulation_report_sha256"] = expected_simulation_hash
    report = simulation_report
    report_common_hash = _validate_sha256(
        "saved common_case_sha256", report.get("common_case_sha256")
    )
    manifest_path = _resolve_saved_path(
        root,
        report.get("input_manifest_path", "inputs/manifest.json"),
        label="input manifest",
    )
    manifest = _load_json_mapping(manifest_path, label="input manifest")
    manifest_arrays = manifest.get("arrays")
    if not isinstance(manifest_arrays, Mapping) or not manifest_arrays:
        raise ValueError("input manifest must describe saved arrays")
    verified_hashes: Dict[str, str] = {}
    for name, metadata in manifest_arrays.items():
        if not isinstance(metadata, Mapping):
            raise ValueError(f"input manifest entry {name!r} must be a mapping")
        path = _resolve_saved_path(
            root, metadata.get("path"), label=f"input array {name}"
        )
        if _file_hash(path) != _validate_sha256(
            f"input array {name} file_sha256", metadata.get("file_sha256")
        ):
            raise ValueError(f"input array {name} file hash mismatch")
        array = np.load(path, allow_pickle=False)
        array_hash = _array_hash(array)
        if array_hash != _validate_sha256(
            f"input array {name} array_sha256", metadata.get("array_sha256")
        ):
            raise ValueError(f"input array {name} content hash mismatch")
        if list(array.shape) != metadata.get("shape") or str(array.dtype) != metadata.get(
            "dtype"
        ):
            raise ValueError(f"input array {name} metadata mismatch")
        verified_hashes[str(name)] = array_hash
    strength = validate_material_strength(
        manifest.get("material_strength"),
        material_name=protocol.orthotropic_material_id,
    )
    recomputed_common_hash = _strict_common_case_hash(
        protocol, verified_hashes, strength
    )
    manifest_common_hash = _validate_sha256(
        "manifest common_case_sha256", manifest.get("common_case_sha256")
    )
    if not (
        recomputed_common_hash == manifest_common_hash == report_common_hash
    ):
        raise ValueError("saved common-case hash verification failed")

    designs = report.get("designs")
    if not isinstance(designs, Mapping) or set(designs) != set(DESIGN_IDS):
        raise ValueError("saved comparison must contain Designs A, B, and C")
    for design_id in DESIGN_IDS:
        record = designs[design_id]
        if not isinstance(record, Mapping):
            raise ValueError(f"saved Design {design_id} record must be a mapping")
        if record.get("common_case_sha256") != report_common_hash:
            raise ValueError(f"saved Design {design_id} common-case hash mismatch")
        for path_field, hash_field in (
            ("projected_density_path", "projected_density_sha256"),
            ("binary_density_path", "binary_topology_sha256"),
        ):
            saved_hash = record.get(hash_field)
            saved_path = record.get(path_field)
            if saved_hash is None and saved_path is None:
                continue
            path = _resolve_saved_path(
                root, saved_path, label=f"Design {design_id} {path_field}"
            )
            array = np.load(path, allow_pickle=False)
            if _array_hash(array) != _validate_sha256(
                f"Design {design_id} {hash_field}", saved_hash
            ):
                raise ValueError(f"saved Design {design_id} topology hash mismatch")
    return _json_safe(simulation_report)


def attach_external_validation(
    comparison_dir: str | Path,
    *,
    experimental_measurements: Optional[Sequence[Mapping[str, Any]]] = None,
    ansys_results: Optional[Sequence[Mapping[str, Any]]] = None,
) -> ResearchComparisonResult:
    """Attach ANSYS/specimen evidence to an existing comparison without rerunning.

    A source argument left as ``None`` preserves already attached records.  Pass
    an empty sequence explicitly to clear that source from the derived report.
    """
    root = Path(comparison_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"comparison directory does not exist: {root}")
    protocol = load_comparison_protocol(root / "protocol.json")
    report = _load_json_mapping(root / "comparison.json", label="comparison report")
    verified_simulation_report = _verify_saved_comparison(root, protocol, report)

    if experimental_measurements is None:
        experimental_path = root / "experimental_measurements.csv"
        experimental_measurements = (
            load_experimental_measurements_csv(experimental_path)
            if experimental_path.is_file()
            else []
        )
    if ansys_results is None:
        ansys_path = root / "ansys_results.json"
        ansys_results = (
            load_ansys_results_json(ansys_path) if ansys_path.is_file() else []
        )
    experiments = validate_experimental_measurements(
        protocol, experimental_measurements
    )
    ansys = validate_ansys_results(protocol, ansys_results)
    # All parsing, saved-artifact verification, and evidence binding occurs
    # before the first write inside _apply_external_evidence.
    return _apply_external_evidence(
        root, protocol, verified_simulation_report, experiments, ansys
    )
