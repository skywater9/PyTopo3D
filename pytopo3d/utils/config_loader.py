"""
Allow for the use of custom presets in the config folder

This module contains functions that use user made presets to build parameters
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

current_dir = Path(__file__).parent
protected_zones_path = current_dir.parent.parent / "config" / "protected_zone_ranges.yml"
with open(protected_zones_path, "r") as f:
    protected_zones = yaml.safe_load(f)

def get_protected_zone_ranges(zone_names: tuple):
    """
    Pass the protected zone ranges for multiple presets

    Parameter
    ----------
    zone_names
        Tuple of protected zone preset names

    Returns
    -------
    protected_zone_ranges
        Tuple of tuples, each containing the range for a protected zone
    """
    ranges = []
    for name in zone_names:
        zone = protected_zones.get(name.lower())
        if zone is None:
            raise ValueError(f"Protected zone '{name}' not found.")
        zone_range = (
            zone.get("x1"),
            zone.get("x2"),
            zone.get("y1"),
            zone.get("y2"),
            zone.get("z1"),
            zone.get("z2"),
        )
        ranges.append(zone_range)
    return tuple(ranges)

current_dir = Path(__file__).parent
material_presets_path = current_dir.parent.parent / "config" / "material_presets.yml"
with open(material_presets_path, "r") as f:
    materials = yaml.safe_load(f)

def to_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return val
        

def get_material_params(material_name: str):
    """
    Pass the material properties of preset

    Parameter
    ----------
    material_name
        Name of the target material preset

    Returns
    -------
    material_properties
        Tuple containing the material's stiffness and Poisson ratio parameters.
    """
    material = materials.get(material_name.lower())
    if material is None:
        raise ValueError(f"Material '{material_name}' not found.")

    material_properties = (
        to_float(material.get("E_x", None)),
        to_float(material.get("E_y", None)),
        to_float(material.get("E_z", None)),
        to_float(material.get("G_xy", None)),
        to_float(material.get("G_yz", None)),
        to_float(material.get("G_zx", None)),
        to_float(material.get("nu_xy", None)),
        to_float(material.get("nu_yz", None)),
        to_float(material.get("nu_zx", None)),
    )

    return material_properties


@dataclass(frozen=True)
class MaterialStrength:
    """Orthotropic maximum-stress allowables in material axes, in pascals."""

    X_t: float
    X_c: float
    Y_t: float
    Y_c: float
    Z_t: float
    Z_c: float
    S_xy: float
    S_yz: float
    S_zx: float
    criterion: str = "maximum_stress"
    units: str = "Pa"

    def as_dict(self) -> dict:
        """Return a JSON/YAML-friendly snapshot of the validated allowables."""
        return {
            "X_t": self.X_t,
            "X_c": self.X_c,
            "Y_t": self.Y_t,
            "Y_c": self.Y_c,
            "Z_t": self.Z_t,
            "Z_c": self.Z_c,
            "S_xy": self.S_xy,
            "S_yz": self.S_yz,
            "S_zx": self.S_zx,
            "criterion": self.criterion,
            "units": self.units,
        }


_STRENGTH_FIELDS = (
    "X_t",
    "X_c",
    "Y_t",
    "Y_c",
    "Z_t",
    "Z_c",
    "S_xy",
    "S_yz",
    "S_zx",
)


def validate_material_strength(
    strength: Mapping[str, object],
    *,
    material_name: str = "material",
) -> MaterialStrength:
    """Validate and normalize a material strength mapping.

    Every allowable must be finite, strictly positive, and expressed in Pa.
    Rejecting zeros is intentional: zero-valued schema placeholders must never
    silently turn into divide-by-zero or infinite failure indices.
    """
    if not isinstance(strength, Mapping):
        raise ValueError(
            f"Material '{material_name}' must define strength as a mapping."
        )

    missing = [field for field in _STRENGTH_FIELDS if field not in strength]
    if missing:
        raise ValueError(
            f"Material '{material_name}' strength is missing required fields: "
            + ", ".join(missing)
        )

    values = {}
    for field in _STRENGTH_FIELDS:
        raw_value = strength[field]
        if isinstance(raw_value, bool):
            raise ValueError(
                f"Material '{material_name}' strength {field} must be a number in Pa."
            )
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Material '{material_name}' strength {field} must be a number in Pa."
            ) from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Material '{material_name}' strength {field} must be finite and > 0 Pa; "
                f"got {raw_value!r}."
            )
        values[field] = value

    criterion = str(strength.get("criterion", "")).strip().lower()
    if criterion != "maximum_stress":
        raise ValueError(
            f"Material '{material_name}' has unsupported strength criterion "
            f"{criterion!r}; expected 'maximum_stress'."
        )

    units = str(strength.get("units", "Pa")).strip()
    if units.lower() != "pa":
        raise ValueError(
            f"Material '{material_name}' strength units must be Pa; got {units!r}."
        )

    return MaterialStrength(**values, criterion=criterion, units="Pa")


def get_material_strength(material_name: str) -> MaterialStrength:
    """Load validated material-axis strength allowables for a preset."""
    material = materials.get(material_name.lower())
    if material is None:
        raise ValueError(f"Material '{material_name}' not found.")
    if "strength" not in material:
        raise ValueError(
            f"Material '{material_name}' has no strength data. Add measured Pa-valued "
            "allowables before enabling failure evaluation."
        )
    return validate_material_strength(
        material["strength"],
        material_name=material_name,
    )


def material_has_strength(material_name: str) -> bool:
    """Return whether a known preset declares strength data.

    Declaration and validity are intentionally separate: callers should then
    use :func:`get_material_strength`, which rejects incomplete or invalid data.
    """
    material = materials.get(material_name.lower())
    if material is None:
        raise ValueError(f"Material '{material_name}' not found.")
    return "strength" in material


def parse_material_orientation_xyz(material_orientation_xyz: Optional[str]) -> Optional[str]:
    """
    Validate and normalize a material orientation mapping string.

    Parameters
    ----------
    material_orientation_xyz
        String containing exactly one each of x, y, z (axis permutation), or None.

    Returns
    -------
    Optional[str]
        Normalized lowercase orientation string, or None if no value was provided.
    """
    if material_orientation_xyz is None:
        return None

    orientation = material_orientation_xyz.strip().lower()
    if len(orientation) != 3 or any(axis not in "xyz" for axis in orientation):
        raise ValueError(
            "material_orientation_xyz must be exactly 3 letters using only x, y, z "
            "(for example: xyz, zxy)."
        )
    if sorted(orientation) != ["x", "y", "z"]:
        raise ValueError(
            "material_orientation_xyz must be a permutation of xyz with no repeated letters "
            "(allowed: xyz, xzy, yxz, yzx, zxy, zyx)."
        )

    return orientation


def material_orientation_matrix(
    material_orientation_xyz: Optional[str],
) -> np.ndarray:
    """Return the material-to-global orthogonal axis mapping matrix.

    Column ``m`` contains material basis vector ``m`` in global coordinates,
    so ``v_global = R @ v_material``. The permutation strings intentionally
    preserve the existing axis-label convention and may have determinant -1;
    maximum-stress failure is insensitive to the resulting shear signs.
    """
    orientation = parse_material_orientation_xyz(material_orientation_xyz) or "xyz"
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    rotation = np.zeros((3, 3), dtype=float)
    for material_axis, global_axis in enumerate(orientation):
        rotation[axis_to_index[global_axis], material_axis] = 1.0
    return rotation


def apply_material_orientation(
    material_params: Sequence[Optional[float]],
    material_orientation_xyz: Optional[str],
) -> Tuple[Optional[float], ...]:
    """
    Remap material properties onto global axes.

    The mapping string is interpreted as:
    - char 0: global axis receiving material x-direction values
    - char 1: global axis receiving material y-direction values
    - char 2: global axis receiving material z-direction values

    For shear/Poisson directional pairs, axes are grouped as:
    xy -> x, yz -> y, zx -> z.
    """
    orientation = parse_material_orientation_xyz(material_orientation_xyz)
    if orientation is None:
        return tuple(material_params)

    if len(material_params) != 9:
        raise ValueError("material_params must contain 9 values in preset order.")

    params = tuple(material_params)
    if any(value is None for value in params):
        raise ValueError(
            "Oriented material presets must provide all 9 orthotropic stiffness values."
        )

    E_x, E_y, E_z, G_xy, G_yz, G_zx, nu_xy, nu_yz, nu_zx = (
        float(value) for value in params
    )

    # Build the normal compliance block using the exact convention consumed by
    # make_C_matrix, then permute it. This correctly handles reciprocal Poisson
    # effects for both cyclic and reversed axis mappings.
    compliance_normal_material = np.array(
        [
            [1.0 / E_x, -nu_xy / E_x, -nu_zx / E_z],
            [-nu_xy / E_x, 1.0 / E_y, -nu_yz / E_y],
            [-nu_zx / E_z, -nu_yz / E_y, 1.0 / E_z],
        ],
        dtype=float,
    )
    rotation = material_orientation_matrix(orientation)
    compliance_normal_global = rotation @ compliance_normal_material @ rotation.T

    E_global = 1.0 / np.diag(compliance_normal_global)
    nu_global = (
        -compliance_normal_global[0, 1] * E_global[0],
        -compliance_normal_global[1, 2] * E_global[1],
        -compliance_normal_global[2, 0] * E_global[2],
    )

    shear_by_material_plane = {
        frozenset((0, 1)): G_xy,
        frozenset((1, 2)): G_yz,
        frozenset((2, 0)): G_zx,
    }
    global_to_material = np.argmax(rotation, axis=1)
    shear_global = tuple(
        shear_by_material_plane[
            frozenset((global_to_material[a], global_to_material[b]))
        ]
        for a, b in ((0, 1), (1, 2), (2, 0))
    )

    return tuple(E_global) + shear_global + nu_global


current_dir = Path(__file__).parent
force_fields_path = current_dir.parent.parent / "config" / "force_fields.yml"
with open(force_fields_path, "r") as f:
    force_fields = yaml.safe_load(f)

def get_force_field_params(force_field_name: str):
    """
    Pass the force field preset

    Parameter
    ----------
    force_field_name
        Name of the target force_field preset

    Returns
    -------
    force_field_params
        Tuple containing the range and force vector of the force field
    """
    force_field = force_fields.get(force_field_name.lower())
    if force_field is None:
        raise ValueError(f"Force field '{force_field_name}' not found.")

    force_field_params = (
        force_field.get("x1"),
        force_field.get("x2"),
        force_field.get("y1"),
        force_field.get("y2"),
        force_field.get("z1"),
        force_field.get("z2"),
        force_field.get("F_x"),
        force_field.get("F_y"),
        force_field.get("F_z")
    )

    return force_field_params


current_dir = Path(__file__).parent
support_masks_path = current_dir.parent.parent / "config" / "support_masks.yml"
with open(support_masks_path, "r") as f:
    support_masks = yaml.safe_load(f)

def get_support_mask_params(support_mask_name: str):
    """
    Pass the support mask preset

    Parameter
    ----------
    support_mask_name
        Name of the target support_mask preset

    Returns
    -------
    support_mask_params
        Tuple containing the range of the support mask
    """
    support_mask = support_masks.get(support_mask_name.lower())
    if support_mask is None:
        raise ValueError(f"Support mask '{support_mask_name}' not found.")

    support_mask_params = (
        support_mask.get("x1"),
        support_mask.get("x2"),
        support_mask.get("y1"),
        support_mask.get("y2"),
        support_mask.get("z1"),
        support_mask.get("z2"),
    )

    return support_mask_params
