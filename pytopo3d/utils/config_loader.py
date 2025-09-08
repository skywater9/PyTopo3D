"""
Allow for the use of custom presets in the config folder

This module contains functions that use user made presets to build parameters
"""

from pathlib import Path
import yaml

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
        Tuple containing the material's mechanical properties
    """
    material = materials.get(material_name.lower())
    if material is None:
        raise ValueError(f"Material '{material_name}' not found.")

    material_properties = (
        to_float(material.get("sigma_x_yield", None)),
        to_float(material.get("sigma_y_yield", None)),
        to_float(material.get("sigma_z_yield", None)),
        to_float(material.get("tau_xy_yield", None)),
        to_float(material.get("tau_yz_yield", None)),
        to_float(material.get("tau_zx_yield", None)),
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

current_dir = Path(__file__).parent
odr_path = current_dir.parent.parent / "config" / "output_displacement_ranges.yml"
with open(odr_path, "r") as f:
    output_displacement_ranges = yaml.safe_load(f)

def get_output_displacement_range(output_displacement_range_name: str):

    output_displacement_range = output_displacement_ranges.get(output_displacement_range_name.lower())
    if output_displacement_range is None:
        raise ValueError(f"Output displacement range '{output_displacement_range_name}' not found.")

    odr_vals = (
        output_displacement_range.get("x1"),
        output_displacement_range.get("x2"),
        output_displacement_range.get("y1"),
        output_displacement_range.get("y2"),
        output_displacement_range.get("z1"),
        output_displacement_range.get("z2"),
    )

    return odr_vals