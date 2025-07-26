"""
Allow for the use of material presets when assembling stiffness matrix.

This module contains the function that maps material presets to parameters 
for stiffness matrix creation
"""

from pathlib import Path
import yaml

current_dir = Path(__file__).parent
material_presets_path = current_dir.parent.parent / "material_presets.yml"


with open(material_presets_path, "r") as f:
    materials = yaml.safe_load(f)

def get_material(material_name: str):
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
    material = materials.get(material_name)
    if material is None:
        raise ValueError(f"Material '{material_name}' not found.")

    material_properties = (
        material.get("E_x", None),
        material.get("E_y", None),
        material.get("E_z", None),
        material.get("nu_xy", None),
        material.get("nu_yz", None),
        material.get("nu_zx", None),
        material.get("G_xy", None),
        material.get("G_yz", None),
        material.get("G_zx", None),
        material.get("material_type", None),
    )

    return material_properties