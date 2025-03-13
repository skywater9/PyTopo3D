from typing import List, Optional

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure


def voxel_to_stl(
    input_file: str,
    output_file: str,
    level: float = 0.5,
    padding: int = 1,
    fix_mesh: bool = True,
    smooth_mesh: bool = True,
    smooth_iterations: int = 5,
    upscale_factor: Optional[int] = None,
) -> None:
    """
    Convert a voxel representation (.npy file) to an STL mesh file.

    Parameters:
    -----------
    input_file : str
        Path to the input .npy file containing voxel data
    output_file : str
        Path where the output STL file will be saved
    level : float, optional
        Contour level for the marching cubes algorithm (default: 0.5)
    padding : int, optional
        Number of zero-voxel layers to pad around the volume (default: 1)
    fix_mesh : bool, optional
        Whether to apply mesh repair operations (default: True)
    smooth_mesh : bool, optional
        Whether to apply Laplacian smoothing to the final mesh (default: True)
    smooth_iterations : int, optional
        Number of iterations for mesh smoothing (default: 5)
    upscale_factor : Optional[int], optional
        Factor to upscale voxel resolution before meshing (default: None)

    Returns:
    --------
    None
    """
    # 1. Load the voxel data from the .npy file
    voxel_data: np.ndarray = np.load(input_file)

    # 2. Pad the voxel data with zeros to ensure a closed mesh
    if padding > 0:
        padded_data: np.ndarray = np.pad(
            voxel_data, padding, mode="constant", constant_values=0
        )
    else:
        padded_data = voxel_data

    # 3. Upscale the voxel data if requested
    if upscale_factor and upscale_factor > 1:
        # Get the original shape
        original_shape: np.ndarray = np.array(padded_data.shape)
        # Calculate the new shape
        new_shape: np.ndarray = original_shape * upscale_factor
        # Create coordinates for the original data
        orig_coords: List[np.ndarray] = [np.arange(s) for s in original_shape]
        # Create coordinates for the upscaled data
        new_coords: List[np.ndarray] = [
            np.linspace(0, s - 1, ns) for s, ns in zip(original_shape, new_shape)
        ]
        # Use scipy's map_coordinates for smooth interpolation
        grid: List[np.ndarray] = np.meshgrid(*new_coords, indexing="ij")
        upscaled_data: np.ndarray = ndimage.map_coordinates(
            padded_data, grid, order=3, mode="nearest"
        )
        padded_data = upscaled_data

    # 4. Generate a triangulated mesh using marching cubes algorithm
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    vertices, faces, normals, _ = measure.marching_cubes(padded_data, level=level)

    # 5. Create a mesh object
    mesh: trimesh.Trimesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, normals=normals
    )

    # 6. Apply Laplacian smoothing to the mesh if requested
    if smooth_mesh and smooth_iterations > 0:
        # Check if the mesh has any vertices to smooth
        if len(mesh.vertices) > 0:
            # Laplacian smoothing while preserving volume
            for _ in range(smooth_iterations):
                trimesh.smoothing.filter_laplacian(mesh, iterations=1, lamb=0.5)

    # 7. Fix mesh if requested (fill holes, remove duplicate vertices, etc.)
    if fix_mesh:
        # Make sure mesh is watertight
        if not mesh.is_watertight:
            print("Attempting to fix non-watertight mesh...")
            # Fill holes
            mesh.fill_holes()
            # Remove duplicate vertices
            mesh = mesh.process(validate=True)

    # 8. If we padded the data, adjust the vertices to compensate
    if padding > 0:
        # Calculate the scale factor if we upscaled
        scale_factor: float = 1.0
        if upscale_factor and upscale_factor > 1:
            scale_factor = 1.0 / upscale_factor

        # Shift the vertices back by the padding amount, adjusted for any scaling
        mesh.vertices = (mesh.vertices * scale_factor) - (padding * scale_factor)

    # 9. Save the mesh as an STL file
    mesh.export(output_file)

    if mesh.is_watertight:
        print(f"Conversion complete! Watertight mesh saved as '{output_file}'")
    else:
        print(
            f"Conversion complete! Mesh saved as '{output_file}' (note: mesh may not be completely watertight)"
        )


# Example usage
if __name__ == "__main__":
    voxel_to_stl(
        input_file="results/pytopo3d_60x30x20_obstacles_config_cylinder/optimized_design.npy",
        output_file="results/pytopo3d_60x30x20_obstacles_config_cylinder/output_mesh.stl",
        padding=1,
        smooth_mesh=True,
        smooth_iterations=5,
    )
