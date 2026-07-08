"""
Metrics collection and management for topology optimization.

This module provides functions for collecting and reporting metrics from
topology optimization runs.
"""

from typing import Any, Dict, List, Optional

import numpy as np

def collect_metrics(
    terminal_input: str,
    nelx: int,
    nely: int,
    nelz: int,
    volfrac: float,
    penal: float,
    rmin: float,
    disp_thres: float,
    material_preset: str = None,
    material_orientation_xyz: str = None,
    force_field_preset: str = None,
    support_mask_preset: str = None,
    elem_size: float = 0.01, # 1 cm 
    tolx: float = 0.01,
    maxloop: int = 2000,
    design_space_stl: Optional[str] = None,
    pitch: float = 1.0,
    obstacle_config: Optional[str] = None,
    animation_fps: int = 5,
    stl_level: float = 0.5,
    stl_export_mode: str = "density",
    smooth_stl: bool = False,
    smooth_iterations: int = 3,
    xPhys: np.ndarray = None,
    design_space_mask: np.ndarray = None,
    obstacle_mask: np.ndarray = None,
    combined_obstacle_mask: np.ndarray = None,
    run_time: float = 0.0,
    final_compliance: Optional[float] = None,
    final_ux_avg_load_patch: Optional[float] = None,
    final_uy_avg_load_patch: Optional[float] = None,
    final_uz_avg_load_patch: Optional[float] = None,
    final_k_avg_x: Optional[float] = None,
    final_k_avg_y: Optional[float] = None,
    final_k_avg_z: Optional[float] = None,
    final_k_avg: Optional[float] = None,
    final_voxel_eval: Optional[List[Dict[str, Any]]] = None,
    final_binary_voxel_eval: Optional[List[Dict[str, Any]]] = None,
    gif_path: Optional[str] = None,
    stl_exported: bool = False,
    skip_optimization: bool = False,
) -> Dict[str, Any]:
    """
    Collect metrics about the optimization run.

    Args:
        terminal_input: full string input of terminal for reference
        nelx: Number of elements in x direction
        nely: Number of elements in y direction
        nelz: Number of elements in z direction
        volfrac: Volume fraction constraint
        penal: Penalization factor
        rmin: Filter radius
        disp_thres: Threshold for displaying elements
        tolx: Convergence tolerance
        maxloop: Maximum number of iterations
        design_space_stl: Path to STL file for design space
        pitch: Voxelization pitch for STL
        obstacle_config: Path to obstacle configuration file
        animation_fps: Frames per second for animation
        stl_level: Threshold level for STL export
        stl_export_mode: STL export mode (density, binary, blocky)
        smooth_stl: Whether to smooth the STL mesh
        smooth_iterations: Number of smoothing iterations
        xPhys: Optimized design
        design_space_mask: Design space mask
        obstacle_mask: Obstacle mask
        combined_obstacle_mask: Combined obstacle and design space mask
        run_time: Optimization runtime in seconds
        final_compliance: Final compliance/objective value from the optimizer
        final_ux_avg_load_patch: Mean displacement on loaded x-direction DOFs
        final_uy_avg_load_patch: Mean displacement on loaded y-direction DOFs
        final_uz_avg_load_patch: Mean displacement on loaded z-direction DOFs
        final_k_avg_x: Equivalent stiffness F_x / abs(final_ux_avg_load_patch)
        final_k_avg_y: Equivalent stiffness F_y / abs(final_uy_avg_load_patch)
        final_k_avg_z: Equivalent stiffness F_z / abs(final_uz_avg_load_patch)
        final_k_avg: Equivalent stiffness on dominant loading direction
        final_voxel_eval: Optional list of fixed-geometry evaluations under alternate materials
        final_binary_voxel_eval: Optional list of thresholded binary fixed-geometry
            evaluations under one or more materials
        gif_path: Path to animation GIF if created
        stl_exported: Whether STL export was successful

    Returns:
        Dictionary of metrics
    """
    # Create metrics dictionary
    metrics = {
        "terminal_input": terminal_input,
        "nelx": nelx,
        "nely": nely,
        "nelz": nelz,
        "volfrac": volfrac,
        "penal": penal,
        "rmin": rmin,
        "disp_thres": disp_thres,
        "material_preset": material_preset,
        "material_orientation_xyz": material_orientation_xyz,
        "force_field_preset": force_field_preset,
        "suport_mask_preset": support_mask_preset,
        "elem_size": elem_size,
        "tolx": tolx,
        "maxloop": maxloop,
        "runtime_seconds": run_time,
        "has_obstacles": obstacle_config is not None,
        "skip_optimization": skip_optimization,
    }

    if final_compliance is not None:
        metrics["final_compliance"] = final_compliance
    if final_ux_avg_load_patch is not None:
        metrics["ux_avg_load_patch"] = final_ux_avg_load_patch
    if final_uy_avg_load_patch is not None:
        metrics["uy_avg_load_patch"] = final_uy_avg_load_patch
    if final_uz_avg_load_patch is not None:
        metrics["uz_avg_load_patch"] = final_uz_avg_load_patch
    if final_k_avg_x is not None:
        metrics["k_avg_x"] = final_k_avg_x
    if final_k_avg_y is not None:
        metrics["k_avg_y"] = final_k_avg_y
    if final_k_avg_z is not None:
        metrics["k_avg_z"] = final_k_avg_z
    if final_k_avg is not None:
        metrics["k_avg"] = final_k_avg
    if final_voxel_eval:
        metrics["final_voxel_eval"] = final_voxel_eval
    if final_binary_voxel_eval:
        metrics["final_binary_voxel_eval"] = final_binary_voxel_eval

    # Add obstacle info to metrics
    if obstacle_config:
        metrics["obstacle_config"] = obstacle_config
        metrics["obstacle_elements"] = int(np.sum(obstacle_mask))

    # Add design space info to metrics
    if design_space_stl:
        metrics["design_space_stl"] = design_space_stl
        metrics["design_space_elements"] = int(np.sum(design_space_mask))
        metrics["combined_restricted_elements"] = int(np.sum(combined_obstacle_mask))
        metrics["pitch"] = pitch

    # Add animation metrics if available
    if gif_path:
        metrics["animation_created"] = True
        metrics["animation_path"] = gif_path
        metrics["animation_fps"] = animation_fps

    # Add STL export metrics
    if stl_exported:
        effective_smoothed = (
            stl_export_mode == "density" and smooth_stl and smooth_iterations > 0
        )
        metrics["stl_exported"] = True
        metrics["stl_level"] = stl_level
        metrics["stl_export_mode"] = stl_export_mode
        metrics["stl_smoothed"] = effective_smoothed
        metrics["stl_smooth_iterations"] = smooth_iterations

    return metrics


def calculate_compliance(xPhys, u, KE, penal) -> float:
    """
    Calculate compliance for the optimized structure.

    Args:
        xPhys: Physical density array
        u: Displacement vector
        KE: Element stiffness matrix
        penal: Penalization factor

    Returns:
        Compliance value
    """
    # This is a placeholder for the actual compliance calculation
    # In a real implementation, this would compute the compliance
    # based on the physical design and displacement
    return 0.0  # TODO: Replace with actual calculation


def summarize_optimization_results(metrics: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of optimization results.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Formatted string summarizing the optimization results
    """
    summary = [
        "Topology Optimization Results Summary",
        "======================================",
        f"Design space: {metrics['nelx']} x {metrics['nely']} x {metrics['nelz']} elements",
        f"Volume fraction: {metrics['volfrac']:.2f}",
        f"Filter radius: {metrics['rmin']:.2f}",
        f"Penalization: {metrics['penal']:.2f}",
        f"Threshold: {metrics['disp_thres']:.2f}",
        f"Runtime: {metrics['runtime_seconds']:.2f} seconds",
    ]

    if metrics.get("has_obstacles", False):
        summary.append(f"Obstacles: {metrics.get('obstacle_elements', 0)} elements")

    if "design_space_stl" in metrics:
        summary.append(
            f"Design space restricted by STL: {metrics.get('design_space_elements', 0)} elements"
        )
        summary.append(
            f"Total restricted elements: {metrics.get('combined_restricted_elements', 0)}"
        )

    if metrics.get("stl_exported", False):
        summary.append("STL export: Yes")

    if metrics.get("animation_created", False):
        summary.append("Animation: Yes")

    return "\n".join(summary)
