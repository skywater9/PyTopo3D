"""
Metrics collection and management for topology optimization.

This module provides functions for collecting and reporting metrics from
topology optimization runs.
"""

from typing import Any, Dict, Optional

import numpy as np


def collect_metrics(
    args,
    xPhys,
    design_space_mask,
    obstacle_mask,
    combined_obstacle_mask,
    run_time,
    gif_path,
    stl_exported,
) -> Dict[str, Any]:
    """
    Collect metrics about the optimization run.

    Args:
        args: Command-line arguments
        xPhys: Optimized design
        design_space_mask: Design space mask
        obstacle_mask: Obstacle mask
        combined_obstacle_mask: Combined obstacle and design space mask
        run_time: Optimization runtime in seconds
        gif_path: Path to animation GIF if created
        stl_exported: Whether STL export was successful

    Returns:
        Dictionary of metrics
    """
    # Create metrics dictionary
    metrics = {
        "nelx": args.nelx,
        "nely": args.nely,
        "nelz": args.nelz,
        "volfrac": args.volfrac,
        "penal": args.penal,
        "rmin": args.rmin,
        "disp_thres": args.disp_thres,
        "tolx": getattr(args, "tolx", 0.01),
        "maxloop": getattr(args, "maxloop", 2000),
        "runtime_seconds": run_time,
        "has_obstacles": args.obstacle_config is not None,
    }

    # Add obstacle info to metrics
    if args.obstacle_config:
        metrics["obstacle_config"] = args.obstacle_config
        metrics["obstacle_elements"] = int(np.sum(obstacle_mask))

    # Add design space info to metrics
    if hasattr(args, "design_space_stl") and args.design_space_stl:
        metrics["design_space_stl"] = args.design_space_stl
        metrics["design_space_elements"] = int(np.sum(design_space_mask))
        metrics["combined_restricted_elements"] = int(np.sum(combined_obstacle_mask))
        metrics["pitch"] = args.pitch

    # Add animation metrics if available
    if gif_path:
        metrics["animation_created"] = True
        metrics["animation_path"] = gif_path
        metrics["animation_fps"] = getattr(args, "animation_fps", 5)

    # Add STL export metrics
    if stl_exported:
        metrics["stl_exported"] = True
        metrics["stl_level"] = args.stl_level
        metrics["stl_smoothed"] = args.smooth_stl
        metrics["stl_smooth_iterations"] = args.smooth_iterations

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
    return 0.0  # Replace with actual calculation


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
