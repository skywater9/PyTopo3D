"""
Visualization utilities for 3D topology optimization.

This module provides functions for creating visualizations of topology optimization
results, boundary conditions, and creating animations.
"""

import os
from typing import Optional, Tuple

import numpy as np

from pytopo3d.visualization.animation import save_optimization_gif
from pytopo3d.visualization.runner import create_visualization


def visualize_initial_setup(
    args, logger, results_mgr, combined_obstacle_mask
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Create and save initial visualization showing boundary conditions and obstacles.

    Args:
        args: Command-line arguments
        logger: Configured logger
        results_mgr: Results manager instance
        combined_obstacle_mask: Combined obstacle and design space mask

    Returns:
        Tuple containing loads array, constraints array, and path to visualization
    """
    # Create obstacle array for visualization
    obstacle_array = combined_obstacle_mask.astype(float)

    # Create boundary condition arrays for visualization
    logger.debug("Creating boundary condition arrays")
    from pytopo3d.utils.boundary import create_boundary_arrays

    loads_array, constraints_array = create_boundary_arrays(
        args.nelx, args.nely, args.nelz
    )

    # Save configuration
    from pytopo3d.cli.parser import create_config_dict

    config = create_config_dict(args)
    results_mgr.save_config(config)
    logger.debug("Configuration saved")

    # Create visualization for boundary conditions, loads, constraints, and obstacle geometry
    logger.debug("Creating boundary condition visualization")
    arrays_to_visualize = [obstacle_array, loads_array, constraints_array]
    thresholds = [0.5, 0.5, 0.5]
    colors = ["yellow", "blue", "red"]
    labels = ["Obstacles", "Loads", "Constraints"]
    alphas = [0.3, 0.9, 0.9]  # Make obstacles transparent

    boundary_viz_path = create_visualization(
        arrays=arrays_to_visualize,
        thresholds=thresholds,
        colors=colors,
        labels=labels,
        alphas=alphas,  # Add transparency
        experiment_name=args.experiment_name,
        results_mgr=results_mgr,
        filename="boundary_conditions_and_obstacles",
        title="Boundary Conditions and Obstacles",
    )
    logger.info(f"Boundary conditions visualization saved to {boundary_viz_path}")

    return loads_array, constraints_array, boundary_viz_path


def visualize_final_result(
    args,
    logger,
    results_mgr,
    xPhys,
    combined_obstacle_mask,
    loads_array,
    constraints_array,
) -> str:
    """
    Create and save visualization of the final optimization result.

    Args:
        args: Command-line arguments
        logger: Configured logger
        results_mgr: Results manager instance
        xPhys: Optimized design
        combined_obstacle_mask: Combined obstacle and design space mask
        loads_array: Array showing load positions
        constraints_array: Array showing constraint positions

    Returns:
        Path to the saved visualization
    """
    # Create design_only array (optimized design without obstacles)
    design_only = xPhys.copy()
    design_only[combined_obstacle_mask] = (
        0  # Remove design elements where obstacles are
    )

    # Create combined visualization with optimized design, loads, constraints, and obstacles
    logger.debug("Creating combined visualization")
    obstacle_array = combined_obstacle_mask.astype(float)
    combined_arrays = [design_only, obstacle_array, loads_array, constraints_array]
    combined_thresholds = [args.disp_thres, 0.5, 0.5, 0.5]
    combined_colors = ["gray", "yellow", "blue", "red"]
    combined_labels = ["Optimized Design", "Obstacles", "Loads", "Constraints"]
    combined_alphas = [0.9, 0.3, 0.9, 0.9]  # Make obstacles transparent

    combined_viz_path = create_visualization(
        arrays=combined_arrays,
        thresholds=combined_thresholds,
        colors=combined_colors,
        labels=combined_labels,
        alphas=combined_alphas,  # Add transparency
        experiment_name=args.experiment_name,
        results_mgr=results_mgr,
        filename="optimized_design_with_boundary_conditions",
        title="Optimized Design with Boundary Conditions",
    )
    logger.info(f"Combined visualization saved to {combined_viz_path}")
    return combined_viz_path


def create_optimization_animation(
    args,
    logger,
    results_mgr,
    history,
    combined_obstacle_mask,
    loads_array,
    constraints_array,
) -> Optional[str]:
    """
    Create an animation of the optimization process if history was captured.

    Args:
        args: Command-line arguments
        logger: Configured logger
        results_mgr: Results manager instance
        history: Optimization history data
        combined_obstacle_mask: Combined obstacle and design space mask
        loads_array: Array showing load positions
        constraints_array: Array showing constraint positions

    Returns:
        Path to the generated GIF file, or None if animation failed
    """
    if not history:
        return None

    try:
        logger.info("Creating GIF visualization of optimization process...")

        # Default to a reasonable number of frames for the animation
        frames_to_include = getattr(args, "animation_frames", 50)

        # If there are more frames than we want to include, sample them
        history_frames = history["density_history"]
        history_iterations = history["iteration_history"]
        history_compliances = history["compliance_history"]

        logger.debug(
            f"Animation data: {len(history_frames)} density frames, "
            f"{len(history_iterations)} iterations, "
            f"{len(history_compliances)} compliance values"
        )

        if len(history_frames) > frames_to_include:
            # Calculate sampling frequency
            sample_rate = max(1, len(history_frames) // frames_to_include)
            logger.debug(f"Sampling animation frames (every {sample_rate} frames)")
        else:
            sample_rate = 1

        # Create the animation
        try:
            gif_path = save_optimization_gif(
                frames=history_frames,
                obstacle_mask=combined_obstacle_mask,
                loads_array=loads_array,
                constraints_array=constraints_array,
                compliances=history_compliances,
                disp_thres=args.disp_thres,
                results_mgr=results_mgr,
                filename="optimization_animation",
                fps=getattr(args, "animation_fps", 5),
                every_n_iterations=sample_rate,
            )

            if os.path.exists(gif_path) and os.path.getsize(gif_path) > 0:
                logger.info(f"Optimization animation saved to {gif_path}")
                return gif_path
            else:
                logger.error(
                    f"Animation GIF file was not created properly or is empty: {gif_path}"
                )
                return None

        except Exception as e:
            logger.error(f"Error in save_optimization_gif function: {e}")
            import traceback

            logger.debug(f"Animation creation error details: {traceback.format_exc()}")
            return None

    except Exception as e:
        logger.error(f"Error creating optimization animation: {e}")
        import traceback

        logger.debug(f"Animation error details: {traceback.format_exc()}")
        return None
