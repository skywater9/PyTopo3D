#!/usr/bin/env python3
"""
Main entry point for the 3D topology optimization package.

This script provides a command-line interface to run the topology optimization.
"""

import logging
import os
import time

import numpy as np

from pytopo3d.cli.parser import create_config_dict, generate_experiment_name, parse_args
from pytopo3d.core.optimizer import top3d
from pytopo3d.utils.boundary import create_boundary_arrays
from pytopo3d.utils.export import voxel_to_stl
from pytopo3d.utils.logger import setup_logger
from pytopo3d.utils.obstacles import parse_obstacle_config_file
from pytopo3d.utils.results_manager import ResultsManager
from pytopo3d.visualization.runner import create_visualization


def main():
    """
    Main function to run the optimization from command-line arguments.
    """
    # Parse command-line arguments
    args = parse_args()

    # Configure logging from command-line arguments
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = getattr(logging, args.log_level)

    # Setup logger
    logger = setup_logger(level=log_level, log_file=args.log_file)

    logger.debug("Command-line arguments parsed successfully")

    # Generate experiment name
    args.experiment_name = generate_experiment_name(args)
    logger.info(f"Experiment name: {args.experiment_name}")

    # Create a results manager for this experiment
    results_mgr = ResultsManager(
        experiment_name=args.experiment_name, description=args.description
    )
    logger.debug(
        f"Results manager created with experiment directory: {results_mgr.experiment_dir}"
    )

    # Create obstacle mask if requested
    obstacle_mask = None

    # Handle obstacle config file case
    if args.obstacle_config:
        try:
            shape = (args.nely, args.nelx, args.nelz)
            obstacle_mask = parse_obstacle_config_file(args.obstacle_config, shape)
            n_obstacle_elements = np.count_nonzero(obstacle_mask)
            logger.info(
                f"Loaded {n_obstacle_elements} obstacle elements from {args.obstacle_config}"
            )

            # Copy the obstacle config file to the experiment directory
            results_mgr.copy_file(args.obstacle_config, "obstacle_config.json")
            logger.debug("Copied obstacle config file to experiment directory")

        except Exception as e:
            logger.error(f"Error loading obstacle configuration: {e}")
            return
    else:
        logger.info(
            "No obstacle configuration provided, creating a default empty obstacle mask"
        )
        obstacle_mask = np.zeros((args.nely, args.nelx, args.nelz), dtype=bool)

    # Create obstacle array for visualization
    obstacle_array = obstacle_mask.astype(float)

    # Create boundary condition arrays for visualization
    logger.debug("Creating boundary condition arrays")
    loads_array, constraints_array = create_boundary_arrays(
        args.nelx, args.nely, args.nelz
    )

    # Save configuration
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

    # Run the optimization with timing
    logger.info(
        f"Starting optimization with {args.nelx}x{args.nely}x{args.nelz} elements..."
    )
    start_time = time.time()

    # Get tolx and maxloop from args if available
    tolx = getattr(args, "tolx", 0.01)  # Default to 0.01 if not provided
    maxloop = getattr(args, "maxloop", 2000)  # Default to 2000 if not provided

    logger.debug(f"Optimization parameters: tolx={tolx}, maxloop={maxloop}")

    xPhys = top3d(
        args.nelx,
        args.nely,
        args.nelz,
        args.volfrac,
        args.penal,
        args.rmin,
        args.disp_thres,
        obstacle_mask=obstacle_mask,
        tolx=tolx,
        maxloop=maxloop,
    )

    end_time = time.time()
    run_time = end_time - start_time
    logger.debug(f"Optimization finished in {run_time:.2f} seconds")

    # Save the result to the experiment directory
    result_path = results_mgr.save_result(xPhys, "optimized_design.npy")
    logger.debug(f"Optimization result saved to {result_path}")

    # Create design_only array (optimized design without obstacles)
    design_only = xPhys.copy()
    design_only[obstacle_mask] = 0  # Remove design elements where obstacles are

    # Create combined visualization with optimized design, loads, constraints, and obstacles
    logger.debug("Creating combined visualization")
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

    # Create metrics dictionary
    metrics = {
        "nelx": args.nelx,
        "nely": args.nely,
        "nelz": args.nelz,
        "volfrac": args.volfrac,
        "penal": args.penal,
        "rmin": args.rmin,
        "disp_thres": args.disp_thres,
        "tolx": tolx,
        "maxloop": maxloop,
        "runtime_seconds": run_time,
        "has_obstacles": args.obstacle_config is not None,
    }

    # Add obstacle info to metrics
    if args.obstacle_config:
        metrics["obstacle_config"] = args.obstacle_config
        metrics["obstacle_elements"] = int(np.sum(obstacle_mask))

    # Export the result as an STL file if requested
    if getattr(args, "export_stl", False):
        try:
            # Create the STL filename
            stl_filename = os.path.join(
                results_mgr.experiment_dir, "optimized_design.stl"
            )

            # Export the result as an STL file
            logger.info("Exporting optimization result as STL file...")
            voxel_to_stl(
                input_file=result_path,
                output_file=stl_filename,
                level=args.stl_level,
                smooth_mesh=args.smooth_stl,
                smooth_iterations=args.smooth_iterations,
            )
            logger.info(f"STL file exported to {stl_filename}")

            # Add STL export info to metrics
            metrics["stl_exported"] = True
            metrics["stl_level"] = args.stl_level
            metrics["stl_smoothed"] = args.smooth_stl
            metrics["stl_smooth_iterations"] = args.smooth_iterations
        except Exception as e:
            logger.error(f"Error exporting STL file: {e}")
            metrics["stl_export_error"] = str(e)

    results_mgr.update_metrics(metrics)
    logger.debug("Metrics updated")

    logger.info(f"Optimization complete in {run_time:.2f} seconds.")
    logger.info(f"Result saved to {result_path}")
    logger.info(f"All experiment files are in {results_mgr.experiment_dir}")

    # # Create/update the experiments database
    # try:
    #     db = ResultsManager.create_experiments_database()
    #     logger.info("Experiment database updated: experiments.csv")
    # except Exception as e:
    #     logger.warning(f"Could not update experiments database: {e}")


if __name__ == "__main__":
    main()
