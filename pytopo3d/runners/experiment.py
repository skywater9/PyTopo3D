"""
Experiment setup and management for topology optimization.

This module contains functions for setting up and managing topology optimization experiments.
"""

import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

from pytopo3d.cli.parser import generate_experiment_name
from pytopo3d.core.optimizer import top3d
from pytopo3d.utils.export import voxel_to_stl
from pytopo3d.utils.logger import setup_logger
from pytopo3d.utils.results_manager import ResultsManager


def setup_experiment(args) -> Tuple[logging.Logger, ResultsManager]:
    """
    Set up experiment name, logging, and results manager.

    Args:
        args: Command-line arguments

    Returns:
        Tuple containing configured logger and results manager
    """
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

    return logger, results_mgr


def execute_optimization(
    args, logger, combined_obstacle_mask
) -> Tuple[np.ndarray, Optional[Dict], float]:
    """
    Run the topology optimization process.

    Args:
        args: Command-line arguments
        logger: Configured logger
        combined_obstacle_mask: Combined obstacle and design space mask

    Returns:
        Tuple containing optimization result, history (if saved), and runtime in seconds
    """
    # Run the optimization with timing
    logger.info(
        f"Starting optimization with {args.nelx}x{args.nely}x{args.nelz} elements..."
    )
    start_time = time.time()

    # Get tolx and maxloop from args if available
    tolx = getattr(args, "tolx", 0.01)  # Default to 0.01 if not provided
    maxloop = getattr(args, "maxloop", 2000)  # Default to 2000 if not provided

    # Check if creating animation is enabled
    save_history = getattr(args, "create_animation", False)
    history_frequency = getattr(args, "animation_frequency", 10)

    logger.debug(
        f"Optimization parameters: tolx={tolx}, maxloop={maxloop}, "
        f"save_history={save_history}, history_frequency={history_frequency}"
    )

    # Run the optimization with history if requested
    optimization_result = top3d(
        args.nelx,
        args.nely,
        args.nelz,
        args.volfrac,
        args.penal,
        args.rmin,
        args.disp_thres,
        obstacle_mask=combined_obstacle_mask,
        tolx=tolx,
        maxloop=maxloop,
        save_history=save_history,
        history_frequency=history_frequency,
    )

    # Check if we got history back
    history = None
    if save_history:
        xPhys, history = optimization_result
        logger.info(
            f"Optimization history captured with {len(history['density_history'])} frames"
        )
    else:
        xPhys = optimization_result

    end_time = time.time()
    run_time = end_time - start_time
    logger.debug(f"Optimization finished in {run_time:.2f} seconds")

    return xPhys, history, run_time


def export_result_to_stl(args, logger, results_mgr, result_path) -> bool:
    """
    Export the optimization result as an STL file if requested.

    Args:
        args: Command-line arguments
        logger: Configured logger
        results_mgr: Results manager instance
        result_path: Path to the saved optimization result

    Returns:
        True if STL export was successful, False otherwise
    """
    if not getattr(args, "export_stl", False):
        return False

    try:
        # Create the STL filename
        stl_filename = os.path.join(results_mgr.experiment_dir, "optimized_design.stl")

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
        return True

    except Exception as e:
        logger.error(f"Error exporting STL file: {e}")
        return False
