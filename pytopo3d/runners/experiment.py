"""
Experiment setup and management for topology optimization.

This module contains functions for setting up and managing topology optimization experiments.
"""

import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

from pytopo3d.core.optimizer import top3d
from pytopo3d.utils.export import voxel_to_stl
from pytopo3d.utils.logger import setup_logger
from pytopo3d.utils.results_manager import ResultsManager


def setup_experiment(
    verbose: bool = False,
    quiet: bool = False,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    experiment_name: Optional[str] = None,
    description: Optional[str] = None,
    nelx: int = 40,
    nely: int = 20,
    nelz: int = 10,
    volfrac: float = 0.3,
    penal: float = 3.0,
    rmin: float = 1.5,
) -> Tuple[logging.Logger, ResultsManager]:
    """
    Set up experiment name, logging, and results manager.

    Args:
        verbose: Whether to enable verbose logging
        quiet: Whether to enable quiet mode
        log_level: Logging level
        log_file: Path to log file
        experiment_name: Name of the experiment (if None, will be generated)
        description: Description of the experiment
        nelx: Number of elements in x direction (for name generation)
        nely: Number of elements in y direction (for name generation)
        nelz: Number of elements in z direction (for name generation)
        volfrac: Volume fraction (for name generation)
        penal: Penalization factor (for name generation)
        rmin: Filter radius (for name generation)

    Returns:
        Tuple containing configured logger and results manager
    """
    # Configure logging from parameters
    if verbose:
        log_level_value = logging.DEBUG
    elif quiet:
        log_level_value = logging.WARNING
    else:
        log_level_value = getattr(logging, log_level)

    # Setup logger
    logger = setup_logger(level=log_level_value, log_file=log_file)
    logger.debug("Logging configured successfully")

    # Generate experiment name if not provided
    if experiment_name is None:
        import hashlib
        from datetime import datetime

        # Generate a name based on parameters and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create parameter string
        param_str = f"{nelx}x{nely}x{nelz}_vf{volfrac}_p{penal}_r{rmin}"

        # Generate a short hash of the parameters
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:6]

        experiment_name = f"topo3d_{timestamp}_{param_hash}"

    logger.info(f"Experiment name: {experiment_name}")

    # Create a results manager for this experiment
    results_mgr = ResultsManager(
        experiment_name=experiment_name, description=description
    )
    logger.debug(
        f"Results manager created with experiment directory: {results_mgr.experiment_dir}"
    )

    return logger, results_mgr


def execute_optimization(
    nelx: int,
    nely: int,
    nelz: int,
    volfrac: float,
    penal: float,
    rmin: float,
    disp_thres: float,
    material_preset: str = None,
    force_field: Optional[np.ndarray] = None,
    support_mask: Optional[np.ndarray] = None,
    tolx: float = 0.01,
    maxloop: int = 2000,
    create_animation: bool = False,
    animation_frequency: int = 10,
    logger: logging.Logger = None,
    combined_obstacle_mask: Optional[np.ndarray] = None,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, Optional[Dict], float]:
    """
    Run the topology optimization process.

    Args:
        nelx: Number of elements in x direction
        nely: Number of elements in y direction
        nelz: Number of elements in z direction
        volfrac: Volume fraction constraint
        penal: Penalization factor
        rmin: Filter radius
        disp_thres: Threshold for displaying elements
        force_field: Optional force field array (nely, nelx, nelz, 3)
        support_mask: Optional support mask array (nely, nelx, nelz)
        tolx: Convergence tolerance
        maxloop: Maximum number of iterations
        create_animation: Whether to save optimization history for animation
        animation_frequency: Frequency of saving frames for animation
        logger: Configured logger
        combined_obstacle_mask: Combined obstacle and design space mask
        use_gpu: Whether to use GPU acceleration if available

    Returns:
        Tuple containing optimization result, history (if saved), and runtime in seconds
    """
    # Run the optimization with timing
    if logger:
        logger.info(f"Starting optimization with {nelx}x{nely}x{nelz} elements...")
    start_time = time.time()

    # Determine ndof needed for logging/debug (though top3d calculates it internally)
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    logger.debug(
        f"Optimization parameters: tolx={tolx}, maxloop={maxloop}, "
        f"save_history={create_animation}, history_frequency={animation_frequency}, "
        f"ndof={ndof}, use_gpu={use_gpu}"
    )
    if force_field is not None:
        logger.debug(f"Using provided force_field with shape {force_field.shape}")
    else:
        logger.debug("Using default force settings")
    if support_mask is not None:
        logger.debug(f"Using provided support_mask with shape {support_mask.shape}")
    else:
        logger.debug("Using default support settings")

    # Run the optimization with history if requested
    optimization_result = top3d(
        nelx,
        nely,
        nelz,
        volfrac,
        penal,
        rmin,
        disp_thres,
        material_preset=material_preset,
        force_field=force_field,
        support_mask=support_mask,
        obstacle_mask=combined_obstacle_mask,
        tolx=tolx,
        maxloop=maxloop,
        save_history=create_animation,
        history_frequency=animation_frequency,
        use_gpu=use_gpu,
    )

    # Check if we got history back
    history = None
    if create_animation:
        xPhys, history = optimization_result
        if logger:
            logger.info(
                f"Optimization history captured with {len(history['density_history'])} frames"
            )
    else:
        xPhys = optimization_result

    end_time = time.time()
    run_time = end_time - start_time
    if logger:
        logger.debug(f"Optimization finished in {run_time:.2f} seconds")

    return xPhys, history, run_time


def export_result_to_stl(
    export_stl: bool = False,
    stl_level: float = 0.5,
    smooth_stl: bool = False,
    smooth_iterations: int = 3,
    logger: logging.Logger = None,
    results_mgr: ResultsManager = None,
    result_path: str = None,
) -> bool:
    """
    Export the optimization result as an STL file if requested.

    Args:
        export_stl: Whether to export as STL
        stl_level: Threshold level for STL export
        smooth_stl: Whether to smooth the STL mesh
        smooth_iterations: Number of smoothing iterations
        logger: Configured logger
        results_mgr: Results manager instance
        result_path: Path to the saved optimization result

    Returns:
        True if STL export was successful, False otherwise
    """
    if not export_stl:
        return False

    try:
        # Create the STL filename
        stl_filename = os.path.join(results_mgr.experiment_dir, "optimized_design.stl")

        # Export the result as an STL file
        if logger:
            logger.info("Exporting optimization result as STL file...")
        voxel_to_stl(
            input_file=result_path,
            output_file=stl_filename,
            level=stl_level,
            smooth_mesh=smooth_stl,
            smooth_iterations=smooth_iterations,
        )
        if logger:
            logger.info(f"STL file exported to {stl_filename}")
        return True

    except Exception as e:
        if logger:
            logger.error(f"Error exporting STL file: {e}")
        return False
