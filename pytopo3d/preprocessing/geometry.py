"""
Geometry processing utilities for 3D topology optimization.

This module provides functions for loading and processing geometry data from STL files
and creating boundary conditions.
"""

import os
from typing import Tuple

import numpy as np

from pytopo3d.utils.boundary import create_boundary_arrays
from pytopo3d.utils.import_design_space import stl_to_design_space
from pytopo3d.utils.obstacles import parse_obstacle_config_file


def load_geometry_data(
    args, logger, results_mgr
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load design space and obstacle data.

    Args:
        args: Command-line arguments
        logger: Configured logger
        results_mgr: Results manager instance

    Returns:
        Tuple containing design space mask, obstacle mask, and combined obstacle mask
    """
    # Handle design space from STL if provided
    design_space_mask = None
    if hasattr(args, "design_space_stl") and args.design_space_stl:
        try:
            logger.info(f"Loading design space from STL file: {args.design_space_stl}")

            # Use the voxelization pitch
            pitch = args.pitch
            logger.info(f"Using voxelization pitch: {pitch}")

            # Invert flag
            invert = getattr(args, "invert_design_space", False)
            if invert:
                logger.info("Design space will be inverted (STL represents void space)")

            # Generate design space from STL
            # Resolution is determined by the mesh and pitch
            design_space_mask = stl_to_design_space(
                args.design_space_stl, pitch=pitch, invert=invert
            )

            # Update nelx, nely, nelz based on the voxelized shape
            args.nely, args.nelx, args.nelz = design_space_mask.shape
            logger.info(
                f"Resolution from voxelization: {args.nely}x{args.nelx}x{args.nelz}"
            )

            # Save design space mask
            design_space_path = os.path.join(
                results_mgr.experiment_dir, "design_space_mask.npy"
            )
            np.save(design_space_path, design_space_mask)
            logger.info(f"Design space mask saved to {design_space_path}")

            # Copy the STL file to the experiment directory
            results_mgr.copy_file(args.design_space_stl, "design_space.stl")
            logger.debug("Copied design space STL file to experiment directory")

        except Exception as e:
            logger.error(f"Error loading design space from STL: {e}")
            import traceback

            logger.debug(f"STL loading error details: {traceback.format_exc()}")
            raise
    else:
        logger.debug("No STL design space provided, using full rectangular domain")
        design_space_mask = np.ones((args.nely, args.nelx, args.nelz), dtype=bool)

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
            raise
    else:
        logger.info(
            "No obstacle configuration provided, creating a default empty obstacle mask"
        )
        obstacle_mask = np.zeros((args.nely, args.nelx, args.nelz), dtype=bool)

    # Combine design space and obstacle masks
    # Elements outside the design space are treated as obstacles
    combined_obstacle_mask = obstacle_mask.copy()
    if design_space_mask is not None:
        # Areas outside design space (False values) become obstacles (True in obstacle mask)
        combined_obstacle_mask = np.logical_or(
            combined_obstacle_mask, ~design_space_mask
        )
        logger.info(
            f"Combined obstacle and design space masks, {np.count_nonzero(combined_obstacle_mask)} elements restricted"
        )

    return design_space_mask, obstacle_mask, combined_obstacle_mask


def create_boundary_conditions(nelx, nely, nelz) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create boundary condition arrays for loads and constraints.

    A wrapper around create_boundary_arrays with standardized output.

    Args:
        nelx: Number of elements in x direction
        nely: Number of elements in y direction
        nelz: Number of elements in z direction

    Returns:
        Tuple containing loads array and constraints array
    """
    return create_boundary_arrays(nelx, nely, nelz)
