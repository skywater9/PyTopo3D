#!/usr/bin/env python3
"""
Main entry point for the 3D topology optimization package.

This script provides a command-line interface to run the topology optimization.
"""

import sys

from pytopo3d.cli.parser import parse_args
from pytopo3d.preprocessing.geometry import load_geometry_data
from pytopo3d.runners.experiment import (
    execute_optimization,
    export_result_to_stl,
    setup_experiment,
)
from pytopo3d.utils.metrics import collect_metrics
from pytopo3d.visualization.visualizer import (
    create_optimization_animation,
    visualize_final_result,
    visualize_initial_setup,
)


def main():
    """
    Main function to run the optimization from command-line arguments.
    """
    # Parse command-line arguments
    args = parse_args()

    try:
        # Setup experiment, logging and results manager
        logger, results_mgr = setup_experiment(args)

        # Load design space and obstacle data
        design_space_mask, obstacle_mask, combined_obstacle_mask = load_geometry_data(
            args, logger, results_mgr
        )

        # Create and save initial visualization
        loads_array, constraints_array, _ = visualize_initial_setup(
            args, logger, results_mgr, combined_obstacle_mask
        )

        # Run the optimization
        xPhys, history, run_time = execute_optimization(
            args, logger, combined_obstacle_mask
        )

        # Save the result to the experiment directory
        result_path = results_mgr.save_result(xPhys, "optimized_design.npy")
        logger.debug(f"Optimization result saved to {result_path}")

        # Create visualization of the final result
        visualize_final_result(
            args,
            logger,
            results_mgr,
            xPhys,
            combined_obstacle_mask,
            loads_array,
            constraints_array,
        )

        # Create animation if history was captured
        gif_path = None
        if history:
            gif_path = create_optimization_animation(
                args,
                logger,
                results_mgr,
                history,
                combined_obstacle_mask,
                loads_array,
                constraints_array,
            )

        # Export result as STL if requested
        stl_exported = export_result_to_stl(args, logger, results_mgr, result_path)

        # Collect and save metrics
        metrics = collect_metrics(
            args,
            xPhys,
            design_space_mask,
            obstacle_mask,
            combined_obstacle_mask,
            run_time,
            gif_path,
            stl_exported,
        )
        results_mgr.update_metrics(metrics)
        logger.debug("Metrics updated")

        logger.info(f"Optimization complete in {run_time:.2f} seconds.")
        logger.info(f"Result saved to {result_path}")
        logger.info(f"All experiment files are in {results_mgr.experiment_dir}")

    except Exception as e:
        if "logger" in locals():
            logger.error(f"Error in main function: {e}")
            import traceback

            logger.debug(f"Error details: {traceback.format_exc()}")
        else:
            print(f"Error during initialization: {e}", file=sys.stderr)
            import traceback

            print(f"Error details: {traceback.format_exc()}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
