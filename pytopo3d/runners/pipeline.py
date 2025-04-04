"""
High-level pipeline for running topology optimization.

This module provides a complete pipeline for running topology optimization
from configuration to results generation.
"""

from typing import Any, Dict

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


def run_optimization_pipeline(args) -> Dict[str, Any]:
    """
    Run the complete topology optimization pipeline.

    This is a high-level function that orchestrates the entire optimization process
    from setup to results generation. It can be used as an alternative to the main script.

    Args:
        args: Command-line arguments or configuration object

    Returns:
        Dictionary containing results and metrics
    """
    # Setup experiment, logging and results manager
    logger, results_mgr = setup_experiment(args)

    # Load design space and obstacle data
    design_space_mask, obstacle_mask, combined_obstacle_mask = load_geometry_data(
        args, logger, results_mgr
    )

    # Create and save initial visualization
    loads_array, constraints_array, boundary_viz_path = visualize_initial_setup(
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
    viz_path = visualize_final_result(
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

    # Log completion
    logger.info(f"Optimization complete in {run_time:.2f} seconds.")
    logger.info(f"Result saved to {result_path}")
    logger.info(f"All experiment files are in {results_mgr.experiment_dir}")

    # Return a dictionary with all the important results and paths
    return {
        "xPhys": xPhys,
        "history": history,
        "metrics": metrics,
        "result_path": result_path,
        "visualization_path": viz_path,
        "animation_path": gif_path,
        "experiment_dir": results_mgr.experiment_dir,
        "stl_exported": stl_exported,
    }


def run_batch_optimization(configs_list: list) -> Dict[str, Dict[str, Any]]:
    """
    Run multiple optimization jobs in sequence.

    Args:
        configs_list: List of configuration objects or argument objects

    Returns:
        Dictionary mapping config names to results dictionaries
    """
    results = {}

    for i, config in enumerate(configs_list):
        name = getattr(config, "experiment_name", f"batch_job_{i}")
        try:
            results[name] = run_optimization_pipeline(config)
        except Exception as e:
            import traceback

            print(f"Error in job {name}: {e}")
            print(traceback.format_exc())
            results[name] = {"error": str(e), "traceback": traceback.format_exc()}

    return results
