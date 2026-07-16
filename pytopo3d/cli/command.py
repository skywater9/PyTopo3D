#!/usr/bin/env python3
"""
Command-line entry point for the pytopo3d package.
This module provides the main entry point when installed as a package.
"""

import sys
import time
from typing import List, Optional

import numpy as np

# Import the main function from the main module
from pytopo3d.analysis.postprocessing import evaluate_failure_representations
from pytopo3d.cli.parser import parse_args
from pytopo3d.core.optimizer import (
    evaluate_fixed_geometry_compliance,
    evaluate_fixed_geometry_metrics,
)
from pytopo3d.preprocessing.geometry import load_geometry_data
from pytopo3d.runners.experiment import (
    execute_optimization,
    export_result_to_stl,
    setup_experiment,
)
from pytopo3d.utils.assembly import (
    build_force_field,
    build_force_vector,
    build_support_mask,
    build_supports,
)
from pytopo3d.utils.boundary import create_bc_visualization_arrays_from_masks
from pytopo3d.utils.config_loader import (
    apply_material_orientation,
    get_force_field_params,
    get_material_params,
    get_material_strength,
    get_support_mask_params,
    material_has_strength,
    material_orientation_matrix,
    parse_material_orientation_xyz,
)
from pytopo3d.utils.metrics import collect_metrics
from pytopo3d.visualization.visualizer import (
    create_optimization_animation,
    visualize_final_result,
    visualize_initial_setup,
)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main function to run the optimization from command-line arguments.

    Args:
        args: Command line arguments (sys.argv[1:] by default)

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Parse command-line arguments
    if args is None:
        args = sys.argv[1:]

    parsed_args = parse_args(args)

    try:
        # Setup experiment, logging and results manager
        logger, results_mgr = setup_experiment(
            verbose=parsed_args.verbose,
            quiet=parsed_args.quiet,
            log_level=parsed_args.log_level,
            log_file=parsed_args.log_file,
            experiment_name=getattr(parsed_args, "experiment_name", None),
            description=parsed_args.description,
            nelx=parsed_args.nelx,
            nely=parsed_args.nely,
            nelz=parsed_args.nelz,
            volfrac=parsed_args.volfrac,
            penal=parsed_args.penal,
            rmin=parsed_args.rmin,
        )

        # Update args.experiment_name if it was generated in setup_experiment
        if (
            not hasattr(parsed_args, "experiment_name")
            or not parsed_args.experiment_name
        ):
            parsed_args.experiment_name = results_mgr.experiment_name

        # Load design space and obstacle data
        (
            design_space_mask,
            obstacle_mask,
            combined_obstacle_mask,
            parsed_args.nelx,
            parsed_args.nely,
            parsed_args.nelz,
        ) = load_geometry_data(
            nelx=parsed_args.nelx,
            nely=parsed_args.nely,
            nelz=parsed_args.nelz,
            design_space_stl=getattr(parsed_args, "design_space_stl", None),
            target_nelx=getattr(parsed_args, "target_nelx", None),
            invert_design_space=getattr(parsed_args, "invert_design_space", False),
            obstacle_config=getattr(parsed_args, "obstacle_config", None),
            experiment_name=parsed_args.experiment_name,
            logger=logger,
            results_mgr=results_mgr,
        )

        if getattr(parsed_args, "target_physical_x", None) is not None:
            parsed_args.elem_size = parsed_args.target_physical_x / parsed_args.nelx

        ndof = 3 * (parsed_args.nelx + 1) * (parsed_args.nely + 1) * (parsed_args.nelz + 1)

        material_preset = parsed_args.material_preset
        material_orientation_xyz = parse_material_orientation_xyz(
            getattr(parsed_args, "material_orientation_xyz", None)
        )
        if material_preset is not None:
            material_params = get_material_params(material_preset)
            material_params = apply_material_orientation(
                material_params,
                material_orientation_xyz,
            )
            if material_orientation_xyz is not None:
                logger.info(
                    "Applied material orientation mapping (material x/y/z -> global %s/%s/%s)",
                    material_orientation_xyz[0],
                    material_orientation_xyz[1],
                    material_orientation_xyz[2],
                )
        else:
            material_params = None
            if material_orientation_xyz is not None:
                logger.warning(
                    "--material-orientation-xyz was provided without --material-preset; orientation mapping is ignored."
                )

        material_strength = None
        if material_preset is not None and material_has_strength(material_preset):
            material_strength = get_material_strength(material_preset)
            logger.info(
                "Enabled final maximum-stress failure post-processing for material '%s'",
                material_preset,
            )

        eval_material_orientation_xyz = parse_material_orientation_xyz(
            getattr(parsed_args, "eval_material_orientation_xyz", None)
        )
        if eval_material_orientation_xyz is None:
            eval_material_orientation_xyz = material_orientation_xyz

        eval_material_queue = []
        if getattr(parsed_args, "eval_material_presets", None):
            if material_preset is not None:
                eval_material_queue.append(material_preset)
            eval_material_queue.extend(parsed_args.eval_material_presets)
            deduped = []
            seen = set()
            for preset_name in eval_material_queue:
                key = preset_name.lower()
                if key not in seen:
                    deduped.append(preset_name)
                    seen.add(key)
            eval_material_queue = deduped

        force_field_preset = parsed_args.force_field_preset
        if force_field_preset is not None:
            force_field = build_force_field(
                parsed_args.nelx,
                parsed_args.nely,
                parsed_args.nelz,
                *get_force_field_params(force_field_preset),
            )
        else:
            force_field = None

        support_mask_preset = parsed_args.support_mask_preset
        if support_mask_preset is not None:
            support_mask = build_support_mask(
                parsed_args.nelx,
                parsed_args.nely,
                parsed_args.nelz,
                *get_support_mask_params(support_mask_preset),
            )
        else:
            support_mask = None

        protected_zone_mask = None
        if getattr(parsed_args, "protected_zones", None):
            from pytopo3d.utils.config_loader import get_protected_zone_ranges

            zone_names = tuple(parsed_args.protected_zones)
            protected_zone_ranges = get_protected_zone_ranges(zone_names)
            protected_zone_mask = np.zeros(
                (parsed_args.nely, parsed_args.nelx, parsed_args.nelz), dtype=bool
            )
            for zone in protected_zone_ranges:
                x1, x2, y1, y2, z1, z2 = zone
                protected_zone_mask[y1:y2, x1:x2, z1:z2] = True

        logger.info("Building force vector")
        build_force_vector(
            parsed_args.nelx,
            parsed_args.nely,
            parsed_args.nelz,
            ndof,
            force_field=force_field,
        )
        logger.info("Building support constraints")
        build_supports(
            parsed_args.nelx,
            parsed_args.nely,
            parsed_args.nelz,
            ndof,
            support_mask=support_mask,
        )

        loads_array, constraints_array = create_bc_visualization_arrays_from_masks(
            parsed_args.nelx,
            parsed_args.nely,
            parsed_args.nelz,
            ndof,
            force_field,
            support_mask,
        )
        logger.info("Generated boundary condition visualization arrays")

        # Create and save initial visualization
        visualize_initial_setup(
            nelx=parsed_args.nelx,
            nely=parsed_args.nely,
            nelz=parsed_args.nelz,
            loads_array=loads_array,
            constraints_array=constraints_array,
            experiment_name=parsed_args.experiment_name,
            logger=logger,
            results_mgr=results_mgr,
            combined_obstacle_mask=combined_obstacle_mask,
            protected_zone_mask=protected_zone_mask,
        )

        optimization_diagnostics = {}

        # Run the optimization
        if getattr(parsed_args, "skip_optimization", False):
            # Skip optimization - create a solid block for FEA testing
            logger.info("Skipping optimization - creating solid block for FEA testing")
            start_time = time.time()
            xPhys = np.ones((parsed_args.nely, parsed_args.nelx, parsed_args.nelz))  # All solid (density = 1.0)
            xPhys[combined_obstacle_mask] = 0.0
            history = None
            final_compliance = 0.0  # Placeholder
            run_time = time.time() - start_time
            optimization_diagnostics["projection_enabled"] = False
        else:
            xPhys, history, final_compliance, _, run_time = execute_optimization(
                nelx=parsed_args.nelx,
                nely=parsed_args.nely,
                nelz=parsed_args.nelz,
                volfrac=parsed_args.volfrac,
                penal=parsed_args.penal,
                rmin=parsed_args.rmin,
                disp_thres=parsed_args.disp_thres,
                elem_size=parsed_args.elem_size,
                material_params=material_params,
                force_field=force_field,
                support_mask=support_mask,
                tolx=getattr(parsed_args, "tolx", 0.01),
                maxloop=getattr(parsed_args, "maxloop", 2000),
                create_animation=getattr(parsed_args, "create_animation", False),
                animation_frequency=getattr(parsed_args, "animation_frequency", 10),
                logger=logger,
                combined_obstacle_mask=combined_obstacle_mask,
                use_gpu=parsed_args.gpu,
                protected_zone_mask=protected_zone_mask,
                beta_schedule=getattr(
                    parsed_args, "beta_schedule", (1.0, 2.0, 4.0, 8.0)
                ),
                projection_eta=getattr(parsed_args, "projection_eta", 0.5),
                move=getattr(parsed_args, "move_limit", 0.2),
                diagnostics_out=optimization_diagnostics,
                optimization_mode=getattr(
                    parsed_args, "optimization_mode", "compliance"
                ),
                optimizer=getattr(parsed_args, "optimizer", "oc"),
                material_strength=material_strength,
                material_orientation=material_orientation_matrix(
                    material_orientation_xyz
                ),
                failure_limit=getattr(parsed_args, "failure_limit", 1.0),
                failure_aggregate_exponent=getattr(
                    parsed_args, "failure_aggregate_exponent", 8.0
                ),
                failure_relaxation_exponent=getattr(
                    parsed_args, "failure_relaxation_exponent", 0.5
                ),
                mma_move=getattr(parsed_args, "mma_move_limit", 0.05),
                mma_min_density=getattr(
                    parsed_args, "mma_min_density", 1.0e-3
                ),
                failure_limit_schedule=getattr(
                    parsed_args, "failure_limit_schedule", None
                ),
                failure_aggregate_exponent_schedule=getattr(
                    parsed_args,
                    "failure_aggregate_exponent_schedule",
                    None,
                ),
            )

        if material_strength is not None:
            failure_postprocessing = evaluate_failure_representations(
                x_projected=xPhys,
                binary_threshold=float(getattr(parsed_args, "stl_level", 0.5)),
                penal=parsed_args.penal,
                material_params=material_params,
                strength=material_strength,
                orientation_matrix=material_orientation_matrix(
                    material_orientation_xyz
                ),
                elem_size=parsed_args.elem_size,
                force_field=force_field,
                support_mask=support_mask,
                obstacle_mask=combined_obstacle_mask,
                protected_zone_mask=protected_zone_mask,
                smooth_failure_aggregate=optimization_diagnostics.get(
                    "failure_aggregate"
                ),
                smooth_failure_limit=optimization_diagnostics.get("failure_limit"),
                use_gpu=parsed_args.gpu,
                results_manager=results_mgr,
            )
            final_response_metrics = failure_postprocessing.projected_response
            optimization_diagnostics.update(failure_postprocessing.metrics)
            projected_fi = failure_postprocessing.metrics[
                "failure_index_max_projected"
            ]
            binary_fi = failure_postprocessing.metrics["failure_index_max_binary"]
            logger.info(
                "Failure post-processing: projected FI=%s, binary FI=%s, "
                "internal verification=%s",
                "n/a" if projected_fi is None else f"{projected_fi:.6e}",
                "n/a" if binary_fi is None else f"{binary_fi:.6e}",
                failure_postprocessing.metrics[
                    "stage10_internal_verification_status"
                ],
            )
        else:
            final_response_metrics = evaluate_fixed_geometry_metrics(
                xPhys=xPhys,
                penal=parsed_args.penal,
                material_params=material_params,
                elem_size=parsed_args.elem_size,
                force_field=force_field,
                support_mask=support_mask,
                obstacle_mask=combined_obstacle_mask,
                protected_zone_mask=protected_zone_mask,
                use_gpu=parsed_args.gpu,
            )

        if getattr(parsed_args, "skip_optimization", False):
            final_compliance = final_response_metrics["compliance"]

        final_voxel_eval = None
        if eval_material_queue:
            final_voxel_eval = []
            for eval_material_preset in eval_material_queue:
                eval_material_params = get_material_params(eval_material_preset)
                eval_material_params = apply_material_orientation(
                    eval_material_params,
                    eval_material_orientation_xyz,
                )
                eval_metrics = evaluate_fixed_geometry_metrics(
                    xPhys=xPhys,
                    penal=parsed_args.penal,
                    material_params=eval_material_params,
                    elem_size=parsed_args.elem_size,
                    force_field=force_field,
                    support_mask=support_mask,
                    obstacle_mask=combined_obstacle_mask,
                    protected_zone_mask=protected_zone_mask,
                    use_gpu=parsed_args.gpu,
                )
                final_voxel_eval.append(
                    {
                        "material_preset": eval_material_preset,
                        "material_orientation_xyz": eval_material_orientation_xyz,
                        "compliance": eval_metrics["compliance"],
                        "ux_avg_load_patch": eval_metrics["ux_avg_load_patch"],
                        "uy_avg_load_patch": eval_metrics["uy_avg_load_patch"],
                        "uz_avg_load_patch": eval_metrics["uz_avg_load_patch"],
                        "k_avg_x": eval_metrics["k_avg_x"],
                        "k_avg_y": eval_metrics["k_avg_y"],
                        "k_avg_z": eval_metrics["k_avg_z"],
                        "k_avg": eval_metrics["k_avg"],
                    }
                )

            baseline_compliance = None
            if material_preset is not None:
                for row in final_voxel_eval:
                    if row["material_preset"].lower() == material_preset.lower():
                        baseline_compliance = row["compliance"]
                        break
            if baseline_compliance is None and final_compliance is not None:
                baseline_compliance = final_compliance

            if baseline_compliance is not None and baseline_compliance != 0.0:
                for row in final_voxel_eval:
                    row["relative_to_baseline"] = row["compliance"] / baseline_compliance

            sorted_rows = sorted(final_voxel_eval, key=lambda row: row["compliance"])
            for rank, row in enumerate(sorted_rows, start=1):
                row["rank"] = rank

            logger.info(
                "Final voxel cross-material evaluation (lower compliance means stiffer):"
            )
            for row in sorted_rows:
                ratio_text = ""
                if "relative_to_baseline" in row:
                    ratio_text = f", ratio={row['relative_to_baseline']:.4f}"
                logger.info(
                    "  rank=%d, material=%s, compliance=%.6e%s",
                    row["rank"],
                    row["material_preset"],
                    row["compliance"],
                    ratio_text,
                )

        # Save the exact projected physical field used by FEA.
        result_path = results_mgr.save_result(xPhys, "optimized_design.npy")
        logger.debug(f"Optimization result saved to {result_path}")

        # Create visualization of the final result
        visualize_final_result(
            nelx=parsed_args.nelx,
            nely=parsed_args.nely,
            nelz=parsed_args.nelz,
            experiment_name=parsed_args.experiment_name,
            disp_thres=parsed_args.disp_thres,
            logger=logger,
            results_mgr=results_mgr,
            xPhys=xPhys,
            combined_obstacle_mask=combined_obstacle_mask,
            loads_array=loads_array,
            constraints_array=constraints_array,
        )

        # Create animation if history was captured
        gif_path = None
        if history:
            gif_path = create_optimization_animation(
                nelx=parsed_args.nelx,
                nely=parsed_args.nely,
                nelz=parsed_args.nelz,
                experiment_name=parsed_args.experiment_name,
                disp_thres=parsed_args.disp_thres,
                animation_frames=getattr(parsed_args, "animation_frames", 50),
                animation_fps=getattr(parsed_args, "animation_fps", 5),
                logger=logger,
                results_mgr=results_mgr,
                history=history,
                combined_obstacle_mask=combined_obstacle_mask,
                loads_array=loads_array,
                constraints_array=constraints_array,
            )

        # Export result as STL if requested
        stl_exported = export_result_to_stl(
            export_stl=getattr(parsed_args, "export_stl", False),
            stl_level=getattr(parsed_args, "stl_level", 0.5),
            export_mode=getattr(parsed_args, "export_mode", "density"),
            smooth_stl=getattr(parsed_args, "smooth_stl", False),
            smooth_iterations=getattr(parsed_args, "smooth_iterations", 3),
            combined_obstacle_mask=combined_obstacle_mask,
            logger=logger,
            results_mgr=results_mgr,
            result_path=result_path,
            elem_size=parsed_args.elem_size,
        )

        terminal_input = " ".join(args)

        # Collect and save metrics
        metrics = collect_metrics(
            terminal_input=terminal_input,
            nelx=parsed_args.nelx,
            nely=parsed_args.nely,
            nelz=parsed_args.nelz,
            volfrac=parsed_args.volfrac,
            penal=parsed_args.penal,
            rmin=parsed_args.rmin,
            disp_thres=parsed_args.disp_thres,
            material_preset=parsed_args.material_preset,
            material_orientation_xyz=material_orientation_xyz,
            force_field_preset=parsed_args.force_field_preset,
            support_mask_preset=parsed_args.support_mask_preset,
            elem_size=parsed_args.elem_size,
            tolx=getattr(parsed_args, "tolx", 0.01),
            maxloop=getattr(parsed_args, "maxloop", 2000),
            design_space_stl=getattr(parsed_args, "design_space_stl", None),
            pitch=getattr(parsed_args, "pitch", 1.0),
            obstacle_config=getattr(parsed_args, "obstacle_config", None),
            animation_fps=getattr(parsed_args, "animation_fps", 5),
            stl_level=getattr(parsed_args, "stl_level", 0.5),
            stl_export_mode=getattr(parsed_args, "export_mode", "density"),
            smooth_stl=getattr(parsed_args, "smooth_stl", False),
            smooth_iterations=getattr(parsed_args, "smooth_iterations", 3),
            skip_optimization=getattr(parsed_args, "skip_optimization", False),
            xPhys=xPhys,
            design_space_mask=design_space_mask,
            obstacle_mask=obstacle_mask,
            combined_obstacle_mask=combined_obstacle_mask,
            run_time=run_time,
            final_compliance=final_compliance,
            final_ux_avg_load_patch=final_response_metrics["ux_avg_load_patch"],
            final_uy_avg_load_patch=final_response_metrics["uy_avg_load_patch"],
            final_uz_avg_load_patch=final_response_metrics["uz_avg_load_patch"],
            final_k_avg_x=final_response_metrics["k_avg_x"],
            final_k_avg_y=final_response_metrics["k_avg_y"],
            final_k_avg_z=final_response_metrics["k_avg_z"],
            final_k_avg=final_response_metrics["k_avg"],
            final_voxel_eval=final_voxel_eval,
            optimization_metrics=optimization_diagnostics,
            gif_path=gif_path,
            stl_exported=stl_exported,
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
