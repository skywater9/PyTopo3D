#!/usr/bin/env python3
"""
Main entry point for the 3D topology optimization package.

This script provides a command-line interface to run the topology optimization.
"""

import sys
import time
import numpy as np

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
from pytopo3d.utils.assembly import build_force_field, build_force_vector, build_support_mask, build_supports
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
from pytopo3d.utils.boundary import create_bc_visualization_arrays, create_bc_visualization_arrays_from_masks
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
        logger, results_mgr = setup_experiment(
            verbose=args.verbose,
            quiet=args.quiet,
            log_level=args.log_level,
            log_file=args.log_file,
            experiment_name=getattr(args, "experiment_name", None),
            description=args.description,
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            volfrac=args.volfrac,
            penal=args.penal,
            rmin=args.rmin,
        )

        # Update args.experiment_name if it was generated in setup_experiment
        if not hasattr(args, "experiment_name") or not args.experiment_name:
            args.experiment_name = results_mgr.experiment_name

        # Load design space and obstacle data
        design_space_mask, obstacle_mask, combined_obstacle_mask, args.nelx, args.nely, args.nelz = load_geometry_data(
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            design_space_stl=getattr(args, "design_space_stl", None),
            target_nelx=getattr(args, "target_nelx", None),
            invert_design_space=getattr(args, "invert_design_space", False),
            obstacle_config=getattr(args, "obstacle_config", None),
            experiment_name=args.experiment_name,
            logger=logger,
            results_mgr=results_mgr,
        )

        if getattr(args, "target_physical_x", None) is not None:  
            args.elem_size = args.target_physical_x / args.nelx # in meters

        # Determine number of DOFs
        ndof = 3 * (args.nelx + 1) * (args.nely + 1) * (args.nelz + 1)

        # set material parameters/preset
        material_preset = args.material_preset
        material_orientation_xyz = parse_material_orientation_xyz(
            getattr(args, "material_orientation_xyz", None)
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
            getattr(args, "eval_material_orientation_xyz", None)
        )
        if eval_material_orientation_xyz is None:
            eval_material_orientation_xyz = material_orientation_xyz

        eval_material_queue = []
        if getattr(args, "eval_material_presets", None):
            if material_preset is not None:
                eval_material_queue.append(material_preset)
            eval_material_queue.extend(args.eval_material_presets)
            deduped = []
            seen = set()
            for preset_name in eval_material_queue:
                key = preset_name.lower()
                if key not in seen:
                    deduped.append(preset_name)
                    seen.add(key)
            eval_material_queue = deduped

        # --- Build Boundary Conditions ---
        force_field_preset = args.force_field_preset
        if force_field_preset is not None:
            force_field = build_force_field(
                args.nelx, 
                args.nely, 
                args.nelz, 
                *get_force_field_params(force_field_preset)
            )

        else:
            force_field = None

        support_mask_preset = args.support_mask_preset
        if support_mask_preset is not None:
            support_mask = build_support_mask(
                args.nelx, 
                args.nely, 
                args.nelz, 
                *get_support_mask_params(support_mask_preset)
            )

        else:
            support_mask = None

        # Load protected zone masks if specified
        protected_zone_mask = None
        if getattr(args, "protected_zones", None):
            from pytopo3d.utils.config_loader import get_protected_zone_ranges
            zone_names = tuple(args.protected_zones)
            protected_zone_ranges = get_protected_zone_ranges(zone_names)
            # Build a combined mask for all protected zones
            protected_zone_mask = np.zeros((args.nely, args.nelx, args.nelz), dtype=bool)
            for zone in protected_zone_ranges:
                x1, x2, y1, y2, z1, z2 = zone
                protected_zone_mask[y1:y2, x1:x2, z1:z2] = True

        logger.info("Building force vector")
        F = build_force_vector(
            args.nelx, args.nely, args.nelz, ndof, force_field=force_field
        )
        logger.info("Building support constraints")
        freedofs0, fixeddof0 = build_supports(
            args.nelx, args.nely, args.nelz, ndof, support_mask=support_mask
        )

        # Create visualization arrays from actual BCs
        loads_array, constraints_array = create_bc_visualization_arrays_from_masks(
            args.nelx, args.nely, args.nelz, ndof, force_field, support_mask
        )
        logger.info("Generated boundary condition visualization arrays")

        # Create and save initial visualization
        visualize_initial_setup(
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            loads_array=loads_array,
            constraints_array=constraints_array,
            experiment_name=args.experiment_name,
            logger=logger,
            results_mgr=results_mgr,
            combined_obstacle_mask=combined_obstacle_mask,
            protected_zone_mask=protected_zone_mask,
        )

        optimization_diagnostics = {}

        # Run the optimization - Passing force_field and support_mask
        if getattr(args, "skip_optimization", False):
            # Skip optimization - create a solid block for FEA testing
            logger.info("Skipping optimization - creating solid block for FEA testing")
            start_time = time.time()
            xPhys = np.ones((args.nely, args.nelx, args.nelz))  # All solid (density = 1.0)
            xPhys[combined_obstacle_mask] = 0.0
            history = None
            final_compliance = 0.0  # Placeholder
            failure_force = 0.0
            run_time = time.time() - start_time
            optimization_diagnostics["projection_enabled"] = False
        else:
            xPhys, history, final_compliance, failure_force, run_time = execute_optimization(
                nelx=args.nelx,
                nely=args.nely,
                nelz=args.nelz,
                volfrac=args.volfrac,
                penal=args.penal,
                rmin=args.rmin,
                disp_thres=args.disp_thres,
                elem_size=args.elem_size,
                material_params=material_params,
                force_field=force_field,
                support_mask=support_mask,
                tolx=getattr(args, "tolx", 0.01),
                maxloop=getattr(args, "maxloop", 2000),
                create_animation=getattr(args, "create_animation", False),
                animation_frequency=getattr(args, "animation_frequency", 10),
                logger=logger,
                combined_obstacle_mask=combined_obstacle_mask,
                use_gpu=args.gpu,
                protected_zone_mask=protected_zone_mask,
                beta_schedule=getattr(
                    args, "beta_schedule", (1.0, 2.0, 4.0, 8.0)
                ),
                projection_eta=getattr(args, "projection_eta", 0.5),
                move=getattr(args, "move_limit", 0.2),
                diagnostics_out=optimization_diagnostics,
            )

        failure_postprocessing = None
        if material_strength is not None:
            failure_postprocessing = evaluate_failure_representations(
                x_projected=xPhys,
                binary_threshold=float(getattr(args, "stl_level", 0.5)),
                penal=args.penal,
                material_params=material_params,
                strength=material_strength,
                orientation_matrix=material_orientation_matrix(
                    material_orientation_xyz
                ),
                elem_size=args.elem_size,
                force_field=force_field,
                support_mask=support_mask,
                obstacle_mask=combined_obstacle_mask,
                protected_zone_mask=protected_zone_mask,
                use_gpu=args.gpu,
                results_manager=results_mgr,
            )
            final_response_metrics = failure_postprocessing.projected_response
            optimization_diagnostics.update(failure_postprocessing.metrics)
            failure_force = failure_postprocessing.metrics[
                "predicted_failure_load_projected"
            ]
            projected_fi = failure_postprocessing.metrics[
                "failure_index_max_projected"
            ]
            binary_fi = failure_postprocessing.metrics["failure_index_max_binary"]
            logger.info(
                "Failure post-processing: projected FI=%s, binary FI=%s",
                "n/a" if projected_fi is None else f"{projected_fi:.6e}",
                "n/a" if binary_fi is None else f"{binary_fi:.6e}",
            )
        else:
            final_response_metrics = evaluate_fixed_geometry_metrics(
                xPhys=xPhys,
                penal=args.penal,
                material_params=material_params,
                elem_size=args.elem_size,
                force_field=force_field,
                support_mask=support_mask,
                obstacle_mask=combined_obstacle_mask,
                protected_zone_mask=protected_zone_mask,
                use_gpu=args.gpu,
            )

        if getattr(args, "skip_optimization", False):
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
                    penal=args.penal,
                    material_params=eval_material_params,
                    elem_size=args.elem_size,
                    force_field=force_field,
                    support_mask=support_mask,
                    obstacle_mask=combined_obstacle_mask,
                    protected_zone_mask=protected_zone_mask,
                    use_gpu=args.gpu,
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

        # Binary evaluation mirrors the STL threshold exactly so the reported
        # compliance matches the pre-smoothing binary voxel behavior.
        binary_eval_level = float(getattr(args, "stl_level", 0.5))

        if eval_material_queue:
            binary_eval_material_queue = list(eval_material_queue)
            binary_eval_orientation = eval_material_orientation_xyz
        elif material_preset is not None:
            binary_eval_material_queue = [material_preset]
            binary_eval_orientation = material_orientation_xyz
        else:
            binary_eval_material_queue = [None]
            binary_eval_orientation = material_orientation_xyz

        final_binary_voxel_eval = []
        if failure_postprocessing is not None:
            x_binary = failure_postprocessing.binary_density.copy()
        else:
            x_binary = (xPhys >= binary_eval_level).astype(float)
            if protected_zone_mask is not None:
                x_binary[protected_zone_mask] = 1.0
            x_binary[combined_obstacle_mask] = 0.0
        voxel_fill_fraction = float(np.mean(x_binary))

        for eval_material_preset in binary_eval_material_queue:
            if eval_material_preset is None:
                eval_material_params = material_params
                eval_material_name = "optimizer_material"
            else:
                eval_material_params = get_material_params(eval_material_preset)
                eval_material_params = apply_material_orientation(
                    eval_material_params,
                    binary_eval_orientation,
                )
                eval_material_name = eval_material_preset

            can_reuse_failure_solve = (
                failure_postprocessing is not None
                and eval_material_preset is not None
                and material_preset is not None
                and eval_material_preset.lower() == material_preset.lower()
                and binary_eval_orientation == material_orientation_xyz
            )
            if can_reuse_failure_solve:
                eval_binary_metrics = failure_postprocessing.binary_response
            else:
                eval_binary_metrics = evaluate_fixed_geometry_metrics(
                    xPhys=x_binary,
                    penal=args.penal,
                    material_params=eval_material_params,
                    elem_size=args.elem_size,
                    force_field=force_field,
                    support_mask=support_mask,
                    obstacle_mask=combined_obstacle_mask,
                    protected_zone_mask=protected_zone_mask,
                    use_gpu=args.gpu,
                )

            final_binary_voxel_eval.append(
                {
                    "material_preset": eval_material_name,
                    "material_orientation_xyz": binary_eval_orientation,
                    "voxel_fill_fraction": voxel_fill_fraction,
                    "compliance": eval_binary_metrics["compliance"],
                    "ux_avg_load_patch": eval_binary_metrics["ux_avg_load_patch"],
                    "uy_avg_load_patch": eval_binary_metrics["uy_avg_load_patch"],
                    "uz_avg_load_patch": eval_binary_metrics["uz_avg_load_patch"],
                    "k_avg_x": eval_binary_metrics["k_avg_x"],
                    "k_avg_y": eval_binary_metrics["k_avg_y"],
                    "k_avg_z": eval_binary_metrics["k_avg_z"],
                    "k_avg": eval_binary_metrics["k_avg"],
                }
            )

        baseline_row = None
        if material_preset is not None:
            for row in final_binary_voxel_eval:
                if row["material_preset"].lower() == material_preset.lower():
                    baseline_row = row
                    break
        if baseline_row is None and final_binary_voxel_eval:
            baseline_row = final_binary_voxel_eval[0]

        if baseline_row is not None and baseline_row["compliance"] != 0.0:
            baseline_compliance = baseline_row["compliance"]
            for row in final_binary_voxel_eval:
                row["relative_to_baseline"] = row["compliance"] / baseline_compliance

        sorted_rows = sorted(final_binary_voxel_eval, key=lambda row: row["compliance"])
        for rank, row in enumerate(sorted_rows, start=1):
            row["rank"] = rank

        logger.info(
            "Binary voxel evaluation at STL level=%.3f (lower compliance means stiffer):",
            binary_eval_level,
        )
        for row in sorted_rows:
            ratio_text = ""
            if "relative_to_baseline" in row:
                ratio_text = f", ratio={row['relative_to_baseline']:.4f}"
            logger.info(
                "  rank=%d, material=%s, compliance=%.6e, fill=%.4f%s",
                row["rank"],
                row["material_preset"],
                row["compliance"],
                row["voxel_fill_fraction"],
                ratio_text,
            )

        # xPhys is already the final projected physical field used by FEA.
        result_path = results_mgr.save_result(xPhys, "optimized_design.npy")
        logger.debug(f"Optimization result saved to {result_path}")

        # Create visualization of the final result
        visualize_final_result(
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            experiment_name=args.experiment_name,
            disp_thres=args.disp_thres,
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
                nelx=args.nelx,
                nely=args.nely,
                nelz=args.nelz,
                experiment_name=args.experiment_name,
                disp_thres=args.disp_thres,
                animation_frames=getattr(args, "animation_frames", 50),
                animation_fps=getattr(args, "animation_fps", 5),
                logger=logger,
                results_mgr=results_mgr,
                history=history,
                combined_obstacle_mask=combined_obstacle_mask,
                loads_array=loads_array,
                constraints_array=constraints_array,
            )

        # Export result as STL if requested
        stl_exported = export_result_to_stl(
            export_stl=getattr(args, "export_stl", False),
            stl_level=getattr(args, "stl_level", 0.5),
            export_mode=getattr(args, "export_mode", "density"),
            smooth_stl=getattr(args, "smooth_stl", False),
            smooth_iterations=getattr(args, "smooth_iterations", 3),
            combined_obstacle_mask=combined_obstacle_mask,
            logger=logger,
            results_mgr=results_mgr,
            result_path=result_path,
            elem_size=args.elem_size,
        )

        terminal_input = ' '.join(sys.argv[1:])

        # Collect and save metrics
        metrics = collect_metrics(
            terminal_input=terminal_input,
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            volfrac=args.volfrac,
            penal=args.penal,
            rmin=args.rmin,
            material_preset=args.material_preset,
            material_orientation_xyz=material_orientation_xyz,
            force_field_preset=args.force_field_preset,
            support_mask_preset=args.support_mask_preset,
            elem_size=args.elem_size,
            disp_thres=args.disp_thres,
            tolx=getattr(args, "tolx", 0.01),
            maxloop=getattr(args, "maxloop", 2000),
            design_space_stl=getattr(args, "design_space_stl", None),
            pitch=getattr(args, "pitch", 1.0),
            obstacle_config=getattr(args, "obstacle_config", None),
            animation_fps=getattr(args, "animation_fps", 5),
            stl_level=getattr(args, "stl_level", 0.5),
            stl_export_mode=getattr(args, "export_mode", "density"),
            smooth_stl=getattr(args, "smooth_stl", False),
            smooth_iterations=getattr(args, "smooth_iterations", 3),
            skip_optimization=getattr(args, "skip_optimization", False),
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
            final_binary_voxel_eval=final_binary_voxel_eval,
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
