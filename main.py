#!/usr/bin/env python3
"""
Main entry point for the 3D topology optimization package.

This script provides a command-line interface to run the topology optimization.
"""

import sys
import numpy as np

from pytopo3d.cli.parser import parse_args
from pytopo3d.core.optimizer import evaluate_fixed_geometry_compliance
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
    get_support_mask_params,
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

        # Run the optimization - Passing force_field and support_mask
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
            protected_zone_mask=protected_zone_mask
        )

        final_voxel_eval = None
        if eval_material_queue:
            final_voxel_eval = []
            for eval_material_preset in eval_material_queue:
                eval_material_params = get_material_params(eval_material_preset)
                eval_material_params = apply_material_orientation(
                    eval_material_params,
                    eval_material_orientation_xyz,
                )
                eval_compliance = evaluate_fixed_geometry_compliance(
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
                        "compliance": eval_compliance,
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

        # Save the result to the experiment directory

        
        # Make a union array for stl export that can be tested
        force_mask_bool = np.any(force_field != 0, axis=-1)
        support_mask_bool = support_mask.astype(bool)
        union_mask = force_mask_bool | support_mask_bool

        xPhys_union = xPhys.copy()
        xPhys_union[union_mask] = 1.0

        result_path = results_mgr.save_result(xPhys_union, "optimized_design.npy")
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
            smooth_stl=getattr(args, "smooth_stl", False),
            smooth_iterations=getattr(args, "smooth_iterations", 3),
            logger=logger,
            results_mgr=results_mgr,
            result_path=result_path,
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
            smooth_stl=getattr(args, "smooth_stl", False),
            smooth_iterations=getattr(args, "smooth_iterations", 3),
            xPhys=xPhys,
            design_space_mask=design_space_mask,
            obstacle_mask=obstacle_mask,
            combined_obstacle_mask=combined_obstacle_mask,
            run_time=run_time,
            final_compliance=final_compliance,
            final_voxel_eval=final_voxel_eval,
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
