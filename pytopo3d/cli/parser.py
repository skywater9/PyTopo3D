"""
Command-line argument parsing for the 3D topology optimization package.
"""

import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the topology optimization.

    Parameters
    ----------
    args : Optional[List[str]], optional
        Command line arguments, by default None (uses sys.argv[1:])

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="3D Topology Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Basic parameters
    basic_group = parser.add_argument_group("Basic parameters")
    basic_group.add_argument(
        "--nelx", type=int, default=60, help="Number of elements in x direction"
    )
    basic_group.add_argument(
        "--nely", type=int, default=30, help="Number of elements in y direction"
    )
    basic_group.add_argument(
        "--nelz", type=int, default=20, help="Number of elements in z direction"
    )
    basic_group.add_argument(
        "--volfrac", type=float, default=0.3, help="Volume fraction constraint"
    )
    basic_group.add_argument(
        "--penal", type=float, default=3.0, help="Penalty parameter"
    )
    basic_group.add_argument("--rmin", type=float, default=3.0, help="Filter radius")
    basic_group.add_argument(
        "--disp_thres",
        type=float,
        default=0.5,
        help="Threshold for displaying elements in visualization",
    )
    basic_group.add_argument(
        "--tolx",
        type=float,
        default=0.01,
        help="Convergence tolerance on design change",
    )
    basic_group.add_argument(
        "--maxloop",
        type=int,
        default=2000,
        help="Maximum number of iterations",
    )

    # Performance parameters
    performance_group = parser.add_argument_group("Performance parameters")
    performance_group.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Enable GPU acceleration when available (requires CuPy)",
    )

    # Output parameters
    output_group = parser.add_argument_group("Output parameters")
    output_group.add_argument(
        "--output",
        type=str,
        default="optimized_design.npy",
        help="Output filename for the optimized design",
    )
    output_group.add_argument(
        "--export-stl",
        action="store_true",
        help="Export the final optimization result as an STL file",
    )
    output_group.add_argument(
        "--stl-level",
        type=float,
        default=0.5,
        help="Contour level for STL export (default: 0.5)",
    )
    output_group.add_argument(
        "--smooth-stl",
        action="store_true",
        default=True,
        help="Apply smoothing to the exported STL (default: True)",
    )
    output_group.add_argument(
        "--smooth-iterations",
        type=int,
        default=5,
        help="Number of smoothing iterations for STL export (default: 5)",
    )
    output_group.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom name for the experiment (optional)",
    )
    output_group.add_argument(
        "--description",
        type=str,
        default=None,
        help="Description of the experiment (optional)",
    )

    # Animation parameters
    animation_group = parser.add_argument_group("Animation parameters")
    animation_group.add_argument(
        "--create-animation",
        action="store_true",
        help="Create a GIF animation of the optimization process",
    )
    animation_group.add_argument(
        "--animation-frequency",
        type=int,
        default=10,
        help="Store every N iterations for the animation (default: 10)",
    )
    animation_group.add_argument(
        "--animation-frames",
        type=int,
        default=50,
        help="Target number of frames to include in the animation (default: 50)",
    )
    animation_group.add_argument(
        "--animation-fps",
        type=int,
        default=5,
        help="Frames per second in the animation (default: 5)",
    )

    # Design space parameters
    design_space_group = parser.add_argument_group("Design space parameters")
    design_space_group.add_argument(
        "--design-space-stl",
        type=str,
        help="Path to an STL file defining the design space geometry",
    )
    design_space_group.add_argument(
        "--pitch",
        type=float,
        default=1.0,
        help="Distance between voxel centers when voxelizing STL (smaller values create finer detail)",
    )
    design_space_group.add_argument(
        "--invert-design-space",
        action="store_true",
        help="Invert the design space (treat STL as void space rather than design space)",
    )

    # Obstacle related arguments
    obstacle_group = parser.add_argument_group("Obstacle parameters")
    obstacle_group.add_argument(
        "--obstacle-config", type=str, help="Path to a JSON file defining obstacles"
    )

    # Logging parameters
    log_group = parser.add_argument_group("Logging parameters")
    log_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    log_group.add_argument("--log-file", type=str, default=None, help="Log file path")
    log_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (DEBUG level)",
    )
    log_group.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output (WARNING level)"
    )

    return parser.parse_args(args)


def generate_experiment_name(args: argparse.Namespace) -> str:
    """
    Generate an experiment name from command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    str
        Generated experiment name.
    """
    if args.experiment_name:
        return args.experiment_name

    dims = f"{args.nelx}x{args.nely}x{args.nelz}"

    # Include obstacle info in experiment name
    obstacle_type = "no_obstacle"
    if args.obstacle_config:
        obstacle_type = os.path.basename(args.obstacle_config).replace(".json", "")

    # Include design space STL info in experiment name if provided
    design_space = ""
    if hasattr(args, "design_space_stl") and args.design_space_stl:
        stl_name = os.path.basename(args.design_space_stl).replace(".stl", "")
        pitch_info = f"_p{args.pitch}".replace(".", "p")
        design_space = f"_ds_{stl_name}{pitch_info}"

    return f"{dims}_{obstacle_type}{design_space}"


def create_config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Create a configuration dictionary from command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    """
    config = vars(args)
    config["timestamp"] = datetime.now().isoformat()
    return config
