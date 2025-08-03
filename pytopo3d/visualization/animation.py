"""
Animation utilities for 3D topology optimization.

This module contains functions for creating animations of the optimization process.
"""

import logging
import os
from typing import List, Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np

from pytopo3d.utils.results_manager import ResultsManager
from pytopo3d.visualization.display import display_3D


def create_frame(
    density: np.ndarray,
    obstacle_mask: Optional[np.ndarray] = None,
    loads_array: Optional[np.ndarray] = None,
    constraints_array: Optional[np.ndarray] = None,
    disp_thres: float = 0.5,
    iteration: int = 0,
    compliance: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Create a frame for the optimization animation.

    Parameters
    ----------
    density : ndarray
        Current density distribution.
    obstacle_mask : ndarray, optional
        Boolean array marking obstacle regions.
    loads_array : ndarray, optional
        Array representing load positions.
    constraints_array : ndarray, optional
        Array representing constraint positions.
    disp_thres : float, default=0.5
        Display threshold for the density array.
    iteration : int, default=0
        Current iteration number (for title).
    compliance : float, optional
        Current compliance value (for title).
    figsize : tuple of int, default=(10, 8)
        Figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    plt.figure(figsize=figsize)

    # Prepare arrays and visualization parameters
    arrays = [density]
    thresholds = [disp_thres]
    colors = ["gray"]
    labels = ["Design"]
    alphas = [0.9]

    # Add obstacles if provided
    if obstacle_mask is not None:
        obstacle_array = obstacle_mask.astype(float)
        arrays.append(obstacle_array)
        thresholds.append(0.5)
        colors.append("yellow")
        labels.append("Obstacles")
        alphas.append(0.3)  # Make obstacles transparent

    # Add loads if provided
    if loads_array is not None:
        arrays.append(loads_array)
        thresholds.append(0.5)
        colors.append("blue")
        labels.append("Loads")
        alphas.append(0.9)

    # Add constraints if provided
    if constraints_array is not None:
        arrays.append(constraints_array)
        thresholds.append(0.5)
        colors.append("red")
        labels.append("Constraints")
        alphas.append(0.9)

    # Create the 3D visualization
    fig = display_3D(
        densities=arrays,
        thresholds=thresholds,
        colors=colors,
        labels=labels,
        alphas=alphas,
    )

    # Add title with iteration and compliance info
    title = f"Iteration: {iteration}"
    if compliance is not None:
        title += f" | Compliance: {compliance:.2f}"
    plt.title(title)

    return fig


def save_optimization_gif(
    frames: List[np.ndarray],
    obstacle_mask: Optional[np.ndarray] = None,
    loads_array: Optional[np.ndarray] = None,
    constraints_array: Optional[np.ndarray] = None,
    compliances: Optional[List[float]] = None,
    disp_thres: float = 0.5,
    results_mgr: ResultsManager = None,
    filename: str = "optimization_process",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
    fps: int = 5,
    every_n_iterations: int = 1,
) -> str:
    """
    Create and save a GIF animation of the optimization process.

    Parameters
    ----------
    frames : list of ndarray
        Density distributions at different iterations.
    obstacle_mask : ndarray, optional
        Boolean array marking obstacle regions.
    loads_array : ndarray, optional
        Array representing load positions.
    constraints_array : ndarray, optional
        Array representing constraint positions.
    compliances : list of float, optional
        Compliance values for each iteration.
    disp_thres : float, default=0.5
        Display threshold for density arrays.
    results_mgr : ResultsManager, default=None
        Results manager for saving the animation.
    filename : str, default="optimization_process"
        Base filename for the animation.
    figsize : tuple of int, default=(10, 8)
        Figure size (width, height) in inches.
    dpi : int, default=100
        DPI for rendering frames.
    fps : int, default=5
        Frames per second in the animation.
    every_n_iterations : int, default=1
        Include every n-th frame to control animation length.

    Returns
    -------
    str
        Path to the saved GIF.
    """
    logger = logging.getLogger(__name__)

    if results_mgr is None:
        raise ValueError("ResultsManager must be provided to save the GIF")

    # Create the visualizations directory if it doesn't exist
    viz_dir = os.path.join(results_mgr.experiment_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Create the frames directory for temporary frame storage
    frames_dir = os.path.join(viz_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Prepare for GIF creation
    png_paths = []

    # Use fewer frames for efficiency by selecting every n-th frame
    selected_frames = frames[::every_n_iterations]
    selected_iterations = list(range(0, len(frames), every_n_iterations))

    # If compliances are provided, select the corresponding ones
    selected_compliances = None
    if compliances is not None:
        selected_compliances = compliances[::every_n_iterations]

    # Create and save each frame
    logger.info(f"Creating {len(selected_frames)} frames for animation...")
    for i, (density, iteration) in enumerate(zip(selected_frames, selected_iterations)):
        compliance = (
            selected_compliances[i] if selected_compliances is not None else None
        )

        # Create the frame
        fig = create_frame(
            density=density,
            obstacle_mask=obstacle_mask,
            loads_array=loads_array,
            constraints_array=constraints_array,
            disp_thres=disp_thres,
            iteration=iteration,
            compliance=compliance,
            figsize=figsize,
        )

        # Save the frame as a PNG file
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        png_paths.append(frame_path)

        # Close the figure to free memory
        plt.close(fig)

    # Create the GIF
    gif_path = os.path.join(viz_dir, f"{filename}.gif")

    try:
        logger.info("Trying alternative method for GIF creation...")
        # Try using PIL directly
        from PIL import Image

        images = []
        for png_path in png_paths:
            if os.path.exists(png_path):
                try:
                    img = Image.open(png_path)
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Could not open image {png_path} with PIL: {e}")

        if images:
            # Save as GIF using PIL
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                optimize=False,
                duration=int(1000 / fps),
                loop=0,
            )
            logger.info(f"GIF animation saved using PIL to {gif_path}")
        else:
            logger.error("No valid frame images found for GIF creation")
    except Exception as e2:
        logger.error(f"GIF creation method using PIL failed: {e2}")

    # Optionally clean up the temporary PNG files
    for png_path in png_paths:
        try:
            os.remove(png_path)
        except Exception as e:
            logger.warning(f"Could not remove temporary file {png_path}: {e}")

    # Try to remove the frames directory if it's empty
    try:
        if os.path.exists(frames_dir) and not os.listdir(frames_dir):
            os.rmdir(frames_dir)
    except Exception as e:
        logger.warning(f"Could not remove frames directory: {e}")

    return gif_path
