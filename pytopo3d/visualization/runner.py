"""
Visualization execution utilities for 3D topology optimization.

This module contains functions for creating and saving visualizations.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from pytopo3d.utils.results_manager import ResultsManager
from pytopo3d.visualization.display import display_3D


def create_visualization(
    arrays: Union[np.ndarray, List[np.ndarray]],
    thresholds: Union[float, List[float]],
    colors: Union[str, List[str]],
    labels: Union[str, List[str]],
    experiment_name: str,
    results_mgr: ResultsManager,
    filename: str,
    title: Optional[str] = None,
    format: str = "png",
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8),
    alphas: Union[float, List[float]] = 0.9,
) -> str:
    """
    Create and save a visualization using the display_3D function.

    This is a general-purpose visualization function that can handle all visualization tasks
    by configuring its parameters appropriately.

    Parameters
    ----------
    arrays : ndarray or list of ndarray
        3D array(s) to visualize. For combined visualizations, provide a list of arrays.
    thresholds : float or list of float
        Threshold(s) for each array.
    colors : str or list of str
        Color(s) for each array.
    labels : str or list of str
        Label(s) for each array.
    experiment_name : str
        Name of the experiment (used in the title).
    results_mgr : ResultsManager
        Results manager for saving the visualization.
    filename : str
        Base filename for saving the visualization (without extension).
    title : str, optional
        Custom title. If None, a default title will be generated based on the filename and experiment_name.
    format : str, default="png"
        File format for saving the visualization.
    dpi : int, default=300
        DPI for saving the visualization.
    figsize : tuple of int, default=(10, 8)
        Figure size (width, height) in inches.
    alphas : float or list of float, default=0.9
        Transparency levels for each array (0.0 = fully transparent, 1.0 = fully opaque).
        Use lower values for obstacles/non-design spaces (e.g., 0.3) and higher values for
        the design (e.g., 0.9) for better visualization.

    Returns
    -------
    str
        Path where the visualization was saved.

    Examples
    --------
    # Visualize optimized design
    viz_path = create_visualization(
        arrays=xPhys,
        thresholds=disp_thres,
        colors='gray',
        labels='Optimized Design',
        experiment_name=experiment_name,
        results_mgr=results_mgr,
        filename="final_topology"
    )

    # Visualize with boundary conditions and transparent obstacles
    viz_path = create_visualization(
        arrays=[xPhys, obstacle_array, loads_array, constraints_array],
        thresholds=[disp_thres, 0.5, 0.5, 0.5],
        colors=['gray', 'yellow', 'blue', 'red'],
        labels=['Optimized Design', 'Obstacles', 'Loads', 'Constraints'],
        alphas=[0.9, 0.3, 0.9, 0.9],  # Make obstacles transparent
        experiment_name=experiment_name,
        results_mgr=results_mgr,
        filename="topology_with_transparent_obstacles"
    )

    # Visualize combined view with design, obstacles, loads, and constraints
    combined_viz_path = create_visualization(
        arrays=[design_only, obstacle_array, loads_array, constraints_array],
        thresholds=[disp_thres, 0.5, 0.5, 0.5],
        colors=['gray', 'yellow', 'blue', 'red'],
        labels=['Optimized Design', 'Obstacles', 'Loads', 'Constraints'],
        experiment_name=experiment_name,
        results_mgr=results_mgr,
        filename="combined_view"
    )
    """
    plt.figure(figsize=figsize)

    # Create visualization
    fig = display_3D(
        densities=arrays,
        thresholds=thresholds,
        colors=colors,
        labels=labels,
        alphas=alphas,
    )

    # Set title
    if title is None:
        # Convert filename with underscores to a more readable title
        readable_name = " ".join(
            word.capitalize() for word in filename.replace("_", " ").split()
        )
        title = f"{readable_name} - {experiment_name}"

    plt.title(title)

    # Save visualization
    viz_path = results_mgr.save_visualization(fig, filename, format=format, dpi=dpi)

    # Close figure to free memory
    plt.close()

    return viz_path
