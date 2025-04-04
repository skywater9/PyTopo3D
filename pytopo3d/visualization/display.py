"""
Visualization utilities for 3D topology optimization.

This module contains functions for displaying the 3D structure.
"""

from typing import Any, List, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray


class Arrow3D(FancyArrowPatch):
    """
    A 3D arrow for use in matplotlib 3D plots.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def _add_axis_legend(ax, fig):
    """
    Add axis direction indicators as a separate legend/annotation in the corner
    of the figure rather than on the plot itself.
    
    Parameters
    ----------
    ax : Axes3D
        The 3D axes to add the indicators to
    fig : Figure
        The figure object
    """
    # Create a small axis for the legend in the bottom right corner
    # Use a smaller area and position it to not interfere with the title
    legend_ax = fig.add_axes([0.75, 0.05, 0.2, 0.2], projection='3d')
    
    # Create short arrows for X, Y, Z with equal length
    # Start from origin point
    origin = np.array([0, 0, 0])
    
    # Create arrows for each axis with same length
    arrow_length = 0.8
    
    # X-axis (Red)
    x_arrow = Arrow3D([0, arrow_length], [0, 0], [0, 0], 
                      mutation_scale=15, lw=2, arrowstyle='-|>', color='red')
    legend_ax.add_artist(x_arrow)
    legend_ax.text(arrow_length + 0.1, 0, 0, "X", color='red', fontsize=12, fontweight='bold')
    
    # Z-axis (Blue) - Swapped with Y
    z_arrow = Arrow3D([0, 0], [0, arrow_length], [0, 0], 
                      mutation_scale=15, lw=2, arrowstyle='-|>', color='blue')
    legend_ax.add_artist(z_arrow)
    legend_ax.text(0, arrow_length + 0.1, 0, "Z", color='blue', fontsize=12, fontweight='bold')
    
    # Y-axis (Green) - Swapped with Z
    y_arrow = Arrow3D([0, 0], [0, 0], [0, arrow_length], 
                      mutation_scale=15, lw=2, arrowstyle='-|>', color='green')
    legend_ax.add_artist(y_arrow)
    legend_ax.text(0, 0, arrow_length + 0.1, "Y", color='green', fontsize=12, fontweight='bold')
    
    # Set axis limits and view angle for the legend
    legend_ax.set_xlim([-0.2, 1.2])
    legend_ax.set_ylim([-0.2, 1.2])
    legend_ax.set_zlim([-0.2, 1.2])
    
    # Use the same view angle as the main plot
    elev, azim = ax.elev, ax.azim
    legend_ax.view_init(elev=elev, azim=azim)
    
    # Hide spines, ticks, and labels
    legend_ax.set_axis_off()
    
    return legend_ax


def display_3D(
    densities: Union[NDArray[np.float64], List[NDArray[np.float64]]],
    thresholds: Union[float, List[float]] = 0.5,
    colors: Union[str, List[str]] = "#000000",
    common_threshold: Optional[float] = None,
    labels: Optional[Union[str, List[str]]] = None,
    alphas: Union[float, List[float]] = 0.9,
) -> Figure:
    """
    Display multiple 3D structures using Poly3DCollection with color gradients.

    Parameters
    ----------
    densities : ndarray or list of ndarray
        3D array or list of 3D arrays of element densities.
        All arrays must have the same shape.
        To visualize loads, include a density array where elements at load positions have values > 0.
        To visualize constraints, include a density array where elements at constraint positions have values > 0.
    thresholds : float or list of float, default=0.5
        Display thresholds for each density array. Elements with density > threshold are displayed.
        If a single float is provided and densities is a list, it will be used for all arrays.
    colors : str or list of str, default='#000000'
        Colors for each density array. Can be string names (like 'red') or hex codes.
        If a single color is provided and densities is a list, it will be used for all arrays.
        Use 'blue' for loads and 'red' for constraints for consistency.
    common_threshold : float, optional
        If provided, uses a single shared threshold for all arrays.
    labels : str or list of str, optional
        Labels for each array in the legend. If not provided, default labels will be used.
    alphas : float or list of float, default=0.9
        Transparency levels for each array (0.0 = fully transparent, 1.0 = fully opaque).
        If a single float is provided and densities is a list, it will be used for all arrays.
        Use lower values for obstacles/non-design spaces (e.g., 0.3) and higher values for
        the design (e.g., 0.9) for better visualization.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    # Convert single array input to list
    if not isinstance(densities, list):
        densities = [densities]

    # Handle thresholds
    if not isinstance(thresholds, list):
        thresholds = [thresholds] * len(densities)
    if common_threshold is not None:
        thresholds = [common_threshold] * len(densities)

    # Handle colors
    if not isinstance(colors, list):
        colors = [colors] * len(densities)

    # Handle alphas (transparency)
    if not isinstance(alphas, list):
        alphas = [alphas] * len(densities)
    if len(alphas) < len(densities):
        alphas.extend([0.9] * (len(densities) - len(alphas)))

    # Handle labels
    if labels is None:
        labels = [f"Array {i + 1}" for i in range(len(densities))]
    elif not isinstance(labels, list):
        labels = [labels]

    if len(labels) < len(densities):
        labels.extend([f"Array {i + 1}" for i in range(len(labels), len(densities))])

    # Ensure all inputs have the same length
    n_arrays = len(densities)
    if (
        len(thresholds) != n_arrays
        or len(colors) != n_arrays
        or len(alphas) != n_arrays
    ):
        raise ValueError(
            "densities, thresholds, colors, and alphas must have the same length"
        )

    # Check that all arrays have the same shape
    shape = densities[0].shape
    for i, density in enumerate(densities):
        if density.shape != shape:
            raise ValueError(
                f"All density arrays must have the same shape. Array {i} has shape {density.shape} but expected {shape}"
            )

    # Extract shape dimensions
    nely, nelx, nelz = shape

    # Create color maps for each color
    color_maps = []
    for color in colors:
        # Create a colormap from white to the specified color
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"gradient_{color}", ["#FFFFFF", color], N=256
        )
        color_maps.append(cmap)

    fig = plt.gcf()
    
    # Make the main axes slightly smaller to accommodate the axis legend
    # without affecting the title position
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((nelx, nelz, nely))

    faces = np.array(
        [
            [0, 1, 2, 3],
            [1, 5, 6, 2],
            [3, 2, 6, 7],
            [0, 4, 5, 1],
            [0, 3, 7, 4],
            [4, 7, 6, 5],
        ]
    )
    local_verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )

    poly_list: List[List[List[float]]] = []
    face_colors: List[Any] = []
    face_alphas: List[float] = []

    # Store actual density values for gradients
    density_values: List[float] = []
    array_indices: List[int] = []

    # For each voxel, determine which array has the highest normalized density
    # and use that array's color
    for k in range(nelz):
        for i in range(nelx):
            for j in range(nely):
                # Check if any of the arrays has density > threshold
                display_voxel = False
                array_idx = -1
                max_normalized_density = -1.0

                for idx, (density, threshold) in enumerate(zip(densities, thresholds)):
                    if density[j, i, k] > threshold:
                        # Calculate normalized density (how far above threshold)
                        norm_density = (density[j, i, k] - threshold) / (
                            1.0 - threshold
                        )
                        if norm_density > max_normalized_density:
                            max_normalized_density = norm_density
                            array_idx = idx
                            display_voxel = True

                if display_voxel:
                    base_x = i
                    base_y = k
                    base_z = nely - 1 - j
                    verts_global = local_verts + [base_x, base_y, base_z]
                    for f in faces:
                        poly_verts = verts_global[f].tolist()
                        poly_list.append(poly_verts)

                        # Store the density value, array index, and alpha for coloring and transparency
                        density_value = densities[array_idx][j, i, k]
                        density_values.append(density_value)
                        array_indices.append(array_idx)
                        face_alphas.append(alphas[array_idx])

    # Create face colors based on density values and array indices
    for density_val, array_idx, alpha in zip(
        density_values, array_indices, face_alphas
    ):
        threshold = thresholds[array_idx]
        # Normalize density between 0 and 1 (from threshold to 1.0)
        norm_density = min(1.0, max(0.0, (density_val - threshold) / (1.0 - threshold)))
        # Use the colormap to get a color based on the normalized density
        rgba_color = list(color_maps[array_idx](norm_density))
        # Set the alpha component (the 4th value in RGBA)
        rgba_color[3] = alpha
        face_colors.append(rgba_color)

    pc = Poly3DCollection(
        poly_list,
        facecolors=face_colors,
        edgecolors="k",  # Black edges
        linewidths=0.1,
    )

    ax.add_collection3d(pc)
    ax.set_xlim(0, nelx)
    ax.set_ylim(0, nelz)  # Y is now depth (z)
    ax.set_zlim(0, nely)  # Z is now height (y)

    # Set appropriate axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y (Depth)")
    ax.set_zlabel("Z (Height)")

    # Isometric view
    ax.view_init(elev=30, azim=30)
    
    # Add color legend patches with custom gradient swatches
    legend_elements = []

    # Add design elements to legend
    for i, (color, label, alpha) in enumerate(zip(colors, labels, alphas)):
        # Create a rectangle with the specified color and alpha
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha, label=label)
        )

    # Only show legend if we have multiple arrays
    if len(densities) > 1:
        ax.legend(handles=legend_elements, loc="upper right", fontsize="small")

    # Add colorbar for single array visualizations
    if len(densities) == 1:
        sm = plt.cm.ScalarMappable(
            cmap=color_maps[0], norm=plt.Normalize(thresholds[0], 1.0)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label(f"Density ({labels[0]})")

    # Add axis direction indicators as a legend AFTER other elements
    _add_axis_legend(ax, fig)
    
    plt.axis("off")

    return fig
