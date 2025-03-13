#!/usr/bin/env python3
"""
Results manager for the topology optimization package.

This module provides a class for managing experiment output organization

```pytopo3d/utils/results_manager.py
<code_block_to_apply_changes_from>
"""

import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd


class ResultsManager:
    """
    Manages the storage and retrieval of experiment results.

    This class creates a structured directory for each experiment and
    provides methods for saving configurations, results, visualizations,
    and exports.
    """

    def __init__(self, base_dir="results", experiment_name=None, description=None):
        """
        Initialize a results manager.

        Parameters
        ----------
        base_dir : str
            Base directory for all results
        experiment_name : str
            Name of the experiment (default: auto-generated using timestamp)
        description : str
            Description of the experiment
        """
        # Create a unique experiment directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.timestamp = timestamp
        self.experiment_name = experiment_name or f"experiment_{timestamp}"
        self.description = description

        # Create the full experiment path
        self.experiment_dir = os.path.join(base_dir, self.experiment_name)

        # Create directory structure
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "exports"), exist_ok=True)

        # Initialize metrics dictionary
        self.metrics = {
            "timestamp": timestamp,
            "experiment_name": self.experiment_name,
            "description": description,
        }

    def save_config(self, config_dict):
        """
        Save configuration parameters.

        Parameters
        ----------
        config_dict : dict
            Dictionary of configuration parameters
        """
        # Add timestamp if not present
        if "timestamp" not in config_dict:
            config_dict["timestamp"] = self.timestamp

        # Add description if available
        if self.description and "description" not in config_dict:
            config_dict["description"] = self.description

        # Save to JSON file
        with open(os.path.join(self.experiment_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        return os.path.join(self.experiment_dir, "config.json")

    def save_result(self, result_array, filename="optimized_design.npy"):
        """
        Save the primary result array.

        Parameters
        ----------
        result_array : numpy.ndarray
            The result array to save
        filename : str
            Filename to save as (default: optimized_design.npy)

        Returns
        -------
        str
            Path to the saved file
        """
        filepath = os.path.join(self.experiment_dir, filename)
        np.save(filepath, result_array)
        return filepath

    def save_visualization(self, fig, name, format="png", **kwargs):
        """
        Save a visualization.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure or matplotlib.figure.Figure
            The figure to save
        name : str
            Name of the visualization
        format : str
            File format (default: png)
        **kwargs : dict
            Additional parameters to pass to write_image or savefig

        Returns
        -------
        str
            Path to the saved file
        """
        import copy
        from typing import Any, Union

        viz_path = os.path.join(
            self.experiment_dir, "visualizations", f"{name}.{format}"
        )

        # Create a copy of the figure to avoid modifying the original
        fig_copy: Union[Any, None] = None

        # Handle different figure types
        if hasattr(fig, "write_image"):
            # Plotly figure - reduce margins
            fig_copy = copy.deepcopy(fig)

            # Update layout with tighter margins if not already set
            if "margin" not in fig_copy.layout:
                fig_copy.update_layout(
                    margin=dict(l=5, r=5, t=5, b=5, pad=0), autosize=True
                )

            fig_copy.write_image(viz_path, **kwargs)
        elif hasattr(fig, "savefig"):
            # Matplotlib figure - use tight_layout
            fig_copy = copy.deepcopy(fig)

            # Apply tight layout if not already applied
            if not hasattr(fig_copy, "_tight") or not fig_copy._tight:
                fig_copy.tight_layout()

            # Ensure bbox_inches='tight' is used for saving
            if "bbox_inches" not in kwargs:
                kwargs["bbox_inches"] = "tight"

            fig_copy.savefig(viz_path, **kwargs)
        else:
            raise TypeError(
                "Unsupported figure type. Must be plotly.Figure or matplotlib.Figure"
            )

        return viz_path

    def save_export(self, export_object, name, format="stl"):
        """
        Save an exported file.

        Parameters
        ----------
        export_object : object
            Object with an export method (e.g., trimesh.Trimesh)
        name : str
            Name of the export
        format : str
            File format (default: stl)

        Returns
        -------
        str
            Path to the saved file
        """
        export_path = os.path.join(self.experiment_dir, "exports", f"{name}.{format}")

        # Handle different export object types
        if hasattr(export_object, "export"):
            export_object.export(export_path)
        else:
            raise TypeError(
                "Unsupported export object type. Must have an export method."
            )

        return export_path

    def update_metrics(self, metrics_dict):
        """
        Update metrics for the experiment.

        Parameters
        ----------
        metrics_dict : dict
            Dictionary of metrics to update
        """
        self.metrics.update(metrics_dict)

        # Save to JSON file
        with open(os.path.join(self.experiment_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)

        return os.path.join(self.experiment_dir, "metrics.json")

    def copy_file(self, source_path, destination_name=None):
        """
        Copy a file to the experiment directory.

        Parameters
        ----------
        source_path : str
            Path to the source file
        destination_name : str
            Name to save the file as (default: same as source)

        Returns
        -------
        str
            Path to the copied file
        """
        destination_name = destination_name or os.path.basename(source_path)
        destination_path = os.path.join(self.experiment_dir, destination_name)
        shutil.copy2(source_path, destination_path)
        return destination_path

    @staticmethod
    def list_experiments(base_dir="results"):
        """
        List all experiments in the base directory.

        Parameters
        ----------
        base_dir : str
            Base directory for all results

        Returns
        -------
        list
            List of experiment directories
        """
        if not os.path.exists(base_dir):
            return []

        return [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]

    @staticmethod
    def load_experiment_config(experiment_dir):
        """
        Load configuration for an experiment.

        Parameters
        ----------
        experiment_dir : str
            Path to the experiment directory

        Returns
        -------
        dict
            Configuration dictionary
        """
        config_path = os.path.join(experiment_dir, "config.json")
        if not os.path.exists(config_path):
            return {}

        with open(config_path, "r") as f:
            return json.load(f)

    @staticmethod
    def create_experiments_database(base_dir="results", output_file="experiments.csv"):
        """
        Create a CSV database of all experiments.

        Parameters
        ----------
        base_dir : str
            Base directory for all results
        output_file : str
            Path to save the CSV file

        Returns
        -------
        pandas.DataFrame
            DataFrame of all experiments
        """
        experiments = ResultsManager.list_experiments(base_dir)

        # Initialize list to store experiment data
        experiment_data = []

        for exp in experiments:
            exp_dir = os.path.join(base_dir, exp)
            config = ResultsManager.load_experiment_config(exp_dir)

            # Load metrics if available
            metrics_path = os.path.join(exp_dir, "metrics.json")
            metrics = {}
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)

            # Combine config and metrics
            data = {**config, **metrics, "experiment_dir": exp_dir}
            experiment_data.append(data)

        # Create DataFrame
        if not experiment_data:
            # Create empty dataframe with some expected columns
            df = pd.DataFrame(
                columns=[
                    "experiment_name",
                    "timestamp",
                    "description",
                    "experiment_dir",
                ]
            )
        else:
            df = pd.DataFrame(experiment_data)

        # Save to CSV
        df.to_csv(output_file, index=False)

        return df
