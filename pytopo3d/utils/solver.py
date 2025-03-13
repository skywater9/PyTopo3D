"""
Solver configuration for topology optimization.

This module provides solver configuration, trying to use PyPardiso if available,
otherwise falling back to SciPy's spsolve.
"""

import os
import numpy as np
from scipy.sparse.linalg import spsolve as seq_solve

# Try to use PyPardiso; if not, fall back to SciPy's spsolve.
try:
    import psutil
    import pypardiso
    from pypardiso import spsolve as parallel_solve

    num_cores = os.cpu_count()
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Number of CPU cores: {num_cores}")
    print(f"Available RAM: {available_ram_gb:.2f} GB")

    solver = parallel_solve
    solver_name = "PyPardiso"
except ImportError as e:
    print(f"Error importing pypardiso: {e}. Falling back to SciPy spsolve.")
    solver = seq_solve
    solver_name = "SciPy spsolve"