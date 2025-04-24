"""
Solver configuration for topology optimization.

This module provides solver configuration for various backends:
1. CuPy (GPU-accelerated on CUDA-capable GPUs)
2. PyPardiso (multi-core CPU acceleration)
3. SciPy's spsolve (single-core CPU, fallback)

The module detects available solvers but defaults to CPU solvers unless explicitly requested.
"""

import os
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve as seq_solve

# Dictionary to hold available solvers
solvers: Dict[str, Callable] = {
    "cpu": seq_solve,
    "cpu_name": "SciPy spsolve (Single-core CPU)",
}

# Try to import PyPardiso for CPU parallelism
try:
    import psutil
    import pypardiso
    from pypardiso import spsolve as parallel_solve

    num_cores = os.cpu_count()
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Number of CPU cores: {num_cores}")
    print(f"Available RAM: {available_ram_gb:.2f} GB")

    solvers["cpu"] = parallel_solve
    solvers["cpu_name"] = "PyPardiso (Multi-core CPU)"
except ImportError as e:
    print(f"Error importing pypardiso: {e}. Using SciPy spsolve for CPU calculations.")

# Try to detect CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import spsolve as gpu_spsolve

    # Get GPU information
    gpu_info = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = gpu_info["name"].decode("utf-8")
    gpu_memory_total = gpu_info["totalGlobalMem"] / (1024**3)  # Convert to GB
    print(f"CUDA GPU detected: {gpu_name}")
    print(f"GPU memory: {gpu_memory_total:.2f} GB")

    # Convert SciPy CSR -> CuPy CSR only once per iteration
    # This minimizes host-device transfers, which can be a bottleneck
    def cupy_solver(A_csr: csr_matrix, b_np: np.ndarray) -> np.ndarray:
        """
        Solve linear system Ax=b using CuPy's GPU-accelerated sparse solver.

        Args:
            A_csr: SciPy sparse CSR matrix
            b_np: NumPy array for right-hand side

        Returns:
            NumPy array containing solution vector
        """
        # Transfer data to GPU
        A_gpu = cp.sparse.csr_matrix(
            (
                cp.asarray(A_csr.data),
                cp.asarray(A_csr.indices),
                cp.asarray(A_csr.indptr),
            ),
            shape=A_csr.shape,
        )
        b_gpu = cp.asarray(b_np)

        # Solve on GPU
        x_gpu = gpu_spsolve(A_gpu, b_gpu)

        # Transfer result back to CPU
        return cp.asnumpy(x_gpu)

    solvers["gpu"] = cupy_solver
    solvers["gpu_name"] = "CuPy-cuSOLVER (GPU)"

except ImportError as e:
    print(f"Error importing CuPy: {e}. GPU acceleration not available.")

# Set default solver to CPU version
solver = solvers["cpu"]
solver_name = solvers["cpu_name"]


def get_solver(use_gpu: bool = False) -> Tuple[Callable, str]:
    """
    Get the appropriate solver based on user preference and availability.

    Args:
        use_gpu: Whether to use GPU acceleration if available

    Returns:
        Tuple containing the solver function and its name
    """
    if use_gpu and "gpu" in solvers:
        return solvers["gpu"], solvers["gpu_name"]
    return solvers["cpu"], solvers["cpu_name"]
