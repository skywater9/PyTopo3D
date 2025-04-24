# solver.py  ───────── 2025-04-24 refactor

"""
GPU/CPU solver factory.
✅ GPU : CG + Jacobi(preconditioner)  (cupyx ≥ 13.0)
✅ CPU : PyPardiso -or- SciPy spsolve
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve as seq_solve

# ───────────────────────────────────────────────────────────────────── CPU
solvers: Dict[str, Callable] = {
    "cpu": seq_solve,
    "cpu_name": "SciPy spsolve (single-core)",
}

try:
    import pypardiso

    solvers["cpu"] = pypardiso.spsolve
    solvers["cpu_name"] = "PyPardiso (multi-core)"
except ImportError:
    pass


# ───────────────────────────────────────────────────────────────────── GPU
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    from cupyx.scipy.sparse.linalg import (
        LinearOperator as LinearOperatorGPU,
    )
    from cupyx.scipy.sparse.linalg import (
        cg as cg_gpu,
    )

    def _gpu_cg_solver(
        A: Union[cusp.csr_matrix, csr_matrix], b: Union[cp.ndarray, np.ndarray]
    ):
        """Symmetric CG with Jacobi preconditioner."""
        if isinstance(A, csr_matrix):  # 히든 복사 방지
            A = cusp.csr_matrix(A)

        if isinstance(b, np.ndarray):
            b = cp.asarray(b)

        diag_M = 1.0 / (A.diagonal() + 1e-20)  # Jacobi
        M = LinearOperatorGPU(A.shape, matvec=lambda x: diag_M * x)

        x, info = cg_gpu(A, b, M=M, maxiter=2000, atol=0.0, tol=1e-6)
        if info != 0:
            raise RuntimeError(f"CG failed to converge: info={info}")
        return x

    solvers["gpu"] = _gpu_cg_solver
    solvers["gpu_name"] = "CuPy CG + Jacobi (GPU)"

except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────── API
def get_solver(use_gpu: bool = False) -> Tuple[Callable, str]:
    """Return (solver_func, solver_name)."""
    if use_gpu and "gpu" in solvers:
        return solvers["gpu"], solvers["gpu_name"]
    return solvers["cpu"], solvers["cpu_name"]
