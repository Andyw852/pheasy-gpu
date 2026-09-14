"""Resident RIDGE must work on FLOAT32 factors.

Regression test for the bug that killed PHEASY_GPU_RIDGE_RESIDENT=1 twice on the
c6.5/c3=4.5 fit with

    CUDA error: operation not supported when calling cusparseSpMM_bufferSize(...)

_iterative_ridge_tensor wraps the operator in a local Augmented class.  That class
did not carry _value_dtype, so _iterative_lstsq_tensor (and _operator_norm_estimate,
which norm_estimate() calls before the loop) fell back to torch.float64 while
GpuTwoLevelOperator keeps float32 factors for an fp32 sensing matrix.  Every
A.matvec() then became torch.sparse.mm(float32_csr, float64_dense), which cuSPARSE
rejects as a mixed-dtype SpMM.  Same omission as GpuSubsetOperator._value_dtype.

Runs on any CUDA host; no dataset needed.
"""
import os
import unittest

import numpy as np
import scipy.sparse as sp
from types import SimpleNamespace

import torch
from core import gpu_backend as gb


@unittest.skipUnless(gb.available(), "CUDA unavailable")
class ResidentRidgeFloat32Test(unittest.TestCase):
    def test_augmented_operator_carries_factor_dtype(self):
        os.environ["PHEASY_SM_DTYPE"] = "float32"
        rng = np.random.default_rng(0)
        nrow, nmid, ncol = 400, 60, 30
        SMp = sp.random(nrow, nmid, density=0.06, format="csr",
                        random_state=1, dtype=np.float32)
        NS = sp.random(nmid, ncol, density=0.20, format="csr",
                       random_state=2, dtype=np.float32)
        factors = SimpleNamespace(SM_prime=SMp, NS=NS, shape=(nrow, ncol))

        X = np.asarray((SMp @ NS).todense(), dtype=np.float64)
        c_true = rng.standard_normal(ncol)
        y = X @ c_true + 0.01 * rng.standard_normal(nrow)
        alpha = 1e-2
        c_ref = np.linalg.solve(X.T @ X + alpha * np.eye(ncol), X.T @ y)

        op = gb.GpuTwoLevelOperator(factors)
        try:
            self.assertEqual(op._value_dtype, torch.float32)
            # Would raise RuntimeError("... not supported ...") before the fix.
            coef, info = gb.iterative_ridge(op, y, alpha, atol=1e-10,
                                            btol=1e-10, maxiter=2000)
        finally:
            op.close()
        self.assertEqual(info["backend"], "gpu_resident_iterative")
        rel = float(np.linalg.norm(coef - c_ref) / np.linalg.norm(c_ref))
        self.assertLess(rel, 1e-3, "resident ridge disagrees with the dense "
                        "reference by %.3e" % rel)


if __name__ == "__main__":
    unittest.main()
