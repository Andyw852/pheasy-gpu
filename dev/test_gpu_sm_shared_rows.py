"""GPU-SM CV slices must share the uploaded parent instead of allocating folds."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse as sp
from core import optimizer as om

class SharedGpuRowsTest(unittest.TestCase):
    def test_gpu_rows_share_parent_and_preserve_adjoint(self):
        rng = np.random.default_rng(14)
        prime = rng.normal(size=(12, 7))
        ns = rng.normal(size=(7, 4))
        with patch.dict("os.environ", {"PHEASY_GPU_SM": "0", "PHEASY_TWOLEVEL_CACHE_T": "0"}):
            parent = om.TwoLevelSM(sp.csr_matrix(prime), sp.csr_matrix(ns))
        class FakeGpu:
            def matvec(self, x): return prime @ x
            def rmatvec(self, x): return prime.T @ x
        parent._gpu_mv = FakeGpu()
        rows = np.array([9, 1, 7, 9])
        with patch.object(om, "TwoLevelSM", side_effect=AssertionError("fold reuploads matrix")):
            child = parent.row_slice(rows)
            nested = om._row_slice(child, [2, 0])
        x = rng.normal(size=4)
        u = rng.normal(size=4)
        expected = (prime @ ns)[rows]
        np.testing.assert_allclose(child @ x, expected @ x, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(child.rmatvec(u), expected.T @ u, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(nested @ x, expected[[2, 0]] @ x, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(u @ (child @ x)), float(x @ child.rmatvec(u)), places=11)

if __name__ == "__main__":
    unittest.main()
