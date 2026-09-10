#!/usr/bin/env python3
import unittest
import numpy as np
from scipy.linalg import lstsq
from unittest.mock import patch
from core import gpu_backend as gb

class TestGpuTSQR(unittest.TestCase):
    @unittest.skipUnless(gb.available(), "CUDA required")
    def test_parity_and_block_invariance(self):
        rng=np.random.default_rng(7); A=rng.normal(size=(23,4)); y=A@np.array([1.,-2.,.5,3.])
        ref=lstsq(A,y,cond=None,lapack_driver="gelsd")[0]
        for b in (5,7,23):
            got, info=gb.gpu_tsqr(A,y,b)
            np.testing.assert_allclose(got,ref,rtol=1e-11,atol=1e-11)
            self.assertEqual(info["backend"],"gpu_dense_tsqr")
    def test_memory_rejection_precedes_upload(self):
        with patch.object(gb, "enabled", return_value=True), patch.object(gb, "available", return_value=True), patch.object(gb, "available_memory_bytes", return_value=1), patch.object(gb, "_to_torch") as upload:
            with self.assertRaises(MemoryError):
                gb.gpu_tsqr(np.ones((12,3)), np.ones(12), 4)
            upload.assert_not_called()

    @unittest.skipUnless(gb.available(), "CUDA required")
    def test_rank_deficient_rejects(self):
        A=np.ones((12,3)); A[:,2]=A[:,1]
        with self.assertRaises(np.linalg.LinAlgError): gb.gpu_tsqr(A,np.arange(12.),4)
    @unittest.skipUnless(gb.available(), "CUDA required")
    def test_sparse_twolevel_blocks_without_full_densification(self):
        import scipy.sparse as sp
        from core.optimizer import TwoLevelSM
        rng = np.random.default_rng(42)
        prime = sp.csr_matrix(rng.normal(size=(29, 8)))
        ns = sp.csr_matrix(rng.normal(size=(8, 4)))
        dense = (prime @ ns).toarray()
        y = rng.normal(size=29)
        ref = lstsq(dense, y)[0]
        original = sp.csr_matrix.toarray
        seen = []
        def bounded(matrix, *args, **kwargs):
            self.assertLessEqual(matrix.shape[0], 7)
            seen.append(matrix.shape)
            return original(matrix, *args, **kwargs)
        for A, kind in ((sp.csr_matrix(dense), "sparse"), (TwoLevelSM(prime, ns), "twolevel")):
            with patch.object(sp.csr_matrix, "toarray", bounded):
                coef, info = gb.gpu_tsqr(A, y, block_rows=7)
            np.testing.assert_allclose(coef, ref, rtol=1e-11, atol=1e-11)
            self.assertEqual(info["backend"], "gpu_" + kind + "_tsqr")
            self.assertEqual(info["n_blocks"], 5)
        self.assertEqual(len(seen), 10)

    @unittest.skipUnless(gb.available(), "CUDA required")
    def test_public_dispatch_and_memory_fallback(self):
        import os
        import scipy.sparse as sp
        from core.optimizer import Optimizer, TwoLevelSM
        rng = np.random.default_rng(13)
        dense = rng.normal(size=(31, 4))
        y = rng.normal(size=31)
        for A, kind in ((sp.csr_matrix(dense), "sparse"), (TwoLevelSM(sp.csr_matrix(dense), sp.eye(4, format="csr")), "twolevel")):
            model = Optimizer("OLS", use_gpu=True)
            with patch.dict(os.environ, {"PHEASY_GPU_TSQR": "1", "PHEASY_TSQR_BLOCK_ROWS": "7", "PHEASY_GPU_OLS_RESIDENT": "0"}):
                model.fit(A, y)
                self.assertEqual(model.results["execution_backend"], "gpu_" + kind + "_tsqr")
                np.testing.assert_allclose(model.predict(A), dense @ lstsq(dense, y)[0], atol=1e-10)
                with patch.object(gb, "available_memory_bytes", return_value=1):
                    model.fit(A, y)
                self.assertIn("MemoryError", model.results["fallback_reason"])
                with patch.object(gb, "gpu_tsqr", side_effect=np.linalg.LinAlgError("rank deficiency")):
                    model.fit(A, y)
                self.assertIn("rank deficiency", model.results["fallback_reason"])
                self.assertTrue(model.results["execution_backend"].startswith("cpu_"))
                np.testing.assert_allclose(model.predict(A), dense @ lstsq(dense, y)[0], atol=1e-7)

    @unittest.skipUnless(gb.available(), "CUDA required")
    def test_sparse_formats_and_nonfinite_blocks(self):
        import scipy.sparse as sp
        rng = np.random.default_rng(81)
        dense = rng.normal(size=(17, 3))
        y = rng.normal(size=17)
        for fmt in (sp.coo_matrix, sp.dia_matrix, sp.csc_matrix, sp.csr_array):
            got, _ = gb.gpu_tsqr(fmt(dense), y, block_rows=4)
            np.testing.assert_allclose(got, lstsq(dense, y)[0], atol=1e-11)
        dense[-1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite A"):
            gb.gpu_tsqr(sp.coo_matrix(dense), y, block_rows=4)

    def test_sparse_memory_gate_precedes_format_conversion(self):
        import scipy.sparse as sp
        A = sp.coo_matrix(np.eye(5))
        with patch.object(gb, "enabled", return_value=True), patch.object(gb, "available", return_value=True), patch.object(gb, "available_memory_bytes", return_value=1), patch.object(sp.coo_matrix, "tocsr", side_effect=AssertionError("conversion before gate")):
            with self.assertRaises(MemoryError):
                gb.gpu_tsqr(A, np.ones(5), block_rows=3)

    @unittest.skipUnless(gb.available(), "CUDA required")
    def test_acceptance_helper_understands_direct_diagnostics(self):
        import os
        import scipy.sparse as sp
        from core.optimizer import Optimizer, TwoLevelSM
        from dev.validate_gpu_backends import _fit
        A = TwoLevelSM(sp.eye(12, format="csr"), sp.eye(12, format="csr"))
        with patch.dict(os.environ, {"PHEASY_GPU_TSQR": "1"}):
            coef, record = _fit(Optimizer, "OLS", A, np.arange(12.), True)
        self.assertEqual(record["solver_info"]["solver_kind"], "direct")
        self.assertNotIn("converged", record["solver_info"])
        np.testing.assert_allclose(coef, np.arange(12.), atol=1e-10)

if __name__ == "__main__": unittest.main()
