#!/usr/bin/env python3
"""Small deterministic regressions for large-matrix optimizer paths.

Run from any directory with the project's Python environment. No CUDA hardware
or material dataset is required; the GPU allocation failure is simulated.
"""
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_allclose
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import optimizer as opt


class OptimizerLargeRegressions(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "PHEASY_USE_GPU": "0", "PHEASY_GPU_SM": "0",
            "PHEASY_TWOLEVEL_CACHE_T": "0",
            "PHEASY_MAX_CORES": "1", "PHEASY_N_JOBS": "1",
            "PHEASY_RFE_1SE": "1", "PHEASY_RFE_PATIENCE": "20",
            "PHEASY_CV_GROUP_SIZE": "0", "PHEASY_TSQR_CRITERION": "cv",
            "PHEASY_LSQR_ATOL": "1e-12", "PHEASY_LSQR_BTOL": "1e-12",
            "PHEASY_LSQR_MAXITER": "1000",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_operator_ridge_subset_matches_dense_solution(self):
        rng = np.random.default_rng(17)
        A = rng.normal(size=(30, 7))
        y = rng.normal(size=30)
        rows = np.arange(0, 30, 2)
        cols = np.array([0, 2, 3, 6])
        alpha = 0.7
        subset = A[np.ix_(rows, cols)]
        expected = np.linalg.solve(
            subset.T @ subset + alpha * np.eye(len(cols)),
            subset.T @ y[rows])
        actual = opt._solve_subset(aslinearoperator(A), y, rows, cols,
                                   ridge_alpha=alpha)
        assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_rfe_evaluates_minimum_feature_candidate(self):
        rng = np.random.default_rng(18)
        A = rng.normal(size=(40, 5))
        y = np.zeros(40)
        for criterion in ("cv", "bic", "aic"):
            with self.subTest(criterion=criterion):
                model = opt.PheasyRFE_OLS_TSQR(
                    step=0.6, min_features=1, cv=4, patience=20,
                    n_jobs=1, verbose=False, random_state=0)
                model._criterion = criterion
                model.fit(A, y)
                self.assertEqual(int(model.support_.sum()), 1)

    def test_gpu_construction_failure_does_not_cache_host_transpose(self):
        fake_backend = types.ModuleType("pheasy_gpu.core.gpu_backend")

        def fail_gpu_allocation(_matrix):
            raise MemoryError("simulated insufficient VRAM")

        fake_backend.GpuSparseMV = fail_gpu_allocation
        fake_core = types.ModuleType("pheasy_gpu.core")
        fake_core.gpu_backend = fake_backend
        fake_package = types.ModuleType("pheasy_gpu")
        fake_package.core = fake_core
        modules = {"pheasy_gpu": fake_package, "pheasy_gpu.core": fake_core,
                   "pheasy_gpu.core.gpu_backend": fake_backend}
        prime = sp.csr_matrix([[1., 2., 0.], [0., 3., 4.]])
        ns = sp.csr_matrix([[2., 0.], [1., 3.], [0., 5.]])
        with patch.dict(os.environ, {"PHEASY_GPU_SM": "1",
                                     "PHEASY_TWOLEVEL_CACHE_T": "1"}), \
                patch.dict(sys.modules, modules):
            A = opt.TwoLevelSM(prime, ns)
        expected = (prime @ ns).toarray()
        assert_allclose(A @ np.array([2., 3.]), expected @ [2., 3.])
        assert_allclose(A.T @ np.array([4., 5.]), expected.T @ [4., 5.])
        self.assertFalse(A._cache_T)
        self.assertIsNone(A._SMpT)
        self.assertIsNone(A._NST)

    def test_gpu_runtime_failure_closes_backend_before_cpu_fallback(self):
        prime = sp.csr_matrix([[1., 2.], [3., 4.]])
        ns = sp.eye(2, format="csr")
        for operation in ("matvec", "rmatvec"):
            with self.subTest(operation=operation):
                A = opt.TwoLevelSM(prime, ns)
                failing_gpu = Mock()
                getattr(failing_gpu, operation).side_effect = RuntimeError("GPU lost")
                A._gpu_mv = failing_gpu
                v = np.array([2., 3.])
                actual = getattr(A, operation)(v)
                expected = prime @ v if operation == "matvec" else prime.T @ v
                assert_allclose(actual, expected)
                failing_gpu.close.assert_called_once_with()
                self.assertIsNone(A._gpu_mv)
                self.assertFalse(A._cache_T)
                self.assertIsNone(A._SMpT)

    def test_iterative_subsolvers_warn_on_iteration_limit(self):
        rng = np.random.default_rng(21)
        dense = rng.normal(size=(30, 8))
        A, y = aslinearoperator(dense), rng.normal(size=30)
        with patch.dict(os.environ, {"PHEASY_LSQR_MAXITER": "1"}):
            calls = (
                lambda: opt._solve_sparse_lsqr(A, y),
                lambda: opt._solve_subset(A, y, None, np.arange(8)),
                lambda: opt._ridge_solve(A, y, 0.3),
            )
            for solve in calls:
                with self.assertWarnsRegex(RuntimeWarning, "istop=7, iterations=1"):
                    solve()

    def test_ols_results_expose_convergence_and_do_not_reuse_stale_info(self):
        rng = np.random.default_rng(22)
        dense, y = rng.normal(size=(30, 8)), rng.normal(size=30)
        model = opt.Optimizer("OLS", use_gpu=False)
        with patch.dict(os.environ, {"PHEASY_OLS_MAXITER": "1",
                                     "PHEASY_OLS_JACOBI": "0"}):
            with self.assertWarnsRegex(RuntimeWarning, "LSMR did not converge"):
                model.fit(aslinearoperator(dense), y)
        info = model.results["solver_info"]
        self.assertEqual(info["istop"], 7)
        self.assertEqual(info["itn"], 1)
        self.assertFalse(info["converged"])
        self.assertEqual(model.results["n_iter"], 1)
        model.fit(dense, y)
        self.assertNotIn("solver_info", model.results)
        with patch.dict(os.environ, {"PHEASY_MAX_DENSE": "0"}):
            model.fit(sp.csr_matrix(dense), y)
        self.assertEqual(model.results["solver_info"]["solver"], "LSQR")
        self.assertTrue(model.results["solver_info"]["converged"])
        assert_allclose(model.results["coef"], np.linalg.lstsq(dense, y,
                        rcond=None)[0], rtol=1e-8, atol=1e-10)

    def test_twolevel_adjoint_and_subset_equal_materialized_product(self):
        rng = np.random.default_rng(19)
        prime = sp.csr_matrix(rng.normal(size=(31, 11)))
        ns = sp.csr_matrix(rng.normal(size=(11, 7)))
        A = opt.TwoLevelSM(prime, ns)
        dense = (prime @ ns).toarray()
        v, u = rng.normal(size=7), rng.normal(size=31)
        assert_allclose(A @ v, dense @ v, rtol=1e-12, atol=1e-12)
        assert_allclose(A.T @ u, dense.T @ u, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(u @ (A @ v)), float((A.T @ u) @ v))
        assert_allclose(opt._col_norms(A), np.linalg.norm(dense, axis=0))
        rows, cols = np.arange(0, 31, 2), np.array([0, 2, 5])
        actual = opt._solve_subset(A, u, rows, cols)
        expected = np.linalg.lstsq(dense[np.ix_(rows, cols)], u[rows],
                                   rcond=None)[0]
        assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)

    def test_twolevel_column_norms_stream_products_with_cancellation(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                prime = sp.csr_matrix(np.array([
                    [1., 1., 3.], [2., 2., 1.], [3., 1., 0.],
                    [0., 0., 0.], [0.5, 1., -2.]], dtype=dtype))
                ns = sp.csr_matrix(np.array([
                    [1., 0., 1., 1e-20], [-1., 0., 2., -1e-20],
                    [0., 0., 3., 0.]], dtype=dtype))
                A = opt.TwoLevelSM(prime, ns)
                expected = np.linalg.norm(prime.toarray().astype(np.float64)
                                          @ ns.toarray().astype(np.float64), axis=0)
                with patch.object(A, "_matvec", side_effect=AssertionError(
                        "column norms must not call full matrix matvec")):
                    assert_allclose(A.col_norms(block_rows=1), expected,
                                    rtol=1e-14, atol=0.)
                    with patch.dict(os.environ, {"PHEASY_COL_NORM_BLOCK_BYTES": "32"}):
                        assert_allclose(opt._col_norms(A), expected,
                                        rtol=1e-14, atol=0.)
                self.assertEqual(A.col_norms()[1], 0.)
        for shape in ((0, 4), (3, 0)):
            A = opt.TwoLevelSM(sp.csr_matrix((shape[0], 2)),
                               sp.csr_matrix((2, shape[1])))
            assert_allclose(A.col_norms(), np.zeros(shape[1]))

    def test_jacobi_preconditioner_preserves_ridge_objective(self):
        rng = np.random.default_rng(23)
        dense = rng.normal(size=(50, 5)) * np.array([0.01, 0.1, 1., 10., 100.])
        y = rng.normal(size=50)
        ridge = 0.2
        expected = np.linalg.solve(dense.T @ dense + ridge * 50 * np.eye(5),
                                   dense.T @ y)
        model = opt.Optimizer("OLS", use_gpu=False)
        with patch.dict(os.environ, {"PHEASY_OLS_JACOBI": "1",
                                     "PHEASY_OLS_RIDGE": str(ridge),
                                     "PHEASY_OLS_ATOL": "1e-13",
                                     "PHEASY_OLS_BTOL": "1e-13",
                                     "PHEASY_OLS_MAXITER": "1000"}):
            model.fit(aslinearoperator(dense), y)
        assert_allclose(model.results["coef"], expected, rtol=1e-9, atol=1e-11)
        self.assertTrue(model.results["solver_info"]["converged"])

    def test_tsqr_matches_svd_for_tall_wide_and_deficient_inputs(self):
        rng = np.random.default_rng(20)
        for shape, duplicate in (((65, 7), False), ((9, 15), False),
                                 ((65, 7), True)):
            with self.subTest(shape=shape, duplicate=duplicate):
                A = rng.normal(size=shape)
                if duplicate:
                    A[:, -1] = A[:, 0]
                y = rng.normal(size=shape[0])
                actual = opt._solve_qr(A, y, block_rows=5)
                expected = np.linalg.lstsq(A, y, rcond=None)[0]
                assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)

    def test_jacobi_solver_failure_is_not_retried_unscaled(self):
        # A solve/audit exception is not a column-norm preparation failure.
        model = opt.Optimizer("OLS", use_gpu=False)
        A = aslinearoperator(np.eye(3))
        with patch.dict(os.environ, {"PHEASY_OLS_JACOBI": "1"}), \
                patch.object(opt, "_lsmr", side_effect=RuntimeError("solve failed")) as solve:
            with self.assertRaisesRegex(RuntimeError, "solve failed"):
                model._ols_lsmr(A, np.ones(3))
        self.assertEqual(solve.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
