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

    def test_debias_end_to_end_matches_selected_support_ols(self):
        rng = np.random.default_rng(1907)
        dense = rng.normal(size=(48, 6))
        y = dense @ np.array([1., -.7, .3, 0., 0., 0.]) + rng.normal(size=48) * .002
        A = aslinearoperator(dense)
        original = opt.Optimizer._debias
        for method in ("LASSO", "ALASSO"):
            captured = {}
            def capture(model, matrix, target, coef):
                captured["before"] = coef.copy()
                return original(model, matrix, target, coef)
            with self.subTest(method=method), patch.dict(os.environ, {
                    "PHEASY_LASSO_DEBIAS": "1", "PHEASY_GRAM_MAX_GB": "0"}), \
                    patch.object(opt.Optimizer, "_debias", capture):
                model = opt.Optimizer(method, alpha=[.01, .1], alpha_auto=False,
                                      cv=2, tol=1e-8, max_iter=5000, use_gpu=False)
                model.fit(A, y)
                support = np.flatnonzero(captured["before"])
                expected = np.zeros(6)
                expected[support] = np.linalg.lstsq(dense[:, support], y, rcond=None)[0]
                actual = model.results["coef"]
                assert_allclose(actual, expected, atol=1e-8)
                assert_allclose(model.predict(A), dense @ actual, atol=1e-10)
                self.assertLessEqual(np.linalg.norm(dense @ actual - y),
                                     np.linalg.norm(dense @ captured["before"] - y) + 1e-10)

    def test_pre_debias_capture_preserves_physical_scale_and_clears_on_refit(self):
        dense = np.diag([1., 3., 10., 30.])
        A = aslinearoperator(dense)
        y = np.array([1., -2., 3., -4.])
        original = opt.Optimizer._debias
        for method in ("LASSO", "ALASSO"):
            for standardize in (False, True):
                captured = {}
                def capture(model, matrix, target, coef):
                    captured["prediction"] = matrix @ coef
                    return original(model, matrix, target, coef)
                model = opt.Optimizer(method, alpha=[.001, .01], alpha_auto=False,
                                      cv=2, standardize=standardize, use_gpu=False)
                with patch.dict(os.environ, {"PHEASY_LASSO_DEBIAS": "1"}), patch.object(opt.Optimizer, "_debias", capture):
                    model.fit(A, y)
                before = model.results["pre_debias_coef"]
                assert_allclose(dense @ before, captured["prediction"], atol=1e-10)
                self.assertFalse(np.shares_memory(before, model.results["coef"]))
                with patch.dict(os.environ, {"PHEASY_LASSO_DEBIAS": "0"}):
                    model.fit(A, y)
                self.assertNotIn("pre_debias_coef", model.results)

    def test_debias_rejects_worse_refit_and_skips_empty_support(self):
        A = aslinearoperator(np.eye(4))
        y = np.array([1., 0., 0., 0.])
        coef = np.array([.5, 0., 0., 0.])
        model = opt.Optimizer.__new__(opt.Optimizer)
        with patch.object(opt, "_solve_sparse_lsqr", return_value=np.array([10.])) as solve:
            assert_allclose(model._debias(A, y, coef), coef)
            solve.assert_called_once()
        with patch.object(opt, "_solve_sparse_lsqr") as solve:
            assert_allclose(model._debias(A, y, np.zeros(4)), np.zeros(4))
            solve.assert_not_called()

    def test_debias_full_support_refits_instead_of_skipping(self):
        dense = np.diag([1., 2., 3.])
        y = np.array([1., -4., 9.])
        shrunk = np.array([.5, -1., 2.])
        for A in (dense, sp.csr_matrix(dense), aslinearoperator(dense)):
            with self.subTest(kind=type(A).__name__):
                model = opt.Optimizer.__new__(opt.Optimizer)
                actual = model._debias(A, y, shrunk)
                assert_allclose(actual, [1., -2., 3.], atol=1e-9)

    def test_fista_does_not_accept_small_steps_with_large_relative_kkt(self):
        dense = np.diag([1., 1e-6])
        y = np.array([0., 1e-6])
        for gram in (None, (dense.T @ dense, dense.T @ y)):
            info = {}
            opt._fista_lasso(dense, y, 1e-16, max_iter=80, tol=1e-7,
                             lipschitz=1., gram=gram, _info=info)
            self.assertEqual(info["n_iter"], 80)
            self.assertFalse(info["converged"])
            self.assertGreater(info["kkt_relative"], 0.9)

    def test_fista_weighted_solution_has_certified_stationarity(self):
        dense = np.diag([1., 2., 3.])
        y = np.array([1., -2., .1])
        weights = np.array([1., 2., 0.])
        expected = opt._soft_threshold(dense.T @ y, .03 * weights) / np.diag(dense)**2
        info = {}
        actual = opt._fista_lasso(dense, y, .01, penalty_weights=weights,
                                  max_iter=2000, tol=1e-9, _info=info)
        assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
        self.assertTrue(info["converged"])
        self.assertLessEqual(info["kkt_relative"], 1e-9)

    def test_iterative_lasso_final_refit_honors_requested_limits(self):
        model = opt._LassoCVIterative([.01], cv=2, tol=1e-10,
                                     max_iter=6789, rand_seed=0, n_jobs=1)
        calls = []
        def fake_fista(A, y, alpha, **kwargs):
            calls.append(kwargs)
            if kwargs.get('_info') is not None:
                kwargs['_info'].update(n_iter=1, converged=True, kkt_relative=0.)
            return np.zeros(A.shape[1])
        with patch.object(opt, '_fista_lasso', side_effect=fake_fista):
            model.fit(aslinearoperator(np.eye(6)), np.ones(6))
        self.assertEqual(calls[-1]['max_iter'], 6789)
        self.assertEqual(calls[-1]['tol'], 1e-10)

    def test_nonflat_dense_cv_does_not_mislabel_iteration_cap(self):
        model = types.SimpleNamespace(alphas_=np.array([.1, .01]),
                                      mse_path_=np.array([[2., 2.], [1., 1.]]),
                                      n_iter_=50, max_iter=50, alpha_=.01)
        with patch('builtins.print') as printed:
            opt._reselect_alpha(model, np.eye(2), np.ones(2))
        self.assertTrue(model._alpha_at_min_hitcap)
        messages = ' '.join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn('CONVERGENCE', messages)
        self.assertNotIn('Treat this fit as effectively unregularized', messages)

    def test_fista_unconverged_result_warns_even_without_info(self):
        with self.assertWarnsRegex(RuntimeWarning, 'FISTA did not converge'):
            opt._fista_lasso(np.diag([1., 1e-6]), np.array([0., 1e-6]),
                             1e-16, max_iter=40, tol=1e-7, lipschitz=1.)

    def test_operator_options_are_not_silently_ignored(self):
        A = aslinearoperator(np.eye(8))
        for method in ('RIDGE', 'LASSO', 'ALASSO'):
            with self.subTest(method=method, option='intercept'):
                with self.assertRaisesRegex(NotImplementedError, 'fit_intercept'):
                    opt.Optimizer(method, fit_intercept=True, use_gpu=False).fit(A, np.ones(8))
            with self.subTest(method=method, option='weights'):
                with self.assertRaisesRegex(NotImplementedError, 'weight'):
                    opt.Optimizer(method, use_gpu=False).fit(A, np.ones(8), weights=np.arange(1.,9.))

    def test_zero_alpha_fista_reports_stationarity(self):
        info = {}
        coef = opt._fista_lasso(np.eye(3), np.arange(3.), 0., _info=info)
        assert_allclose(coef, np.arange(3.), atol=1e-12)
        self.assertTrue(info["converged"])
        self.assertLessEqual(info["kkt_relative"], 1e-7)

    def test_lasso_results_expose_predebias_stationarity(self):
        rng = np.random.default_rng(811)
        A = aslinearoperator(rng.normal(size=(30, 4)))
        y = A @ np.array([1., -.5, 0., 0.])
        for method in ("LASSO", "ALASSO"):
            model = opt.Optimizer(method, alpha=[.01, .1], alpha_auto=False, cv=2,
                                  tol=1e-7, max_iter=5000, use_gpu=False)
            model.fit(A, y)
            info = model.results["regularized_solver_info"]
            self.assertTrue(info["converged"])
            self.assertEqual(info["stage"], "regularized_refit_before_debias")
            self.assertLessEqual(info["kkt_relative"], 1e-7)

    def test_dense_adaptive_weights_and_weighted_debias_fail_explicitly(self):
        A = np.eye(8)
        y = np.arange(8.)
        w = np.arange(1., 9.)
        with patch.dict(os.environ, {"PHEASY_LASSO_DEBIAS": "0"}):
            with self.assertRaisesRegex(NotImplementedError, "sample weight"):
                opt.Optimizer("ALASSO", use_gpu=False).fit(A, y, weights=w)
        with patch.dict(os.environ, {"PHEASY_LASSO_DEBIAS": "1"}):
            with self.assertRaisesRegex(NotImplementedError, "debias"):
                opt.Optimizer("LASSO", use_gpu=False).fit(A, y, weights=w)

    def test_dense_weighted_lasso_without_debias_still_matches_sklearn(self):
        from sklearn.linear_model import Lasso
        rng = np.random.default_rng(291)
        A = rng.normal(size=(40, 5))
        y = A @ np.arange(5.) + rng.normal(size=40) * .2
        w = np.arange(1., 41.)
        with patch.dict(os.environ, {"PHEASY_LASSO_DEBIAS": "0"}):
            model = opt.Optimizer("LASSO", alpha=[.01], alpha_auto=False,
                                  tol=1e-10, max_iter=20000, cv=2, use_gpu=False)
            model.fit(A, y, weights=w)
        reference = Lasso(alpha=.01, fit_intercept=False, tol=1e-10, max_iter=20000)
        reference.fit(A, y, sample_weight=w)
        assert_allclose(model.results["coef"], reference.coef_, atol=1e-8, rtol=1e-8)

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

    def test_subset_column_scaling_converges_and_preserves_ridge(self):
        rng = np.random.default_rng(3090)
        q, _ = np.linalg.qr(rng.normal(size=(160, 40)))
        norms = np.geomspace(1, 1e6, 40)
        A = q * norms
        y = A @ (1 / norms)
        with patch.dict(os.environ, {"PHEASY_LSQR_MAXITER": "80"}):
            with self.assertWarnsRegex(RuntimeWarning, "did not converge"):
                raw = opt._solve_subset(aslinearoperator(A), y, None, np.arange(40))
            self.assertGreater(np.linalg.norm(A @ raw - y), 1.0)
            actual = opt._solve_subset(aslinearoperator(A), y, None, np.arange(40),
                                       column_scale=norms)
        assert_allclose(A @ actual, y, rtol=1e-10, atol=1e-12)
        rows, cols = np.arange(0, 160, 2), np.array([0, 3, 7, 12])
        subset = A[np.ix_(rows, cols)]
        alpha = 0.7
        expected = np.linalg.solve(subset.T @ subset + alpha * np.eye(len(cols)),
                                   subset.T @ y[rows])
        actual = opt._solve_subset(aslinearoperator(A), y, rows, cols,
                                   ridge_alpha=alpha, column_scale=norms)
        assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)

    def test_subset_scaling_handles_zero_columns_and_rejects_invalid_norms(self):
        dense = np.array([[1., 0., 2.], [2., 0., -1.], [0., 0., 3.]])
        y = np.array([1., 2., 3.])
        A = aslinearoperator(dense)
        actual = opt._solve_subset(A, y, None, np.arange(3),
                                   column_scale=np.linalg.norm(dense, axis=0))
        assert_allclose(dense @ actual, dense @ np.linalg.lstsq(dense, y, rcond=None)[0],
                        rtol=1e-10, atol=1e-12)
        for invalid in (np.array([1., -1., 1.]), np.array([1., np.nan, 1.])):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                opt._solve_subset(A, y, None, np.arange(3), column_scale=invalid)

    def test_rfe_jacobi_reaches_subset_solver(self):
        rng = np.random.default_rng(3091)
        q, _ = np.linalg.qr(rng.normal(size=(160, 40)))
        dense = q * np.geomspace(1, 1e6, 40)
        A = opt.TwoLevelSM(sp.csr_matrix(dense), sp.eye(40, format="csr"))
        y = dense @ (1 / np.geomspace(1, 1e6, 40))
        model = opt.PheasyRFE_OLS_TSQR(min_features=40, cv=3, verbose=False)
        with patch.dict(os.environ, {"PHEASY_RFE_JACOBI": "1",
                                     "PHEASY_LSQR_MAXITER": "80"}), \
                patch.object(opt, "_iterative_solver_info", wraps=opt._iterative_solver_info) as info:
            model.fit(A, y)
        self.assertGreater(info.call_count, 0)
        for call in info.call_args_list:
            self.assertIn(call.args[0][1], (0, 1, 2, 4, 5))
        assert_allclose(A @ model.coef_, y, rtol=1e-10, atol=1e-12)

    def test_scaled_grouped_rfe_matches_dense_feature_selection(self):
        rng = np.random.default_rng(3092)
        dense = rng.normal(size=(120, 8)) * np.geomspace(1, 1e4, 8)
        coef = np.zeros(8)
        coef[[0, 3, 6]] = np.array([1., 0.4, -0.7]) / np.geomspace(1, 1e4, 8)[[0, 3, 6]]
        y = dense @ coef + rng.normal(scale=1e-4, size=120)
        models = []
        with patch.dict(os.environ, {"PHEASY_RFE_JACOBI": "1",
                                     "PHEASY_CV_GROUP_SIZE": "6"}):
            splits = opt._make_cv_splits(120, 3, 3092, 6)
            for train, valid in splits:
                self.assertFalse(set(train // 6) & set(valid // 6))
            for matrix in (dense, opt.TwoLevelSM(sp.csr_matrix(dense), sp.eye(8, format="csr"))):
                model = opt.PheasyRFE_OLS_TSQR(step=0.5, min_features=2, cv=3,
                                             random_state=3092, n_jobs=1, verbose=False)
                with patch.object(opt, "_iterative_solver_info", wraps=opt._iterative_solver_info) as info:
                    model.fit(matrix, y)
                for call in info.call_args_list:
                    self.assertIn(call.args[0][1], (0, 1, 2, 4, 5))
                models.append(model)
        np.testing.assert_array_equal(models[0].support_, models[1].support_)
        self.assertGreater(models[1].n_iter_, 0)
        assert_allclose(models[0].best_rmse_cv_, models[1].best_rmse_cv_, rtol=1e-7, atol=1e-12)
        assert_allclose(dense @ models[0].coef_, dense @ models[1].coef_, rtol=1e-8, atol=1e-10)

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
