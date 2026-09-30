#!/usr/bin/env python3
"""[FIX CV-RES / CV-FOLD / RIDGE-GRID / ARD-STD / RFE-JACOBI] regression tests.

The large-system symptom these lock in: LASSO/ALASSO cross-validation pinned
alpha* to the bottom of every auto grid ("effectively unregularized") because
the CV solves stopped on a relative KKT tolerance far above the L1 penalty of
the small alphas, so the CV curve followed the iteration count, not alpha.
The resident path is exercised with CPU torch (numerical emulation, not CUDA
evidence).
"""
import contextlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import optimizer as opt          # noqa: E402
from core import gpu_backend as gb         # noqa: E402

try:
    import torch
except ImportError:                        # pragma: no cover
    torch = None


def illcond_problem(seed=0, n_cfg=60, rows_per=24, p=120, noise=0.15, frac_zero=0.6):
    """FC2-like (40x) and FC3-like (1x) column blocks, sparse truth, grouped rows."""
    rng = np.random.default_rng(seed)
    n = n_cfg * rows_per
    X = rng.normal(size=(n, p))
    cn = np.concatenate([np.full(p // 4, 40.0), np.full(p - p // 4, 1.0)])
    cn *= np.exp(rng.normal(scale=0.3, size=p))
    X = X * cn
    beta = rng.normal(size=p) / cn
    beta[rng.random(p) < frac_zero] = 0.0
    f = X @ beta
    y = f + noise * rng.normal(size=n) * np.std(f)
    return X, y, rows_per


def exact_cv(X, y, alphas, splits):
    from sklearn.linear_model import Lasso
    out = np.zeros((len(alphas), len(splits)))
    for i, a in enumerate(alphas):
        for k, (tr, va) in enumerate(splits):
            c = Lasso(alpha=a, fit_intercept=False, tol=1e-12, max_iter=200000,
                      precompute=True).fit(X[tr], y[tr]).coef_
            out[i, k] = np.mean((X[va] @ c - y[va]) ** 2)
    return out.mean(axis=1)


class FoldTests(unittest.TestCase):
    def test_contiguous_folds_are_config_blocks(self):
        splits = opt._make_cv_splits(10 * 6, 5, 0, 6)
        self.assertEqual(len(splits), 5)
        seen = np.zeros(60, dtype=int)
        for tr, va in splits:
            self.assertEqual(len(np.intersect1d(tr, va)), 0)
            cfg = np.unique(va // 6)
            # two whole, ADJACENT configurations per validation fold
            np.testing.assert_array_equal(np.diff(cfg), np.ones(cfg.size - 1))
            self.assertEqual(len(va), 12)
            seen[va] += 1
        np.testing.assert_array_equal(seen, np.ones(60))

    def test_interleaved_legacy_mode(self):
        with patch.dict(os.environ, {"PHEASY_CV_FOLD_MODE": "interleaved"}):
            splits = opt._make_cv_splits(10 * 6, 5, 0, 6)
        cfg = np.unique(splits[0][1] // 6)
        self.assertGreater(int(np.diff(cfg).max()), 1)

    def test_gpu_backend_uses_the_same_splits(self):
        a = opt._make_cv_splits(120, 4, 3, 6)
        b = gb._make_cv_splits(120, 4, 3, 6)
        for (t1, v1), (t2, v2) in zip(a, b):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)

    def test_bad_mode_rejected(self):
        with patch.dict(os.environ, {"PHEASY_CV_FOLD_MODE": "shuffle"}):
            with self.assertRaises(ValueError):
                opt._make_cv_splits(60, 5, 0, 6)


class SelectionTests(unittest.TestCase):
    def quiet(self, *args, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return opt._select_cv_alpha(*args, **kw)

    def test_unresolved_alphas_are_excluded(self):
        alphas = np.array([1e-4, 1e-3, 1e-2, 1e-1])
        mse = np.array([[0.5, 0.5], [0.9, 0.9], [0.8, 0.8], [1.2, 1.2]])
        res = np.array([[False, False], [True, True], [True, True], [True, True]])
        sel = self.quiet(alphas, mse, res)
        self.assertEqual(sel["best"], 2)          # 1e-4 had the lowest MSE but is unresolved
        self.assertEqual(sel["n_valid"], 3)

    def test_ols_limit_wins_and_is_flagged(self):
        alphas = np.array([1e-3, 1e-2])
        mse = np.array([[1.0, 1.0], [1.1, 1.1]])
        res = np.ones((2, 2), dtype=bool)
        sel = self.quiet(alphas, mse, res, ols_mse=np.array([0.9, 0.95]))
        self.assertEqual(sel["best"], -1)
        self.assertTrue(sel["at_min"])

    def test_interior_minimum_beats_ols(self):
        alphas = np.array([1e-3, 1e-2, 1e-1])
        mse = np.array([[1.0, 1.0], [0.8, 0.8], [1.5, 1.5]])
        sel = self.quiet(alphas, mse, np.ones((3, 2), bool), ols_mse=np.array([1.05, 1.05]))
        self.assertEqual(sel["best"], 1)
        self.assertFalse(sel["at_min"])

    def test_one_se_rule_picks_sparsest_within_one_se(self):
        alphas = np.array([1e-3, 1e-2, 1e-1])
        mse = np.array([[1.0, 1.2], [1.05, 1.25], [2.0, 2.0]])
        with patch.dict(os.environ, {"PHEASY_LASSO_1SE": "1"}):
            sel = self.quiet(alphas, mse, np.ones((3, 2), bool), ols_mse=np.array([0.99, 1.2]))
        self.assertEqual(sel["best"], 1)


class FistaToleranceTests(unittest.TestCase):
    def test_alpha_aware_tolerance_resolves_small_alpha(self):
        X, y, _ = illcond_problem(0)
        G, b = X.T @ X, X.T @ y
        amax = np.abs(b).max() / X.shape[0]
        a = 1e-4 * amax
        loose, aware = {}, {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt._fista_lasso(X, y, a, gram=(G, b), tol=1e-3, max_iter=4000, _info=loose,
                             warn_nonconvergence=False, auto_floor=False)
            opt._fista_lasso(X, y, a, gram=(G, b), tol=1e-3, max_iter=4000, _info=aware,
                             warn_nonconvergence=False, auto_floor=False,
                             alpha_tol_ratio=0.1)
        # a relative KKT of 1e-3 is ~10x ABOVE the penalty level at 1e-4*alpha_max
        self.assertFalse(loose["resolved"])
        self.assertLessEqual(aware["tol_effective"], 0.1 * aware["penalty_relative"] * 1.0001)
        self.assertLess(aware["kkt_relative"], loose["kkt_relative"])


def _emulated(fn):
    return unittest.skipIf(torch is None, "torch not installed")(fn)


class ResidentResolutionTests(unittest.TestCase):
    def _fit(self, env, ols=True):
        X, y, g = illcond_problem(0, n_cfg=40, p=80)
        A = opt.TwoLevelSM(sp.csr_matrix(X), sp.eye(X.shape[1], format="csr"))
        envd = {"PHEASY_CV_GROUP_SIZE": str(g)}
        envd.update(env)
        with patch.object(gb, "available", return_value=True), \
                patch.object(gb, "enabled", return_value=True), \
                patch.object(gb, "device", return_value="cpu"), \
                patch.dict(os.environ, envd), warnings.catch_warnings(), \
                contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            m = gb.GpuTwoLevelLassoCV(np.logspace(-6, -2, 17), 5, 1e-6, 20000, 0,
                                      group_size=g, standardize=False, nalpha=17,
                                      decades=4.0, alpha_auto=True, ols_reference=ols)
            m.fit(A, y)
        splits = opt._make_cv_splits(X.shape[0], 5, 0, g)
        return m, X, y, splits

    @_emulated
    def test_legacy_behaviour_pins_the_grid_bottom(self):
        # documents the bug: 400-iteration CV solves at relative KKT 1e-3
        m, X, y, splits = self._fit({"PHEASY_CV_ALPHA_AWARE": "0",
                                     "PHEASY_RESIDENT_JACOBI": "0"}, ols=False)
        self.assertEqual(m.alpha_, float(np.min(m.alphas_)))
        ex = exact_cv(X, y, m.alphas_, splits)
        self.assertGreater(int(np.argmin(ex)), 3)       # the true optimum is interior

    @_emulated
    def test_resolved_cv_finds_the_exact_optimum(self):
        m, X, y, splits = self._fit({})
        ex = exact_cv(X, y, m.alphas_, splits)
        self.assertAlmostEqual(m.alpha_, float(m.alphas_[int(np.argmin(ex))]))
        self.assertFalse(m._alpha_is_zero)
        self.assertEqual(m.regularized_solver_info_["dtype"], "float64")
        self.assertTrue(m.regularized_solver_info_["jacobi_scaled"])


class OptimizerOlsLimitTests(unittest.TestCase):
    def test_dense_truth_low_noise_returns_ols_limit(self):
        # noiseless dense truth: every alpha > 0 only adds shrinkage bias, so
        # the exact OLS limit must win the CV (run_pheasy's calling convention:
        # derived grid handed in as alpha=, ols_limit=--alpha_auto)
        rng = np.random.default_rng(5)
        X = rng.normal(size=(40 * 6, 12))
        y = X @ rng.normal(size=12)
        with patch.dict(os.environ, {"PHEASY_CV_GROUP_SIZE": "6"}), \
                contextlib.redirect_stdout(io.StringIO()):
            grid = opt.derive_alpha_grid(X, y, nalpha=9, decades=4.0)
            o = opt.Optimizer("LASSO", alpha=grid, cv=5, rand_seed=0, tol=1e-10,
                              max_iter=100000, ols_limit=True)
            o.fit(X, y)
        r = o.results
        self.assertTrue(r["cv_selected_ols_limit"])
        self.assertEqual(r["alpha"], 0.0)
        self.assertEqual(r["debias_backend"], "not_needed_ols_limit")
        np.testing.assert_allclose(r["coef"], np.linalg.lstsq(X, y, rcond=None)[0],
                                   rtol=1e-8, atol=1e-10)
        self.assertTrue(np.all(np.isfinite(o.metrics["mse_path"])))

    def test_explicit_grid_is_fitted_as_given(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(40 * 6, 12))
        y = X @ rng.normal(size=12) + 1e-3 * rng.normal(size=X.shape[0])
        with patch.dict(os.environ, {"PHEASY_CV_GROUP_SIZE": "6"}), \
                contextlib.redirect_stdout(io.StringIO()):
            o = opt.Optimizer("LASSO", alpha=[1e-3, 1e-2], cv=5, rand_seed=0, tol=1e-10)
            o.fit(X, y)
        self.assertGreater(o.results["alpha"], 0.0)
        self.assertNotIn("cv_selected_ols_limit", o.results)


class RidgeGridTests(unittest.TestCase):
    def test_grid_reaches_the_spectrum_of_a_large_fit(self):
        rng = np.random.default_rng(1)
        X = 100.0 * rng.normal(size=(30 * 6, 8))     # lambda_max(X^T X) ~ 2e6
        y = X @ rng.normal(size=8) + rng.normal(size=X.shape[0])
        with patch.dict(os.environ, {"PHEASY_CV_GROUP_SIZE": "6"}), \
                contextlib.redirect_stdout(io.StringIO()):
            o = opt.Optimizer("RIDGE", cv=5, rand_seed=0)
            o.fit(X, y)
        lam = np.linalg.eigvalsh(X.T @ X)[-1]
        self.assertGreater(o.results["ridge_lambda_max"], 0.5 * lam)
        self.assertIn("ridge_alpha_at_grid_min", o.results)

    def test_manual_grid_kept_when_disabled(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(30 * 6, 8))
        y = X @ rng.normal(size=8)
        with patch.dict(os.environ, {"PHEASY_CV_GROUP_SIZE": "6",
                                     "PHEASY_RIDGE_ALPHA_AUTO": "0"}), \
                contextlib.redirect_stdout(io.StringIO()):
            o = opt.Optimizer("RIDGE", cv=5, rand_seed=0)
            o.fit(X, y)
        self.assertNotIn("ridge_lambda_max", o.results)
        self.assertLessEqual(o.results["alpha"], 1e-2)


class ArdAndRfeTests(unittest.TestCase):
    def test_ardr_standardizes_to_unit_variance(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(20 * 6, 6))
        y = X @ np.array([1.0, 0.0, 0.5, 0.0, 0.0, -2.0]) + 0.01 * rng.normal(size=120)
        with patch.dict(os.environ, {"PHEASY_CV_GROUP_SIZE": "6"}), \
                contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o = opt.Optimizer("ARDR", cv=3, rand_seed=0, standardize=True)
            o.fit(X, y)
        self.assertEqual(o.results.get("ard_standardization"), "unit_variance")
        np.testing.assert_allclose(o.results["coef"][[0, 2, 5]], [1.0, 0.5, -2.0], atol=0.05)

    def test_rfe_jacobi_defaults_on_for_operators(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(20 * 6, 10)) * np.r_[np.full(3, 40.0), np.ones(7)]
        y = X @ rng.normal(size=10)
        A = opt.TwoLevelSM(sp.csr_matrix(X), sp.eye(10, format="csr"))
        with patch.dict(os.environ, {"PHEASY_CV_GROUP_SIZE": "6", "PHEASY_GPU_RFE_RESIDENT": "0",
                                     "PHEASY_RFE_FINAL_TSQR": "0"}), \
                contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o = opt.Optimizer("RFE", cv=3, rand_seed=0, use_gpu=False)
            o.fit(A, y)
        self.assertTrue(o.results["backend_metadata"]["jacobi_applied"])



class GpuMemoryAndOffloadTests(unittest.TestCase):
    """[FIX GPU-MEM / RFE-MEM / GPU-PRED] memory budgets and host-work removal."""

    def _gpu_on(self, avail):
        return (patch.object(gb, "available", return_value=True),
                patch.object(gb, "enabled", return_value=True),
                patch.object(gb, "device", return_value="cpu"),
                patch.object(gb, "available_memory_bytes", return_value=avail))

    @_emulated
    def test_lasso_preflight_counts_the_grams(self):
        A = np.random.default_rng(0).normal(size=(50, 400))   # wide: p^2 >> n p
        need = gb.lasso_gram_footprint_bytes(*A.shape)
        avail = int(need / 2)          # A itself fits (_gpu_dense), the Grams do not
        p1, p2, p3, p4 = self._gpu_on(avail)
        with p1, p2, p3, p4, contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(opt._gpu_dense(A))
            with patch.dict(os.environ, {"PHEASY_GPU_MODE": "auto"}):
                self.assertEqual(opt._lasso_backend(A), "dense")
            with patch.dict(os.environ, {"PHEASY_GPU_MODE": "required"}):
                with self.assertRaises(MemoryError):
                    opt._lasso_backend(A)
        p1, p2, p3, p4 = self._gpu_on(10 * need)
        with p1, p2, p3, p4, patch.dict(os.environ, {"PHEASY_GPU_MODE": "auto"}):
            self.assertEqual(opt._lasso_backend(A), "gpu")

    @_emulated
    def test_fold_outer_gpu_lasso_matches_exact_cv(self):
        X, y, g = illcond_problem(1, n_cfg=30, p=40)
        Xs = X / np.linalg.norm(X, axis=0)
        alphas = np.logspace(-5, -1, 9)
        splits = opt._make_cv_splits(X.shape[0], 5, 0, g)
        p1, p2, p3, p4 = self._gpu_on(None)
        with p1, p2, p3, p4, contextlib.redirect_stdout(io.StringIO()), \
                patch.dict(os.environ, {"PHEASY_CV_TOL": "1e-11", "PHEASY_CV_MAX_ITER": "20000"}):
            m = gb.GpuLassoCV(alphas, 5, 1e-11, 20000, 0, group_size=g, ols_reference=True)
            m.fit(Xs, y)
        ex = exact_cv(Xs, y, alphas, splits)
        np.testing.assert_allclose(m.mse_path_.mean(axis=1), ex, rtol=1e-5)
        from sklearn.linear_model import Lasso
        ref = Lasso(alpha=m.alpha_, fit_intercept=False, tol=1e-13, max_iter=500000,
                    precompute=True).fit(Xs, y).coef_
        np.testing.assert_allclose(m.coef_, ref, atol=1e-6)

    def test_rfe_final_refit_inplace_syrk_is_exact(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(500, 30))
        y = X @ rng.normal(size=30)
        A = opt.TwoLevelSM(sp.csr_matrix(X), sp.eye(30, format="csr"))
        idx = np.arange(0, 30, 2)
        with patch.dict(os.environ, {"PHEASY_RFE_FINAL_BLOCK_ROWS": "64"}), \
                contextlib.redirect_stdout(io.StringIO()):
            c = opt._rfe_final_refit_exact(A, y, idx)
        np.testing.assert_allclose(c, np.linalg.lstsq(X[:, idx], y, rcond=None)[0],
                                   rtol=1e-9, atol=1e-10)

    def test_prediction_is_reused_for_the_same_operator(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(60, 5))
        y = X @ rng.normal(size=5)
        o = opt.Optimizer("OLS")
        o.fit(X, y)
        with patch.object(o._model, "predict", side_effect=AssertionError("recomputed")):
            np.testing.assert_allclose(o.predict(X), X @ o.results["coef"])
        np.testing.assert_allclose(o.predict(X.copy()), X @ o.results["coef"])

    def test_predict_rows_uses_the_resident_view(self):
        A = opt.TwoLevelSM(sp.eye(6, format="csr"), sp.eye(6, format="csr"))
        A._gpu_ridge_op = object()
        calls = []

        class _View(object):
            def __matmul__(self, c):
                calls.append(1)
                return np.full(3, 7.0)

        with patch.object(opt, "_row_slice", return_value=_View()):
            out = opt._predict_rows(A, np.ones(6), np.array([0, 2, 4]))
        self.assertEqual(calls, [1])
        np.testing.assert_allclose(out, 7.0)

    @_emulated
    def test_debias_residuals_come_from_the_resident_factors(self):
        from types import SimpleNamespace
        rng = np.random.default_rng(8)
        X = rng.normal(size=(40, 6))
        y = X @ rng.normal(size=6) + 0.01 * rng.normal(size=40)
        A = opt.TwoLevelSM(sp.csr_matrix(X), sp.eye(6, format="csr"))
        with patch.object(gb, "available", return_value=True), \
                patch.object(gb, "enabled", return_value=True), \
                patch.object(gb, "device", return_value="cpu"):
            op = gb.GpuTwoLevelOperator(A)
            sup = np.array([0, 2, 3, 5])
            ref_sub = np.linalg.lstsq(X[:, sup], y, rcond=None)[0]
            o = opt.Optimizer("LASSO")
            o._model = SimpleNamespace(_operator=op, column_scale_=None)
            with patch.object(gb, "solve_resident_subset",
                              return_value=(op.torch.as_tensor(ref_sub), {"converged": True})):
                old = np.linspace(0.1, 0.6, 6)
                sub, res = o._debias_resident_gpu(y, sup, old)
        new = np.zeros(6)
        new[sup] = sub
        self.assertIsNotNone(res)
        np.testing.assert_allclose(res, [np.linalg.norm(X @ new - y),
                                         np.linalg.norm(X @ old - y)], rtol=1e-10)
        self.assertIsNone(o._model._operator)



class GridExtensionTests(unittest.TestCase):
    """[FIX CV-EXT] a grid that stops above the CV optimum is extended downward
    (the MoS2 / Mg4C60 symptom: alpha* "at the grid minimum, effectively
    unregularized", while a smaller -- still nonzero -- alpha is optimal)."""

    @classmethod
    def setUpClass(cls):
        from sklearn.linear_model import Lasso
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50 * 6, 40))
        b = rng.normal(size=40)
        b[rng.random(40) < 0.6] = 0.0
        cls.y = X @ b + 1.0 * rng.normal(size=300)
        cls.Xs = X / np.linalg.norm(X, axis=0)
        cls.splits = opt._make_cv_splits(300, 5, 0, 6)
        cls.amax = float(np.abs(cls.Xs.T @ cls.y).max() / 300)
        # handed-in grid stops a decade ABOVE the optimum (~1e-2 alpha_max)
        cls.grid = cls.amax * np.logspace(-1, 0, 9)

    def exact(self, alphas):
        from sklearn.linear_model import Lasso
        return np.array([np.mean([np.mean((self.Xs[va] @ Lasso(
            alpha=a, fit_intercept=False, tol=1e-12, max_iter=200000).fit(
                self.Xs[tr], self.y[tr]).coef_ - self.y[va]) ** 2)
            for tr, va in self.splits]) for a in alphas])

    def check(self, alpha, alphas):
        self.assertLess(alpha, self.grid.min())                 # went below the grid
        ex = self.exact(alphas)
        self.assertAlmostEqual(alpha, float(alphas[int(np.argmin(ex))]), delta=1e-12)

    def test_sklearn_dense_path_extends(self):
        with patch.dict(os.environ, {"PHEASY_CV_GROUP_SIZE": "6"}), \
                contextlib.redirect_stdout(io.StringIO()):
            o = opt.Optimizer("LASSO", alpha=self.grid, cv=5, rand_seed=0, tol=1e-10,
                              max_iter=200000, ols_limit=True, use_gpu=False)
            o.fit(self.Xs, self.y)
        self.check(o.results["alpha"], np.sort(o._model.alphas_))
        self.assertFalse(o.results["alpha_at_grid_edge"])
        self.assertGreater(o.results["cv_selection"]["extended_decades"], 0)

    def test_cpu_fista_path_extends(self):
        A = opt.TwoLevelSM(sp.csr_matrix(self.Xs), sp.eye(40, format="csr"))
        with patch.dict(os.environ, {"PHEASY_CV_GROUP_SIZE": "6", "PHEASY_CV_TOL": "1e-10",
                                     "PHEASY_GPU_LASSO_RESIDENT": "0"}), \
                contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = opt._LassoCVIterative(self.grid, 5, 1e-10, 20000, 0, group_size=6,
                                      ols_reference=True)
            m.fit(A, self.y)
        self.check(m.alpha_, m.alphas_)

    @_emulated
    def test_gpu_gram_path_extends(self):
        with patch.object(gb, "available", return_value=True), \
                patch.object(gb, "enabled", return_value=True), \
                patch.object(gb, "device", return_value="cpu"), \
                patch.dict(os.environ, {"PHEASY_CV_TOL": "1e-10",
                                        "PHEASY_CV_MAX_ITER": "20000"}), \
                contextlib.redirect_stdout(io.StringIO()):
            m = gb.GpuLassoCV(self.grid, 5, 1e-10, 20000, 0, group_size=6,
                              ols_reference=True)
            m.fit(self.Xs, self.y)
        self.check(m.alpha_, m.alphas_)
        self.assertGreater(m.cv_selection_["extended_decades"], 0)

    @_emulated
    def test_resident_path_extends(self):
        A = opt.TwoLevelSM(sp.csr_matrix(self.Xs), sp.eye(40, format="csr"))
        with patch.object(gb, "available", return_value=True), \
                patch.object(gb, "enabled", return_value=True), \
                patch.object(gb, "device", return_value="cpu"), \
                patch.dict(os.environ, {"PHEASY_CV_TOL": "1e-10",
                                        "PHEASY_CV_MAX_ITER": "20000"}), \
                contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = gb.GpuTwoLevelLassoCV(self.grid, 5, 1e-10, 20000, 0, group_size=6,
                                      standardize=False, alpha_auto=False,
                                      ols_reference=True)
            m.fit(A, self.y)
        self.check(m.alpha_, m.alphas_)
        self.assertGreater(m.cv_selection_["extended_decades"], 0)


if __name__ == "__main__":
    unittest.main()
