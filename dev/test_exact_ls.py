#!/usr/bin/env python3
"""[FIX GRAM-F64 / EXACT-LS / RVM-BETA] regression tests.

Large systems reach the fitters as a float32 two-level operator
(SM = SM_prime @ NS).  Its matvec/rmatvec run in float32, and the fitters used
them where float64 was needed:

* the Gram builders (ARDR, RVM, the CPU FISTA LASSO/ALASSO Gram path) formed
  b = A^T y -- and on the block route G itself -- in float32;
* OLS, RIDGE CV and the LASSO debias solved with CGLS/LSMR/LSQR that stop at a
  float32 precision floor instead of at the least-squares / ridge solution.

Every test compares against an exact float64 reference on float32-representable
data, so the only admissible error is float64 round-off.
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
from core.fast_rvm import fast_rvm, _evidence_beta   # noqa: E402


def problem(seed=0, n_cfg=24, rows_per=40, p2=24, p3=96, k3=40, noise=0.02,
            corr=3.0, nlat=30):
    """Correlated FC2-like (40x norm) / FC3-like columns, float32-representable."""
    r = np.random.default_rng(seed)
    n = n_cfg * rows_per
    L = r.standard_normal((n, nlat))
    A = r.standard_normal((n, p2 + p3)) + corr * L @ r.standard_normal((nlat, p2 + p3))
    A[:, :p2] *= 40.0
    c = np.zeros(p2 + p3)
    c[:p2] = r.standard_normal(p2) * 0.5
    nz = r.choice(np.arange(p2, p2 + p3), k3, replace=False)
    c[nz] = r.standard_normal(k3) * np.geomspace(20, 0.05, k3)
    A = A.astype(np.float32).astype(np.float64)
    y = A @ c
    y = y + noise * np.std(y) * r.standard_normal(n)
    return A, y, rows_per


def twolevel32(A):
    A32 = A.astype(np.float32)
    return opt.TwoLevelSM(sp.csr_matrix(A32),
                          sp.eye(A.shape[1], format="csr", dtype=np.float32))


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - b) / max(np.linalg.norm(b), 1e-300))


def fit(method, A, y, env=None, **kw):
    m = opt.Optimizer(method, cv=4, rand_seed=0, use_gpu=False, **kw)
    with patch.dict(os.environ, dict({"PHEASY_GPU_MODE": "cpu"}, **(env or {}))), \
            warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        m.fit(A, y)
    return m


class GramF64Tests(unittest.TestCase):
    def setUp(self):
        self.A, self.y, _ = problem()
        self.G = self.A.T @ self.A
        self.b = self.A.T @ self.y

    def _check(self, G, b, D=None):
        Ge, be = self.G, self.b
        if D is not None:
            Ge, be = Ge * D[:, None] * D[None, :], be * D
        self.assertLess(np.abs(G - Ge).max() / np.abs(Ge).max(), 1e-13)
        self.assertLess(np.abs(b - be).max() / np.abs(be).max(), 1e-13)

    def test_every_gram_route_is_float64(self):
        T = twolevel32(self.A)
        with contextlib.redirect_stdout(io.StringIO()):
            G, b, how = opt._build_gram_matrix(T, self.y)
            self.assertEqual(how, "P")
            self._check(G, b)
            self._check(*opt._compute_gram_blockwise(T, self.y))
            self._check(*opt._compute_gram(sp.csr_matrix(self.A.astype(np.float32)), self.y)[:2])
            cn = np.linalg.norm(self.A, axis=0)
            S = opt._scale_operator(T, cn)
            self._check(*opt._build_gram_matrix(S, self.y)[:2], D=1 / cn)
            self._check(*opt._compute_gram_blockwise(S, self.y), D=1 / cn)
            self._check(*opt._compute_gram(S, self.y)[:2], D=1 / cn)

    def test_rhs_rounding_was_the_error(self):
        # the float32 rmatvec is what the Gram builders used for b
        T = twolevel32(self.A)
        b32 = np.asarray(T.T @ self.y, dtype=np.float64)
        self.assertGreater(np.abs(b32 - self.b).max() / np.abs(self.b).max(), 1e-8)
        np.testing.assert_allclose(opt._rmatvec_f64(T, self.y), self.b, rtol=1e-12,
                                   atol=1e-12 * np.abs(self.b).max())


class ExactLeastSquaresTests(unittest.TestCase):
    def setUp(self):
        self.A, self.y, g = problem()
        self.env = {"PHEASY_CV_GROUP_SIZE": str(g), "PHEASY_RIDGE_LCURVE": "0"}

    def test_ols_on_float32_twolevel_is_the_least_squares_solution(self):
        exact = np.linalg.lstsq(self.A, self.y, rcond=None)[0]
        m = fit("OLS", twolevel32(self.A), self.y, self.env)
        self.assertEqual(m.results["execution_backend"], "cpu_gram_exact")
        self.assertTrue(m.results["fit_accepted"])
        self.assertLess(rel(m.results["coef"], exact), 1e-8)
        # the iterative solver it replaces (PHEASY_EXACT_GRAM_GB=0) is not
        old = fit("OLS", twolevel32(self.A), self.y, dict(self.env, PHEASY_EXACT_GRAM_GB="0"))
        self.assertNotEqual(old.results.get("execution_backend"), "cpu_gram_exact")
        self.assertGreater(rel(old.results["coef"], exact), 100 * rel(m.results["coef"], exact))

    def test_ridge_cv_on_float32_twolevel_matches_the_exact_dense_path(self):
        dense = fit("RIDGE", self.A, self.y, self.env)
        got = fit("RIDGE", twolevel32(self.A), self.y, self.env)
        self.assertEqual(got.results["execution_backend"], "cpu_gram_ridge_exact")
        # the auto grid is anchored on lambda_max(A^T A), estimated separately by
        # each path (they agree to ~1e-8): same grid point, same curve
        self.assertAlmostEqual(got.results["alpha"] / dense.results["alpha"], 1.0, delta=1e-6)
        np.testing.assert_allclose(np.asarray(got.results["mse_path"]),
                                   np.asarray(dense.results["mse_path"]), rtol=1e-6)
        a = got.results["alpha"]
        ridge = np.linalg.solve(self.A.T @ self.A + a * np.eye(self.A.shape[1]),
                                self.A.T @ self.y)
        self.assertLess(rel(got.results["coef"], ridge), 1e-8)

    def test_explicit_resident_request_keeps_the_iterative_ridge(self):
        A, y, _ = problem(n_cfg=8, p3=24, k3=8)
        with patch.object(opt, "_exact_ridge_cv",
                          side_effect=AssertionError("exact path used")):
            m = fit("RIDGE", twolevel32(A), y,
                    dict(self.env, PHEASY_GPU_RIDGE_RESIDENT="1"),
                    alpha=[1e2, 1e0])
        self.assertNotEqual(m.results.get("execution_backend"), "cpu_gram_ridge_exact")

    def test_lasso_debias_on_float32_twolevel_is_ols_on_the_support(self):
        A, y, _ = problem(n_cfg=12, p3=40, k3=16)
        m = fit("LASSO", twolevel32(A), y, dict(self.env, PHEASY_LASSO_DEBIAS="1"),
                alpha=[1e-1, 1e-2])
        c = np.asarray(m.results["coef"])
        sup = np.flatnonzero(c)
        self.assertEqual(m.results.get("debias_backend"), "cpu_gram_exact")
        ref = np.linalg.lstsq(A[:, sup], y, rcond=None)[0]
        self.assertLess(rel(c[sup], ref), 1e-8)

    def test_budget_zero_disables_every_exact_path(self):
        env = dict(self.env, PHEASY_EXACT_GRAM_GB="0")
        with patch.dict(os.environ, env):
            self.assertIsNone(opt._exact_normal_solve(twolevel32(self.A), self.y))
            self.assertIsNone(opt._exact_ridge_cv(
                twolevel32(self.A), self.y, [1.0],
                opt._make_cv_splits(self.A.shape[0], 4, 0, 40)))


class RvmTests(unittest.TestCase):
    def _data(self, seed):
        r = np.random.default_rng(seed)
        n, p, k, sig = 240, 150, 25, 0.5
        X = r.standard_normal((n, p))
        w = np.zeros(p)
        w[r.choice(p, k, replace=False)] = r.standard_normal(k) * 2
        y = X @ w + sig * r.standard_normal(n)
        return X.T @ X, X.T @ y, float(y @ y), n, float(np.var(y)), sig

    def test_supplied_beta_is_fixed(self):
        G, b, yty, n, var, sig = self._data(0)
        out = fast_rvm(G, b, yty, n, y_var=var, beta=3.0)
        self.assertEqual(out["beta"], 3.0)

    def test_noise_update_is_the_evidence_fixed_point(self):
        G, b, yty, n, var, sig = self._data(1)
        out = fast_rvm(G, b, yty, n, y_var=var, beta_iters=60, prune_threshold=0)
        c = out["coef"]
        rss = yty - 2 * float(c @ b) + float(c @ (G @ c))
        self.assertAlmostEqual(out["beta"], _evidence_beta(G, out, n, rss),
                               delta=1e-3 * out["beta"])
        # n / rss (the old update) is strictly larger: it ignores sum(gamma)
        self.assertGreater(n / rss, 1.05 * out["beta"])

    def test_noise_estimate_improves_over_n_over_rss(self):
        est = []
        for seed in range(4):
            G, b, yty, n, var, sig = self._data(seed)
            est.append(fast_rvm(G, b, yty, n, y_var=var, beta_iters=30)["beta"] * sig ** 2)
        # old update: 1.94 on these seeds; the evidence update ~1.46 (sklearn's
        # batch ARD lands at ~1.29 on the same data)
        self.assertLess(float(np.mean(est)), 1.7)

    def test_batch_addition_uses_fresh_statistics(self):
        G, b, yty, n, var, sig = self._data(2)
        one = fast_rvm(G, b, yty, n, y_var=var, beta=1 / sig ** 2)
        many = fast_rvm(G, b, yty, n, y_var=var, beta=1 / sig ** 2, add_batch=8)
        self.assertTrue(np.all(np.isfinite(many["coef"])))
        self.assertTrue(np.all(many["alpha"] > 0))
        self.assertLess(rel(many["coef"], one["coef"]), 0.1)


if __name__ == "__main__":
    unittest.main()
