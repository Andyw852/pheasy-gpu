#!/usr/bin/env python3
"""[HARM_DENSE] contract tests: sparsify the anharmonic block only.

The reduced IFC vector is [HARM | ANHARM3 ...].  With an unpenalized (free)
harmonic block every sparse method must

  * never zero, shrink or eliminate a free coefficient (LASSO, ALASSO, ARDR,
    RVM, RFE, RFE-OLS-TSQR, and the debias / zero-tolerance post-processing);
  * anchor the alpha grid at the PENALIZED block's KKT threshold on the
    residual of the free-block OLS (not at max|A^T y| over every column);
  * for LASSO, reproduce the Frisch-Waugh-Lovell reference exactly: sklearn
    Lasso on the data with the free columns projected out, then OLS of the free
    block on the remaining residual.

No CUDA is needed.  The GPU classes are exercised through the same explicit
CPU-torch emulation the resident tests use (skipped when torch is missing).

    python -m unittest dev.test_harm_dense
"""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import optimizer as opt       # noqa: E402

try:
    import torch                        # noqa: F401
    from core import gpu_backend as gb  # noqa: E402
except ImportError:                     # pragma: no cover
    torch = None
    gb = None

N_CFG, ROWS, P_H, P_3 = 40, 12, 10, 30
ENV = {
    "PHEASY_USE_GPU": "0", "PHEASY_GPU_SM": "0", "PHEASY_TWOLEVEL_CACHE_T": "0",
    "PHEASY_MAX_CORES": "1", "PHEASY_N_JOBS": "1",
    "PHEASY_CV_GROUP_SIZE": str(ROWS), "PHEASY_RFE_PATIENCE": "50",
    "PHEASY_RVM_VERBOSE": "0", "PHEASY_ARDR_VERBOSE": "0",
    "PHEASY_LSQR_ATOL": "1e-14", "PHEASY_LSQR_BTOL": "1e-14",
    "PHEASY_LSQR_MAXITER": "20000",
}


def make_problem(seed=7, noise=2e-3):
    """FC2-like block (unit-scale columns, dense truth incl. tiny terms) plus a
    FC3-like block (columns ~20x smaller, sparse truth), mildly correlated."""
    rng = np.random.default_rng(seed)
    n = N_CFG * ROWS
    A_h = rng.normal(size=(n, P_H))
    A_3 = 0.05 * (rng.normal(size=(n, P_3)) + 0.3 * A_h[:, rng.integers(0, P_H, P_3)])
    A = np.hstack([A_h, A_3])
    beta = np.zeros(P_H + P_3)
    beta[:P_H] = rng.normal(size=P_H)
    beta[P_H - 2:P_H] = [3e-4, -2e-4]          # tiny but real harmonic terms
    support3 = P_H + np.array([0, 3, 7, 11, 19, 25])
    beta[support3] = rng.choice([-1.0, 1.0], size=support3.size) * rng.uniform(1, 4, support3.size)
    y = A @ beta + noise * rng.normal(size=n)
    free = np.zeros(P_H + P_3, dtype=bool)
    free[:P_H] = True
    return A, y, beta, free


def make_illcond_problem(seed=3, noise=2e-3):
    """Raw-column scales of a real FC fit: FC3 columns ~40x smaller than FC2,
    correlated with them, and nearly collinear among themselves (cond(G)~4e6,
    as measured on MnIn2Se4 raw columns: 1.4e6)."""
    rng = np.random.default_rng(seed)
    n, k = N_CFG * ROWS, 8
    A_h = rng.normal(size=(n, P_H))
    Z = rng.normal(size=(n, k))
    M = rng.normal(size=(k, P_3))
    A_3 = 0.025 * (Z @ M / np.sqrt(k) + 0.03 * rng.normal(size=(n, P_3))
                   + 0.5 * A_h[:, rng.integers(0, P_H, P_3)])
    A = np.hstack([A_h, A_3])
    beta = np.zeros(P_H + P_3)
    beta[:P_H] = rng.normal(size=P_H)
    sup = P_H + np.array([0, 3, 7, 11, 19, 25])
    beta[sup] = rng.choice([-1.0, 1.0], size=sup.size) * rng.uniform(20, 80, sup.size)
    y = A @ beta + noise * rng.normal(size=n)
    free = np.zeros(P_H + P_3, dtype=bool)
    free[:P_H] = True
    return A, y, free


def fwl_reference(A, y, free, alpha):
    """Exact unpenalized-block LASSO via Frisch-Waugh-Lovell + sklearn."""
    from sklearn.linear_model import Lasso
    A_f, A_p = A[:, free], A[:, ~free]
    Q, _ = np.linalg.qr(A_f)
    proj = lambda M: M - Q @ (Q.T @ M)       # noqa: E731
    las = Lasso(alpha=alpha, fit_intercept=False, tol=1e-14, max_iter=1_000_000)
    las.fit(proj(A_p), proj(y))
    x = np.zeros(A.shape[1])
    x[~free] = las.coef_
    x[free] = np.linalg.lstsq(A_f, y - A_p @ las.coef_, rcond=None)[0]
    return x


def penalized_alpha_max(A, y, free):
    A_f = A[:, free]
    r = y - A_f @ np.linalg.lstsq(A_f, y, rcond=None)[0]
    return float(np.max(np.abs(A[:, ~free].T @ r))) / A.shape[0]


class HarmDenseBase(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, ENV)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cwd = os.getcwd()
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="harm_dense_")
        os.chdir(self.tmp)                     # RFE writes rfe_support.npy
        self.addCleanup(os.chdir, self.cwd)
        warnings.simplefilter("ignore")
        self.A, self.y, self.beta, self.free = make_problem()

    def assert_free_kept(self, coef, free=None):
        free = self.free if free is None else free
        self.assertTrue(np.all(coef[free] != 0.0),
                        "a free (harmonic) coefficient was zeroed: %s" % coef[free])


class GridTests(HarmDenseBase):
    def test_mask_normalization(self):
        self.assertIsNone(opt._as_free_mask(None, 5))
        self.assertIsNone(opt._as_free_mask(np.zeros(5, bool), 5))
        np.testing.assert_array_equal(opt._as_free_mask([0, 2], 5),
                                      [True, False, True, False, False])
        with self.assertRaises(ValueError):
            opt._as_free_mask(np.ones(5, bool), 5)
        with self.assertRaises(ValueError):
            opt._as_free_mask(np.ones(4, bool), 5)

    def test_alpha_max_is_the_penalized_block_threshold(self):
        a_ref = penalized_alpha_max(self.A, self.y, self.free)
        grid = opt.derive_alpha_grid(self.A, self.y, nalpha=11, decades=4.0,
                                     unpenalized=self.free)
        self.assertAlmostEqual(grid.max() / a_ref, 1.0, places=9)
        # The unmasked top is set by the harmonic gradient and sits far higher.
        grid_all = opt.derive_alpha_grid(self.A, self.y, nalpha=11, decades=4.0)
        self.assertGreater(grid_all.max() / a_ref, 10.0)
        # KKT: just above alpha_max the penalized block is exactly zero and the
        # free block is its plain OLS; just below, something enters.
        pw = opt._free_penalty_weights(self.free)
        G, b, _ = opt._compute_gram(self.A, self.y)
        x_hi = opt._fista_lasso(self.A, self.y, a_ref * 1.001, penalty_weights=pw,
                                gram=(G, b), tol=1e-12, max_iter=200000,
                                x0=opt._free_block_x0(G, b, self.free))
        self.assertTrue(np.all(x_hi[~self.free] == 0.0))
        ols_f = np.linalg.lstsq(self.A[:, self.free], self.y, rcond=None)[0]
        np.testing.assert_allclose(x_hi[self.free], ols_f, rtol=1e-8, atol=1e-10)
        x_lo = opt._fista_lasso(self.A, self.y, a_ref * 0.9, penalty_weights=pw,
                                gram=(G, b), tol=1e-12, max_iter=200000)
        self.assertGreater(np.count_nonzero(x_lo[~self.free]), 0)

    def test_twolevel_grid_matches_dense(self):
        prime = sp.csr_matrix(self.A)
        A2 = opt.TwoLevelSM(prime, sp.eye(self.A.shape[1], format="csr"))
        g_dense = opt.derive_alpha_grid(self.A, self.y, nalpha=5, unpenalized=self.free)
        g_op = opt.derive_alpha_grid(A2, self.y, nalpha=5, unpenalized=self.free)
        np.testing.assert_allclose(g_op, g_dense, rtol=1e-7)


class LassoTests(HarmDenseBase):
    def _fit(self, method, A=None, **kw):
        kw.setdefault("cv", 3)
        kw.setdefault("rand_seed", 0)
        o = opt.Optimizer(method, unpenalized=self.free, **kw)
        o.fit(self.A if A is None else A, self.y)
        return o

    def test_lasso_equals_fwl_reference(self):
        a = 0.2 * penalized_alpha_max(self.A, self.y, self.free)
        with patch.dict(os.environ, {"PHEASY_LASSO_DEBIAS": "0",
                                     "PHEASY_LASSO_EDGE_RELAXED": "0",
                                     "PHEASY_COEF_ZERO_TOL": "0"}):
            o = self._fit("LASSO", alpha=[a], tol=1e-13, max_iter=400000)
        ref = fwl_reference(self.A, self.y, self.free, a)
        np.testing.assert_allclose(o.results["coef"], ref, rtol=1e-6, atol=1e-8)
        # same support in the penalized block, and it IS sparse
        np.testing.assert_array_equal(o.results["coef"][~self.free] != 0,
                                      ref[~self.free] != 0)
        self.assertLess(o.results["nnz_penalized"], P_3)
        self.assert_free_kept(o.results["coef"])

    def test_lasso_cv_keeps_free_block_and_reports_per_block(self):
        o = self._fit("LASSO", alpha_auto=True, tol=1e-9, max_iter=100000,
                      alpha=opt.derive_alpha_grid(self.A, self.y, nalpha=13,
                                                  unpenalized=self.free))
        c = o.results["coef"]
        self.assert_free_kept(c)
        self.assertEqual(o.results["unpenalized_columns"], P_H)
        self.assertEqual(o.results["nnz_unpenalized"], P_H)
        self.assertEqual(o.results["n_penalized"], P_3)
        self.assertEqual(o.results["nnz_penalized"], int(np.count_nonzero(c[~self.free])))

    def test_debias_and_zero_tol_never_drop_free_columns(self):
        # A free coefficient that is exactly zero in truth, and an aggressive
        # zero tolerance: the free block must survive both post-fit stages.
        beta = self.beta.copy()
        beta[0] = 0.0
        y = self.A @ beta
        a = 0.5 * penalized_alpha_max(self.A, y, self.free)
        with patch.dict(os.environ, {"PHEASY_COEF_ZERO_TOL": "1e-1"}):
            o = opt.Optimizer("LASSO", alpha=[a], cv=3, rand_seed=0, tol=1e-12,
                              max_iter=200000, unpenalized=self.free)
            o.fit(self.A, y)
        c = o.results["coef"]
        # tiny free coefficients (|beta| ~ 1e-4 < 1e-1) are NOT zeroed
        self.assertTrue(np.all(c[P_H - 2:P_H] != 0.0))
        # debias support contained the whole free block (OLS on it is exact)
        self.assertTrue(o.results.get("debias_accepted", True))

    def test_all_zero_penalized_block_warns(self):
        a = 10.0 * penalized_alpha_max(self.A, self.y, self.free)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            o = self._fit("LASSO", alpha=[a], tol=1e-10, max_iter=50000)
        self.assertEqual(o.results["nnz_penalized"], 0)
        self.assert_free_kept(o.results["coef"])
        self.assertTrue(any("three-phonon" in str(w.message) for w in rec))

    def test_alasso_weights_and_weighted_grid(self):
        with patch.dict(os.environ, {"PHEASY_ALASSO_RIDGE_ALPHA": "1e-6"}):
            o = self._fit("ALASSO", alpha_auto=True, nalpha=9, decades=4.0,
                          tol=1e-9, max_iter=100000)
        m = o._model
        self.assertTrue(np.all(m._weights[self.free] == 0.0))
        self.assertTrue(np.all(m._weights[~self.free] > 0.0))
        al = np.asarray(m.alphas_)
        self.assertTrue(np.all(np.isfinite(al)))
        # the old 1e-300-guard grid spanned ~300 decades; this one is sane
        self.assertLess(np.log10(al.max() / al.min()), 12.0)
        # top of the weighted grid = weighted KKT threshold of the penalized
        # block on the free-block OLS residual
        A_f = self.A[:, self.free]
        r = self.y - A_f @ np.linalg.lstsq(A_f, self.y, rcond=None)[0]
        g = np.abs(self.A.T @ r)[~self.free] / m._weights[~self.free]
        self.assertAlmostEqual(al.max() / (g.max() / self.A.shape[0]), 1.0, places=6)
        self.assert_free_kept(o.results["coef"])

    def test_twolevel_operator_lasso_matches_dense(self):
        a = 0.2 * penalized_alpha_max(self.A, self.y, self.free)
        env = {"PHEASY_LASSO_DEBIAS": "0", "PHEASY_LASSO_EDGE_RELAXED": "0",
               "PHEASY_COEF_ZERO_TOL": "0", "PHEASY_GRAM_MAX_GB": "0"}
        prime = sp.csr_matrix(self.A)
        A2 = opt.TwoLevelSM(prime, sp.eye(self.A.shape[1], format="csr"))
        with patch.dict(os.environ, env):
            o = self._fit("LASSO", A=A2, alpha=[a], tol=1e-11, max_iter=400000)
        ref = fwl_reference(self.A, self.y, self.free, a)
        np.testing.assert_allclose(o.results["coef"], ref, rtol=1e-5, atol=1e-7)
        self.assert_free_kept(o.results["coef"])


class IllConditionedTests(HarmDenseBase):
    """Raw (unstandardized) FC2/FC3 column scales.

    Iterating the free block inside FISTA is correct but a first-order method
    pays for the conditioning; the reduced solver (exact Schur-complement
    elimination of the free block + Jacobi scaling) must reach the FWL
    reference within an ordinary iteration budget.
    """

    def test_reduced_gram_fista_reaches_fwl_on_raw_scales(self):
        env = {"PHEASY_LASSO_DEBIAS": "0", "PHEASY_LASSO_EDGE_RELAXED": "0",
               "PHEASY_COEF_ZERO_TOL": "0"}
        for seed in (3, 11):
            A, y, free = make_illcond_problem(seed)
            G, b = A.T @ A, A.T @ y
            red = opt._FreeBlockReduction(G, b, free)
            self.assertGreater(np.linalg.cond(G) / np.linalg.cond(red.S), 50.0)
            amax = penalized_alpha_max(A, y, free)
            for frac in (0.05, 0.005):
                a = frac * amax
                ref = fwl_reference(A, y, free, a)
                # the old way: free block inside the iteration, same budget --
                # documents the failure mode (and that its KKT reads ~1e-5)
                info = {}
                x_old = opt._fista_lasso(A, y, a, penalty_weights=opt._free_penalty_weights(free),
                                         gram=(G, b), tol=1e-14, max_iter=3000,
                                         x0=opt._free_block_x0(G, b, free), _info=info,
                                         warn_nonconvergence=False, auto_floor=False)
                self.assertGreater(np.linalg.norm(x_old - ref) / np.linalg.norm(ref), 1e-2)
                with patch.dict(os.environ, env):
                    o = opt.Optimizer("LASSO", alpha=[a], cv=3, rand_seed=0, tol=1e-12,
                                      max_iter=3000, unpenalized=free)
                    o.fit(A, y)
                c = o.results["coef"]
                self.assertLess(np.linalg.norm(c - ref) / np.linalg.norm(ref), 1e-7,
                                "seed %d frac %g" % (seed, frac))
                self.assertEqual(o.results["regularized_solver_info"].get("harm_dense_reduction"),
                                 "schur_complement+jacobi")
                self.assert_free_kept(c, free)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_gpu_dense_reduced_fista_on_raw_scales(self):
        A, y, free = make_illcond_problem(11)
        a = 0.005 * penalized_alpha_max(A, y, free)
        ref = fwl_reference(A, y, free, a)
        with patch.object(gb, "available", return_value=True), \
                patch.object(gb, "enabled", return_value=True), \
                patch.object(gb, "device", return_value="cpu"), \
                patch.dict(os.environ, {"PHEASY_CV_TOL": "1e-10", "PHEASY_CV_MAX_ITER": "3000"}):
            m = gb.GpuLassoCV([a], 3, 1e-12, 3000, 0, group_size=ROWS,
                              penalty_weights=opt._free_penalty_weights(free))
            m.fit(A, y)
        self.assertLess(np.linalg.norm(m.coef_ - ref) / np.linalg.norm(ref), 1e-7)
        self.assertEqual(m.regularized_solver_info_.get("harm_dense_reduction"),
                         "schur_complement+jacobi")
        self.assert_free_kept(m.coef_, free)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_resident_internal_jacobi_on_raw_scales(self):
        # matrix-free path: the free block stays in the iteration, so the fix
        # there is the internal Jacobi scaling of an unstandardized fit.
        # Measured on this problem: 400 iterations to 4e-10 with it; without it
        # 2000 iterations leave 48 % error (at a KKT of 1e-5).
        A, y, free = make_illcond_problem(11)
        a = 0.005 * penalized_alpha_max(A, y, free)
        ref = fwl_reference(A, y, free, a)
        prime = sp.csr_matrix(A)
        ns = sp.eye(A.shape[1], format="csr")
        with patch.object(gb, "available", return_value=True), \
                patch.object(gb, "enabled", return_value=True), \
                patch.object(gb, "device", return_value="cpu"), \
                patch.object(gb, "_iterative_lstsq_tensor",
                             GpuEmulationTests._dense_lstsq_tensor), \
                patch.dict(os.environ, {"PHEASY_CV_TOL": "1e-12",
                                        "PHEASY_CV_MAX_ITER": "2000",
                                        "PHEASY_FISTA_AUTO_FLOOR": "0"}):
            m = gb.GpuTwoLevelLassoCV([a], 3, 1e-13, 2000, 0, group_size=ROWS,
                                      standardize=False, alpha_auto=False,
                                      unpenalized=free)
            m.fit(opt.TwoLevelSM(prime, ns), y)
        self.assertLess(np.linalg.norm(m.coef_ - ref) / np.linalg.norm(ref), 1e-7)
        self.assert_free_kept(m.coef_, free)


class EliminationAndBayesTests(HarmDenseBase):
    def test_rfe_and_tsqr_never_eliminate_free_columns(self):
        for method, env in (("RFE", {"PHEASY_RFE_STEP": "0.2",
                                     "PHEASY_RFE_MIN_FEATURES": "1"}),
                            ("RFE-OLS-TSQR", {"PHEASY_TSQR_STEP": "0.2",
                                              "PHEASY_TSQR_MIN_FEATURES": "2",
                                              "PHEASY_TSQR_CRITERION": "cv"})):
            with patch.dict(os.environ, env):
                o = opt.Optimizer(method, cv=4, rand_seed=0, unpenalized=self.free)
                o.fit(self.A, self.y)
            c = o.results["coef"]
            self.assert_free_kept(c)
            # something in the penalized block was actually eliminated, and the
            # true FC3 support survived
            self.assertLess(np.count_nonzero(c[~self.free]), P_3, method)
            truth = self.beta[~self.free] != 0
            self.assertTrue(np.all(c[~self.free][truth] != 0), method)

    def test_rfe_support_override_is_completed_with_free_block(self):
        sup = np.array([P_H + 0, P_H + 3])       # a saved support without FC2
        np.save("sup.npy", sup)
        with patch.dict(os.environ, {"PHEASY_RFE_SUPPORT_NPY": "sup.npy"}):
            o = opt.Optimizer("RFE", cv=3, rand_seed=0, unpenalized=self.free)
            o.fit(self.A, self.y)
        c = o.results["coef"]
        self.assert_free_kept(c)
        self.assertEqual(np.count_nonzero(c[~self.free]), 2)

    def test_ardr_keeps_free_block_under_a_harsh_threshold(self):
        # threshold 1e2: without the flat prior the tiny harmonic terms
        # (|beta| ~ 1e-4, lambda ~ 1/beta^2 ~ 1e7) would be pruned
        with patch.dict(os.environ, {"PHEASY_ARDR_THRESHOLD": "1e2"}):
            o = opt.Optimizer("ARDR", cv=3, rand_seed=0, tol=1e-6,
                              unpenalized=self.free)
            o.fit(self.A, self.y)
            ref = opt.Optimizer("ARDR", cv=3, rand_seed=0, tol=1e-6)
            ref.fit(self.A, self.y)
        c = o.results["coef"]
        self.assert_free_kept(c)
        self.assertLess(np.count_nonzero(c[~self.free]), P_3)
        # the reference (no mask) really does prune harmonic terms here
        self.assertTrue(np.any(ref.results["coef"][self.free] == 0.0))

    def test_rvm_keeps_free_block(self):
        with patch.dict(os.environ, {"PHEASY_RVM_THRESHOLD": "1e2"}):
            o = opt.Optimizer("RVM", unpenalized=self.free)
            o.fit(self.A, self.y)
            ref = opt.Optimizer("RVM")
            ref.fit(self.A, self.y)
        c = o.results["coef"]
        self.assert_free_kept(c)
        self.assertLess(np.count_nonzero(c[~self.free]), P_3)
        truth = self.beta[~self.free] != 0
        self.assertTrue(np.all(c[~self.free][truth] != 0))
        self.assertTrue(np.any(ref.results["coef"][self.free] == 0.0))

    def test_ols_and_ridge_ignore_the_mask(self):
        o = opt.Optimizer("OLS", unpenalized=self.free)
        o.fit(self.A, self.y)
        ref = opt.Optimizer("OLS")
        ref.fit(self.A, self.y)
        np.testing.assert_allclose(o.results["coef"], ref.results["coef"])
        self.assertNotIn("nnz_penalized", o.results)


@unittest.skipIf(torch is None, "torch not installed")
class GpuEmulationTests(HarmDenseBase):
    """GPU classes on CPU torch (numerical emulation, not CUDA evidence)."""

    def _gpu_patches(self):
        return (patch.object(gb, "available", return_value=True),
                patch.object(gb, "enabled", return_value=True),
                patch.object(gb, "device", return_value="cpu"))

    @staticmethod
    def _dense_lstsq_tensor(A, y, atol=1e-8, btol=1e-8, maxiter=5000, x0=None):
        """CPU stand-in for the CUDA-only CGLS (_iterative_lstsq_tensor refuses
        a non-CUDA operator by design): exact dense least squares on the
        materialized subset operator."""
        t = A.torch
        p = A.shape[1]
        eye = t.eye(p, dtype=y.dtype, device=y.device)
        M = t.stack([A.matvec(eye[:, j]) for j in range(p)], dim=1)
        x = t.linalg.lstsq(M, y.reshape(-1, 1)).solution.reshape(-1)
        return x, {"converged": True, "stop_reason": "emulated_dense_lstsq",
                   "n_iter": 0}

    def test_gpu_dense_lasso_matches_fwl(self):
        a = 0.2 * penalized_alpha_max(self.A, self.y, self.free)
        pw = opt._free_penalty_weights(self.free)
        p1, p2, p3 = self._gpu_patches()
        with p1, p2, p3, patch.dict(os.environ, {"PHEASY_CV_TOL": "1e-10",
                                                 "PHEASY_CV_MAX_ITER": "100000"}):
            m = gb.GpuLassoCV([a], 3, 1e-13, 400000, 0, group_size=ROWS,
                              penalty_weights=pw)
            m.fit(self.A, self.y)
        ref = fwl_reference(self.A, self.y, self.free, a)
        np.testing.assert_allclose(m.coef_, ref, rtol=1e-6, atol=1e-8)
        self.assert_free_kept(m.coef_)

    def test_resident_lasso_and_alasso_keep_free_block(self):
        prime = sp.csr_matrix(self.A)
        ns = sp.eye(self.A.shape[1], format="csr")
        a = 0.2 * penalized_alpha_max(self.A, self.y, self.free)
        ref = fwl_reference(self.A, self.y, self.free, a)
        p1, p2, p3 = self._gpu_patches()
        p4 = patch.object(gb, "_iterative_lstsq_tensor", self._dense_lstsq_tensor)
        env = {"PHEASY_CV_TOL": "1e-10", "PHEASY_CV_MAX_ITER": "20000",
               "PHEASY_FISTA_AUTO_FLOOR": "0"}
        with p1, p2, p3, p4, patch.dict(os.environ, env):
            m = gb.GpuTwoLevelLassoCV([a], 3, 1e-12, 200000, 0, group_size=ROWS,
                                      standardize=False, alpha_auto=False,
                                      unpenalized=self.free)
            m.fit(opt.TwoLevelSM(prime, ns), self.y)
        np.testing.assert_allclose(m.coef_, ref, rtol=1e-5, atol=1e-7)
        self.assert_free_kept(m.coef_)
        # auto grid: top = penalized-block threshold (standardize=False)
        with p1, p2, p3, p4, patch.dict(os.environ, env):
            m2 = gb.GpuTwoLevelLassoCV([1.0], 3, 1e-9, 50000, 0, group_size=ROWS,
                                       standardize=False, alpha_auto=True, nalpha=7,
                                       unpenalized=self.free)
            m2.fit(opt.TwoLevelSM(prime, ns), self.y)
        self.assertAlmostEqual(float(np.max(m2.alphas_)) /
                               penalized_alpha_max(self.A, self.y, self.free), 1.0, places=6)
        self.assert_free_kept(m2.coef_)
        # ALASSO: finite weighted grid, zero weight on the free block
        with p1, p2, p3, p4, patch.dict(os.environ, env):
            m3 = gb.GpuTwoLevelLassoCV([1.0], 3, 1e-9, 50000, 0, group_size=ROWS,
                                       standardize=False, adaptive=True,
                                       alpha_auto=True, nalpha=7,
                                       unpenalized=self.free)
            m3.fit(opt.TwoLevelSM(prime, ns), self.y)
        self.assertTrue(np.all(m3.penalty_weights_[self.free] == 0.0))
        self.assertTrue(np.all(np.isfinite(m3.alphas_)))
        self.assertLess(np.log10(np.max(m3.alphas_) / np.min(m3.alphas_)), 12.0)
        self.assert_free_kept(m3.coef_)


if __name__ == "__main__":
    unittest.main()
