#!/usr/bin/env python3
"""[FIX P46] The automatic LASSO alpha-grid policy, and its two failure modes.

What this pins down (all CPU, no CUDA):

  grid TOP    alpha_max must be the exact KKT threshold max_j|X_j^T y|/n.
  grid BOTTOM overdetermined problems must reach the under-regularized regime
              (>= 6 decades, ALASSO P37's rule), underdetermined ones must not.
  density     widening the span must not coarsen the step.
  pinned      alpha* at the grid bottom == "no sparsity supported": the relaxed
              refit must run EVEN IF the wrapper disabled debias (that is how the
              shipped Mg8C120 v4 fit lost 6x generalization), and the result must
              match the unpenalized solution.
  sparse      on genuinely sparse data the policy must be a NO-OP.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import optimizer as opt


class TestGridPolicy(unittest.TestCase):
    def test_top_of_grid_is_the_kkt_threshold(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((300, 40))
        y = A @ rng.normal(size=40) + 0.05 * rng.standard_normal(300)
        grid = opt.derive_alpha_grid(A, y, nalpha=20, decades=4.0)
        self.assertAlmostEqual(float(grid[-1]),
                               float(np.abs(A.T @ y).max() / A.shape[0]), places=12)

    def test_overdetermined_grid_reaches_six_decades(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((300, 40))
        y = rng.standard_normal(300)
        amax = float(np.abs(A.T @ y).max() / A.shape[0])
        grid = opt.derive_alpha_grid(A, y, nalpha=20, decades=4.0)
        self.assertLessEqual(float(grid[0]), amax * 10.0 ** -6.0 * (1 + 1e-12))
        # density is preserved: (nalpha-1)/4 points per decade
        self.assertGreaterEqual(len(grid), 1 + int(np.ceil(6 * (19 / 4.0))) - 1)

    def test_underdetermined_grid_is_left_alone(self):
        rng = np.random.default_rng(2)
        A = rng.standard_normal((30, 200))
        y = rng.standard_normal(30)
        amax = float(np.abs(A.T @ y).max() / A.shape[0])
        grid = opt.derive_alpha_grid(A, y, nalpha=20, decades=4.0)
        self.assertAlmostEqual(float(grid[0]), amax * 10.0 ** -4.0, places=12)

    def test_env_overrides(self):
        with patch.dict(os.environ, {"PHEASY_LASSO_GRID_FLOOR": "0"}):
            self.assertEqual(opt.lasso_grid_min_decades(300, 40, 4.0), 4.0)
        with patch.dict(os.environ, {"PHEASY_LASSO_GRID_MIN_DECADES": "9"}):
            self.assertEqual(opt.lasso_grid_min_decades(300, 40, 4.0), 9.0)
        # an explicitly wider user span is never narrowed
        self.assertEqual(opt.lasso_grid_min_decades(300, 40, 10.0), 10.0)
        # underdetermined stays untouched even with the floor enabled
        self.assertEqual(opt.lasso_grid_min_decades(30, 200, 4.0), 4.0)

    def test_grid_density_cap_and_bounds(self):
        g = opt.lasso_alpha_grid(1e-12, 1e-4, 20)
        self.assertEqual(len(g), 39)          # 1 + ceil(8 * 4.75)
        with patch.dict(os.environ, {"PHEASY_ALPHA_NMAX": "25"}):
            self.assertEqual(len(opt.lasso_alpha_grid(1e-12, 1e-4, 20)), 25)
        with self.assertRaises(ValueError):
            opt.lasso_alpha_grid(1.0, 0.1, 20)


class TestPinnedAlphaFallback(unittest.TestCase):
    """A pinned alpha* must ship the relaxed (unpenalized) coefficients."""

    def _problem(self, kind):
        rng = np.random.default_rng(7)
        n, p = 400, 60
        A = rng.standard_normal((n, p))
        if kind == "dense":                     # high SNR, no sparsity to find
            b = rng.normal(0, 0.5, p)
        else:                                   # genuinely sparse
            b = np.zeros(p)
            b[rng.choice(p, 6, replace=False)] = rng.uniform(1, 3, 6)
        y = A @ b + 0.02 * rng.standard_normal(n)
        return A, y, b

    def test_pinned_alpha_forces_relaxed_refit_even_with_debias_off(self):
        A, y, b = self._problem("dense")
        ols = np.linalg.lstsq(A, y, rcond=None)[0]
        with patch.dict(os.environ, {"PHEASY_LASSO_DEBIAS": "0",
                                     "PHEASY_LASSO_EDGE_RELAXED": "1"}):
            model = opt.Optimizer("LASSO", alpha=[1e-12, 2e-12], alpha_auto=False,
                                  cv=2, tol=1e-10, max_iter=20000, use_gpu=False)
            model.fit(A, y)
            res = model.results
            self.assertTrue(res["alpha_at_grid_edge"])
            self.assertFalse(res["sparsity_supported"])
            self.assertNotEqual(res.get("debias_backend", "disabled"), "disabled")
            self.assertIn("debias_forced_reason", res)
            coef = np.asarray(res["coef"], dtype=float)
            # the relaxed refit must be the least-squares solution
            self.assertLess(np.linalg.norm(coef - ols) / np.linalg.norm(ols), 1e-3)

    def test_sparse_problem_is_a_no_op(self):
        A, y, b = self._problem("sparse")
        with patch.dict(os.environ, {"PHEASY_LASSO_DEBIAS": "0"}):
            model = opt.Optimizer("LASSO", nalpha=20, alpha_auto=True, cv=2,
                                  tol=1e-10, max_iter=20000, use_gpu=False)
            model.fit(A, y)
            res = model.results
            self.assertFalse(res["alpha_at_grid_edge"])
            self.assertNotIn("debias_forced_reason", res)
            self.assertEqual(res.get("debias_backend", "disabled"), "disabled")
            # and it still beats OLS on the true coefficients
            ols = np.linalg.lstsq(A, y, rcond=None)[0]
            coef = np.asarray(res["coef"], dtype=float)
            self.assertLess(np.linalg.norm(coef - b), np.linalg.norm(ols - b))


if __name__ == "__main__":
    unittest.main(verbosity=2)
