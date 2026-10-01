#!/usr/bin/env python3
"""[FIX ARD-YSTD] / [FIX RVM-CYCLE] regression tests.

* ARDR / RVM prune on an absolute precision threshold (lambda_t = 1e4, from
  hiphive / trainstation).  trainstation standardizes the columns AND the
  target; --std here scaled only the columns, so the threshold sat in force
  units and the selected model depended on the units of F (measured: 18 of 1303
  features and 17 % relative error, against 586 and 0.86 % without --std).
* fast_rvm clamped a re-estimated precision above alpha_ceiling back onto the
  ceiling the basis already sat on: a no-op that won every step until
  max_steps (and the add path had the same stall).
* [FIX ARD-CV] lambda_t is chosen by grouped CV from per-fold Grams unless one
  value is given (PHEASY_ARDR_THRESHOLD / PHEASY_RVM_THRESHOLD).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import optimizer as opt          # noqa: E402
from core.fast_rvm import fast_rvm         # noqa: E402


def problem(seed=0, n=600, p=60, k=25, noise=1e-3):
    r = np.random.default_rng(seed)
    A = r.standard_normal((n, p)) * np.geomspace(0.05, 0.5, p)   # displacement-like scales
    w = np.zeros(p)
    w[r.choice(p, k, replace=False)] = r.standard_normal(k) * np.geomspace(3, 0.01, k)
    y = A @ w
    return A, y + noise * np.std(y) * r.standard_normal(n)


def fit(method, A, y, **env):
    m = opt.Optimizer(method, cv=3, rand_seed=0, use_gpu=False, standardize=True)
    with patch.dict(os.environ, dict({"PHEASY_GPU_MODE": "cpu", "PHEASY_CV_GROUP_SIZE": "6"}, **env)), \
            warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        m.fit(A, y)
    return m


class TargetStandardizationTests(unittest.TestCase):
    def setUp(self):
        self.A, self.y = problem()

    def test_selection_does_not_depend_on_force_units(self):
        for method in ("ARDR", "RVM"):
            with self.subTest(method=method):
                a = fit(method, self.A, self.y)
                b = fit(method, self.A, 1000.0 * self.y)
                ca, cb = np.asarray(a.results["coef"]), np.asarray(b.results["coef"])
                np.testing.assert_array_equal(ca != 0, cb != 0)
                np.testing.assert_allclose(cb, 1000.0 * ca, rtol=1e-6,
                                           atol=1e-9 * np.abs(cb).max())
                self.assertAlmostEqual(b.results["ard_y_scale"] / a.results["ard_y_scale"],
                                       1000.0, delta=1e-6)

    def test_unscaled_target_was_unit_dependent(self):
        # PHEASY_ARD_YSTD=0 with the fixed lambda_t = 1e4 is the old behaviour:
        # same data in other units, different model
        old = dict(PHEASY_ARD_YSTD="0", PHEASY_ARDR_THRESHOLD="1e4")
        a = fit("ARDR", self.A, self.y, **old)
        b = fit("ARDR", self.A, 1000.0 * self.y, **old)
        self.assertNotEqual(int(np.count_nonzero(a.results["coef"])),
                            int(np.count_nonzero(b.results["coef"])))

    def test_reported_fit_quality_is_in_force_units(self):
        # the model is fitted on y / std(y): its reported RMSE (CV or in-sample)
        # must come back in the units of y, i.e. follow a change of units exactly
        for method in ("ARDR", "RVM"):
            with self.subTest(method=method):
                a = fit(method, self.A, self.y)
                b = fit(method, self.A, 1000.0 * self.y)
                self.assertAlmostEqual(b._model.best_rmse_cv_ / a._model.best_rmse_cv_,
                                       1000.0, delta=1e-3)
                self.assertAlmostEqual(b.metrics["rmse_path_mean"] / a.metrics["rmse_path_mean"],
                                       1000.0, delta=1e-3)
        m = fit("RVM", self.A, self.y, PHEASY_RVM_THRESHOLD="1e4")   # one lambda_t: in-sample
        rmse = float(np.sqrt(np.mean((self.A @ np.asarray(m.results["coef"]) - self.y) ** 2)))
        self.assertAlmostEqual(m._model.best_rmse_cv_ / rmse, 1.0, delta=1e-6)


class ThresholdCvTests(unittest.TestCase):
    def setUp(self):
        self.A, self.y = problem(seed=3)

    def test_ardr_cv_matches_explicit_fold_refits(self):
        m = fit("ARDR", self.A, self.y)
        info = m.results["regularized_solver_info"]
        self.assertTrue(m._model.cv_evaluated)
        self.assertEqual(sorted(float(t) for t in info["threshold_cv"]),
                         sorted(opt._ARD_LAMBDA_GRID))
        rmse = {float(t): v for t, v in info["threshold_cv"].items()}
        self.assertEqual(m._model.threshold_, min(rmse, key=rmse.get))
        # the per-fold Gram bookkeeping equals refitting on the training rows
        cs = np.linalg.norm(self.A, axis=0) / np.sqrt(self.A.shape[0])
        X, s = self.A / cs, float(np.std(self.y))
        yv = self.y / s
        splits = opt._make_cv_splits(X.shape[0], 3, 0, 6)
        mse = []
        for tr, va in splits:
            with contextlib.redirect_stdout(io.StringIO()):
                c = opt._ardr_evidence_gram(X[tr].T @ X[tr], X[tr].T @ yv[tr], float(yv[tr] @ yv[tr]),
                                            len(tr), float(np.var(yv[tr])),
                                            threshold_lambda=m._model.threshold_,
                                            max_iter=info["maxiter"], tol=info["tol"])[0]
            mse.append(np.mean((X[va] @ c - yv[va]) ** 2))
        self.assertAlmostEqual(float(np.sqrt(np.mean(mse))) * s,
                               rmse[m._model.threshold_], delta=1e-6 * rmse[m._model.threshold_])

    def test_rvm_cv_selects_from_the_grid(self):
        m = fit("RVM", self.A, self.y)
        info = m.results["regularized_solver_info"]
        self.assertTrue(m._model.cv_evaluated)
        rmse = {float(t): v for t, v in info["threshold_cv"].items()}
        self.assertEqual(sorted(rmse), sorted(opt._ARD_LAMBDA_GRID))
        self.assertEqual(info["threshold_lambda"], min(rmse, key=rmse.get))

    def test_explicit_threshold_disables_the_cv(self):
        # one lambda_t: no grid is searched (dense ARDR still reports the CV
        # score of that one value, as before)
        for method, key in (("ARDR", "PHEASY_ARDR_THRESHOLD"), ("RVM", "PHEASY_RVM_THRESHOLD")):
            with self.subTest(method=method):
                m = fit(method, self.A, self.y, **{key: "1e5"})
                info = m.results["regularized_solver_info"]
                self.assertIsNone(info.get("threshold_cv"))
                self.assertEqual(info["threshold_lambda"], 1e5)

    def test_no_memory_for_fold_grams_falls_back_to_one_threshold(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = opt.Optimizer("ARDR", cv=3, rand_seed=0, use_gpu=False, standardize=True)
            with patch.dict(os.environ, {"PHEASY_GPU_MODE": "cpu", "PHEASY_CV_GROUP_SIZE": "6",
                                         "PHEASY_EXACT_GRAM_GB": "1e-9"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                m.fit(self.A, self.y)
        self.assertFalse(m._model.cv_evaluated)
        self.assertEqual(m._model.threshold_, 1e4)
        self.assertTrue(any("lambda_t CV unavailable" in str(x.message) for x in w))


class RvmCycleTests(unittest.TestCase):
    def test_precision_at_the_ceiling_does_not_stall_the_loop(self):
        # seed 2 stalled the old loop at max_steps (an add / re-estimate whose
        # precision lands above alpha_ceiling was a no-op that kept winning)
        r = np.random.default_rng(2)
        n, p, k = 200, 60, 20
        X = r.standard_normal((n, p))
        w = np.zeros(p)
        w[:k] = r.standard_normal(k) * np.geomspace(3, 1e-3, k)
        y = X @ w + 1e-3 * r.standard_normal(n)
        out = fast_rvm(X.T @ X, X.T @ y, float(y @ y), n, y_var=float(np.var(y)),
                       beta_iters=3, alpha_ceiling=1e6, max_steps=3000)
        self.assertTrue(out["converged"])
        self.assertLess(out["n_steps"], 3000)


if __name__ == "__main__":
    unittest.main()
