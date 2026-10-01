#!/usr/bin/env python3
"""[FIX RFE-GRAM] regression tests for RFE / RFE-OLS-TSQR.

The large-system symptom: on operator / sparse input every RFE round ranked
the features and scored the folds with float32 iterative subset solves
(CGLS / LSMR / LSQR) that stop at a precision floor, so the CV compared solver
states rather than supports -- fit_scripts/fit_3090.sh records the same
round-0 support scoring CV_RMSE 2.505e-01 or 3.401e-01 depending only on the
solver tolerance.  The exact per-fold Gram engine must reproduce the exact
dense RFE (QR/SVD per subset) bit for bit in its selection.
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


def rfe_problem(seed=0, n_cfg=24, rows_per=30, p2=20, p3=100, k3=35, noise=0.02):
    """FC2-like (40x norm) and FC3-like columns, correlated, grouped rows."""
    r = np.random.default_rng(seed)
    n = n_cfg * rows_per
    L = r.standard_normal((n, 10))
    A = r.standard_normal((n, p2 + p3)) + 0.7 * L @ r.standard_normal((10, p2 + p3))
    A[:, :p2] *= 40.0
    c = np.zeros(p2 + p3)
    c[:p2] = r.standard_normal(p2) * 0.5
    nz = r.choice(np.arange(p2, p2 + p3), k3, replace=False)
    c[nz] = r.standard_normal(k3) * np.geomspace(20, 0.05, k3)
    y = A @ c
    y = y + noise * np.std(y) * r.standard_normal(n)
    return A, y, rows_per


def fit_rfe(A, y, env=None, cls=None, **kw):
    cls = cls or opt.PheasyRFECV
    kw.setdefault("step", 0.1)
    model = cls(cv=4, n_jobs=1, verbose=True, random_state=0, patience=4, **kw)
    buf = io.StringIO()
    with patch.dict(os.environ, env or {}), warnings.catch_warnings(), \
            contextlib.redirect_stdout(buf):
        warnings.simplefilter("ignore")
        model.fit(A, y)
    rounds = [(int(l.split("n_active=")[1].split()[0]), float(l.split("CV_RMSE=")[1].split()[0]))
              for l in buf.getvalue().splitlines() if "Round" in l and "CV_RMSE" in l]
    return model, rounds, buf.getvalue()


class GramEngineTests(unittest.TestCase):
    def setUp(self):
        self.A, self.y, g = rfe_problem()
        self.env = {"PHEASY_CV_GROUP_SIZE": str(g), "PHEASY_GPU_MODE": "cpu"}
        self.ref, self.ref_rounds, _ = fit_rfe(self.A, self.y, self.env)

    def _assert_matches_reference(self, got, rounds):
        self.assertEqual([r[0] for r in rounds], [r[0] for r in self.ref_rounds])
        np.testing.assert_allclose([r[1] for r in rounds], [r[1] for r in self.ref_rounds],
                                   rtol=1e-5)
        np.testing.assert_array_equal(got.support_, self.ref.support_)
        np.testing.assert_allclose(got.coef_, self.ref.coef_, rtol=0,
                                   atol=1e-6 * np.abs(self.ref.coef_).max())

    def test_float32_sparse_and_twolevel_match_exact_dense_rfe(self):
        A32 = self.A.astype(np.float32)
        inputs = {"csr32": sp.csr_matrix(A32),
                  "twolevel32": opt.TwoLevelSM(sp.csr_matrix(A32),
                                               sp.eye(A32.shape[1], format="csr",
                                                      dtype=np.float32))}
        for name, M in inputs.items():
            with self.subTest(name):
                got, rounds, log = fit_rfe(M, self.y, self.env)
                meta = got.backend_metadata_
                self.assertEqual(meta["subset_solver"], "cpu_gram_exact")
                self.assertEqual(meta["gram_from_round"], 0)
                self.assertGreater(meta["gram_exact_rounds"], len(rounds))
                self.assertIn("exact Gram CV from round 0", log)
                self._assert_matches_reference(got, rounds)

    def test_iterative_float32_path_is_what_the_engine_replaces(self):
        # The same float32 sparse input on the old iterative path (engine off)
        # delivers an LSQR iterate; the engine delivers the exact refit.
        M = sp.csr_matrix(self.A.astype(np.float32))
        env = dict(self.env, PHEASY_MAX_DENSE="1000", PHEASY_RFE_FINAL_TSQR="0",
                   PHEASY_RFE_GRAM_GB="0")
        old, _, _ = fit_rfe(M, self.y, env)
        self.assertEqual(old.backend_metadata_["subset_solver"], "cpu")
        sup = old.support_
        exact = np.linalg.lstsq(self.A[:, sup], self.y, rcond=None)[0]
        err_old = np.linalg.norm(old.coef_[sup] - exact) / np.linalg.norm(exact)
        new, _, _ = fit_rfe(M, self.y, dict(self.env, PHEASY_MAX_DENSE="1000"))
        sup_n = new.support_
        exact_n = np.linalg.lstsq(self.A[:, sup_n], self.y, rcond=None)[0]
        err_new = np.linalg.norm(new.coef_[sup_n] - exact_n) / np.linalg.norm(exact_n)
        self.assertLess(err_new, 1e-6)
        self.assertGreater(err_old, 10 * err_new)

    def test_late_engagement_is_exact_from_that_round(self):
        M = sp.csr_matrix(self.A)
        K = 4
        p_cut = 80                          # engages once n_active <= 80
        env = dict(self.env, PHEASY_RFE_GRAM_GB=repr((K + 4) * 8 * p_cut ** 2 / 1e9))
        got, rounds, log = fit_rfe(M, self.y, env)
        meta = got.backend_metadata_
        self.assertGreater(meta["gram_from_round"], 0)
        self.assertLessEqual(meta["gram_n_features"], p_cut)
        # float64 input: the iterative rounds before it agree with the exact
        # path too, so the whole run still reproduces the reference
        self._assert_matches_reference(got, rounds)

    def test_ridge_rfe_keeps_the_penalty_in_the_final_refit(self):
        M = sp.csr_matrix(self.A)
        alpha = 50.0
        got, _, _ = fit_rfe(M, self.y, self.env, ridge_alpha=alpha)
        sup = got.support_
        As = self.A[:, sup]
        ridge = np.linalg.solve(As.T @ As + alpha * np.eye(sup.sum()), As.T @ self.y)
        np.testing.assert_allclose(got.coef_[sup], ridge, rtol=1e-7,
                                   atol=1e-9 * np.abs(ridge).max())

    def test_bic_criterion_uses_the_exact_rss(self):
        M = sp.csr_matrix(self.A)
        env = dict(self.env, PHEASY_TSQR_CRITERION="bic")
        dense, _, _ = fit_rfe(self.A, self.y, env, cls=opt.PheasyRFE_OLS_TSQR, min_features=10)
        got, _, _ = fit_rfe(M, self.y, env, cls=opt.PheasyRFE_OLS_TSQR, min_features=10)
        self.assertEqual(got.backend_metadata_["subset_solver"], "cpu_gram_exact")
        np.testing.assert_array_equal(got.support_, dense.support_)
        self.assertAlmostEqual(got.best_rmse_cv_, dense.best_rmse_cv_, delta=1e-6 * dense.best_rmse_cv_)

    def test_gram_solve_rank_deficient_support_returns_min_norm(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 6))
        X[:, 5] = X[:, 4]                   # exactly duplicated column
        yv = rng.normal(size=50)
        x, how = opt._gram_solve(lambda: X.T @ X, X.T @ yv)
        self.assertEqual(how, "eigh_min_norm")
        ref = np.linalg.lstsq(X, yv, rcond=None)[0]
        np.testing.assert_allclose(X @ x, X @ ref, rtol=1e-9, atol=1e-10)
        self.assertAlmostEqual(x[4], x[5], places=10)


class RfeDispatchTests(unittest.TestCase):
    def test_sample_weights_are_refused(self):
        A, y, _ = rfe_problem(n_cfg=6, p3=10, k3=4)
        with self.assertRaises(NotImplementedError):
            opt.PheasyRFECV(cv=3, verbose=False).fit(A, y, sample_weight=np.ones(len(y)))

    def _twolevel(self):
        A, y, g = rfe_problem(n_cfg=6, p3=10, k3=4)
        return opt.TwoLevelSM(sp.csr_matrix(A), sp.eye(A.shape[1], format="csr")), y, g

    def test_explicit_resident_request_serializes_instead_of_failing(self):
        # pheasy_fit.sh exports PHEASY_N_JOBS=$NCPU; in auto GPU mode an explicit
        # PHEASY_GPU_RFE_RESIDENT=1 used to die with "requires ... n_jobs=1"
        # before the resident operator was even tried.
        T, y, g = self._twolevel()
        env = {"PHEASY_GPU_MODE": "auto", "PHEASY_GPU_RFE_RESIDENT": "1",
               "PHEASY_N_JOBS": "4", "PHEASY_RFE_GRAM_GB": "0",
               "PHEASY_CV_GROUP_SIZE": str(g)}
        with patch.dict(os.environ, env), patch.object(gb, "available", return_value=True), \
                patch.object(gb, "enabled", return_value=True), \
                patch.object(gb, "GpuTwoLevelOperator",
                             side_effect=RuntimeError("injected setup failure")), \
                warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(RuntimeError, "injected setup failure"):
                opt.PheasyRFECV(cv=3, verbose=False).fit(T, y)

    def test_exact_engine_replaces_the_resident_operator_when_it_fits(self):
        T, y, g = self._twolevel()
        env = {"PHEASY_GPU_MODE": "required", "PHEASY_RFE_N_JOBS": "1",
               "PHEASY_CV_GROUP_SIZE": str(g)}
        with patch.dict(os.environ, env), patch.object(gb, "available", return_value=True), \
                patch.object(gb, "enabled", return_value=True), \
                patch.object(gb, "GpuTwoLevelOperator",
                             side_effect=AssertionError("resident operator built")), \
                warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            m = opt.Optimizer("RFE", use_gpu=True)
            m.fit(T, y)
        self.assertEqual(m.results["execution_backend"], "cpu_rfe_gram_exact")
        self.assertGreater(m.results["rfe_gram_exact_rounds"], 0)


if __name__ == "__main__":
    unittest.main()
