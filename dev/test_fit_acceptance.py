import os
import unittest
import warnings
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
from core.optimizer import Optimizer, TwoLevelSM

class FitAcceptanceTest(unittest.TestCase):
    def _make_twolevel(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((40, 6))
        x_true = rng.standard_normal(6)
        y = A @ x_true + 1e-4 * rng.standard_normal(40)
        sm = sp.csr_matrix(A)
        ns = sp.eye(6, format="csr")
        return TwoLevelSM(sm, ns), y

    def test_converged_ols_is_accepted(self):
        A, y = self._make_twolevel()
        opt = Optimizer("OLS", max_iter=5000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.fit(A, y)
        self.assertIn("fit_accepted", opt.results)
        self.assertIn("status", opt.results)
        self.assertTrue(opt.results["fit_accepted"])
        self.assertEqual(opt.results["status"], "fit_returned")

    def test_iteration_limited_ols_is_not_accepted(self):
        A, y = self._make_twolevel()
        opt = Optimizer("OLS", max_iter=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.fit(A, y)
        self.assertFalse(opt.results["fit_accepted"])
        self.assertEqual(opt.results["status"], "fit_returned_not_accepted")

    def test_converged_ridge_certifies_the_delivered_alpha(self):
        """CPU LSMR RIDGE must publish a certificate for the alpha it returns."""
        A, y = self._make_twolevel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Pin the CPU LSMR branch: on a CUDA box the resident path would
            # satisfy the fit while leaving the certificate untested.
            with patch.dict(os.environ, {"PHEASY_RIDGE_LCURVE": "1",
                                         "PHEASY_GPU_RIDGE_RESIDENT": "0"}):
                opt = Optimizer("RIDGE", alpha=[1e-1, 1e-3, 1e-5], cv=2)
                opt.fit(A, y)
        info = opt.results.get("regularized_solver_info")
        self.assertIsNotNone(info, "RIDGE solve published no certificate")
        self.assertEqual(info["backend"], "cpu_lsmr_ridge")
        # The L-curve walk ends on the SMALLEST alpha, so a certificate taken
        # from the operator after the loop describes a fit that was discarded.
        self.assertEqual(float(info["alpha"]), float(opt.results["alpha"]))
        self.assertTrue(opt.results["fit_accepted"], info)
        self.assertEqual(opt.results["execution_backend"], "cpu_lsmr_ridge")

    def test_iteration_limited_ridge_is_not_accepted(self):
        """A RIDGE solve that hit its cap is not a certified fit on CPU either."""
        A, y = self._make_twolevel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch.dict(os.environ, {"PHEASY_LSQR_MAXITER": "1",
                                         "PHEASY_RIDGE_LCURVE": "0",
                                         "PHEASY_GPU_RIDGE_RESIDENT": "0"}):
                opt = Optimizer("RIDGE", alpha=[1e-2, 1e-3], cv=2)
                opt.fit(A, y)
        info = opt.results.get("regularized_solver_info")
        self.assertIsNotNone(info, "the iteration-capped solve must be reported")
        self.assertIs(info["converged"], False)
        self.assertEqual(info["istop"], 7)
        self.assertFalse(opt.results["fit_accepted"])
        self.assertEqual(opt.results["status"], "fit_returned_not_accepted")

    def _dense_lasso_problem(self):
        rng = np.random.default_rng(5)
        A = rng.standard_normal((80, 10))
        y = A @ rng.standard_normal(10) + 1e-3 * rng.standard_normal(80)
        return A, y

    def test_dense_lasso_publishes_a_certificate(self):
        """The sklearn coordinate-descent backend iterated too; certify it.

        It was the only LASSO path that reported nothing, so a coordinate
        descent that stopped at max_iter read as fit_accepted=True while the
        FISTA and resident backends vetoed on the same condition.
        PHEASY_GPU_LASSO=0 pins the sklearn branch on a CUDA box, where
        _lasso_backend would otherwise select the GPU Gram solver.
        """
        A, y = self._dense_lasso_problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch.dict(os.environ, {"PHEASY_GPU_LASSO": "0"}):
                opt = Optimizer("LASSO", alpha=[1e-2, 1e-3], cv=2, max_iter=5000)
                opt.fit(A, y)
        info = opt.results.get("regularized_solver_info")
        self.assertIsNotNone(info, "the dense CD backend must certify")
        self.assertEqual(info["backend"], "cpu_dense_coordinate_descent")
        self.assertTrue(info["converged"], info)
        self.assertTrue(opt.results["fit_accepted"])

    def test_dense_alasso_publishes_a_certificate_too(self):
        """The adaptive branch runs its own LassoCV and used to report nothing.

        Measured on c7 it returned regularized_solver_info={} and
        execution_backend=None, so the entire adaptive solve was invisible to the
        acceptance gate.
        """
        A, y = self._dense_lasso_problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch.dict(os.environ, {"PHEASY_GPU_LASSO": "0"}):
                opt = Optimizer("ALASSO", alpha=[1e-2, 1e-3], cv=2, max_iter=5000,
                                alpha_auto=False)
                opt.fit(A, y)
        info = opt.results.get("regularized_solver_info")
        self.assertTrue(info, "the adaptive dense backend must certify")
        self.assertEqual(info["backend"], "cpu_dense_coordinate_descent")
        self.assertIn("weight_dispersion", info)
        self.assertTrue(opt.results["fit_accepted"], info)

    def test_dense_lasso_that_hits_its_cap_is_not_accepted(self):
        A, y = self._dense_lasso_problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch.dict(os.environ, {"PHEASY_GPU_LASSO": "0"}):
                opt = Optimizer("LASSO", alpha=[1e-2, 1e-3], cv=2, max_iter=1)
                opt.fit(A, y)
        info = opt.results.get("regularized_solver_info")
        self.assertIsNotNone(info)
        self.assertFalse(info["converged"], info)
        self.assertEqual(info["stop_reason"], "iteration_limit")
        self.assertFalse(opt.results["fit_accepted"])
        self.assertEqual(opt.results["status"], "fit_returned_not_accepted")

    def test_lasso_debias_reports_whether_the_refit_was_kept(self):
        """A declared post-fit stage that did not run must not read as if it had.
        The residual guard correctly rejects a corrupted support refit, but it did
        so silently: on c7 float32 the pristine CGLS debias came back
        invalid_search_direction, the guard kept the L1 coefficients, and the
        results still reported debias_backend=gpu_cgls with nothing saying the
        shrinkage was still in place.
        """
        A, y = self._make_twolevel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt = Optimizer("LASSO", alpha=[1e-3, 1e-4], cv=2, max_iter=2000)
            opt.fit(A, y)
        r = opt.results
        self.assertTrue(r.get("fit_accepted"), r.get("status"))
        self.assertIn("debias_accepted", r)
        self.assertIn(r["debias_backend"], ("cpu_lsmr", "gpu_cgls", "cpu_gram_solve"))
        self.assertIn("debias_stage", r)
        if r["debias_accepted"]:
            self.assertLessEqual(r["debias_residual_refit"],
                             r["debias_residual_shrunk"])
        else:
            self.assertIn("debias_fallback_reason", r)

    def test_rfe_does_not_bake_the_ambient_precision_into_its_tolerance(self):
        """RFE has no operator at construction, so it must not guess a floor.

        Applying _lsmr_tol there made _array_precision fall back to
        PHEASY_SM_DTYPE and baked the ambient precision into the model default,
        which then over-raised the tolerance of every later float64 subset.
        """
        A, y = self._make_twolevel()          # float64 factors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Pin the CPU RFE path: on a CUDA box the resident path would take
            # over and the tolerance default under test would never be built.
            with patch.dict(os.environ, {"PHEASY_SM_DTYPE": "float32",
                                         "PHEASY_RFE_MIN_FEATURES": "1",
                                         "PHEASY_GPU_RFE_RESIDENT": "0"}):
                opt = Optimizer("RFE", cv=2)
                opt.fit(A, y)
        self.assertEqual(float(opt._model.lsmr_atol), 1e-8)
        self.assertEqual(float(opt._model.lsmr_btol), 1e-8)

if __name__ == "__main__":
    unittest.main()
