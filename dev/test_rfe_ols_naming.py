#!/usr/bin/env python3
"""[RFE-OLS] canonical method name and the RFE-OLS-TSQR feature-count rule.

* "RFE-OLS" is the canonical name.  The CLI still maps "RFE" to it; the shell
  wrappers (pheasy_fit.sh, fit_3090.sh) accept only the full name and also
  take ARDR / RVM.
* RFE-OLS-TSQR shares RFE-OLS's elimination and differs in how the feature
  count is chosen: AIC by default (PHEASY_TSQR_CRITERION=aic|bic|cv).
* The AIC/BIC n is the number of force-component rows: with the
  configuration count (PHEASY_BIC_N_EFF=groups) n is tens while k is hundreds
  and both criteria prune far too hard.
"""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import optimizer as opt          # noqa: E402


def problem(seed=0, n_cfg=24, rows_per=40, p=120, k=60, noise=0.02):
    """Grouped rows, k real features with geometric magnitudes."""
    r = np.random.default_rng(seed)
    n = n_cfg * rows_per
    A = r.standard_normal((n, p)) + 0.5 * r.standard_normal((n, 8)) @ r.standard_normal((8, p))
    c = np.zeros(p)
    nz = r.choice(p, k, replace=False)
    c[nz] = r.standard_normal(k) * np.geomspace(5, 0.05, k)
    y = A @ c
    y = y + noise * np.std(y) * r.standard_normal(n)
    return A, y, rows_per


def quiet(fn):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return fn()


class CriterionTests(unittest.TestCase):
    def setUp(self):
        self.A, self.y, g = problem()
        self.env = {"PHEASY_CV_GROUP_SIZE": str(g), "PHEASY_GPU_MODE": "cpu"}

    def tsqr(self, **env):
        with patch.dict(os.environ, dict(self.env, **env)):
            for key in ("PHEASY_TSQR_CRITERION", "PHEASY_BIC_N_EFF"):
                if key not in env:
                    os.environ.pop(key, None)
            m = opt.PheasyRFE_OLS_TSQR(step=0.1, cv=4, min_features=5, n_jobs=1,
                                      verbose=False, random_state=0)
            quiet(lambda: m.fit(self.A, self.y))
        return m

    def test_default_criterion_is_aic_on_rows(self):
        m = self.tsqr()
        self.assertEqual(m.backend_metadata_["criterion"], "aic")
        self.assertEqual(m.backend_metadata_["ic_n_eff"], self.A.shape[0])
        for crit in ("aic", "bic", "cv"):
            self.assertEqual(self.tsqr(PHEASY_TSQR_CRITERION=crit).backend_metadata_["criterion"], crit)
        with patch.dict(os.environ, {"PHEASY_TSQR_CRITERION": "cv2"}):
            with self.assertRaises(ValueError):
                opt.PheasyRFE_OLS_TSQR()

    def test_cv_criterion_is_rfe_ols(self):
        # the old default: same elimination, same CV + 1-SE rule, same model
        tsqr = self.tsqr(PHEASY_TSQR_CRITERION="cv")
        with patch.dict(os.environ, self.env):
            rfe = opt.PheasyRFECV(step=0.1, cv=4, min_features=5, n_jobs=1,
                                  verbose=False, random_state=0)
            quiet(lambda: rfe.fit(self.A, self.y))
        np.testing.assert_array_equal(tsqr.support_, rfe.support_)
        np.testing.assert_allclose(tsqr.coef_, rfe.coef_, rtol=1e-8,
                                   atol=1e-10 * np.abs(rfe.coef_).max())

    def test_rows_aic_matches_cv_quality_configuration_count_prunes_too_hard(self):
        cv = self.tsqr(PHEASY_TSQR_CRITERION="cv")
        aic = self.tsqr()
        grp = self.tsqr(PHEASY_BIC_N_EFF="groups")
        self.assertEqual(grp.backend_metadata_["ic_n_eff"], 24)
        # rows: within a few percent of the CV rule (measured 0.5% here)
        self.assertLess(aic.best_rmse_cv_, 1.03 * cv.best_rmse_cv_)
        # configuration count: far sparser and much worse (measured +35%)
        self.assertLess(grp.support_.sum(), cv.support_.sum())
        self.assertGreater(grp.best_rmse_cv_, 1.2 * cv.best_rmse_cv_)

    def test_optimizer_reports_the_criterion(self):
        with patch.dict(os.environ, self.env):
            os.environ.pop("PHEASY_TSQR_CRITERION", None)
            m = opt.Optimizer("RFE-OLS-TSQR", cv=4, rand_seed=0, use_gpu=False)
            quiet(lambda: m.fit(self.A, self.y))
            r = opt.Optimizer("RFE-OLS", cv=4, rand_seed=0, use_gpu=False)
            quiet(lambda: r.fit(self.A, self.y))
        self.assertEqual(m.results["rfe_criterion"], "aic")
        self.assertEqual(m.results["rfe_ic_n_eff"], self.A.shape[0])
        self.assertEqual(r.results["rfe_criterion"], "cv")
        self.assertNotIn("rfe_ic_n_eff", r.results)


class NamingTests(unittest.TestCase):
    def test_rfe_is_an_alias_of_rfe_ols_in_the_optimizer(self):
        A, y, g = problem(seed=1, n_cfg=8, p=30, k=12)
        coefs = []
        with patch.dict(os.environ, {"PHEASY_CV_GROUP_SIZE": str(g), "PHEASY_GPU_MODE": "cpu"}):
            for name in ("RFE", "RFE-OLS", "rfe_ols"):
                m = opt.Optimizer(name, cv=4, rand_seed=0, use_gpu=False)
                quiet(lambda: m.fit(A, y))
                coefs.append(np.asarray(m.results["coef"]))
        for c in coefs[1:]:
            np.testing.assert_array_equal(c, coefs[0])

    def test_cli_rewrites_rfe_to_rfe_ols(self):
        try:
            import basic_io
        except ImportError as exc:      # f90nml etc. missing in a bare env
            self.skipTest("basic_io not importable: %s" % exc)
        with tempfile.TemporaryDirectory() as d, patch.object(sys, "argv", ["pheasy", "-l", "RFE"]):
            cwd = os.getcwd()
            os.chdir(d)
            try:
                parser = basic_io.InputParser()
                parser.read()
            finally:
                os.chdir(cwd)
        self.assertEqual(parser.settings.MODEL, "RFE-OLS")

    def _fit_sh(self, script, method):
        # an empty directory: the method check runs before the input-file check,
        # so an accepted method stops at the missing POSCAR, a rejected one first
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, PHEASY_EXECUTABLE="true", PYTHON=sys.executable)
            out = subprocess.run(["bash", str(ROOT / script), "FIT_METHOD=" + method],
                                 cwd=d, capture_output=True, text=True, timeout=60, env=env)
        return out.returncode, out.stdout + out.stderr

    def test_wrappers_take_the_full_names_and_ardr_rvm(self):
        for script in ("pheasy_fit.sh", "fit_scripts/fit_3090.sh"):
            for method in ("RFE-OLS", "RFE-OLS-TSQR", "ARDR", "RVM"):
                with self.subTest(script=script, method=method):
                    rc, text = self._fit_sh(script, method)
                    self.assertIn("POSCAR", text)      # got past the method check
            for method in ("RFE", "RFE-TSQR"):
                with self.subTest(script=script, method=method):
                    rc, text = self._fit_sh(script, method)
                    self.assertEqual(rc, 2)
                    self.assertNotIn("POSCAR", text)   # refused before the input check


if __name__ == "__main__":
    unittest.main()
