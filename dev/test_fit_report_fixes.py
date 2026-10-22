#!/usr/bin/env python3
"""Regression tests for the FC2-pruning / symmetry / CV-gap / manifest fixes.

  * [HARM_DENSE] the sparse methods protect the harmonic block by default;
    an explicit PHEASY_HARM_DENSE=0 opts out with a loud warning; OLS / RIDGE
    are untouched.
  * [SYMPREC] a slightly distorted cell found in a subgroup at the fit
    tolerance is flagged by the looser-tolerance probe.
  * [FIX CV-GAP] alpha* = 0 chosen only because the interval below the
    resolved path was never resolved is reported as a fallback, not a CV
    conclusion -- and the selection itself is unchanged.
  * fit_manifest.json keeps the values of small arrays (per-fold CV errors).

    python -m unittest dev.test_fit_report_fixes
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import optimizer as opt               # noqa: E402
from core.utilities import manifest_value       # noqa: E402
from structure import symmetry as sym           # noqa: E402

SPARSE = ("LASSO", "ALASSO", "RFE-OLS", "RFE-OLS-TSQR", "ARDR", "RVM")


def _env_without_harm_dense():
    env = dict(os.environ)
    env.pop("PHEASY_HARM_DENSE", None)
    return env


class HarmDenseDefaultTests(unittest.TestCase):
    def test_sparse_methods_protected_by_default(self):
        with patch.dict(os.environ, _env_without_harm_dense(), clear=True):
            for m in SPARSE + ("rfe_ols_tsqr", "Lasso"):
                self.assertEqual(opt.harm_dense_setting(m), (True, "default"), m)
                self.assertIsNone(opt.harm_dense_prune_warning(m), m)

    def test_ols_ridge_and_no_method_stay_off(self):
        with patch.dict(os.environ, _env_without_harm_dense(), clear=True):
            for m in ("OLS", "RIDGE", None):
                self.assertEqual(opt.harm_dense_setting(m), (False, "default"), m)
                self.assertIsNone(opt.harm_dense_prune_warning(m), m)
            self.assertFalse(opt.harm_dense_enabled())

    def test_explicit_zero_opts_out_with_warning(self):
        for raw in ("0", "false", "no"):
            with patch.dict(os.environ, {"PHEASY_HARM_DENSE": raw}):
                for m in SPARSE:
                    self.assertEqual(opt.harm_dense_setting(m), (False, "env"), m)
                    msg = opt.harm_dense_prune_warning(m)
                    self.assertIsNotNone(msg, m)
                    self.assertIn("PRUNE the second-order", msg)
                self.assertIsNone(opt.harm_dense_prune_warning("OLS"))

    def test_explicit_one_wins_for_every_method(self):
        with patch.dict(os.environ, {"PHEASY_HARM_DENSE": "1"}):
            for m in SPARSE + ("OLS", "RIDGE"):
                self.assertEqual(opt.harm_dense_setting(m), (True, "env"), m)

    def test_ols_and_ridge_bitwise_identical_with_and_without_mask(self):
        # whatever the switch says, OLS and RIDGE must not change
        rng = np.random.default_rng(11)
        A = rng.normal(size=(240, 20))
        y = A @ rng.normal(size=20) + 1e-3 * rng.normal(size=240)
        free = np.zeros(20, dtype=bool)
        free[:6] = True
        env = {"PHEASY_USE_GPU": "0", "PHEASY_GPU_SM": "0", "PHEASY_CV_GROUP_SIZE": "12",
               "PHEASY_N_JOBS": "1", "PHEASY_MAX_CORES": "1"}
        for method, kw in (("OLS", {}), ("RIDGE", {"alpha": [1e-4, 1e-3, 1e-2]})):
            coefs = []
            for mask in (None, free):
                with patch.dict(os.environ, env), \
                        contextlib.redirect_stdout(io.StringIO()):
                    o = opt.Optimizer(method, cv=3, rand_seed=0, unpenalized=mask, **kw)
                    o.fit(A, y)
                coefs.append(np.asarray(o.results["coef"]))
            np.testing.assert_array_equal(coefs[0], coefs[1], err_msg=method)


class _Cell(object):
    """Minimal stand-in for pheasy.Atoms: just the spglib tuple."""

    def __init__(self, lattice, positions, numbers):
        self._t = (np.asarray(lattice, float), np.asarray(positions, float),
                   np.asarray(numbers, int))

    def to_spglib_tuple(self):
        return self._t


def mos2_monolayer(distort_ang=0.0):
    """1H-MoS2 (P-6m2, 12 operations) with one S moved by distort_ang along x."""
    a, c = 3.19, 20.0
    lat = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]
    dz = 1.56 / c
    pos = np.array([[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5 + dz], [2 / 3, 1 / 3, 0.5 - dz]])
    pos[1, 0] += distort_ang / a
    return _Cell(lat, pos, [42, 16, 16])


def fit_summary(cell, symprec, **kw):
    sg = sym.get_spacegroup(cell, symprec=symprec)
    nsym = sym.get_symmetry(cell, symprec=symprec)["translations"].shape[0]
    return sym.symmetry_summary(cell, symprec, sg, nsym, **kw)


class SymmetryProbeTests(unittest.TestCase):
    def test_record_parses_spglib_string(self):
        r = sym.spacegroup_record("P-6m2 (187)", 12, 1e-3)
        self.assertEqual((r["symbol"], r["number"], r["n_operations"]), ("P-6m2", 187, 12))
        self.assertEqual(sym.spacegroup_record(None, None, 1e-5)["number"], None)

    def test_distorted_cell_is_flagged(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PHEASY_SYMPREC_PROBE", None)
            s = fit_summary(mos2_monolayer(5e-5), 1e-5)
        self.assertLess(s["fit"]["n_operations"], 12)
        self.assertEqual(s["probe"]["symprec"], 1e-3)
        self.assertEqual((s["probe"]["number"], s["probe"]["n_operations"]), (187, 12))
        self.assertTrue(s["probe_more_symmetric"])
        msg = sym.symmetry_warning(s)
        self.assertIn("P-6m2 (187)", msg)
        self.assertIn("--symprec 0.001", msg)
        json.dumps(s)                              # manifest-ready as is

    def test_clean_cell_is_not_flagged(self):
        s = fit_summary(mos2_monolayer(0.0), 1e-5)
        self.assertEqual(s["fit"]["n_operations"], 12)
        self.assertFalse(s["probe_more_symmetric"])
        self.assertIsNone(sym.symmetry_warning(s))

    def test_probe_skipped_when_not_looser_or_disabled(self):
        self.assertIsNone(fit_summary(mos2_monolayer(5e-5), 1e-3)["probe"])
        with patch.dict(os.environ, {"PHEASY_SYMPREC_PROBE": "0"}):
            s = fit_summary(mos2_monolayer(5e-5), 1e-5)
        self.assertIsNone(s["probe"])
        self.assertIsNone(sym.symmetry_warning(s))

    def test_probe_tolerance_override(self):
        with patch.dict(os.environ, {"PHEASY_SYMPREC_PROBE": "1e-2"}):
            s = fit_summary(mos2_monolayer(5e-5), 1e-5)
        self.assertEqual(s["probe"]["symprec"], 1e-2)


class CvGapTests(unittest.TestCase):
    alphas = np.array([1e-4, 1e-3, 1e-2, 1e-1])
    res = np.array([[False] * 5, [True] * 5, [True] * 5, [True] * 5])

    def select(self, mse, ols, **env):
        out = io.StringIO()
        with patch.dict(os.environ, env), contextlib.redirect_stdout(out):
            sel = opt._select_cv_alpha(self.alphas, mse, self.res, ols_mse=ols)
        return sel, out.getvalue()

    def test_large_gap_is_a_fallback_not_a_cv_conclusion(self):
        # Mn2In2Se5 5.71 A: best resolved 7.07e-5 vs OLS 2.15e-6 (33x)
        mse = np.vstack([np.full(5, 1e-6), np.full(5, 7.07e-5),
                         np.full(5, 9e-5), np.full(5, 2e-4)])
        sel, log = self.select(mse, np.full(5, 2.15e-6))
        self.assertEqual(sel["best"], -1)
        self.assertAlmostEqual(sel["gap_ratio"], 7.07e-5 / 2.15e-6, places=6)
        self.assertTrue(sel["gap_unresolved"])
        self.assertIn("FALLBACK, not a CV conclusion", log)
        self.assertNotIn("a CV conclusion, not an optimiser artefact", log)
        # selection is independent of the warning threshold
        sel2, log2 = self.select(mse, np.full(5, 2.15e-6), PHEASY_CV_GAP_WARN="100")
        self.assertEqual(sel2["best"], sel["best"])
        self.assertFalse(sel2["gap_unresolved"])
        self.assertIn("a CV conclusion, not an optimiser artefact", log2)

    def test_small_gap_keeps_the_cv_conclusion(self):
        mse = np.vstack([np.full(5, 1.0), np.full(5, 1.05),
                         np.full(5, 1.2), np.full(5, 2.0)])
        sel, log = self.select(mse, np.full(5, 1.0))
        self.assertEqual(sel["best"], -1)
        self.assertAlmostEqual(sel["gap_ratio"], 1.05)
        self.assertFalse(sel["gap_unresolved"])
        self.assertIn("a CV conclusion, not an optimiser artefact", log)
        self.assertNotIn("FALLBACK", log)

    def test_nothing_resolved_with_ols_reference(self):
        mse = np.full((4, 5), 1.0)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            sel = opt._select_cv_alpha(self.alphas, mse, np.zeros((4, 5), bool),
                                       ols_mse=np.full(5, 0.5))
        self.assertEqual(sel["best"], -1)
        self.assertIsNone(sel["gap_ratio"])
        self.assertTrue(sel["gap_unresolved"])
        self.assertIn("by DEFAULT", out.getvalue())

    def test_regularized_optimum_is_not_flagged(self):
        mse = np.vstack([np.full(5, 1.0), np.full(5, 0.8),
                         np.full(5, 0.9), np.full(5, 2.0)])
        sel, _ = self.select(mse, np.full(5, 1.0))
        self.assertEqual(sel["best"], 1)
        self.assertFalse(sel["gap_unresolved"])

    def test_results_carry_the_gap_fields(self):
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
        self.assertIn("cv_gap_ratio", r)
        self.assertIn("cv_gap_unresolved", r)
        self.assertIsInstance(r["cv_gap_unresolved"], bool)
        self.assertEqual(r["cv_selection"].get("gap_unresolved"), r["cv_gap_unresolved"])


class ManifestValueTests(unittest.TestCase):
    def test_small_array_keeps_values_and_metadata(self):
        rmse = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = manifest_value({"rmse_path": rmse, "rmse_path_mean": np.float64(rmse.mean())})
        self.assertEqual(v["rmse_path"]["type"], "ndarray")
        self.assertEqual(v["rmse_path"]["shape"], [5])
        self.assertEqual(v["rmse_path"]["dtype"], "float64")
        self.assertEqual(len(v["rmse_path"]["values"]), 5)
        self.assertEqual(np.mean(v["rmse_path"]["values"]), v["rmse_path_mean"])
        json.loads(json.dumps(v, allow_nan=False))

    def test_nonfinite_entries_become_null(self):
        v = manifest_value(np.array([[1.0, np.nan], [np.inf, -np.inf]]))
        self.assertEqual(v["values"], [[1.0, None], [None, None]])
        json.dumps(v, allow_nan=False)

    def test_large_arrays_stay_metadata_only(self):
        v = manifest_value(np.zeros(1001))
        self.assertNotIn("values", v)
        self.assertEqual(v["shape"], [1001])
        with patch.dict(os.environ, {"PHEASY_MANIFEST_MAX_ARRAY": "0"}):
            self.assertNotIn("values", manifest_value(np.zeros(3)))
        with patch.dict(os.environ, {"PHEASY_MANIFEST_MAX_ARRAY": "2000"}):
            self.assertIn("values", manifest_value(np.zeros(1001)))

    def test_int_and_bool_arrays(self):
        self.assertEqual(manifest_value(np.array([1, 2], dtype=np.int32))["values"], [1, 2])
        self.assertEqual(manifest_value(np.array([True, False]))["values"], [True, False])


if __name__ == "__main__":
    unittest.main()
