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

    def test_resolved_bottom_is_not_a_gap_however_large_the_ratio(self):
        # bottom alpha resolved and the curve falls toward OLS: monotone path
        res = np.ones((4, 5), dtype=bool)
        mse = np.vstack([np.full(5, 1e-20), np.full(5, 1e-18),
                         np.full(5, 1e-10), np.full(5, 1e-4)])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            sel = opt._select_cv_alpha(self.alphas, mse, res, ols_mse=np.full(5, 1e-30))
        self.assertEqual(sel["best"], -1)
        self.assertGreater(sel["gap_ratio"], 1e9)
        self.assertFalse(sel["gap_unresolved"])
        self.assertIn("a CV conclusion, not an optimiser artefact", out.getvalue())

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


class SparsityVerdictTests(unittest.TestCase):
    """sparsity_supported=False is a CV verdict; an unresolved gap is not one."""

    def fit(self, n_unresolved=0, ols_limit=True):
        # noiseless dense truth: the OLS limit wins.  The unpenalized block
        # sends LASSO through the CPU FISTA path, i.e. through _select_cv_alpha.
        # n_unresolved > 0 emulates a solver that could not resolve the lowest
        # grid alphas (the Mn2In2Se5 symptom); the Gram path here resolves all.
        rng = np.random.default_rng(5)
        X = rng.normal(size=(40 * 6, 12))
        y = X @ rng.normal(size=12)
        free = np.zeros(12, dtype=bool)
        free[:3] = True
        real = opt._select_cv_alpha

        def select(alphas, mse_path, resolved, *a, **kw):
            resolved = np.array(resolved, dtype=bool)
            resolved[:n_unresolved] = False
            return real(alphas, mse_path, resolved, *a, **kw)

        env = {"PHEASY_CV_GROUP_SIZE": "6", "PHEASY_USE_GPU": "0", "PHEASY_GPU_SM": "0"}
        with patch.dict(os.environ, env), patch.object(opt, "_select_cv_alpha", select), \
                contextlib.redirect_stdout(io.StringIO()) as log:
            grid = opt.derive_alpha_grid(X, y, nalpha=9, decades=4.0, unpenalized=free)
            o = opt.Optimizer("LASSO", alpha=grid, cv=5, rand_seed=0, tol=1e-10,
                              max_iter=100000, ols_limit=ols_limit, unpenalized=free)
            o.fit(X, y)
        return o.results, log.getvalue()

    def test_resolved_monotone_path_keeps_the_false_verdict(self):
        # every alpha resolved, curve falling all the way to OLS: the gap ratio
        # is huge (OLS CV MSE ~1e-30) but alpha* = 0 IS a CV conclusion
        r, log = self.fit()
        self.assertTrue(r["cv_selected_ols_limit"])
        self.assertGreater(r["cv_gap_ratio"], 1e3)
        self.assertFalse(r["cv_gap_unresolved"])
        self.assertIs(r["sparsity_supported"], False)
        self.assertNotIn("sparsity_undetermined_reason", r)
        self.assertNotIn("FALLBACK", log)

    def test_unresolved_gap_is_undetermined_and_fit_unchanged(self):
        ref, _ = self.fit()
        r, log = self.fit(n_unresolved=3)
        self.assertTrue(r["cv_gap_unresolved"])
        self.assertIn("FALLBACK", log)
        self.assertIsNone(r["sparsity_supported"])
        self.assertIn("by default", r["sparsity_undetermined_reason"])
        self.assertEqual(r["alpha"], ref["alpha"])
        np.testing.assert_array_equal(r["coef"], ref["coef"])
        json.dumps(manifest_value(r), allow_nan=False)

    def test_nothing_resolved_without_ols_reference_is_undetermined(self):
        r, log = self.fit(n_unresolved=10 ** 6, ols_limit=False)
        self.assertIn("no alpha was RESOLVED in every fold", log)
        self.assertTrue(r["cv_selection"]["fallback"])
        self.assertTrue(r["alpha_at_grid_edge"])
        self.assertIsNone(r["sparsity_supported"])
        self.assertIn("plain argmin", r["sparsity_undetermined_reason"])



class SolveScopeWarningTests(unittest.TestCase):
    """A non-converged RFE ranking / fold solve must not read like a rejected fit."""

    @staticmethod
    def _lsmr_result(istop=7, itn=5000):
        # scipy lsmr: (x, istop, itn, normr, normar, norma, conda, normx)
        return (np.zeros(3), istop, itn, 1.0, 0.1, 1.0, 10.0, 1.0)

    def _warn(self, scope):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            info = opt._iterative_solver_info(self._lsmr_result(), "LSMR", scope=scope)
        self.assertFalse(info["converged"])
        self.assertEqual(len(w), 1)
        return str(w[0].message)

    def test_scope_is_named_with_its_acceptance_role(self):
        for scope in ("ranking", "fold"):
            msg = self._warn(scope)
            self.assertIn("LSMR did not converge [%s]" % scope, msg)
            self.assertIn("not part of the acceptance gate", msg)
            self.assertIn("istop=7, iterations=5000", msg)
        msg = self._warn("full")
        self.assertIn("[full]", msg)
        self.assertIn("fit_accepted=False", msg)

    def test_unscoped_message_is_unchanged(self):
        msg = self._warn(None)
        self.assertIn("LSMR did not converge: iteration limit reached", msg)
        self.assertIn("before using these coefficients", msg)

    def _rfe(self, A, y, **extra):
        import warnings
        env = {"PHEASY_LSQR_MAXITER": "1", "PHEASY_RFE_GRAM_GB": "0",
               "PHEASY_USE_GPU": "0", "PHEASY_GPU_SM": "0",
               "PHEASY_CV_GROUP_SIZE": "12", "PHEASY_RFE_STEP": "0.2"}
        env.update(extra)
        clean = {k: v for k, v in os.environ.items()
                 if k != "PHEASY_RFE_FINAL_LSMR_MAXITER"}
        with patch.dict(os.environ, clean, clear=True), patch.dict(os.environ, env), \
                contextlib.redirect_stdout(io.StringIO()) as log, \
                warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            o = opt.Optimizer("RFE-OLS", cv=3, rand_seed=0, use_gpu=False)
            o.fit(A, y)
        msgs = [str(x.message) for x in w if "did not converge" in str(x.message)]
        diags = o.results["backend_metadata"]["iterative_diagnostics"]
        bad = [d["fit_scope"] for d in diags if d.get("converged") is False]
        self._diags = diags
        return o.results, msgs, bad, log.getvalue()

    @staticmethod
    def _problem():
        rng = np.random.default_rng(3)
        dense = rng.normal(size=(120, 10))
        beta = np.zeros(10)
        beta[:4] = rng.normal(size=4)
        return dense, dense @ beta + 1e-3 * rng.normal(size=120)

    def test_rfe_ranking_nonconvergence_is_labelled_and_fit_accepted(self):
        # production operator: the final refit is exact, so only ranking / fold
        # LSMR solves fail -- labelled, and they do not veto the fit
        import scipy.sparse as sp
        dense, y = self._problem()
        A = opt.TwoLevelSM(sp.csr_matrix(dense), sp.identity(10, format="csr"))
        r, msgs, bad, _ = self._rfe(A, y)
        self.assertTrue(any("[ranking]" in m for m in msgs), msgs[:3])
        self.assertFalse(any("[full]" in m for m in msgs), msgs[:3])
        self.assertTrue(bad)
        self.assertTrue(set(bad) <= {"ranking", "fold"}, bad)
        self.assertTrue(r["fit_accepted"])

    def test_rfe_final_lsmr_fallback_is_labelled_full_and_vetoes(self):
        # an operator the exact refit cannot slice falls back to LSMR for the
        # DELIVERED coefficients: that one is [full] and does reject the fit
        from scipy.sparse.linalg import aslinearoperator
        dense, y = self._problem()
        r, msgs, bad, log = self._rfe(aslinearoperator(dense), y)
        self.assertIn("LSMR fallback", log)
        self.assertIn("final refit: delivering a full-data LSMR solve", log)
        self.assertEqual(sum("[full]" in m for m in msgs), 1)
        self.assertIn("full", bad)
        self.assertFalse(r["fit_accepted"])
        self.assertIn("PHEASY_LSQR_MAXITER=1, shared with the ranking rounds", log)

    def test_final_cap_alone_rejects_when_too_small(self):
        # ranking / fold solves converge (shared cap 5000); only the delivered
        # fallback solve is capped at 1 -> it alone fails and rejects the fit
        from scipy.sparse.linalg import aslinearoperator
        dense, y = self._problem()
        r, msgs, bad, log = self._rfe(aslinearoperator(dense), y,
                                      PHEASY_LSQR_MAXITER="5000",
                                      PHEASY_RFE_FINAL_LSMR_MAXITER="1")
        self.assertEqual(bad, ["full"])
        self.assertFalse(r["fit_accepted"])
        full = [d for d in self._diags if d["fit_scope"] == "full"]
        self.assertEqual([d["itn"] for d in full], [1])
        self.assertIn("iteration cap PHEASY_RFE_FINAL_LSMR_MAXITER=1", log)

    def test_final_cap_rescues_the_fit_without_touching_ranking(self):
        # shared cap 1 (every ranking / fold solve fails, as on Mg4C60), final
        # cap 5000: the delivered solve converges and the fit is accepted
        from scipy.sparse.linalg import aslinearoperator
        dense, y = self._problem()
        r, msgs, bad, _ = self._rfe(aslinearoperator(dense), y,
                                    PHEASY_RFE_FINAL_LSMR_MAXITER="5000")
        self.assertTrue(bad)
        self.assertTrue(set(bad) <= {"ranking", "fold"}, bad)
        self.assertTrue(r["fit_accepted"])
        ranking = [d["itn"] for d in self._diags if d["fit_scope"] == "ranking"]
        self.assertTrue(ranking and max(ranking) == 1, ranking)
        full = [d for d in self._diags if d["fit_scope"] == "full"]
        self.assertEqual(len(full), 1)
        self.assertTrue(full[0]["converged"])
        self.assertGreater(full[0]["itn"], 1)

    def test_final_cap_must_be_positive(self):
        for raw in ("0", "-3"):
            with patch.dict(os.environ, {"PHEASY_RFE_FINAL_LSMR_MAXITER": raw}):
                with self.assertRaises(ValueError):
                    opt._rfe_final_lsmr_maxiter()
        with patch.dict(os.environ, {"PHEASY_RFE_FINAL_LSMR_MAXITER": " "}):
            self.assertIsNone(opt._rfe_final_lsmr_maxiter())


class AvailableMemoryTests(unittest.TestCase):
    """[FIX MEMAVAIL] size allocations by MemAvailable (+ cgroup), not MemFree."""

    GB = 1024 ** 3

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="memavail_")
        self.cg = os.path.join(self.tmp, "cgroup")
        os.makedirs(self.cg)

    def _write(self, rel, text):
        path = os.path.join(self.tmp, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def _env(self, free_gb, avail_gb=None, proc_cgroup="0::/\n"):
        lines = "MemTotal: %d kB\nMemFree: %d kB\n" % (512 * 1024 ** 2, free_gb * 1024 ** 2)
        if avail_gb is not None:
            lines += "MemAvailable: %d kB\n" % (avail_gb * 1024 ** 2)
        meminfo = self._write("meminfo", lines)
        cgfile = self._write("self_cgroup", proc_cgroup)
        return (patch.object(opt, "_MEMINFO_PATH", meminfo),
                patch.object(opt, "_PROC_CGROUP_PATH", cgfile),
                patch.object(opt, "_CGROUP_ROOT", self.cg))

    def _avail(self, *args, **kw):
        a, b, c = self._env(*args, **kw)
        with a, b, c:
            return opt._available_memory_bytes()

    def test_memavailable_not_memfree(self):
        # the GPU box: 148 GB page cache, MemFree 36 GB, MemAvailable 166 GB
        self.assertEqual(self._avail(36, 166), 166 * self.GB)

    def test_cgroup_v2_limit_caps_the_node(self):
        self._write("cgroup/job/memory.max", str(64 * self.GB))
        self._write("cgroup/job/memory.current", str(10 * self.GB))
        self.assertEqual(self._avail(36, 166, "0::/job\n"), 54 * self.GB)

    def test_cgroup_v2_tightest_ancestor_wins(self):
        self._write("cgroup/slurm/memory.max", str(30 * self.GB))
        self._write("cgroup/slurm/memory.current", str(5 * self.GB))
        self._write("cgroup/slurm/step/memory.max", "max")
        self._write("cgroup/slurm/step/memory.current", str(4 * self.GB))
        self.assertEqual(self._avail(36, 166, "0::/slurm/step\n"), 25 * self.GB)

    def test_unlimited_cgroups_leave_memavailable(self):
        self._write("cgroup/job/memory.max", "max")
        self._write("cgroup/job/memory.current", str(10 * self.GB))
        self._write("cgroup/memory/job/memory.limit_in_bytes", str(1 << 62))
        self._write("cgroup/memory/job/memory.usage_in_bytes", str(self.GB))
        self.assertEqual(self._avail(36, 166, "4:memory:/job\n0::/job\n"),
                         166 * self.GB)

    def test_cgroup_v1_limit(self):
        self._write("cgroup/memory/job/memory.limit_in_bytes", str(32 * self.GB))
        self._write("cgroup/memory/job/memory.usage_in_bytes", str(2 * self.GB))
        self.assertEqual(self._avail(36, 166, "4:memory:/job\n"), 30 * self.GB)

    def test_no_memavailable_falls_back_to_sysconf(self):
        got = self._avail(36, None)
        self.assertIsNotNone(got)
        self.assertGreater(got, 0)

    def test_rfe_final_exact_refit_runs_under_page_cache_pressure(self):
        # MemFree tiny, MemAvailable large: the exact refit must run (it used to
        # fall back to LSMR, and with a 1-iteration cap that fallback is rejected)
        import scipy.sparse as sp
        rng = np.random.default_rng(3)
        dense = rng.normal(size=(120, 10))
        beta = np.zeros(10)
        beta[:4] = rng.normal(size=4)
        y = dense @ beta + 1e-3 * rng.normal(size=120)
        env = {"PHEASY_LSQR_MAXITER": "1", "PHEASY_RFE_GRAM_GB": "0",
               "PHEASY_USE_GPU": "0", "PHEASY_GPU_SM": "0",
               "PHEASY_CV_GROUP_SIZE": "12", "PHEASY_RFE_STEP": "0.2"}
        results = {}
        for label, free_kb, avail_kb in (("cache", 1, 64 * 1024 ** 2),
                                         ("tight", 1, 1)):
            meminfo = self._write("meminfo_" + label,
                                  "MemFree: %d kB\nMemAvailable: %d kB\n" % (free_kb, avail_kb))
            cgfile = self._write("self_cgroup_" + label, "0::/\n")
            A = opt.TwoLevelSM(sp.csr_matrix(dense), sp.identity(10, format="csr"))
            with patch.object(opt, "_MEMINFO_PATH", meminfo), \
                    patch.object(opt, "_PROC_CGROUP_PATH", cgfile), \
                    patch.object(opt, "_CGROUP_ROOT", self.cg), \
                    patch.dict(os.environ, env), \
                    contextlib.redirect_stdout(io.StringIO()) as log:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    o = opt.Optimizer("RFE-OLS", cv=3, rand_seed=0, use_gpu=False)
                    o.fit(A, y)
            full = [d for d in o.results["backend_metadata"]["iterative_diagnostics"]
                    if d["fit_scope"] == "full"]
            results[label] = (o.results["fit_accepted"], [d["solver"] for d in full],
                              log.getvalue())
        ok, solvers, log = results["cache"]
        self.assertTrue(ok)
        self.assertNotIn("iterative fallback", log)
        self.assertIn(solvers[-1], ("Gram-Cholesky", "TSQR"))
        ok, solvers, log = results["tight"]
        self.assertIn("iterative fallback", log)
        self.assertFalse(ok)


class Float32IterativeAcceptanceTests(unittest.TestCase):
    """[FIX F32-ITER] a float32 iterate delivered as the answer is not accepted;
    exact paths run by default whenever 0.8 x available RAM allows them."""

    @staticmethod
    def _problem(dtype, n=240, p=12, seed=4):
        import scipy.sparse as sp
        rng = np.random.default_rng(seed)
        dense = rng.normal(size=(n, p)).astype(np.float32).astype(np.float64)
        beta = np.zeros(p)
        beta[:5] = rng.normal(size=5)
        y = dense @ beta + 1e-3 * rng.normal(size=n)
        A = opt.TwoLevelSM(sp.csr_matrix(dense.astype(dtype)),
                           sp.eye(p, format="csr", dtype=dtype))
        return A, y, dense

    def _fit(self, method, A, y, **env):
        import warnings
        base = {"PHEASY_GPU_MODE": "cpu", "PHEASY_USE_GPU": "0", "PHEASY_GPU_SM": "0",
                "PHEASY_CV_GROUP_SIZE": "12", "PHEASY_RFE_GRAM_GB": "0",
                "PHEASY_RIDGE_LCURVE": "0"}
        base.update(env)
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("PHEASY_ACCEPT_FLOAT32_ITERATIVE", "PHEASY_EXACT_GRAM_GB")}
        kw = {"alpha": [1e-4, 1e-3, 1e-2]} if method == "RIDGE" else {}
        o = opt.Optimizer(method, cv=3, rand_seed=0, use_gpu=False, **kw)
        with patch.dict(os.environ, clean, clear=True), patch.dict(os.environ, base), \
                warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            o.fit(A, y)
        return o.results

    def test_budget_defaults_to_most_of_available_ram(self):
        with patch.object(opt, "_available_memory_bytes", lambda: 100e9), \
                patch.dict(os.environ, {"PHEASY_EXACT_GRAM_GB": ""}):
            self.assertAlmostEqual(opt._exact_gram_budget_bytes(), 80e9)
        with patch.dict(os.environ, {"PHEASY_EXACT_GRAM_GB": "5"}):
            self.assertAlmostEqual(opt._exact_gram_budget_bytes(), 5e9)

    def test_float32_ols_takes_the_exact_path_and_is_accepted(self):
        A, y, dense = self._problem(np.float32)
        r = self._fit("OLS", A, y)
        self.assertEqual(r["execution_backend"], "cpu_gram_exact")
        self.assertTrue(r["fit_accepted"])
        self.assertNotIn("float32_iterative_delivered", r)

    def test_float32_iterative_ols_is_not_accepted(self):
        A, y, _ = self._problem(np.float32)
        r = self._fit("OLS", A, y, PHEASY_EXACT_GRAM_GB="0")
        self.assertEqual(r["float32_iterative_delivered"], ["LSMR"])
        self.assertFalse(r["fit_accepted"])
        self.assertEqual(r["status"], "fit_returned_not_accepted")
        self.assertIn("float32 iterative solve", r["not_accepted_reason"])
        ok = self._fit("OLS", A, y, PHEASY_EXACT_GRAM_GB="0",
                       PHEASY_ACCEPT_FLOAT32_ITERATIVE="1")
        self.assertTrue(ok["fit_accepted"])
        self.assertTrue(ok["float32_iterative_override"])

    def test_float64_iterative_ols_is_still_accepted(self):
        A, y, _ = self._problem(np.float64)
        r = self._fit("OLS", A, y, PHEASY_EXACT_GRAM_GB="0")
        self.assertTrue(r["fit_accepted"])
        self.assertNotIn("float32_iterative_delivered", r)

    def test_float32_iterative_ridge_is_not_accepted(self):
        A, y, _ = self._problem(np.float32)
        r = self._fit("RIDGE", A, y, PHEASY_EXACT_GRAM_GB="0")
        self.assertTrue(r.get("float32_iterative_delivered"), r.get("regularized_solver_info"))
        self.assertFalse(r["fit_accepted"])

    def test_float32_rfe_lsmr_fallback_is_not_accepted_even_if_converged(self):
        A, y, _ = self._problem(np.float32)
        # the final exact refit is refused (no memory), the LSMR fallback gets a
        # generous cap and converges -- at the float32 floor, so not accepted
        with patch.object(opt, "_available_memory_bytes", lambda: 1.0):
            r = self._fit("RFE-OLS", A, y, PHEASY_EXACT_GRAM_GB="0",
                          PHEASY_RFE_FINAL_LSMR_MAXITER="20000")
            # accepting it knowingly takes both opt-outs: the float32 rule and
            # the [FIX LSMR-CERT] optimality gate (record-only tolerance)
            ok = self._fit("RFE-OLS", A, y, PHEASY_EXACT_GRAM_GB="0",
                           PHEASY_RFE_FINAL_LSMR_MAXITER="20000",
                           PHEASY_ACCEPT_FLOAT32_ITERATIVE="1",
                           PHEASY_RFE_FINAL_OPT_TOL="0")
        self.assertIn("LSMR", r["float32_iterative_delivered"])
        self.assertFalse(r["fit_accepted"])
        self.assertTrue(ok["fit_accepted"])
        # with memory the same fit takes the exact refit and is accepted
        exact = self._fit("RFE-OLS", A, y)
        self.assertTrue(exact["fit_accepted"])
        self.assertNotIn("float32_iterative_delivered", exact)

    def test_float64_certified_rfe_iterate_is_exempt(self):
        # an iterate that passed the float64 optimality certificate is the LS
        # answer of the stored matrix: not listed, whatever the storage dtype
        A, _, _ = self._problem(np.float32)
        ok = {"solver": "LSMR", "optimality": 3e-7, "optimality_tol": 1e-6}
        meta = {"iterative_diagnostics": [dict(ok, fit_scope="full")]}
        self.assertEqual(opt._float32_iterative_deliveries(
            {"backend_metadata": meta}, A, "RFE"), [])
        for bad in ({"optimality": 3e-3, "optimality_tol": 1e-6},
                    {"optimality": 3e-7, "optimality_tol": 0.0}, {}):
            meta = {"iterative_diagnostics": [dict(bad, solver="LSMR", fit_scope="full")]}
            self.assertEqual(opt._float32_iterative_deliveries(
                {"backend_metadata": meta}, A, "RFE"), ["LSMR"], bad)

    def test_float32_iterative_lasso_debias_is_not_accepted(self):
        A, y, _ = self._problem(np.float32)
        kw = dict(PHEASY_LASSO_DEBIAS="1")
        o = opt.Optimizer("LASSO", alpha=[1e-4, 1e-3], cv=3, rand_seed=0, use_gpu=False)
        r = self._fit_with(o, A, y, PHEASY_EXACT_GRAM_GB="0", **kw)
        self.assertEqual(r["debias_backend"], "cpu_lsmr")
        self.assertTrue(r["float32_iterative_delivered"])
        self.assertFalse(r["fit_accepted"])
        o = opt.Optimizer("LASSO", alpha=[1e-4, 1e-3], cv=3, rand_seed=0, use_gpu=False)
        r = self._fit_with(o, A, y, **kw)
        self.assertEqual(r["debias_backend"], "cpu_gram_exact")
        self.assertTrue(r["fit_accepted"])

    def _fit_with(self, o, A, y, **env):
        import warnings
        base = {"PHEASY_GPU_MODE": "cpu", "PHEASY_USE_GPU": "0", "PHEASY_GPU_SM": "0",
                "PHEASY_CV_GROUP_SIZE": "12"}
        base.update(env)
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("PHEASY_ACCEPT_FLOAT32_ITERATIVE", "PHEASY_EXACT_GRAM_GB")}
        with patch.dict(os.environ, clean, clear=True), patch.dict(os.environ, base), \
                warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            o.fit(A, y)
        return o.results


class ExactRefitScalingTests(unittest.TestCase):
    """[FIX RFE-CHOL] Jacobi-scaled in-place Cholesky of the RFE final refit."""

    @staticmethod
    def _problem(seed=11, m=400, n=24, spread=1e5):
        import scipy.sparse as sp
        rng = np.random.default_rng(seed)
        base = rng.normal(size=(m, n))
        base[:, 1] = base[:, 0] + 1e-3 * rng.normal(size=m)   # a soft direction
        scale = np.geomspace(1.0 / spread, spread, n)
        X = base * scale[None, :]
        y = X @ (rng.normal(size=n) / scale) + 1e-2 * rng.normal(size=m)
        A = opt.TwoLevelSM(sp.csr_matrix(X), sp.identity(n, format="csr"))
        return A, X, y

    def _refit(self, A, y, idx, avail=None, **patches):
        diags = []
        ctx = [patch.object(opt, "_available_memory_bytes", return_value=avail)]
        ctx += [patch.object(*k) if isinstance(k, tuple) else k for k in patches.values()]
        with contextlib.ExitStack() as st, contextlib.redirect_stdout(io.StringIO()) as log:
            for c in ctx:
                st.enter_context(c)
            st.enter_context(patch.dict(os.environ, {"PHEASY_RFE_FINAL_BLOCK_ROWS": "64"}))
            coef = opt._rfe_final_refit_exact(A, y, idx, iterative_diagnostics=diags)
        return coef, diags, log.getvalue()

    def test_ill_scaled_support_is_least_squares_optimal(self):
        A, X, y = self._problem()
        idx = np.arange(X.shape[1])
        coef, diags, log = self._refit(A, y, idx)
        self.assertIn("Jacobi-scaled Cholesky ok", log)
        self.assertEqual(diags[-1]["solver"], "Gram-Cholesky")
        self.assertLess(diags[-1]["optimality"], 1e-8)
        # independent certificate against A itself, column-normalized
        r = X @ coef - y
        g = X.T @ r
        self.assertLess(np.max(np.abs(g) / np.linalg.norm(X, axis=0)) / np.linalg.norm(r),
                        1e-8)
        # reference in normalized columns (an SVD of the raw X truncates the
        # soft direction: its singular value sits below rcond * s_max)
        cn = np.linalg.norm(X, axis=0)
        ref = np.linalg.lstsq(X / cn, y, rcond=None)[0]
        np.testing.assert_allclose(coef * cn, ref, rtol=1e-7, atol=1e-7)

    def test_cholesky_is_in_place_and_the_gate_counts_one_gram(self):
        self.assertTrue(opt._cholesky_in_place())
        A, X, y = self._problem(n=20)
        idx = np.arange(20)
        n, blk = 20, 64
        one = 8.0 * n * n + 16.0 * blk * n
        # fits once (in place), not twice
        avail = 1.05 * one / 0.8
        coef, diags, log = self._refit(A, y, idx, avail=avail)
        self.assertIsNotNone(coef)
        self.assertNotIn("iterative fallback", log)
        coef, diags, log = self._refit(A, y, idx, avail=avail,
                                       p=(opt, "_cholesky_in_place", lambda: False))
        self.assertIsNone(coef)
        self.assertIn("iterative fallback", log)

    def test_memory_error_in_cholesky_falls_back_to_lsmr(self):
        A, X, y = self._problem(n=12)

        def boom(*a, **k):
            raise MemoryError
        coef, diags, log = self._refit(A, y, np.arange(12),
                                       p=patch.object(opt.spla, "cholesky", boom))
        self.assertIsNone(coef)
        self.assertEqual(diags, [])
        self.assertIn("Cholesky out of memory (MemoryError); LSMR fallback", log)

    def test_memory_error_in_accumulation_falls_back_to_lsmr(self):
        A, X, y = self._problem(n=12)
        real = opt._rfe_row_source

        def rows_then_oom(A_, cols):
            def rows(i0, i1):
                raise MemoryError
            return rows
        coef, diags, log = self._refit(A, y, np.arange(12),
                                       p=(opt, "_rfe_row_source", rows_then_oom))
        self.assertIs(opt._rfe_row_source, real)
        self.assertIsNone(coef)
        self.assertIn("Gram accumulation out of memory (MemoryError); LSMR fallback", log)

    def test_not_pd_gram_goes_to_tsqr_then_lsmr(self):
        A, X, y = self._problem(n=10)
        import scipy.sparse as sp
        Xd = np.hstack([X, X[:, :1]])                  # exact duplicate column
        A = opt.TwoLevelSM(sp.csr_matrix(Xd), sp.identity(11, format="csr"))
        coef, diags, log = self._refit(A, y, np.arange(11))
        self.assertIn("scaled Gram not PD", log)
        self.assertIn("trying TSQR", log)
        self.assertIsNone(coef)                        # TSQR flags the rank deficiency
        coef, diags, log = self._refit(A, y, np.arange(11), avail=1.0)
        self.assertIn("iterative fallback", log)

    def test_refinement_is_logged_and_never_worsens(self):
        A, X, y = self._problem(seed=5, spread=1e6)
        with patch.dict(os.environ, {"PHEASY_RFE_FINAL_REFINE": "3"}):
            coef, diags, log = self._refit(A, y, np.arange(X.shape[1]))
        d = diags[-1]
        self.assertIn("optimality", d)
        self.assertLessEqual(d["refine_steps"], 3)
        self.assertLess(d["optimality"], 1e-8)
        self.assertIn("optimality max|A_j^T r|/(||A_j|| ||r||)", log)

    def test_check_can_be_disabled(self):
        A, X, y = self._problem(n=8)
        with patch.dict(os.environ, {"PHEASY_RFE_FINAL_CHECK": "0"}):
            coef, diags, log = self._refit(A, y, np.arange(8))
        self.assertNotIn("optimality", diags[-1])
        self.assertNotIn("optimality max", log)


class LsmrCertificationTests(unittest.TestCase):
    """[FIX LSMR-CERT] a delivered RFE final LSMR solve must be LS-optimal."""

    @staticmethod
    def _operator(dtype=np.float32, seed=7, m=240, n=16):
        import scipy.sparse as sp
        rng = np.random.default_rng(seed)
        prime = rng.normal(size=(m, n + 4)) * np.geomspace(1e-2, 1e2, n + 4)[None, :]
        ns = sp.csr_matrix(rng.normal(size=(n + 4, n)))
        A = opt.TwoLevelSM(sp.csr_matrix(prime.astype(dtype)), ns)
        dense = prime.astype(dtype).astype(np.float64) @ ns.toarray()
        y = dense @ rng.normal(size=n) + 1e-2 * rng.normal(size=m)
        return A, dense, y

    def test_support_gradient_is_float64_and_handles_the_scaled_wrapper(self):
        A, dense, y = self._operator()
        cols = np.array([0, 3, 5, 9, 15])
        coef = np.random.default_rng(1).normal(size=cols.size)
        g, rn = opt._support_ls_gradient(A, y, cols, coef)
        r = dense[:, cols] @ coef - y
        np.testing.assert_allclose(g, dense[:, cols].T @ r, rtol=1e-11, atol=0)
        self.assertAlmostEqual(rn, float(np.linalg.norm(r)), places=10)
        w = np.linspace(0.5, 3.0, dense.shape[1])
        W = opt._scale_operator(A, w)
        W._twolevel_base, W._twolevel_scale = A, w
        g2, rn2 = opt._support_ls_gradient(W, y, cols, coef)
        Ad = dense / w[None, :]
        r2 = Ad[:, cols] @ coef - y
        np.testing.assert_allclose(g2, Ad[:, cols].T @ r2, rtol=1e-11, atol=0)
        self.assertAlmostEqual(rn2, float(np.linalg.norm(r2)), places=10)

    def _fit(self, dtype, extra):
        A, dense, y = self._operator(dtype=dtype)
        env = {"PHEASY_RFE_FINAL_TSQR": "0", "PHEASY_RFE_GRAM_GB": "0",
               "PHEASY_USE_GPU": "0", "PHEASY_GPU_SM": "0",
               "PHEASY_CV_GROUP_SIZE": "12", "PHEASY_RFE_STEP": "0.2"}
        env.update(extra)
        import warnings
        with patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()) as log, \
                warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o = opt.Optimizer("RFE-OLS", cv=3, rand_seed=0, use_gpu=False)
            o.fit(A, y)
        full = [d for d in o.results["backend_metadata"]["iterative_diagnostics"]
                if d["fit_scope"] == "full"]
        return o.results, full[-1], log.getvalue()

    def test_early_istop_without_optimality_is_rejected(self):
        res, d, log = self._fit(np.float32, {"PHEASY_LSQR_ATOL": "1e-2",
                                             "PHEASY_LSQR_BTOL": "1e-2"})
        self.assertIn("delivering a full-data LSMR solve", log)
        self.assertTrue(d["lsmr_converged"])           # LSMR itself said "converged"
        self.assertGreater(d["optimality"], 1e-6)
        self.assertFalse(d["converged"])
        self.assertEqual(d["stop_reason"], "optimality_not_certified")
        self.assertFalse(res["fit_accepted"])
        self.assertIn("is not least-squares optimal", log)

    def test_a_converged_lsmr_solve_is_certified(self):
        res, d, log = self._fit(np.float64, {"PHEASY_LSQR_ATOL": "1e-14",
                                             "PHEASY_LSQR_BTOL": "1e-14",
                                             "PHEASY_LSQR_MAXITER": "2000"})
        self.assertLessEqual(d["optimality"], 1e-6)
        self.assertTrue(d["converged"])
        self.assertTrue(res["fit_accepted"])
        self.assertIn("LSMR solve optimality", log)

    def test_zero_tolerance_records_without_gating(self):
        # float32 input: an uncertified iterate also needs the [FIX F32-ITER]
        # override to be accepted (the record-only tolerance alone is not enough)
        res, d, log = self._fit(np.float32, {"PHEASY_LSQR_ATOL": "1e-2",
                                             "PHEASY_LSQR_BTOL": "1e-2",
                                             "PHEASY_RFE_FINAL_OPT_TOL": "0",
                                             "PHEASY_ACCEPT_FLOAT32_ITERATIVE": "1"})
        self.assertGreater(d["optimality"], 1e-6)
        self.assertTrue(d["converged"])
        self.assertTrue(res["fit_accepted"])
        self.assertIn("not gated", log)



class RfeFinalSolveKindTests(unittest.TestCase):
    """[FIX RFE-LSQR-DIAG] the final-solve log names the solver that runs, the
    support is always saved, and a sparse LSQR subset solve is recorded."""

    def _fit(self, A, y, **env):
        import tempfile
        import warnings
        base = {"PHEASY_USE_GPU": "0", "PHEASY_GPU_SM": "0", "PHEASY_GPU_MODE": "cpu",
                "PHEASY_CV_GROUP_SIZE": "12", "PHEASY_RFE_STEP": "0.2",
                "PHEASY_RFE_GRAM_GB": "0"}
        base.update(env)
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("PHEASY_ACCEPT_FLOAT32_ITERATIVE", "PHEASY_MAX_DENSE")}
        cwd = os.getcwd()
        tmp = tempfile.mkdtemp(prefix="rfe_kind_")
        os.chdir(tmp)
        try:
            o = opt.Optimizer("RFE-OLS", cv=3, rand_seed=0, use_gpu=False)
            with patch.dict(os.environ, clean, clear=True), patch.dict(os.environ, base), \
                    contextlib.redirect_stdout(io.StringIO()) as log, \
                    warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                o.fit(A, y)
            saved = os.path.exists(os.path.join(tmp, "rfe_support.npy"))
        finally:
            os.chdir(cwd)
        msgs = [str(x.message) for x in w if "did not converge" in str(x.message)]
        return o.results, log.getvalue(), saved, msgs

    @staticmethod
    def _problem(seed=6, n=240, p=14):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(n, p)).astype(np.float32).astype(np.float64)
        beta = np.zeros(p)
        beta[:5] = rng.normal(size=5)
        return A, A @ beta + 1e-3 * rng.normal(size=n)

    def test_dense_input_logs_a_direct_solve_and_saves_the_support(self):
        A, y = self._problem()
        r, log, saved, _ = self._fit(A, y)
        self.assertIn("final refit: direct least-squares solve", log)
        self.assertNotIn("LSMR solve", log)
        self.assertTrue(r["fit_accepted"])
        self.assertTrue(saved)
        meta = r["backend_metadata"]
        self.assertEqual(meta["support_file"], "rfe_support.npy")
        self.assertEqual(meta["n_active"], int(np.count_nonzero(r["coef"])))
        self.assertEqual(meta["n_candidates"], A.shape[1])

    def _sparse_fallback(self, **env):
        import scipy.sparse as sp
        A, y = self._problem()
        A32 = sp.csr_matrix(A.astype(np.float32))
        # never densify; the exact final refit is refused (no memory)
        with patch.object(opt, "_available_memory_bytes", lambda: 1.0):
            return self._fit(A32, y, PHEASY_MAX_DENSE="0", **env)

    def test_sparse_lsqr_fallback_is_recorded_and_certified(self):
        # scipy's float32 CSR times a float64 vector computes in float64, so
        # this LSQR reaches the least-squares answer: the float64 optimality
        # certificate passes and the fit is accepted on that evidence
        r, log, saved, msgs = self._sparse_fallback()
        self.assertIn("delivering a full-data LSQR solve", log)
        diags = r["backend_metadata"]["iterative_diagnostics"]
        full = [d for d in diags if d.get("fit_scope") == "full"]
        self.assertTrue(full and full[-1]["solver"] == "LSQR", full)
        self.assertLessEqual(full[-1]["optimality"], full[-1]["optimality_tol"])
        self.assertTrue(r["fit_accepted"])
        self.assertTrue(saved)
        ranking = [d for d in diags if d.get("fit_scope") == "ranking"]
        self.assertTrue(ranking and all(d["solver"] == "LSQR" for d in ranking))

    def test_uncertified_sparse_lsqr_fallback_is_not_accepted(self):
        # stopped early: not least-squares optimal, so not accepted (it used to
        # be invisible to the gate and shipped as the RFE answer)
        r, log, saved, msgs = self._sparse_fallback(PHEASY_LSQR_ATOL="1e-2",
                                                    PHEASY_LSQR_BTOL="1e-2")
        full = [d for d in r["backend_metadata"]["iterative_diagnostics"]
                if d.get("fit_scope") == "full"]
        self.assertGreater(full[-1]["optimality"], full[-1]["optimality_tol"])
        self.assertIn("LSQR", r.get("float32_iterative_delivered", []))
        self.assertFalse(r["fit_accepted"])


class RfeDenseGramTests(unittest.TestCase):
    """[FIX RFE-GRAM-DENSE] the per-fold Gram engine on dense input selects the
    same support as the per-subset QR/SVD rounds; the final solve stays QR."""

    def _fit(self, A, y, gram_dense):
        import tempfile
        import warnings
        env = {"PHEASY_USE_GPU": "0", "PHEASY_GPU_SM": "0", "PHEASY_GPU_MODE": "cpu",
               "PHEASY_CV_GROUP_SIZE": "12", "PHEASY_RFE_STEP": "0.1",
               "PHEASY_RFE_GRAM_GB": "4", "PHEASY_RFE_GRAM_DENSE": gram_dense}
        cwd = os.getcwd()
        os.chdir(tempfile.mkdtemp(prefix="rfe_gram_dense_"))
        try:
            o = opt.Optimizer("RFE-OLS", cv=5, rand_seed=0, use_gpu=False)
            with patch.dict(os.environ, env), warnings.catch_warnings(), \
                    contextlib.redirect_stdout(io.StringIO()) as log:
                warnings.simplefilter("ignore")
                o.fit(A, y)
        finally:
            os.chdir(cwd)
        return o.results, log.getvalue()

    def test_same_support_and_coefficients_as_qr_rounds(self):
        rng = np.random.default_rng(12)
        n_cfg, rows, p = 50, 12, 40
        A = rng.normal(size=(n_cfg * rows, p)) * np.geomspace(1.0, 30.0, p)
        beta = np.zeros(p)
        beta[rng.choice(p, 9, replace=False)] = rng.normal(size=9)
        y = A @ beta + 0.05 * rng.normal(size=A.shape[0])
        g, glog = self._fit(A, y, "1")
        q, _ = self._fit(A, y, "0")
        gm, qm = g["backend_metadata"], q["backend_metadata"]
        self.assertGreater(gm["gram_exact_rounds"], 0)
        self.assertEqual(qm["gram_exact_rounds"], 0)
        self.assertEqual(gm["n_active"], qm["n_active"])
        np.testing.assert_array_equal(np.flatnonzero(g["coef"]), np.flatnonzero(q["coef"]))
        np.testing.assert_allclose(g["coef"], q["coef"], rtol=1e-9, atol=1e-12)
        self.assertIn("final refit: direct least-squares solve", glog)
        self.assertTrue(g["fit_accepted"])


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
