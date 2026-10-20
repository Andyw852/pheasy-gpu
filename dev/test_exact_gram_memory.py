#!/usr/bin/env python3
"""[FIX EXACT-LS-MEM / F64-FLOOR] regression tests.

* The exact float64 normal-equation solve factors the Gram IN PLACE, so its
  memory gate is ~1.1 Gram instead of the old 3 Grams.  On Mg4C60 (84557 to
  121438 columns) the old gate refused the exact path at every cutoff, even
  with PHEASY_EXACT_GRAM_GB=130, and OLS silently went back to CGLS.
* The residual is now taken from the operator (the Gram no longer exists after
  the in-place factorization); it must equal ||A x - y|| exactly.
* A singular Gram re-streams it for the min-norm eigh fallback, which has its
  own (4 Gram) gate; past that the solve hands back to the iterative solver.
* The CGLS measured-floor cap is a float32 statement; float64 gets 1e-6.

    python -m unittest dev.test_exact_gram_memory
"""
import contextlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import optimizer as opt          # noqa: E402

try:
    import torch                            # noqa: F401
    from core import gpu_backend as gb      # noqa: E402
except ImportError:                         # pragma: no cover
    torch = None
    gb = None


def problem(seed=0, n=600, p=40):
    r = np.random.default_rng(seed)
    A = r.standard_normal((n, p)) * np.logspace(0, -2, p)[None, :]
    A = A.astype(np.float32).astype(np.float64)
    y = A @ r.standard_normal(p) + 0.01 * r.standard_normal(n)
    return A, y


def twolevel32(A):
    return opt.TwoLevelSM(sp.csr_matrix(A.astype(np.float32)),
                          sp.eye(A.shape[1], format="csr", dtype=np.float32))


def gram_gb(p):
    return 8.0 * p * p / 1e9


def solve(A, y, budget_gb, **kw):
    out = io.StringIO()
    with patch.dict(os.environ, {"PHEASY_EXACT_GRAM_GB": repr(budget_gb)}), \
            contextlib.redirect_stdout(out):
        res = opt._exact_normal_solve(A, y, **kw)
    return res, out.getvalue()


class ExactGramMemoryTests(unittest.TestCase):
    def setUp(self):
        self.A, self.y = problem()
        self.op = twolevel32(self.A)
        self.p = self.A.shape[1]

    def test_gate_admits_1p1_gram_that_the_old_3x_gate_refused(self):
        budget = 1.5 * gram_gb(self.p)          # old gate needed 3.0x
        res, log = solve(self.op, self.y, budget, label="OLS")
        self.assertIsNotNone(res, log)
        x, info = res
        ref = np.linalg.lstsq(self.A, self.y, rcond=None)[0]
        self.assertLess(np.linalg.norm(x - ref) / np.linalg.norm(ref), 1e-9)
        self.assertEqual(info["backend"], "cpu_gram_exact")
        self.assertEqual(info["stop_reason"], "normal_equations")
        self.assertIn("exact float64 normal equations: %d features" % self.p, log)

    def test_gate_refusal_is_announced(self):
        res, log = solve(self.op, self.y, 1.0 * gram_gb(self.p), label="OLS")
        self.assertIsNone(res)
        self.assertIn("exact float64 normal equations skipped", log)
        self.assertIn("PHEASY_EXACT_GRAM_GB", log)

    def test_residual_is_the_operator_residual(self):
        cols = np.arange(5, 33)
        for ridge in (0.0, 3.0):
            res, log = solve(self.op, self.y, 10 * gram_gb(self.p), cols=cols, ridge=ridge)
            x, info = res
            As = self.A[:, cols]
            ref = np.linalg.solve(As.T @ As + ridge * np.eye(cols.size), As.T @ self.y)
            self.assertLess(np.linalg.norm(x - ref) / np.linalg.norm(ref), 1e-9, ridge)
            self.assertAlmostEqual(info["normr"] / np.linalg.norm(As @ x - self.y), 1.0,
                                   places=10)

    def test_cholesky_factors_the_accumulated_gram_itself(self):
        seen = {}
        real_acc = opt._accumulate_fold_grams
        real_cho = opt.spla.cho_factor

        def acc(*a, **k):
            out = real_acc(*a, **k)
            seen.setdefault("grams", []).append(out[0][0])
            return out

        def cho(a, *args, **kw):
            seen["factored"] = a
            seen["overwrite"] = kw.get("overwrite_a")
            return real_cho(a, *args, **kw)

        with patch.object(opt, "_accumulate_fold_grams", acc), \
                patch.object(opt.spla, "cho_factor", cho):
            res, log = solve(self.op, self.y, 1.5 * gram_gb(self.p))
        self.assertIsNotNone(res, log)
        self.assertEqual(len(seen["grams"]), 1)               # streamed once
        self.assertIs(seen["factored"], seen["grams"][0])     # no copy
        self.assertTrue(seen["overwrite"])

    def test_singular_gram_restreams_for_the_eigh_fallback(self):
        A = self.A.copy()
        A[:, -3:] = A[:, :3]                                  # exact duplicates
        y = A @ np.random.default_rng(1).standard_normal(self.p)
        op = twolevel32(A)
        calls = []
        real_acc = opt._accumulate_fold_grams

        def acc(*a, **k):
            calls.append(1)
            return real_acc(*a, **k)

        def cho_fail(*a, **k):
            raise np.linalg.LinAlgError("forced: not positive definite")

        with patch.object(opt, "_accumulate_fold_grams", acc), \
                patch.object(opt.spla, "cho_factor", cho_fail):
            res, log = solve(op, y, 5 * gram_gb(self.p))
        self.assertIsNotNone(res, log)
        x, info = res
        self.assertEqual(len(calls), 2)                       # re-streamed once
        self.assertIn("eigh", info["solver"])
        self.assertLess(np.linalg.norm(A @ x - y) / np.linalg.norm(y), 1e-6)
        # the fallback gate: 1.5 Gram admits the Cholesky but not the eigh
        with patch.object(opt.spla, "cho_factor", cho_fail):
            res, log = solve(op, y, 1.5 * gram_gb(self.p))
        self.assertIsNone(res)
        self.assertIn("eigh fallback", log)

    def test_ols_on_a_float32_operator_uses_the_exact_path(self):
        m = opt.Optimizer("OLS", cv=4, rand_seed=0, use_gpu=False)
        with patch.dict(os.environ, {"PHEASY_GPU_MODE": "cpu",
                                     "PHEASY_EXACT_GRAM_GB": repr(1.5 * gram_gb(self.p))}), \
                contextlib.redirect_stdout(io.StringIO()):
            m.fit(self.op, self.y)
        ref = np.linalg.lstsq(self.A, self.y, rcond=None)[0]
        self.assertLess(np.linalg.norm(m.results["coef"] - ref) / np.linalg.norm(ref), 1e-8)


@unittest.skipIf(torch is None, "torch not installed")
class CglsFloorCapTests(unittest.TestCase):
    def test_cap_is_per_precision(self):
        import torch as t
        self.assertEqual(gb._cgls_floor_cap(t, t.float32), gb._CGLS_FLOOR_MAX)
        self.assertEqual(gb._cgls_floor_cap(t, t.float64), gb._CGLS_FLOOR_MAX_F64)
        self.assertLess(gb._CGLS_FLOOR_MAX_F64, 1e-4)
        with patch.dict(os.environ, {"PHEASY_CGLS_FLOOR_MAX_F64": "1e-5"}):
            self.assertEqual(gb._cgls_floor_cap(t, t.float64), 1e-5)

    def test_analytic_floor_respects_the_float64_cap(self):
        import torch as t
        # an absurd condition bound must not push the float64 floor to 1e-3
        f = gb._cgls_relative_floor(t, t.float64, 1e20, alpha=1e-20)
        self.assertLessEqual(f, gb._CGLS_FLOOR_MAX_F64)
        f32 = gb._cgls_relative_floor(t, t.float32, 1e20, alpha=1e-20)
        self.assertEqual(f32, gb._CGLS_FLOOR_MAX)


if __name__ == "__main__":
    unittest.main()
