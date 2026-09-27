#!/usr/bin/env python3
"""Regression tests for the CGLS convergence certificate.

The certificate has two failure modes that a shipped c7 resident-ridge CV sweep
hit at once, so both are pinned here:

1. the tolerance floor that _lsmr_tol exists to enforce was bypassed whenever
   the operator was wrapped (--std -> _scale_columns -> _scale_operator, and
   every CV fold through _row_slice of it), leaving a float32 solve with a 1e-8
   request it can never meet;
2. the stopping test runs on the recurrence residual, which in float32 drifts
   below the true y - A x -- and the ||x||-scaled second branch of the same test
   is unbounded in ||x||, so an iterate that had come apart could certify
   itself.
"""
import os
import unittest
import warnings

from unittest import mock

import numpy as np
import scipy.sparse as sp

from core import optimizer as opt


class TestToleranceFloor(unittest.TestCase):
    def test_precision_detection_follows_the_wrapper_chain(self):
        """--std must not downgrade a float32 operator to a float64 tolerance."""
        prime = sp.eye(4, format="csr", dtype=np.float32)
        ns = sp.eye(4, format="csr", dtype=np.float64)
        base = opt.TwoLevelSM(prime, ns)
        self.assertEqual(np.dtype(opt._array_precision(base)).name, "float32")

        scaled = opt._scale_columns(base, np.ones(4))
        # _scale_operator declares dtype float64 whatever it multiplies; the
        # factors are what the arithmetic can carry.
        self.assertEqual(np.dtype(opt._array_precision(scaled)).name, "float32")
        folded = opt._row_slice(scaled, np.array([0, 1], dtype=np.intp))
        self.assertEqual(np.dtype(opt._array_precision(folded)).name, "float32")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for operator in (base, scaled, folded):
                self.assertGreaterEqual(
                    float(opt._lsmr_tol("PHEASY_LSQR_ATOL", 1e-8, operator)),
                    10.0 * float(np.finfo(np.float32).eps) - 1e-12)

    def test_precision_detection_still_reports_float64_for_dense(self):
        """The floor must not be applied to library float64 callers."""
        self.assertEqual(
            np.dtype(opt._array_precision(np.zeros(3))).name, "float64")
        # PHEASY_SM_DTYPE is only a fallback for an operator that carries no
        # array at all; a real float64 array always wins over it.
        with mock.patch.dict(os.environ, {"PHEASY_SM_DTYPE": "float32"}):
            self.assertEqual(
                np.dtype(opt._array_precision(np.zeros(3))).name, "float64")
        env = {k: v for k, v in os.environ.items() if k != "PHEASY_SM_DTYPE"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                np.dtype(opt._array_precision(None)).name, "float64")


class TestCglsFloorModel(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("torch required")

    def test_floor_is_eps_times_the_augmented_condition_bound(self):
        import torch
        from core import gpu_backend as gb
        eps32 = float(torch.finfo(torch.float32).eps)
        # cond(A_aug) <= ||A||/sqrt(alpha) for the ridge-augmented operator.
        self.assertAlmostEqual(
            gb._cgls_relative_floor(torch, torch.float32, 3.0, 0.01),
            eps32 * 3.0 / 0.1, places=12)
        self.assertAlmostEqual(
            gb._cgls_relative_floor(torch, torch.float32, 3.0, 1e-4),
            eps32 * 3.0 / 1e-2, places=12)
        # No penalty: no sigma_min bound and therefore no condition number, so
        # the one-decade rule _lsmr_tol uses.  The two paths must agree on what
        # the stored precision can reach, or a solve would be asked for a
        # tolerance the other path already knows is unreachable.
        self.assertAlmostEqual(
            gb._cgls_relative_floor(torch, torch.float64, 3.0, None),
            float(torch.finfo(torch.float64).eps) * gb._CGLS_UNPENALIZED_COND,
            places=18)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for dtype in (np.float32, np.float64):
                self.assertAlmostEqual(
                    gb._cgls_relative_floor(torch, getattr(torch, np.dtype(dtype).name),
                                            3.0, None),
                    float(opt._lsmr_tol("PHEASY_CGLS_TEST_TOL", 0.0,
                                        np.zeros(1, dtype=dtype))),
                    places=15)
        # Uncapped, alpha -> 0 would ask for a tolerance that certifies nothing.
        self.assertEqual(
            gb._cgls_relative_floor(torch, torch.float32, 1e8, 1e-20),
            gb._CGLS_FLOOR_MAX)


class TestCertifiedSolution(unittest.TestCase):
    """CUDA: a certificate is only worth what the coefficients it certifies."""

    def _ridge_float32(self, alpha, atol=1e-8, maxiter=20000, spread=3):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        rng = np.random.default_rng(11)
        nrow, ncol = 400, 120
        matrix = rng.normal(size=(nrow, ncol)).astype(np.float32)
        # Spread the column scales so the augmented system is genuinely
        # ill-conditioned at small alpha, as the real sensing matrix is.
        # spread=0 leaves it well conditioned, where the floor stays far below
        # anything that would cost accuracy.
        if spread:
            matrix = matrix * np.logspace(0, -spread, ncol, dtype=np.float32)[None, :]
        x_true = rng.normal(size=ncol)
        y = matrix.astype(np.float64) @ x_true

        # Bind the module outside the class body: a class-body name that is also
        # assigned there is not resolved from the enclosing function scope.
        t = torch

        class Operator:
            shape = (nrow, ncol)
            device = t.device("cuda:0")
            torch = t
            _value_dtype = t.float32
            def __init__(self):
                self.matrix = t.as_tensor(matrix, device=self.device)
            def matvec(self, x): return self.matrix @ x
            def rmatvec(self, x): return self.matrix.T @ x
            def norm_estimate(self, iters=10): return float(np.linalg.norm(matrix))

        coef, info = gb._iterative_ridge_tensor(
            Operator(), y, alpha, atol=atol, btol=atol, maxiter=maxiter)
        reference = np.linalg.solve(
            matrix.astype(np.float64).T @ matrix.astype(np.float64)
            + alpha * np.eye(ncol),
            matrix.astype(np.float64).T @ y)
        # _iterative_ridge_tensor returns a CUDA tensor; iterative_ridge is the
        # wrapper that copies to host.
        return coef.detach().cpu().numpy(), info, reference

    def test_converged_never_reports_a_residual_above_the_trivial_solution(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        for alpha in (1e-2, 1e-4, 1e-6):
            with self.subTest(alpha=alpha):
                coef, info, reference = self._ridge_float32(alpha)
                if info["converged"]:
                    # x = 0 already achieves ||b||, so a certified least-squares
                    # residual can never exceed it.  The old ||x||-scaled branch
                    # could pass with normr/||b|| = 1e17 and ||x|| = inf.
                    self.assertLessEqual(info["normr"], info["normb"] * (1 + 1e-6))
                    self.assertTrue(np.isfinite(coef).all())
                    # A certified solve is not garbage even on this fixture,
                    # where the floor is capped and the certificate is coarse
                    # on purpose: the point is that it is not 1e17 wrong.
                    error = (np.linalg.norm(coef - reference)
                             / np.linalg.norm(reference))
                    self.assertLess(error, 1.0,
                                    "certified coefficients are %.3e off" % error)
                else:
                    self.assertNotEqual(info["stop_reason"], "converged")

    def test_certified_float32_solution_stays_accurate_when_precision_allows(self):
        """The floor must cost nothing where the arithmetic can reach 1e-8."""
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        coef, info, reference = self._ridge_float32(1e-2, spread=0)
        self.assertTrue(info["converged"], info["stop_reason"])
        # eps * ||A||/sqrt(alpha) over-estimates the condition number of a
        # well-conditioned A, so the floor here is looser than float32 really
        # needs -- it is a bound, not a measurement.  What must hold is that the
        # delivered coefficients are still an accurate solution of THIS system.
        self.assertGreater(info["tolerance_floor"], 1e-8)
        self.assertLessEqual(info["tolerance_floor"], 1e-3)
        error = np.linalg.norm(coef - reference) / np.linalg.norm(reference)
        self.assertLess(error, 1e-2, "certified coefficients are %.3e off" % error)
        self.assertGreater(info["tolerance_floor"], 0)
        self.assertEqual(info["tolerance_floor_dtype"], "float32")

    def test_float32_request_is_raised_to_the_floor_and_stops_early(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        coef, info, reference = self._ridge_float32(1e-6)
        self.assertGreaterEqual(info["atol_effective"], info["tolerance_floor"])
        self.assertGreater(info["tolerance_floor"], 1e-8)
        self.assertEqual(info["atol"], 1e-8)          # the request is untouched
        self.assertLess(info["n_iter"], 20000)
        self.assertIn(info["honest_residual_check"],
                      ("recurrence_agreed", "recurrence_replaced"))


class TestMeasuredFloor(unittest.TestCase):
    """When the analytic floor is unreachable, measure it instead of grinding."""

    def _unpenalized_float32(self, maxiter=20000):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        rng = np.random.default_rng(41)
        nrow, ncol = 800, 200
        matrix = rng.normal(size=(nrow, ncol)).astype(np.float32)
        # No penalty and a wide spectrum: sigma_min is unbounded from below, so the
        # analytic floor (10*eps, no conditioning information) sits below what a
        # float32 CGLS can actually reach and the honest test never fires.
        matrix = matrix * np.logspace(0, -6, ncol, dtype=np.float32)[None, :]
        y = matrix.astype(np.float64) @ rng.normal(size=ncol)
        t = torch

        class Operator:
            shape = (nrow, ncol)
            device = t.device("cuda:0")
            torch = t
            _value_dtype = t.float32
            def __init__(self):
                self.matrix = t.as_tensor(matrix, device=self.device)
            def matvec(self, x): return self.matrix @ x
            def rmatvec(self, x): return self.matrix.T @ x
            def norm_estimate(self, iters=10): return float(np.linalg.norm(matrix))

        coef, info = gb._iterative_lstsq_tensor(
            Operator(), y, atol=1e-8, btol=1e-8, maxiter=maxiter)
        return coef.detach().cpu().numpy(), info, gb

    def test_forced_stall_certifies_or_refuses_but_never_grinds(self):
        """The stall mechanism must produce ONE of the two honest verdicts.
        PHEASY_CGLS_VERIFY_EVERY=1 with STALL_POINTS=2 forces the plateau: once
        the recurrence test has been caught lying, consecutive single-iteration
        verifications cannot improve by the progress margin, so the solve ends at
        the measured floor instead of at the iteration cap.  Either that floor is
        inside _CGLS_FLOOR_MAX (certified, and atol_effective must carry it) or it
        is past it (refused) -- but never a silent march to maxiter.
        The margins that decide "improving" and the headroom added to the
        certified floor both have to clear the measurement's own reproducibility
        (~2% on c7 from the sparse-kernel accumulation order), or the verdict
        follows kernel scheduling instead of the iteration.
        """
        with mock.patch.dict(os.environ, {"PHEASY_CGLS_STALL_POINTS": "2",
                                          "PHEASY_CGLS_VERIFY_EVERY": "1"}):
            coef, info, gb = self._unpenalized_float32()
        self.assertEqual(info["stop_reason"], "precision_floor", info)
        measured = info["tolerance_floor_measured"]
        self.assertIsNotNone(measured)
        self.assertLess(info["n_iter"], 20000)
        self.assertTrue(np.isfinite(coef).all())
        self.assertEqual(bool(info["converged"]),
                         bool(measured <= gb._CGLS_FLOOR_MAX))
        if info["converged"]:
            self.assertGreaterEqual(info["atol_effective"], measured * (1 - 1e-9))

    def test_unreachable_floor_stops_at_the_measured_one(self):
        coef, info, gb = self._unpenalized_float32()
        self.assertTrue(np.isfinite(coef).all())
        self.assertEqual(info["atol"], 1e-8)          # the request is untouched
        self.assertGreater(info["tolerance_floor"], 1e-8)
        # The whole point: no march to the iteration cap.
        self.assertLess(info["n_iter"], 20000, info)
        self.assertIn(info["stop_reason"],
                      ("precision_floor", "converged",
                       "converged_on_true_residual", "residual_growth"))
        if info["stop_reason"] == "precision_floor":
            measured = info["tolerance_floor_measured"]
            self.assertIsNotNone(measured)
            self.assertLessEqual(measured, gb._CGLS_FLOOR_MAX)
            if info["converged"]:
                # A certified measured floor is what the certificate used.
                self.assertGreaterEqual(info["atol_effective"],
                                        measured * (1.0 - 1e-9))


class TestSubsetLsmrFloor(unittest.TestCase):
    """The RFE subset path reads env tolerances directly; it must not."""

    def _twolevel(self, dtype):
        rng = np.random.default_rng(17)
        dense = rng.normal(size=(500, 40)) * np.logspace(0, -2, 40)
        prime = sp.csr_matrix(dense.astype(dtype))
        ns = sp.eye(40, format="csr", dtype=np.float64)
        return opt.TwoLevelSM(prime, ns), dense, dtype

    def test_wrappers_carry_the_factor_precision(self):
        """A float64-declared wrapper around float32 factors is still float32."""
        A, _, _ = self._twolevel(np.float32)
        masked = opt._make_masked_op(A, None, list(range(A.shape[1])))
        scaled = opt._scale_operator(masked, np.ones(A.shape[1]))
        folded = opt._row_slice_op(scaled, np.arange(10))
        for name, operator in (("TwoLevelSM", A), ("masked", masked),
                               ("scaled", scaled), ("folded", folded)):
            self.assertEqual(np.dtype(opt._array_precision(operator)).name,
                             "float32", name)

    def test_subset_lsmr_is_raised_to_the_floor_and_converges(self):
        A, dense, _ = self._twolevel(np.float32)
        rng = np.random.default_rng(23)
        x_true = rng.normal(size=40)
        y = dense @ x_true
        reference = np.linalg.lstsq(dense, y, rcond=None)[0]
        diag = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            coef = opt._solve_subset(A, y, None, list(range(40)),
                                     lsmr_atol=1e-8, lsmr_btol=1e-8,
                                     diag_sink=diag, diag_scope="full")
        self.assertEqual(len(diag), 1)
        self.assertTrue(diag[0]["converged"], diag[0])
        self.assertEqual(diag[0]["fit_scope"], "full")
        self.assertTrue(
            any("float32 reachable floor" in str(w.message) for w in caught),
            "the subset solve must report that 1e-8 was raised to the floor")
        # float32 stops where float32 can: this fixture has a 1e-2 column spread,
        # so 1e-2 on the coefficients is the right order for a certified solve and
        # the point of the test is that it is not garbage (it was 1e17 before).
        self.assertLess(
            np.linalg.norm(coef - reference) / np.linalg.norm(reference), 1e-2)

    def test_subset_float64_is_not_given_the_float32_floor(self):
        A, dense, _ = self._twolevel(np.float64)
        y = dense @ np.random.default_rng(29).normal(size=40)
        diag = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt._solve_subset(A, y, None, list(range(40)),
                              lsmr_atol=1e-8, lsmr_btol=1e-8, diag_sink=diag)
        self.assertTrue(diag[0]["converged"], diag[0])
        self.assertFalse(
            any("reachable floor" in str(w.message) for w in caught))

class TestSparseLsqrFloor(unittest.TestCase):
    """The sparse/operator LSQR path must read the precision off the MATRIX."""

    def _problem(self):
        rng = np.random.default_rng(3)
        dense = (rng.normal(size=(600, 80))
                 * np.logspace(0, -2, 80)).astype(np.float32)
        y = dense.astype(np.float64) @ rng.normal(size=80)
        ref = np.linalg.lstsq(dense.astype(np.float64), y, rcond=None)[0]
        return dense, y, ref

    def test_float32_sparse_lsqr_is_raised_to_the_floor_and_converges(self):
        dense, y, ref = self._problem()
        info = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            coef = opt._solve_sparse_lsqr(sp.csr_matrix(dense), y, info=info)
        self.assertTrue(info["converged"], info)
        self.assertTrue(
            any("float32 reachable floor" in str(w.message) for w in caught),
            "a float32 matrix must report that 1e-8 was raised")
        self.assertLess(
            np.linalg.norm(coef - ref) / np.linalg.norm(ref), 1e-2)

    def test_float64_sparse_lsqr_is_not_given_the_float32_floor(self):
        """The ambient PHEASY_SM_DTYPE must not loosen a float64 solve.

        Reading the precision from the matrix is the whole point: passing no
        matrix made _array_precision fall back to the environment, so a float64
        sparse solve inherited the float32 floor whenever PHEASY_SM_DTYPE was
        set -- it stopped ~190 iterations early for ~100x the coefficient error.
        """
        dense, y, ref = self._problem()
        results = {}
        for dtype in (np.float32, np.float64):
            info = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with mock.patch.dict(os.environ, {"PHEASY_SM_DTYPE": "float32"}):
                    coef = opt._solve_sparse_lsqr(
                        sp.csr_matrix(dense.astype(dtype)), y, info=info)
            results[dtype] = (info,
                              float(np.linalg.norm(coef - ref) / np.linalg.norm(ref)))
        info32, err32 = results[np.float32]
        info64, err64 = results[np.float64]
        self.assertTrue(info32["converged"] and info64["converged"])
        self.assertGreater(info64["itn"], info32["itn"],
                           "the float64 solve must not stop where float32 had to")
        self.assertLess(err64 * 10.0, err32)

class TestUnconditionalProbe(unittest.TestCase):
    """A solve whose CHEAP test never fires must still report a measured floor.

    Production case (Mg8C120 OLS, 454656x69487, float32): 20000 CGLS iterations
    reached a criterion of 1.33e-03 against an analytic floor of 1.19e-06 -- three
    orders below what the arithmetic can express.  The recurrence test never fired,
    so the honest block -- and with it the stall detector -- never ran either, and
    the fit was refused as a bare iteration_limit with nothing to act on.  The
    unconditional probe from probe_start is what fixes that; these tests pin that
    it runs, that it reports, and that it never certifies above the cap.
    """

    def _lstsq_float32(self, atol=1e-12, maxiter=1000, spread=3, probe_start=None):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        rng = np.random.default_rng(23)
        nrow, ncol = 400, 120
        matrix = rng.normal(size=(nrow, ncol)).astype(np.float32)
        if spread:
            matrix = matrix * np.logspace(0, -spread, ncol, dtype=np.float32)[None, :]
        x_true = rng.normal(size=ncol)
        y = matrix.astype(np.float64) @ x_true
        t = torch

        class Operator:
            shape = (nrow, ncol)
            device = t.device("cuda:0")
            torch = t
            _value_dtype = t.float32

            def __init__(self):
                self.matrix = t.as_tensor(matrix, device=self.device)

            def matvec(self, x):
                return self.matrix @ x

            def rmatvec(self, x):
                return self.matrix.T @ x

            def norm_estimate(self, iters=10):
                return float(np.linalg.norm(matrix))

        env = {} if probe_start is None else {
            "PHEASY_CGLS_PROBE_START": str(int(probe_start))}
        with mock.patch.dict(os.environ, env):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coef, info = gb._iterative_lstsq_tensor(
                    Operator(), y, atol=atol, btol=atol, maxiter=maxiter)
        return coef.detach().cpu().numpy(), info, gb

    def test_probe_reports_a_floor_instead_of_a_bare_iteration_limit(self):
        coef, info, gb = self._lstsq_float32()
        self.assertTrue(np.isfinite(coef).all())
        self.assertGreater(info["probe_count"], 0, info)
        self.assertIsNotNone(info["stall_floor"], info)
        self.assertIsNotNone(info["stall_iteration"], info)
        self.assertLessEqual(info["stall_iteration"], info["n_iter"], info)
        self.assertTrue(np.isfinite(info["criterion_value"]))
        if info["converged"]:
            # Only a floor inside the cap may be certified.
            self.assertEqual(info["stop_reason"], "precision_floor", info)
            self.assertLessEqual(info["tolerance_floor_measured"],
                                 gb._CGLS_FLOOR_MAX)
        else:
            # Above the cap the verdict must SAY so rather than blame the budget.
            self.assertIn(info["stop_reason"],
                          ("stall_above_floor", "precision_floor", "iteration_limit"),
                          info)

    def test_probe_can_be_positioned_and_turned_off(self):
        _, info_off, _ = self._lstsq_float32(maxiter=400, probe_start=10 ** 9)
        self.assertEqual(info_off["probe_count"], 0)
        _, info_on, _ = self._lstsq_float32(maxiter=400, probe_start=50)
        self.assertGreater(info_on["probe_count"], 0)
        self.assertEqual(info_on["probe_start"], 50)

    def test_probe_window_scales_with_the_budget(self):
        """Never before half the budget, and never before _CGLS_PROBE_WINDOW_MAX.

        A window that is short compared with the budget declares a "floor" on a
        criterion that is still creeping down (the FISTA lesson), so the default
        start is min(8000, max(maxiter // 2, verify_every)).
        """
        from core import gpu_backend as gb
        _, info, _ = self._lstsq_float32(maxiter=400)
        self.assertGreaterEqual(info["probe_start"], 200)
        self.assertLessEqual(info["probe_start"], 400)
        self.assertLessEqual(info["probe_start"], gb._CGLS_PROBE_WINDOW_MAX)


    def test_above_cap_floor_is_reported_and_never_stops_early(self):
        """The new verdict: a floor past the cap is named, not acted on.

        A measured floor above _CGLS_FLOOR_MAX cannot be certified, but it must
        not END the run either -- the recurrence is still creeping down there, so
        stopping would deliver a worse iterate than the remaining budget reaches.
        Measured on the MgC-class fixture: stall_floor 3.68e-03 (> cap 1e-03), the
        delivered iterate IS the best one the probes saw, and n_iter equals the
        budget.
        """
        from core import gpu_backend as gb
        coef, info, _ = self._lstsq_float32(maxiter=1000)
        if info["converged"]:
            self.skipTest("this fixture certified; the above-cap path needs the other branch")
        self.assertEqual(info["stop_reason"], "stall_above_floor", info)
        self.assertIsNone(info["tolerance_floor_measured"],
                          "an uncertified floor must not be promoted")
        self.assertEqual(int(info["n_iter"]), 1000, info)          # never stops early
        self.assertGreater(info["stall_floor"], gb._CGLS_FLOOR_MAX, info)
        # The best iterate is what was delivered, not the last one the loop made.
        self.assertLessEqual(info["criterion_value"],
                             info["best_criterion"] * (1.0 + 1e-9), info)
        self.assertGreater(info["probe_count"], 1, info)

    def test_probe_position_does_not_change_the_measured_floor(self):
        """Probing earlier must not move the number: it measures, it does not act.

        With the probe starting at iteration 50 instead of the default midpoint,
        the floor reported for the same system is the same value -- the detector
        keeps the BEST criterion over the run rather than whatever it saw first.
        """
        _, late, _ = self._lstsq_float32(maxiter=1000)
        _, early, _ = self._lstsq_float32(maxiter=1000, probe_start=50)
        self.assertEqual(early["stop_reason"], late["stop_reason"])
        if late["stall_floor"] is not None and early["stall_floor"] is not None:
            self.assertAlmostEqual(early["stall_floor"], late["stall_floor"],
                                   delta=0.25 * late["stall_floor"])
        self.assertGreaterEqual(early["probe_count"], late["probe_count"])


class TestSubsetFloorAcceptance(unittest.TestCase):
    """A RANKING subset solve may deliver at the measured floor; a budget limit may not.

    Measured on the production MgC operator in required GPU mode: the first RFE round
    lands on stall_above_floor after 5000 iterations, and the fail-closed subset solve
    aborted the whole fit for a solve whose only job is to rank features.  The two
    reasons a solve can end uncertified must therefore be separated: a MEASURED floor
    (precision_floor / stall_above_floor, i.e. the criterion was seen to stop improving
    at this precision) is acceptable for ranking and is recorded; an exhausted budget
    with no floor evidence still raises.
    """

    def _info(self, stop_reason, converged=False):
        return dict(solver="GPU CGLS", converged=converged, stop_reason=stop_reason,
                    itn=5000, n_iter=5000, stall_floor=3.7e-03, stall_iteration=800,
                    normr=1.0, normar=1.0)

    def _base(self):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        prime = sp.eye(6, format="csr", dtype=np.float32)
        op = gb.GpuTwoLevelOperator(opt.TwoLevelSM(prime, sp.eye(6, format="csr")),
                                    device_ids=[0])
        cols = torch.arange(6, dtype=torch.long, device=op.device)
        scale = torch.ones(6, dtype=op._value_dtype, device=op.device)
        return gb, op, cols, scale

    def test_measured_floor_is_accepted_for_ranking_but_budget_is_not(self):
        import torch
        gb, op, cols, scale = self._base()
        zero = torch.zeros(6, dtype=op._value_dtype, device=op.device)
        stalled = self._info("stall_above_floor")
        with mock.patch.object(gb, "_iterative_lstsq_tensor",
                               return_value=(zero, dict(stalled))):
            with self.assertRaises(RuntimeError):
                gb.solve_resident_subset(op, np.zeros(6), cols,
                                         column_scale=scale)
            coef, info = gb.solve_resident_subset(
                op, np.zeros(6), cols, column_scale=scale,
                accept_measured_floor=True)
            self.assertTrue(info["floor_accepted"], info)
            self.assertIn("MEASURED precision floor", info["floor_note"])
            self.assertEqual(info["stop_reason"], "stall_above_floor")
        budget = self._info("iteration_limit")
        with mock.patch.object(gb, "_iterative_lstsq_tensor",
                               return_value=(zero, dict(budget))):
            with self.assertRaises(RuntimeError):
                gb.solve_resident_subset(op, np.zeros(6), cols,
                                         column_scale=scale,
                                         accept_measured_floor=True)


if __name__ == "__main__":
    unittest.main()
