#!/usr/bin/env python3
"""Regression test for the transpose of a resident row view ([FIX P50]).

A resident row view is what _row_slice returns once a RIDGE CV has cached
_gpu_ridge_op on the operator.  It is deliberately NOT a scipy LinearOperator:
it exposes matvec/rmatvec and returns numpy.  The transpose, however, is needed
by every other consumer of a row slice -- above all FISTA, which asks for
A.T @ y, A.T @ (Az - y) and A.T @ u.  It used to be missing (AttributeError),
which killed any LASSO fit that ran after a RIDGE fit on the same operator
(measured: the Mg2C60 2x2x2 fc2 fit, jobid 2119).

These tests are CPU-only and never silently skip: the resident device is
simulated with a fake torch + fake subset operator that compute the exact same
row-selected matrix the real GpuSubsetOperator multiplies, so the numbers are
checkable against the dense slice.
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import gpu_backend as GB
from core import optimizer as opt


class _FakeTorch:
    @staticmethod
    def as_tensor(x, dtype=None, device=None):
        return np.asarray(x, dtype=np.dtype(dtype).type if dtype is not None else None)


class _FakeTensor:
    """Mimics the torch tensor surface _ResidentRowView unwraps."""

    def __init__(self, a):
        self._a = np.asarray(a, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _FakeSubset:
    """The exact operator the view stands for: (prime @ ns)[rows, :]."""

    def __init__(self, resident, cols, rows):
        self._parent = resident
        self.cols = np.asarray(cols, dtype=np.intp)
        self.rows = np.asarray(rows, dtype=np.intp)
        self.device = "cpu"
        self._value_dtype = resident._value_dtype

    def matvec(self, v):
        dense = self._parent.dense
        return _FakeTensor(dense[np.ix_(self.rows, self.cols)] @ np.asarray(v))

    def rmatvec(self, u):
        dense = self._parent.dense
        full = np.zeros(dense.shape[0], dtype=np.float64)
        np.add.at(full, self.rows, np.asarray(u))
        return _FakeTensor(dense.T @ full)


class _FakeResident:
    """Stands in for GpuTwoLevelOperator, carrying the dense matrix it views."""

    def __init__(self, dense, n_gpu=None):
        self.dense = np.asarray(dense, dtype=np.float64)
        self.torch = _FakeTorch()
        self.device = "cpu"
        self._value_dtype = np.dtype(np.float64)
        self.n_gpu = n_gpu


def _dense(A):
    """TwoLevelSM is a LinearOperator: materialise it through matmat."""
    return np.asarray(A.matmat(np.eye(int(A.shape[1]), dtype=np.float64)),
                      dtype=np.float64)


def _operator(seed=0, n_rows=24, n_mid=7, n_cols=5):
    rng = np.random.default_rng(seed)
    prime = sp.csr_matrix(rng.normal(size=(n_rows, n_mid)))
    ns = sp.csr_matrix(rng.normal(size=(n_mid, n_cols)))
    return opt.TwoLevelSM(prime, ns)


class ResidentRowViewTransposeTest(unittest.TestCase):
    def _view(self, A):
        fake = _FakeResident(_dense(A))
        A._gpu_ridge_op = fake
        with mock.patch.object(GB, "GpuSubsetOperator", _FakeSubset):
            view = opt._row_slice(A, np.array([3, 0, 2], dtype=np.intp))
            view._subset()      # build inside the patch: _subset is lazy
        return view, fake

    def test_row_slice_of_a_resident_operator_is_a_resident_view(self):
        A = _operator()
        view, _ = self._view(A)
        self.assertIsInstance(view, opt._ResidentRowView)

    def test_transpose_exists_and_swaps_matvec_rmatvec(self):
        A = _operator()
        view, _ = self._view(A)
        AT = view.T                      # AttributeError before [FIX P50]
        self.assertEqual(AT.shape, view.shape[::-1])
        u = np.arange(view.shape[0], dtype=np.float64)
        v = np.arange(view.shape[1], dtype=np.float64) + 1.0
        np.testing.assert_allclose(AT @ u, view.rmatvec(u))
        np.testing.assert_allclose(AT.rmatvec(v), view.matvec(v))

    def test_transpose_matches_the_dense_slice(self):
        A = _operator()
        rows = np.array([3, 0, 2], dtype=np.intp)
        view, _ = self._view(A)
        dense = _dense(A)[rows, :]
        AT = view.T
        u = np.arange(len(rows), dtype=np.float64)
        np.testing.assert_allclose(AT @ u, dense.T @ u, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(view @ np.ones(dense.shape[1]), dense @ np.ones(dense.shape[1]))

    def test_transpose_does_not_recurse(self):
        """The adjoint is a class over bound methods, never a closure on .T.

        b6b7f92 had to fix exactly this class of recursion (a closure that
        captured a name later rebound to the transpose) in _solve_subset.  A
        second, shadowing definition of __matmul__/T in one class body is the
        same failure mode in another disguise ([FIX P50b]): Python keeps the
        LAST definition, so the effective operator silently changes.
        """
        A = _operator()
        view, _ = self._view(A)
        self.assertIs(type(view.T), opt._ResidentRowAdjoint)
        self.assertIsNot(view.T, view)
        # .T round-trips through the SAME objects: no new operator is built
        # from a closure, and no chain of transposes grows.
        self.assertIs(view.T.T, view)
        self.assertIs(view.transpose().T, view)
        # behavioural: the adjoint dispatches to the VIEW bound methods
        # (patched with sentinels here).  A closure over self.T would either
        # recurse or never call these.
        calls = {"rmatvec": 0, "matvec": 0}
        real_r, real_m = view.rmatvec, view.matvec

        def spy_r(u):
            calls["rmatvec"] += 1
            return real_r(u)

        def spy_m(v):
            calls["matvec"] += 1
            return real_m(v)

        view.rmatvec, view.matvec = spy_r, spy_m
        try:
            view.T @ (np.arange(view.shape[0], dtype=np.float64) + 1.0)
            self.assertEqual(calls["rmatvec"], 1)
            self.assertEqual(calls["matvec"], 0)
        finally:
            view.rmatvec, view.matvec = real_r, real_m
        v = np.arange(view.shape[1], dtype=np.float64) + 1.0
        np.testing.assert_allclose(view.T.T @ v, view @ v)

    def test_operator_product_supports_2d_operands(self):
        """[FIX P50b] A @ B and A.T @ B must work for a 2-D B.

        _compute_gram multiplies I_blk (n_features x blk) and then A.T
        multiplies that block, so a view that forwards only 1-D operands
        makes the FISTA Gram path raise and fall back to the slow matvec
        loop.  The duplicate __matmul__ that shadowed the 2-D one did that.
        """
        A = _operator()
        view, _ = self._view(A)
        dense = _dense(A)[view._rows, :]      # the view's OWN row selection
        B = (np.arange(view.shape[1] * 4, dtype=np.float64)
             .reshape(view.shape[1], 4) / 7.0)
        np.testing.assert_allclose(view @ B, dense @ B,
                                   rtol=1e-12, atol=1e-12)
        C = (np.arange(view.shape[0] * 3, dtype=np.float64)
             .reshape(view.shape[0], 3) / 5.0)
        np.testing.assert_allclose(view.T @ C, dense.T @ C,
                                   rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError):
            view @ np.zeros((2, 2, 2))

    def test_compute_gram_runs_on_a_resident_view(self):
        """The real consumer: _compute_gram(view) must build the SAME G and b.

        _fista_lasso calls _compute_gram on the fold views inside try/except,
        so a broken 2-D product does not crash -- it prints "Gram build
        failed" and drops to the matvec path, which is how the shadowing bug
        could hide.  Assert the Gram itself.
        """
        A = _operator()
        view, _ = self._view(A)
        dense = _dense(A)[view._rows, :]      # the view's OWN row selection
        y = np.arange(view.shape[0], dtype=np.float64) * 0.25
        G, b, _how = opt._compute_gram(view, y)
        G_ref, b_ref, _ = opt._compute_gram(dense, y)
        np.testing.assert_allclose(G, G_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(b, b_ref, rtol=1e-10, atol=1e-12)

    def test_factor_dtype_provenance_carries_through_view_and_adjoint(self):
        """The floor must read the FACTOR dtype, not the wrapper's shell."""
        A = _operator()
        view, _ = self._view(A)
        self.assertEqual(opt._array_precision(view), np.float64)
        # the adjoint must not look like a FRESH float64 operator: it carries
        # the same provenance as the view it came from.
        self.assertEqual(opt._array_precision(view.T),
                         opt._array_precision(view))
        self.assertEqual(np.dtype(view.T._data_dtype), np.dtype(np.float64))
        # [FIX P50c] a float32 FACTOR parent must be reported as float32, not
        # as this view's declared float64 shell: the tolerance floor reads it,
        # and a 1e-7 request against a float32 operator is unreachable.
        A32 = _operator()
        A32.SM_prime.data = A32.SM_prime.data.astype(np.float32)
        view32, _ = self._view(A32)
        self.assertEqual(opt._array_precision(view32._gpu_ridge_parent[3]),
                         np.float32)
        self.assertEqual(opt._array_precision(view32), np.float32)
        self.assertEqual(opt._array_precision(view32.T), np.float32)

    def test_fista_lasso_runs_on_a_resident_view(self):
        """The exact crash path: FISTA straight after a cached resident op.

        Before [FIX P50] this raised AttributeError on A.T; adding only .T
        moves the failure one line down to A @ z, which is why the view needs
        __matmul__ as well.  The coefficients must match the dense slice,
        because the view multiplies exactly that matrix.
        """
        A = _operator()
        rows = np.array([3, 0, 2], dtype=np.intp)
        view, _ = self._view(A)
        y = np.arange(len(rows), dtype=np.float64) * 0.5
        info = {}
        coef = opt._fista_lasso(view, y, 1e-3, max_iter=60, tol=1e-12,
                                _info=info)
        self.assertEqual(np.asarray(coef).shape, (view.shape[1],))
        self.assertTrue(np.all(np.isfinite(coef)))
        self.assertIn("kkt_relative", info)
        ref = opt._fista_lasso(_dense(A)[rows, :], y, 1e-3, max_iter=60,
                               tol=1e-12, _info={})
        np.testing.assert_allclose(coef, ref, rtol=1e-8, atol=1e-10)

    def test_cached_resident_op_does_not_change_the_lasso_fit(self):
        """The end-to-end contract of the reported failure (jobid 2119).

        A RIDGE CV caches _gpu_ridge_op on the operator; the LASSO fit that
        follows must slice its CV folds through that cache and produce EXACTLY
        the same answer as a pristine operator.  Before [FIX P50] the fit
        raised AttributeError on the first fold; the cache has to stay
        invisible in the numbers, not merely stop crashing.
        """
        y_model = np.array([1., -.5, 0., .3, 0.])

        def _fit(cached):
            A = _operator()
            y = _dense(A) @ y_model
            if cached:
                A._gpu_ridge_op = _FakeResident(_dense(A))
            with mock.patch.dict(os.environ, {
                    "PHEASY_GPU_LASSO_RESIDENT": "0",
                    "PHEASY_GPU_TWOLEVEL_LASSO": "1",
                    "PHEASY_LASSO_DEBIAS": "0",
                    "PHEASY_CV_MAX_ITER": "100",
                    "PHEASY_CV_TOL": "1e-7",
                    "PHEASY_SEED": "7"}):
                with mock.patch.object(GB, "GpuSubsetOperator", _FakeSubset):
                    model = opt.Optimizer("lasso", alpha=[.01, .05], nalpha=2,
                                          cv=2, tol=1e-7, max_iter=200,
                                          use_gpu=False, standardize=False,
                                          rand_seed=7)
                    model.fit(A, y)
            return model

        pristine = _fit(False)
        cached = _fit(True)
        self.assertEqual(pristine.results["execution_backend"],
                         cached.results["execution_backend"])
        self.assertEqual(cached.results["execution_backend"],
                         "cpu_iterative_fista")
        np.testing.assert_array_equal(np.asarray(cached.results["coef"]),
                                      np.asarray(pristine.results["coef"]))

    def test_lasso_backend_after_a_cached_resident_op_is_iterative(self):
        """resident=false + GPU_SM=1 keeps the documented CPU-FISTA path."""
        A = _operator()
        with mock.patch.dict(os.environ, {"PHEASY_GPU_LASSO_RESIDENT": "0",
                                          "PHEASY_GPU_TWOLEVEL_LASSO": "1"}):
            self.assertEqual(opt._lasso_backend(A), "iterative")
            view, _ = self._view(A)
            self.assertIsInstance(view, opt._ResidentRowView)
            self.assertEqual(opt._lasso_backend(view), "iterative")
            self.assertIsNotNone(view.T)      # the fold path FISTA then needs

    def test_ridge_fold_solve_still_uses_the_parent_view(self):
        """_ridge_solve must keep taking the _gpu_ridge_parent shortcut."""
        A = _operator()
        view, _ = self._view(A)
        self.assertIsNotNone(view._gpu_ridge_parent)
        parent = view._gpu_ridge_parent
        self.assertEqual(len(parent), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
