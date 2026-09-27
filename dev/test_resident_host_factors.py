"""Focused tests for the cached canonical host factors (CPU-only, no pytest).

`twolevel_host_factors` exists to remove one O(nnz) column-slice transpose PER
SHARD from every resident construction outside the LASSO path.  Measured on the
MgC operator (454656x69487, nnz(SM_prime) 1.127e9, 2 shards) that was 456.9 s of
a 1298.8 s full-scale RIDGE fit, so these tests pin the behaviour that makes it
safe: the adjoint handed to the operator is the true transpose of the canonical
prime, retention on the operator is a host-memory decision that an environment
variable can override, and the RIDGE construction site actually passes it.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import gpu_backend as GB
from core import optimizer as O


def _operator(n_rows=40, n_mid=12, n_cols=9, seed=0):
    prime = sp.random(n_rows, n_mid, density=0.4, format="csr",
                      random_state=seed, dtype=np.float64)
    prime.data = np.round(prime.data, 3) + 0.1
    prime.sum_duplicates()
    ns = sp.random(n_mid, n_cols, density=0.5, format="csr",
                   random_state=seed + 1, dtype=np.float64)
    return O.TwoLevelSM(prime, ns)


class HostFactorTest(unittest.TestCase):
    def test_canonical_prime_t_is_the_true_adjoint(self):
        A = _operator()
        prime_c, ns_c, prime_t = GB.twolevel_host_factors(A)
        self.assertEqual(prime_t.shape, (A.SM_prime.shape[1], A.SM_prime.shape[0]))
        self.assertEqual(prime_t.nnz, prime_c.nnz)
        self.assertTrue(np.array_equal(prime_t.toarray(), prime_c.toarray().T))
        self.assertTrue(np.array_equal(ns_c.toarray(), A.NS.toarray()))

    def test_factors_are_kept_on_the_operator_when_asked(self):
        with mock.patch.dict(os.environ, {"PHEASY_HOST_FACTOR_CACHE": "1"}):
            A = _operator()
            first = GB.twolevel_host_factors(A)
            # A second caller (RIDGE then OLS on the same operator) must reuse
            # the same arrays instead of paying another copy plus transpose.
            self.assertIs(GB.twolevel_host_factors(A), first)
        self.assertIs(getattr(A, "_canonical_host_factors", None), first)

    def test_cache_can_be_switched_off(self):
        with mock.patch.dict(os.environ, {"PHEASY_HOST_FACTOR_CACHE": "0"}):
            A = _operator()
            first = GB.twolevel_host_factors(A)
            second = GB.twolevel_host_factors(A)
        self.assertIsNone(getattr(A, "_canonical_host_factors", None))
        self.assertIsNot(first, second)
        self.assertTrue(np.array_equal(first[0].toarray(), second[0].toarray()))
        self.assertTrue(np.array_equal(first[2].toarray(), second[2].toarray()))

    def test_auto_keeps_only_under_the_host_headroom(self):
        env = {k: v for k, v in os.environ.items()
               if k != "PHEASY_HOST_FACTOR_CACHE"}
        with mock.patch.dict(os.environ, env, clear=True):
            tight = _operator()
            roomy = _operator()
            # The fixture is tiny, so the "is it worth keeping at all" floor has
            # to be taken out of the way to test the HEADROOM rule in isolation.
            with mock.patch.object(GB, "_HOST_FACTOR_CACHE_MIN_BYTES", 0):
                with mock.patch.object(GB, "_host_available_bytes", lambda: 1):
                    GB.twolevel_host_factors(tight)
                with mock.patch.object(GB, "_host_available_bytes", lambda: 1 << 40):
                    GB.twolevel_host_factors(roomy)
        self.assertIsNone(getattr(tight, "_canonical_host_factors", None))
        self.assertIsNotNone(getattr(roomy, "_canonical_host_factors", None))

    def test_tiny_factor_sets_are_not_kept(self):
        """Retention buys nothing when the transpose is cheap to rebuild."""
        env = {k: v for k, v in os.environ.items()
               if k != "PHEASY_HOST_FACTOR_CACHE"}
        with mock.patch.dict(os.environ, env, clear=True):
            A = _operator()
            with mock.patch.object(GB, "_host_available_bytes", lambda: 1 << 40):
                GB.twolevel_host_factors(A)
        self.assertIsNone(getattr(A, "_canonical_host_factors", None))
        self.assertGreater(GB._HOST_FACTOR_CACHE_MIN_BYTES, 0)

    def test_unreadable_meminfo_is_not_treated_as_room(self):
        env = {k: v for k, v in os.environ.items()
               if k != "PHEASY_HOST_FACTOR_CACHE"}
        with mock.patch.dict(os.environ, env, clear=True):
            A = _operator()
            with mock.patch.object(GB, "_host_available_bytes", lambda: None):
                GB.twolevel_host_factors(A)
        self.assertIsNone(getattr(A, "_canonical_host_factors", None))

    def test_resident_ridge_op_hands_over_the_canonical_factors(self):
        """The RIDGE construction site must pass host_factors, not raw factors."""
        seen = {}

        class _FakeTorch:
            @staticmethod
            def as_tensor(x, **kw):
                return np.asarray(x)

        class _FakeOp:
            def __init__(self, base, **kw):
                seen.update(kw)
                seen["base"] = base
                self.torch = _FakeTorch()
                self._value_dtype = np.dtype(np.float64)
                self.device = "cpu"
                self.input_scale = None

        A = _operator()
        with mock.patch.dict(os.environ, {"PHEASY_HOST_FACTOR_CACHE": "1"}):
            with mock.patch.object(GB, "GpuTwoLevelOperator", _FakeOp):
                with mock.patch.object(GB, "resident_device_ids", lambda: [0]):
                    op = O._resident_ridge_op(A)
                    self.assertIsInstance(op, _FakeOp)
                    factors = seen.get("host_factors")
                    self.assertIsNotNone(
                        factors, "resident RIDGE must pass canonical host factors")
                    self.assertTrue(np.array_equal(
                        factors[0].toarray(), A.SM_prime.toarray()))
                    self.assertTrue(np.array_equal(
                        factors[2].toarray(), A.SM_prime.toarray().T))
                    # cached on the operator: a second fit reuses the same
                    # operator AND the same host factors.
                    self.assertIs(O._resident_ridge_op(A), op)
        self.assertIs(GB.twolevel_host_factors(A), factors)


    def test_adjoint_cache_round_trip_and_stale_rejection(self):
        """The on-disk adjoint cache must round-trip and never serve a stale entry."""
        A = _operator(n_rows=60, n_mid=14, n_cols=9)
        with tempfile.TemporaryDirectory() as d:
            env = {"PHEASY_HT_CACHE_DIR": d,
                   "PHEASY_HT_CACHE_KEY": "unit-test-key"}
            with mock.patch.dict(os.environ, env):
                first = GB.twolevel_host_factors(A)[2]
                paths = [os.path.join(d, f) for f in os.listdir(d)]
                self.assertEqual(len(paths), 1, paths)
                _mtime = os.path.getmtime(paths[0])
                second = GB.twolevel_host_factors(A)[2]
                self.assertTrue(np.array_equal(first.toarray(), second.toarray()))
                self.assertEqual(second.nnz, A.SM_prime.nnz)
                # A cache HIT must not rewrite the entry (a rewrite would mean the
                # entry was rejected and rebuilt, i.e. the cache never pays off).
                self.assertEqual(os.path.getmtime(paths[0]), _mtime)
                # A truncated entry must be rebuilt, not uploaded.
                with open(paths[0], "wb") as fh:
                    fh.write(b"not an npz")
                third = GB.twolevel_host_factors(A)[2]
                self.assertTrue(np.array_equal(first.toarray(), third.toarray()))

    def test_adjoint_cache_is_off_without_a_directory(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("PHEASY_HT_CACHE_DIR", "PHEASY_HT_CACHE_KEY")}
        with mock.patch.dict(os.environ, env, clear=True):
            A = _operator()
            adj = GB.twolevel_host_factors(A)[2]
            self.assertTrue(np.array_equal(adj.toarray(), A.SM_prime.toarray().T))

if __name__ == "__main__":
    unittest.main()
