#!/usr/bin/env python3
"""[FIX COLNORM-STREAM] a streamed block pass for the column norms.

The resident two-level operator holds SM_prime (sharded) for the whole fit.  At
a large third-order cutoff that no longer fits any consumer card -- c3=5.0 needs
27.6 GB per card against a 24.9 GB budget on a 23.68 GiB RTX3090 -- while the
norms themselves are just a sum of per-row-block squares and need no residency.

So a failed RESIDENT pass now retries as a STREAMED pass: one block on the
device, computed, squared, dropped.  Device memory is O(NS + block).  It is
still the GPU path -- the CPU loop is NOT restored (0008/ad8ed52 removed it for
required mode).

What these tests pin down:
  * the streamed result equals the CPU definition (same block loop, float64);
  * the resident path is still preferred and is not bypassed;
  * the streamed pass is tried before the hard error, in required mode;
  * a host with no CUDA still uses the CPU definition (nothing to stream on);
  * required mode still fails closed when BOTH GPU paths fail.

The float64 comparisons need a CUDA device and are skipped without one; the
policy tests are pure host code.

    python -m unittest dev.test_colnorm_streaming
"""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import gpu_backend as GB      # noqa: E402
from core import optimizer as O         # noqa: E402


def _twolevel(n_rows=200, n_cols=7, k=13, seed=0, sparse_ns=False, dtype=np.float32):
    rng = np.random.default_rng(seed)
    prime = sp.random(n_rows, k, density=0.15, format="csr", random_state=seed)
    prime.data = prime.data.astype(dtype)
    ns = rng.normal(size=(k, n_cols))
    if sparse_ns:
        ns = sp.csr_matrix(np.where(np.abs(ns) > 0.5, ns, 0.0))
    A = O.TwoLevelSM(prime, ns)
    exact = np.linalg.norm(np.asarray(prime @ (ns.toarray() if sp.issparse(ns) else ns),
                                      dtype=np.float64), axis=0)
    return A, exact


def _cuda():
    try:
        return bool(GB.enabled() and GB.available())
    except Exception:
        return False


class _BrokenResident:
    """A resident operator that always fails to build."""

    def __init__(self, *a, **kw):
        raise GB.ResidentFootprintError("estimate 27596127127 bytes exceeds budget 24887820288")


class _FakeTensor:
    """Just enough of the torch tensor surface for _col_norms_on_gpu."""

    def __init__(self, arr):
        self._a = np.asarray(arr, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _WorkingResident:
    """A resident operator that reports a sentinel so we can see it was used."""

    used = 0

    def __init__(self, A, **kw):
        self.shape = A.shape
        self.torch = None
        _WorkingResident.used += 1

    def col_norms(self):
        return _FakeTensor(np.full(self.shape[1], 42.0))


class StreamingMathTests(unittest.TestCase):
    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_streamed_matches_the_cpu_definition(self):
        for sparse_ns in (False, True):
            A, exact = _twolevel(sparse_ns=sparse_ns, seed=1 if sparse_ns else 0)
            got = GB.col_norms_streaming(A)
            np.testing.assert_allclose(got, exact, rtol=1e-12, atol=0.0)

    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_streamed_matches_the_cpu_loop_on_a_wide_operator(self):
        A, exact = _twolevel(n_rows=4096, n_cols=97, k=41, seed=3)
        got = GB.col_norms_streaming(A)
        np.testing.assert_allclose(got, exact, rtol=1e-12, atol=0.0)

    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_a_tiny_block_budget_gives_the_same_answer(self):
        A, exact = _twolevel(n_rows=512, n_cols=11, k=17, seed=4)
        got = GB.col_norms_streaming(A, block_bytes=4096)
        np.testing.assert_allclose(got, exact, rtol=1e-12, atol=0.0)

    @unittest.skipUnless(_cuda(), "needs CUDA")
    def test_shape_mismatch_is_refused(self):
        A, _ = _twolevel(seed=5)
        A.NS = np.zeros((3, 7))
        with self.assertRaises(ValueError):
            GB.col_norms_streaming(A)

    def test_without_cuda_it_refuses_instead_of_running_something_else(self):
        A, _ = _twolevel(seed=6)
        with mock.patch.object(GB, "available", return_value=False):
            with self.assertRaises(RuntimeError):
                GB.col_norms_streaming(A)


class ResidentIsPreferredTests(unittest.TestCase):
    """The streamed pass must never take over from a working resident operator."""

    def test_resident_is_used_when_it_builds(self):
        A, _ = _twolevel(seed=7)
        _WorkingResident.used = 0
        with mock.patch.object(GB, "enabled", return_value=True), \
                mock.patch.object(GB, "available", return_value=True), \
                mock.patch.object(GB, "resident_device_ids", return_value=[0]), \
                mock.patch.object(GB, "twolevel_host_factors", return_value=None), \
                mock.patch.object(GB, "GpuTwoLevelOperator", _WorkingResident), \
                mock.patch.object(GB, "col_norms_streaming",
                                  side_effect=AssertionError("streamed pass used")), \
                contextlib.redirect_stdout(io.StringIO()):
            out = A.col_norms()
        self.assertEqual(_WorkingResident.used, 1)
        np.testing.assert_allclose(out, np.full(A.shape[1], 42.0))


class RequiredModeStreamsBeforeFailingTests(unittest.TestCase):
    def _patched(self, mode):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.dict(os.environ, {"PHEASY_GPU_MODE": mode}))
        stack.enter_context(mock.patch.object(GB, "enabled", return_value=True))
        stack.enter_context(mock.patch.object(GB, "available", return_value=True))
        stack.enter_context(mock.patch.object(GB, "resident_device_ids", return_value=[0]))
        stack.enter_context(mock.patch.object(GB, "twolevel_host_factors", return_value=None))
        stack.enter_context(mock.patch.object(GB, "GpuTwoLevelOperator", _BrokenResident))
        return stack

    def test_required_mode_streams_instead_of_raising(self):
        A, exact = _twolevel(seed=8)
        with self._patched("required"), \
                mock.patch.object(GB, "col_norms_streaming", return_value=exact.copy()), \
                contextlib.redirect_stdout(io.StringIO()):
            got = A.col_norms()
        np.testing.assert_allclose(got, exact, rtol=1e-12)

    def test_required_mode_fails_closed_when_streaming_fails_too(self):
        A, _ = _twolevel(seed=9)
        with self._patched("required"), \
                mock.patch.object(GB, "col_norms_streaming",
                                  side_effect=RuntimeError("no room for NS")), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError) as cm:
                A.col_norms()
        msg = str(cm.exception)
        self.assertIn("fallback", msg)
        self.assertIn("streamed block pass", msg)
        self.assertIn("no room for NS", msg)

    def test_auto_mode_streams_too(self):
        A, exact = _twolevel(seed=10)
        with self._patched("auto"), \
                mock.patch.object(GB, "col_norms_streaming", return_value=exact.copy()), \
                contextlib.redirect_stdout(io.StringIO()):
            got = A.col_norms()
        np.testing.assert_allclose(got, exact, rtol=1e-12)

    def test_auto_mode_still_falls_back_to_the_cpu_definition(self):
        A, exact = _twolevel(seed=11)
        with self._patched("auto"), \
                mock.patch.object(GB, "col_norms_streaming",
                                  side_effect=RuntimeError("no room")), \
                contextlib.redirect_stdout(io.StringIO()):
            got = A.col_norms()
        np.testing.assert_allclose(got, exact, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
