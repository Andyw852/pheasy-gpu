#!/usr/bin/env python3
"""[FIX COLNORM-NO-CPU-FALLBACK] a failed GPU column-norm pass is a hard error.

History: on Mg4C60 c3=3.5 (26 GB factors) the resident operator could not be
built, the column norms fell back to the bounded CPU loop (~55 h estimated) and
the job kept running on the host while holding its GPUs.  0008 then made that a
fast failure only when the loop exceeded PHEASY_COLNORM_CPU_MAX_WORK, and added
an opt-in (PHEASY_COLNORM_CPU_FALLBACK=1) for the rest.  Measured again on
Mg2C60 c3=4.0: the loop still ran >6 h before being killed by hand.

Both escape hatches are now gone.  Under PHEASY_GPU_MODE=required a GPU
column-norm failure always raises: either the resident operator fits and the
norms are exact and fast, or the fit stops with the size of the problem.  Auto
mode and CPU-only runs are unaffected -- they never select the GPU operator.

    python -m unittest dev.test_colnorm_failfast
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


def _op(big=False):
    rng = np.random.default_rng(0)
    if big:
        prime = sp.random(400, 3000, density=0.02, format="csr", random_state=1)
    else:
        prime = sp.random(60, 15, density=0.4, format="csr", random_state=1)
    ns = sp.csr_matrix(rng.normal(size=(prime.shape[1], 6)))
    A = O.TwoLevelSM(prime, ns)
    exact = np.linalg.norm(prime.toarray() @ ns.toarray(), axis=0)
    return A, exact


class _BrokenOp:
    def col_norms(self):
        raise RuntimeError("CUDA out of memory (simulated)")


class ColNormNoCpuFallbackTests(unittest.TestCase):
    def _patched(self, mode, build_error=True):
        env = {"PHEASY_GPU_MODE": mode}
        boom = mock.Mock(side_effect=RuntimeError(
            "scipy materialises csr.T (buffers are not shared) (simulated)"))
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.dict(os.environ, env))
        stack.enter_context(mock.patch.object(GB, "enabled", return_value=True))
        stack.enter_context(mock.patch.object(GB, "available", return_value=True))
        stack.enter_context(mock.patch.object(GB, "resident_device_ids", return_value=[0]))
        stack.enter_context(mock.patch.object(GB, "twolevel_host_factors", return_value=None))
        if build_error:
            stack.enter_context(mock.patch.object(GB, "GpuTwoLevelOperator", boom))
        return stack

    def test_required_mode_raises_when_the_operator_cannot_be_built(self):
        A, _ = _op()
        with self._patched("required"):
            with self.assertRaises(RuntimeError) as cm:
                A.col_norms()
        msg = str(cm.exception)
        self.assertIn("CPU column-norm fallback", msg)
        self.assertIn("has been removed", msg)
        self.assertIn("buffers are not shared", msg)
        self.assertIn("Make the resident two-level operator fit", msg)

    def test_required_mode_raises_when_the_gpu_pass_itself_fails(self):
        A, _ = _op()
        with self._patched("required", build_error=False):
            with self.assertRaises(RuntimeError) as cm:
                A.col_norms(gpu_op=_BrokenOp())
        self.assertIn("out of memory", str(cm.exception))

    def test_required_mode_raises_even_for_a_tiny_operator(self):
        """There is no size exemption any more: small or large, no CPU loop."""
        A, _ = _op()
        with self._patched("required"):
            with self.assertRaises(RuntimeError):
                A.col_norms()

    def test_required_mode_raises_for_a_large_operator(self):
        A, _ = _op(big=True)
        with self._patched("required"):
            with self.assertRaises(RuntimeError) as cm:
                A.col_norms()
        self.assertIn("sparse products", str(cm.exception))

    def test_the_cpu_loop_size_is_still_reported(self):
        A, _ = _op(big=True)
        prime = (A.SM_prime != 0).astype(np.float64)
        ns = (A.NS != 0).astype(np.float64)
        work = float((prime @ ns).sum())
        self.assertEqual(O._colnorm_cpu_work(A), work)
        self.assertEqual(O._colnorm_cpu_work(A, GB), work)

    def test_auto_mode_keeps_the_cpu_loop(self):
        A, exact = _op()
        with self._patched("auto"), contextlib.redirect_stdout(io.StringIO()):
            cn = A.col_norms()
        np.testing.assert_allclose(cn, exact, rtol=1e-12)

    def test_no_gpu_at_all_is_not_this_failure(self):
        # enabled()/available() false (or raising) is handled by the solver's own
        # error surface; the column-norm pass just uses the CPU definition.
        A, exact = _op()
        with mock.patch.dict(os.environ, {"PHEASY_GPU_MODE": "required"}), \
                mock.patch.object(GB, "available", return_value=False):
            np.testing.assert_allclose(A.col_norms(), exact, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
