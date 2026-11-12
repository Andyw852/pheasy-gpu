#!/usr/bin/env python3
"""[FIX COLNORM-FAILFAST] a failed GPU column-norm pass under required mode.

Mg4C60 c3=3.5: the resident operator could not be built, the column norms fell
back to the bounded CPU loop (~55 h estimated) and the job kept running on the
host while holding its GPUs.  Under PHEASY_GPU_MODE=required that must now be a
fast, explicit failure whenever the CPU loop is larger than
PHEASY_COLNORM_CPU_MAX_WORK (0 = always); small operators, auto mode and
PHEASY_COLNORM_CPU_FALLBACK=1 keep the CPU loop.  CPU-only: the GPU backend is
patched.

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


def _op():
    rng = np.random.default_rng(0)
    prime = sp.random(60, 15, density=0.4, format="csr", random_state=1)
    ns = sp.csr_matrix(rng.normal(size=(15, 6)))
    A = O.TwoLevelSM(prime, ns)
    exact = np.linalg.norm(prime.toarray() @ ns.toarray(), axis=0)
    return A, exact


class _BrokenOp:
    def col_norms(self):
        raise RuntimeError("CUDA out of memory (simulated)")


class ColNormFailFastTests(unittest.TestCase):
    def _patched(self, mode, extra=None, build_error=True):
        env = {"PHEASY_GPU_MODE": mode}
        env.update(extra or {})
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

    def test_required_mode_fails_fast_when_the_operator_cannot_be_built(self):
        A, _ = _op()
        with self._patched("required", {"PHEASY_COLNORM_CPU_MAX_WORK": "0"}):
            with self.assertRaises(RuntimeError) as cm:
                A.col_norms()
        msg = str(cm.exception)
        self.assertIn("PHEASY_GPU_MODE=required", msg)
        self.assertIn("PHEASY_COLNORM_CPU_FALLBACK=1", msg)
        self.assertIn("buffers are not shared", msg)
        self.assertIn("PHEASY_COLNORM_CPU_MAX_WORK", msg)

    def test_required_mode_fails_fast_when_the_gpu_pass_itself_fails(self):
        A, _ = _op()
        with self._patched("required", {"PHEASY_COLNORM_CPU_MAX_WORK": "0"},
                           build_error=False):
            with self.assertRaises(RuntimeError) as cm:
                A.col_norms(gpu_op=_BrokenOp())
        self.assertIn("out of memory", str(cm.exception))

    def test_explicit_opt_in_keeps_the_cpu_loop(self):
        A, exact = _op()
        out = io.StringIO()
        with self._patched("required", {"PHEASY_COLNORM_CPU_FALLBACK": "1",
                                        "PHEASY_COLNORM_CPU_MAX_WORK": "0"}), \
                contextlib.redirect_stdout(out):
            cn = A.col_norms()
        np.testing.assert_allclose(cn, exact, rtol=1e-12)
        self.assertIn("bounded CPU column-norm loop", out.getvalue())

    def test_auto_mode_keeps_the_cpu_loop(self):
        A, exact = _op()
        with self._patched("auto"), contextlib.redirect_stdout(io.StringIO()):
            cn = A.col_norms()
        np.testing.assert_allclose(cn, exact, rtol=1e-12)

    def test_small_operator_keeps_the_cpu_loop_under_the_default_limit(self):
        A, exact = _op()
        with self._patched("required"), contextlib.redirect_stdout(io.StringIO()):
            cn = A.col_norms()
        np.testing.assert_allclose(cn, exact, rtol=1e-12)

    def test_the_limit_is_compared_with_the_real_loop_size(self):
        A, _ = _op()
        prime = (A.SM_prime != 0).astype(np.float64)
        ns = (A.NS != 0).astype(np.float64)
        work = float((prime @ ns).sum())
        self.assertEqual(O._colnorm_cpu_work(A), work)
        self.assertEqual(O._colnorm_cpu_work(A, GB), work)
        with self._patched("required", {"PHEASY_COLNORM_CPU_MAX_WORK": repr(work)}), \
                contextlib.redirect_stdout(io.StringIO()):
            A.col_norms()                                   # == limit: allowed
        A2, _ = _op()
        with self._patched("required", {"PHEASY_COLNORM_CPU_MAX_WORK": repr(work - 1)}):
            with self.assertRaises(RuntimeError):
                A2.col_norms()

    def test_no_gpu_at_all_is_not_this_failure(self):
        # enabled()/available() false (or raising) is handled by the solver's own
        # error surface; the column-norm pass just uses the CPU definition.
        A, exact = _op()
        with mock.patch.dict(os.environ, {"PHEASY_GPU_MODE": "auto"}), \
                mock.patch.object(GB, "available", return_value=False):
            np.testing.assert_allclose(A.col_norms(), exact, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
