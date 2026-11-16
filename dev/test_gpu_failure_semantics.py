import os
import unittest
from unittest.mock import patch
from core.optimizer import TwoLevelSM

class GpuFailureSemanticsTest(unittest.TestCase):
    def test_strict_mode_propagates_matvec_failure(self):
        import numpy as np
        from scipy import sparse as sp
        A=TwoLevelSM(sp.eye(3,format="csr"),sp.eye(3,format="csr"))
        class Broken:
            def matvec(self, x): raise RuntimeError("cuda lost")
            def rmatvec(self, x): raise RuntimeError("cuda lost")
            def close(self): pass
        A._gpu_mv=Broken()
        with patch.dict(os.environ,{"PHEASY_GPU_FALLBACK":"0"}):
            with self.assertRaisesRegex(RuntimeError,"cuda lost"):
                A @ np.ones(3)

    def test_explicit_fallback_keeps_cpu_compatibility(self):
        import numpy as np
        from scipy import sparse as sp
        A=TwoLevelSM(sp.eye(3,format="csr"),sp.eye(3,format="csr"))
        class Broken:
            def matvec(self, x): raise RuntimeError("cuda lost")
            def close(self): pass
        A._gpu_mv=Broken()
        with patch.dict(os.environ,{"PHEASY_GPU_FALLBACK":"1"}):
            np.testing.assert_allclose(A @ np.ones(3), np.ones(3))

class ConstructionFailurePolicyTest(unittest.TestCase):
    """GpuSparseMV failing inside TwoLevelSM.__init__ (int32 refusal, busy cards).

    The c3=4.5 fit ran with PHEASY_GPU_MODE=required AND PHEASY_GPU_FALLBACK=1;
    construction only read the fallback knob and silently put the RFE on one CPU
    core.  Required mode must abort here exactly as in matvec/rmatvec.
    """

    def _build(self, env):
        from scipy import sparse as sp
        import core.gpu_backend as gb

        def refuse(*_a, **_k):
            raise ValueError("CUDA CSR block nnz=2408182532 exceeds the int32 index range")

        base = {"PHEASY_GPU_SM": "1", "PHEASY_GPU_LASSO_RESIDENT": "0"}
        base.update(env)
        with patch.dict(os.environ, base, clear=True), \
                patch.object(gb, "GpuSparseMV", side_effect=refuse):
            return TwoLevelSM(sp.eye(3, format="csr"), sp.eye(3, format="csr"))

    def test_required_mode_beats_fallback_knob(self):
        for env in ({"PHEASY_GPU_MODE": "required", "PHEASY_GPU_FALLBACK": "1"},
                    {"PHEASY_USE_GPU": "1", "PHEASY_GPU_FALLBACK": "1"}):
            with self.subTest(env=env):
                with self.assertRaisesRegex(RuntimeError, "int32 index range"):
                    self._build(env)

    def test_fallback_disabled_aborts(self):
        with self.assertRaisesRegex(RuntimeError, "fallback disabled"):
            self._build({"PHEASY_GPU_FALLBACK": "0"})

    def test_explicit_fallback_outside_required_mode_still_uses_cpu(self):
        import numpy as np
        A = self._build({"PHEASY_GPU_MODE": "auto", "PHEASY_GPU_FALLBACK": "1"})
        self.assertIsNone(A._gpu_mv)
        np.testing.assert_allclose(A @ np.ones(3), np.ones(3))


class DefaultFailurePolicyTest(unittest.TestCase):
    def test_default_cuda_errors_propagate_for_both_operator_directions(self):
        import numpy as np
        from scipy import sparse as sp
        for transpose in (False, True):
            with self.subTest(transpose=transpose), patch.dict(os.environ, {}, clear=True):
                A = TwoLevelSM(sp.eye(3, format="csr"), sp.eye(3, format="csr"))
                class Broken:
                    def matvec(self, x): raise RuntimeError("cuda lost")
                    def rmatvec(self, x): raise RuntimeError("cuda lost")
                    def close(self): pass
                A._gpu_mv = Broken()
                with self.assertRaisesRegex(RuntimeError, "cuda lost"):
                    (A.T if transpose else A) @ np.ones(3)

if __name__=="__main__": unittest.main()
