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
