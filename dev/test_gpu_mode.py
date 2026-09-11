import os
import unittest
from unittest.mock import patch
from core import gpu_backend as gb

class GpuModeTest(unittest.TestCase):
    def test_cpu_mode_disables_gpu(self):
        with patch.dict(os.environ,{"PHEASY_GPU_MODE":"cpu"},clear=False):
            self.assertEqual(gb.gpu_mode_from_env(),"cpu")
            self.assertFalse(gb.enabled())
    def test_auto_mode_is_not_required(self):
        with patch.dict(os.environ,{"PHEASY_GPU_MODE":"auto"},clear=False):
            self.assertEqual(gb.gpu_mode_from_env(),"auto")
            self.assertFalse(gb.gpu_mode_required())
    def test_required_mode_fails_closed_without_cuda(self):
        with patch.dict(os.environ,{"PHEASY_GPU_MODE":"required"},clear=False), patch.object(gb,"available",return_value=False):
            with self.assertRaisesRegex(RuntimeError,"required"):
                gb.enabled()

class RfeFailureTest(unittest.TestCase):
    def test_twolevel_resident_setup_failure_does_not_run_cpu_lsmr(self):
        import numpy as np
        from scipy import sparse as sp
        from core.optimizer import Optimizer, TwoLevelSM

        A = TwoLevelSM(sp.eye(6, format="csr"), sp.eye(6, format="csr"))
        y = np.arange(6.)

        with patch.dict(os.environ, {"PHEASY_GPU_MODE": "required", "PHEASY_RFE_N_JOBS": "1"}, clear=True), \
                patch.object(gb, "enabled", return_value=True), \
                patch.object(gb, "available", return_value=True), \
                patch.object(gb, "GpuTwoLevelOperator", side_effect=RuntimeError("injected setup failure")):
            with self.assertRaisesRegex(RuntimeError, "fallback disabled"):
                Optimizer("RFE", use_gpu=True).fit(A, y)


class RidgeFailureTest(unittest.TestCase):
    def test_ridge_cv_cuda_oom_does_not_run_cpu_svd(self):
        import contextlib
        import numpy as np
        import torch
        from core.optimizer import Optimizer

        with patch.dict(os.environ, {"PHEASY_GPU_MODE": "auto"}, clear=True), \
                patch.object(gb, "available", return_value=True), \
                patch.object(gb, "available_memory_bytes", return_value=10**10), \
                patch.object(gb, "_multi_gpu_devices", return_value=[0]), \
                patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()), \
                patch.object(torch.cuda, "empty_cache"), \
                patch.object(torch, "as_tensor", side_effect=torch.cuda.OutOfMemoryError("injected OOM")), \
                patch.object(np.linalg, "svd", side_effect=AssertionError("CPU SVD was called")):
            with self.assertRaisesRegex(RuntimeError, "fallback disabled"):
                Optimizer("RIDGE", alpha=[0.1, 1.0], cv=2, use_gpu=True).fit(
                    np.eye(6), np.arange(6.))


if __name__=="__main__": unittest.main()
