#!/usr/bin/env python3
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp
import torch
import unittest
from core import optimizer as opt
from core import gpu_backend as gb

class TestMultiGPUHardware(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2,
                         "two CUDA devices required for hardware acceptance")
    def test_actual_two_card_cv_and_single_card_parity(self):
        A = opt.TwoLevelSM(sp.csr_matrix(np.random.default_rng(83).normal(size=(24, 5))),
                           sp.eye(5, format="csr"))
        y = np.asarray(A @ np.array([1., -.3, 0., .1, 0.]), dtype=np.float64)
        old = {key: os.environ.get(key) for key in
               ("PHEASY_GPU_DEVICES", "PHEASY_GPU_NGPU", "PHEASY_CV_TOL", "PHEASY_CV_MAX_ITER")}
        try:
            os.environ.update({"PHEASY_GPU_DEVICES": "0,1", "PHEASY_GPU_NGPU": "2",
                               "PHEASY_CV_TOL": "1e-8", "PHEASY_CV_MAX_ITER": "1000"})
            parallel = gb.GpuTwoLevelLassoCV([.001, .02, .1], 4, 1e-8, 1000, 42,
                                              group_size=3, standardize=True).fit(A, y)
            self.assertEqual(parallel.cv_devices_, ["cuda:0", "cuda:1"])
            self.assertEqual(set(parallel.cv_fold_devices_), {"cuda:0", "cuda:1"})
            self.assertNotEqual(parallel.cv_fold_devices_[0], parallel.cv_fold_devices_[1])
            self.assertEqual(parallel.regularized_solver_info_["backend"], "gpu_twolevel_resident")
            self.assertEqual(parallel.regularized_solver_info_["dtype"], "float64")
            self.assertEqual(parallel.regularized_solver_info_["device"], "cuda:0")
            self.assertTrue(all(x["converged"] for fold in parallel.cv_solver_info_ for x in fold))
            self.assertTrue(parallel.regularized_solver_info_["converged"])
            os.environ["PHEASY_GPU_NGPU"] = "1"
            serial = gb.GpuTwoLevelLassoCV([.001, .02, .1], 4, 1e-8, 1000, 42,
                                            group_size=3, standardize=True).fit(A, y)
            self.assertEqual(serial.cv_devices_, ["cuda:0"])
            self.assertEqual(serial.alpha_, parallel.alpha_)
            np.testing.assert_allclose(parallel.coef_, serial.coef_, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(parallel.mse_path_, serial.mse_path_, rtol=1e-8, atol=1e-10)
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2,
                         "two CUDA devices required for hardware acceptance")
    def test_resident_twolevel_rfe_when_device_is_not_the_resident_card(self):
        # [FIX AUTO-DEVICE] the resident operator lives on PHEASY_GPU_DEVICES[0]
        # while device() is cuda:0 here (in production: _auto_device's pick).
        # y used to be uploaded to device(), so the setup probe hit a
        # cross-device error and required mode aborted the fit.
        from unittest.mock import patch
        import warnings
        rng = np.random.default_rng(77)
        prime = sp.csr_matrix(rng.normal(size=(48, 12)))
        ns = sp.csr_matrix(rng.normal(size=(12, 8)) * np.geomspace(.1, 10, 8))
        y = (prime.toarray() @ ns.toarray() @ np.array([1., -.5, .25, 0, 0, .1, 0, 0])
             + rng.normal(scale=.01, size=48))
        env = {"PHEASY_GPU_DEVICES": "1", "PHEASY_GPU_NGPU": "1", "PHEASY_GPU_AUTO_DEVICE": "0",
               "PHEASY_GPU_MODE": "required", "PHEASY_GPU_RFE_RESIDENT": "1",
               "PHEASY_RFE_GRAM_GB": "0", "PHEASY_RFE_N_JOBS": "1", "PHEASY_CV_GROUP_SIZE": "4",
               "PHEASY_RFE_STEP": ".5", "PHEASY_RFE_MIN_FEATURES": "2"}
        with patch.dict(os.environ, env), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.environ.pop("PHEASY_GPU_DEVICE", None)
            self.assertEqual(str(gb.device()), "cuda:0")
            self.assertEqual(gb.resident_device_ids(), [1])
            model = opt.Optimizer("RFE", cv=3, rand_seed=42, use_gpu=True)
            model.fit(opt.TwoLevelSM(prime, ns), y)
        meta = model.results["backend_metadata"]
        self.assertEqual(model.results["execution_backend"], "gpu_rfe_resident_iterative")
        self.assertEqual(meta["resident_input_kind"], "twolevel")
        self.assertEqual(meta["cv_fold_scoring"], "gpu")

if __name__ == "__main__":
    unittest.main()
