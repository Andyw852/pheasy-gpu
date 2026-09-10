#!/usr/bin/env python3
"""CPU-safe dynamic resident CV contracts; emulation is not CUDA evidence."""
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import gpu_backend as gb
from core import optimizer as opt
try:
    import torch
except ImportError:
    torch = None


class SchedulerTests(unittest.TestCase):
    def test_fast_worker_claims_next_fold_without_waiting_for_slow_fold(self):
        slow_started = threading.Event()
        third_done = threading.Event()
        assignments = {}

        def solve(device, k, fold):
            assignments[k] = device
            if k == 0:
                slow_started.set()
                self.assertTrue(third_done.wait(5), "batch barrier blocked next fold")
            elif k == 1:
                self.assertTrue(slow_started.wait(5))
            elif k == 2:
                third_done.set()
            return fold * 2

        self.assertEqual(gb._dynamic_fold_map([0, 1], list(range(6)), solve), list(range(0, 12, 2)))
        self.assertNotEqual(assignments[0], assignments[1])
        self.assertEqual(assignments[1], assignments[2])
        self.assertEqual(len(assignments), 6)

    def test_scheduler_stays_in_process_and_never_uses_process_pool(self):
        pid = os.getpid()
        def solve(resource, k, fold):
            self.assertEqual(os.getpid(), pid)
            return fold
        with patch("concurrent.futures.ProcessPoolExecutor", side_effect=AssertionError("CUDA process pool forbidden")):
            self.assertEqual(gb._dynamic_fold_map([0, 1], [1, 2, 3], solve), [1, 2, 3])

    def test_single_worker_is_synchronous_and_failure_propagates(self):
        caller = threading.get_ident()
        seen = []
        def solve(resource, k, fold):
            self.assertEqual(threading.get_ident(), caller)
            seen.append(k)
            if k == 1:
                raise RuntimeError("fold failed")
            return fold
        with self.assertRaisesRegex(RuntimeError, "fold failed"):
            gb._dynamic_fold_map([0], [1, 2, 3], solve)
        self.assertEqual(seen, [0, 1])

    def test_parallel_failure_joins_workers(self):
        started = threading.Event()
        done = threading.Event()
        def solve(resource, k, fold):
            if k == 0:
                self.assertTrue(started.wait(5))
                raise RuntimeError("worker error")
            started.set()
            done.set()
        with self.assertRaisesRegex(RuntimeError, "worker error"):
            gb._dynamic_fold_map([0, 1], [0, 1], solve)
        self.assertTrue(done.is_set())


@unittest.skipIf(torch is None, "Torch not installed")
class DeviceTests(unittest.TestCase):
    def selected(self, env):
        with patch.dict(os.environ, env, clear=True), patch.object(torch.cuda, "device_count", return_value=4):
            return [str(d) for d in gb._resident_cv_devices()]

    def test_default_explicit_list_and_count(self):
        self.assertEqual(self.selected({"PHEASY_GPU_DEVICE": "2"}), ["cuda:2"])
        self.assertEqual(self.selected({"PHEASY_GPU_DEVICE": "ignored", "PHEASY_GPU_DEVICES": "1"}), ["cuda:1"])
        self.assertEqual(self.selected({"PHEASY_GPU_DEVICES": "3, 1,2"}), ["cuda:3", "cuda:1", "cuda:2"])
        self.assertEqual(self.selected({"PHEASY_GPU_DEVICES": "3,1,2", "PHEASY_GPU_NGPU": "2"}), ["cuda:3", "cuda:1"])
        self.assertEqual(self.selected({"PHEASY_GPU_DEVICE": "2", "PHEASY_GPU_NGPU": "3"}), ["cuda:2", "cuda:0", "cuda:1"])

    def test_invalid_explicit_request_fails_before_any_factor_allocation(self):
        A = opt.TwoLevelSM(sp.eye(4, format="csr"), sp.eye(4, format="csr"))
        # Validate the whole explicit list, not just the prefix NGPU will use.
        for env in [{"PHEASY_GPU_DEVICES": "0,4", "PHEASY_GPU_NGPU": "1"},
                    {"PHEASY_GPU_DEVICES": "1", "PHEASY_GPU_NGPU": "2"}]:
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True), patch.object(torch.cuda, "device_count", return_value=4), patch.object(gb, "available", return_value=True), patch.object(gb, "enabled", return_value=True), patch.object(gb, "GpuTwoLevelOperator") as upload:
                with self.assertRaises(ValueError):
                    gb.GpuTwoLevelLassoCV([.1], 2, 1e-8, 100, 42).fit(A, np.ones(4))
                upload.assert_not_called()

    def test_invalid_configuration_rejected(self):
        for value in ["0,0", "-1", "4", "1,", "x"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.selected({"PHEASY_GPU_DEVICES": value})
        for value in ["0", "-1", "5", "x"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.selected({"PHEASY_GPU_NGPU": value})


@unittest.skipIf(torch is None, "Torch not installed")
class EmulatedFitTests(unittest.TestCase):
    def test_parallel_matches_serial_and_refits_on_primary(self):
        rng = np.random.default_rng(83)
        A = opt.TwoLevelSM(sp.csr_matrix(rng.normal(size=(24, 5))), sp.eye(5, format="csr"))
        y = np.asarray(A @ np.array([1., -.3, 0., .1, 0.]))
        created = []
        refits = []
        factory = gb.GpuTwoLevelOperator
        fista = gb._fista_twolevel
        def operator(*args, **kwargs):
            result = factory(*args, **kwargs)
            created.append(result)
            return result
        def solve(op, *args, **kwargs):
            if len(args) < 7 and kwargs.get("rows") is None:
                refits.append(op)
            return fista(op, *args, **kwargs)
        with patch.object(gb, "available", return_value=True), patch.object(gb, "enabled", return_value=True), patch.object(gb, "device", return_value="cpu"), patch.dict(os.environ, {"PHEASY_GPU_DEVICES": "", "PHEASY_GPU_NGPU": "", "PHEASY_CV_TOL": "1e-8", "PHEASY_CV_MAX_ITER": "1000"}):
            serial = gb.GpuTwoLevelLassoCV([.001, .02, .1], 4, 1e-8, 1000, 42, group_size=3, standardize=True).fit(A, y)
            with patch.object(gb, "_resident_cv_devices", return_value=["cpu", "cpu"]), patch.object(gb, "GpuTwoLevelOperator", side_effect=operator), patch.object(gb, "_fista_twolevel", side_effect=solve):
                parallel = gb.GpuTwoLevelLassoCV([.001, .02, .1], 4, 1e-8, 1000, 42, group_size=3, standardize=True).fit(A, y)
        np.testing.assert_allclose(parallel.mse_path_, serial.mse_path_, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(parallel.coef_, serial.coef_, rtol=1e-12, atol=1e-14)
        self.assertEqual(parallel.alpha_, serial.alpha_)
        self.assertEqual(parallel.cv_solver_info_, serial.cv_solver_info_)
        self.assertEqual(len(created), 2)
        self.assertTrue(refits)
        self.assertTrue(all(op is created[0] for op in refits))
        self.assertTrue(all(op.prime is None for op in created))

    def test_secondary_upload_failure_closes_primary(self):
        A = opt.TwoLevelSM(sp.eye(4, format="csr"), sp.eye(4, format="csr"))
        created = []
        factory = gb.GpuTwoLevelOperator
        def operator(*args, **kwargs):
            if created:
                raise MemoryError("secondary preflight")
            result = factory(*args, **kwargs)
            created.append(result)
            return result
        with patch.object(gb, "available", return_value=True), patch.object(gb, "enabled", return_value=True), patch.object(gb, "_resident_cv_devices", return_value=["cpu", "cpu"]), patch.object(gb, "GpuTwoLevelOperator", side_effect=operator) as upload, patch.object(gb, "_dynamic_fold_map") as scheduler:
            with self.assertRaisesRegex(MemoryError, "secondary preflight"):
                gb.GpuTwoLevelLassoCV([.1], 2, 1e-8, 100, 42).fit(A, np.ones(4))
            self.assertEqual(upload.call_count, 2)  # no retry on another card
            scheduler.assert_not_called()  # no partial CV after setup failure
        self.assertIsNone(created[0].prime)


if __name__ == "__main__":
    unittest.main()
