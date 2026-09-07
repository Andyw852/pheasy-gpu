#!/usr/bin/env python3
"""CPU-only regressions for safe construction of large multi-GPU sparse ops.

Run: python dev/test_gpu_memory_regressions.py
CUDA allocation and kernels are replaced by a small scipy-backed fake; these
tests cover splitting, index limits, preflight memory checks and cleanup.
"""
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy import sparse as sp


_spec = importlib.util.spec_from_file_location(
    "gpu_backend_under_test", Path(__file__).resolve().parents[1] / "core/gpu_backend.py")
gb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gb)


class _Result:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _SparseTensor:
    def __init__(self, matrix):
        self.matrix = matrix

    def __matmul__(self, vector):
        return _Result(self.matrix @ vector)


class _FakeTorch:
    float64 = np.float64
    float32 = np.float32
    int32 = np.int32

    def __init__(self):
        self.cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 6,
            device=lambda _device: contextlib.nullcontext(),
            empty_cache=Mock(),
        )
        self.as_tensor = Mock(side_effect=lambda a, dtype, device: np.asarray(a, dtype=dtype))

    @staticmethod
    def device(name):
        return name

    @staticmethod
    def sparse_csr_tensor(crow, ccol, values, size, dtype, device):
        return _SparseTensor(sp.csr_matrix((values, ccol, crow), shape=size))


class GpuMemoryRegressions(unittest.TestCase):
    def setUp(self):
        self.torch = _FakeTorch()
        self.torch_patch = patch.object(gb, "_torch", return_value=self.torch)
        self.free_patch = patch.object(gb, "_device_free_bytes", return_value=8 << 30)
        self.torch_patch.start()
        self.free_patch.start()
        self.addCleanup(self.torch_patch.stop)
        self.addCleanup(self.free_patch.stop)

    def test_oversized_nnz_is_rejected_before_tensor_allocation(self):
        matrix = SimpleNamespace(shape=(2, 2), nnz=2**31)
        instance = gb.GpuSparseMV.__new__(gb.GpuSparseMV)
        instance._t = self.torch
        with self.assertRaisesRegex(ValueError, "int32 index range"):
            instance._csr_to_torch(matrix, "cuda:0")
        self.torch.as_tensor.assert_not_called()

    def test_oversized_column_dimension_is_rejected(self):
        # Large sparse shapes do not require a correspondingly large array.
        matrix = sp.csr_matrix(([1.0], ([0], [2**31])), shape=(1, 2**31 + 1))
        with self.assertRaisesRegex(ValueError, "int32 index range"):
            gb._check_cuda_csr_indices(matrix)

    def test_budget_uses_actual_nonzeros_and_cuda_index_dtype(self):
        # Both row and transpose blocks can contain nearly all the nonzeros.
        row = SimpleNamespace(shape=(1000, 2000), nnz=300_000_000)
        transposed = SimpleNamespace(shape=(1000, 2000), nnz=300_000_000)
        budget = gb._cuda_spmv_block_budget(row, transposed, 8)
        self.assertGreater(budget, 7_200_000_000)
        self.assertEqual(gb._cuda_csr_bytes(row, 8), 3_600_004_004)

    def test_skewed_blocks_fail_before_upload(self):
        matrix = sp.csr_matrix(([1.0] * 9, ([0] * 9, list(range(9)))), shape=(12, 24))
        actual = gb._cuda_spmv_block_budget(matrix[:6], matrix[:, :12].T.tocsr(), 8)
        with patch.object(gb, "_device_free_bytes", return_value=actual - 1):
            with self.assertRaisesRegex(MemoryError, "actual CSR row/transpose blocks"):
                gb.GpuSparseMV(matrix, n_gpu=2, device_ids=[0, 1])
        self.torch.as_tensor.assert_not_called()

    def test_later_card_failure_releases_earlier_uploads(self):
        instance = gb.GpuSparseMV.__new__(gb.GpuSparseMV)
        matrix = sp.eye(12, format="csr")
        with patch.object(gb, "_device_free_bytes", side_effect=lambda d: (8 << 30) if d == 0 else 0):
            with self.assertRaises(MemoryError):
                instance.__init__(matrix, n_gpu=2, device_ids=[0, 1])
        self.assertGreater(self.torch.as_tensor.call_count, 0)
        self.assertEqual(instance._R, [])
        self.assertEqual(instance._T, [])
        self.torch.cuda.empty_cache.assert_called()

    def test_multiple_gpu_blocks_match_cpu_and_do_not_repeat_devices(self):
        rng = np.random.default_rng(19)
        matrix = sp.random(31, 17, density=0.3, random_state=rng, format="csr")
        gpu = gb.GpuSparseMV(matrix, n_gpu=3, device_ids=[0, 0, 2, 4])
        self.assertEqual(gpu._devs, [0, 2, 4])
        x, u = rng.normal(size=17), rng.normal(size=31)
        np.testing.assert_allclose(gpu.matvec(x), matrix @ x, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(gpu.rmatvec(u), matrix.T @ u, rtol=1e-13, atol=1e-13)
        gpu.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
