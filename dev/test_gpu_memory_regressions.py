#!/usr/bin/env python3
"""CPU-only regressions for safe construction of large multi-GPU sparse ops.

Run: python dev/test_gpu_memory_regressions.py
CUDA allocation and kernels are replaced by a small scipy-backed fake; these
tests cover splitting, index limits, preflight memory checks and cleanup.
"""
import contextlib
import importlib.util
import os
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

    def t(self):
        # mirrors torch's zero-copy CSC view of a CSR tensor
        return _SparseTensor(self.matrix.T)


class _FakeTorch:
    float64 = np.float64
    float32 = np.float32
    int32 = np.int32

    def __init__(self):
        self.set_num_threads = Mock()
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

    def test_sparse_dispatch_threads_default_and_opt_out(self):
        matrix = sp.eye(4, format="csr")
        for setting, expected in (("1", True), ("0", False)):
            with self.subTest(setting=setting), patch.dict(
                    os.environ, {"PHEASY_GPU_SM_TORCH_THREADS": setting}):
                self.torch.set_num_threads.reset_mock()
                gpu = gb.GpuSparseMV(matrix, n_gpu=1, device_ids=[0])
                try:
                    if expected:
                        self.torch.set_num_threads.assert_called_once_with(1)
                    else:
                        self.torch.set_num_threads.assert_not_called()
                finally:
                    gpu.close()

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
        # The pre-flight is per SHARD, at the boundaries the operator actually
        # splits at -- nnz-balanced since the c3=5.0 skew fix (_row/col_shard_edges),
        # not the uniform linspace the original version of this test hard-coded.
        rs = gb._row_shard_edges(matrix, 2)
        cs = gb._col_shard_edges(matrix, 2)
        actual = gb._cuda_spmv_block_budget(
            matrix[int(rs[0]):int(rs[1])],
            matrix[:, int(cs[0]):int(cs[1])].T.tocsr(), 8)
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

    def test_row_shard_count_covers_the_int32_index_limit(self):
        # c3=4.5: nnz=4,816,356,048.  Two cards -> 2.41e9 per block, which int32
        # cannot represent; the split must grow to 4 shards (2 per card).
        nnz = 4816356048
        limit = gb._CUDA_INT32_LIMIT
        safe = gb._CUDA_SHARD_SAFE_NNZ
        self.assertLess(safe, limit)
        # The table in REPORT_gpu_sm_sharding_20261116.md section 3.2, every column.
        table = {
            4816356048: [3, 4, 3, 4, 5],     # c3=4.5 Mg2C60 joint (1040112 x 92079)
            3569731584: [2, 2, 3, 4, 5],     # c3=5.0 (454656 x 265662), GPU.md
            6_000_000_000: [4, 4, 6, 4, 5],  # hypothetical 6.0e9: 3 cards -> 6, not 3
        }
        for n, expected in table.items():
            got = [gb.row_shard_count(n, g) for g in range(1, 6)]
            self.assertEqual(got, expected, msg="nnz=%d" % n)
            for g, n_shards in zip(range(1, 6), got):
                self.assertEqual(n_shards % g, 0)
                # average shard under the safe target, hence under int32
                self.assertLessEqual(-(-n // n_shards), safe)
        self.assertEqual(gb.row_shard_count(nnz, 1), 3)
        self.assertEqual(gb.row_shard_count(nnz, 2), 4)
        # small matrices keep the one-block-per-card layout.
        self.assertEqual(gb.row_shard_count(10_000, 3), 3)
        with patch.dict(os.environ, {"PHEASY_GPU_SM_SHARDS": "7"}):
            self.assertEqual(gb.row_shard_count(10_000, 3), 7)

    def test_more_shards_than_cards_match_cpu(self):
        # Two cards, four shards: every block below the int32 limit, two blocks
        # per card, matvec concatenates in ROW order and the adjoint sums.
        rng = np.random.default_rng(23)
        matrix = sp.random(37, 19, density=0.35, random_state=rng, format="csr")
        with patch.dict(os.environ, {"PHEASY_GPU_SM_SHARDS": "4"}):
            gpu = gb.GpuSparseMV(matrix, n_gpu=2, device_ids=[0, 1])
            try:
                self.assertEqual(gpu._devs, [0, 1])
                self.assertEqual(gpu._n_shards, 4)
                self.assertEqual(gpu._block_devs, [0, 1, 0, 1])
                self.assertEqual(len(gpu._R), 4)
                x, u = rng.normal(size=19), rng.normal(size=37)
                np.testing.assert_allclose(gpu.matvec(x), matrix @ x,
                                           rtol=1e-13, atol=1e-13)
                np.testing.assert_allclose(gpu.rmatvec(u), matrix.T @ u,
                                           rtol=1e-13, atol=1e-13)
            finally:
                gpu.close()

    def test_low_mem_more_shards_than_cards_match_cpu(self):
        rng = np.random.default_rng(29)
        matrix = sp.random(41, 23, density=0.3, random_state=rng, format="csr")
        with patch.dict(os.environ, {"PHEASY_GPU_SM_SHARDS": "4",
                                     "PHEASY_GPU_SM_LOWMEM": "1"}):
            gpu = gb.GpuSparseMV(matrix, n_gpu=2, device_ids=[0, 1])
            try:
                self.assertEqual(gpu._T, [])
                x, u = rng.normal(size=23), rng.normal(size=41)
                np.testing.assert_allclose(gpu.matvec(x), matrix @ x,
                                           rtol=1e-13, atol=1e-13)
                np.testing.assert_allclose(gpu.rmatvec(u), matrix.T @ u,
                                           rtol=1e-13, atol=1e-13)
            finally:
                gpu.close()

    def test_real_gpu_check_script_logic(self):
        # dev/check_gpu_sm_shards.py is the real-GPU validation of the path above;
        # run its comparison under the fake torch so the script itself is tested.
        spec = importlib.util.spec_from_file_location(
            "check_gpu_sm_shards", Path(__file__).resolve().parent / "check_gpu_sm_shards.py")
        chk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chk)
        rng = np.random.default_rng(31)
        matrix = sp.random(53, 17, density=0.3, random_state=rng, format="csr")
        for lowmem in (False, True):
            with self.subTest(lowmem=lowmem):
                res = chk.check(matrix, [0, 1], shards=4, lowmem=lowmem,
                                backend=gb, log=lambda _s: None)
                self.assertEqual(res["n_shards"], 4)
                self.assertEqual(res["block_devs"], [0, 1, 0, 1])
                self.assertEqual(sum(res["row_nnz"]), matrix.nnz)
                self.assertLess(res["matvec_rel_err"], 1e-12)
                self.assertLess(res["rmatvec_rel_err"], 1e-12)
        # The script must not leak its env overrides into the caller.
        self.assertNotIn("PHEASY_GPU_SM_SHARDS", os.environ)
        # ... and it must flag a wrong result: drop one shard's rows.
        broken = gb.GpuSparseMV.matvec
        with patch.object(gb.GpuSparseMV, "matvec",
                          lambda self, x: broken(self, x) * np.r_[
                              np.zeros(int(self._rs[1])),
                              np.ones(matrix.shape[0] - int(self._rs[1]))]):
            res = chk.check(matrix, [0, 1], shards=4, backend=gb, log=lambda _s: None)
        self.assertGreater(res["matvec_rel_err"], 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
