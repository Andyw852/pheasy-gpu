#!/usr/bin/env python3
"""Pre-flight estimate tests for the resident two-level footprint.

The c3=5.0 sharded resident estimate used to be 1.43x the memory the GPU can
actually allocate, because it charged every shard for int64 indices:

* the index width was read off the GLOBAL nnz (3.57e9 > 2**31 -> 8 bytes), but
  each shard uploads ~892M nonzeros and upload() casts those to int32;
* the estimate multiplied ONE global bytes-per-nonzero by the FULL nnz and then
  divided by n_shards, so the SM_prime pair and the NS replica were modelled with
  the same width;
* GpuTwoLevelOperator.upload() itself used the same global flag, so a sharded
  float32 fit shipped int64 index tensors (+4 B/nnz = +3.57 GB per card).

The tests pin: the per-BLOCK width rule, that sharding lowers the estimate, that
the estimate stays >= the operator own per-block budget (it is a PRE-flight, so
it must not be optimistic), and that uploaded blocks use int32 even when the
matrix as a whole would need int64.
"""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import gpu_backend as GB
from core import optimizer as O


_REAL_NNZ = sp.csr_matrix.nnz


class _LyingNnz(sp.csr_matrix):
    """A real CSR that REPORTS the production c3=5.0 nnz after construction.

    Only .nnz is faked -- shape/dtype/indptr/indices stay the small real arrays --
    so the int32/int64 decision can be exercised without a 43 GB fixture.  The
    fake is installed after __init__ because scipy own check_format/prune reads
    .nnz and would reject a matrix claiming more entries than it holds.
    """

    @property
    def nnz(self):
        return getattr(self, "_fake_nnz", _REAL_NNZ.__get__(self))


def _lying_nnz(matrix, nnz):
    """Install a fake .nnz on a real sparse matrix (shape/arrays stay real)."""
    assert isinstance(matrix, _LyingNnz), type(matrix)
    matrix._fake_nnz = int(nnz)
    return matrix


def _lying_factor(nnz, shape, dtype=np.float32):
    """A real 2x2 CSR reporting the c3=5.0 nnz AND shape (estimate tests only).

    Slicing such a matrix would be inconsistent, so only the footprint arithmetic
    may touch it.
    """
    m = _LyingNnz(sp.eye(2, dtype=dtype))
    m._fake_nnz = int(nnz)
    m._shape = tuple(shape)
    return m


class _Duck(object):
    """Duck-typed sparse: _csr_device_bytes only reads .nnz and .shape."""

    def __init__(self, nnz, shape):
        self.nnz = nnz
        self.shape = shape


class CsrDeviceBytesTest(unittest.TestCase):
    def test_float32_shard_indices_are_int32(self):
        m = _Duck(3569731584, (454656, 265662))
        full = GB._csr_device_bytes(m, 4, 1)
        one = GB._csr_device_bytes(m, 4, 4)
        self.assertEqual(full, (4 + 8) * m.nnz + 8 * (sum(m.shape) + 2))
        self.assertEqual(one, (4 + 4) * m.nnz + 8 * (sum(m.shape) + 2))
        self.assertEqual(GB._csr_device_bytes(m, 4, 2), one)
        # the old global-width rule, for contrast
        old = (4 + 8) * m.nnz + 8 * (sum(m.shape) + 2)
        self.assertLess(one, old * 0.7)

    def test_index_width_boundary_is_per_block(self):
        limit = 2 ** 31
        tail = 8 * (1 + 1 + 2)
        self.assertEqual(GB._csr_device_bytes(_Duck(limit - 1, (1, 1)), 4, 1),
                         8 * (limit - 1) + tail)
        self.assertEqual(GB._csr_device_bytes(_Duck(limit, (1, 1)), 4, 1),
                         12 * limit + tail)
        # the same total, but split in two: both blocks fit int32
        self.assertEqual(GB._csr_device_bytes(_Duck(limit, (1, 1)), 4, 2),
                         8 * limit + tail)

    def test_float64_values_are_charged_eight_bytes(self):
        m = _Duck(1000, (10, 10))
        self.assertEqual(GB._csr_device_bytes(m, 8, 1), 12 * 1000 + 8 * 22)


class ResidentEstimateTest(unittest.TestCase):
    def _estimate(self, A, n_shards):
        with mock.patch.object(GB, "_device_free_bytes", return_value=1 << 40), \
             mock.patch.object(GB, "_device_available_bytes", return_value=1 << 40):
            return GB.resident_twolevel_estimate(A, device_id=0, n_shards=n_shards)

    @staticmethod
    def _small():
        rng = np.random.default_rng(5)
        prime = sp.csr_matrix(rng.normal(size=(60, 20)) * (rng.random((60, 20)) < 0.35))
        prime = prime.astype(np.float32)
        ns = sp.csr_matrix(np.asarray(rng.normal(size=(20, 8)), dtype=np.float32))
        return SimpleNamespace(SM_prime=prime, NS=ns, shape=(60, 8))

    def test_sharding_lowers_the_estimate(self):
        A = self._small()
        peak1 = self._estimate(A, 1)[0]
        peak4 = self._estimate(A, 4)[0]
        self.assertLess(peak4, peak1)
        self.assertEqual(self._estimate(A, 4)[3], 0.8)

    def test_estimate_is_not_optimistic_about_the_real_blocks(self):
        """peak must cover every shard block pair the operator will upload."""
        A = self._small()
        peak = self._estimate(A, 4)[0]
        rs = GB._row_shard_edges(A.SM_prime, 4)
        cs = GB._col_shard_edges(A.SM_prime, 4)
        worst = 0
        for i in range(4):
            R = A.SM_prime[int(rs[i]):int(rs[i + 1])]
            T = A.SM_prime[:, int(cs[i]):int(cs[i + 1])].T.tocsr()
            worst = max(worst, GB._cuda_spmv_block_budget(R, T, 4))
        self.assertGreaterEqual(peak, worst)

    def test_c50_shaped_estimate_matches_the_per_card_allocation(self):
        """Float32 3.57e9-nnz factors: ~17 GB/card at 4 shards, not ~22.5 GB."""
        prime = _lying_factor(3569731584, (454656, 265662))
        ns = _lying_factor(86793504, (265662, 224443))
        A = SimpleNamespace(SM_prime=prime, NS=ns, shape=(454656, 224443))
        peak1 = self._estimate(A, 1)[0]
        peak4 = self._estimate(A, 4)[0]
        pair4 = 2 * GB._csr_device_bytes(prime, 4, 4) / 4
        self.assertAlmostEqual(peak4, int(np.ceil(pair4)) + int(np.ceil(pair4)) // 10
                               + 2 * GB._csr_device_bytes(ns, 4, 1)
                               + GB._norm_workspace_bytes()
                               + 8 * 32 * sum(A.shape), delta=1024)
        self.assertLess(peak4, 18 * 10 ** 9)
        self.assertGreater(peak4, 15 * 10 ** 9)
        self.assertGreater(peak1, 4 * peak4)


class _Device(object):
    def __init__(self, index):
        self.index = index

    @property
    def type(self):
        return "cuda"


class _Ctx(object):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeCuda(object):
    @staticmethod
    def device(d):
        return _Ctx()

    @staticmethod
    def empty_cache():
        pass

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 2


class _FakeTorch(object):
    float64 = np.float64
    float32 = np.float32
    int32 = np.int32
    int64 = np.int64
    device = _Device

    def __init__(self):
        self.uploaded = []
        self.as_tensor_inputs = []
        self.cuda = _FakeCuda()
        self.set_num_threads = lambda n: None

    def as_tensor(self, x, dtype=None, device=None):
        self.as_tensor_inputs.append(x)
        return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)

    def sparse_csr_tensor(self, crow, ccol, cval, size=None, dtype=None, device=None):
        self.uploaded.append(dict(shape=tuple(size), indptr=np.asarray(crow),
                                  indices=np.asarray(ccol), data=np.asarray(cval)))
        return self.uploaded[-1]

    @staticmethod
    def ones(n, dtype=None, device=None):
        return np.ones(n, dtype=dtype if dtype is not None else np.float64)

    @staticmethod
    def ones_like(x):
        return np.ones_like(x)

    @staticmethod
    def isfinite(x):
        return np.isfinite(x)


class UploadIndexWidthTest(unittest.TestCase):
    def test_sharded_upload_uses_int32_when_only_the_whole_matrix_needs_int64(self):
        prime = _lying_nnz(_LyingNnz(sp.eye(4, dtype=np.float32)), 3569731584)
        ns = sp.csr_matrix(np.asarray(np.eye(4), dtype=np.float32))
        A = SimpleNamespace(SM_prime=prime, NS=ns, shape=(4, 4))
        factors = GB._canonical_twolevel_host(A)
        fake = _FakeTorch()
        with mock.patch.object(GB, "_torch", lambda: fake), \
             mock.patch.object(GB, "enabled", lambda: True), \
             mock.patch.object(GB, "available", lambda: True), \
             mock.patch.object(GB, "_device_free_bytes", lambda d: 1 << 40), \
             mock.patch.object(GB, "_device_available_bytes", lambda d: 1 << 40):
            op = GB.GpuTwoLevelOperator(A, device_ids=[0, 1], host_factors=factors)
        try:
            self.assertEqual(op._index_dtype, np.int64,
                             "the GLOBAL flag must still be the pessimistic one")
            self.assertTrue(fake.uploaded)
            for block in fake.uploaded:
                self.assertEqual(block["indices"].dtype, np.int32,
                                 "blocks smaller than 2**31 must ship int32 indices")
            # NS and NS.T are (4, 4); each shard contributes one row block and
            # one adjoint row block, both (2, 4) for this square factor.
            split = [b for b in fake.uploaded if b["shape"] == (2, 4)]
            full = [b for b in fake.uploaded if b["shape"] == (4, 4)]
            self.assertEqual(len(split), 4, "two shards -> 2 row + 2 adjoint blocks")
            self.assertEqual(len(full), 2, "NS and its transpose")
            self.assertEqual(sum(int(b["indptr"][-1]) for b in split), 8,
                             "row+adjoint blocks must tile all real nonzeros")
        finally:
            op.close()

class SequentialBlockBuildTest(unittest.TestCase):
    """One block at a time: at c3=5.0 the two host blocks are 10.7 GB each.

    Building both before uploading them doubled the host transient (and kept two
    numpy buffers alive through the cast) for no benefit -- the count-based
    budget already answers the fit question.  The order is only observable in
    the calls, so record build/upload and collapse the consecutive duplicates
    that the redundant index check inside _csr_to_torch produces.
    """

    @staticmethod
    def _collapse(events):
        out = []
        for event in events:
            if not out or out[-1] != event:
                out.append(event)
        return out

    def test_count_budget_equals_block_budget(self):
        row = sp.csr_matrix((np.ones(7, np.float32),
                             ([0, 0, 1, 2, 2, 2, 4], [0, 3, 1, 0, 2, 3, 4])),
                            shape=(5, 6))
        trans = row.T.tocsr()
        for itemsize in (4, 8):
            with self.subTest(itemsize=itemsize):
                self.assertEqual(
                    GB._cuda_spmv_block_budget(row, trans, itemsize),
                    GB._cuda_spmv_block_budget_counts(
                        row.shape, row.nnz, trans.shape, trans.nnz, itemsize),
                    "count-based pre-flight must not drift from the block budget")

    def test_resident_operator_builds_one_block_at_a_time(self):
        prime = _lying_nnz(_LyingNnz(sp.eye(4, dtype=np.float32)), 3569731584)
        ns = sp.csr_matrix(np.asarray(np.eye(4), dtype=np.float32))
        A = SimpleNamespace(SM_prime=prime, NS=ns, shape=(4, 4))
        factors = GB._canonical_twolevel_host(A)
        fake = _FakeTorch()
        events = []
        real_block = GB._canonical_block
        # upload() is a CLOSURE inside __init__ (not a method), so the upload
        # event is recorded where it lands: the sparse_csr_tensor call itself.
        real_csr = fake.sparse_csr_tensor

        def _build(m):
            events.append(("build", tuple(m.shape)))
            return real_block(m)

        def _csr(crow, ccol, cval, size=None, dtype=None, device=None):
            events.append(("upload", tuple(size)))
            return real_csr(crow, ccol, cval, size=size, dtype=dtype, device=device)

        fake.sparse_csr_tensor = _csr
        with mock.patch.object(GB, "_canonical_block", _build), \
             mock.patch.object(GB, "_torch", lambda: fake), \
             mock.patch.object(GB, "enabled", lambda: True), \
             mock.patch.object(GB, "available", lambda: True), \
             mock.patch.object(GB, "_device_free_bytes", lambda d: 1 << 40), \
             mock.patch.object(GB, "_device_available_bytes", lambda d: 1 << 40):
            op = GB.GpuTwoLevelOperator(A, device_ids=[0, 1], host_factors=factors)
        try:
            # two shards, each building its (2, 4) row block and then its (2, 4)
            # adjoint block: 8 events, strictly alternating.
            self.assertEqual(self._collapse(events)[:8],
                             [("build", (2, 4)), ("upload", (2, 4))] * 4,
                             "row block must be uploaded and released before adjoint")
        finally:
            op.close()

    def test_spmv_builds_one_block_at_a_time(self):
        fake = _FakeTorch()
        events = []
        real_check = GB._check_cuda_csr_indices
        real_convert = GB.GpuSparseMV._csr_to_torch
        state = {"uploading": False}
        # Rectangular, so the row block (r x 10) and the transpose block
        # (c x 6) are distinguishable by shape in the event log.
        matrix = sp.csr_matrix(
            (np.ones(12, np.float32),
             (np.repeat(np.arange(6), 2),
              np.array([0, 9, 1, 8, 2, 7, 3, 6, 4, 5, 0, 1]))),
            shape=(6, 10))
        rs = GB._row_shard_edges(matrix, 2)
        cs = GB._col_shard_edges(matrix, 2)
        # the nnz-balanced split gives the two shards different column widths
        expected = []
        for i in range(2):
            row_shape = (int(rs[i + 1] - rs[i]), int(matrix.shape[1]))
            col_shape = (int(cs[i + 1] - cs[i]), int(matrix.shape[0]))
            self.assertNotEqual(row_shape, col_shape,
                                "fixture must separate the two blocks by shape")
            expected += [("build", row_shape), ("upload", row_shape),
                         ("build", col_shape), ("upload", col_shape)]

        def _check(m):
            # _csr_to_torch re-checks the indices as part of the upload; suppress
            # that one so the log holds the BUILD order only.
            if not state["uploading"]:
                events.append(("build", tuple(m.shape)))
            return real_check(m)

        def _convert(self, m, dev):
            events.append(("upload", tuple(m.shape)))
            state["uploading"] = True
            try:
                return real_convert(self, m, dev)
            finally:
                state["uploading"] = False

        with mock.patch.object(GB, "_torch", lambda: fake), \
             mock.patch.object(GB, "_device_free_bytes", lambda d: 1 << 40), \
             mock.patch.object(GB, "_check_cuda_csr_indices", _check), \
             mock.patch.object(GB.GpuSparseMV, "_csr_to_torch", _convert):
            gpu = GB.GpuSparseMV(matrix, n_gpu=2, device_ids=[0, 1])
        try:
            self.assertEqual(events, expected,
                             "column block must not be built before the row block is on the device")
        finally:
            gpu.close()


class UploadValueBufferTest(unittest.TestCase):
    """A contiguous float32 block transfers straight from the scipy buffer.

    `np.array(csr.data, copy=True)` made a second 10.7 GB host copy of every
    c3=5.0 shard block (the dtype/copy loop needs to cast the INDICES, not the
    values).  The strided case still copies, so assert identity, not just dtype.
    """

    def test_contiguous_float32_values_are_not_copied(self):
        data = np.asarray(np.eye(4), dtype=np.float32)
        prime = sp.csr_matrix(data)
        ns = sp.csr_matrix(data)
        A = SimpleNamespace(SM_prime=prime, NS=ns, shape=(4, 4))
        factors = GB._canonical_twolevel_host(A)
        fake = _FakeTorch()
        with mock.patch.object(GB, "_torch", lambda: fake), \
             mock.patch.object(GB, "enabled", lambda: True), \
             mock.patch.object(GB, "available", lambda: True), \
             mock.patch.object(GB, "_device_free_bytes", lambda d: 1 << 40), \
             mock.patch.object(GB, "_device_available_bytes", lambda d: 1 << 40):
            op = GB.GpuTwoLevelOperator(A, device_ids=[0], host_factors=factors)
        try:
            self.assertTrue(any(x is prime.data for x in fake.as_tensor_inputs),
                            "contiguous float32 values must not be copied")
        finally:
            op.close()



if __name__ == "__main__":
    unittest.main(verbosity=2)
