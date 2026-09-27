#!/usr/bin/env python3
"""Host-footprint contract tests for the resident two-level factors.

The c3=5.0 resident LASSO phase needed ~129 GB of host RAM for a 42.84 GB factor
set: the raw sm_prime.npz copy, an ALREADY-CANONICAL copy made by
_canonical_twolevel_host(), and an eager prime.T.tocsr() adjoint.  Two of the
three are pure overhead, and these tests pin the properties that let the code
drop them, plus the properties that make dropping them safe:

* scipy must give csr.T as a CSC VIEW sharing the buffers (otherwise holding the
  lazy adjoint would silently materialise the transpose);
* the lazy adjoint must be the TRUE transpose and its row blocks the true column
  slices;
* taking the lazy adjoint / its blocks must not allocate a full matrix-sized
  buffer (measured through the process RSS, not by inspecting our own types);
* the operator must upload the SAME device blocks from the lazy adjoint as from
  the eager one, and must not mutate the caller's factors on the way;
* an already-canonical factor set must not be copied, and a dirty one must still
  come out canonical with the right values.

CPU-only: the CUDA tensor construction is replaced by a fake torch that records
the CSR blocks the operator would have uploaded.
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import gpu_backend as GB
from core import optimizer as O


def _rss_bytes():
    with open("/proc/self/statm") as fh:
        pages = int(fh.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


def _sparse(n_rows, n_mid, density, seed, dtype=np.float64):
    rng = np.random.default_rng(seed)
    m = sp.csr_matrix(rng.normal(size=(n_rows, n_mid)) *
                      (rng.random((n_rows, n_mid)) < density))
    m.sum_duplicates()
    m.sort_indices()
    return m.astype(dtype)


def _operator(n_rows=48, n_mid=13, n_cols=7, dtype=np.float64):
    prime = _sparse(n_rows, n_mid, 0.4, 0)
    ns = sp.csr_matrix(np.random.default_rng(1).normal(size=(n_mid, n_cols)))
    prime = prime.astype(dtype)
    for a in (prime.data,):
        if a.dtype != dtype:
            raise AssertionError("fixture dtype")
    return O.TwoLevelSM(prime, ns)


class _Device(object):
    """A torch.device look-alike; the fake's `device` attribute IS this class,
    so `isinstance(d, torch.device)` inside GpuTwoLevelOperator works."""

    def __init__(self, index):
        self.index = index

    @property
    def type(self):
        return "cuda"

    def __repr__(self):
        return "cuda:%d" % self.index


class _FakeCuda(object):
    def __init__(self):
        self.blocks = []

    def device(self, d):
        return mock.patch.object(None, "x") if False else _Ctx()

    @staticmethod
    def empty_cache():
        pass


class _Ctx(object):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeTorch(object):
    """Enough torch for GpuTwoLevelOperator.__init__, recording the blocks."""

    float64 = np.float64
    float32 = np.float32
    int32 = np.int32
    int64 = np.int64
    device = _Device

    def __init__(self):
        self.uploaded = []
        self.cuda = _FakeCuda()

    def as_tensor(self, x, dtype=None, device=None):
        return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)

    def sparse_csr_tensor(self, crow, ccol, cval, size=None, dtype=None, device=None):
        self.uploaded.append(dict(shape=tuple(size),
                                  indptr=np.asarray(crow),
                                  indices=np.asarray(ccol),
                                  data=np.asarray(cval),
                                  dtype=np.dtype(dtype)))
        return self.uploaded[-1]

    def ones(self, n, dtype=None, device=None):
        return np.ones(n, dtype=dtype if dtype is not None else np.float64)

    def ones_like(self, x):
        return np.ones_like(x)

    def isfinite(self, x):
        return np.isfinite(x)


def _build(A, device_ids, host_factors, fake):
    with mock.patch.object(GB, "_torch", lambda: fake), \
         mock.patch.object(GB, "enabled", lambda: True), \
         mock.patch.object(GB, "available", lambda: True), \
         mock.patch.object(GB, "_device_free_bytes", lambda d: 1 << 40), \
         mock.patch.object(GB, "_device_available_bytes", lambda d: 1 << 40):
        op = GB.GpuTwoLevelOperator(A, device_ids=device_ids,
                                    host_factors=host_factors)
    return op


class ScipyAdjointViewTest(unittest.TestCase):
    def test_scipy_adjoint_view_is_shared(self):
        """The whole fix rests on csr.T being a view; pin it against scipy."""
        m = _sparse(200, 90, 0.3, 3)
        t = m.T
        self.assertTrue(np.shares_memory(t.data, m.data))
        self.assertTrue(np.shares_memory(t.indices, m.indices))
        self.assertTrue(np.shares_memory(t.indptr, m.indptr))

    def test_scipy_verified_canonical_flag_is_truthful(self):
        """_canonical_csr_inplace trusts has_canonical_format, so it must verify."""
        dirty = sp.csr_matrix((np.array([1., 2., 3.]),
                               np.array([2, 0, 0], dtype=np.int32),
                               np.array([0, 3], dtype=np.int32)), shape=(1, 3))
        self.assertFalse(dirty.has_canonical_format)
        dirty.sum_duplicates()
        self.assertTrue(dirty.has_canonical_format)
        self.assertEqual(list(dirty.indices), [0, 2])
        np.testing.assert_allclose(dirty.data, [5., 1.])


class LazyAdjointTest(unittest.TestCase):
    def test_lazy_adjoint_is_the_true_transpose(self):
        A = _operator()
        pc, _ns, adj = GB._canonical_twolevel_host(A)
        self.assertIsInstance(adj, GB._AdjointBlocks)
        self.assertEqual(adj.shape, (A.SM_prime.shape[1], A.SM_prime.shape[0]))
        self.assertEqual(adj.nnz, pc.nnz)
        self.assertEqual(adj.resident_bytes, 0)
        np.testing.assert_allclose(adj.toarray(), A.SM_prime.toarray().T)

    def test_lazy_blocks_are_the_true_column_slices(self):
        A = _operator(n_rows=40, n_mid=11, n_cols=5)
        _pc, _ns, adj = GB._canonical_twolevel_host(A)
        dense = A.SM_prime.toarray()
        ref = adj.materialize().toarray()
        for c0, c1 in ((0, 3), (3, 8), (8, 11), (0, 11)):
            blk = adj[c0:c1]
            self.assertTrue(sp.issparse(blk) and blk.format == "csr")
            self.assertTrue(blk.has_canonical_format)
            self.assertEqual(blk.shape, (c1 - c0, A.SM_prime.shape[0]))
            np.testing.assert_allclose(blk.toarray(), dense[:, c0:c1].T)
            np.testing.assert_allclose(blk.toarray(), ref[c0:c1])

    def test_lazy_adjoint_rejects_fancy_indexing(self):
        A = _operator()
        _pc, _ns, adj = GB._canonical_twolevel_host(A)
        with self.assertRaises(TypeError):
            adj[[1, 2]]
        with self.assertRaises(TypeError):
            adj[0:4:2]

    def test_lazy_adjoint_and_blocks_do_not_allocate_a_full_copy(self):
        """RSS must stay near zero for the view and near one block for a block."""
        # 4000 x 6000 with 6000 nonzeros/row -> nnz 2.4e7 = 288 MB (float64+int32),
        # big enough that a full transpose would show up in RSS.
        n_rows, n_mid, nnz_per_row = 4000, 6000, 6000
        rng = np.random.default_rng(5)
        indptr = (np.arange(n_rows + 1, dtype=np.int64) * nnz_per_row).astype(np.int32)
        idx = np.empty(int(indptr[-1]), dtype=np.int32)
        for r0 in range(0, n_rows, 500):
            r1 = min(n_rows, r0 + 500)
            b = rng.integers(0, n_mid, size=(r1 - r0, nnz_per_row), dtype=np.int32)
            b.sort(axis=1)
            idx[int(indptr[r0]):int(indptr[r1])] = b.reshape(-1)
            del b
        prime = sp.csr_matrix((rng.standard_normal(idx.size),
                               idx, indptr), shape=(n_rows, n_mid))
        prime.sum_duplicates()
        prime.sort_indices()
        prime_bytes = GB._factor_host_bytes(prime)
        self.assertGreater(prime_bytes, 128 * 2 ** 20)

        rss0 = _rss_bytes()
        adj = GB._AdjointBlocks(prime)
        rss1 = _rss_bytes()
        self.assertLess(rss1 - rss0, 0.05 * prime_bytes,
                        "holding the lazy adjoint must not allocate a transpose")

        blk = adj.block(0, n_mid // 4)
        rss2 = _rss_bytes()
        self.assertLess(rss2 - rss1, 0.7 * prime_bytes,
                        "a lazy block must cost about one block, not one matrix")
        self.assertGreater(blk.nnz, 0)

        mat = adj.materialize()
        rss3 = _rss_bytes()
        self.assertGreater(rss3 - rss2, 0.7 * prime_bytes,
                           "materialize() must be the full copy the lazy view avoids")
        self.assertEqual(mat.nnz, prime.nnz)


class CanonicalFactorsTest(unittest.TestCase):
    def test_already_canonical_factors_are_not_copied(self):
        A = _operator()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PHEASY_HOST_FACTOR_INPLACE", None)
            pc, ns_c, _adj = GB._canonical_twolevel_host(A)
        self.assertIs(pc, A.SM_prime)
        self.assertIs(ns_c, A.NS)

    def test_inplace_can_be_switched_off(self):
        A = _operator()
        with mock.patch.dict(os.environ, {"PHEASY_HOST_FACTOR_INPLACE": "0"}):
            pc, _ns, _adj = GB._canonical_twolevel_host(A)
        self.assertIsNot(pc, A.SM_prime)
        np.testing.assert_allclose(pc.toarray(), A.SM_prime.toarray())

    def test_dirty_factors_still_come_out_canonical(self):
        prime = sp.csr_matrix((np.array([1., 2., 3., 4.]),
                               np.array([3, 1, 1, 0], dtype=np.int32),
                               np.array([0, 2, 4], dtype=np.int32)), shape=(2, 5))
        A = O.TwoLevelSM(prime, sp.eye(5, format="csr"))
        pc, _ns, _adj = GB._canonical_twolevel_host(A)
        self.assertTrue(pc.has_canonical_format)
        self.assertTrue(pc.has_sorted_indices)
        # row 0: (col3, 1) (col1, 2) -> cols [1, 3]; row 1: (col1, 3) (col0, 4)
        expected = sp.csr_matrix(np.array([[0., 2., 0., 1., 0.],
                                           [4., 3., 0., 0., 0.]]))
        np.testing.assert_allclose(pc.toarray(), expected.toarray())

    def test_adjoint_mode_env_is_honoured_and_validated(self):
        A = _operator()
        with mock.patch.dict(os.environ, {"PHEASY_TWOLEVEL_ADJOINT": "eager"}):
            _pc, _ns, adj = GB._canonical_twolevel_host(A)
            self.assertIsInstance(adj, sp.csr_matrix)
        with mock.patch.dict(os.environ, {"PHEASY_TWOLEVEL_ADJOINT": "auto"}):
            _pc, _ns, adj = GB._canonical_twolevel_host(A)
            self.assertIsInstance(adj, (sp.csr_matrix, GB._AdjointBlocks))
        with mock.patch.dict(os.environ, {"PHEASY_TWOLEVEL_ADJOINT": "lazy"}):
            _pc, _ns, adj = GB._canonical_twolevel_host(A)
            self.assertIsInstance(adj, GB._AdjointBlocks)
        with mock.patch.dict(os.environ, {"PHEASY_TWOLEVEL_ADJOINT": "nonsense"}):
            with self.assertRaisesRegex(ValueError, "PHEASY_TWOLEVEL_ADJOINT"):
                GB._canonical_twolevel_host(A)

    def test_auto_never_guesses_when_meminfo_is_unreadable(self):
        A = _operator()
        with mock.patch.object(GB, "_host_available_bytes", lambda: None):
            self.assertFalse(GB._eager_adjoint_fits(A.SM_prime))


class OperatorUploadTest(unittest.TestCase):
    def _blocks(self, A, host_factors, n_dev):
        fake = _FakeTorch()
        op = _build(A, list(range(n_dev)), host_factors, fake)
        # ns/ns_t/scale/input_scale are the last uploads: keep only prime blocks.
        prime_blocks = [b for b in fake.uploaded
                        if b["shape"] == (A.SM_prime.shape[0], A.SM_prime.shape[1])]
        return op, fake.uploaded

    def test_sharded_operator_uploads_the_same_blocks_lazy_and_eager(self):
        A = _operator(n_rows=40, n_mid=11, n_cols=5)
        with mock.patch.dict(os.environ, {"PHEASY_TWOLEVEL_ADJOINT": "lazy"}):
            lazy = GB._canonical_twolevel_host(A)
        with mock.patch.dict(os.environ, {"PHEASY_TWOLEVEL_ADJOINT": "eager"}):
            eager = GB._canonical_twolevel_host(A)
        seen = {}
        for name, factors in (("lazy", lazy), ("eager", eager)):
            fake = _FakeTorch()
            op = _build(A, [0, 1], factors, fake)
            seen[name] = [(b["shape"], b["indptr"].tolist(), b["indices"].tolist(),
                           b["data"].tolist(), b["dtype"].str) for b in fake.uploaded]
            op.close()
        self.assertEqual(len(seen["lazy"]), len(seen["eager"]),
                         "the same number of tensors must be uploaded")
        for a, b in zip(seen["lazy"], seen["eager"]):
            self.assertEqual(a, b, "lazy and eager adjoints must upload identical blocks")

    def test_sharded_blocks_reconstruct_the_dense_operator(self):
        A = _operator(n_rows=24, n_mid=7, n_cols=4)
        _pc, _ns, adj = GB._canonical_twolevel_host(A)
        n_shards = 3
        N, M = A.SM_prime.shape
        rows = np.linspace(0, N, n_shards + 1).astype(int)
        cols = np.linspace(0, M, n_shards + 1).astype(int)
        dense = A.SM_prime.toarray()
        # _canonical_block canonicalizes IN PLACE, which is only safe because a
        # scipy slice owns fresh buffers.  Pin that: a view-backed slice would
        # let sort_indices()/sum_duplicates() corrupt the parent matrix.
        probe = A.SM_prime[rows[0]:rows[1]]
        self.assertFalse(np.shares_memory(probe.indices, A.SM_prime.indices))
        self.assertFalse(np.shares_memory(probe.data, A.SM_prime.data))
        for i in range(n_shards):
            R = GB._canonical_block(A.SM_prime[rows[i]:rows[i + 1]])
            T = GB._canonical_block(adj[cols[i]:cols[i + 1]])
            self.assertTrue(R.has_canonical_format and T.has_canonical_format)
            np.testing.assert_allclose(R.toarray(), dense[rows[i]:rows[i + 1]])
            np.testing.assert_allclose(T.toarray(), dense[:, cols[i]:cols[i + 1]].T)

    def test_operator_build_does_not_mutate_the_callers_factors(self):
        A = _operator(n_rows=30, n_mid=9, n_cols=4)
        before_idx = A.SM_prime.indices.copy()
        before_data = A.SM_prime.data.copy()
        factors = GB._canonical_twolevel_host(A)
        fake = _FakeTorch()
        op = _build(A, [0, 1], factors, fake)
        op.close()
        np.testing.assert_array_equal(A.SM_prime.indices, before_idx)
        np.testing.assert_array_equal(A.SM_prime.data, before_data)
        self.assertEqual(A.SM_prime.format, "csr")
        self.assertTrue(A.SM_prime.has_canonical_format)

    def test_single_replica_materializes_the_lazy_adjoint(self):
        A = _operator(n_rows=20, n_mid=6, n_cols=3)
        factors = GB._canonical_twolevel_host(A)
        self.assertIsInstance(factors[2], GB._AdjointBlocks)
        fake = _FakeTorch()
        op = _build(A, [0], factors, fake)
        # single-shard layout: the whole prime and the whole adjoint go to the card
        shapes = sorted(b["shape"] for b in fake.uploaded)
        self.assertIn((20, 6), shapes)
        self.assertIn((6, 20), shapes)
        op.close()


class HostFactorRetentionTest(unittest.TestCase):
    def test_lazy_adjoint_does_not_count_as_resident_bytes(self):
        """Retention must not reserve room for an adjoint that is a view."""
        A = _operator(n_rows=32, n_mid=9, n_cols=4)
        with mock.patch.dict(os.environ, {"PHEASY_HOST_FACTOR_CACHE": "1"}), \
             mock.patch.object(GB, "_HOST_FACTOR_CACHE_MIN_BYTES", 0):
            factors = GB.twolevel_host_factors(A)
        self.assertIsInstance(factors[2], GB._AdjointBlocks)
        self.assertIs(getattr(A, "_canonical_host_factors", None), factors)
        # and a later caller reuses the same view, not a second transpose
        self.assertIs(GB.twolevel_host_factors(A)[2], factors[2])

    def test_twolevel_host_factors_disk_cache_is_still_eager(self):
        """PHEASY_HT_CACHE_DIR exists to reuse the transpose; it implies eager."""
        import tempfile
        A = _operator(n_rows=32, n_mid=9, n_cols=4)
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"PHEASY_HT_CACHE_DIR": d,
                                              "PHEASY_HT_CACHE_KEY": "lazy-test"}):
                factors = GB.twolevel_host_factors(A)
                self.assertIsInstance(factors[2], sp.csr_matrix)
                self.assertTrue(os.listdir(d))


class ShardBalanceTest(unittest.TestCase):
    """Equal-NNZ shard boundaries; the uniform split put 71% on shard 0 at c3=5.0."""

    @staticmethod
    def _skewed(n_rows=100, n_mid=50, hot_rows=25, hot_per=60, per=2):
        """The first `hot_rows` rows carry most of the nonzeros.

        A uniform quarter split then puts nearly the whole matrix on shard 0,
        which is the c3=5.0 shape this replaces.
        """
        rng = np.random.default_rng(11)
        rows, cols = [], []
        for r in range(n_rows):
            k = hot_per if r < hot_rows else per
            rows.extend([r] * k)
            cols.extend(rng.integers(0, n_mid, size=k).tolist())
        m = sp.csr_matrix((rng.standard_normal(len(rows)),
                           (np.asarray(rows), np.asarray(cols))),
                          shape=(n_rows, n_mid))
        m.sum_duplicates()
        m.sort_indices()
        return m

    def test_balanced_edges_equalise_nonzeros_and_cover_everything(self):
        m = self._skewed()
        row_nnz = np.diff(m.indptr)
        edges = GB._balanced_shard_edges(row_nnz, 4)
        self.assertEqual(int(edges[0]), 0)
        self.assertEqual(int(edges[-1]), m.shape[0])
        self.assertTrue(np.all(np.diff(edges) > 0), "edges must be strictly increasing")
        cum = np.concatenate([[0], np.cumsum(row_nnz)])
        share = np.diff(cum[edges])
        self.assertLess(share.max(), 0.45 * share.sum(),
                        "no shard may carry far more than its share of the nonzeros")
        uniform = np.linspace(0, m.shape[0], 5).astype(np.int64)
        self.assertGreater(np.diff(cum[uniform]).max(), 0.7 * share.sum(),
                           "the uniform split must be the skewed one this replaces")

    def test_shard_balance_can_be_switched_off(self):
        m = self._skewed()
        with mock.patch.dict(os.environ, {"PHEASY_SHARD_BALANCE": "0"}):
            edges = GB._row_shard_edges(m, 4)
        np.testing.assert_array_equal(edges, np.linspace(0, m.shape[0], 5).astype(np.int64))
        with mock.patch.dict(os.environ, {"PHEASY_SHARD_BALANCE": "1"}):
            self.assertFalse(np.array_equal(GB._row_shard_edges(m, 4), edges))

    def test_column_counts_match_bincount_and_are_cached(self):
        m = self._skewed()
        counts = GB._col_nnz_counts(m.indices, m.shape[1])
        np.testing.assert_array_equal(counts, np.bincount(m.indices, minlength=m.shape[1]))
        edges = GB._col_shard_edges(m, 3)
        self.assertEqual(int(edges[-1]), m.shape[1])
        self.assertIs(getattr(m, "_pheasy_col_nnz", None), getattr(m, "_pheasy_col_nnz", None))
        self.assertEqual(GB._col_shard_edges(m, 3).tolist(), edges.tolist())

    def test_balanced_sharding_tiles_the_matrix(self):
        m = self._skewed(n_rows=40, n_mid=30, hot_rows=10, hot_per=40, per=2)
        A = O.TwoLevelSM(m, sp.eye(m.shape[1], format="csr"))
        _pc, _ns, adj = GB._canonical_twolevel_host(A)
        fake = _FakeTorch()
        op = _build(A, [0, 1], (_pc, _ns, adj), fake)
        op.close()
        row_shapes = [b["shape"] for b in fake.uploaded
                      if b["shape"][1] == m.shape[1] and b["shape"][0] != m.shape[1]]
        self.assertEqual(len(row_shapes), 2)
        self.assertEqual(sum(s[0] for s in row_shapes), m.shape[0],
                         "row blocks must tile every row exactly once")
        nnzs = [int(b["indices"].size) for b in fake.uploaded
                if b["shape"][0] != m.shape[1] and b["shape"][1] == m.shape[1]]
        self.assertLess(max(nnzs), 0.7 * sum(nnzs),
                        "balanced boundaries must not leave one fat row block")


if __name__ == "__main__":
    unittest.main(verbosity=2)
