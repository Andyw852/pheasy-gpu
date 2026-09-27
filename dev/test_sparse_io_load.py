#!/usr/bin/env python3
"""Tests for core.sparse_io: narrow int32 column indices on a huge CSR.

The motive is the c3=5.0 sensing matrix (nnz 3.57e9, n_cols 265662): scipy
saves and reloads it with int64 column indices because indptr needs int64,
costing 26.6 GiB of index buffers where 13.3 GiB suffice.  The loader streams
the npz members so no full-size int64 temporary exists, and the resulting
mixed dtype (indices int32 + indptr int64) must keep working for every op the
fit uses -- and is expected to FAIL for the indptr-expanding ones, which is
pinned here so a scipy change cannot surprise us silently.
"""
import os
import pickle
import sys
import tempfile
import unittest
import warnings
import zipfile
from unittest import mock

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import sparse_io
from core import gpu_backend


def _write_scipy_npz(path, data, indices, indptr, shape, compressed=False,
                     data_dtype=None):
    """A scipy-compatible CSR npz, written with explicit member dtypes."""
    kwargs = dict(format=np.array(b"csr"),
                  data=np.asarray(data, dtype=data_dtype),
                  indices=np.asarray(indices),
                  indptr=np.asarray(indptr),
                  shape=np.asarray(shape, dtype=np.int64))
    if compressed:
        np.savez_compressed(path, **kwargs)
    else:
        np.savez(path, **kwargs)
    return path


class LoadCsrTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _path(self, name):
        return os.path.join(self._tmp.name, name)

    def _fixture(self, name="sm.npz", compressed=False, data_dtype=np.float32):
        """3x6 CSR whose index members are int64 (the scipy big-matrix layout).

        Rows: [0, 3], [1], [2, 4, 5] -- canonical (sorted, no duplicates), which
        is what the pipeline always saves and what the mixed layout requires.
        """
        data = np.array([1.5, -2.0, 3.0, 4.5, 5.25, -6.0], dtype=data_dtype)
        indices = np.array([0, 3, 1, 2, 4, 5], dtype=np.int64)
        indptr = np.array([0, 2, 3, 6], dtype=np.int64)
        path = _write_scipy_npz(self._path(name), data, indices, indptr, (3, 6),
                                compressed=compressed, data_dtype=data_dtype)
        dense = np.zeros((3, 6), dtype=np.float64)
        for row in range(3):
            for k in range(indptr[row], indptr[row + 1]):
                dense[row, indices[k]] += data[k]
        return path, dense

    def test_narrow_load_matches_scipy_values_and_indptr(self):
        path, dense = self._fixture()
        m = sparse_io.load_csr(path)
        self.assertTrue(sp.issparse(m))
        self.assertEqual(m.format, "csr")
        self.assertEqual(m.indices.dtype, np.dtype(np.int32))
        self.assertEqual(m.indptr.dtype, np.dtype(np.int64), "indptr must stay int64")
        self.assertEqual(m.data.dtype, np.dtype(np.float32))
        self.assertTrue(m.has_canonical_format)
        np.testing.assert_allclose(m.toarray(), dense)
        # the FILE stores int64 (the c3=5.0 layout) while scipy normalizes a small
        # matrix back to int32 on load; values must agree either way.
        with np.load(path) as loaded:
            self.assertEqual(loaded["indices"].dtype, np.dtype(np.int64))
            self.assertEqual(loaded["indptr"].dtype, np.dtype(np.int64))
        ref = sp.load_npz(path)
        np.testing.assert_allclose(m.toarray(), ref.toarray())

    def test_one_element_chunks_exercise_the_streaming_loop(self):
        path, dense = self._fixture(name="chunk1.npz")
        m = sparse_io.load_csr(path, chunk_elems=1)
        self.assertEqual(m.indices.dtype, np.dtype(np.int32))
        np.testing.assert_allclose(m.toarray(), dense)

    def test_multi_chunk_roundtrip_matches_scipy(self):
        # 300x500 with 40 entries per row: several chunks of 997 elements, and a
        # file whose stored indices are int64 (the c3=5.0 layout).
        rng = np.random.default_rng(4)
        n_rows, n_cols, per = 300, 500, 40
        indptr = np.arange(n_rows + 1, dtype=np.int64) * per
        indices = np.empty(n_rows * per, dtype=np.int64)
        for row in range(n_rows):
            indices[row * per:(row + 1) * per] = np.sort(
                rng.choice(n_cols, size=per, replace=False))
        data = rng.standard_normal(n_rows * per)
        path = _write_scipy_npz(self._path("multi.npz"), data, indices, indptr,
                                (n_rows, n_cols), data_dtype=np.float32)
        m = sparse_io.load_csr(path, chunk_elems=997)
        ref = sp.load_npz(path)
        self.assertEqual(m.indices.dtype, np.dtype(np.int32))
        self.assertTrue(m.has_canonical_format)
        np.testing.assert_allclose(m.toarray(), ref.toarray())
        self.assertEqual(m.nnz, ref.nnz)

    def test_compressed_members_are_streamed_too(self):
        path, dense = self._fixture(name="sm_c.npz", compressed=True)
        with zipfile.ZipFile(path) as zf:
            self.assertTrue(all(i.compress_type == zipfile.ZIP_DEFLATED
                                for i in zf.infolist()))
        m = sparse_io.load_csr(path)
        self.assertEqual(m.indices.dtype, np.dtype(np.int32))
        np.testing.assert_allclose(m.toarray(), dense)

    def test_float64_data_dtype_is_preserved(self):
        path, dense = self._fixture(name="sm64.npz", data_dtype=np.float64)
        m = sparse_io.load_csr(path)
        self.assertEqual(m.data.dtype, np.dtype(np.float64))
        self.assertEqual(m.indices.dtype, np.dtype(np.int32))
        np.testing.assert_allclose(m.toarray(), dense)

    def test_int64_mode_matches_the_scipy_layout(self):
        # For a small matrix scipy itself normalizes to int32 (the big c3=5.0
        # file is the one that gets int64), so compare against scipy, not a
        # hard-coded width.
        path, dense = self._fixture()
        ref = sp.load_npz(path)
        m = sparse_io.load_csr(path, index_dtype="int64")
        self.assertEqual(m.indices.dtype, ref.indices.dtype)
        self.assertEqual(m.indptr.dtype, ref.indptr.dtype)
        np.testing.assert_allclose(m.toarray(), dense)

    def test_env_switch_is_honoured(self):
        path, _dense = self._fixture()
        ref = sp.load_npz(path)
        for value, expected in (("auto", np.int32), ("int32", np.int32),
                                ("int64", ref.indices.dtype),
                                ("legacy", ref.indices.dtype)):
            with self.subTest(value=value):
                old = os.environ.get("PHEASY_SM_INDEX_DTYPE")
                os.environ["PHEASY_SM_INDEX_DTYPE"] = value
                try:
                    self.assertEqual(sparse_io.load_csr(path).indices.dtype,
                                     np.dtype(expected))
                finally:
                    if old is None:
                        os.environ.pop("PHEASY_SM_INDEX_DTYPE", None)
                    else:
                        os.environ["PHEASY_SM_INDEX_DTYPE"] = old
        with self.assertRaises(ValueError):
            sparse_io.load_csr(path, index_dtype="banana")

    def test_auto_falls_back_when_int32_cannot_address_the_shape(self):
        path = _write_scipy_npz(self._path("wide.npz"), np.array([1.0]),
                                np.array([2 ** 31], dtype=np.int64),
                                np.array([0, 1], dtype=np.int64),
                                (1, 2 ** 31 + 1), data_dtype=np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = sparse_io.load_csr(path)
        self.assertTrue(any("narrow sensing-matrix indices" in str(w.message)
                            for w in caught), [str(w.message) for w in caught])
        self.assertEqual(m.indices.dtype, np.dtype(np.int64),
                         "scipy keeps int64 when the shape needs it")
        self.assertEqual(m.shape, (1, 2 ** 31 + 1))
        # strict int32 refuses instead of silently allocating the int64 layout
        with self.assertRaises(ValueError):
            sparse_io.load_csr(path, index_dtype="int32")

    def test_unsorted_file_falls_back_in_auto_and_raises_in_strict(self):
        path = _write_scipy_npz(self._path("unsorted.npz"),
                                np.array([1.0, 2.0], dtype=np.float32),
                                np.array([4, 1], dtype=np.int64),
                                np.array([0, 2], dtype=np.int64), (1, 6))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = sparse_io.load_csr(path)
        self.assertTrue(any("narrow sensing-matrix indices" in str(w.message)
                            for w in caught), [str(w.message) for w in caught])
        self.assertTrue(np.isfinite(m.toarray()).all())
        self.assertEqual(m.indices.dtype, sp.load_npz(path).indices.dtype)
        with self.assertRaises(ValueError):
            sparse_io.load_csr(path, index_dtype="int32")

    def test_non_scipy_npz_is_rejected_in_strict_mode(self):
        path = self._path("other.npz")
        np.savez(path, a=np.arange(4.0))
        with self.assertRaises(ValueError):
            sparse_io.load_csr(path, index_dtype="int32")

    def test_out_of_range_index_is_rejected(self):
        path = _write_scipy_npz(self._path("bad.npz"), np.array([1.0]),
                                np.array([7], dtype=np.int64),
                                np.array([0, 1], dtype=np.int64), (1, 6),
                                data_dtype=np.float32)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            m = sparse_io.load_csr(path)          # auto: scipy handles it
        self.assertEqual(m.indices[0], 7)         # scipy does not validate
        with self.assertRaises(ValueError):
            sparse_io.load_csr(path, index_dtype="int32")


class NarrowLayoutOpsTest(unittest.TestCase):
    """The mixed layout must work where the fit uses it and fail loudly elsewhere."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        data = np.array([1.5, -2.0, 3.0, 4.5, 5.25, -6.0], dtype=np.float32)
        # canonical rows: [0, 3], [1], [2, 4, 5]
        indices = np.array([0, 3, 1, 2, 4, 5], dtype=np.int64)
        indptr = np.array([0, 2, 3, 6], dtype=np.int64)
        cls.path = _write_scipy_npz(os.path.join(cls._tmp.name, "ops.npz"), data,
                                    indices, indptr, (3, 6))
        cls.m = sparse_io.load_csr(cls.path)
        cls.dense = cls.m.toarray()
        assert cls.m.indices.dtype == np.dtype(np.int32), "fixture must reach the mixed layout"
        assert cls.m.indptr.dtype == np.dtype(np.int64)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_fit_path_operations_work(self):
        m = self.m
        x = np.arange(6, dtype=np.float32)
        u = np.arange(3, dtype=np.float32)
        np.testing.assert_allclose(m @ x, self.dense @ x)
        np.testing.assert_allclose(m.T @ u, self.dense.T @ u)
        np.testing.assert_allclose(m[0:2].toarray(), self.dense[0:2])
        np.testing.assert_allclose(m[:, 1:4].T.tocsr().toarray(),
                                   self.dense[:, 1:4].T)
        self.assertEqual(m.T.tocsr().indices.dtype, np.dtype(np.int32))
        self.assertEqual(m.astype(np.float64).indices.dtype, np.dtype(np.int32))
        self.assertEqual(m.copy().indices.dtype, np.dtype(np.int32))
        self.assertEqual(sp.hstack([m, m], format="csr").indices.dtype, np.dtype(np.int32))
        self.assertEqual(sp.vstack([m, m], format="csr").indices.dtype, np.dtype(np.int32))
        self.assertEqual(m.multiply(2.0).indices.dtype, np.dtype(np.int32))
        self.assertTrue(np.isfinite(lsmr(m, u)[0]).all())
        m2 = pickle.loads(pickle.dumps(m))
        self.assertEqual(m2.indices.dtype, np.dtype(np.int32))
        np.testing.assert_allclose(m2.toarray(), self.dense)
        m3 = m.copy()
        m3.sum_duplicates()
        m3.sort_indices()
        self.assertEqual(m3.indices.dtype, np.dtype(np.int32))
        np.testing.assert_allclose(m3.toarray(), self.dense)

    def test_indptr_expanding_routines_raise_the_documented_error(self):
        # NB a fresh load each time: copy()/astype() normalize the dtypes (and
        # then these routines work), so the mixed layout must be tested directly.
        for name, call in (("tocoo", lambda m: m.tocoo()),
                           ("nonzero", lambda m: m.nonzero()),
                           ("eliminate_zeros",
                            lambda m: sparse_io.load_csr(self.path).eliminate_zeros())):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as ctx:
                    call(self.m)
                self.assertIn("dtype", str(ctx.exception))


class CanonicalFormatHelperTest(unittest.TestCase):
    """canonical_format() must not pay scipy's int64 upcast on narrow matrices.

    Measured: csr_has_sorted_indices()/csr_has_canonical_format() convert int32
    indices to indptr's int64 internally, so the first query on the c3=5.0
    c3=5.0-shaped narrow matrix allocates a 26.6 GiB temporary (VmHWM doubled).
    The helper verifies in chunks and caches the verdict.
    """

    @staticmethod
    def _mixed(sorted_rows=True):
        m = sp.csr_matrix((3, 5), dtype=np.float32)
        m.data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        if sorted_rows:
            m.indices = np.array([0, 3, 1, 4], dtype=np.int32)
        else:
            m.indices = np.array([3, 0, 1, 4], dtype=np.int32)
        m.indptr = np.array([0, 2, 3, 4], dtype=np.int64)
        return m

    def test_matches_scipy_and_caches_the_verdict(self):
        for sorted_rows in (True, False):
            with self.subTest(sorted_rows=sorted_rows):
                m = self._mixed(sorted_rows=sorted_rows)
                expected = sp.csr_matrix(
                    (m.data, m.indices.astype(np.int64), m.indptr), shape=m.shape
                ).has_canonical_format
                self.assertEqual(sparse_io.canonical_format(m), expected)
                # the verdict is cached, so the next query cannot re-run anything
                self.assertIsNotNone(getattr(m, "_has_canonical_format", None))
                self.assertEqual(sparse_io.canonical_format(m), expected)

    def test_does_not_call_scipys_compiled_check_for_mixed_indices(self):
        # scipy 1.17 imports the compiled checks into _compressed; 1.15 calls
        # them through scipy.sparse._sparsetools.  Patch every binding that
        # exists so the assertion holds on both.
        import scipy
        from scipy.sparse import _compressed
        modules = [_compressed]
        try:
            from scipy.sparse import _sparsetools
            modules.append(_sparsetools)
        except ImportError:
            pass
        patchers = [mock.patch.object(mod, name,
                                      side_effect=AssertionError("scipy check used"))
                    for mod in modules
                    for name in ("csr_has_canonical_format", "csr_has_sorted_indices")
                    if hasattr(mod, name)]
        if not patchers:
            self.skipTest("scipy %s exposes no patchable compiled check"
                          % scipy.__version__)
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        m = self._mixed()
        self.assertTrue(sparse_io.canonical_format(m))

    def test_unsorted_narrow_matrix_is_repaired_by_canonicalize(self):
        m = self._mixed(sorted_rows=False)
        self.assertFalse(sparse_io.canonical_format(m))
        sparse_io.canonicalize(m)
        self.assertEqual(m.indices.dtype, np.dtype(np.int64), "widened for the C kernels")
        self.assertTrue(m.has_canonical_format)
        # row 0 was stored as (col 3, val 1), (col 0, val 2): sorting keeps the
        # data paired with its own column, so the dense row is [2, 0, 0, 1, 0].
        np.testing.assert_allclose(
            m.toarray(), np.array([[2.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.0, 3.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 4.0]]))

    def test_duplicate_entries_are_detected(self):
        m = sp.csr_matrix((2, 5), dtype=np.float32)
        m.data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        m.indices = np.array([1, 1, 3], dtype=np.int32)
        m.indptr = np.array([0, 2, 3], dtype=np.int64)
        self.assertFalse(sparse_io.canonical_format(m))
        sparse_io.canonicalize(m)
        np.testing.assert_allclose(m.toarray(), [[0.0, 3.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 3.0, 0.0]])

    def test_chunk_boundaries_do_not_mark_rows_wrong(self):
        # one row per chunk boundary, with chunk_elems=1
        m = self._mixed()
        self.assertTrue(sparse_io.canonical_format(m, chunk_elems=1))
        m2 = self._mixed(sorted_rows=False)
        self.assertFalse(sparse_io.canonical_format(m2, chunk_elems=1))


class RunPheasyWiringTest(unittest.TestCase):
    """The fit must go through the narrow loader, not scipy's int64 layout."""

    def test_sensing_matrix_load_uses_sparse_io(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "run_pheasy.py"), "r", encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("sparse_io import load_csr", src)
        self.assertNotIn("self.SM_prime = spmat.load_npz(self.SensingMatrixFile)",
                         src, "the load site must not fall back to scipy silently")


class MajorIndexExpansionTest(unittest.TestCase):
    """gpu_backend._csr_major_indices: the COO fallback for the narrow layout."""

    @staticmethod
    def _mixed():
        m = sp.csr_matrix((3, 5), dtype=np.float32)
        m.data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        m.indices = np.array([0, 3, 1, 4], dtype=np.int32)
        m.indptr = np.array([0, 2, 3, 4], dtype=np.int64)
        return m

    def test_matches_tocoo_for_a_consistent_matrix(self):
        ref = sp.csr_matrix(np.array([[0, 1, 0], [2, 0, 3]], dtype=np.float32))
        np.testing.assert_array_equal(gpu_backend._csr_major_indices(ref),
                                      ref.tocoo().row)

    def test_handles_the_mixed_layout_that_tocoo_rejects(self):
        m = self._mixed()
        with self.assertRaises(ValueError):
            m.tocoo()
        rows = gpu_backend._csr_major_indices(m)
        self.assertEqual(rows.dtype, np.dtype(np.int32), "rows follow indices")
        np.testing.assert_array_equal(rows, np.array([0, 0, 1, 2], dtype=np.int32))
        coo = sp.coo_matrix((m.data, (rows, m.indices)), shape=m.shape)
        np.testing.assert_allclose(coo.toarray(), m.toarray())


if __name__ == "__main__":
    unittest.main(verbosity=2)
