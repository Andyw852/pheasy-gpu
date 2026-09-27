"""Memory-lean sparse I/O for the huge sensing matrices.

scipy.sparse.load_npz() + csr_matrix() force ONE index dtype for indices and
indptr, so a CSR whose nnz exceeds 2**31 gets int64 COLUMN indices even when
every column index fits int32.  The c3=5.0 sensing matrix is exactly that case:
shape (454656, 265662), nnz 3.57e9, so its column indices occupy 26.6 GiB where
13.3 GiB suffice, and the fit holds them until the factors are dropped.

load_csr() streams the npz members (stored or compressed) and builds the CSR as
    data as stored (float32) + indices int32 + indptr int64
with no full-size int64 temporary: the peak is the destination arrays alone
(26.7 GiB for c3=5.0 instead of a 39.9 GiB scipy load).

The mixed dtype is legal for every kernel the fit uses -- matvec, slicing,
transposes, tocsr, lsmr/lsqr, sum_duplicates/sort_indices, hstack, vstack and
pickle round-trips (all verified in dev/test_sparse_io_load.py) -- but NOT for
the indptr-expanding routines tocoo(), nonzero() and eliminate_zeros(), which
raise "Output dtype not compatible with inputs".  Nothing in the fit calls those
on SM_prime (the one GPU COO fallback in gpu_backend.load_sensing_matrix was
made dtype-lean), and PHEASY_SM_INDEX_DTYPE=int64 restores the old layout.

  PHEASY_SM_INDEX_DTYPE=auto   (default) narrow to int32 when representable,
                               fall back to scipy for any format surprise
  PHEASY_SM_INDEX_DTYPE=int32  force the narrow layout (errors propagate)
  PHEASY_SM_INDEX_DTYPE=int64  plain scipy.sparse.load_npz
"""
import os
import zipfile

import numpy as np
import scipy.sparse as sp

__all__ = ["load_csr", "read_index_dtype", "canonical_format", "canonicalize"]

_MODE_ENV = "PHEASY_SM_INDEX_DTYPE"
_CHUNK_ELEMS = 1 << 24
_MEMBERS = ("data", "indices", "indptr", "shape")


def read_index_dtype(mode=None):
    """Resolve PHEASY_SM_INDEX_DTYPE (or an explicit mode) to int32/int64."""
    if mode is None:
        mode = os.environ.get(_MODE_ENV, "auto")
    mode = str(mode).strip().lower()
    if mode in ("", "auto", "narrow", "int32", "32"):
        return "int32"
    if mode in ("int64", "64", "legacy", "off", "0"):
        return "int64"
    raise ValueError("%s must be auto, int32 or int64 (got %r)" % (_MODE_ENV, mode))


def _header(fp):
    version = np.lib.format.read_magic(fp)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fp)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fp)
    else:
        raise ValueError("unsupported .npy version %r" % (version,))
    return tuple(int(x) for x in shape), bool(fortran), np.dtype(dtype)


def _stream(zf, name, out, chunk_elems, limit=None):
    """Fill the preallocated 1-D `out` from the stored `name`.npy member."""
    with zf.open(name + ".npy") as fp:
        shape, fortran, src_dtype = _header(fp)
        if fortran or shape != (int(out.size),):
            raise ValueError("%s member shape %s does not match %d"
                             % (name, shape, out.size))
        itemsize = src_dtype.itemsize
        done = 0
        while done < out.size:
            take = min(max(1, int(chunk_elems)), out.size - done)
            raw = fp.read(take * itemsize)
            if len(raw) != take * itemsize:
                raise ValueError("%s member is truncated" % name)
            piece = np.frombuffer(raw, dtype=src_dtype, count=take)
            if limit is not None and piece.size:
                if int(piece.min()) < 0 or int(piece.max()) > int(limit):
                    raise ValueError("%s holds an index outside [0, %d]"
                                     % (name, limit))
            out[done:done + take] = piece
            done += take
    return out


def canonical_format(matrix, chunk_elems=_CHUNK_ELEMS):
    """matrix.has_canonical_format WITHOUT scipy's int64 upcast for mixed dtypes.

    scipy's compiled csr_has_sorted_indices/csr_has_canonical_format convert the
    indices to the indptr dtype internally, so on the narrow layout (int32
    indices + int64 indptr) the FIRST query allocates a full int64 copy of the
    indices -- 26.6 GiB at c3=5.0, worse than everything the narrow layout
    saves.  This verifies in chunks (temporaries of one chunk only) and caches
    the verdict through scipy's public setter, so later queries are free.
    """
    cached = getattr(matrix, "_has_canonical_format", None)
    if cached is not None:
        return bool(cached)
    if matrix.indices.dtype == matrix.indptr.dtype:
        return bool(matrix.has_canonical_format)
    ok = _rows_are_canonical(matrix.indices, matrix.indptr, chunk_elems)
    matrix.has_canonical_format = ok          # public setter caches BOTH flags
    return ok


def _rows_are_canonical(indices, indptr, chunk_elems):
    """Sorted, duplicate-free indices within every row (chunk-sized temporaries)."""
    nnz = int(indices.size)
    if nnz == 0:
        return True
    indptr = np.asarray(indptr)
    step = max(1, int(chunk_elems))
    last = None
    for start in range(0, nnz, step):
        stop = min(start + step, nnz)
        chunk = indices[start:stop]
        if chunk.size == 0:
            continue
        is_row_start = np.zeros(chunk.size, dtype=bool)
        lo = max(0, int(np.searchsorted(indptr, start, side="right")) - 1)
        hi = int(np.searchsorted(indptr, stop, side="left"))
        starts = np.asarray(indptr[lo:hi], dtype=np.int64) - start
        starts = starts[(starts >= 0) & (starts < chunk.size)]
        if starts.size:
            is_row_start[starts] = True
        # A pair that spans a chunk boundary compares only when this chunk's
        # first element does not itself start a row (otherwise they are in
        # different rows and any order is fine).
        if last is not None and not bool(is_row_start[0]) and chunk[0] <= last:
            return False
        if chunk.size > 1:
            unsorted = chunk[1:] <= chunk[:-1]
            unsorted &= ~is_row_start[1:]
            if bool(unsorted.any()):
                return False
        last = chunk[-1]
    return True


def canonicalize(matrix, chunk_elems=_CHUNK_ELEMS):
    """sum_duplicates + sort_indices, widening the narrow layout first.

    scipy's sort/dedupe kernels need indptr and indices to share one dtype, so a
    narrow matrix is widened (one full index copy) before it is canonicalized.
    """
    if matrix.indices.dtype != matrix.indptr.dtype:
        matrix.indices = matrix.indices.astype(matrix.indptr.dtype)
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.has_canonical_format = True
    return matrix


def load_csr(path, index_dtype=None, chunk_elems=_CHUNK_ELEMS):
    """Load a save_npz() CSR, narrowing int64 column indices to int32.

    Returns a scipy CSR whose `indices` are int32 (when asked and possible) and
    whose `indptr` stays int64 (cumulative nnz does exceed int32).  In auto mode
    any unexpected file falls back to scipy.sparse.load_npz() with a warning.
    """
    mode = read_index_dtype(index_dtype)
    strict = str(index_dtype).strip().lower() in ("int32", "32") if index_dtype else False
    if mode == "int64":
        matrix = sp.load_npz(path)
        _report(path, matrix, "scipy")
        return matrix
    try:
        matrix = _load_narrow(path, chunk_elems)
    except Exception as exc:
        if strict:
            raise
        import warnings
        warnings.warn("narrow sensing-matrix indices unavailable (%s: %s); "
                      "loading with scipy" % (type(exc).__name__, exc),
                      RuntimeWarning, stacklevel=2)
        matrix = sp.load_npz(path)
        _report(path, matrix, "scipy-fallback")
        return matrix
    _report(path, matrix, "narrow")
    return matrix


def _load_narrow(path, chunk_elems):
    dtype = np.dtype(np.int32)
    limit = int(np.iinfo(dtype).max)
    with zipfile.ZipFile(os.fspath(path), "r") as zf:
        names = set(zf.namelist())
        missing = [m for m in _MEMBERS if m + ".npy" not in names]
        if missing:
            raise ValueError("not a scipy CSR npz (missing %s)" % ", ".join(missing))
        shape_buf = np.empty(2, dtype=np.int64)
        _stream(zf, "shape", shape_buf, 2)
        n_rows, n_cols = int(shape_buf[0]), int(shape_buf[1])
        if max(n_rows, n_cols) > limit:
            raise ValueError("int32 cannot address shape %s" % ((n_rows, n_cols),))
        indptr = np.empty(n_rows + 1, dtype=np.int64)
        _stream(zf, "indptr", indptr, chunk_elems)
        nnz = int(indptr[-1])
        with zf.open("data.npy") as fp:
            data_shape, _fortran, data_dtype = _header(fp)
        if data_shape != (nnz,):
            raise ValueError("data member shape %s does not match nnz %d"
                             % (data_shape, nnz))
        data = np.empty(nnz, dtype=data_dtype)
        _stream(zf, "data", data, chunk_elems)
        indices = np.empty(nnz, dtype=dtype)
        _stream(zf, "indices", indices, chunk_elems, limit=n_cols - 1)
    matrix = sp.csr_matrix((n_rows, n_cols), dtype=data.dtype)
    # Direct assignment is deliberate: csr_matrix((data, indices, indptr))
    # re-derives a COMMON index dtype from max(nnz, n_rows) (scipy
    # _get_index_dtype) and would upcast the int32 indices straight back to
    # int64, which is the 26.6 GiB layout this loader exists to avoid.  The
    # checks below then compute the format flags from THESE arrays.
    matrix.data = data
    matrix.indices = indices
    matrix.indptr = indptr
    if not canonical_format(matrix, chunk_elems):
        # sum_duplicates()/sort_indices() reorder through compressed C kernels
        # that require indptr and indices to share ONE dtype, so the mixed
        # layout cannot canonicalize itself in place (ValueError: "Output dtype
        # not compatible with inputs").  Every save_npz() this pipeline writes
        # is canonical, so refuse loudly and let auto mode fall back to scipy,
        # which loads the int64 layout and can sort it.
        raise ValueError(
            "stored CSR is not canonical (nnz=%d); the int32 indices / int64 "
            "indptr layout cannot be sorted in place, use "
            "PHEASY_SM_INDEX_DTYPE=int64 for this file" % (nnz,))
    return matrix


def _report(path, matrix, how):
    total = (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    print("[SM] %s load (%s): shape=%s nnz=%d data=%.2f GB indices=%s (%.2f GB) "
          "indptr=%s total=%.2f GB"
          % (os.path.basename(os.fspath(path)), how, tuple(matrix.shape), matrix.nnz,
             matrix.data.nbytes / 1e9, matrix.indices.dtype,
             matrix.indices.nbytes / 1e9, matrix.indptr.dtype, total / 1e9),
          flush=True)
