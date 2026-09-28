"""GPU (CUDA via PyTorch) backend for pheasy's dense linear algebra.

Drop-in GPU replacements for the dense CPU primitives in core/optimizer.py.
Dense convenience functions take and return NumPy arrays, so the
optimizer's control flow (CV grouping, alpha grids, standardization, debias,
RFE elimination) is unchanged -- only the heavy dense linear algebra moves to
the GPU. The opt-in GpuTwoLevelOperator/GpuTwoLevelLassoCV path instead
keeps both sparse factors and all iterative vectors resident on CUDA; its
normalization, grouped CV, FISTA and KKT math never round-trip host vectors.
Only setup, returned results and the separate optional debias/metrics are host-side.

Activation (in priority order):

1. Optimizer(..., use_gpu=True/False) -- scoped to that instance's fit(): it
   sets the global mode for the duration of fit() and restores it afterward, so
   constructing an Optimizer no longer clobbers a caller's earlier
   set_gpu_mode(...).
2. set_gpu_mode(mode) / PHEASY_USE_GPU env var -- the process-wide switch
   ("0"/"false"/"off" forces CPU, "1"/"true"/"on" forces GPU, unset -> auto).
   Direct gb.* calls (e.g. load_sensing_matrix) follow ONLY this switch and
   never see an Optimizer's use_gpu; likewise Optimizer.predict() outside fit()
   runs under the ambient mode.
3. Auto mode uses the GPU when torch.cuda.is_available() is true.

Tuning knobs:

* PHEASY_GPU_DEVICE -- CUDA device index (default 0 / first visible device);
  read fresh on every device() call (no caching).
* Resident LASSO: PHEASY_GPU_DEVICES="2,3,1" explicitly enables dynamic
  fold scheduling (first device is primary); PHEASY_GPU_NGPU caps the list or,
  alone, selects that many visible cards starting with PHEASY_GPU_DEVICE.
  Unset both for the original single-GPU behavior. IDs are CUDA-visible logical
  indices; each card must fit the full resident factors, not a row shard.

Design notes
------------
* torch.linalg.lstsq on CUDA only supports driver="gels" (QR without
  pivoting). For the SVD-stable least squares that spla.lstsq(driver="gelsd")
  provides, lstsq() below implements the same rcond-thresholded SVD solve
  with torch.linalg.svd (cuSOLVER) -- numerically equivalent to scipy's
  gelsd, and it handles both over- and under-determined systems.
* LASSO / ALASSO run the same Gram-based FISTA as _LassoCVIterative, but with
  x, G = A^T A and b = A^T y resident on the GPU so each iteration is a dense
  BLAS matvec instead of two CPU sparse multiplies.
* RIDGE CV reproduces sklearn RidgeCV(cv=None) generalized CV (leave-one-out
  via the SVD hat-matrix diagonal) so the selected alpha matches the CPU
  control.
"""
import os
import time

import numpy as np


def _lasso_grid_helpers():
    """[FIX P46] shared auto-grid policy, imported lazily.

    core/optimizer.py owns the rule (span floor for overdetermined problems +
    span-independent density); importing it at module scope would tie this
    backend to the optimizer's import order, so resolve it on first use.
    """
    from .optimizer import lasso_grid_min_decades, lasso_alpha_grid
    return lasso_grid_min_decades, lasso_alpha_grid


def _lasso_grid(amax, anchor, decades, nalpha, n_samples, n_features):
    """Grid from the KKT threshold amax down to the P46-floored bottom.

    anchor is the threshold the bottom is measured from (min of the weighted and
    unweighted KKT thresholds for ALASSO, amax itself for plain LASSO).
    """
    min_decades, make_grid = _lasso_grid_helpers()
    dec = min_decades(n_samples, n_features, decades)
    lo = float(anchor) * 10.0 ** (-float(dec))
    if dec > float(decades):
        print("[gpu_resident] alpha grid widened from %.1f to %.1f decades below "
              "the KKT threshold (overdetermined %d x %d; PHEASY_LASSO_GRID_FLOOR=0 "
              "restores the old span): a 4-decade grid pins alpha* to its own "
              "bottom on high-SNR data." % (float(decades), dec, n_samples,
                                            n_features), flush=True)
    return make_grid(lo, float(amax), nalpha)


__all__ = [
    "available",
    "available_memory_bytes",
    "enabled",
    "set_gpu_mode",
    "get_gpu_mode",
    "gpu_mode_from_env",
    "gpu_mode_required",
    "device",
    "lstsq",
    "qr_solve",
    "gpu_tsqr",
    "ridge_solve",
    "gram",
    "top_eigval",
    "predict",
    "GpuLassoCV",
    "GpuRidgeCV",
    "load_sensing_matrix",
    "GpuSparseMV",
    "GpuTwoLevelOperator",
    "GpuTwoLevelLassoCV",
    "iterative_lstsq",
    "iterative_ridge",
]

_torch_mod = None
_mode = None          # None = auto, True = force on, False = force off
_WARNED_UNGROUPED_CV = False   # _make_cv_splits: group_size does not tile rows
_WARNED_NO_SEED = False        # _make_cv_splits: shuffled KFold without a seed


def _torch():
    """Lazily import torch (returns None if unavailable)."""
    global _torch_mod
    if _torch_mod is None:
        try:
            import torch
            _torch_mod = torch
        except Exception:
            _torch_mod = False
    return _torch_mod if _torch_mod is not False else None


def set_gpu_mode(mode):
    """Set the GPU dispatch mode: None (auto), True (force on), False (force off)."""
    global _mode
    _mode = mode


def get_gpu_mode():
    """Return the current process-global dispatch mode (None/True/False)."""
    return _mode


def available():
    """True when torch + a CUDA device are importable."""
    t = _torch()
    if t is None:
        return False
    try:
        return bool(t.cuda.is_available())
    except Exception:
        return False


def available_memory_bytes():
    """Usable VRAM on the current device in bytes (None when unknown).

    mem_get_info is driver-level (cudaMemGetInfo), which counts blocks the torch
    caching allocator has RESERVED-but-not-allocated as "used"; those are
    immediately reusable without a cudaMalloc, so add them back. Otherwise the
    first big fit parks cache in memory, "free" shrinks for the rest of the run,
    and later dense solves silently fall back to the CPU.
    """
    t = _torch()
    if t is None:
        return None
    try:
        if not t.cuda.is_available():
            return None
        dev = device()
        free, _total = t.cuda.mem_get_info(dev)
        reusable = t.cuda.memory_reserved(dev) - t.cuda.memory_allocated(dev)
        return int(free + reusable)
    except Exception:
        return None


class ResidentFootprintError(MemoryError):
    """The pre-flight says the resident factors do not fit this card.

    A distinct type (not a bare MemoryError) so callers can retry with a SHARDED
    layout -- which needs only ~1/G of the memory on each of the SAME cards --
    while still refusing to continue after a genuine upload/allocation failure.
    """


def _norm_workspace_bytes():
    """Workspace col_norms()/normalize() actually allocates.

    One constant for one budget: the pre-flight and col_norms() used to read
    different defaults for PHEASY_GPU_NORM_WORKSPACE_MB (64 MB vs 512 MB), so
    the default standardized path under-reserved ~448 MB -- exactly the margin
    that turns a borderline "fits" verdict into an OOM.
    """
    raw = os.environ.get("PHEASY_GPU_NORM_WORKSPACE_MB", "512")
    value = int(raw)
    if value <= 0:
        raise ValueError("PHEASY_GPU_NORM_WORKSPACE_MB must be positive")
    return value * 1024**2


def _csr_device_bytes(matrix, value_bytes, n_shards=1):
    """CUDA CSR bytes for ONE card's copy of a scipy matrix (or of one shard).

    The index width is decided PER BLOCK, exactly as GpuTwoLevelOperator.upload()
    does: a block with fewer than 2**31 nonzeros is cast to int32.  Sharding a
    3.57e9-nnz float32 matrix therefore still uses 4-byte indices on each card,
    while a SINGLE replica of it genuinely needs int64.  Reading the width off
    the global nnz made the c3=5.0 sharded pre-flight 1.43x too large (22.5e9 B
    against the operator's real 16.9e9 B) and rejected GPU-sized fits.
    """
    nnz = int(matrix.nnz)
    shards = max(1, int(n_shards))
    per_block = -(-nnz // shards)          # ceil
    index_bytes = 4 if per_block < 2**31 else 8
    return ((int(value_bytes) + index_bytes) * nnz
            + 8 * (sum(int(x) for x in matrix.shape) + 2))


def resident_twolevel_estimate(A, device_id=None, extra_workspace_bytes=0, n_shards=1):
    """Estimate the resident two-level footprint WITHOUT uploading anything.

    n_shards > 1 describes a SHARDED resident operator: SM_prime is split by
    rows (matvec) and by columns (adjoint) across that many devices, so each
    card holds ~2*nnz_total/n_shards nonzeros instead of the whole matrix.

    Pre-flighting matters because the resident backend refuses an oversized
    input only when the operator is built -- after the sensing matrix has been
    materialised and the factors loaded (measured: a third-order fit died 2
    minutes in with "estimate 29.5 GB exceeds budget 19.9 GB", long after a
    3.6 GB SM_prime had been written).  The formula below is exactly the one
    GpuTwoLevelOperator uses for its own check.

    Returns (estimated_peak_bytes, budget_bytes or None, free_bytes or None,
    fraction).  A None budget means the device budget could not be queried;
    callers must treat that as unknown, not as fits.
    """
    import scipy.sparse as _sp
    base = getattr(A, "_twolevel_base", A)
    if not (hasattr(base, "SM_prime") and hasattr(base, "NS")):
        raise TypeError("resident_twolevel_estimate requires TwoLevelSM")
    if (not np.isfinite(extra_workspace_bytes) or extra_workspace_bytes < 0
            or int(extra_workspace_bytes) != extra_workspace_bytes):
        raise ValueError("extra_workspace_bytes must be a nonnegative integer")
    fraction = float(os.environ.get("PHEASY_GPU_MEM_FRACTION", "0.8"))
    if not 0 < fraction <= 1:
        raise ValueError("PHEASY_GPU_MEM_FRACTION must be in (0, 1]")
    # Bytes per nonzero follow the FACTOR dtype instead of assuming float64 +
    # int64.  A float32 sensing matrix (PHEASY_SM_DTYPE=float32, the production
    # setting) was upcast to float64 on the device, which doubled the values and
    # the indices -- that is exactly what made the resident backend report 29.5 GB
    # for a 6.5 GB factor and refuse every fit that the sharded/GPU-SM paths run
    # happily.  int32 indices are used while nnz fits in int32.
    _prime = base.SM_prime
    _value_bytes = 4 if getattr(_prime, "dtype", None) is not None and _prime.dtype == np.float32 else 8
    n_shards = max(1, int(n_shards))
    # Per-card device bytes: SM_prime is sharded (int32 indices per BLOCK),
    # NS is replicated whole (its own width).  The old code multiplied ONE
    # global (_value_bytes + _index_bytes) by the FULL nnz and divided by
    # n_shards, so the c3=5.0 sharded estimate came out 22.5e9 B where the
    # operator's own per-block budget is 16.9e9 B (1.43x), and every float32
    # sharded fit was pre-flighted against a number the GPU cannot reach.
    if _sp.issparse(_prime):
        _prime_bytes = _csr_device_bytes(_prime, _value_bytes, n_shards)
    else:
        _prime_bytes = _value_bytes * int(np.prod(_prime.shape))
    if _sp.issparse(base.NS):
        # NS and NS.T both live on the PRIMARY card (the sharded layout keeps the
        # two-level matvec local), and the peak is compared against every card.
        _ns_bytes = 2 * _csr_device_bytes(base.NS, _value_bytes, 1)
    else:
        _ns_bytes = _value_bytes * int(np.prod(base.NS.shape))
    # Each shard holds one ROW block and one COLUMN block of SM_prime (matvec
    # needs rows, the adjoint needs columns), hence the factor of 2; each pair
    # also needs the same 10% SpMV workspace _cuda_spmv_block_budget reserves.
    _pair = 2.0 * _prime_bytes / n_shards
    _spmv_workspace = max(256 << 20, int(np.ceil(_pair)) // 10)
    peak = (int(np.ceil(_pair)) + _spmv_workspace + _ns_bytes
            + _norm_workspace_bytes() + 8 * 32 * sum(base.shape)
            + int(extra_workspace_bytes))
    if device_id is None:
        # Budget the card the fit will actually UPLOAD to, which is
        # _resident_cv_devices()[0] == PHEASY_GPU_DEVICES[0], not
        # PHEASY_GPU_DEVICE / cuda:0.  With PHEASY_GPU_DEVICES="2,3" the old
        # code measured card 0 and then uploaded to card 2: the decision could
        # abort on a card that was never used, or miss the real overflow.
        try:
            device_id = _resident_cv_devices()[0]
        except Exception:
            device_id = device()
    # None means "cannot query" (no torch/CUDA, or the card is unqueryable);
    # callers treat None as unknown, never as "fits".
    free = _device_free_bytes(device_id)
    # [FIX R1] Budget the memory this fit could actually use, not the leftover
    # AFTER this process has already taken what it needs.  The residency decision
    # is taken more than once per run and the operator may already be resident when
    # it is re-taken; the live factors then count as used and shrink the budget.
    # Measured at c3=5.0: the identical estimate 88029528920 bytes was compared
    # against 23643429273 (22.02 GiB, before the upload) and then against
    # 2505749427 (2.33 GiB, after ~21 GB/card of our own sharded factors were
    # resident), and the second verdict demoted a GPU-sized CV to CPU FISTA.
    # `own` is this process own live device memory -- the very factors this
    # estimate describes -- so it is headroom here, not another user memory.
    available = _device_available_bytes(device_id)
    budget = None if available is None else int(available * fraction)
    return peak, budget, free, fraction


def resident_twolevel_error_message(peak, budget):
    """Actionable text for a resident-LASSO footprint overflow."""
    return ("Resident two-level GPU estimate %d bytes exceeds budget %d bytes; no CPU fallback. "
            "The resident backend must hold SM_prime AND NS (plus their CSR transposes) on ONE "
            "device, so a large third-order problem can exceed a consumer card. Remedies: set "
            "PHEASY_GPU_LASSO_RESIDENT=0 to use the two-level GPU-SM matvec instead (SM_prime "
            "sharded across PHEASY_GPU_SM_NGPU cards, the path validated on this host), or "
            "reduce the third-order cutoff, or use a card with more memory." % (peak, budget))


def resident_device_ids():
    """Device list for a resident (device-side) operator: single or sharded.

    Priority: PHEASY_GPU_DEVICES / PHEASY_GPU_NGPU (the documented resident
    knobs) > PHEASY_GPU_SM_DEVICES (the GPU-SM knob the production templates
    already export, so a sharded resident solve needs no new variable) >
    the single device().
    """
    raw = os.environ.get("PHEASY_GPU_DEVICES", "").strip()
    count = os.environ.get("PHEASY_GPU_NGPU", "").strip()
    if raw or count:
        return [d.index for d in _resident_cv_devices()]
    sm = os.environ.get("PHEASY_GPU_SM_DEVICES", "").strip()
    if sm:
        try:
            ids = [int(x) for x in sm.split(",") if x.strip() != ""]
        except ValueError:
            raise ValueError("PHEASY_GPU_SM_DEVICES must contain integer device IDs") from None
        if ids:
            return list(dict.fromkeys(ids))
    dev = device()
    return [dev.index if dev.index is not None else 0]


def _canonical_csr(matrix):
    """tocsr + sum_duplicates + sort_indices on the HOST (cheap, one pass)."""
    import scipy.sparse as sp
    if not sp.issparse(matrix):
        return matrix
    csr = matrix.tocsr(copy=True)
    csr.sum_duplicates()
    csr.sort_indices()
    return csr


def _mem_trace(tag):
    """Host RSS/VmHWM after a phase, when PHEASY_MEM_TRACE=1.

    The 129 GB host peak of the c3=5.0 resident phase was invisible until the
    kernel OOM-killed it.  VmHWM is the only number that says how close the host
    came to the limit, and the per-phase deltas say WHICH copy caused it.  Off by
    default: one /proc read per phase.
    """
    if os.environ.get("PHEASY_MEM_TRACE", "0").lower() not in ("1", "true", "yes", "on"):
        return
    try:
        rss = hwm = float("nan")
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1048576.0
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1]) / 1048576.0
        print("[mem] %-34s RSS=%.2f HWM=%.2f GiB" % (tag, rss, hwm), flush=True)
    except Exception:
        pass


def _factor_host_bytes(matrix):
    """Resident host bytes of a sparse factor (None/0 for a dense or missing one)."""
    if matrix is None or not hasattr(matrix, "data"):
        return 0
    total = int(matrix.data.nbytes)
    for name in ("indices", "indptr"):
        arr = getattr(matrix, name, None)
        if arr is not None:
            total += int(arr.nbytes)
    return total


def _eager_adjoint_fits(prime_c):
    """Is there host room for the eager transpose ON TOP of what is resident?

    The eager adjoint adds one more full copy of prime's arrays.  Ask for twice
    that (so the run still has room to finish) plus the same headroom the factor
    retention rule uses.  Unknown MemAvailable means NO -- never guess.
    """
    avail = _host_available_bytes()
    if avail is None:
        return False
    return avail >= 2 * _factor_host_bytes(prime_c) + _HOST_FACTOR_CACHE_HEADROOM


class _AdjointBlocks:
    """Row blocks of prime.T, built on demand instead of materialised.

    prime.T of a canonical CSR is a CSC *view*: scipy shares indptr, indices and
    data (test_scipy_adjoint_view_is_shared asserts the sharing), so HOLDING it
    costs nothing.  Every consumer of the adjoint wants row blocks -- one per
    shard for the sharded resident layout, or the whole matrix for a single
    replica -- and each block costs one O(nnz) filtered pass plus the block
    itself (measured 1.8 s/block at nnz 2.2e8, i.e. ~2 min for all four shards at
    the c3=5.0 size).  Materialising the whole transpose up front instead costs a
    FULL second copy of the factors (42.84 GB at c3=5.0) held for the entire host
    phase; together with the canonical copy, that was the 129 GB peak that made
    the c3=5.0 resident fit need a fully idle box.

    The on-disk adjoint cache (PHEASY_HT_CACHE_DIR) remains the explicit opt-in
    for paying that transpose once across runs.
    """

    __slots__ = ("_prime", "_view", "shape", "dtype", "nnz")

    def __init__(self, prime_c):
        import scipy.sparse as sp
        if not sp.issparse(prime_c):
            raise TypeError("_AdjointBlocks requires a sparse prime")
        self._prime = prime_c
        self._view = prime_c.T
        # A scipy that copied here would silently double the host footprint, so
        # fail loudly instead of pretending the fix applied.
        if not (np.shares_memory(self._view.data, prime_c.data)
                and np.shares_memory(self._view.indices, prime_c.indices)):
            raise RuntimeError(
                "scipy materialises csr.T (buffers are not shared), so the lazy "
                "adjoint cannot save host memory; set PHEASY_TWOLEVEL_ADJOINT=eager")
        self.shape = tuple(int(x) for x in self._view.shape)
        self.dtype = prime_c.dtype
        self.nnz = int(prime_c.nnz)

    @property
    def resident_bytes(self):
        """Extra resident host bytes: zero -- the view shares prime's arrays."""
        return 0

    def block(self, c0, c1):
        """Canonical CSR for rows [c0, c1) of the adjoint (a PRIVATE copy)."""
        b = self._view[int(c0):int(c1)].tocsr()
        if not b.has_canonical_format:
            b.sum_duplicates()
            b.sort_indices()
        return b

    def materialize(self):
        """The whole adjoint as a canonical CSR (one full copy; single replica only)."""
        adj = self._view.tocsr()
        if not adj.has_canonical_format:
            adj.sum_duplicates()
            adj.sort_indices()
        return adj

    def toarray(self):
        """Dense adjoint. Diagnostics/tests ONLY -- 1e11 elements at c3=5.0."""
        return self.materialize().toarray()

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("the lazy adjoint supports step-1 row slices only")
        c0, c1, _ = key.indices(self.shape[0])
        return self.block(c0, c1)

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return "<lazy adjoint blocks shape=%s nnz=%d>" % (self.shape, self.nnz)


def _make_adjoint(prime_c, mode):
    """The adjoint for a canonical sparse prime: lazy blocks or an eager CSR."""
    import scipy.sparse as sp
    if not sp.issparse(prime_c):
        return None
    return _prime_adjoint(prime_c) if mode == "eager" else _AdjointBlocks(prime_c)


_SPARSE_IO = None


def _sparse_io():
    """core.sparse_io, imported lazily (numpy/scipy/zipfile only, no cycle)."""
    global _SPARSE_IO
    if _SPARSE_IO is None:
        try:
            from . import sparse_io as _mod
        except (ImportError, ValueError):
            try:
                from pheasy_gpu.core import sparse_io as _mod
            except (ImportError, ValueError):
                # Loaded standalone (dev tools use spec_from_file_location).
                import importlib.util as _ilu
                _path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "sparse_io.py")
                _spec = _ilu.spec_from_file_location("sparse_io", _path)
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
        _SPARSE_IO = _mod
    return _SPARSE_IO


def _canonical_block(matrix):
    """Canonicalize a block that is ALREADY a private copy (a fresh slice).

    A row slice of a canonical CSR is sorted but not marked canonical, and the
    old call sites passed canonical=False, which paid ANOTHER full block copy
    (tocsr(copy=True)) per shard just to re-canonicalise it.  The slices this is
    called on are freshly allocated by scipy's _get_submatrix (copy=True), so
    sum_duplicates()/sort_indices() can run in place.
    """
    import scipy.sparse as sp
    if not sp.issparse(matrix):
        return matrix
    block = matrix.tocsr()          # no copy: already CSR
    # canonical_format() is scipy's check without the int64 upcast that a
    # narrow (int32 indices + int64 indptr) block would otherwise pay.
    _sio = _sparse_io()
    if not _sio.canonical_format(block):
        _sio.canonicalize(block)
    return block


def _balanced_shard_edges(nnz_per_index, n_shards):
    """Split boundaries that equalise NONZEROS per shard, not index counts.

    np.linspace splits by row/column COUNT, which is only fair when the nonzeros
    are spread evenly.  The c3=5.0 factors are not: shard 0 of the uniform split
    needed 19.00 GiB of CUDA CSR for two blocks whose average share was 22.54 GB
    of host factors, i.e. the first quarter of the rows AND columns carried most
    of the matrix.  A single fat shard then decides both the host transient and
    the per-card VRAM (and makes the even-split footprint estimate a lie).
    Equal-nnz boundaries keep every shard at ~nnz/G.
    """
    n = int(len(nnz_per_index))
    if n_shards <= 1 or n == 0:
        return np.array([0, n], dtype=np.int64)
    counts = np.asarray(nnz_per_index, dtype=np.int64)
    cum = np.cumsum(counts)
    total = int(cum[-1])
    edges = [0]
    for k in range(1, int(n_shards)):
        target = total * k // int(n_shards)
        e = int(np.searchsorted(cum, target, side="left")) + 1
        e = max(e, edges[-1] + 1)
        e = min(e, n - (int(n_shards) - k))
        edges.append(e)
    edges.append(n)
    return np.array(edges, dtype=np.int64)


def _col_nnz_counts(indices, n_cols):
    """Per-column nonzero counts of a CSR matrix, chunked (no nnz-sized temp)."""
    counts = np.zeros(int(n_cols), dtype=np.int64)
    step = 1 << 24
    for s in range(0, int(indices.size), step):
        counts += np.bincount(indices[s:s + step], minlength=int(n_cols))
    return counts


def _row_shard_edges(prime, n_shards):
    """Edges over prime's ROWS that equalise nonzeros (matvec blocks)."""
    import scipy.sparse as sp
    if not sp.issparse(prime) or _shard_balance_off():
        return np.linspace(0, prime.shape[0], int(n_shards) + 1).astype(np.int64)
    return _balanced_shard_edges(np.diff(prime.indptr), n_shards)


def _col_nnz_cached(prime):
    """Per-column nonzero counts of prime, computed at most once per object.

    Kept separate from _col_shard_edges because the count-based GPU pre-flight
    needs the same numbers even when the balanced edges are switched off.
    """
    counts = getattr(prime, "_pheasy_col_nnz", None)
    if counts is None or int(counts.size) != int(prime.shape[1]):
        counts = _col_nnz_counts(prime.indices, prime.shape[1])
        try:
            prime._pheasy_col_nnz = counts   # one O(nnz) scan per prime
        except Exception:
            pass
    return counts


def _col_shard_edges(prime, n_shards):
    """Edges over prime's COLUMNS that equalise nonzeros (adjoint blocks)."""
    import scipy.sparse as sp
    if not sp.issparse(prime) or _shard_balance_off():
        return np.linspace(0, prime.shape[1], int(n_shards) + 1).astype(np.int64)
    return _balanced_shard_edges(_col_nnz_cached(prime), n_shards)


def _shard_balance_off():
    return os.environ.get("PHEASY_SHARD_BALANCE", "1").lower() in ("0", "false", "no", "off")


def _canonical_csr_inplace(matrix):
    """Canonicalize WITHOUT a second full copy when that is safe.

    upload() only needs a canonical CSR and canonicalizing is mathematically the
    identity, so the 42.84 GB tocsr(copy=True) that used to sit next to the raw
    factors is pure overhead.  scipy's has_canonical_format is a VERIFIED
    compiled check (csr_has_canonical_format), so an already-canonical factor set
    -- the shipped sm_prime.npz is one -- costs one O(nnz) scan and no allocation
    at all.  Set PHEASY_HOST_FACTOR_INPLACE=0 for the copying behaviour.
    """
    import scipy.sparse as sp
    if not sp.issparse(matrix):
        return matrix
    if os.environ.get("PHEASY_HOST_FACTOR_INPLACE", "1").lower() in ("0", "false", "no", "off"):
        return _canonical_csr(matrix)
    if matrix.format != "csr":
        matrix = matrix.tocsr()
    _sio = _sparse_io()
    if not _sio.canonical_format(matrix):
        _sio.canonicalize(matrix)
    return matrix


def _canonical_twolevel_host(A, adjoint=None):
    """Host-side canonical factors for the resident operator, built ONCE.

    GpuTwoLevelOperator.upload() runs tocsr(copy=True)/sum_duplicates()/
    sort_indices() over the WHOLE SM_prime and NS. Built per card, an N-card
    resident fit pays (N-1) extra full host passes plus (N-1) transient host
    arrays the size of the factors -- the same class of blow-up that OOM-killed
    the large config (exit 137). Build it once, hand it to every replica, and
    drop it as soon as the last card has uploaded.

    Two further host copies used to make the c3=5.0 phase need ~129 GB: the
    canonical tocsr(copy=True) of an already-canonical sm_prime.npz (removed by
    _canonical_csr_inplace) and the eager prime.T.tocsr() adjoint.  The adjoint is
    now lazy by default (see _AdjointBlocks) and only materialised when
    PHEASY_TWOLEVEL_ADJOINT=eager/auto asks for it.
    """
    import scipy.sparse as sp
    base = getattr(A, "_twolevel_base", A)
    _mem_trace("canonical_twolevel_host enter")
    prepared = [_canonical_csr_inplace(base.SM_prime),
                _canonical_csr_inplace(base.NS)]
    _mem_trace("canonical factors (no copy if already canonical)")
    _env_mode = os.environ.get("PHEASY_TWOLEVEL_ADJOINT", "").strip()
    mode = (adjoint or _env_mode or "lazy").strip().lower()
    if adjoint is None and not _env_mode and sp.issparse(prepared[0]):
        # PHEASY_HT_CACHE_DIR exists to pay the transpose ONCE and reuse it from
        # disk, which is meaningless for a lazy view: honour the configured
        # cache by materialising the adjoint (the cache then applies as before).
        _cdir, _cpath = _ht_cache_location(prepared[0])
        if _cpath is not None:
            mode = "eager"
    if mode in ("eager", "full", "materialized", "1", "true", "yes", "on"):
        resolved = "eager"
    elif mode in ("auto",):
        resolved = "eager" if _eager_adjoint_fits(prepared[0]) else "lazy"
    elif mode in ("lazy", "deferred", "view", "0", "false", "no", "off"):
        resolved = "lazy"
    else:
        raise ValueError(
            "PHEASY_TWOLEVEL_ADJOINT must be lazy (default), eager or auto; got %r" % mode)
    prepared.append(_make_adjoint(prepared[0], resolved))
    _mem_trace("adjoint policy=%s (resolved %s)" % (mode, resolved))
    return tuple(prepared)


def _ht_cache_location(prime):
    """(dir, path) for the cached adjoint, or (None, None) when caching is off.

    The transpose is the expensive host step of a resident upload (measured 155.2 s
    for nnz 1.127e9 on the MgC factors) and it is the SAME matrix for every fit that
    reuses the same sensing matrix, so it is worth keeping on disk.  The cache is
    keyed by an IDENTITY THE CALLER KNOWS -- the file the factors came from plus its
    size and mtime (PHEASY_HT_CACHE_KEY) -- because the matrix itself is already in
    memory by then and hashing 9 GB to name the cache would cost more than the
    transpose it is meant to save.  Caching stays OFF unless PHEASY_HT_CACHE_DIR is
    set (9 GB per entry) and PHEASY_HT_CACHE=0 can veto it.
    """
    import scipy.sparse as sp
    cdir = os.environ.get("PHEASY_HT_CACHE_DIR", "").strip()
    key = os.environ.get("PHEASY_HT_CACHE_KEY", "").strip()
    if not cdir or not key:
        return None, None
    if os.environ.get("PHEASY_HT_CACHE", "1").lower() in ("0", "false", "no", "off"):
        return None, None
    if not sp.issparse(prime):
        return None, None
    import hashlib
    tag = hashlib.sha256(("%s|%s|%s|%s" % (key, prime.shape, prime.nnz, prime.dtype)).encode()).hexdigest()[:24]
    try:
        os.makedirs(cdir, exist_ok=True)
    except Exception:
        return None, None
    return cdir, os.path.join(cdir, "prime_t_%s.npz" % tag)


def _prime_adjoint(prime):
    """Canonical prime.T, from the disk cache when it is there, else transposed once.

    Only the three CSR arrays are stored, uncompressed (np.savez): compression of a
    9 GB factor set would cost more CPU than the transpose it replaces, and the load
    is what the cache is for.  A cache entry is trusted only if its shape and nnz
    match the matrix in hand, so a stale or truncated file is rebuilt rather than
    uploaded as the wrong operator.
    """
    import scipy.sparse as sp
    _cdir, path = _ht_cache_location(prime)
    if path is not None and os.path.exists(path):
        try:
            t0 = time.monotonic()
            with np.load(path) as z:
                indptr, indices, data = z["indptr"], z["indices"], z["data"]
                shape = tuple(int(x) for x in z["shape"])
            # The entry holds the ADJOINT, so its shape is prime.shape reversed.
            _want = tuple(reversed(tuple(prime.shape)))
            if shape == _want and len(data) == int(prime.nnz):
                adj = sp.csr_matrix((data, indices, indptr), shape=shape)
                print("[host] adjoint from disk cache %s (%.1fs, nnz=%d)"
                      % (os.path.basename(path), time.monotonic() - t0, adj.nnz), flush=True)
                return adj
            print("[host] ignoring stale adjoint cache (shape %s vs %s, nnz %d vs %d)"
                  % (shape, tuple(prime.shape), len(data), int(prime.nnz)), flush=True)
        except Exception as exc:
            print("[host] ignoring unreadable adjoint cache %s: %s"
                  % (os.path.basename(path), exc), flush=True)
    _t0 = time.monotonic()
    adj = prime.T.tocsr()
    print("[host] built adjoint host-side in %.1fs (nnz=%d%s)"
          % (time.monotonic() - _t0, adj.nnz,
             ", disk cache off" if path is None else ""), flush=True)
    if path is not None:
        try:
            t0 = time.monotonic()
            # np.savez appends ".npz" when the name lacks it, so the temporary
            # name must carry the extension or os.replace cannot find it.
            tmp = path + ".tmp-%d.npz" % os.getpid()
            np.savez(tmp, indptr=adj.indptr, indices=adj.indices, data=adj.data,
                     shape=np.asarray(adj.shape))
            os.replace(tmp, path)
            print("[host] wrote adjoint cache %s (%.1fs, %.2f GiB)"
                  % (os.path.basename(path), time.monotonic() - t0,
                     (adj.data.nbytes + adj.indices.nbytes) / 2**30), flush=True)
        except Exception as exc:
            print("[host] could not write adjoint cache: %s" % exc, flush=True)
    return adj


# Smallest canonical factor set worth retaining on the operator: below this the
# extra transpose is cheap to rebuild, so holding ~2x the factors in host RAM buys
# nothing (see twolevel_host_factors).
_HOST_FACTOR_CACHE_MIN_BYTES = 256 * 1024 ** 2
# Free host memory that must remain AFTER the retained factors, for the fit itself
# (host factor copies, CV index arrays, metrics).  Retention is not allowed to eat
# the room the run needs to finish.
_HOST_FACTOR_CACHE_HEADROOM = 8 * 1024 ** 3


def _host_available_bytes():
    """MemAvailable from /proc/meminfo in bytes, or None if unreadable."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def twolevel_host_factors(A, cache=None):
    """Canonical host factors for a resident operator, reused from A when kept.

    GpuTwoLevelOperator built from RAW factors pays, PER SHARD, one O(nnz)
    column-slice transpose for the adjoint (prime[:, c0:c1].T).  The resident
    LASSO path already avoids that by handing every replica the canonical set
    from _canonical_twolevel_host(), but RIDGE / OLS / the column-norm pass built
    their own operator from the raw factors.  Measured on the MgC operator
    (454656x69487, nnz(SM_prime) 1.127e9, 2 shards, one idle pair of cards): the
    per-shard transposes cost 251.7 s, one canonical copy plus transpose 167.6 s
    (of which .T.tocsr() 155.2 s and the tocsr copy 16.8 s), and __init__ as a
    whole was 456.9 s of a 1298.8 s full-scale RIDGE fit.  So duplication of the
    adjoint transpose -- not the upload -- is what scales with the shard count:
    ~84 s at 2 shards, ~460 s at 5, for a fit whose solves are ~350 s.  The
    remaining seconds are per-shard tocsr/dedupe/sort and the H2D copies, which
    the canonical factors do not remove.

    Keeping the canonical set costs a full copy of the factors plus the
    transpose in host RAM (measured 18.2 GiB for that operator), so retention is
    a HOST MEMORY decision, not a speed one: it is stored on the operator -- so
    that a second operator over the same factors (RIDGE then OLS, or a repeated
    fit) uploads with no host pass at all -- only while MemAvailable can afford
    roughly twice what it adds.  PHEASY_HOST_FACTOR_CACHE=0 disables retention,
    =1 forces it; the default ("auto") keeps it only under that headroom.

    Returning the factors is always safe; only the caching is conditional.
    """
    base = getattr(A, "_twolevel_base", A)
    cached = getattr(base, "_canonical_host_factors", None)
    if cached is not None:
        return cached
    factors = _canonical_twolevel_host(base)
    want = os.environ.get("PHEASY_HOST_FACTOR_CACHE", "auto") if cache is None else cache
    if isinstance(want, str):
        _w = want.strip().lower()
        if _w in ("0", "false", "no", "off"):
            keep = False
        elif _w in ("1", "true", "yes", "on"):
            keep = True
        else:
            added = 0
            for m in factors:
                if m is None:
                    continue
                if hasattr(m, "data"):
                    added += int(m.data.nbytes) + int(
                        getattr(m, "indices", np.zeros(0, dtype=np.int32)).nbytes)
                else:
                    # A lazy adjoint (_AdjointBlocks) shares prime's buffers, so
                    # retaining it adds nothing and the headroom rule below must
                    # not pretend it is another full factor set.
                    added += int(getattr(m, "resident_bytes", 0))
            avail = _host_available_bytes()
            # Retention only pays when the factor set is big enough for the
            # duplicate transpose to be worth holding at all, and when the host can
            # afford the copy PLUS the headroom the rest of the fit still needs.
            # Below the floor the transpose is cheap to rebuild, so there is
            # nothing to keep; without the headroom term a host that is merely
            # large enough for twice the factors would keep them and then run the
            # fit itself out of memory.
            keep = bool(added >= _HOST_FACTOR_CACHE_MIN_BYTES
                        and avail is not None
                        and avail >= 2 * added + _HOST_FACTOR_CACHE_HEADROOM)
            if keep:
                print("[host] keeping canonical resident factors (%.2f GiB; "
                      "MemAvailable %.1f GiB)" % (added / 2**30, avail / 2**30),
                      flush=True)
    else:
        keep = bool(want)
    if keep:
        try:
            base._canonical_host_factors = factors
        except Exception:
            pass
    return factors


def _device_free_bytes(dev):
    """Free (usable) VRAM in bytes on a specific CUDA device (None if unknown).

    Same accounting as available_memory_bytes() (adds back torch's
    reserved-but-unallocated cache) but for an arbitrary device index.
    """
    t = _torch()
    if t is None:
        return None
    try:
        if not t.cuda.is_available():
            return None
        free, _total = t.cuda.mem_get_info(dev)
        reusable = t.cuda.memory_reserved(dev) - t.cuda.memory_allocated(dev)
        return int(free + reusable)
    except Exception:
        return None


def _device_own_bytes(dev):
    """Device bytes this process currently holds as live tensors (None if unknown)."""
    t = _torch()
    if t is None:
        return None
    try:
        if not t.cuda.is_available():
            return None
        return int(t.cuda.memory_allocated(dev))
    except Exception:
        return None


def _device_available_bytes(dev):
    """Bytes this process could use on `dev`, counting what it already holds.

    _device_free_bytes() answers "how much room is left for someone else", which
    is the wrong question once this process's own operator is resident: the live
    factors are reported as used.  The residency checks below are re-taken while
    the factors they describe may already be uploaded, so they must add this
    process's own live bytes back; otherwise the identical fit is called too big
    the second time it is asked.  Memory held by OTHER processes stays excluded:
    it is in neither the free pool nor our own allocation.
    """
    free = _device_free_bytes(dev)
    if free is None:
        return None
    own = _device_own_bytes(dev)
    return free if own is None else free + own


def _multi_gpu_devices(min_free_bytes=0):
    """Device indices for fold-parallel CV, filtered by free VRAM.

    PHEASY_GPU_DEVICES="1,2,4" pins an explicit list (caller order preserved,
    duplicates removed -- two folds on one card would break the documented
    "one fold per card" memory budget); otherwise all visible devices are
    considered. A device whose usable VRAM cannot be measured, or is below
    min_free_bytes, is dropped, so a busy or dead shared GPU is skipped.

    If nothing qualifies the result is EMPTY and the caller must fail closed.
    (An empty list used to be turned back into device(), i.e. straight back to
    a card that had just failed the memory test; and an unavailable CUDA was
    reported as device 0, which need not exist at all.)
    """
    import torch
    if not available():
        return []
    n = torch.cuda.device_count()
    if n <= 1:
        enough = min_free_bytes <= 0 or (_device_free_bytes(0) or 0) >= min_free_bytes
        return [0] if enough else []
    raw = os.environ.get("PHEASY_GPU_DEVICES", "").strip()
    if raw:
        devs = [int(x) for x in raw.split(",") if x.strip() != ""]
    else:
        devs = list(range(n))
    devs = list(dict.fromkeys(d for d in devs if 0 <= d < n))
    if min_free_bytes > 0:
        devs = [d for d in devs
                if (_device_free_bytes(d) or 0) >= min_free_bytes]
    return devs


def gpu_mode_from_env():
    """Return the public GPU mode: auto, cpu, or required."""
    raw = os.environ.get("PHEASY_GPU_MODE")
    if raw is None:
        raw = os.environ.get("PHEASY_USE_GPU")
        if raw is None:
            return "auto"
        return "required" if raw.lower() in ("1", "true", "yes", "on") else "cpu"
    mode = raw.strip().lower()
    if mode not in ("auto", "cpu", "required"):
        raise ValueError("PHEASY_GPU_MODE must be auto, cpu, or required")
    return mode

def gpu_mode_required():
    return gpu_mode_from_env() == "required"

def _env_wants():
    mode = gpu_mode_from_env()
    return None if mode == "auto" else mode == "required"


def enabled():
    """Whether GPU dispatch should be used right now."""
    if _mode is not None:
        want = _mode
    else:
        want = _env_wants()
        if want is None:
            want = True        # auto: use GPU when available
    if (gpu_mode_required() or _mode is True) and not available():
        raise RuntimeError("GPU mode is required but CUDA is unavailable")
    return bool(want) and available()


def device():
    """CUDA device, read fresh from PHEASY_GPU_DEVICE each call (no caching).

    An explicit PHEASY_GPU_DEVICE is validated against the VISIBLE device
    count, like _resident_cv_devices() already did; otherwise "cuda:9" on a
    4-card box sailed through every entry point and then died deep inside
    torch with an "Invalid device ordinal" that names nothing.
    """
    import torch
    dev = os.environ.get("PHEASY_GPU_DEVICE", None)
    if dev is None:
        return torch.device("cuda:0")
    try:
        idx = int(dev)
    except ValueError:
        raise ValueError("PHEASY_GPU_DEVICE must be an integer device index, got %r" % (dev,)) from None
    if idx < 0:
        raise ValueError("PHEASY_GPU_DEVICE must be nonnegative, got %d" % idx)
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n and idx >= n:
            raise ValueError("PHEASY_GPU_DEVICE must name a visible CUDA device "
                             "(0..%d of %d visible), got %d" % (n - 1, n, idx))
    return torch.device("cuda:%d" % idx)


def _dtype():
    """Torch dtype used by the backend -- always float64.

    A float32 mode (PHEASY_GPU_DTYPE=float32) was removed: it only affected a
    few entry points (gram / predict / top_eigval) while the dense solvers and
    CV classes stayed float64, silently mixing precisions and breaking the
    ~1e-7 agreement with the CPU reference. The backend is uniformly float64.
    """
    import torch
    return torch.float64


def _to_torch(A, dtype=None):
    import torch
    if dtype is None:
        dtype = _dtype()
    if hasattr(A, "toarray"):        # scipy sparse -> dense
        A = A.toarray()
    arr = np.ascontiguousarray(A)
    if arr.dtype.kind not in "fc":
        arr = arr.astype(np.float64)
    return torch.as_tensor(arr, dtype=dtype, device=device())


def _to_numpy(t, dtype=np.float64):
    if t is None:
        return None
    if hasattr(t, "detach"):
        t = t.detach().cpu()
    return np.asarray(t, dtype=dtype)


def _csr_major_indices(matrix):
    """Row index of every stored element, in the matrix's OWN index dtype.

    scipy's tocoo() expands indptr through a compiled routine that requires
    indptr and indices to share ONE dtype, so it raises "Output dtype not
    compatible with inputs" for the narrow sensing-matrix layout (int32 column
    indices + int64 indptr, see core/sparse_io.py).  numpy expands the same
    thing and does not care about the mixed widths.
    """
    return np.repeat(np.arange(matrix.shape[0], dtype=matrix.indices.dtype),
                     np.diff(matrix.indptr))


def _is_dense(A):
    return isinstance(A, np.ndarray)


# ---------------------------------------------------------------------------
# CV splits (identical to optimizer._make_cv_splits so GPU and CPU agree)
# ---------------------------------------------------------------------------
def _make_cv_splits(n_samples, cv, random_state=None, group_size=None):
    """Identical to optimizer._make_cv_splits (GPU and CPU CV must agree)."""
    import warnings
    from sklearn.model_selection import GroupKFold, KFold
    global _WARNED_UNGROUPED_CV, _WARNED_NO_SEED
    if cv is None or cv <= 1:
        cv = min(3, n_samples)
    cv = int(cv)
    if group_size and group_size > 1:
        if n_samples % group_size == 0:
            groups = np.arange(n_samples) // group_size
            n_groups = int(groups[-1]) + 1
            if n_groups >= 2:
                eff_cv = int(min(cv, n_groups))
                gkf = GroupKFold(n_splits=eff_cv)
                return list(gkf.split(np.zeros(n_samples, dtype=np.int8),
                                      np.zeros(n_samples, dtype=np.int8), groups))
        elif not _WARNED_UNGROUPED_CV:
            _WARNED_UNGROUPED_CV = True
            warnings.warn(
                "PHEASY_CV_GROUP_SIZE=%s does not divide n_samples=%d: grouped "
                "cross-validation is impossible, so this fit falls back to "
                "shuffled ROW-based KFold.  Rows of one configuration then appear "
                "in both folds, which leaks information and biases alpha*/ridge "
                "toward 0.  Fix PHEASY_CV_GROUP_SIZE (it must be 3*natom and must "
                "divide the row count)." % (group_size, n_samples),
                RuntimeWarning, stacklevel=2)
    cv = max(2, min(cv, n_samples))
    if random_state is None and not _WARNED_NO_SEED:
        _WARNED_NO_SEED = True
        warnings.warn(
            "CV splits use KFold(shuffle=True, random_state=None): the fold "
            "assignment (and therefore alpha*, the CV curve and every reported "
            "score) changes from run to run.  Pass --seed / PHEASY_SEED to make "
            "the fit reproducible.", RuntimeWarning, stacklevel=2)
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    return list(kf.split(np.arange(n_samples)))


# ---------------------------------------------------------------------------
# Core dense solvers
# ---------------------------------------------------------------------------
def lstsq(A, y):
    """SVD-based least squares (== scipy lstsq with driver="gelsd").

    Returns the min-norm solution for under-determined systems, exactly like
    scipy.linalg.lstsq(cond=None). Inputs/outputs are float64 NumPy.
    """
    import torch
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    m, n = A.shape
    At = _to_torch(A, torch.float64)
    yt = _to_torch(y, torch.float64).reshape(-1)
    U, S, Vh = torch.linalg.svd(At, full_matrices=False)
    rcond = max(m, n) * torch.finfo(torch.float64).eps
    cutoff = rcond * S.max()
    Sinv = torch.where(S > cutoff, 1.0 / S, torch.zeros_like(S))
    coef = Vh.T @ (Sinv * (U.T @ yt))
    return _to_numpy(coef, np.float64)




def gpu_tsqr(A, y, block_rows=40000, diag_floor=1e-12):
    """Opt-in float64 binary-tree TSQR with bounded host row blocks.

    Sparse/TwoLevel block assembly is CPU-side; QR and tree reduction are CUDA.
    The complete sensing matrix is never explicitly densified.

    Returns (coef, diagnostics). Only tall full-rank systems use TSQR; callers
    retain SVD semantics for wide or rank-deficient inputs.
    """
    import torch
    import scipy.sparse as sp
    twolevel = hasattr(A, "SM_prime") and hasattr(A, "NS")
    sparse = sp.issparse(A)
    if not (isinstance(A, np.ndarray) or sparse or twolevel):
        raise TypeError("gpu_tsqr requires ndarray, scipy sparse, or TwoLevelSM")
    input_kind = "twolevel" if twolevel else "sparse" if sparse else "dense"
    y = np.asarray(y, dtype=np.float64).ravel()
    if A.ndim != 2 or y.size != A.shape[0] or not np.isfinite(y).all():
        raise ValueError("gpu_tsqr requires finite 2-D A and matching finite y")
    m, n = A.shape
    if m == 0 or n == 0:
        raise ValueError("gpu_tsqr requires nonempty dimensions")
    if not np.isfinite(diag_floor) or diag_floor < 0 or int(block_rows) <= 0:
        raise ValueError("invalid TSQR block_rows or diag_floor")
    if m < n:
        raise np.linalg.LinAlgError("gpu_tsqr requires a tall matrix")
    block_rows = max(int(block_rows), n)
    if not enabled() or not available():
        raise RuntimeError("GPU TSQR requires enabled CUDA")
    fraction = float(os.environ.get("PHEASY_GPU_MEM_FRACTION", "0.8"))
    if not 0 < fraction <= 1:
        raise ValueError("PHEASY_GPU_MEM_FRACTION must be in (0, 1]")
    blocks = (m + block_rows - 1) // block_rows
    # Conservative estimate: current block/Q, merge workspaces, tree and margin.
    estimated_peak = 8 * (4 * min(m, block_rows) * (n + 1) +
                          (blocks.bit_length() + 16) * n * (n + 1)) + 64 * 1024**2
    _free_for_tsqr = available_memory_bytes()
    if _free_for_tsqr is None:
        # available_memory_bytes() returns None when mem_get_info raises (a
        # dead or unqueryable card while torch.cuda.is_available() is still
        # True). None * fraction raised TypeError before, so the intended gate
        # never ran; fail closed like the resident path instead.
        raise RuntimeError("Cannot query GPU memory for TSQR; refusing unchecked upload")
    if estimated_peak > _free_for_tsqr * fraction:
        raise MemoryError("GPU TSQR estimated workspace exceeds memory budget")
    # Normalize only after the workspace gate; never expand whole factors.
    if sparse:
        A = A.tocsr(copy=False)
    prime = ns = None
    if twolevel:
        prime = A.SM_prime.tocsr(copy=False) if sp.issparse(A.SM_prime) else A.SM_prime
        ns = A.NS.astype(np.float64, copy=False)
    stack = []
    xb = yb = q = r = z = ro = zo = rn = zn = diag = coef = None
    try:
        for start in range(0, m, block_rows):
            if twolevel:
                block = prime[start:start + block_rows].astype(np.float64, copy=False) @ ns
            else:
                block = A[start:start + block_rows]
            if sp.issparse(block):
                block = block.toarray()
            block = np.asarray(block, dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError("gpu_tsqr requires finite A")
            xb = _to_torch(block, torch.float64)
            del block
            yb = _to_torch(y[start:start + block_rows], torch.float64)
            q, r = torch.linalg.qr(xb, mode="reduced")
            z = q.T @ yb
            level = 0
            while stack and stack[-1][0] == level:
                _, ro, zo = stack.pop()
                q, r = torch.linalg.qr(torch.cat((ro, r), dim=0), mode="reduced")
                z = q.T @ torch.cat((zo, z), dim=0)
                level += 1
            stack.append((level, r, z))
            del xb, yb, q
        while len(stack) > 1:
            _, ro, zo = stack.pop(0)
            _, rn, zn = stack.pop(0)
            q, r = torch.linalg.qr(torch.cat((ro, rn), dim=0), mode="reduced")
            z = q.T @ torch.cat((zo, zn), dim=0)
            stack.insert(0, (0, r, z))
            del q
        r, z = stack[0][1], stack[0][2]
        diag = r.diagonal().abs()
        dmax = float(diag.max().item()) if diag.numel() else 0.0
        rank_safe = bool(dmax > 0 and float(diag.min().item()) > diag_floor * max(dmax, 1.0))
        if not rank_safe:
            raise np.linalg.LinAlgError("gpu_tsqr detected rank deficiency")
        coef = torch.linalg.solve_triangular(r, z[:, None], upper=True).flatten()
        if not bool(torch.isfinite(coef).all().item()):
            raise np.linalg.LinAlgError("GPU TSQR produced nonfinite coefficients")
        device_name = str(coef.device)
        return _to_numpy(coef, np.float64), {"backend":"gpu_" + input_kind + "_tsqr", "solver":"TSQR", "solver_kind":"direct", "block_assembly":"cpu", "device":device_name, "dtype":"float64", "block_rows":block_rows, "rank_safe":True, "n_blocks":blocks, "estimated_peak_bytes":estimated_peak}
    finally:
        stack.clear()
        xb = yb = q = r = z = ro = zo = rn = zn = diag = coef = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def qr_solve(A, y):
    """QR least squares with an SVD fallback for rank-deficient / wide systems.

    Matches _solve_qr semantics: QR for full-rank tall systems, SVD (min-norm)
    when the system is underdetermined or rank deficient. CUDA gels SILENTLY
    returns NaN/inf for rank-deficient inputs (pytorch#117122), so we first
    factorize R only (mode="r", no Q materialization), check its diagonal with
    the same threshold _solve_qr uses, then run the fast gels solve.
    """
    import torch
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    At = _to_torch(A, torch.float64)
    yt = _to_torch(y, torch.float64).reshape(-1)
    return _to_numpy(_qr_solve_tensor(At, yt), np.float64)


def _qr_solve_tensor(At, yt):
    """Solve from resident float64 tensors; return coefficients on the same device."""
    import torch
    m, n = At.shape

    def svd_fallback():
        # Reuse the uploaded inputs for rank-deficient/wide subsets.
        U, S, Vh = torch.linalg.svd(At, full_matrices=False)
        cutoff = max(m, n) * torch.finfo(torch.float64).eps * S.max()
        inv = torch.where(S > cutoff, S.reciprocal(), torch.zeros_like(S))
        return Vh.T @ (inv * (U.T @ yt))

    if m < n:
        return svd_fallback()          # underdetermined -> min-norm SVD
    try:
        _R = torch.linalg.qr(At, mode="r")
        # mode="r" returns only R, but the container differs across torch
        # versions (named tuple with .R, or a plain (R,) tuple).
        R = _R.R if hasattr(_R, "R") else (_R[-1] if isinstance(_R, tuple) else _R)
        diag = R.diagonal().abs()
        if diag.numel() == 0:
            return svd_fallback()
        dmax = float(diag.max().item())
        if dmax == 0.0:
            return svd_fallback()
        tol = torch.finfo(torch.float64).eps * max(m, n) * dmax
        if float(diag.min().item()) <= tol:
            return svd_fallback()      # rank deficient -> SVD
        coef = torch.linalg.lstsq(At, yt, driver="gels").solution
    except (torch.cuda.OutOfMemoryError, MemoryError):
        raise
    except Exception:
        return svd_fallback()
    if not bool(torch.isfinite(coef).all()):
        return svd_fallback()
    return coef


def ridge_solve(A, y, alpha):
    """min ||A x - y||^2 + alpha ||x||^2 via the normal equations (GPU)."""
    import torch
    if float(alpha) <= 0:
        return lstsq(A, y)
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    At = _to_torch(A, torch.float64)
    yt = _to_torch(y, torch.float64).reshape(-1)
    return _to_numpy(_ridge_solve_tensor(At, yt, alpha), np.float64)


def _ridge_solve_tensor(At, yt, alpha):
    """Positive-alpha Ridge using resident inputs and resident output."""
    import torch
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("resident Ridge requires finite positive alpha")
    n = At.shape[1]
    G = At.T @ At
    b = At.T @ yt
    Gp = G + float(alpha) * torch.eye(n, dtype=torch.float64, device=At.device)
    try:
        L = torch.linalg.cholesky(Gp)
        x = torch.cholesky_solve(b.reshape(-1, 1), L).reshape(-1)
    except (torch.cuda.OutOfMemoryError, MemoryError):
        raise
    except Exception:
        x = torch.linalg.solve(Gp, b)
    return x


def gram(A, y=None):
    """Return G = A^T A (and b = A^T y when y is given) as GPU tensors."""
    At = _to_torch(np.asarray(A, dtype=np.float64))
    G = At.T @ At
    if y is None:
        return G
    yt = _to_torch(np.asarray(y, dtype=np.float64).ravel()).reshape(-1)
    return G, At.T @ yt


def top_eigval(G):
    """Largest eigenvalue of a symmetric PSD matrix (NumPy or GPU tensor in)."""
    import torch
    if isinstance(G, np.ndarray):
        Gt = _to_torch(G, torch.float64)
    else:
        Gt = G
    e = torch.linalg.eigvalsh(Gt)
    return float(e[-1].item())


def predict(A, coef):
    """A @ coef on the GPU (returns float64 NumPy)."""
    At = _to_torch(np.asarray(A, dtype=np.float64))
    ct = _to_torch(np.asarray(coef, dtype=np.float64).ravel()).reshape(-1)
    return _to_numpy(At @ ct, np.float64)


# ---------------------------------------------------------------------------
# FISTA (GPU) -- mirrors optimizer._fista_lasso with a precomputed Gram
# ---------------------------------------------------------------------------
def _power_lipschitz(Gt, power_iters=15):
    """lambda_max(G) -- the Lipschitz constant of the Gram-form gradient.

    The Gram-form objective is 0.5 x^T G x - b^T x; its gradient G x - b has
    Lipschitz constant ||G||_2 = lambda_max(G), so the FISTA step is
    1/lambda_max. The CPU Gram path uses the exact scipy.linalg.eigvalsh(G)[-1]
    (no safety factor); mirror it with a torch eigendecomposition. A previous
    power-iteration version applied the Gram one extra time and returned
    ||G v||^2 = lambda_max^2, shrinking the step by ~lambda_max and stalling
    FISTA inside cv_max_iter. The power_iters argument is kept for signature
    compatibility and ignored.

    Note: eigvalsh is a full O(p^3) decomposition plus a p-by-p workspace,
    called once per CV fold (n_splits + 1 times per fit). That is cheap at
    p=3678 but not at the PHEASY_MAX_DENSE ceiling (p ~ 1e4), so beyond
    PHEASY_LIPSCHITZ_POWER_P (default 8192) this falls back to the power
    iteration described above: the Rayleigh quotient v . (G v) -- NOT
    ||G v||^2 -- times a safety factor. An underestimate there is not fatal:
    _fista_gram()/_fista_twolevel() enforce objective monotonicity and inflate
    L if a step turns out to be too large.
    """
    import torch
    p = int(Gt.shape[0])
    if p == 0:
        return 0.0
    threshold = int(os.environ.get("PHEASY_LIPSCHITZ_POWER_P", "8192"))
    if p <= threshold:
        return float(torch.linalg.eigvalsh(Gt)[-1].item())
    safety = float(os.environ.get("PHEASY_FISTA_LIPSCHITZ_SAFETY", "1.05"))
    if not safety >= 1.0:
        raise ValueError("PHEASY_FISTA_LIPSCHITZ_SAFETY must be >= 1")
    gen = torch.Generator(device=Gt.device).manual_seed(0)
    v = torch.randn(p, generator=gen, dtype=Gt.dtype, device=Gt.device)
    v = v / torch.clamp(v.norm(), min=torch.finfo(Gt.dtype).tiny)
    for _ in range(max(1, int(power_iters))):
        w = Gt @ v
        v = w / torch.clamp(w.norm(), min=torch.finfo(Gt.dtype).tiny)
    rayleigh = float(torch.dot(v, Gt @ v).item())
    return max(rayleigh * safety, 1e-12)


def _soft_threshold_t(x, thr):
    import torch
    # soft-threshold(x, thr) == x - clamp(x, -thr, thr): 2 elementwise ops
    # instead of sign(x)*max(|x|-thr, 0) (5 ops). thr may be a scalar or a
    # per-coordinate vector (penalty weights).
    return x - torch.clamp(x, min=-thr, max=thr)


class _FreeBlockReductionT(object):
    """[HARM_DENSE] torch twin of optimizer._FreeBlockReduction.

    Eliminates the unpenalized block F exactly from (G, b) -- Schur complement
    S = G_PP - G_PF G_FF^-1 G_FP, s = b_P - G_PF G_FF^-1 b_F (Frisch-Waugh-Lovell)
    -- and Jacobi-scales the rest (z = d x_P, d = sqrt(diag S), weights w_P / d).
    Exact reparametrization; FISTA then never iterates the free block, and its
    step is no longer set by the large FC2 columns.
    """

    def __init__(self, Gt, bt, free_t, pw_t):
        import torch
        self.p = int(Gt.shape[0])
        self.F = torch.nonzero(free_t, as_tuple=False).reshape(-1)
        self.P = torch.nonzero(~free_t, as_tuple=False).reshape(-1)
        Gff = Gt.index_select(0, self.F).index_select(1, self.F)
        Gfp = Gt.index_select(0, self.F).index_select(1, self.P)
        Gpp = Gt.index_select(0, self.P).index_select(1, self.P)
        bf = bt.index_select(0, self.F)
        bp = bt.index_select(0, self.P)
        C = beta = None
        try:
            L = torch.linalg.cholesky(Gff)
            C = torch.cholesky_solve(Gfp, L)
            beta = torch.cholesky_solve(bf.unsqueeze(1), L).squeeze(1)
            if not (bool(torch.isfinite(C).all()) and bool(torch.isfinite(beta).all())):
                C = None
        except RuntimeError:
            C = None
        if C is None:
            # CUDA cholesky / gels can fail silently or loudly on a singular
            # block; the rcond-thresholded pseudo-inverse is the safe fallback.
            Gp = torch.linalg.pinv(Gff)
            C = Gp @ Gfp
            beta = Gp @ bf
        S = Gpp - Gfp.T @ C
        S = 0.5 * (S + S.T)
        s = bp - Gfp.T @ beta
        dg = torch.clamp(torch.diagonal(S), min=0.0)
        top = float(dg.max().item()) if dg.numel() else 0.0
        d = torch.where(dg > 1e-300 * max(top, 1.0), torch.sqrt(dg), torch.ones_like(dg))
        self.d = d
        self.C = C
        self.beta = beta
        self.S = S / d[:, None] / d[None, :]
        self.s = s / d
        self.w = pw_t.index_select(0, self.P).to(dtype=S.dtype) / d
        self.lip = _power_lipschitz(self.S)

    def to_full(self, z):
        import torch
        x_p = z / self.d
        x = torch.zeros(self.p, dtype=z.dtype, device=z.device)
        x[self.P] = x_p
        x[self.F] = self.beta - self.C @ x_p
        return x


def _fista_gram(Gt, bt, alpha, x0=None, max_iter=3000, tol=1e-7,
                lipschitz=None, penalty_weights=None, n_samples=None, _info=None):
    """FISTA LASSO on the precomputed Gram: min 0.5||Ax-y||^2 + alpha sum w|x|.

    Mirrors optimizer._fista_lasso (Gram path) on GPU tensors (same fixed point).
    Returns (x, n_iter) with x a GPU tensor.

    The adaptive-restart overshoot check needs one host scalar per iteration;
    on a 3090 that sync (~1 ms) dwarfs the ~0.3 ms Gram matvec, so it is
    evaluated only every PHEASY_FISTA_RESTART_EVERY iterations (default 5).
    The restart test is a heuristic, so the 5-step cadence leaves the fixed
    point (and the converged solution) unchanged while cutting the per-iter
    cost ~3x.
    """
    import torch
    n = Gt.shape[0]
    if x0 is None:
        x = torch.zeros(n, dtype=Gt.dtype, device=Gt.device)
    else:
        x = x0.clone()

    if alpha <= 0:
        coef = torch.linalg.lstsq(Gt, bt, driver="gels").solution
        if not bool(torch.isfinite(coef).all()):
            # CUDA gels silently returns NaN/inf on a rank-deficient Gram
            # (pytorch#117122); fall back to the rcond-thresholded SVD like
            # lstsq() / qr_solve().
            U, S, Vh = torch.linalg.svd(Gt, full_matrices=False)
            rcond = Gt.shape[0] * torch.finfo(Gt.dtype).eps
            cutoff = rcond * S.max()
            Sinv = torch.where(S > cutoff, 1.0 / S, torch.zeros_like(S))
            coef = Vh.T @ (Sinv * (U.T @ bt))
        scale = torch.clamp(bt.abs().max(), min=torch.finfo(bt.dtype).tiny)
        kkt = float(((Gt @ coef - bt).abs().max() / scale).item())
        converged = bool(np.isfinite(kkt) and kkt <= tol)
        if _info is not None:
            _info.update(n_iter=0, converged=converged, kkt_relative=kkt)
        if not converged:
            import warnings
            warnings.warn("FISTA zero-alpha least-squares result lacks stationarity: relative KKT=%g" % kkt,
                          RuntimeWarning, stacklevel=2)
        return coef, 0

    if lipschitz is None:
        lipschitz = _power_lipschitz(Gt)
    lipschitz = max(float(lipschitz), 1e-12)
    step = 1.0 / lipschitz
    # penalty = alpha * n_samples * w is INDEPENDENT of the step size, so the
    # KKT certificate below stays valid when the monotonicity guard raises L.
    if penalty_weights is None:
        penalty = float(alpha) * float(n_samples)
    elif isinstance(penalty_weights, torch.Tensor):
        penalty = float(alpha) * float(n_samples) * penalty_weights.to(
            device=Gt.device, dtype=Gt.dtype)
    else:
        penalty = float(alpha) * float(n_samples) * torch.as_tensor(
            np.ascontiguousarray(penalty_weights, dtype=np.float64),
            dtype=Gt.dtype, device=Gt.device)
    thr_vec = penalty * step
    kkt_scale = torch.clamp(bt.abs().max(), min=torch.finfo(bt.dtype).tiny)

    def kkt_relative(coef):
        gradient = Gt @ coef - bt
        violation = torch.where(coef != 0, (gradient + penalty * coef.sign()).abs(),
                                torch.clamp(gradient.abs() - penalty, min=0.0))
        return float((violation.max() / kkt_scale).item())

    def objective(coef):
        """LASS objective up to the constant 0.5||y||^2 (step-size guard)."""
        gx = Gt @ coef
        return float((0.5 * torch.dot(coef, gx) - torch.dot(bt, coef)
                      + (penalty * coef.abs()).sum()).item())

    converged = False
    kkt = float("inf")
    z = x.clone()
    t = 1.0
    x_prev = x.clone()
    prev_f = None
    n_inflate = 0
    n_blowup = 0
    x_finite = None
    n_iter = 0
    restart_every = int(os.environ.get("PHEASY_FISTA_RESTART_EVERY", "5"))
    restart_every = max(1, restart_every)

    for it in range(int(max_iter)):
        n_iter = it + 1
        grad = Gt @ z
        grad.sub_(bt)
        x_new = _soft_threshold_t(z - step * grad, thr_vec)

        if it % restart_every == 0:
            if float(((z - x_new) * (x_new - x)).sum().item()) > 0.0:
                z = x_new
                t = 1.0
            else:
                t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
                z = x_new + ((t - 1.0) / t_new) * (x_new - x)
                t = t_new
        else:
            t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
            z = x_new + ((t - 1.0) / t_new) * (x_new - x)
            t = t_new
        x = x_new

        if it % 20 == 19:
            if not bool(torch.isfinite(x).all().item()):
                # A step so large that the iterate overflowed.  Restart from the
                # last finite iterate with a much larger L instead of returning
                # NaN coefficients with only a warning (fail loud if even that
                # does not recover).
                n_blowup += 1
                if x_finite is None or n_blowup > 3:
                    raise RuntimeError(
                        "GPU FISTA diverged (nonfinite iterate): the Lipschitz "
                        "estimate is far too small. Raise "
                        "PHEASY_FISTA_LIPSCHITZ_SAFETY or lower "
                        "PHEASY_LIPSCHITZ_POWER_P so the exact estimate is used.")
                x = x_finite.clone()
                z = x.clone()
                x_prev = x.clone()
                t = 1.0
                lipschitz = max(lipschitz * 8.0, 1e-12)
                step = 1.0 / lipschitz
                thr_vec = penalty * step
                prev_f = None
                n_inflate += 1
                continue
            x_finite = x.clone()
            f_new = objective(x)
            if prev_f is not None and f_new > prev_f + (_FISTA_F_REFINE * float(
                    torch.finfo(Gt.dtype).eps)) * (abs(prev_f) + 1e-300):
                # Step too large: _power_lipschitz under-resolved lambda_max(G)
                # (40 power iterations converge from below).  Halve the step,
                # restart the momentum from the current iterate, and re-check.
                # This is what the removed per-step backtracking certificate
                # used to guarantee; sweeping to max_iter instead would return
                # coefficients with converged=False.
                lipschitz = max(lipschitz * 2.0, 1e-12)
                step = 1.0 / lipschitz
                thr_vec = penalty * step
                z = x.clone()
                t = 1.0
                x_prev = x.clone()
                prev_f = None
                n_inflate += 1
            else:
                prev_f = f_new
            dx = (x - x_prev).norm()
            xn = torch.clamp(x.norm(), min=1.0)
            if bool((dx <= tol * xn).item()):
                kkt = kkt_relative(x)
                if np.isfinite(kkt) and kkt <= tol:
                    converged = True
                    break
            x_prev = x.clone()
    if not converged:
        kkt = kkt_relative(x)
        converged = bool(np.isfinite(kkt) and kkt <= tol)
    if _info is not None:
        _info.update(n_iter=n_iter, converged=converged, kkt_relative=kkt,
                     lipschitz=lipschitz, lipschitz_inflations=n_inflate)
    if not converged:
        import warnings
        warnings.warn("FISTA did not converge: iterations=%d, relative KKT=%g, tol=%g"
                      % (n_iter, kkt, tol), RuntimeWarning, stacklevel=2)
    return x, n_iter


# ---------------------------------------------------------------------------
# LASSO CV (GPU FISTA) -- drop-in for _LassoCVIterative
# ---------------------------------------------------------------------------
class GpuLassoCV(object):
    """LASSO over an alpha grid with grouped CV via GPU Gram-based FISTA.

    Public interface matches _LassoCVIterative (and the attributes
    Optimizer.fit / holdout_eval read): coef_, alpha_, alphas_, mse_path_,
    n_iter_, n_features_in_, predict.
    """

    def __init__(self, alphas, cv, tol, max_iter, rand_seed, n_jobs=1,
                 fit_intercept=False, group_size=None, selection="cyclic",
                 penalty_weights=None, grid_diag=None):
        self.alphas = np.sort(np.asarray(alphas, dtype=np.float64))
        self.cv = cv
        self.tol = tol
        self.max_iter = int(max_iter)
        self.rand_seed = rand_seed
        self.n_jobs = int(n_jobs)
        self.fit_intercept = fit_intercept
        self.group_size = group_size
        self.selection = selection
        self.penalty_weights = penalty_weights
        self.grid_diag = grid_diag
        self._lipschitz = None
        self._gram = None          # deliberately None: debias uses GPU lstsq

    def fit(self, A, y, sample_weight=None):
        import torch
        if sample_weight is not None:
            raise NotImplementedError(
                "GpuLassoCV does not support sample_weight; use the CPU path")
        if self.fit_intercept:
            raise NotImplementedError(
                "GpuLassoCV does not support fit_intercept (intercept_ is "
                "always 0); use the CPU path")
        y64 = np.asarray(y, dtype=np.float64).ravel()
        n_samples, m = A.shape
        At = _to_torch(A, torch.float64)
        yt = _to_torch(y64, torch.float64).reshape(-1)

        splits = _make_cv_splits(n_samples, self.cv, self.rand_seed,
                                 self.group_size)
        n_alphas = len(self.alphas)

        # [ACC] convert the penalty weights ONCE per fit, not once per
        # (fold, alpha): with nalpha=100 and cv=5 the old code performed 600
        # host->device transfers plus 600 device allocations of a p-vector
        # inside the hot loop.
        pw = self.penalty_weights
        if pw is not None and not isinstance(pw, torch.Tensor):
            pw = torch.as_tensor(np.ascontiguousarray(pw, dtype=np.float64),
                                 dtype=torch.float64, device=At.device)
        # [HARM_DENSE] zero weight == unpenalized column.  With such a block
        # every solve below runs on the reduced Gram (_FreeBlockReductionT), which
        # carries its own Lipschitz constant, so lambda_max of the unreduced
        # Grams (an eigvalsh per fold) would never be used: skip it.
        _free_t = (pw == 0) if pw is not None else None
        _reduce = _free_t is not None and bool(_free_t.any())

        G_full = At.T @ At
        b_full = At.T @ yt
        lip_full = None if _reduce else _power_lipschitz(G_full)

        gram_folds = []
        lip_folds = []
        A_va_list = []
        for tr, va in splits:
            va_t = torch.as_tensor(np.asarray(va), dtype=torch.long, device=At.device)
            A_va = At[va_t]
            G_va = A_va.T @ A_va
            b_va = A_va.T @ yt[va_t]
            gram_folds.append((G_full - G_va, b_full - b_va))
            lip_folds.append(None if _reduce else _power_lipschitz(G_full - G_va))
            A_va_list.append(A_va)

        cv_tol = float(os.environ.get(
            "PHEASY_CV_TOL", str(max(float(self.tol), 1e-3))))
        cv_max_iter = int(os.environ.get(
            "PHEASY_CV_MAX_ITER", str(min(self.max_iter, 800))))

        mse_path = np.zeros((n_alphas, len(splits)))
        x_folds = [None] * len(splits)
        x_full = None
        best_i = 0
        best_mean = float("inf")
        best_x = None
        _cv_max_n_iter = 0
        # [D1] "hit the cap" must mean "hit the cap WITHOUT converging": the
        # periodic check runs at it % 20 == 19, so a fully converged fit can
        # report n_iter == cv_max_iter and used to be flagged as a convergence
        # problem on a perfectly converged path.
        _cv_hit_cap = False

        red_full = red_folds = None
        if _reduce:
            # [HARM_DENSE] eliminate the unpenalized block exactly from every
            # Gram (each fold from ITS OWN training Gram, so nothing leaks from
            # the validation rows) and Jacobi-scale the rest: FISTA then iterates
            # on the penalized block only, starting from z = 0, which is the
            # exact top-of-grid solution.  See optimizer._FreeBlockReduction for
            # why iterating the free block inside FISTA is too slow on raw
            # FC2/FC3 column scales.
            red_folds = [_FreeBlockReductionT(G_tr, b_tr, _free_t, pw)
                         for G_tr, b_tr in gram_folds]
            red_full = _FreeBlockReductionT(G_full, b_full, _free_t, pw)
            print("[HARM_DENSE] GPU Gram FISTA on the penalized block only: %d "
                  "free columns eliminated exactly (Schur complement), %d "
                  "penalized columns Jacobi-scaled"
                  % (int(_free_t.sum().item()), int((~_free_t).sum().item())),
                  flush=True)

        for a_i in range(n_alphas - 1, -1, -1):
            alpha = float(self.alphas[a_i])
            fold_mse = np.zeros(len(splits))
            for k, (tr, va) in enumerate(splits):
                _fold_info = {}
                if red_folds is not None:
                    _r = red_folds[k]
                    coef, nit = _fista_gram(_r.S, _r.s, alpha, x0=x_folds[k],
                                            max_iter=cv_max_iter, tol=cv_tol,
                                            lipschitz=_r.lip, penalty_weights=_r.w,
                                            n_samples=len(tr), _info=_fold_info)
                    coef_full = _r.to_full(coef)
                else:
                    coef, nit = _fista_gram(gram_folds[k][0], gram_folds[k][1], alpha,
                                            x0=x_folds[k], max_iter=cv_max_iter, tol=cv_tol,
                                            lipschitz=lip_folds[k], penalty_weights=pw,
                                            n_samples=len(tr), _info=_fold_info)
                    coef_full = coef
                va_t = torch.as_tensor(np.asarray(va), dtype=torch.long, device=At.device)
                pred = A_va_list[k] @ coef_full
                x_folds[k] = coef
                _cv_max_n_iter = max(_cv_max_n_iter, nit)
                if nit >= cv_max_iter and not _fold_info.get("converged", False):
                    _cv_hit_cap = True
                err = pred - yt[va_t]
                fold_mse[k] = float((err * err).mean().item())
            mse_path[a_i] = fold_mse
            mean = float(fold_mse.mean())

            _full_info = {}
            if red_full is not None:
                x_full, nit = _fista_gram(red_full.S, red_full.s, alpha, x0=x_full,
                                          max_iter=cv_max_iter, tol=cv_tol,
                                          lipschitz=red_full.lip,
                                          penalty_weights=red_full.w,
                                          n_samples=n_samples, _info=_full_info)
            else:
                x_full, nit = _fista_gram(G_full, b_full, alpha, x0=x_full,
                                          max_iter=cv_max_iter, tol=cv_tol,
                                          lipschitz=lip_full, penalty_weights=pw,
                                          n_samples=n_samples, _info=_full_info)
            # Full-data warm-start solve: feeds the "max iterations" message but
            # must not set the hit-cap flag (that is a CV-convergence diagnosis).
            _cv_max_n_iter = max(_cv_max_n_iter, nit)
            if mean <= best_mean:
                best_mean = mean
                best_i = a_i
                best_x = x_full.clone()

        # tie / edge diagnostics (mirror _LassoCVIterative so holdout flags work)
        mean_path = mse_path.mean(axis=1)
        rtol = float(os.environ.get("PHEASY_LASSO_TIE_RTOL", "1e-9"))
        tied = np.flatnonzero(mean_path <= best_mean * (1.0 + rtol) + 1e-300)
        if tied.size > 1:
            if _cv_hit_cap:
                print("[CV] WARNING: %d alphas tie at CV MSE %.6e (%.3e ... %.3e). "
                      "The CV solver (FISTA, tol=%.0e) hit cv_max_iter (%d) and is "
                      "too loose to separate them -- lower PHEASY_CV_TOL / raise "
                      "PHEASY_CV_MAX_ITER."
                      % (tied.size, best_mean, float(self.alphas[tied].min()),
                         float(self.alphas[tied].max()), cv_tol, cv_max_iter),
                      flush=True)
            else:
                print("[CV] WARNING: %d alphas tie at CV MSE %.6e (%.3e ... %.3e); "
                      "FISTA already converged (max %d iters < %d), so the CV tail "
                      "is genuinely flat."
                      % (tied.size, best_mean, float(self.alphas[tied].min()),
                         float(self.alphas[tied].max()), _cv_max_n_iter,
                         cv_max_iter), flush=True)
        # PHEASY_LASSO_1SE: one-standard-error rule (largest alpha within 1 SE
        # of the CV minimum) -- matches _reselect_alpha on the dense path.
        if os.environ.get("PHEASY_LASSO_1SE", "0").lower() in ("1", "true", "yes"):
            se = float(mse_path[best_i].std(ddof=1) / np.sqrt(mse_path.shape[1])) \
                if mse_path.shape[1] > 1 else 0.0
            cand = np.flatnonzero(mean_path <= best_mean + se)
            best_i = int(cand[np.argmax(self.alphas[cand])])
        self._alpha_at_min = (best_i == 0)
        self._alpha_at_min_flat = (
            self._alpha_at_min and tied.size > 1
            and float(self.alphas[tied].min()) <= float(self.alphas.min()) * (1.0 + 1e-12))
        self._alpha_at_min_hitcap = self._alpha_at_min_flat and _cv_hit_cap
        if self._alpha_at_min:
            if self.grid_diag:
                print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM; %s"
                      % (float(self.alphas[0]), self.grid_diag), flush=True)
            elif self._alpha_at_min_hitcap:
                print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM via the "
                      "tie-break on a FLAT CV tail AND FISTA hit cv_max_iter; this "
                      "is a CONVERGENCE problem (lower PHEASY_CV_TOL / raise "
                      "PHEASY_CV_MAX_ITER), not a model-density conclusion."
                      % float(self.alphas[0]), flush=True)
            elif self._alpha_at_min_flat:
                print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM on a flat "
                      "CV tail, but FISTA already converged (max %d iters): alpha* "
                      "is not well-determined by CV (not a convergence problem)."
                      % (float(self.alphas[0]), _cv_max_n_iter), flush=True)
            else:
                print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM; the CV "
                      "curve is still falling at the low end, so widening the grid "
                      "only pushes alpha* toward OLS. Treat this fit as effectively "
                      "unregularized (compare with OLS/RFE)."
                      % float(self.alphas[0]), flush=True)

        self.alpha_ = float(self.alphas[best_i])
        final_info = {}
        if red_full is not None:
            z_t, nfin = _fista_gram(red_full.S, red_full.s, self.alpha_, x0=best_x,
                                    max_iter=self.max_iter,
                                    tol=float(self.tol),
                                    lipschitz=red_full.lip,
                                    penalty_weights=red_full.w,
                                    n_samples=n_samples, _info=final_info)
            coef_t = red_full.to_full(z_t)
            # certificate of the reduced problem: relative to the penalized
            # block's own gradient, not the FC2-dominated max|A^T y|
            final_info["harm_dense_reduction"] = "schur_complement+jacobi"
        else:
            coef_t, nfin = _fista_gram(G_full, b_full, self.alpha_, x0=best_x,
                                       max_iter=self.max_iter,
                                       tol=float(self.tol),
                                       lipschitz=lip_full, penalty_weights=pw,
                                       n_samples=n_samples, _info=final_info)
        self.coef_ = _to_numpy(coef_t, np.float64)
        self.intercept_ = 0.0
        self.alphas_ = self.alphas
        self.mse_path_ = mse_path
        self.n_iter_ = int(nfin)
        self.regularized_solver_info_ = dict(final_info, solver="GPU FISTA",
                                            backend="gpu_dense_fista",
                                            stage="regularized_refit_before_debias", tol=float(self.tol))
        self.n_features_in_ = m
        return self

    def predict(self, A):
        if isinstance(A, np.ndarray) and enabled():
            return predict(A, self.coef_)
        return np.asarray(A @ self.coef_).ravel()


# ---------------------------------------------------------------------------
# RIDGE CV (GPU) -- drop-in for sklearn RidgeCV(cv=None) generalized CV
# ---------------------------------------------------------------------------
class GpuRidgeCV(object):
    """Ridge CV over an alpha grid via grouped K-fold CV (closed form, GPU).

    [FIX P46/P47] grouped CV -- one torch SVD per fold -- instead of the old
    leave-one-out GCV. LOO leaks the other 3N-1 rows of the same configuration
    into training and biases alpha* toward 0. The Optimizer pre-scales A and y
    by sqrt(weights) for weighted ridge, so fit() takes no sample_weight.
    """

    def __init__(self, alphas, cv=5, rand_seed=None, group_size=None):
        # sort ascending so the CV tie-break leans toward the SMALLEST alpha
        # (matching GpuLassoCV), independent of the caller's grid order.
        self.alphas = np.sort(np.asarray(alphas, dtype=np.float64))
        self.cv = cv
        self.rand_seed = rand_seed
        self.group_size = group_size

    def fit(self, A, y, sample_weight=None):
        import torch
        if sample_weight is not None:
            raise NotImplementedError(
                "GpuRidgeCV: pre-scale A,y by sqrt(weights) before calling")
        y64 = np.asarray(y, dtype=np.float64).ravel()
        A64 = np.ascontiguousarray(A, dtype=np.float64)
        n, m = A64.shape
        splits = _make_cv_splits(n, self.cv, self.rand_seed, self.group_size)
        alphas = self.alphas  # sorted ascending (see __init__)
        mse_path = np.zeros((len(alphas), len(splits)), dtype=np.float64)

        # [multi-GPU] fold-parallel CV: one thread per device, each fold's dense
        # SVD on its device. Memory guard: a device is used only when its usable
        # VRAM covers the fold's ~4x SVD footprint (matching _gpu_footprint_ok);
        # otherwise we fall back to fewer devices (ultimately single). Each
        # device's folds run serially inside its own thread, so the per-device
        # peak is exactly one fold (At + U + Vh + workspace).
        n_train = max(int(len(tr)) for tr, _ in splits)
        footprint = 4 * n_train * m * 8
        frac = float(os.environ.get("PHEASY_GPU_MEM_FRACTION", "0.8"))
        min_free = footprint / frac if frac > 0 else footprint
        devs = _multi_gpu_devices(min_free_bytes=min_free)
        devs = devs[:len(splits)]
        if not devs:
            raise RuntimeError(
                "GPU RIDGE CV: no CUDA device has the %.2f GB free required for one "
                "fold (PHEASY_GPU_DEVICES=%r, PHEASY_GPU_MEM_FRACTION=%g); free a "
                "card or lower the fraction instead of running on a card that "
                "failed the memory test" %
                (footprint / 1e9, os.environ.get("PHEASY_GPU_DEVICES", ""), frac))
        if len(devs) > 1:
            print("[GPU] RIDGE CV fold-parallel on %d device(s); fold footprint "
                  "~%.2f GB" % (len(devs), footprint / 1e9), flush=True)

        # [M2] pre-slice fold arrays in the PARENT: numpy fancy indexing
        # (A64[tr]) holds the GIL, so doing it per-thread serialized the
        # parallel SVD work. Cost: all folds' slices resident in host RAM
        # (~cv x n_train x m x 8; c7 5-fold ~3 GB -- acceptable).
        pre_sliced = [
            (np.ascontiguousarray(A64[tr]), np.ascontiguousarray(y64[tr]),
             np.ascontiguousarray(A64[va]), np.ascontiguousarray(y64[va]))
            for tr, va in splits
        ]

        def _fold_cpu(k):
            Atr, ytr, Ava, yva = pre_sliced[k]
            U, S, Vh = np.linalg.svd(Atr, full_matrices=False)
            Uty = U.T @ ytr
            AvV = Ava @ Vh.T                  # A_va @ V  (Vh = V^H; real -> V^T)
            col = np.empty(len(alphas), dtype=np.float64)
            for j, a in enumerate(alphas):
                pred = AvV @ ((S / (S * S + float(a))) * Uty)
                col[j] = float(((pred - yva) ** 2).mean())
            return col

        def _fold(k, dev):
            Atr, ytr, Ava, yva = pre_sliced[k]
            # [M1] torch's CURRENT device is thread-local and inherits cuda:0;
            # cuSOLVER handles/workspace follow the current device, so pin the
            # thread to the fold's device (avoids wrong-device workspace).
            # [M3] the VRAM guard is a snapshot; on a shared box another job can
            # grab VRAM in between -- fall back to a CPU fold instead of failing
            # the whole CV ("a slow fold beats a dead fit").
            try:
                with torch.cuda.device(dev):
                    d = torch.device("cuda:%d" % dev)
                    At = torch.as_tensor(Atr, dtype=torch.float64, device=d)
                    yt = torch.as_tensor(ytr, dtype=torch.float64, device=d)
                    Avat = torch.as_tensor(Ava, dtype=torch.float64, device=d)
                    yvat = torch.as_tensor(yva, dtype=torch.float64, device=d)
                    U, S, Vh = torch.linalg.svd(At, full_matrices=False)
                    Uty = U.T @ yt
                    AvV = Avat @ Vh.T         # A_va @ V  (Vh = V^H; real -> V^T)
                    col = np.empty(len(alphas), dtype=np.float64)
                    for j, a in enumerate(alphas):
                        pred = AvV @ ((S / (S * S + float(a))) * Uty)
                        col[j] = float(((pred - yvat) ** 2).mean().item())
                    return col
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                if gpu_mode_required() or os.environ.get("PHEASY_GPU_FALLBACK", "0").lower() not in ("1", "true", "yes", "on"):
                    raise RuntimeError("GPU RIDGE CV fold %d failed with fallback disabled: %s" % (k, exc)) from exc
                print("[GPU] RIDGE CV fold %d OOM on cuda:%d; explicit CPU fallback" % (k, dev), flush=True)
                return _fold_cpu(k)

        if len(devs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            folds_by_dev = [[] for _ in devs]
            for k in range(len(splits)):
                folds_by_dev[k % len(devs)].append(k)

            def _run_dev(di):
                out = {}
                for k in folds_by_dev[di]:
                    out[k] = _fold(k, devs[di])
                return out

            with ThreadPoolExecutor(max_workers=len(devs)) as ex:
                for out in ex.map(_run_dev, range(len(devs))):
                    for k, col in out.items():
                        mse_path[:, k] = col
        else:
            for k in range(len(splits)):
                col = _fold(k, devs[0])
                mse_path[:, k] = col

        # Tie-break toward the SMALLEST alpha on a flat CV tail: scan from the
        # largest alpha down and accept `<=`, mirroring GpuLassoCV.  (The old
        # np.argmin took the *first* min in the caller's unsorted order.)
        mean_path = mse_path.mean(axis=1)
        best_i = len(alphas) - 1
        best_mean = float(mean_path[best_i])
        for a_i in range(len(alphas) - 1, -1, -1):
            _m = float(mean_path[a_i])
            if _m <= best_mean:
                best_mean = _m
                best_i = a_i
        self._alpha_at_min = (best_i == 0)
        if self._alpha_at_min:
            print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM; the RIDGE "
                  "CV curve is still falling at the low end -- widening the grid "
                  "only pushes alpha* toward OLS."
                  % float(alphas[0]), flush=True)
        self.alpha_ = float(alphas[best_i])
        # final refit at the selected alpha (closed form, full data, device())
        At = _to_torch(A64, torch.float64)
        yt = _to_torch(y64, torch.float64).reshape(-1)
        U, S, Vh = torch.linalg.svd(At, full_matrices=False)
        Uty = U.T @ yt
        self.coef_ = _to_numpy(Vh.T @ ((S / (S * S + self.alpha_)) * Uty),
                               np.float64)
        self.intercept_ = 0.0
        self.mse_path_ = mse_path
        self.regularized_solver_info_ = {
            "solver": "GPU Ridge SVD", "backend": "gpu_dense",
            "device": str(device()), "dtype": "float64",
            "stage": "regularized_refit_before_metrics"
        }
        self.n_features_in_ = m
        return self

    def predict(self, A):
        if isinstance(A, np.ndarray) and enabled():
            return predict(A, self.coef_)
        return np.asarray(A @ self.coef_).ravel()


# ---------------------------------------------------------------------------
# Sensing-matrix loading (sparse SM_prime @ dense NS on the GPU)
# ---------------------------------------------------------------------------
def load_sensing_matrix(sm_prime, ns_harm, ns_anharm, n_rows, dtype=np.float64):
    """Compute SM = sm_prime @ block_diag(ns_harm, ns_anharm) on the GPU.

    sm_prime is a scipy sparse CSR/CSC matrix (float32 or float64),
    ns_harm / ns_anharm are dense 2-D arrays (or scipy sparse). Returns the
    first n_rows rows as a dense NumPy array of the requested dtype.

    Falls back to a CPU scipy multiply when the GPU is unavailable.
    """
    import torch
    # Slice first: holdout_eval only needs the first n_rows, so do not
    # materialize the full product (5.6x the work at n=8/45) or its footprint.
    sm = sm_prime[:n_rows].tocsr()

    if not enabled():
        if gpu_mode_required():
            raise RuntimeError("GPU sensing-matrix construction requires CUDA")
        # Explicit/ambient CPU mode only.
        import scipy.sparse as sp
        NS = sp.block_diag([ns_harm, ns_anharm], format="csr")
        SM = sm @ NS
        return np.asarray(SM.toarray(), dtype=dtype)

    # GPU path: torch.sparse.mm needs a dense RHS, so build the dense NS here.
    nsh = ns_harm.toarray() if hasattr(ns_harm, "toarray") else np.asarray(ns_harm)
    nsa = ns_anharm.toarray() if hasattr(ns_anharm, "toarray") else np.asarray(ns_anharm)
    nh, mh = nsh.shape
    na, ma = nsa.shape
    NS = np.zeros((nh + na, mh + ma), dtype=np.float64)
    NS[:nh, :mh] = nsh
    NS[nh:, mh:] = nsa

    if sm.nnz >= 2 ** 31:
        raise ValueError(
            "nnz=%d exceeds the int32 range torch CSR indices use; split the "
            "scan or load on CPU" % sm.nnz)

    NSt = torch.as_tensor(np.ascontiguousarray(NS), dtype=torch.float64,
                           device=device())
    # CSR path: ~5x faster and ~half the sparse-tensor memory of COO on
    # torch 2.x (indices are int32 and stored once). Fall back to COO if the
    # CSR kernel is unavailable.
    crow = torch.as_tensor(sm.indptr, dtype=torch.int32, device=device())
    ccol = torch.as_tensor(sm.indices, dtype=torch.int32, device=device())
    cval = torch.as_tensor(sm.data, dtype=torch.float64, device=device())
    spt = None
    try:
        spt = torch.sparse_csr_tensor(crow, ccol, cval, size=sm.shape,
                                      dtype=torch.float64, device=device())
        SM = torch.sparse.mm(spt, NSt)
    except Exception:
        # Release the failed CSR tensors before the COO retry so a CSR OOM does
        # not compound with a second (larger) COO allocation. del only drops the
        # refcount; empty_cache() returns the block to the driver.
        del spt, crow, ccol, cval
        torch.cuda.empty_cache()
        # [SM int32] sm can carry int32 indices with an int64 indptr (the narrow
        # sensing-matrix layout, see core/sparse_io.py) and scipy's
        # tocoo()/expandptr rejects that mix ("Output dtype not compatible with
        # inputs"), so expand indptr here in sm's own index dtype instead.
        _rows = _csr_major_indices(sm)
        idx = torch.as_tensor(np.vstack([_rows, sm.indices]), dtype=torch.long,
                              device=device())
        vals = torch.as_tensor(sm.data, dtype=torch.float64, device=device())
        spt = torch.sparse_coo_tensor(idx, vals, sm.shape,
                                      device=device()).coalesce()
        SM = torch.sparse.mm(spt, NSt)
    return _to_numpy(SM, dtype)


# ---------------------------------------------------------------------------
# GPU SpMV for the TwoLevelSM operator (SM_prime row-split across GPUs)
# ---------------------------------------------------------------------------
def _check_cuda_csr_indices(matrix):
    """Reject CSR blocks that cannot be represented by our int32 CUDA path.

    Casting a large CSR indptr to int32 wraps silently.  Check before either
    allocating a CUDA tensor or asking cuSPARSE to read the resulting indices.
    This uses scalar CSR metadata, without scanning/copying a multi-GB array.
    """
    limit = np.iinfo(np.int32).max
    if matrix.nnz > limit or max(matrix.shape, default=0) > limit:
        raise ValueError(
            "CUDA CSR block shape=%s nnz=%d exceeds the int32 index range; "
            "use more GPU blocks or the CPU sparse path"
            % (matrix.shape, matrix.nnz))


def _cuda_csr_bytes(matrix, value_itemsize):
    """Bytes of the actual CUDA CSR allocation, regardless of host dtype."""
    return (int(matrix.nnz) * (int(value_itemsize) + 4)
            + (int(matrix.shape[0]) + 1) * 4)


def _cuda_spmv_block_budget(row_block, transpose_block, value_itemsize):
    """Resident CSR pair, both SpMV vectors, and conservative workspace room.

    Equal row/column ranges need not contain equal numbers of nonzeros.  The
    device-selection average is only a hint; this actual block budget decides
    whether a pair is safe to upload to its selected card.
    """
    return _cuda_spmv_block_budget_counts(
        row_block.shape, row_block.nnz, transpose_block.shape,
        transpose_block.nnz, value_itemsize)


def _cuda_spmv_block_budget_counts(row_shape, row_nnz, trans_shape, trans_nnz,
                                   value_itemsize):
    """Same accounting as _cuda_spmv_block_budget, from COUNTS instead of blocks.

    The upload path needs the verdict BEFORE it materialises the blocks: a
    c3=5.0 shard's row block and column block are 10.7 GB each on the host, so
    holding both just to compute a budget doubled the host transient (21.4 GB of
    the measured 66.76 GiB peak) on top of keeping two 10.7 GB numpy buffers
    alive through the upload.  Both entry points share this one formula, so the
    count-based pre-flight can never drift from the block-based one.
    """
    _vi = int(value_itemsize)
    resident = ((int(row_nnz) + int(trans_nnz)) * (_vi + 4)
                + (int(row_shape[0]) + int(trans_shape[0]) + 2) * 4)
    vectors = (sum(int(x) for x in row_shape)
               + sum(int(x) for x in trans_shape)) * _vi
    workspace = max(256 << 20, (resident + vectors + 9) // 10)
    return resident + vectors + workspace


class GpuSparseMV(object):
    """Row-split SM_prime (+ transpose) across GPUs for TwoLevelSM matvec/rmatvec.

    The memory-heavy half of the TwoLevelSM is SM_prime (6.5-60 GB): matvec is
    SM_prime @ t, rmatvec is SM_prime.T @ u, while NS@v / NS.T@t stay on the
    CPU (small). Each device holds a row block of SM_prime (for matvec) and a
    row block of SM_prime.T (== a column block of SM_prime, for rmatvec), so
    peak VRAM = 2 x SM size / n_gpu.

    Fallback is the CALLER's job: the TwoLevelSM keeps its CPU path and
    disables the GPU mv if construction or a call raises.
    """

    def __init__(self, sm_prime, n_gpu=None, device_ids=None):
        import numpy as np
        t = _torch()
        if t is None or not t.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        self._np = np
        self._t = t
        # GPU sparse kernels are asynchronous; extra PyTorch CPU worker
        # threads only contend with joblib's fold workers and slow host/device
        # synchronization on large matrices. Keep one dispatcher thread.
        if os.environ.get("PHEASY_GPU_SM_TORCH_THREADS", "1").lower() not in ("0", "false", "off"):
            t.set_num_threads(1)
        self._dt64 = t.float64 if sm_prime.dtype == np.float64 else t.float32
        self._value_itemsize = 8 if sm_prime.dtype == np.float64 else 4
        self._sm_bytes_val = _cuda_csr_bytes(sm_prime, self._value_itemsize)
        N, M = sm_prime.shape
        devs = self._pick_devices(device_ids, n_gpu)
        if not devs:
            raise RuntimeError("no usable CUDA device")
        self._devs = devs
        G = len(devs)
        # Equal-NNZ boundaries, not equal index counts: the c3=5.0 factors put
        # most of their nonzeros in the first uniform quarter, so shard 0 needed
        # 19.00 GiB of CUDA CSR while the even-split estimate said 22.54 GB for a
        # shard's whole host footprint (see _balanced_shard_edges).
        self._rs = _row_shard_edges(sm_prime, G)
        self._cs = _col_shard_edges(sm_prime, G)
        _mem_trace("GpuSparseMV nnz-balanced row edges %s (N=%d)"
                   % ([int(x) for x in self._rs], int(self._rs[-1])))
        self._R = []
        self._T = []
        # [low-mem] per-block column slices instead of one full transpose:
        # building smT = SM_prime.T.tocsr() transiently doubles the ~60 GB SM
        # in host RAM and OOM-killed the 699-config fit at ~185 GB on the
        # shared box (exit 137, three times). Column slices are ~97 s per
        # 13275-col block (8 min total vs the transpose) but peak ~75 GB.
        try:
            # Nonzero counts per row/column: the per-shard budget below needs
            # them, and asking for them here costs one O(nnz) pass instead of
            # materialising both blocks first.
            _row_counts = np.diff(sm_prime.indptr)
            _col_counts = _col_nnz_cached(sm_prime)
            for i, d in enumerate(devs):
                dev = t.device("cuda:%d" % d)
                r0, r1 = int(self._rs[i]), int(self._rs[i + 1])
                c0, c1 = int(self._cs[i]), int(self._cs[i + 1])
                _row_nnz = int(_row_counts[r0:r1].sum())
                _col_nnz = int(_col_counts[c0:c1].sum())
                # Per-block host footprint.  This loop is the only host-memory
                # step of the GPU-SM path, so the trace says whether a kill came
                # from the blocks or from somewhere upstream.
                _mem_trace("GpuSparseMV host blocks %d/%d (nnz %d+%d)"
                           % (i + 1, G, _row_nnz, _col_nnz))
                # Pre-flight from COUNTS, before either block exists: at c3=5.0
                # they are 10.7 GB each, so building both just to measure doubled
                # this loop's transient for no benefit.
                required = _cuda_spmv_block_budget_counts(
                    (r1 - r0, M), _row_nnz, (c1 - c0, N), _col_nnz,
                    self._value_itemsize)
                free = _device_free_bytes(d)
                if free is not None and required > free:
                    raise MemoryError(
                        "CUDA device %d: actual CSR row/transpose blocks need "
                        "%.2f GiB including workspace, only %.2f GiB usable; "
                        "nonzeros may be unevenly distributed across devices. "
                        "Use more/free GPUs or the CPU sparse path"
                        % (d, required / 2**30, free / 2**30))
                # [X2/M1] pin the thread-local current device while creating
                # the sparse tensors: cuSPARSE handles/workspace follow it.
                # Build, upload and RELEASE one block at a time so the two
                # 10.7 GB host blocks never coexist.
                with t.cuda.device(d):
                    Ri = sm_prime[r0:r1].tocsr()
                    _check_cuda_csr_indices(Ri)
                    self._R.append(self._csr_to_torch(Ri, dev))
                    del Ri
                    Ti = sm_prime[:, c0:c1].T.tocsr()
                    _check_cuda_csr_indices(Ti)
                    self._T.append(self._csr_to_torch(Ti, dev))
                    del Ti
        except Exception:
            # A failure on card k must release cards 0..k before CPU fallback.
            self.close()
            raise

    def _pick_devices(self, device_ids, n_gpu):
        import os as _os
        t = self._t
        if device_ids is None:
            raw = _os.environ.get("PHEASY_GPU_SM_DEVICES", "").strip()
            device_ids = [int(x) for x in raw.split(",")] if raw else None
        # resolve n_gpu first so the per-card VRAM need is known before
        # filtering the device list.
        if n_gpu is None:
            raw = _os.environ.get("PHEASY_GPU_SM_NGPU", "").strip()
            n_gpu = int(raw) if raw else None
        if n_gpu is None or n_gpu <= 0:
            # auto: 2x SM size (matrix + transpose) over ~20 GB usable/card.
            # [cap 5] the exit-143 SIGTERMs on busy cards are still unexplained
            # (a busy shared card does NOT itself send SIGTERM; a killer does).
            # Cap at 5 as insurance until the killer is identified; the
            # free-VRAM filter below keeps us off busy cards regardless.
            n_gpu = min(5, max(1, int(np.ceil(2.0 * self._sm_bytes_val / 20.0e9))))
        if device_ids is None:
            # [R1] like GpuRidgeCV._multi_gpu_devices: only cards with enough
            # usable free VRAM (a busy shared card caused silent crashes --
            # the 7/5-card instability was card CONTENTION with vasp/gmx, not
            # a torch bug: 9/9 pass on free cards). Absolute 1 GB floor so a
            # small SM does not land on a busy card.
            _per_card = max(1 << 30, int(2.0 * self._sm_bytes_val / max(1, n_gpu)))
            device_ids = _multi_gpu_devices(min_free_bytes=_per_card)
        # Duplicate IDs would place multiple blocks on one card and invalidate
        # the per-card budget. Preserve the caller's order while deduplicating.
        device_ids = list(dict.fromkeys(
            d for d in device_ids if 0 <= d < t.cuda.device_count()))
        if not device_ids:
            return []
        n_gpu = min(n_gpu, len(device_ids))
        return device_ids[:n_gpu]

    def _csr_to_torch(self, m, dev):
        _check_cuda_csr_indices(m)
        t = self._t
        crow = t.as_tensor(m.indptr, dtype=t.int32, device=dev)
        ccol = t.as_tensor(m.indices, dtype=t.int32, device=dev)
        cval = t.as_tensor(m.data, dtype=self._dt64, device=dev)
        return t.sparse_csr_tensor(crow, ccol, cval, size=m.shape,
                                   dtype=self._dt64, device=dev)

    def _mv_blocks(self, blocks, x0):
        """blocks[i] @ x0 on each device.

        [B2] enqueue ALL devices first, then collect: yi.cpu() inside the loop
        synced per block and serialized the cards. torch.sparse.mm segfaults on
        the 540M-nnz blocks after ~300 calls; the R @ xi dispatch is stable and
        periodic empty_cache releases the caching allocator for 10k+ calls.
        [X2/M1] pin the per-card current device during enqueue AND collect (the
        allocator stream bookkeeping and cuSPARSE handles follow the current
        device), and hold xi (and the result) alive until the collect: an
        autograd-free result does not keep xi alive, so the next loop turn
        would free xi while the device kernel may still be reading it.
        """
        t = self._t
        np = self._np
        ys = []
        for i, B in enumerate(blocks):
            with t.cuda.device(self._devs[i]):
                xi = t.as_tensor(x0, dtype=self._dt64, device=self._devs[i])
                ys.append((self._devs[i], B @ xi, xi))
        parts = []
        for _d, _y, _keep in ys:
            with t.cuda.device(_d):
                parts.append(_y.cpu().numpy().astype(np.float64))
        self._n_calls = getattr(self, "_n_calls", 0) + 1
        if self._n_calls % 50 == 0:
            t.cuda.empty_cache()
        return np.concatenate(parts)

    def matvec(self, x):
        """SM_prime @ x -> (N,) numpy f64. x: (M,) numpy."""
        return self._mv_blocks(self._R, np.ascontiguousarray(x, dtype=np.float64))

    def rmatvec(self, u):
        """SM_prime.T @ u -> (M,) numpy f64. u: (N,) numpy."""
        return self._mv_blocks(self._T, np.ascontiguousarray(u, dtype=np.float64))

    def close(self):
        """Release the GPU tensors (clear the lists, not just loop vars)."""
        t = self._t
        try:
            self._R.clear()
            self._T.clear()
            t.cuda.empty_cache()
        except Exception:
            pass


class GpuTwoLevelOperator:
    """CUDA-resident SM_prime @ NS, without a product or Gram matrix.

    device_ids=[d] keeps the original single-device layout (both factors and
    their CSR transposes on one device).  device_ids=[d0, d1, ...] SHARDS it:
    SM_prime is split by rows for matvec and by columns for the adjoint, exactly
    like the validated GpuSparseMV path, so each card holds ~2*nnz/G nonzeros
    instead of the whole matrix -- the same code then runs on one card or on
    many, and the per-card memory requirement falls by G.

    Factor values follow the sensing-matrix dtype (float32 stays float32) and
    indices are int32 while nnz fits; the old hard-coded float64/int64 doubled
    both and is what made a 6.5 GB factor look like 29.5 GB of VRAM.

    Allocation or unsupported sparse-kernel errors propagate: this backend NEVER
    falls back.  Dense NS is supported; sparse NS is never densified.  CV uses
    row masks so folds do not duplicate the factors.  Only setup uploads and
    public result downloads cross the host boundary.
    """
    def __init__(self, A, device_id=None, extra_workspace_bytes=0, host_factors=None,
                 device_ids=None):
        import scipy.sparse as sp
        if not enabled() or not available():
            raise RuntimeError("Resident two-level LASSO requires enabled CUDA; no CPU fallback")
        input_scale = getattr(A, "_twolevel_scale", None)
        A = getattr(A, "_twolevel_base", A)
        if not hasattr(A, "SM_prime") or not hasattr(A, "NS"):
            raise TypeError("GpuTwoLevelOperator requires TwoLevelSM")
        self.torch = torch = _torch()
        if device_ids is None:
            device_ids = [device() if device_id is None else device_id]
        self._devs = [d if isinstance(d, torch.device) else torch.device(d)
                      for d in device_ids]
        if not self._devs:
            raise ValueError("device_ids must name at least one CUDA device")
        self._n_shards = len(self._devs)
        self.device = self._devs[0]
        self.shape = A.shape
        if not sp.issparse(A.SM_prime):
            raise TypeError("Resident two-level SM_prime must be scipy sparse")
        fraction = float(os.environ.get("PHEASY_GPU_MEM_FRACTION", "0.8"))
        if not 0 < fraction <= 1:
            raise ValueError("PHEASY_GPU_MEM_FRACTION must be in (0, 1]")
        # Follow the factor dtype unless PHEASY_GPU_RESIDENT_DTYPE overrides it.
        _dt_env = os.environ.get("PHEASY_GPU_RESIDENT_DTYPE", "auto").strip().lower()
        if _dt_env in ("auto", ""):
            self._value_dtype = (torch.float64 if A.SM_prime.dtype == np.float64
                                 else torch.float32)
        elif _dt_env in ("float64", "fp64", "double"):
            self._value_dtype = torch.float64
        elif _dt_env in ("float32", "fp32", "single"):
            self._value_dtype = torch.float32
        else:
            raise ValueError("PHEASY_GPU_RESIDENT_DTYPE must be auto, float32 or float64")
        self._index_dtype = (torch.int32 if int(A.SM_prime.nnz) < 2**31
                             else torch.int64)
        peak, budget, free, _fraction = resident_twolevel_estimate(
            A, device_id=self.device,
            extra_workspace_bytes=extra_workspace_bytes,
            n_shards=self._n_shards)
        self.estimated_peak_bytes = peak
        if free is None and torch.device(self.device).type == "cuda":
            raise RuntimeError("Cannot query resident CUDA memory budget; refusing unchecked upload")
        if budget is not None and self.estimated_peak_bytes > budget:
            raise ResidentFootprintError(resident_twolevel_error_message(
                self.estimated_peak_bytes, budget))
        for _dev in self._devs[1:]:
            # Every shard must fit its own card: the footprint is per card, not
            # for the whole group.
            _free_other = _device_free_bytes(_dev)
            if _free_other is None:
                raise RuntimeError("Cannot query resident CUDA memory budget on %s; "
                                   "refusing unchecked upload" % _dev)
            # [FIX R2] same own-bytes accounting as device 0 (see
            # _device_available_bytes): a replica may already be resident here.
            _avail_other = _device_available_bytes(_dev)
            if _avail_other is None:
                _avail_other = _free_other
            _budget_other = int(_avail_other * fraction)
            if self.estimated_peak_bytes > _budget_other:
                raise ResidentFootprintError(resident_twolevel_error_message(
                    self.estimated_peak_bytes, _budget_other))

        host_prime_t = None
        if host_factors is None:
            host_prime, host_ns = A.SM_prime, A.NS
        else:
            # [ACC] reuse the canonical host arrays built by
            # _canonical_twolevel_host(); they are already tocsr'd, deduplicated
            # and index-sorted, so the per-card host pass disappears.
            host_prime, host_ns = host_factors[0], host_factors[1]
            host_prime_t = host_factors[2] if len(host_factors) > 2 else None

        def upload(matrix, dev, canonical=False):
            if not sp.issparse(matrix):
                # Dense NS: replicate on every shard so the two-level matvec never
                # needs a cross-device round trip for it.
                return torch.as_tensor(np.asarray(matrix), dtype=self._value_dtype,
                                       device=dev)
            if canonical:
                # Already tocsr'd, deduplicated and index-sorted by
                # _canonical_twolevel_host(): no second host copy.
                csr = matrix
            else:
                csr = matrix.tocsr(copy=True)
                csr.sum_duplicates()
                csr.sort_indices()
            # np.array(..., copy=True) guarantees fresh, owning, C-contiguous
            # buffers: a column-sliced transpose hands back strided views of the
            # parent's arrays and torch then refuses ("expected col_indices to be
            # a contiguous tensor per batch").  GpuSparseMV never hit this because
            # its int32 cast always copied.
            # Index width is PER BLOCK, not per matrix: the sharded layout
            # uploads ~nnz/G per card, so the c3=5.0 matrix (3.57e9 nnz, int64 at
            # full size) ships int32 metadata from its 892M-nnz shards.  Reading
            # the width off the GLOBAL nnz sent int64 indices instead -- +4 B/nnz,
            # +3.57 GB of VRAM per card, the same class of dtype doubling this
            # class's docstring warns about.  _check_cuda_csr_indices below still
            # guards int32 representability of THIS block.
            _idx_np = np.int32 if int(csr.nnz) < 2**31 else np.int64
            _val_np = np.float32 if self._value_dtype == torch.float32 else np.float64
            _check_cuda_csr_indices(csr)
            # Mirror GpuSparseMV's proven construction: int32 metadata with
            # dtype=/size= and no check_invariants (on a transposed slice that
            # flag trips torch's "expected col_indices to be a contiguous tensor
            # per batch" check).  _check_cuda_csr_indices above is the package's
            # own int32-representability guard.
            # The VALUE array needs no host copy when it is already the device
            # dtype and C-contiguous (torch.as_tensor with a device transfers
            # straight from the scipy buffer) -- at c3=5.0 a shard block is
            # 10.7 GB, so the redundant copy showed up in the HWM.  The strided
            # case (column-sliced transpose) still takes the copying path.
            _data = csr.data
            if _data.dtype != _val_np or not _data.flags.c_contiguous:
                _data = np.array(_data, dtype=_val_np, copy=True)
            return torch.sparse_csr_tensor(
                torch.as_tensor(np.array(csr.indptr, dtype=_idx_np, copy=True), device=dev),
                torch.as_tensor(np.array(csr.indices, dtype=_idx_np, copy=True), device=dev),
                torch.as_tensor(_data, device=dev),
                size=csr.shape, dtype=self._value_dtype, device=dev)

        self._R = []
        self._T = []
        try:
            if self._n_shards == 1:
                # Canonicalize the host matrices ONCE and build BOTH the matrix and
                # its transpose on the host.  A device-side
                # prime.transpose(0,1).to_sparse_csr() yields unsorted column
                # indices, and cuSPARSE then refuses the SpMV ("operation not
                # supported when calling cusparseSpMV_bufferSize") -- measured on
                # the production 810M-nnz float32 factor.
                if host_prime_t is not None:
                    # Caller supplied canonical factors, so the adjoint already
                    # exists: uploading it costs nothing extra, while rebuilding
                    # it here transposed a second copy of prime (measured 155.2 s
                    # on the MgC factor, nnz 1.127e9).  _canonical_twolevel_host
                    # only fills this slot for a SPARSE prime, and its .T.tocsr()
                    # is index-sorted, which is what upload(canonical=True)
                    # promises.  A LAZY adjoint (_AdjointBlocks) has to be
                    # materialised for this layout: a single replica needs the
                    # whole adjoint on one card, there is no block to defer.
                    self.prime = upload(host_prime, self.device, canonical=True)
                    if hasattr(host_prime_t, "materialize"):
                        host_prime_t = host_prime_t.materialize()
                    self.prime_t = upload(host_prime_t, self.device,
                                          canonical=True)
                else:
                    _prime_c = _canonical_csr_inplace(host_prime)
                    self.prime = upload(_prime_c, self.device, canonical=True)
                    if sp.issparse(host_prime):
                        self.prime_t = upload(
                            _canonical_block(_prime_c.T), self.device,
                            canonical=True)
                    else:
                        self.prime_t = self.prime.transpose(0, 1).to_sparse_csr()
            else:
                # Sharded: rows for the matvec, columns for the adjoint.  Every
                # output element is produced by exactly one card from its own
                # block, so the concatenation is order-independent of G.
                N, M = A.shape
                # ROW split over N (prime rows) for the matvec, COLUMN split over
                # prime's OWN column count (mid) for the adjoint.  Splitting the
                # adjoint over the effective column count p was wrong: prime has
                # only mid columns, so with p > mid every shard but the first got
                # an empty block (rmatvec still looked right because block 0 then
                # held every column, but column norms came out wrong).
                _mid = int(host_prime.shape[1])
                # Equal-NNZ boundaries over prime's rows (matvec) and prime's own
                # columns (adjoint); a uniform split put most of the c3=5.0
                # nonzeros -- and therefore the whole host transient and VRAM
                # requirement -- on one card.
                self._rs = _row_shard_edges(host_prime, self._n_shards)
                self._cs = _col_shard_edges(host_prime, self._n_shards)
                # Nonzero counts per row/column, so the per-shard budget below
                # needs no materialised block at all.
                _row_counts = np.diff(host_prime.indptr)
                _col_counts = _col_nnz_cached(host_prime)
                for i, dev in enumerate(self._devs):
                    r0, r1 = int(self._rs[i]), int(self._rs[i + 1])
                    c0, c1 = int(self._cs[i]), int(self._cs[i + 1])
                    # Per-SHARD pre-flight, which the uniform estimate cannot give:
                    # the even-split number describes the average shard, so a fat
                    # shard used to reach the allocator with no check at all.  Count
                    # instead of materialise: the two blocks are 10.7 GB each at
                    # c3=5.0, and building them just to measure was half the host
                    # transient this loop is trying to avoid.
                    _row_nnz = int(_row_counts[r0:r1].sum())
                    _col_nnz = int(_col_counts[c0:c1].sum())
                    _need = _cuda_spmv_block_budget_counts(
                        (r1 - r0, _mid), _row_nnz, (c1 - c0, N), _col_nnz,
                        self._item_bytes())
                    _avail = _device_available_bytes(dev)
                    if _avail is not None and _need > int(_avail):
                        raise ResidentFootprintError(resident_twolevel_error_message(
                            _need, int(_avail)))
                    _mem_trace("resident shard %d/%d host blocks (nnz %d+%d, need %.2f GiB)"
                               % (i + 1, self._n_shards, _row_nnz, _col_nnz,
                                  _need / 2 ** 30))
                    with torch.cuda.device(dev):
                        # Build, upload and RELEASE one block at a time.  Holding
                        # both through the upload measured +21.4 GB on the real
                        # c3=5.0 factors (the blocks are 10.7 GB each) on top of
                        # the 42.84 GB factor set, for no benefit: the budget above
                        # already answered the fit question.  _canonical_block
                        # canonicalizes IN PLACE on a freshly sliced (privately
                        # owned) buffer, and a lazy adjoint costs one O(nnz)
                        # filtered pass per shard, so the full 42.84 GB transpose
                        # never exists in RAM.
                        _Rb = _canonical_block(host_prime[r0:r1])
                        self._R.append(upload(_Rb, dev, canonical=True))
                        del _Rb
                        if host_prime_t is not None:
                            _Tb = _canonical_block(host_prime_t[c0:c1])
                        else:
                            _Tb = _canonical_block(host_prime[:, c0:c1].T)
                        self._T.append(upload(_Tb, dev, canonical=True))
                        del _Tb
                        if i == 0:
                            self.prime = None
                            self.prime_t = None
            self.ns = upload(host_ns, self.device, canonical=host_factors is not None)
            if sp.issparse(host_ns):
                # Same rule as prime_t: a DEVICE-side sparse transpose leaves
                # unsorted column indices and cuSPARSE then refuses the SpMV
                # ("operation not supported when calling cusparseSpMV_bufferSize").
                # Measured: the whole 810M-nnz prime SpMV is fine, the transposed
                # NS was the one that failed.
                self.ns_t = upload(_canonical_block(_canonical_csr_inplace(host_ns).T),
                                   self.device, canonical=True)
            else:
                self.ns_t = self.ns.T
            self.scale = torch.ones(A.shape[1], dtype=self._value_dtype, device=self.device)
            self.input_scale = (torch.ones_like(self.scale) if input_scale is None else
                                torch.as_tensor(input_scale, dtype=self._value_dtype,
                                                device=self.device))
            if not bool((torch.isfinite(self.input_scale) & (self.input_scale != 0)).all().item()):
                raise ValueError("two-level column scales must be finite and nonzero")
        except Exception:
            self.close()
            raise

    def close(self):
        """Release owned factor tensors, including partially initialized state."""
        for name in ("prime", "ns", "prime_t", "ns_t", "scale", "input_scale"):
            if hasattr(self, name):
                setattr(self, name, None)
        for name in ("_R", "_T"):
            shards = getattr(self, name, None)
            if shards:
                shards.clear()



    def _item_bytes(self):
        """Bytes per factor VALUE on the device (int32 indices are fixed at 4)."""
        return 4 if self._value_dtype == self.torch.float32 else 8

    def _mm(self, matrix, vector):
        if matrix.layout == self.torch.strided:
            return matrix @ vector
        if vector.ndim == 1:
            return self.torch.sparse.mm(matrix, vector[:, None]).flatten()
        return self.torch.sparse.mm(matrix, vector)

    def matvec(self, vector):
        x = vector / (self.scale * self.input_scale)
        if self._n_shards == 1:
            return self._mm(self.prime, self._mm(self.ns, x))
        # Sharded: NS lives on the primary; the mid vector is broadcast to every
        # card and the disjoint row blocks come back to the primary.  Per
        # iteration this moves ~mid*4 bytes per card, not the factors.
        #
        # Measured, and worth knowing before "using more cards": on THIS host the
        # sharded operator is a MEMORY tool, not a speed tool.  The two half-SpMMs
        # do overlap (interleaved medians on idle cards: 5.45 ms for both halves
        # against 10.60 ms for the same work on one card), but the mandatory
        # broadcast and collect cost ~5.5 ms because there is NO peer access
        # between any pair of cards here (torch.cuda.can_device_access_peer is
        # False for every pair), so each cross-card copy is staged through host
        # memory and lands on the critical path: the whole matvec then costs
        # 10.90 ms (old loop), 10.93 ms (same loop issuing the remote shard on its
        # own stream with non_blocking copies -- bit-identical output, no gain),
        # and 10.61 ms on ONE card doing the entire matrix.  The 1/2/3/4-shard
        # sweep agrees: 22.50/23.02/24.15/24.73 ms per matvec+rmatvec pair.
        # Shard for VRAM, and prefer one card per fit with folds/alphas spread
        # across cards instead.
        t = self._mm(self.ns, x)
        parts = []
        for i, dev in enumerate(self._devs):
            with self.torch.cuda.device(dev):
                ti = t if dev == self.device else t.to(dev)
                yi = self._R[i] @ ti
                parts.append(yi if dev == self.device else yi.to(self.device))
                del ti
        return self.torch.cat(parts)

    def rmatvec(self, vector):
        if self._n_shards == 1:
            return self._mm(self.ns_t, self._mm(self.prime_t, vector)) / (self.scale * self.input_scale)
        parts = []
        for i, dev in enumerate(self._devs):
            with self.torch.cuda.device(dev):
                ui = vector if dev == self.device else vector.to(dev)
                zi = self._T[i] @ ui
                parts.append(zi if dev == self.device else zi.to(self.device))
        z = self.torch.cat(parts)
        return self._mm(self.ns_t, z) / (self.scale * self.input_scale)

    def norm_estimate(self, iters=10):
        """Cached spectral estimate for the current normalized operator."""
        if getattr(self, "_norma", None) is None:
            self._norma = _operator_norm_estimate(self, iters)
        return self._norma

    def _norm_block_budget(self):
        """Workspace for one column block, using the room the device has.

        The block width decides how many times the factors are re-read, and that
        dominates the cost: measured on the MgC operator (36864x69487, 91.4M nnz,
        2 x RTX3090), 409 passes took 23.0s, 26 passes 12.5s and 7 passes 11.6s.
        PHEASY_GPU_NORM_WORKSPACE_MB (512 MB) is a FLOOR here; the device can
        usually spare far more for a transient buffer.  Results are unaffected:
        the accumulation order moves only the last float32 digits (1.1e-07
        between the 512 MB and 32 GB runs).
        """
        budget = _norm_workspace_bytes()
        try:
            free = _device_free_bytes(self.device)
        except Exception:
            free = None
        if free:
            budget = max(budget, min(int(free * _NORM_BLOCK_FREE_FRACTION),
                                     _NORM_BLOCK_MAX_BYTES))
        return budget

    def col_norms(self):
        """Exact full-row column norms of the current effective operator, on CUDA.

        Bounded column blocks avoid sparse-sparse products and never allocate
        the full sensing matrix. Workspace is O(block*(n + mid + p)).

        The block width is a trade-off against the factors being re-read (see
        _norm_block_budget).  It is re-clamped against LIVE free memory, and a
        CUDA OOM falls back once to the conservative configured floor: this box is
        shared, so the room can disappear between the first query and the
        allocation, and that must not turn a norm pass into a hard failure.
        """
        if self._n_shards != 1:
            return self._sharded_col_norms()
        torch = self.torch
        n, p = self.shape
        per_col = max(8 * (n + self.ns.shape[0] + p) * 2, 1)
        floor = _norm_workspace_bytes()
        budget = self._norm_block_budget()
        norms = torch.empty_like(self.scale)
        start = 0
        block = max(1, budget // per_col)
        while start < p:
            count = min(block, p - start)
            # Follow the factor dtype: an fp64 basis against fp32 factors is
            # the same mixed-dtype failure the resident path hit (found by
            # re-running this gate with float32 fixtures).
            try:
                basis = torch.zeros((p, count), dtype=self._value_dtype,
                                    device=self.device)
                idx = torch.arange(count, device=self.device)
                basis[start + idx, idx] = 1
                cols = self._mm(self.prime, self._mm(
                    self.ns, basis / (self.input_scale * self.scale)[:, None]))
                norms[start:start + count] = torch.linalg.vector_norm(cols, dim=0)
            except RuntimeError as _e:
                if "out of memory" not in str(_e).lower() or count <= 1:
                    raise
                # Someone else took the memory: drop to the configured floor
                # (never below one column) and retry this block.
                del basis
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                block = max(1, floor // per_col)
                if block >= count:
                    block = max(1, count // 2)
                continue
            del basis, cols
            start += count
        return norms

    def _sharded_col_norms(self):
        """Exact column norms of the effective operator across shards.

        Column j of the effective operator spans EVERY row, so each shard
        accumulates squared entries over its own row block and the per-shard
        results are SUMMED (concatenating them, as the matvec does, would be
        wrong).  Accumulation is float64 even for float32 factors: the norms feed
        --std scaling, where a relative error is not damped by the solve.
        """
        torch = self.torch
        n, p = self.shape
        budget = self._norm_block_budget()
        block = max(1, budget // max(8 * (n + self.ns.shape[0] + p) * 2, 1))
        total = torch.zeros(p, dtype=torch.float64, device=self.device)
        for start in range(0, p, block):
            count = min(block, p - start)
            basis = torch.zeros((p, count), dtype=self._value_dtype,
                                device=self.device)
            idx = torch.arange(count, device=self.device)
            basis[start + idx, idx] = 1
            scaled = basis / (self.input_scale * self.scale)[:, None]
            t = self._mm(self.ns, scaled)
            partial = torch.zeros(count, dtype=torch.float64, device=self.device)
            for i, dev in enumerate(self._devs):
                with torch.cuda.device(dev):
                    ti = t if dev == self.device else t.to(dev)
                    # _mm (torch.sparse.mm), not the @ operator: for a 2-D dense
                    # operand the sparse @ dense dispatch does NOT match the
                    # single-device path (measured: 22x wrong column norms).
                    cols = self._mm(self._R[i], ti).to(torch.float64)
                    sq = torch.sum(cols * cols, dim=0)
                    partial += sq if dev == self.device else sq.to(self.device)
                    del ti, cols
            total[start:start + count] = partial
        # SQUARES were accumulated above: the single-device path returns the norm,
        # not its square (measured 22x disagreement before this sqrt).  The result
        # is cast back to the FACTOR dtype: normalize() assigns it to self.scale,
        # and an fp64 scale would turn every later vector fp64 -- the mixed-dtype
        # SpMM that cuSPARSE refuses (found by running this gate with fp32 data).
        return torch.sqrt(total).to(self._value_dtype)

    def normalize(self):
        """Apply exact unit-L2 normalization and invalidate the spectral estimate."""
        torch = self.torch
        norms = (self.col_norms() * self.scale).to(self._value_dtype)
        self.scale = torch.where(norms < 1e-30, torch.ones_like(norms), norms).to(self._value_dtype)
        self._norma = None
        return self.scale

    def lipschitz(self):
        """Initial step-size estimate for FISTA (Rayleigh * 1.05, 40 iters).

        Not a certificate: 40 power iterations converge to lambda_max from
        below, so the estimate is only an upper bound while (lambda2/lambda1)^40
        is well under the 5 % margin. _fista_twolevel() enforces monotonicity of
        the objective and inflates L if a step turns out too large.
        """
        torch = self.torch
        gen = torch.Generator(device=self.device).manual_seed(0)
        v = torch.randn(self.shape[1], generator=gen, dtype=self._value_dtype,
                        device=self.device)
        for _ in range(40):
            v = self.rmatvec(self.matvec(v))
            v = v / torch.clamp(v.norm(), min=1e-30)
        return torch.clamp(torch.dot(v, self.rmatvec(self.matvec(v))) * 1.05, min=1e-12)


class GpuCSRResidentOperator(GpuTwoLevelOperator):
    """Single-device CSR adapter reusing TwoLevel upload/budget/lifetime handling.

    A sparse identity provides setup compatibility; matvec bypasses it.
    """

    def __init__(self, matrix, device_id=None, extra_workspace_bytes=0):
        import scipy.sparse as sp
        from types import SimpleNamespace
        if not sp.issparse(matrix):
            raise TypeError("GpuCSRResidentOperator requires scipy sparse input")
        factors = SimpleNamespace(SM_prime=matrix, NS=sp.eye(matrix.shape[1], format="csr"), shape=matrix.shape)
        super().__init__(factors, device_id=device_id, extra_workspace_bytes=extra_workspace_bytes)

    def matvec(self, vector):
        return self._mm(self.prime, vector / (self.scale * self.input_scale))

    def rmatvec(self, vector):
        return self._mm(self.prime_t, vector) / (self.scale * self.input_scale)


class GpuSubsetOperator:
    """Row/column view sharing resident factors; workspace contains vectors only."""

    def __init__(self, base, columns, rows=None, column_scale=None):
        self.base, self.torch, self.device = base, base.torch, base.device
        # Mirror the base operator's value dtype.  This used to be read as
        # self._value_dtype further down without ever being assigned: the class
        # has no __getattr__ delegation and no class-level default, so every
        # construction raised AttributeError('GpuSubsetOperator' object has no
        # attribute '_value_dtype') -- i.e. the resident subset view, the
        # resident LASSO debias path and GpuSubsetOperator's own tests were all
        # dead.  Bind it once here so new call sites cannot miss it again.
        self._value_dtype = getattr(base, "_value_dtype", None) or base.torch.float64
        torch = self.torch
        def index_tensor(values):
            raw = torch.as_tensor(values, device=self.device)
            if raw.numel() and (raw.is_floating_point() or raw.is_complex() or raw.dtype == torch.bool):
                raise ValueError("subset indices must have integer dtype")
            return raw.to(dtype=torch.long).clone()
        self.columns = index_tensor(columns)
        self.rows = None if rows is None else index_tensor(rows)
        for indices, bound in ((self.columns, base.shape[1]), (self.rows, base.shape[0])):
            if indices is not None and (indices.ndim != 1 or bool(((indices < 0) | (indices >= bound)).any().item())):
                raise ValueError("subset indices must be one-dimensional and in bounds")
        self.shape = (base.shape[0] if self.rows is None else self.rows.numel(), self.columns.numel())
        self.column_scale = torch.ones(self.shape[1], dtype=self._value_dtype, device=self.device)
        if column_scale is not None:
            scale = torch.as_tensor(column_scale, dtype=self._value_dtype, device=self.device)
            if scale.shape != self.column_scale.shape or not bool((torch.isfinite(scale) & (scale >= 0)).all().item()):
                raise ValueError("subset column scales must be finite, nonnegative, and match active columns")
            self.column_scale = torch.where(scale < 1e-30, torch.ones_like(scale), scale)

    def norm_estimate(self, iters=10):
        """Cache only this fixed subset; never reuse the base operator norm."""
        if getattr(self, "_norma", None) is None:
            self._norma = _operator_norm_estimate(self, iters)
        return self._norma

    def matvec(self, x):
        full = x.new_zeros(self.base.shape[1])
        full.index_add_(0, self.columns, x / self.column_scale)
        result = self.base.matvec(full)
        return result if self.rows is None else result.index_select(0, self.rows)

    def rmatvec(self, y):
        if self.rows is not None:
            full = y.new_zeros(self.base.shape[0])
            full.index_add_(0, self.rows, y)
            y = full
        return self.base.rmatvec(y).index_select(0, self.columns) / self.column_scale


def solve_resident_subset(base, target, columns, rows=None, column_scale=None,
                          ridge_alpha=0.0, atol=1e-8, btol=1e-8, maxiter=5000,
                          raise_on_nonconvergence=True, accept_measured_floor=False):
    """Return physical CUDA subset coefficients; reject unconverged elimination fits.

    raise_on_nonconvergence keeps elimination fits fail-closed (RFE), while the
    relaxed-LASSO debias sets it False so an unconverged CGLS returns its last
    iterate -- exactly the CPU LSQR path's non-aborting behavior -- and the caller's
    residual check decides whether to keep the refit.

    accept_measured_floor separates the two reasons a solve can end uncertified.
    Measured on the production MgC operator (454656x69487, float32, required GPU
    mode) the FIRST RFE round's subset solve reaches a criterion floor ABOVE the
    certifiable cap after 5000 iterations and ends stall_above_floor, which made the
    whole RFE fit raise "Resident subset solve did not converge" -- for a solve whose
    only job is to RANK the features.  With this flag the coefficients are returned
    when the info carries floor EVIDENCE (precision_floor / stall_above_floor, i.e.
    the criterion was measured to stop improving at the working precision), the
    decision is recorded in info["floor_accepted"] / info["floor_note"], and a solve
    with NO floor evidence (iteration_limit, invalid_search_direction, ...) still
    raises: an exhausted budget says nothing about what the arithmetic can reach.

    target is the full row-space vector; column_scale is in active-column order.
    """
    if not np.isfinite(ridge_alpha) or ridge_alpha < 0:
        raise ValueError("Ridge alpha must be finite and nonnegative")
    view = GpuSubsetOperator(base, columns, rows, column_scale)
    y = base.torch.as_tensor(target,
                             dtype=getattr(base, "_value_dtype", None) or base.torch.float64,
                             device=base.device).reshape(-1)
    if y.numel() != base.shape[0]:
        raise ValueError("subset target must match full operator row count")
    if view.rows is not None:
        y = y.index_select(0, view.rows)
    if ridge_alpha > 0:
        coef, info = _iterative_ridge_tensor(view, y, ridge_alpha, atol, btol, maxiter,
                                             penalty_scale=view.column_scale)
    else:
        coef, info = _iterative_lstsq_tensor(view, y, atol, btol, maxiter)
    info = dict(info, n_samples=view.shape[0], n_features=view.shape[1],
                fit_scope="full" if rows is None else "fold", ridge_alpha=float(ridge_alpha))
    _floor_reasons = ("precision_floor", "stall_above_floor")
    _at_floor = bool(not info["converged"] and info.get("stop_reason") in _floor_reasons)
    if _at_floor:
        info["floor_accepted"] = bool(accept_measured_floor)
        info["floor_note"] = (
            "coefficients returned at the MEASURED precision floor (%s, best "
            "criterion %s at iteration %s); this is a precision limit, not an "
            "exhausted budget -- acceptable for a ranking solve, not a certified "
            "solution" % (info.get("stop_reason"), info.get("stall_floor"),
                          info.get("stall_iteration")))
    if not info["converged"] and raise_on_nonconvergence and not (
            accept_measured_floor and _at_floor):
        raise RuntimeError("Resident subset solve did not converge: " + repr(info))
    return coef / view.column_scale, info


def iterative_ridge(A, y, alpha, atol=1e-8, btol=1e-8, maxiter=5000, rows=None, x0=None):
    """Solve ridge on a CUDA-resident operator, returning NumPy coefficients."""
    x, info = _iterative_ridge_tensor(A, y, alpha, atol, btol, maxiter, rows, x0=x0)
    return x.detach().cpu().numpy(), info


def _iterative_ridge_tensor(A, y, alpha, atol=1e-8, btol=1e-8, maxiter=5000, rows=None, penalty_scale=None, x0=None):
    """Augmented CGLS with CUDA coefficient output."""
    torch = A.torch
    dev = A.device
    _vd = getattr(A, "_value_dtype", None) or torch.float64
    y = torch.as_tensor(y, dtype=_vd, device=dev).reshape(-1)
    mask = torch.ones(A.shape[0], dtype=_vd, device=dev)
    if rows is not None:
        mask.zero_(); mask[torch.as_tensor(rows, dtype=torch.long, device=dev)] = 1
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Ridge alpha must be finite and nonnegative")
    sa = float(np.sqrt(alpha))
    if penalty_scale is not None:
        scale = torch.as_tensor(penalty_scale, dtype=getattr(A, "_value_dtype", None) or torch.float64, device=dev)
        if scale.shape != (A.shape[1],) or not bool((torch.isfinite(scale) & (scale > 0)).all().item()):
            raise ValueError("penalty_scale must be finite positive and match coefficient count")
        sa = sa / scale
    class Augmented:
        def __init__(self):
            self.shape = (A.shape[0] + A.shape[1], A.shape[1])
            self.torch = torch
            self.device = dev
            # Carry the factor dtype through the wrapper.  _iterative_lstsq_tensor
            # (and _operator_norm_estimate, which norm_estimate() calls before the
            # loop) both read getattr(A, "_value_dtype", None) or torch.float64 --
            # without this the augmented operator fell back to float64 while
            # GpuTwoLevelOperator keeps float32 factors for an fp32 sensing matrix,
            # so every A.matvec() became a mixed-dtype torch.sparse.mm and cuSPARSE
            # rejected it with "operation not supported".  That is what killed
            # PHEASY_GPU_RIDGE_RESIDENT=1 on the c6.5/c3=4.5 fit (twice), and it is
            # the same omission as GpuSubsetOperator._value_dtype.
            self._value_dtype = _vd
            # With penalty_scale, sa is a vector (one penalty per coefficient) and
            # the augmented operator's condition bound is set by the SMALLEST
            # effective penalty, so reduce it to that scalar here.  Exposing it on
            # the operator lets _iterative_lstsq_tensor learn the penalty without
            # a new argument.
            self.penalty_alpha = float(torch.as_tensor(sa * sa).min().item())
        def norm_estimate(self, iters=10):
            if getattr(self, "_norma", None) is None:
                self._norma = _operator_norm_estimate(self, iters)
            return self._norma
        def matvec(self, x):
            return torch.cat((A.matvec(x) * mask, sa * x))
        def rmatvec(self, z):
            return A.rmatvec(z[:A.shape[0]] * mask) + sa * z[A.shape[0]:]
    return _iterative_lstsq_tensor(Augmented(),
                                   torch.cat((y * mask, y.new_zeros(A.shape[1]))),
                                   atol, btol, maxiter, x0)


def _operator_norm_estimate(A, iters=10):
    """Deterministic matrix-free spectral estimate, not SciPy's Frobenius estimate."""
    t = A.torch
    gen = t.Generator(device=A.device).manual_seed(0)
    # Follow the FACTOR dtype: a hard-coded float64 probe vector against float32
    # factors is a mixed-dtype SpMM that cuSPARSE rejects ("matA (CUDA_R_32F) and
    # matB (CUDA_R_64F) ... is not supported").  This was the LAST fp64 operand in
    # the resident path -- iterative_lstsq() calls norm_estimate() before its loop.
    v = t.randn(A.shape[1], generator=gen,
                dtype=getattr(A, "_value_dtype", None) or t.float64, device=A.device)
    tiny = t.finfo(v.dtype).tiny
    v = v / t.linalg.vector_norm(v).clamp_min(tiny)
    for _ in range(iters):
        w = A.rmatvec(A.matvec(v))
        v = w / t.linalg.vector_norm(w).clamp_min(tiny)
    return float(t.linalg.vector_norm(A.matvec(v)).item())


def iterative_lstsq(A, y, atol=1e-8, btol=1e-8, maxiter=5000):
    """Solve min_x ||A x-y|| with GPU-resident CGLS, returning NumPy coefficients."""
    x, info = _iterative_lstsq_tensor(A, y, atol, btol, maxiter)
    return x.detach().cpu().numpy(), info


_CGLS_FLOOR_MAX = 1e-3
# Plain least squares carries no penalty, so sigma_min is unbounded from below
# and there is no condition number to build a floor from.  Fall back to the same
# one-decade-above-eps rule _lsmr_tol uses, so the CGLS and LSMR paths agree on
# what the stored data precision can actually reach.
_CGLS_UNPENALIZED_COND = 10.0
# Relative improvement the primary criterion must show between two honest
# verifications to count as progress; below it the solve is stagnating.
# These margins must sit ABOVE the reproducibility of the measurement itself.
# On the c7 operator (float32 factors, RTX 3090) three recomputations of
# ||A^T (y - A x)|| for the SAME x gave 2.715779e-06 / 2.767306e-06 /
# 2.832976e-06, i.e. a +-2% spread from the sparse-kernel accumulation order.
# A 1% margin therefore made the verdict depend on kernel scheduling: the
# certificate was raised by 1% and the re-derivation then missed it by 1.25%.
_CGLS_STALL_MARGIN = 0.05
# Headroom over the measured floor when it is promoted to the certificate.
_CGLS_FLOOR_MARGIN = 0.10
# Earliest iteration at which the UNCONDITIONAL honest probe may run, relative to
# the iteration budget (see the probe_window note in _iterative_lstsq_tensor).
# Same reasoning as _FISTA_STALL_WINDOW_MAX: a criterion that is still creeping
# down must not be declared floored by a window that is short compared with the
# budget.
_CGLS_PROBE_WINDOW_MAX = 8000

# Column-norm workspace: how much of the device free memory one transient block
# may take, and its cap (see GpuTwoLevelOperator._norm_block_budget).
_NORM_BLOCK_FREE_FRACTION = 0.25
_NORM_BLOCK_MAX_BYTES = 8 * 1024 ** 3


def _cgls_relative_floor(torch, dtype, norma, alpha=None):
    """Smallest relative normal residual this working precision can certify.

    CG attains a relative normal residual no better than eps*cond(A) in the
    arithmetic it runs in, and for the ridge-augmented operator

        cond(A_aug)^2 = (s_max^2 + alpha) / (s_min^2 + alpha) <= ||A||^2/alpha + 1

    because sigma_min(A_aug)^2 = sigma_min(A)^2 + alpha >= alpha.  Requesting
    less than this floor asks for something the arithmetic cannot express, and
    on the resident path that is not a harmless stall: measured on the c7
    operator (25515x3678, float32 factors, single RTX3090) the shipped 1e-8 sat
    four to five orders below the floor, the recurrence lost the gradient after
    ~1000 iterations, and the true residual grew to 1e17 before maxiter.

    Capped at _CGLS_FLOOR_MAX: past that the system is not solvable in this
    precision at all and a tolerance that large certifies nothing.
    """
    eps = float(torch.finfo(dtype).eps)
    if alpha is not None and float(alpha) > 0:
        cond = max(1.0, float(norma) / float(np.sqrt(float(alpha))))
    else:
        cond = _CGLS_UNPENALIZED_COND
    return min(eps * cond, _CGLS_FLOOR_MAX)


def _iterative_lstsq_tensor(A, y, atol=1e-8, btol=1e-8, maxiter=5000, x0=None):
    """CGLS core returning CUDA coefficients and the same convergence diagnostics.

    Two things separate this from a textbook CGLS, and both are forced by the
    float32 production runs:

    * the requested atol/btol are raised to the floor the working precision can
      reach (see _cgls_relative_floor), so a tolerance the arithmetic cannot
      express stops the iteration early instead of driving it past the point
      where the recurrence still means anything;
    * the cheap stopping test runs on the RECURRENCE residual, which in float32
      drifts below the true y - A x.  Measured on the c7 operator: the
      recurrence was optimistic by 8x at alpha=1e-2, 48x at 1e-4 and 370x at
      1e-6.  Every verdict it produces is therefore re-checked against a
      recomputed residual, and when the two disagree the recurrence is replaced
      by the honest one and the iteration continues.  That disagreement used to
      be terminal -- stop_reason=true_residual_check_failed, 17 times in one
      shipped c7 CV sweep -- and the coefficients it discarded were correct.

    A ridge caller wraps its operator in an augmented one that names the
    penalty it applied in A.penalty_alpha; plain least squares carries none,
    and then only the bare precision bounds the achievable tolerance.  Reading
    it off the operator rather than taking a new argument keeps this signature
    stable for the callers and test doubles that already patch it.
    """
    torch = getattr(A, "torch", None)
    device = getattr(A, "device", None)
    if torch is None or device is None or torch.device(device).type != "cuda":
        raise RuntimeError("GPU iterative least-squares requires a CUDA operator")
    # The iteration vectors must follow the FACTOR dtype: the resident operator
    # now keeps float32 factors for an fp32 sensing matrix (PHEASY_SM_DTYPE=float32,
    # the production setting), and cuSPARSE refuses a mixed-dtype SpMM
    # ("matA (CUDA_R_32F) and matB (CUDA_R_64F) ... is not supported").  Hard-coded
    # float64 here is what made the resident GPU path unusable on fp32 input.
    _vd = getattr(A, "_value_dtype", None) or torch.float64
    y = torch.as_tensor(y, dtype=_vd, device=device).reshape(-1)
    if y.numel() != A.shape[0]:
        raise ValueError("least-squares target length does not match operator")
    if not bool(torch.isfinite(y).all().item()):
        raise ValueError("least-squares target must be finite")
    if (not np.isfinite(atol) or not np.isfinite(btol) or atol < 0 or btol < 0
            or not np.isfinite(maxiter) or maxiter <= 0 or int(maxiter) != maxiter):
        raise ValueError("atol/btol must be finite nonnegative and maxiter a finite positive integer")
    warm = None
    if x0 is not None:
        # A warm start is an optimization, never a contract: the alpha path hands
        # over the previous (larger) alpha's solution, and a stale or non-finite
        # one must not poison the solve -- it is validated here and dropped when
        # unusable rather than trusted.
        candidate = torch.as_tensor(np.asarray(x0, dtype=np.float64).ravel(),
                                    dtype=y.dtype, device=device)
        if candidate.numel() == A.shape[1] and bool(torch.isfinite(candidate).all().item()):
            warm = candidate
        else:
            import warnings
            warnings.warn("ignoring unusable CGLS warm start: expected %d "
                          "finite values, got %d"
                          % (A.shape[1], candidate.numel()),
                          RuntimeWarning, stacklevel=2)
    x = warm.clone() if warm is not None else torch.zeros(A.shape[1], dtype=y.dtype, device=device)
    # Every recurrence below is rebuilt from this residual, so a warm x must
    # produce it exactly as the loop would on its own.
    r = (y - A.matvec(x)) if warm is not None else y.clone()
    s = A.rmatvec(r)
    p = s.clone()
    gamma = torch.dot(s, s)
    rhs_norm = torch.linalg.vector_norm(y).clamp_min(torch.finfo(y.dtype).tiny)
    normal_norm = torch.linalg.vector_norm(s)
    residual_norm = torch.linalg.vector_norm(r)
    norma = A.norm_estimate() if hasattr(A, "norm_estimate") else _operator_norm_estimate(A)
    tol_floor = _cgls_relative_floor(torch, y.dtype, norma,
                                     getattr(A, "penalty_alpha", None))
    atol_eff = max(float(atol), tol_floor)
    btol_eff = max(float(btol), tol_floor)
    # How often to re-derive the residual honestly once the cheap test has been
    # caught lying.  Bounded so a pathological system cannot make the extra
    # matvec/rmatvec dominate the iteration.
    verify_every = max(1, min(int(os.environ.get("PHEASY_CGLS_VERIFY_EVERY", "50")),
                              int(maxiter)))
    def meets_tolerance():
        # Tried and REVERTED: fetching ||x|| lazily (it is only used by the btol
        # branch) so that the first branch could be answered from scalars already
        # in hand.  Measured no gain -- on a converging CGLS the first branch fails
        # on nearly every iteration, so normx is still computed -- and it invited a
        # dtype trap (np.isfinite returns a numpy bool, which made torch dispatch
        # bitwise_and on float tensors: "bitwise_and_cuda is not implemented for
        # Float").  The per-iteration syncs are therefore still there; batching them
        # is the only way to remove them, and GPU.md records an earlier batching
        # attempt that measured 15.7-16.8% SLOWER on this box.
        normx = torch.linalg.vector_norm(x)
        finite = torch.isfinite(normal_norm) & torch.isfinite(residual_norm) & torch.isfinite(rhs_norm) & torch.isfinite(normx)
        # residual_norm <= rhs_norm is not part of SciPy's LSQR test, but the true
        # CGLS residual is monotone and x=0 already achieves ||b||, so it holds at
        # any genuine minimum.  It is what stops an exploded iterate from
        # certifying itself through the ||x||-scaled branch below: a c7 float32
        # run was measured reporting converged=True with normr/||b||=1.2e17 and
        # ||x||=inf, i.e. the gate would have accepted coefficients that were
        # 1e17 times wrong.
        return bool((finite & bool(np.isfinite(norma)) & (residual_norm <= rhs_norm)
                    & ((normal_norm <= atol_eff * norma * residual_norm)
                       | (residual_norm <= btol_eff * rhs_norm + atol_eff * norma * normx))).item())
    converged = meets_tolerance()
    n_iter = 0
    stop_reason = "iteration_limit"
    honest_only = False        # the recurrence test has been caught lying
    next_verify = 0
    best_true = None           # smallest honest residual reached so far
    # Best honest iterate by the PRIMARY criterion (normar <= atol*norma*normr),
    # which is the one a float32 solve stalls on, plus the stall bookkeeping that
    # turns "stopped improving" into a measured floor.
    best_iterate = None        # (criterion value, x, normr, normar)
    stalls = 0
    stall_limit = max(1, int(os.environ.get("PHEASY_CGLS_STALL_POINTS", "5")))
    measured_floor = None
    # A solve whose CHEAP test never fires never enters the honest block below,
    # so its stall detector never runs and an arithmetic floor is reported as a
    # bare "iteration_limit" with no number attached.  That is the production MgC
    # OLS case (454656x69487, float32): 20000 iterations reached a criterion of
    # 1.33e-03 against a 1.19e-06 analytic floor -- three orders below what the
    # arithmetic can express -- so the fit was refused with no way to see why or
    # what to ask for instead.  An unconditional periodic honest probe therefore
    # runs from probe_start onwards, records the best criterion and the iterate
    # that reached it, and reports both.  It CERTIFIES early when that floor is
    # inside _CGLS_FLOOR_MAX, but it never ends the run on a floor above the cap:
    # the recurrence is still creeping down there (measured 5.31e-03 at 10000
    # iterations -> 1.33e-03 at 20000), so stopping would deliver a worse iterate
    # than the remaining budget can still reach.
    probe_window = min(_CGLS_PROBE_WINDOW_MAX, max(int(maxiter) // 2, verify_every))
    probe_start = int(os.environ.get("PHEASY_CGLS_PROBE_START", probe_window))
    next_probe = probe_start
    probe_count = 0
    stall_floor = None         # best criterion an honest probe reached
    stall_iteration = None

    def _record_honest(r_norm, n_norm):
        """Update the best-criterion bookkeeping from one honest measurement.

        Returns (rel, improved) where rel = ||A^T r|| / (||A|| * ||r||) is the
        quantity the primary criterion tests and the one a float32 solve stalls
        on.  Shared by the honest re-verification and the periodic probe so both
        decide "improving" by the same margin (see _CGLS_STALL_MARGIN).
        """
        nonlocal best_true, best_iterate, stalls
        if bool((r_norm <= rhs_norm).item()):
            best_true = (r_norm if best_true is None
                         else torch.minimum(best_true, r_norm))
        _denom = float((norma * r_norm).item())
        rel = (float(n_norm.item()) / _denom) if _denom > 0 else float("inf")
        improved = (best_iterate is None
                    or rel < best_iterate[0] * (1.0 - _CGLS_STALL_MARGIN))
        if improved:
            best_iterate = (rel, x.clone(), r_norm, n_norm)
            stalls = 0
        else:
            stalls += 1
        return rel, improved
    for it in range(int(maxiter)):
        if not honest_only and not converged and it >= next_probe:
            # Periodic honest probe: the cheap recurrence test has not fired, so
            # nothing else in this loop measures the TRUE residual.  This is what
            # turns an arithmetic floor into a reported number instead of a bare
            # iteration_limit (see the probe_window note above).
            probe_r = y - A.matvec(x)
            probe_rn = torch.linalg.vector_norm(probe_r)
            probe_nn = torch.linalg.vector_norm(A.rmatvec(probe_r))
            probe_count += 1
            rel, _improved = _record_honest(probe_rn, probe_nn)
            if stall_floor is None or rel < stall_floor:
                stall_floor = rel
                stall_iteration = it
            next_probe = it + verify_every
            if stalls >= stall_limit and stall_floor is not None:
                if stall_floor <= _CGLS_FLOOR_MAX:
                    # Certifiable: promote the measured floor exactly as the
                    # honest path does and stop here instead of at the cap.
                    measured_floor = stall_floor
                    _, x, residual_norm, normal_norm = best_iterate
                    _accepted = min(measured_floor * (1.0 + _CGLS_FLOOR_MARGIN),
                                    _CGLS_FLOOR_MAX)
                    atol_eff = max(atol_eff, _accepted)
                    btol_eff = max(btol_eff, _accepted)
                    converged = True
                    stop_reason = "precision_floor"
                    break
                # Above the cap: keep spending the budget (the recurrence is still
                # creeping down) but remember the number for the verdict below,
                # and re-arm the window so the probe does not re-fire every step.
                stalls = 0
        if (converged or honest_only) and it >= next_verify:
            true_r = y - A.matvec(x)
            residual_norm = torch.linalg.vector_norm(true_r)
            normal_norm = torch.linalg.vector_norm(A.rmatvec(true_r))
            if meets_tolerance():
                converged = True
                stop_reason = "converged_on_true_residual" if honest_only else "converged"
                break
            if best_true is not None and bool((residual_norm > 10.0 * rhs_norm).item()):
                # The recurrence has come apart: report the breakdown instead of
                # grinding to maxiter and returning the wreckage.
                converged = False
                stop_reason = "residual_growth"
                break
            # How close this iterate came to the primary criterion.  Stagnation of
            # THIS quantity is what defines the floor in this precision; the
            # analytic eps*cond estimate above is only where the search starts.
            # _record_honest also maintains best_true (the residual-growth guard
            # above reads it) so both paths decide "improving" identically.
            _record_honest(residual_norm, normal_norm)
            if stalls >= stall_limit:
                # The iteration has stopped improving, so what it reached is what
                # this working precision can express.  Deliver the BEST iterate
                # (the current one is usually worse) and certify against the
                # MEASURED floor instead of grinding to maxiter and reporting
                # iteration_limit with an uncertified vector.  Refuse when even
                # the measured floor is past _CGLS_FLOOR_MAX: at that point the
                # system is not solvable in this precision and a certificate would
                # mean nothing.
                measured_floor = best_iterate[0]
                _, x, residual_norm, normal_norm = best_iterate
                if measured_floor <= _CGLS_FLOOR_MAX:
                    # The certificate is re-derived from x after the loop, and
                    # that re-derivation only reproduces to ~2% (see
                    # _CGLS_STALL_MARGIN), so certifying AT the measured value
                    # leaves the verdict inside the measurement noise: measured
                    # on c7, atol_effective 7.825285e-06 against a re-derived
                    # normar 3.242940e-06 missed the test by 1.25%.  The floor is
                    # therefore brought with _CGLS_FLOOR_MARGIN headroom -- still
                    # two orders below the 1e-3 cap, so it certifies far less
                    # than the cap allows.
                    accepted = min(measured_floor * (1.0 + _CGLS_FLOOR_MARGIN),
                                   _CGLS_FLOOR_MAX)
                    atol_eff = max(atol_eff, accepted)
                    btol_eff = max(btol_eff, accepted)
                    converged = True
                else:
                    converged = False
                stop_reason = "precision_floor"
                break
            # Restart the recurrence from the honest residual and keep going.
            # Returning here is what discarded correct float32 coefficients: the
            # recurrence had drifted, so the true certificate could not match it.
            r = true_r
            s = A.rmatvec(r)
            normal_norm = torch.linalg.vector_norm(s)
            p = s.clone()
            gamma = torch.dot(s, s)
            converged = False
            honest_only = True
            next_verify = it + verify_every
        if converged:
            break
        q = A.matvec(p)
        denom = torch.dot(q, q)
        if bool((~torch.isfinite(denom) | (denom <= 0)).item()):
            stop_reason = "invalid_search_direction"
            break
        step = gamma / denom
        x = x + step * p
        r = r - step * q
        s_new = A.rmatvec(r)
        gamma_new = torch.dot(s_new, s_new)
        residual_norm = torch.linalg.vector_norm(r)
        normal_norm = torch.linalg.vector_norm(s_new)
        n_iter = it + 1
        if not honest_only:
            # Do NOT stop here: the verdict is only a recurrence estimate and has
            # to survive the certificate check at the top of the next iteration.
            converged = meets_tolerance()
        if bool((~torch.isfinite(gamma_new) | (gamma <= 0)).item()):
            stop_reason = "invalid_gradient_recurrence"
            break
        p = s_new + (gamma_new / gamma) * p
        s = s_new
        gamma = gamma_new
    # Certify the delivered coefficients, not only the recursively updated residual.
    recurrence_converged = bool(converged)
    true_r = y - A.matvec(x)
    residual_norm = torch.linalg.vector_norm(true_r)
    normal_norm = torch.linalg.vector_norm(A.rmatvec(true_r))
    certified = meets_tolerance()
    _denom_end = float((norma * residual_norm).item())
    _rel_end = ((float(normal_norm.item()) / _denom_end) if _denom_end > 0
                else float("inf"))
    # Deliver the best iterate the honest measurements saw when it beats the
    # last one.  Once the criterion has stalled the loop's final iterate is
    # usually the worse of the two, and the certificate is recomputed from
    # whatever vector is actually returned, so this can only help.  The stall
    # valve above has always done this for the honest path.
    if not certified and best_iterate is not None:
        _denom_now = float((norma * residual_norm).item())
        _rel_now = ((float(normal_norm.item()) / _denom_now) if _denom_now > 0
                    else float("inf"))
        if best_iterate[0] < _rel_now * (1.0 - _CGLS_STALL_MARGIN):
            x = best_iterate[1]
            true_r = y - A.matvec(x)
            residual_norm = torch.linalg.vector_norm(true_r)
            normal_norm = torch.linalg.vector_norm(A.rmatvec(true_r))
            certified = meets_tolerance()
            _denom_end = float((norma * residual_norm).item())
            _rel_end = ((float(normal_norm.item()) / _denom_end) if _denom_end > 0
                        else float("inf"))
    if certified:
        # "precision_floor" is a verdict of its own: the iteration stagnated at
        # the measured floor and atol_eff was raised to it, so re-testing must
        # not relabel it as a stalling recurrence check.
        if stop_reason not in ("converged", "converged_on_true_residual",
                               "precision_floor"):
            stop_reason = "converged_on_true_residual"
    elif stop_reason == "precision_floor":
        # Keep the verdict and its reason: a measured-floor certificate that the
        # re-derivation cannot reproduce is still a measured floor, not a broken
        # recurrence, and conflating them is what made this path look identical
        # to the failure it replaced.
        pass
    elif recurrence_converged or stop_reason in ("converged", "converged_on_true_residual"):
        stop_reason = "true_residual_check_failed"
    elif (stop_reason == "iteration_limit" and stall_floor is not None
          and _rel_end >= stall_floor * (1.0 - _CGLS_STALL_MARGIN)):
        # The honest probes stopped improving ABOVE the cap, and the DELIVERED
        # iterate is no better than what they reached: this precision cannot
        # certify the system.  That is a different verdict from "ran out of
        # iterations while still improving", and reporting it as iteration_limit
        # hid the number that makes it actionable (raise the tolerance to it).
        # The comparison uses the delivered criterion, not the stall counter: the
        # probe re-arms its window after a stall, so a LATER improvement means the
        # stall was spurious and the budget really did run out mid-progress.
        stop_reason = "stall_above_floor"
    converged = certified
    info = {"solver": "GPU CGLS", "itn": n_iter, "n_iter": n_iter,
            "residual_certificate": "recomputed_y_minus_Ax",
            "stop_reason": stop_reason,
            "norma": float(norma), "norma_estimator": "spectral_power_10",
            "normb": float(rhs_norm.item()), "normx": float(torch.linalg.vector_norm(x).item()),
            "criterion": "normar<=atol*norma*normr or normr<=btol*normb+atol*norma*normx",
            "normr": float(residual_norm.item()), "normar": float(normal_norm.item()),
            "converged": bool(converged), "device": str(device),
            "backend": "gpu_resident_iterative", "atol": float(atol),
            "btol": float(btol), "maxiter": int(maxiter),
            # Requested vs applied, so a manifest never hides a loosened test:
            # the applied values are what the certificate above actually used.
            "atol_effective": float(atol_eff), "btol_effective": float(btol_eff),
            "tolerance_floor": float(tol_floor),
            "tolerance_floor_dtype": str(y.dtype).replace("torch.", ""),
            # Analytic floor versus what the iteration actually reached: the
            # second is what a "precision_floor" verdict certified against.
            "tolerance_floor_measured": (None if measured_floor is None
                                         else float(measured_floor)),
            "honest_residual_check": ("recurrence_replaced" if honest_only
                                      else "recurrence_agreed"),
            # What the unconditional honest probe saw, whether or not it could be
            # certified: this is the number a caller needs to choose a reachable
            # tolerance instead of guessing (see probe_window).
            "probe_start": int(probe_start), "probe_count": int(probe_count),
            "stall_floor": (None if stall_floor is None else float(stall_floor)),
            "stall_iteration": (None if stall_iteration is None
                                else int(stall_iteration)),
            "criterion_value": ((float(normal_norm.item())
                                 / float((norma * residual_norm).item()))
                                if float((norma * residual_norm).item()) > 0
                                else float("inf")),
            "best_criterion": (None if best_iterate is None
                               else float(best_iterate[0]))}
    if not converged:
        import warnings
        _hint = ""
        if stall_floor is not None:
            # Say what the number MEANS, in the same spirit as the FISTA valve's
            # "Re-run with --tol >= X": a caller reading only the warning should not
            # have to guess whether the budget or the arithmetic was the limit.  The
            # caveat is not decoration -- the criterion is relative, so a raised
            # tolerance certifies the iterate the probes reached, which was measured
            # 20x worse in residual than the run that reached the floor itself.
            _hint = (", best_criterion=%.3e at iteration %s (analytic floor %.3e,"
                     " certifiable cap %.3e): this is a PRECISION limit, not a"
                     " budget limit, and requesting atol >= %.3e would certify the"
                     " iterate the probe reached -- not a better one"
                     % (stall_floor, stall_iteration, tol_floor, _CGLS_FLOOR_MAX,
                        stall_floor * (1.0 + _CGLS_FLOOR_MARGIN)))
        warnings.warn("GPU CGLS did not converge: reason=%s, iterations=%d, normr=%g, normar=%g%s" %
                      (info["stop_reason"], n_iter, info["normr"], info["normar"], _hint),
                      RuntimeWarning, stacklevel=2)
    return x, info


# ---- FISTA measured-floor valve (mirrors the CGLS stall valve above) --------
# A relative KKT certificate (max violation / |A^T y|_max) computed in float32
# stalls well above the requested tolerance on large, ill-conditioned problems:
# measured on the Mg8C120 c7/c4 LASSO fit, 20000 iterations reached 2.566e-4 and
# 60000 iterations reached 2.523e-4 against a requested 1e-4 -- tripling the
# budget moved the certificate by 1.7 %, so the run could not be certified and
# the entire budget was spent for nothing (3364 s and 2759 s of GPU time, both
# rejected).  LSMR has its own floor (_lsmr_tol) and the debias CGLS has
# _CGLS_STALL_MARGIN; FISTA had neither, so it always ground to max_iter.
_FISTA_STALL_MARGIN = 0.05      # < 5 % certificate improvement = no progress
_FISTA_FLOOR_MARGIN = 0.10      # headroom once the floor becomes the certificate
_FISTA_FLOOR_MAX = 1e-2         # past this the system is unsolvable in this precision
# How long "no progress" must last before the run stops paying for more.  FISTA's
# tail is sublinear, NOT flat: on the c2=7 A/c3=4 A Mg8C120 operator the relative
# KKT certificate runs 9.90e-04 at iteration 80 -> 2.98e-04 at 7160 -> 2.52e-04 at
# 60000, i.e. it keeps creeping down by ~16 % per 1000 iterations for tens of
# thousands of iterations.  A short window therefore declares a "floor" that is
# not one: with the first 60-iteration window this valve stopped at iteration 80
# and delivered ||x||=460.4 / nnz 37285 against the converged run's 608.4 / 25847.
# The window is what makes the verdict meaningful, so it scales with the budget:
# min(_FISTA_STALL_WINDOW_MAX, max(_FISTA_STALL_WINDOW_MIN, max_iter // 2)) --
# never before 8000 iterations unless the whole budget is smaller than that.
_FISTA_STALL_WINDOW_MAX = 8000  # iterations without progress that may end a run
_FISTA_STALL_WINDOW_MIN = 200
_FISTA_KKT_EVERY = 20           # iterations between certificate checks
# The objective-monotonicity test behind the step-size backtracking used a
# 1e-9 *relative* slack.  In torch.float32 the objective carries its own
# evaluation noise (~1e-7 relative), so noise-level increases counted as real
# increases and doubled L on every check until it overflowed: measured on the
# Mg8C120 c2=7 A LASSO, lipschitz=Infinity after 292 inflations, after which
# 1/L == 0 froze the iterate for the rest of a 60000-iteration budget that was
# then rejected.  Compare against the dtype's own resolution instead.
_FISTA_F_REFINE = 8.0           # objective-slack multiplier, in units of eps(dtype)
_FISTA_MAX_INFLATIONS = 40      # beyond this L is chasing rounding noise, not curvature


def _fista_floor_cfg(auto_floor=None, max_iter=None):
    """(use_valve, stall_points, floor_max, floor_margin) for the FISTA valve.

    PHEASY_FISTA_AUTO_FLOOR=0 restores the pre-valve behaviour (grind to
    max_iter, then report non-convergence).  PHEASY_FISTA_STALL_POINTS is how
    many consecutive KKT checks (one every _FISTA_KKT_EVERY iterations) without
    a >= 5 % improvement declare the practical floor; left unset it is derived
    from the budget as min(8000, max(200, max_iter // 2)) / 20, because a short
    window on a sublinear tail certifies a "floor" that is still descending (see
    the constants above).  PHEASY_FISTA_FLOOR_MAX is where stopping early wins
    over certifying a tolerance this precision cannot express.

    auto_floor=False is for the callers where hitting the cap is BY DESIGN (CV
    folds, the alpha path walk): there the cheap approximate solve is the point
    and the hit-cap diagnostic is information, not waste.
    """
    def _num(name, default):
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    env_on = os.environ.get("PHEASY_FISTA_AUTO_FLOOR", "1").strip().lower() in (
        "1", "true", "yes", "on")
    use = env_on if auto_floor is None else (bool(auto_floor) and env_on)
    raw_points = os.environ.get("PHEASY_FISTA_STALL_POINTS")
    if raw_points:
        points = max(1, int(_num("PHEASY_FISTA_STALL_POINTS", 0)))
    elif max_iter is None:
        points = 3
    else:
        budget = max(1, int(max_iter))
        window = min(_FISTA_STALL_WINDOW_MAX,
                     max(_FISTA_STALL_WINDOW_MIN, budget // 2))
        points = max(1, -(-window // _FISTA_KKT_EVERY))
    return (use,
            points,
            _num("PHEASY_FISTA_FLOOR_MAX", _FISTA_FLOOR_MAX),
            _num("PHEASY_FISTA_FLOOR_MARGIN", _FISTA_FLOOR_MARGIN))


def _fista_twolevel(A, y, alpha, x0, max_iter, tol, lipschitz, rows=None, penalty_weights=None, n_samples=None,
                    auto_floor=None):
    """Device FISTA, adaptive restart and exact L1 KKT certificate.

    No NumPy or vector host transfers in the iteration loop. Scalar syncs are
    limited to backtracking acceptance and a KKT check every 20 iterations.
    Masked residuals give the exact training objective without fold CSR copies.

    auto_floor: run the measured-floor valve (see _fista_floor_cfg).  None =
    follow PHEASY_FISTA_AUTO_FLOOR (default on), False = never valve (CV/path).
    """
    torch = A.torch
    n = A.shape[0] if rows is None else rows.numel()
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("LASSO alpha must be finite and nonnegative")
    mask = torch.ones_like(y) if rows is None else torch.zeros_like(y)
    if rows is not None:
        mask[rows] = 1
    rhs = A.rmatvec(y * mask)
    scale = torch.clamp(rhs.abs().max(), min=torch.finfo(y.dtype).tiny)
    penalty = alpha * (n if n_samples is None else n_samples)
    if penalty_weights is None:
        penalty_vec = torch.full((A.shape[1],), penalty, dtype=y.dtype, device=y.device)
    else:
        # Keep adaptive weights resident when provided as a CUDA tensor.
        if isinstance(penalty_weights, torch.Tensor):
            penalty_vec = penalty * penalty_weights.to(device=y.device, dtype=y.dtype)
        else:
            penalty_vec = penalty * torch.as_tensor(np.asarray(penalty_weights), dtype=y.dtype, device=y.device)
        if penalty_vec.numel() != A.shape[1] or not bool(torch.isfinite(penalty_vec).all().item()) or bool((penalty_vec < 0).any().item()):
            raise ValueError("penalty_weights must be finite and nonnegative with one value per feature")
    x = torch.zeros(A.shape[1], dtype=y.dtype, device=y.device) if x0 is None else x0.clone()
    z = x.clone()
    momentum = torch.ones((), dtype=y.dtype, device=y.device)
    L = torch.as_tensor(lipschitz, dtype=y.dtype, device=y.device).clone()
    if not bool(torch.isfinite(L).item()) or float(L) <= 0:
        raise ValueError("FISTA step estimate must be finite and positive")

    def kkt(coef):
        gradient = A.rmatvec((A.matvec(coef) - y) * mask)
        violation = torch.where(coef != 0, (gradient + penalty_vec * coef.sign()).abs(),
                                torch.clamp(gradient.abs() - penalty_vec, min=0))
        return violation.max() / scale

    converged = False
    n_iter = 0
    prev_f = None
    n_inflate = 0
    # --- measured-floor valve state (see _fista_floor_cfg) ------------------
    _valve, _stall_limit, _floor_max, _floor_margin = _fista_floor_cfg(
        auto_floor, max_iter)
    try:
        _max_inflate = int(float(os.environ.get("PHEASY_FISTA_MAX_INFLATIONS",
                                                _FISTA_MAX_INFLATIONS)))
    except (TypeError, ValueError):
        _max_inflate = _FISTA_MAX_INFLATIONS
    tol_requested = float(tol)
    tol_eff = float(tol)
    stop_reason = "iteration_limit"
    best_kkt = None
    best_x = None
    stalls = 0
    measured_floor = None
    for it in range(int(max_iter)):
        n_iter = it + 1
        residual = (A.matvec(z) - y) * mask
        grad = A.rmatvec(residual)
        # lipschitz() returns Rayleigh*1.05 from 40 power iterations, which
        # converge to lambda_max from BELOW: it is a step estimate, not a
        # certificate.  The per-step backtracking was dropped for speed, so the
        # periodic monotonicity guard below is what keeps a too-large step from
        # quietly drifting to max_iter (see the KKT/nonfinite check as well).
        candidate = z - grad / L
        x_new = candidate.sign() * torch.clamp(candidate.abs() - penalty_vec / L, min=0)
        restart = torch.dot(z - x_new, x_new - x) > 0
        next_momentum = (1 + torch.sqrt(1 + 4 * momentum.square())) / 2
        z = torch.where(restart, x_new, x_new + ((momentum - 1) / next_momentum) * (x_new - x))
        momentum = torch.where(restart, torch.ones_like(momentum), next_momentum)
        x = x_new
        if n_iter % 20 == 0:
            certificate = kkt(x)
            if not bool(torch.isfinite(certificate).item()):
                raise RuntimeError("Resident FISTA diverged (nonfinite KKT certificate); Lipschitz estimate too small")
            rx = (A.matvec(x) - y) * mask
            f_new = float((0.5 * torch.dot(rx, rx)
                           + (penalty_vec * x.abs()).sum()).item())
            f_slack = _FISTA_F_REFINE * float(torch.finfo(y.dtype).eps)
            if prev_f is not None and f_new > prev_f + f_slack * (abs(prev_f) + 1e-300):
                # Step too large: inflate L, restart the momentum from the
                # current iterate.  (One measured inflation over 800 iterations
                # on the reference problems, so this costs nothing when L is
                # already adequate.)  The slack is dtype-scaled because a
                # float32 objective cannot resolve differences below ~1e-7.
                L = L * 2.0
                z = x.clone()
                momentum = torch.ones((), dtype=y.dtype, device=y.device)
                prev_f = None
                n_inflate += 1
            else:
                prev_f = f_new
            cert = float(certificate.item())
            if cert <= tol_eff:
                converged = True
                stop_reason = ("converged" if tol_eff <= tol_requested
                               else "converged_on_raised_tol")
                break
            if not _valve:
                continue
            # Solver-health guard, BEFORE the floor valve: a frozen iterate
            # (1/L == 0 after runaway inflations) also leaves the certificate
            # flat, and certifying that as a measured floor would label a
            # step-size failure as a precision limit.
            if not bool(torch.isfinite(L).item()) or n_inflate >= _max_inflate:
                import warnings
                warnings.warn(
                    "FISTA step-size estimate diverged: lipschitz=%g (finite=%s) "
                    "after %d inflations in %s (budget %d).  Backtracking doubled L "
                    "every time the objective rose, i.e. it is chasing rounding "
                    "noise rather than curvature, and at this L the step 1/L can no "
                    "longer move the iterate.  Stopping at iteration %d of "
                    "max_iter=%d WITHOUT certifying -- a frozen iterate is not a "
                    "precision floor.  Re-estimate L (raise "
                    "PHEASY_LIPSCHITZ_POWER_P, lower "
                    "PHEASY_FISTA_LIPSCHITZ_SAFETY) or check the operator."
                    % (float(L), bool(np.isfinite(float(L))), n_inflate,
                       str(y.dtype), int(_max_inflate), n_iter, int(max_iter)),
                    RuntimeWarning, stacklevel=2)
                stop_reason = "step_size_diverged"
                break
            # Certificate stopped improving: measure the floor instead of
            # assuming the requested tolerance is reachable.  Same idiom as the
            # CGLS stall valve -- keep the BEST iterate, because the one the loop
            # happens to stop on is usually worse.
            if best_kkt is None or cert < best_kkt * (1.0 - _FISTA_STALL_MARGIN):
                best_kkt = cert
                best_x = x.clone()
                stalls = 0
            else:
                stalls += 1
            if stalls >= _stall_limit:
                measured_floor = best_kkt
                if measured_floor <= _floor_max:
                    accepted = min(measured_floor * (1.0 + _floor_margin), _floor_max)
                    if accepted > tol_eff:
                        import warnings
                        warnings.warn(
                            "FISTA cannot reach tol=%g in %s: the relative KKT "
                            "certificate improved by less than %g%% in every "
                            "%d-iteration check for the last %d iterations "
                            "(best %g at iteration %d).  The request is below what "
                            "this precision can express, so the run certifies the "
                            "BEST iterate at tol_effective=%g instead of spending "
                            "the remaining %d of max_iter=%d iterations on a "
                            "stopping test it cannot satisfy.  Request --tol >= %g, "
                            "or shorten the window with PHEASY_FISTA_STALL_POINTS, "
                            "to make the request honest."
                            % (tol_requested, str(y.dtype),
                               100.0 * _FISTA_STALL_MARGIN, _FISTA_KKT_EVERY,
                               int(_stall_limit) * _FISTA_KKT_EVERY, measured_floor,
                               n_iter, accepted, int(max_iter) - n_iter,
                               int(max_iter), accepted), RuntimeWarning, stacklevel=2)
                        tol_eff = accepted
                    x = best_x
                    stop_reason = "converged_measured_floor"
                    break
                import warnings
                warnings.warn(
                    "FISTA stalled at relative KKT=%g, past "
                    "PHEASY_FISTA_FLOOR_MAX=%g: this precision cannot certify any "
                    "meaningful tolerance here, so the run stops at iteration %d "
                    "instead of grinding to max_iter=%d.  Re-run with --tol >= %g or "
                    "fix the conditioning (the certificate is not a tolerance "
                    "problem)." % (measured_floor, _floor_max, n_iter,
                                   int(max_iter), measured_floor * 1.05),
                    RuntimeWarning, stacklevel=2)
                x = best_x
                stop_reason = "stall_above_floor"
                break
    certificate = kkt(x)
    value = float(certificate.item())
    if stop_reason == "converged_measured_floor":
        # Certify against the measured floor (the re-derived certificate is the
        # same deterministic quantity, with _FISTA_FLOOR_MARGIN headroom for the
        # +-2 % kernel-accumulation spread the CGLS valve documents).
        converged = bool(np.isfinite(value) and value <= tol_eff)
        if not converged:
            stop_reason = "stall_above_floor"
    else:
        converged = bool(np.isfinite(value) and value <= tol)
    info = dict(n_iter=n_iter, converged=converged, kkt_relative=value,
                lipschitz=float(L), lipschitz_inflations=n_inflate,
                stop_reason=stop_reason, tol_requested=tol_requested,
                tol_effective=tol_eff, measured_floor=measured_floor,
                stall_points=int(_stall_limit))
    if not converged:
        import warnings
        warnings.warn("Resident FISTA did not converge: iterations=%d, relative KKT=%g, tol=%g"
                      " (stop_reason=%s)" %
                      (n_iter, value, tol, stop_reason), RuntimeWarning, stacklevel=2)
    return x, info


def _resident_cv_devices():
    """Explicit opt-in; IDs are logical indices after CUDA_VISIBLE_DEVICES.

    DEVICES order defines the primary (first) device. NGPU truncates that list,
    or selects the primary PHEASY_GPU_DEVICE followed by other visible devices.
    With neither control set, retain the existing single-device behavior.
    Invalid explicit requests fail before factor allocation, even if an invalid
    ID would later be excluded by NGPU or the fold-count cap. Selected active
    devices are never filtered by free memory: an upload/preflight/kernel error
    aborts the fit, without retrying on another device or falling back to CPU.
    """
    raw = os.environ.get("PHEASY_GPU_DEVICES", "").strip()
    count = os.environ.get("PHEASY_GPU_NGPU", "").strip()
    if not raw and not count:
        return [device()]
    n = _torch().cuda.device_count()
    try:
        ids = [int(v.strip()) for v in raw.split(",")] if raw else None
        limit = int(count) if count else None
    except ValueError as exc:
        raise ValueError("PHEASY_GPU_DEVICES and PHEASY_GPU_NGPU must contain integer IDs/counts") from exc
    if ids is not None and (len(set(ids)) != len(ids) or any(d < 0 or d >= n for d in ids)):
        raise ValueError("PHEASY_GPU_DEVICES must contain unique visible CUDA device IDs")
    if ids is None:
        first = _torch().device(device()).index
        if first is None or not 0 <= first < n:
            raise ValueError("PHEASY_GPU_DEVICE must name a visible CUDA device")
        ids = [first] + [d for d in range(n) if d != first]
    if limit is not None:
        if not 1 <= limit <= len(ids):
            raise ValueError("PHEASY_GPU_NGPU must be positive and not exceed selected visible devices")
        ids = ids[:limit]
    return [_torch().device("cuda:%d" % d) for d in ids]


def _resident_device_context(dev):
    from contextlib import nullcontext
    torch = _torch()
    return torch.cuda.device(dev) if torch.device(dev).type == "cuda" else nullcontext()


def _dynamic_fold_map(resources, folds, solve):
    """One long-lived dispatcher per device, claiming the next fold immediately.

    Return in fold order, not completion order. No static chunks or batch
    barriers. Join all workers on failure before the caller releases resources.
    Threads share resident tensors; never use a process pool or fork CUDA.
    This scheduling seam is CPU-testable and does not mutate process GPU mode.
    """
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue, Empty
    from threading import Event
    queue = Queue()
    for k, fold in enumerate(folds):
        queue.put((k, fold))
    results = [None] * len(folds)
    failed = Event()

    def run(resource):
        while not failed.is_set():
            try:
                k, fold = queue.get_nowait()
            except Empty:
                return
            try:
                results[k] = solve(resource, k, fold)
            except BaseException:
                failed.set()
                raise

    if len(resources) == 1:
        run(resources[0])  # preserve synchronous single-GPU execution
    else:
        with ThreadPoolExecutor(max_workers=len(resources)) as pool:
            futures = [pool.submit(run, resource) for resource in resources]
            for future in futures:
                future.result()
    return results


class GpuTwoLevelLassoCV(GpuLassoCV):
    """Opt-in two-level sparse resident CV; outputs physical coefficients.

    One CUDA device by default; opt-in dynamic folds via PHEASY_GPU_DEVICES
    and/or PHEASY_GPU_NGPU. Each card holds full factors; refit uses the first
    selected device. Float64, no intercept/sample weights. CPU grouped split
    construction is shared with the existing solver; all fold gradients,
    predictions, MSE reductions, normalization and alpha selection use Torch.
    Full-data normalization (not per-fold normalization), n_train*alpha penalty,
    descending warm starts, and smallest-alpha exact tie-break are preserved.
    PHEASY_LASSO_TIE_RTOL affects flat-tail diagnostics only.
    """
    def __init__(self, *args, standardize=False, adaptive=False, gamma=1.0,
                 init_alpha=1e-3, eps=1e-8, nalpha=None, decades=4.0,
                 alpha_auto=True, unpenalized=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.standardize = standardize
        self.adaptive = bool(adaptive)
        self.gamma = float(gamma)
        self.init_alpha = float(init_alpha)
        self.eps = float(eps)
        self.nalpha = int(nalpha) if nalpha else len(self.alphas)
        self.decades = float(decades)
        self.alpha_auto = bool(alpha_auto)
        # [HARM_DENSE] bool mask of columns with no L1 penalty (FC2 block)
        self.unpenalized = unpenalized
        self.penalty_weights_ = None

    def fit(self, A, y, sample_weight=None, retain_operator=False):
        if not enabled() or not available():
            raise RuntimeError("Resident two-level LASSO requires enabled CUDA; no CPU fallback")
        devices = _resident_cv_devices()
        owned = []
        try:
            with _resident_device_context(devices[0]):
                return self._fit_resident(A, y, devices, owned, sample_weight, retain_operator)
        except BaseException:
            op = getattr(self, "_operator", None)
            if op is not None:
                op.close()
                self._operator = None
            raise
        finally:
            for operator in owned:
                operator.close()

    def _fit_resident(self, A, y, devices, owned, sample_weight, retain_operator):
        import time
        from .optimizer import _make_cv_splits
        started = time.monotonic()
        if sample_weight is not None or self.fit_intercept:
            raise NotImplementedError("Resident GPU ALASSO does not support sample weights or intercept")
        if self.alphas.size == 0 or not np.isfinite(self.alphas).all() or (self.alphas < 0).any():
            raise ValueError("alphas must be nonempty, finite and nonnegative")
        if self.max_iter < 1 or self.tol <= 0:
            raise ValueError("max_iter and tol must be positive")
        splits = _make_cv_splits(A.shape[0], self.cv, self.rand_seed, self.group_size)
        devices = devices[:max(1, len(splits))]
        print("[gpu_resident] uploading factors shape=%s on %d device(s); CUDA required, no CPU fallback"
              % (A.shape, len(devices)), flush=True)
        # [ACC] canonicalize the host factors ONCE and upload every card from
        # the same arrays: a replica used to re-run tocsr(copy=True)/
        # sum_duplicates()/sort_indices() over the whole SM_prime and NS, i.e.
        # one extra full host pass plus a transient host-RAM spike the size of
        # the factors per card.  The canonical copy is released as soon as the
        # last card has uploaded (the device copies are independent).
        host_factors = _canonical_twolevel_host(A)
        _mem_trace("resident host factors ready (lazy adjoint=%s)"
                   % type(host_factors[2]).__name__)
        replicas = []
        try:
            try:
                op = GpuTwoLevelOperator(A, device_id=devices[0],
                                         host_factors=host_factors)
                owned.append(op)
                for dev in devices[1:]:
                    with _resident_device_context(dev):
                        replica = GpuTwoLevelOperator(A, device_id=dev,
                                                      host_factors=host_factors)
                        owned.append(replica)
                        replicas.append(replica)
            except ResidentFootprintError as _exc:
                # A FULL replica does not fit on every selected card.  Instead of
                # dropping the whole CV to the CPU FISTA path (what used to
                # happen), shard ONE operator across the same devices: folds then
                # run sequentially, but every fold solve stays on the GPU.  The
                # debias subset solve works unchanged because GpuSubsetOperator
                # delegates to base.matvec/rmatvec.
                for _o in owned:
                    _o.close()
                owned.clear()
                del replicas[:]
                print("[gpu_resident] full per-card replica does not fit (%s); "
                      "falling back to ONE sharded operator across %d device(s) "
                      "-- CV folds run sequentially, all solves stay on the GPU"
                      % (_exc, len(devices)), flush=True)
                op = GpuTwoLevelOperator(A, device_ids=list(devices),
                                         host_factors=host_factors)
                owned.append(op)
        finally:
            host_factors = None
        _mem_trace("resident factors uploaded")
        print("[gpu_resident] factors ready on %d device(s) primary=%s shards=%d "
              "dtype=%s estimated_peak_bytes=%d elapsed=%.2fs" %
              (len(devices), op.device, op._n_shards, op._value_dtype,
               op.estimated_peak_bytes, time.monotonic() - started), flush=True)
        torch = op.torch
        yt = torch.as_tensor(np.asarray(y).ravel(), dtype=op._value_dtype, device=op.device)
        if yt.numel() != A.shape[0] or not bool(torch.isfinite(yt).all().item()):
            raise ValueError("target shape or finite values invalid")
        if self.standardize:
            print("[gpu_resident] exact normalization started", flush=True)
            op.normalize()
            print("[gpu_resident] normalization done elapsed=%.2fs" % (time.monotonic() - started), flush=True)
        # [HARM_DENSE] internal Jacobi scaling for an UNstandardized fit.  The
        # free block stays inside the matrix-free iteration here, and on raw
        # columns the FC2/FC3 norm ratio (~40x) squares into the curvature spread
        # FISTA has to crawl through.  Iterate in z = s*x (s = column norms) and
        # carry the model in the weights, w_z = w_x / s: an exact
        # reparametrization, so the unstandardized objective is unchanged.
        jac = None
        if (not self.standardize and self.unpenalized is not None
                and bool(np.asarray(self.unpenalized, dtype=bool).any())
                and os.environ.get("PHEASY_HARM_DENSE_JACOBI", "1").strip().lower()
                not in ("0", "false", "no", "off")):
            op.normalize()
            jac = op.scale.clone()
            print("[HARM_DENSE] resident: internal Jacobi scaling (unstandardized "
                  "model kept through the L1 weights)", flush=True)
        print("[gpu_resident] Lipschitz estimate started", flush=True)
        L = op.lipschitz()
        print("[gpu_resident] Lipschitz estimate ready elapsed=%.2fs" % (time.monotonic() - started), flush=True)
        # [HARM_DENSE] unpenalized block.  Three things change, all derived from
        # ONE resident OLS of y on the free columns (x_top):
        #   * the L1 weight of every free column is exactly 0 (LASSO and ALASSO);
        #   * the KKT thresholds of the alpha grid are taken over the PENALIZED
        #     columns on the residual y - A x_top.  The old ALASSO rule divided
        #     by clamp(w, tiny) = tiny on the free block, overflowed to inf, and
        #     the non-finite amax then silently kept the MANUAL grid;
        #   * x_top is the exact solution at alpha >= alpha_max, so it seeds the
        #     full path, and each fold is seeded with the same solve on its own
        #     training rows (no validation leakage).
        free_np = None
        if self.unpenalized is not None:
            _fm = np.asarray(self.unpenalized, dtype=bool).ravel()
            if _fm.shape != (A.shape[1],):
                raise ValueError("unpenalized mask must have one entry per feature")
            if _fm.any():
                free_np = _fm
        free_t = free_idx_np = pen_mask_t = x_top = g_res = None
        if free_np is not None:
            free_idx_np = np.flatnonzero(free_np)
            free_t = torch.as_tensor(free_np, dtype=torch.bool, device=op.device)
            pen_mask_t = (~free_t).to(op._value_dtype)
            if jac is not None:
                pen_mask_t = pen_mask_t / jac      # unit x-space weight -> z-space
            _xf, _finfo = solve_resident_subset(op, yt, free_idx_np,
                                                raise_on_nonconvergence=False)
            x_top = torch.zeros(A.shape[1], dtype=op._value_dtype, device=op.device)
            x_top[torch.as_tensor(free_idx_np, dtype=torch.long, device=op.device)] = \
                _xf.to(op._value_dtype)
            g_res = op.rmatvec(yt - op.matvec(x_top))
            g_res = torch.where(free_t, torch.zeros_like(g_res), g_res)
            self.free_ols_info_ = dict(_finfo) if isinstance(_finfo, dict) else _finfo
            print("[HARM_DENSE] resident: %d unpenalized columns; free-block OLS "
                  "converged=%s elapsed=%.2fs"
                  % (int(free_idx_np.size),
                     (_finfo or {}).get("converged") if isinstance(_finfo, dict) else _finfo,
                     time.monotonic() - started), flush=True)
        penalty_weights = None
        pilot_info = None
        if self.adaptive:
            # The pilot only sets the adaptive weights (|c_j| + eps)^-gamma, so
            # it needs the coefficient MAGNITUDES, not a tight solution -- and it
            # has its own accuracy budget rather than the FISTA tolerance.  Asking
            # it for 1e-8 on a float32 operator is a request the arithmetic cannot
            # meet: measured on c7 (25515x3678, float32 factors, one RTX 3090) the
            # pilot ran 666 iterations into an exploded iterate and returned
            # ||c|| = 4.0e19, so every weight came out ~1e-19 (|c|+eps)^-1 and the
            # "adaptive" LASSO had a numerically ZERO penalty -- it degenerated
            # into truncated OLS with 3678 nonzeros.  At 1e-5 the pilot converges
            # in 126 iterations (0.3s vs 32.7s) and its weights differ from the
            # 20000-iteration solution by 0.6% peak; at 1e-4 the peak weight
            # difference is already 62%, because the smallest coefficients -- the
            # ones that set the top of the weight range -- are where the relative
            # error concentrates.  1e-5 is the coarsest setting inside a 1% weight
            # budget on that operator.
            pilot_tol = float(os.environ.get("PHEASY_ALASSO_PILOT_TOL", "1e-5"))
            pilot, pilot_info = iterative_lstsq(
                op, yt, atol=pilot_tol, btol=pilot_tol, maxiter=self.max_iter)
            # Keep the pilot certificate: if it did not converge the weights are
            # still usable, but nothing else in the fit would say so.
            self.pilot_info_ = dict(pilot_info) if isinstance(pilot_info, dict) else pilot_info
            pilot = torch.as_tensor(pilot, dtype=op._value_dtype, device=op.device)
            if jac is not None:
                pilot = pilot / jac      # [HARM_DENSE] pilot back in model (x) units
            penalty_weights_t = torch.pow(pilot.abs() + self.eps, -self.gamma)
            if not bool(torch.isfinite(penalty_weights_t).all().item()):
                raise RuntimeError("Resident GPU ALASSO pilot produced nonfinite penalty weights")
            _pen_view = penalty_weights_t
            if free_t is not None:
                _pen_view = penalty_weights_t[~free_t]
                penalty_weights_t = penalty_weights_t * (~free_t).to(penalty_weights_t.dtype)   # [HARM_DENSE]
            # Keep solver weights on CUDA; retain only a diagnostic snapshot
            # (model-space weights; the solver gets them in the iteration space).
            self.penalty_weights_ = _to_numpy(penalty_weights_t, np.float64)
            if jac is not None:
                penalty_weights_t = penalty_weights_t / jac
            penalty_weights = penalty_weights_t
            print("[gpu_resident] adaptive pilot=GPU CGLS gamma=%.6g weight_range=[%.6e, %.6e]%s" % (self.gamma, float(_pen_view.min().item()), float(_pen_view.max().item()), " (penalized block; free block 0)" if free_t is not None else ""), flush=True)
            if self.alpha_auto:
                _g_w = torch.abs(op.rmatvec(yt)) if g_res is None else torch.abs(g_res)
                _w_den = (penalty_weights_t if free_t is None else
                          torch.where(free_t, torch.ones_like(penalty_weights_t), penalty_weights_t))
                weighted_kkt = torch.max(_g_w / torch.clamp(_w_den, min=torch.finfo(yt.dtype).tiny)) / A.shape[0]
                # [FIX P46] the ALASSO rule (P37) anchors the bottom at the MIN of
                # the weighted and unweighted thresholds; the unweighted one comes
                # from the same rmatvec, so it is free here.
                # (with the internal Jacobi the gradient is in z units; the
                # unweighted threshold of the unstandardized model is |A^T r| = s*|g_z|)
                unweighted_kkt = float(torch.max(_g_w if jac is None else _g_w * jac).item()) / A.shape[0]
                amax = float(weighted_kkt.item())
                if amax > 0 and np.isfinite(amax):
                    self.alphas = _lasso_grid(amax, min(amax, unweighted_kkt),
                                              self.decades,
                                              max(self.nalpha, len(self.alphas)),
                                              A.shape[0], A.shape[1])
        elif self.alpha_auto:
            # Non-adaptive LASSO: derive the KKT-threshold grid on the resident
            # operator instead of relying on the CLI's CPU derive_alpha_grid. op
            # is already normalized (when standardize=True), so op.rmatvec(yt)
            # yields A.T y / ||col|| -- the standardized-space gradient the grid
            # needs -- matching derive_alpha_grid(standardize=True) exactly.
            g = op.rmatvec(yt) if g_res is None else g_res
            if jac is not None:
                g = g * jac      # [HARM_DENSE] threshold of the unstandardized model
            amax = float(torch.max(torch.abs(g)).item()) / A.shape[0]
            if amax > 0 and np.isfinite(amax):
                # [FIX P46] same span floor as derive_alpha_grid: this resident
                # grid is the one the shipped GPU runner actually uses, and a bare
                # 4 decades pinned Mg8C120 alpha* 1.2x above its own bottom.
                self.alphas = _lasso_grid(amax, amax, self.decades,
                                          max(self.nalpha, len(self.alphas)),
                                          A.shape[0], A.shape[1])
        if free_t is not None and penalty_weights is None:
            # [HARM_DENSE] plain LASSO: unit weight on the penalized block only
            penalty_weights = pen_mask_t
        cv_tol = float(os.environ.get("PHEASY_CV_TOL", str(max(self.tol, 1e-3))))
        # CV only needs the MSE *ranking* across alphas, not a tight solution per
        # alpha. The mid-grid alphas (the sparse->dense transition) converge
        # extremely slowly under FISTA and hit the cap with KKT ~1e-2 either way;
        # 400 vs 800 iterations leaves the selected alpha unchanged on c2.6
        # (5 seeds + full 99792 rows) while ~halving CV wall time.
        cv_cap = int(os.environ.get("PHEASY_CV_MAX_ITER", str(min(self.max_iter, 400))))
        if cv_cap < 1 or cv_tol <= 0:
            raise ValueError("CV max_iter and tol must be positive")
        # Factors were replicated once per card at upload time, never once per
        # fold. Copy primary normalization and power estimate to keep numerical
        # setup identical. Every selected card fit the FULL factors (preflighted
        # on its own upload).
        resources = [(op, yt, L)]
        for replica, dev in zip(replicas, devices[1:]):
            replica.scale = op.scale.to(dev).clone()
            replica._norma = None
            resources.append((replica, yt.to(dev), L.to(dev)))
        self.cv_devices_ = [str(dev) for dev in devices]
        print("[gpu_resident] dynamic CV devices=%s folds=%d primary=%s" %
              (self.cv_devices_, len(splits), op.device), flush=True)
        # Complete caller-stream setup before handing tensors to dispatchers.
        for dev in devices:
            if torch.device(dev).type == "cuda":
                torch.cuda.synchronize(dev)

        def solve_fold(resource, k, fold):
            worker, target, estimate = resource
            with _resident_device_context(worker.device):
                tr, va = fold
                trt = torch.as_tensor(tr, dtype=torch.int64, device=worker.device)
                vat = torch.as_tensor(va, dtype=torch.int64, device=worker.device)
                values = torch.empty(len(self.alphas), dtype=target.dtype, device=worker.device)
                x = None
                if free_idx_np is not None:
                    # [HARM_DENSE] top-of-grid solution on THIS fold's rows
                    _xk, _ = solve_resident_subset(worker, target, free_idx_np, rows=tr,
                                                   raise_on_nonconvergence=False)
                    x = torch.zeros(A.shape[1], dtype=target.dtype, device=worker.device)
                    x[torch.as_tensor(free_idx_np, dtype=torch.long,
                                      device=worker.device)] = _xk.to(target.dtype)
                infos = []
                for i in range(len(self.alphas) - 1, -1, -1):
                    # CV folds are a capped, ranking-only approximation by design:
                    # hitting the cap is information (see _cv_hit_cap), not waste.
                    x, info = _fista_twolevel(worker, target, float(self.alphas[i]), x,
                                             cv_cap, cv_tol, estimate, trt,
                                             penalty_weights=penalty_weights, n_samples=int(trt.numel()),
                                             auto_floor=False)
                    err = worker.matvec(x)[vat] - target[vat]
                    values[i] = err.square().mean()
                    infos.append(dict(info, alpha=float(self.alphas[i])))
                    print("[gpu_resident] device=%s fold=%d/%d alpha=%.6e n_iter=%d kkt=%.3e converged=%s elapsed=%.2fs" %
                          (worker.device, k + 1, len(splits), self.alphas[i], info["n_iter"],
                           info["kkt_relative"], info["converged"], time.monotonic() - started), flush=True)
                # Complete the fold on its own card, not on another busy GPU.
                if torch.device(worker.device).type == "cuda":
                    torch.cuda.synchronize(worker.device)
                return values, infos, str(worker.device)

        results = _dynamic_fold_map(resources, splits, solve_fold)
        # Only tiny MSE paths cross devices, after all dispatchers have joined.
        # Iterative vectors and factors never round-trip through host memory.
        mse = torch.stack([result[0].to(op.device) for result in results], dim=1)
        self.cv_solver_info_ = [result[1] for result in results]
        self.cv_fold_devices_ = [result[2] for result in results]
        _cv_infos = [info for infos in self.cv_solver_info_ for info in infos]
        max_cv_iterations = max((info["n_iter"] for info in _cv_infos), default=0)
        # [D1] "hit the cap" means "hit the cap without converging": a fold that
        # converges exactly at the cap used to be reported as a convergence
        # problem.  Mirror the dense path's definition.
        _cv_hit_cap = any(info["n_iter"] >= cv_cap and not info.get("converged", False)
                          for info in _cv_infos)
        for replica in owned[1:]:
            replica.close()
        resources.clear()
        means = mse.mean(dim=1)
        if not bool(torch.isfinite(means).all().item()):
            raise RuntimeError("Resident CV produced nonfinite MSE")
        # Match iterative selection: ascending argmin chooses the smallest
        # alpha on exact ties. Near-tie tolerance is diagnostic only.
        best_i = int(torch.argmin(means).item())
        # PHEASY_LASSO_1SE: one-standard-error rule, mirroring GpuLassoCV and
        # _reselect_alpha. It used to be ignored here, so the knob silently did
        # nothing on the path the shipped GPU runner actually uses.
        if os.environ.get("PHEASY_LASSO_1SE", "0").lower() in ("1", "true", "yes"):
            means_cpu = _to_numpy(means, np.float64)
            mse_cpu = _to_numpy(mse, np.float64)
            se = float(mse_cpu[best_i].std(ddof=1) / np.sqrt(mse_cpu.shape[1])) \
                if mse_cpu.shape[1] > 1 else 0.0
            cand = np.flatnonzero(means_cpu <= means_cpu[best_i] + se)
            best_i = int(cand[np.argmax(self.alphas[cand])])
            print("[gpu_resident] PHEASY_LASSO_1SE: alpha* moved to %.6e "
                  "(1 SE = %.3e above the CV minimum)" %
                  (float(self.alphas[best_i]), se), flush=True)
        rtol = float(os.environ.get("PHEASY_LASSO_TIE_RTOL", "1e-9"))
        tied = means <= means[best_i] * (1 + rtol) + 1e-300
        self.alpha_ = float(self.alphas[best_i])
        # Preserve independent full-data descending warm-start path.
        x = None if x_top is None else x_top.clone()   # [HARM_DENSE] exact at the top
        for i in range(len(self.alphas) - 1, best_i - 1, -1):
            # The alpha-path walk is warm-started and capped on purpose (each
            # alpha only needs to be roughly right to rank the grid).
            x, path_info = _fista_twolevel(op, yt, float(self.alphas[i]), x, cv_cap, cv_tol, L,
                                         penalty_weights=penalty_weights, n_samples=A.shape[0],
                                         auto_floor=False)
            print("[gpu_resident] full-path alpha=%.6e n_iter=%d kkt=%.3e converged=%s elapsed=%.2fs" %
                  (self.alphas[i], path_info["n_iter"], path_info["kkt_relative"],
                   path_info["converged"], time.monotonic() - started), flush=True)
        x, info = _fista_twolevel(op, yt, self.alpha_, x, self.max_iter, self.tol, L,
                                  penalty_weights=penalty_weights, n_samples=A.shape[0])
        print("[gpu_resident] final alpha=%.6e n_iter=%d kkt=%.3e converged=%s elapsed=%.2fs" %
              (self.alpha_, info["n_iter"], info["kkt_relative"], info["converged"],
               time.monotonic() - started), flush=True)
        self.coef_ = _to_numpy(x / op.scale, np.float64)
        self.column_scale_ = _to_numpy(op.scale, np.float64)
        self.mse_path_ = _to_numpy(mse, np.float64)
        self.alphas_ = self.alphas
        self.intercept_ = 0.0
        self.n_iter_ = info["n_iter"]
        self.n_features_in_ = A.shape[1]
        self.regularized_solver_info_ = dict(info, solver="FISTA", backend="gpu_twolevel_resident",
            device=str(op.device), dtype="float64", stage="regularized_refit_before_debias", tol=float(self.tol))
        self._alpha_at_min = best_i == 0
        self._alpha_at_min_flat = self._alpha_at_min and int(tied.sum().item()) > 1
        self._alpha_at_min_hitcap = self._alpha_at_min_flat and _cv_hit_cap
        if retain_operator:
            # Keep the primary operator resident so the post-fit OLS debias can reuse
            # its factors (solve_resident_subset) instead of re-uploading the support.
            self._operator = op
            owned.clear()
        if self._alpha_at_min:
            # Same four-way diagnosis as the dense path (it used to collapse
            # into one generic sentence, losing the "lower PHEASY_CV_TOL / raise
            # PHEASY_CV_MAX_ITER" guidance on the path the shipped runner uses).
            import warnings
            if self._alpha_at_min_hitcap:
                warnings.warn(
                    "Resident LASSO alpha* %.3e sits at the grid MINIMUM via a tie "
                    "on a FLAT CV tail AND FISTA hit cv_max_iter (%d): this is a "
                    "CONVERGENCE problem (lower PHEASY_CV_TOL / raise "
                    "PHEASY_CV_MAX_ITER), not a model-density conclusion."
                    % (self.alpha_, cv_cap), RuntimeWarning, stacklevel=2)
            elif self._alpha_at_min_flat:
                warnings.warn(
                    "Resident LASSO alpha* %.3e sits at the grid MINIMUM on a flat "
                    "CV tail, but FISTA already converged (max %d iters < %d): "
                    "alpha* is not well-determined by CV (not a convergence problem)."
                    % (self.alpha_, max_cv_iterations, cv_cap), RuntimeWarning, stacklevel=2)
            else:
                warnings.warn(
                    "Resident LASSO alpha* %.3e sits at the grid MINIMUM; the CV "
                    "curve is still falling at the low end, so widening the grid "
                    "only pushes alpha* toward OLS. Treat this fit as effectively "
                    "unregularized (compare with OLS/RFE)."
                    % self.alpha_, RuntimeWarning, stacklevel=2)
        # Do not retain VRAM after the fit. Predict and optional debias use the
        # ordinary public host interface, explicitly outside the resident stage.
        return self
