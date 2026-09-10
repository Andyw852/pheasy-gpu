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

import numpy as np

__all__ = [
    "available",
    "available_memory_bytes",
    "enabled",
    "set_gpu_mode",
    "get_gpu_mode",
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


def _multi_gpu_devices(min_free_bytes=0):
    """Device indices for fold-parallel CV, filtered by free VRAM.

    PHEASY_GPU_DEVICES="1,2,4" pins an explicit list; otherwise all visible
    devices are considered. Devices with usable VRAM < min_free_bytes are
    dropped (so a busy shared GPU is skipped). Falls back to the single
    device() when nothing qualifies.
    """
    import torch
    if not available():
        return [0]
    n = torch.cuda.device_count()
    if n <= 1:
        return [0]
    raw = os.environ.get("PHEASY_GPU_DEVICES", "").strip()
    if raw:
        devs = [int(x) for x in raw.split(",") if x.strip() != ""]
    else:
        devs = list(range(n))
    devs = [d for d in devs if 0 <= d < n]
    if min_free_bytes > 0:
        devs = [d for d in devs
                if (_device_free_bytes(d) or 0) >= min_free_bytes]
    if not devs:
        return [int(device().index)]
    return devs


def _env_wants():
    v = os.environ.get("PHEASY_USE_GPU", None)
    if v is None:
        return None
    return v.lower() in ("1", "true", "yes", "on")


def enabled():
    """Whether GPU dispatch should be used right now."""
    if _mode is not None:
        want = _mode
    else:
        want = _env_wants()
        if want is None:
            want = True        # auto: use GPU when available
    return bool(want) and available()


def device():
    """CUDA device, read fresh from PHEASY_GPU_DEVICE each call (no caching)."""
    import torch
    dev = os.environ.get("PHEASY_GPU_DEVICE", None)
    if dev is not None:
        return torch.device("cuda:%d" % int(dev))
    return torch.device("cuda:0")


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


def _is_dense(A):
    return isinstance(A, np.ndarray)


# ---------------------------------------------------------------------------
# CV splits (identical to optimizer._make_cv_splits so GPU and CPU agree)
# ---------------------------------------------------------------------------
def _make_cv_splits(n_samples, cv, random_state=None, group_size=None):
    from sklearn.model_selection import GroupKFold, KFold
    if cv is None or cv <= 1:
        cv = min(3, n_samples)
    cv = int(cv)
    if group_size and group_size > 1 and n_samples % group_size == 0:
        groups = np.arange(n_samples) // group_size
        n_groups = int(groups[-1]) + 1
        if n_groups >= 2:
            eff_cv = int(min(cv, n_groups))
            gkf = GroupKFold(n_splits=eff_cv)
            return list(gkf.split(np.zeros(n_samples, dtype=np.int8),
                                  np.zeros(n_samples, dtype=np.int8), groups))
    cv = max(2, min(cv, n_samples))
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
    if estimated_peak > available_memory_bytes() * fraction:
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

    Note: eigvalsh is a full O(p^3) decomposition, called once per CV fold
    (n_splits + 1 times per fit); p=3678 is acceptable. If this ever becomes
    the bottleneck, fall back to a power iteration whose final line is the
    Rayleigh quotient v . (G v) -- not ||G v||^2 -- with a small safety factor.
    """
    import torch
    if Gt.shape[0] == 0:
        return 0.0
    return float(torch.linalg.eigvalsh(Gt)[-1].item())


def _soft_threshold_t(x, thr):
    import torch
    # soft-threshold(x, thr) == x - clamp(x, -thr, thr): 2 elementwise ops
    # instead of sign(x)*max(|x|-thr, 0) (5 ops). thr may be a scalar or a
    # per-coordinate vector (penalty weights).
    return x - torch.clamp(x, min=-thr, max=thr)


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
    step = 1.0 / max(float(lipschitz), 1e-12)
    thr = float(alpha) * float(n_samples) * step
    if penalty_weights is not None:
        thr_vec = thr * torch.as_tensor(np.ascontiguousarray(penalty_weights),
                                        dtype=Gt.dtype, device=Gt.device)
    else:
        thr_vec = thr

    penalty = thr_vec / step
    kkt_scale = torch.clamp(bt.abs().max(), min=torch.finfo(bt.dtype).tiny)

    def kkt_relative(coef):
        gradient = Gt @ coef - bt
        violation = torch.where(coef != 0, (gradient + penalty * coef.sign()).abs(),
                                torch.clamp(gradient.abs() - penalty, min=0.0))
        return float((violation.max() / kkt_scale).item())

    converged = False
    kkt = float("inf")
    z = x.clone()
    t = 1.0
    x_prev = x.clone()
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
        _info.update(n_iter=n_iter, converged=converged, kkt_relative=kkt)
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

        G_full = At.T @ At
        b_full = At.T @ yt
        lip_full = _power_lipschitz(G_full)

        gram_folds = []
        lip_folds = []
        A_va_list = []
        for tr, va in splits:
            va_t = torch.as_tensor(np.asarray(va), dtype=torch.long, device=At.device)
            A_va = At[va_t]
            G_va = A_va.T @ A_va
            b_va = A_va.T @ yt[va_t]
            gram_folds.append((G_full - G_va, b_full - b_va))
            lip_folds.append(_power_lipschitz(G_full - G_va))
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

        pw = self.penalty_weights

        for a_i in range(n_alphas - 1, -1, -1):
            alpha = float(self.alphas[a_i])
            fold_mse = np.zeros(len(splits))
            for k, (tr, va) in enumerate(splits):
                coef, nit = _fista_gram(gram_folds[k][0], gram_folds[k][1], alpha,
                                        x0=x_folds[k], max_iter=cv_max_iter, tol=cv_tol,
                                        lipschitz=lip_folds[k], penalty_weights=pw,
                                        n_samples=len(tr))
                va_t = torch.as_tensor(np.asarray(va), dtype=torch.long, device=At.device)
                pred = A_va_list[k] @ coef
                x_folds[k] = coef
                _cv_max_n_iter = max(_cv_max_n_iter, nit)
                err = pred - yt[va_t]
                fold_mse[k] = float((err * err).mean().item())
            mse_path[a_i] = fold_mse
            mean = float(fold_mse.mean())

            x_full, nit = _fista_gram(G_full, b_full, alpha, x0=x_full,
                                      max_iter=cv_max_iter, tol=cv_tol,
                                      lipschitz=lip_full, penalty_weights=pw,
                                      n_samples=n_samples)
            _cv_max_n_iter = max(_cv_max_n_iter, nit)
            if mean <= best_mean:
                best_mean = mean
                best_i = a_i
                best_x = x_full.clone()

        # tie / edge diagnostics (mirror _LassoCVIterative so holdout flags work)
        mean_path = mse_path.mean(axis=1)
        rtol = float(os.environ.get("PHEASY_LASSO_TIE_RTOL", "1e-9"))
        tied = np.flatnonzero(mean_path <= best_mean * (1.0 + rtol) + 1e-300)
        _cv_hit_cap = _cv_max_n_iter >= cv_max_iter
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
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("[GPU] RIDGE CV fold %d OOM on cuda:%d; falling back to "
                      "CPU (shared box?)" % (k, dev), flush=True)
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
        # CPU path: keep NS sparse (like holdout_eval's own fallback) instead of
        # densifying it -- avoids a large dense block-diagonal allocation.
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
        smc = sm.tocoo()
        idx = torch.as_tensor(np.vstack([smc.row, smc.col]), dtype=torch.long,
                              device=device())
        vals = torch.as_tensor(smc.data, dtype=torch.float64, device=device())
        spt = torch.sparse_coo_tensor(idx, vals, smc.shape,
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
    resident = (_cuda_csr_bytes(row_block, value_itemsize)
                + _cuda_csr_bytes(transpose_block, value_itemsize))
    vectors = sum(sum(block.shape) for block in (row_block, transpose_block)) \
        * int(value_itemsize)
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
        self._rs = np.linspace(0, N, G + 1).astype(np.int64)
        self._cs = np.linspace(0, M, G + 1).astype(np.int64)
        self._R = []
        self._T = []
        # [low-mem] per-block column slices instead of one full transpose:
        # building smT = SM_prime.T.tocsr() transiently doubles the ~60 GB SM
        # in host RAM and OOM-killed the 699-config fit at ~185 GB on the
        # shared box (exit 137, three times). Column slices are ~97 s per
        # 13275-col block (8 min total vs the transpose) but peak ~75 GB.
        try:
            for i, d in enumerate(devs):
                dev = t.device("cuda:%d" % d)
                Ri = sm_prime[self._rs[i]:self._rs[i + 1]].tocsr()
                _check_cuda_csr_indices(Ri)
                Ti = sm_prime[:, self._cs[i]:self._cs[i + 1]].T.tocsr()
                _check_cuda_csr_indices(Ti)
                required = _cuda_spmv_block_budget(Ri, Ti, self._value_itemsize)
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
                with t.cuda.device(d):
                    self._R.append(self._csr_to_torch(Ri, dev))
                    self._T.append(self._csr_to_torch(Ti, dev))
                del Ri, Ti
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
    """CUDA-resident float64 SM_prime @ NS, without a product or Gram matrix.

    Both factors and their CSR transposes live on one device. Allocation or
    unsupported sparse-kernel errors propagate: this backend NEVER falls back.
    Dense NS is supported; sparse NS is never densified. CV uses row masks so
    folds do not duplicate the factors. Only setup uploads and public result
    downloads cross the host boundary.
    """
    def __init__(self, A, device_id=None, extra_workspace_bytes=0):
        import scipy.sparse as sp
        if not enabled() or not available():
            raise RuntimeError("Resident two-level LASSO requires enabled CUDA; no CPU fallback")
        input_scale = getattr(A, "_twolevel_scale", None)
        A = getattr(A, "_twolevel_base", A)
        if not hasattr(A, "SM_prime") or not hasattr(A, "NS"):
            raise TypeError("GpuTwoLevelOperator requires TwoLevelSM")
        self.torch = torch = _torch()
        self.device = device() if device_id is None else torch.device(device_id)
        self.shape = A.shape
        if not sp.issparse(A.SM_prime):
            raise TypeError("Resident two-level SM_prime must be scipy sparse")
        fraction = float(os.environ.get("PHEASY_GPU_MEM_FRACTION", "0.8"))
        if not 0 < fraction <= 1:
            raise ValueError("PHEASY_GPU_MEM_FRACTION must be in (0, 1]")
        factor_bytes = 0
        for matrix in (A.SM_prime, A.NS):
            if sp.issparse(matrix):
                factor_bytes += 32 * int(matrix.nnz) + 8 * (sum(matrix.shape) + 2)
            else:
                factor_bytes += 8 * int(np.prod(matrix.shape))
        workspace = max(1, int(os.environ.get("PHEASY_GPU_NORM_WORKSPACE_MB", "64"))) * 1024**2
        # Extra factor-sized allowance covers CSR transpose conversion scratch;
        # vector margin covers FISTA, masked folds, target and power iteration.
        if not np.isfinite(extra_workspace_bytes) or extra_workspace_bytes < 0 or int(extra_workspace_bytes) != extra_workspace_bytes:
            raise ValueError("extra_workspace_bytes must be a nonnegative integer")
        self.estimated_peak_bytes = 2 * factor_bytes + workspace + 8 * 32 * sum(A.shape) + int(extra_workspace_bytes)
        free = _device_free_bytes(self.device)
        if free is None and torch.device(self.device).type == "cuda":
            raise RuntimeError("Cannot query resident CUDA memory budget; refusing unchecked upload")
        if free is not None and self.estimated_peak_bytes > free * fraction:
            raise MemoryError("Resident two-level GPU estimate %d bytes exceeds budget %d bytes; no CPU fallback" %
                              (self.estimated_peak_bytes, int(free * fraction)))

        def upload(matrix):
            if not sp.issparse(matrix):
                return torch.as_tensor(np.asarray(matrix), dtype=torch.float64, device=self.device)
            csr = matrix.tocsr(copy=True)
            csr.sum_duplicates()
            csr.sort_indices()
            # int64 avoids truncation on genuinely large factors.
            return torch.sparse_csr_tensor(
                torch.as_tensor(csr.indptr, dtype=torch.int64, device=self.device),
                torch.as_tensor(csr.indices, dtype=torch.int64, device=self.device),
                torch.as_tensor(csr.data, dtype=torch.float64, device=self.device),
                size=csr.shape, device=self.device, check_invariants=True)

        try:
            self.prime = upload(A.SM_prime)
            self.ns = upload(A.NS)
            # Convert on device rather than allocating giant host CSR transposes.
            self.prime_t = self.prime.transpose(0, 1).to_sparse_csr()
            self.ns_t = (self.ns.T if self.ns.layout == torch.strided else
                         self.ns.transpose(0, 1).to_sparse_csr())
            self.scale = torch.ones(A.shape[1], dtype=torch.float64, device=self.device)
            self.input_scale = (torch.ones_like(self.scale) if input_scale is None else
                                torch.as_tensor(input_scale, dtype=torch.float64, device=self.device))
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


    def _mm(self, matrix, vector):
        if matrix.layout == self.torch.strided:
            return matrix @ vector
        if vector.ndim == 1:
            return self.torch.sparse.mm(matrix, vector[:, None]).flatten()
        return self.torch.sparse.mm(matrix, vector)

    def matvec(self, vector):
        return self._mm(self.prime, self._mm(self.ns, vector / (self.scale * self.input_scale)))

    def rmatvec(self, vector):
        return self._mm(self.ns_t, self._mm(self.prime_t, vector)) / (self.scale * self.input_scale)

    def norm_estimate(self, iters=10):
        """Cached spectral estimate for the current normalized operator."""
        if getattr(self, "_norma", None) is None:
            self._norma = _operator_norm_estimate(self, iters)
        return self._norma

    def col_norms(self):
        """Exact full-row column norms of the current effective operator, on CUDA.

        Bounded column blocks avoid sparse-sparse products and never allocate
        the full sensing matrix. Workspace is O(block*(n + mid + p)).
        """
        torch = self.torch
        n, p = self.shape
        budget = int(os.environ.get("PHEASY_GPU_NORM_WORKSPACE_MB", "64")) * 1024**2
        if budget <= 0:
            raise ValueError("PHEASY_GPU_NORM_WORKSPACE_MB must be positive")
        block = max(1, min(64, budget // max(8 * (n + self.ns.shape[0] + p) * 2, 1)))
        norms = torch.empty_like(self.scale)
        for start in range(0, p, block):
            count = min(block, p - start)
            basis = torch.zeros((p, count), dtype=torch.float64, device=self.device)
            idx = torch.arange(count, device=self.device)
            basis[start + idx, idx] = 1
            cols = self._mm(self.prime, self._mm(self.ns, basis / (self.input_scale * self.scale)[:, None]))
            norms[start:start + count] = torch.linalg.vector_norm(cols, dim=0)
        return norms

    def normalize(self):
        """Apply exact unit-L2 normalization and invalidate the spectral estimate."""
        torch = self.torch
        norms = self.col_norms() * self.scale
        self.scale = torch.where(norms < 1e-30, torch.ones_like(norms), norms)
        self._norma = None
        return self.scale

    def lipschitz(self):
        """Power estimate; FISTA backtracking certifies every accepted step."""
        torch = self.torch
        gen = torch.Generator(device=self.device).manual_seed(0)
        v = torch.randn(self.shape[1], generator=gen, dtype=torch.float64, device=self.device)
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
        self.column_scale = torch.ones(self.shape[1], dtype=torch.float64, device=self.device)
        if column_scale is not None:
            scale = torch.as_tensor(column_scale, dtype=torch.float64, device=self.device)
            if scale.shape != self.column_scale.shape or not bool((torch.isfinite(scale) & (scale >= 0)).all().item()):
                raise ValueError("subset column scales must be finite, nonnegative, and match active columns")
            self.column_scale = torch.where(scale < 1e-30, torch.ones_like(scale), scale)

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
                          ridge_alpha=0.0, atol=1e-8, btol=1e-8, maxiter=5000):
    """Return physical CUDA subset coefficients; reject unconverged elimination fits.

    target is the full row-space vector; column_scale is in active-column order.
    """
    if not np.isfinite(ridge_alpha) or ridge_alpha < 0:
        raise ValueError("Ridge alpha must be finite and nonnegative")
    view = GpuSubsetOperator(base, columns, rows, column_scale)
    y = base.torch.as_tensor(target, dtype=base.torch.float64, device=base.device).reshape(-1)
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
    if not info["converged"]:
        raise RuntimeError("Resident subset solve did not converge: " + repr(info))
    return coef / view.column_scale, info


def iterative_ridge(A, y, alpha, atol=1e-8, btol=1e-8, maxiter=5000, rows=None):
    """Solve ridge on a CUDA-resident operator, returning NumPy coefficients."""
    x, info = _iterative_ridge_tensor(A, y, alpha, atol, btol, maxiter, rows)
    return x.detach().cpu().numpy(), info


def _iterative_ridge_tensor(A, y, alpha, atol=1e-8, btol=1e-8, maxiter=5000, rows=None, penalty_scale=None):
    """Augmented CGLS with CUDA coefficient output."""
    torch = A.torch
    dev = A.device
    y = torch.as_tensor(y, dtype=torch.float64, device=dev).reshape(-1)
    mask = torch.ones(A.shape[0], dtype=torch.float64, device=dev)
    if rows is not None:
        mask.zero_(); mask[torch.as_tensor(rows, dtype=torch.long, device=dev)] = 1
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Ridge alpha must be finite and nonnegative")
    sa = float(np.sqrt(alpha))
    if penalty_scale is not None:
        scale = torch.as_tensor(penalty_scale, dtype=torch.float64, device=dev)
        if scale.shape != (A.shape[1],) or not bool((torch.isfinite(scale) & (scale > 0)).all().item()):
            raise ValueError("penalty_scale must be finite positive and match coefficient count")
        sa = sa / scale
    class Augmented:
        def __init__(self):
            self.shape = (A.shape[0] + A.shape[1], A.shape[1])
            self.torch = torch
            self.device = dev
        def matvec(self, x):
            return torch.cat((A.matvec(x) * mask, sa * x))
        def rmatvec(self, z):
            return A.rmatvec(z[:A.shape[0]] * mask) + sa * z[A.shape[0]:]
    return _iterative_lstsq_tensor(Augmented(), torch.cat((y * mask, y.new_zeros(A.shape[1]))), atol, btol, maxiter)


def _operator_norm_estimate(A, iters=10):
    """Deterministic matrix-free spectral estimate, not SciPy's Frobenius estimate."""
    t = A.torch
    gen = t.Generator(device=A.device).manual_seed(0)
    v = t.randn(A.shape[1], generator=gen, dtype=t.float64, device=A.device)
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


def _iterative_lstsq_tensor(A, y, atol=1e-8, btol=1e-8, maxiter=5000):
    """CGLS core returning CUDA coefficients and the same convergence diagnostics."""
    torch = getattr(A, "torch", None)
    device = getattr(A, "device", None)
    if torch is None or device is None or torch.device(device).type != "cuda":
        raise RuntimeError("GPU iterative least-squares requires a CUDA operator")
    y = torch.as_tensor(y, dtype=torch.float64, device=device).reshape(-1)
    if y.numel() != A.shape[0]:
        raise ValueError("least-squares target length does not match operator")
    if not bool(torch.isfinite(y).all().item()):
        raise ValueError("least-squares target must be finite")
    if (not np.isfinite(atol) or not np.isfinite(btol) or atol < 0 or btol < 0
            or not np.isfinite(maxiter) or maxiter <= 0 or int(maxiter) != maxiter):
        raise ValueError("atol/btol must be finite nonnegative and maxiter a finite positive integer")
    x = torch.zeros(A.shape[1], dtype=y.dtype, device=device)
    r = y.clone()
    s = A.rmatvec(r)
    p = s.clone()
    gamma = torch.dot(s, s)
    rhs_norm = torch.linalg.vector_norm(y).clamp_min(torch.finfo(y.dtype).tiny)
    normal_norm = torch.linalg.vector_norm(s)
    residual_norm = torch.linalg.vector_norm(r)
    norma = A.norm_estimate() if hasattr(A, "norm_estimate") else _operator_norm_estimate(A)
    def meets_tolerance():
        normx = torch.linalg.vector_norm(x)
        finite = torch.isfinite(normal_norm) & torch.isfinite(residual_norm) & torch.isfinite(rhs_norm) & torch.isfinite(normx)
        return bool((finite & bool(np.isfinite(norma)) & ((normal_norm <= atol * norma * residual_norm)
                    | (residual_norm <= btol * rhs_norm + atol * norma * normx))).item())
    converged = meets_tolerance()
    n_iter = 0
    residual_norm = torch.linalg.vector_norm(r)
    stop_reason = "iteration_limit"
    for it in range(int(maxiter)):
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
        converged = meets_tolerance()
        if converged:
            break
        if bool((~torch.isfinite(gamma_new) | (gamma <= 0)).item()):
            stop_reason = "invalid_gradient_recurrence"
            break
        p = s_new + (gamma_new / gamma) * p
        s = s_new
        gamma = gamma_new
    # Certify the delivered coefficients, not only the recursively updated residual.
    recurrence_converged = converged
    true_r = y - A.matvec(x)
    residual_norm = torch.linalg.vector_norm(true_r)
    normal_norm = torch.linalg.vector_norm(A.rmatvec(true_r))
    converged = meets_tolerance()
    if recurrence_converged and not converged:
        stop_reason = "true_residual_check_failed"
    info = {"solver": "GPU CGLS", "itn": n_iter, "n_iter": n_iter,
            "residual_certificate": "recomputed_y_minus_Ax",
            "stop_reason": "converged" if converged else stop_reason,
            "norma": float(norma), "norma_estimator": "spectral_power_10",
            "normb": float(rhs_norm.item()), "normx": float(torch.linalg.vector_norm(x).item()),
            "criterion": "normar<=atol*norma*normr or normr<=btol*normb+atol*norma*normx",
            "normr": float(residual_norm.item()), "normar": float(normal_norm.item()),
            "converged": bool(converged), "device": str(device),
            "backend": "gpu_resident_iterative", "atol": float(atol),
            "btol": float(btol), "maxiter": int(maxiter)}
    if not converged:
        import warnings
        warnings.warn("GPU CGLS did not converge: reason=%s, iterations=%d, normr=%g, normar=%g" %
                      (info["stop_reason"], n_iter, info["normr"], info["normar"]), RuntimeWarning, stacklevel=2)
    return x, info


def _fista_twolevel(A, y, alpha, x0, max_iter, tol, lipschitz, rows=None, penalty_weights=None, n_samples=None):
    """Device FISTA, adaptive restart and exact L1 KKT certificate.

    No NumPy or vector host transfers in the iteration loop. Scalar syncs are
    limited to backtracking acceptance and a KKT check every 20 iterations.
    Masked residuals give the exact training objective without fold CSR copies.
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
    L = lipschitz.clone()

    def kkt(coef):
        gradient = A.rmatvec((A.matvec(coef) - y) * mask)
        violation = torch.where(coef != 0, (gradient + penalty_vec * coef.sign()).abs(),
                                torch.clamp(gradient.abs() - penalty_vec, min=0))
        return violation.max() / scale

    converged = False
    n_iter = 0
    for it in range(int(max_iter)):
        n_iter = it + 1
        residual = (A.matvec(z) - y) * mask
        grad = A.rmatvec(residual)
        # Rayleigh power estimates are not safe upper bounds; certify the local
        # quadratic majorizer rather than accepting a potentially unstable step.
        for attempt in range(60):
            candidate = z - grad / L
            x_new = candidate.sign() * torch.clamp(candidate.abs() - penalty_vec / L, min=0)
            delta = x_new - z
            Adelta = A.matvec(delta) * mask
            if bool((Adelta.square().sum() <= L * delta.square().sum() * (1 + 1e-12)).item()):
                break
            L = L * 2
        else:
            raise RuntimeError("Resident FISTA backtracking failed; nonfinite data or operator")
        restart = torch.dot(z - x_new, x_new - x) > 0
        next_momentum = (1 + torch.sqrt(1 + 4 * momentum.square())) / 2
        z = torch.where(restart, x_new, x_new + ((momentum - 1) / next_momentum) * (x_new - x))
        momentum = torch.where(restart, torch.ones_like(momentum), next_momentum)
        x = x_new
        if n_iter % 20 == 0:
            certificate = kkt(x)
            if bool((torch.isfinite(certificate) & (certificate <= tol)).item()):
                converged = True
                break
    certificate = kkt(x)
    value = float(certificate.item())
    converged = bool(np.isfinite(value) and value <= tol)
    info = dict(n_iter=n_iter, converged=converged, kkt_relative=value)
    if not converged:
        import warnings
        warnings.warn("Resident FISTA did not converge: iterations=%d, relative KKT=%g, tol=%g" %
                      (n_iter, value, tol), RuntimeWarning, stacklevel=2)
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
                 alpha_auto=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.standardize = standardize
        self.adaptive = bool(adaptive)
        self.gamma = float(gamma)
        self.init_alpha = float(init_alpha)
        self.eps = float(eps)
        self.nalpha = int(nalpha) if nalpha else len(self.alphas)
        self.decades = float(decades)
        self.alpha_auto = bool(alpha_auto)
        self.penalty_weights_ = None

    def fit(self, A, y, sample_weight=None):
        if not enabled() or not available():
            raise RuntimeError("Resident two-level LASSO requires enabled CUDA; no CPU fallback")
        devices = _resident_cv_devices()
        owned = []
        try:
            with _resident_device_context(devices[0]):
                return self._fit_resident(A, y, devices, owned, sample_weight)
        finally:
            for operator in owned:
                operator.close()

    def _fit_resident(self, A, y, devices, owned, sample_weight):
        import time
        from .optimizer import _make_cv_splits
        started = time.monotonic()
        if sample_weight is not None or self.fit_intercept:
            raise NotImplementedError("Resident GPU ALASSO does not support sample weights or intercept")
        if self.alphas.size == 0 or not np.isfinite(self.alphas).all() or (self.alphas < 0).any():
            raise ValueError("alphas must be nonempty, finite and nonnegative")
        if self.max_iter < 1 or self.tol <= 0:
            raise ValueError("max_iter and tol must be positive")
        print("[gpu_resident] uploading factors shape=%s; CUDA required, no CPU fallback" % (A.shape,), flush=True)
        op = GpuTwoLevelOperator(A, device_id=devices[0])
        owned.append(op)
        print("[gpu_resident] factors ready device=%s dtype=float64 estimated_peak_bytes=%d elapsed=%.2fs" %
              (op.device, op.estimated_peak_bytes, time.monotonic() - started), flush=True)
        torch = op.torch
        yt = torch.as_tensor(np.asarray(y).ravel(), dtype=torch.float64, device=op.device)
        if yt.numel() != A.shape[0] or not bool(torch.isfinite(yt).all().item()):
            raise ValueError("target shape or finite values invalid")
        if self.standardize:
            print("[gpu_resident] exact normalization started", flush=True)
            op.normalize()
            print("[gpu_resident] normalization done elapsed=%.2fs" % (time.monotonic() - started), flush=True)
        print("[gpu_resident] Lipschitz estimate started", flush=True)
        L = op.lipschitz()
        print("[gpu_resident] Lipschitz estimate ready elapsed=%.2fs" % (time.monotonic() - started), flush=True)
        penalty_weights = None
        pilot_info = None
        if self.adaptive:
            pilot, pilot_info = iterative_lstsq(op, yt, atol=min(self.tol, 1e-8), btol=min(self.tol, 1e-8), maxiter=self.max_iter)
            pilot = torch.as_tensor(pilot, dtype=torch.float64, device=op.device)
            penalty_weights_t = torch.pow(pilot.abs() + self.eps, -self.gamma)
            if not bool(torch.isfinite(penalty_weights_t).all().item()):
                raise RuntimeError("Resident GPU ALASSO pilot produced nonfinite penalty weights")
            # Keep solver weights on CUDA; retain only a diagnostic snapshot.
            penalty_weights = penalty_weights_t
            self.penalty_weights_ = _to_numpy(penalty_weights_t, np.float64)
            print("[gpu_resident] adaptive pilot=GPU CGLS gamma=%.6g weight_range=[%.6e, %.6e]" % (self.gamma, float(penalty_weights_t.min().item()), float(penalty_weights_t.max().item())), flush=True)
            if self.alpha_auto:
                weighted_kkt = torch.max(torch.abs(op.rmatvec(yt)) / torch.clamp(penalty_weights_t, min=torch.finfo(yt.dtype).tiny)) / A.shape[0]
                amax = float(weighted_kkt.item())
                if amax > 0 and np.isfinite(amax):
                    self.alphas = np.logspace(np.log10(amax) - self.decades, np.log10(amax), max(self.nalpha, len(self.alphas)))
        splits = _make_cv_splits(A.shape[0], self.cv, self.rand_seed, self.group_size)
        cv_tol = float(os.environ.get("PHEASY_CV_TOL", str(max(self.tol, 1e-3))))
        cv_cap = int(os.environ.get("PHEASY_CV_MAX_ITER", str(min(self.max_iter, 800))))
        if cv_cap < 1 or cv_tol <= 0:
            raise ValueError("CV max_iter and tol must be positive")
        # Replicate factors once per card, never once per fold. Copy primary
        # normalization and power estimate to keep numerical setup identical.
        # Every selected card must fit the FULL factors (preflighted on upload).
        devices = devices[:max(1, len(splits))]
        resources = [(op, yt, L)]
        for dev in devices[1:]:
            with _resident_device_context(dev):
                replica = GpuTwoLevelOperator(A, device_id=dev)
                owned.append(replica)
                replica.scale = op.scale.to(dev).clone()
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
                infos = []
                for i in range(len(self.alphas) - 1, -1, -1):
                    x, info = _fista_twolevel(worker, target, float(self.alphas[i]), x,
                                             cv_cap, cv_tol, estimate, trt,
                                             penalty_weights=penalty_weights, n_samples=int(trt.numel()))
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
        max_cv_iterations = max((info["n_iter"] for infos in self.cv_solver_info_ for info in infos), default=0)
        for replica in owned[1:]:
            replica.close()
        resources.clear()
        means = mse.mean(dim=1)
        if not bool(torch.isfinite(means).all().item()):
            raise RuntimeError("Resident CV produced nonfinite MSE")
        # Match iterative selection: ascending argmin chooses the smallest
        # alpha on exact ties. Near-tie tolerance is diagnostic only.
        best_i = int(torch.argmin(means).item())
        rtol = float(os.environ.get("PHEASY_LASSO_TIE_RTOL", "1e-9"))
        tied = means <= means[best_i] * (1 + rtol) + 1e-300
        self.alpha_ = float(self.alphas[best_i])
        # Preserve independent full-data descending warm-start path.
        x = None
        for i in range(len(self.alphas) - 1, best_i - 1, -1):
            x, path_info = _fista_twolevel(op, yt, float(self.alphas[i]), x, cv_cap, cv_tol, L,
                                         penalty_weights=penalty_weights, n_samples=A.shape[0])
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
        self._alpha_at_min_hitcap = self._alpha_at_min_flat and max_cv_iterations >= cv_cap
        if self._alpha_at_min:
            import warnings
            warnings.warn("Resident LASSO selected grid minimum%s" %
                          (" on a flat CV tail" if self._alpha_at_min_flat else ""), RuntimeWarning, stacklevel=2)
        # Do not retain VRAM after the fit. Predict and optional debias use the
        # ordinary public host interface, explicitly outside the resident stage.
        return self
