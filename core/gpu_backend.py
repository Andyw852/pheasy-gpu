"""GPU (CUDA via PyTorch) backend for pheasy's dense linear algebra.

Drop-in GPU replacements for the dense CPU primitives in core/optimizer.py.
Every public function takes NumPy arrays and returns NumPy arrays, so the
optimizer's control flow (CV grouping, alpha grids, standardization, debias,
RFE elimination) is unchanged -- only the heavy dense linear algebra moves to
the GPU.

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
    "ridge_solve",
    "gram",
    "top_eigval",
    "predict",
    "GpuLassoCV",
    "GpuRidgeCV",
    "load_sensing_matrix",
    "GpuSparseMV",
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
    m, n = At.shape
    if m < n:
        return lstsq(A, y)          # underdetermined -> min-norm SVD
    try:
        _R = torch.linalg.qr(At, mode="r")
        # mode="r" returns only R, but the container differs across torch
        # versions (named tuple with .R, or a plain (R,) tuple).
        R = _R.R if hasattr(_R, "R") else (_R[-1] if isinstance(_R, tuple) else _R)
        diag = R.diagonal().abs()
        if diag.numel() == 0:
            return lstsq(A, y)
        dmax = float(diag.max().item())
        if dmax == 0.0:
            return lstsq(A, y)
        tol = torch.finfo(torch.float64).eps * max(m, n) * dmax
        if float(diag.min().item()) <= tol:
            return lstsq(A, y)      # rank deficient -> SVD
        coef = torch.linalg.lstsq(At, yt, driver="gels").solution
    except Exception:
        return lstsq(A, y)
    if not bool(torch.isfinite(coef).all()):
        return lstsq(A, y)
    return _to_numpy(coef, np.float64)


def ridge_solve(A, y, alpha):
    """min ||A x - y||^2 + alpha ||x||^2 via the normal equations (GPU)."""
    import torch
    if float(alpha) <= 0:
        return lstsq(A, y)
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    At = _to_torch(A, torch.float64)
    yt = _to_torch(y, torch.float64).reshape(-1)
    n = At.shape[1]
    G = At.T @ At
    b = At.T @ yt
    Gp = G + float(alpha) * torch.eye(n, dtype=torch.float64, device=At.device)
    try:
        L = torch.linalg.cholesky(Gp)
        x = torch.cholesky_solve(b.reshape(-1, 1), L).reshape(-1)
    except Exception:
        x = torch.linalg.solve(Gp, b)
    return _to_numpy(x, np.float64)


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
                lipschitz=None, penalty_weights=None, n_samples=None):
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
                break
            x_prev = x.clone()
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
        coef_t, nfin = _fista_gram(G_full, b_full, self.alpha_, x0=best_x,
                                   max_iter=min(self.max_iter, 5000),
                                   tol=max(float(self.tol), 1e-7),
                                   lipschitz=lip_full, penalty_weights=pw,
                                   n_samples=n_samples)
        self.coef_ = _to_numpy(coef_t, np.float64)
        self.intercept_ = 0.0
        self.alphas_ = self.alphas
        self.mse_path_ = mse_path
        self.n_iter_ = int(nfin)
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
class GpuSparseMV(object):
    """Row-split SM_prime (+ transpose) across GPUs for TwoLevelSM matvec/rmatvec.

    The memory-heavy half of the TwoLevelSM is SM_prime (6.5-60 GB): matvec is
    SM_prime @ t, rmatvec is SM_prime.T @ u, while NS@v / NS.T@t stay on the
    CPU (small). Each device holds a row block of SM_prime (for matvec) and a
    row block of SM_prime.T (== a column block of SM_prime, for rmatvec), so
    peak VRAM = 2 x SM size / n_gpu. With 24 GB cards: a 60 GB f32 SM fits on
    3 cards; the same SM in f64 needs 6.

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
        self._dt64 = t.float64 if sm_prime.dtype == np.float64 else t.float32
        self._sm_bytes_val = int(sm_prime.data.nbytes + sm_prime.indices.nbytes)
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
        for i, d in enumerate(devs):
            dev = t.device("cuda:%d" % d)
            Ri = sm_prime[self._rs[i]:self._rs[i + 1]].tocsr()
            Ti = sm_prime[:, self._cs[i]:self._cs[i + 1]].T.tocsr()
            self._R.append(self._csr_to_torch(Ri, dev))
            self._T.append(self._csr_to_torch(Ti, dev))

    def _pick_devices(self, device_ids, n_gpu):
        import os as _os
        t = self._t
        if device_ids is None:
            raw = _os.environ.get("PHEASY_GPU_SM_DEVICES", "").strip()
            device_ids = [int(x) for x in raw.split(",")] if raw else None
        if device_ids is None:
            device_ids = list(range(t.cuda.device_count()))
        device_ids = [d for d in device_ids if 0 <= d < t.cuda.device_count()]
        if not device_ids:
            return []
        if n_gpu is None:
            raw = _os.environ.get("PHEASY_GPU_SM_NGPU", "").strip()
            n_gpu = int(raw) if raw else None
        if n_gpu is None or n_gpu <= 0:
            # auto: 2x SM size (matrix + transpose) over ~20 GB usable/card.
            # [FIX torch-stability] multi-GPU CSR @ dense segfaults after ~300
            # calls with 7 devices on the 699-SM blocks (540M nnz, wide T_i);
            # 5 devices is verified stable (1000-iter LSMR), so cap there.
            n_gpu = min(5, max(1, int(np.ceil(2.0 * self._sm_bytes_val / 20.0e9))))
        n_gpu = min(n_gpu, len(device_ids))
        return device_ids[:n_gpu]

    def _csr_to_torch(self, m, dev):
        t = self._t
        crow = t.as_tensor(m.indptr, dtype=t.int32, device=dev)
        ccol = t.as_tensor(m.indices, dtype=t.int32, device=dev)
        cval = t.as_tensor(m.data, dtype=self._dt64, device=dev)
        return t.sparse_csr_tensor(crow, ccol, cval, size=m.shape,
                                   dtype=self._dt64, device=dev)

    def _mv_blocks(self, blocks, x0):
        """blocks[i] @ x0 on each device; torch.sparse.mm on the big 699-SM
        blocks segfaults after ~300 calls (CSR beta bug); R @ xi (the @
        dispatch) is stable and periodic empty_cache releases the caching
        allocator so the fit survives 10k+ calls."""
        t = self._t
        np = self._np
        parts = []
        for i, B in enumerate(blocks):
            xi = t.as_tensor(x0, dtype=self._dt64, device=self._devs[i])
            yi = B @ xi
            parts.append(yi.cpu().numpy().astype(np.float64))
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

    def rmatvec(self, u):
        """SM_prime.T @ u -> (M,) numpy f64. u: (N,) numpy."""
        t = self._t
        np = self._np
        u0 = np.ascontiguousarray(u, dtype=np.float64)
        parts = []
        for i, T in enumerate(self._T):
            ui = t.as_tensor(u0, dtype=self._dt64, device=self._devs[i])
            ti = t.sparse.mm(T, ui.unsqueeze(1)).flatten()
            parts.append(ti.cpu().numpy().astype(np.float64))
        return np.concatenate(parts)

    def close(self):
        t = self._t
        try:
            for R, T in zip(self._R, self._T):
                del R, T
            t.cuda.empty_cache()
        except Exception:
            pass
