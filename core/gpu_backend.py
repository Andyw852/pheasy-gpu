"""GPU (CUDA via PyTorch) backend for pheasy's dense linear algebra.

Drop-in GPU replacements for the dense CPU primitives in core/optimizer.py.
Every public function takes NumPy arrays and returns NumPy arrays, so the
optimizer's control flow (CV grouping, alpha grids, standardization, debias,
RFE elimination) is unchanged -- only the heavy dense linear algebra moves to
the GPU.

Activation (in priority order):

1. Optimizer(..., use_gpu=True/False) -- explicit per-fit override.
2. PHEASY_USE_GPU env var: "0"/"false"/"off" forces CPU, "1"/"true"/"on"
   forces GPU, unset -> auto (GPU if available).
3. Auto mode uses the GPU when torch.cuda.is_available() is true.

Tuning knobs:

* PHEASY_GPU_DEVICE -- CUDA device index (default 0 / first visible device).
* PHEASY_GPU_DTYPE -- float64 (default, matches the CPU path) or float32
  (faster, ~half precision).

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
    "enabled",
    "set_gpu_mode",
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
]

_torch_mod = None
_mode = None          # None = auto, True = force on, False = force off
_device = None        # cached torch.device


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


def available():
    """True when torch + a CUDA device are importable."""
    t = _torch()
    if t is None:
        return False
    try:
        return bool(t.cuda.is_available())
    except Exception:
        return False


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
    global _device
    if _device is None:
        import torch
        dev = os.environ.get("PHEASY_GPU_DEVICE", None)
        if dev is not None:
            _device = torch.device("cuda:%d" % int(dev))
        else:
            _device = torch.device("cuda:0")
    return _device


def _dtype():
    import torch
    return torch.float64 if os.environ.get("PHEASY_GPU_DTYPE", "float64") == "float64" else torch.float32


def _to_torch(A, dtype=None):
    import torch
    if dtype is None:
        dtype = _dtype()
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
    """QR least squares via torch driver="gels" with an SVD fallback.

    Matches _solve_qr semantics: QR for full-rank systems, SVD when the QR
    result is unavailable / rank deficient.
    """
    import torch
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    At = _to_torch(A, torch.float64)
    yt = _to_torch(y, torch.float64).reshape(-1)
    try:
        coef = torch.linalg.lstsq(At, yt, driver="gels").solution
        return _to_numpy(coef, np.float64)
    except Exception:
        return lstsq(A, y)


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
    """||A||_2^2 = lambda_max(A^T A) by power iteration on the Gram (PSD)."""
    import torch
    n = Gt.shape[0]
    v = torch.randn(n, dtype=Gt.dtype, device=Gt.device)
    v = v / (v.norm() + 1e-300)
    for _ in range(power_iters):
        v = Gt @ v
        vn = v.norm()
        if vn < 1e-30:
            break
        v = v / vn
    u = Gt @ v
    L = float(u.dot(u).item())
    safety = float(os.environ.get("PHEASY_FISTA_LIPSCHITZ_SAFETY", "1.02"))
    return max(L, 1e-12) * safety


def _soft_threshold_t(x, thr):
    import torch
    # soft-threshold(x, thr) == x - clamp(x, -thr, thr): 2 elementwise ops
    # instead of sign(x)*max(|x|-thr, 0) (5 ops). thr may be a scalar or a
    # per-coordinate vector (penalty weights).
    return x - torch.clamp(x, min=-thr, max=thr)


def _fista_gram(Gt, bt, alpha, x0=None, max_iter=3000, tol=1e-7,
                lipschitz=None, penalty_weights=None, n_samples=None):
    """FISTA LASSO on the precomputed Gram: min 0.5||Ax-y||^2 + alpha sum w|x|.

    Mirrors optimizer._fista_lasso (Gram path) exactly, on GPU tensors.
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
    """Ridge CV over an alpha grid via generalized (leave-one-out) CV.

    Reproduces sklearn RidgeCV(alphas=..., fit_intercept=False, cv=None):
    for each alpha the ridge solution and the SVD hat-matrix leverage give the
    LOO error mean((y - Xw)^2 / (1 - h)^2); the alpha minimizing it wins.
    """

    def __init__(self, alphas, fit_intercept=False):
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.fit_intercept = fit_intercept

    def fit(self, A, y, sample_weight=None):
        import torch
        y64 = np.asarray(y, dtype=np.float64).ravel()
        At = _to_torch(A, torch.float64)
        yt = _to_torch(y64, torch.float64).reshape(-1)
        U, S, Vh = torch.linalg.svd(At, full_matrices=False)
        Ut_y = U.T @ yt
        S2 = S * S
        scores = []
        coefs = []
        for a in self.alphas:
            a = float(a)
            d = S / (S2 + a)
            w = Vh.T @ (d * Ut_y)
            ratio = S2 / (S2 + a)
            h = (U * U) @ ratio                 # hat-matrix diagonal (leverage)
            r = yt - At @ w
            denom = 1.0 - h
            score = float(((r * r) / (denom * denom)).mean().item())
            scores.append(score)
            coefs.append(w)
        best = int(np.argmin(scores))
        self.coef_ = _to_numpy(coefs[best], np.float64)
        self.alpha_ = float(self.alphas[best])
        self.intercept_ = 0.0
        self._gcv_scores_ = np.asarray(scores, dtype=np.float64)
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
    ns_harm / ns_anharm are dense 2-D arrays (or scipy sparse). Returns
    SM[:n_rows] as a dense NumPy array of the requested dtype.

    Falls back to a CPU scipy multiply when the GPU is unavailable.
    """
    import torch
    nsh = ns_harm.toarray() if hasattr(ns_harm, "toarray") else np.asarray(ns_harm)
    nsa = ns_anharm.toarray() if hasattr(ns_anharm, "toarray") else np.asarray(ns_anharm)
    nh, mh = nsh.shape
    na, ma = nsa.shape
    NS = np.zeros((nh + na, mh + ma), dtype=np.float64)
    NS[:nh, :mh] = nsh
    NS[nh:, mh:] = nsa

    if not enabled():
        SM = sm_prime @ NS
        return np.asarray(SM[:n_rows], dtype=dtype)

    NSt = torch.as_tensor(np.ascontiguousarray(NS), dtype=torch.float64,
                           device=device())
    # CSR path: ~5x faster and ~half the sparse-tensor memory of COO on
    # torch 2.x (indices are int32 and stored once). Fall back to COO if the
    # CSR kernel is unavailable.
    sm = sm_prime.tocsr()
    crow = torch.as_tensor(sm.indptr, dtype=torch.int32, device=device())
    ccol = torch.as_tensor(sm.indices, dtype=torch.int32, device=device())
    cval = torch.as_tensor(sm.data, dtype=torch.float64, device=device())
    try:
        spt = torch.sparse_csr_tensor(crow, ccol, cval, size=sm.shape,
                                      dtype=torch.float64, device=device())
        SM = torch.sparse.mm(spt, NSt)
    except Exception:
        smc = sm.tocoo()
        idx = torch.as_tensor(np.vstack([smc.row, smc.col]), dtype=torch.long,
                              device=device())
        vals = torch.as_tensor(smc.data, dtype=torch.float64, device=device())
        spt = torch.sparse_coo_tensor(idx, vals, smc.shape,
                                      device=device()).coalesce()
        SM = torch.sparse.mm(spt, NSt)
    out = SM[:n_rows]
    return _to_numpy(out, dtype)

