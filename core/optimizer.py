"""Classes and functions for force constant regression.

Implements the five force-constant fitting methods exposed by pheasy:
OLS, RFE, RFE-OLS-TSQR (RFE_TSQR), LASSO and ALASSO (adaptive LASSO), plus
the legacy RIDGE method. The public entry point is the Optimizer class.

References:
  H. Zou, "The Adaptive Lasso and Its Oracle Properties", JASA 101 (2006).
  F. Eriksson et al., Adv. Theory Simul. 2 (2019) (hiphive).
  J. Demmel et al., SIAM J. Sci. Comput. 34 (2012) A206 (TSQR).
"""
import contextlib
import os
import warnings

import numpy as np
import scipy.sparse as sp
from scipy import linalg as spla
from scipy.sparse.linalg import LinearOperator, lsmr as _lsmr

from sklearn.linear_model import LassoCV, Ridge, RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold


# ===== _sm_precision / _lsmr_tol (working-precision floor for LSMR tolerances) =====
def _sm_precision():
    """Working precision of the sensing matrix.

    Only an explicitly requested PHEASY_SM_DTYPE=float32 means float32.  When it
    is unset the caller is a library user (or a test) driving Optimizer directly
    on numpy float64 data, so the floor must not be applied to them -- defaulting
    this to float32 raised the tolerance of every dense float64 solve to 1.2e-6
    and broke the Jacobi equivalence checks, which need a tight tolerance
    precisely because they compare two paths to high accuracy.
    """
    name = os.environ.get("PHEASY_SM_DTYPE")
    if name is not None and str(name).strip().lower() in ("float32", "f32", "single"):
        return np.float32
    return np.float64


def _array_precision(A):
    """Precision actually carried by the operator's stored data.

    PHEASY_SM_DTYPE is only a proxy for it and an unreliable one: SM_prime's
    float32 is hard-coded in run_pheasy.py in ~30 places with no binding to that
    variable, so a production run can perfectly well have the variable unset and
    a float32 matrix -- exactly the case where the floor is needed.  Read the
    array in hand instead: the two-level operator exposes SM_prime, anything else
    exposes dtype.
    """
    for obj in (getattr(A, "SM_prime", None), A):
        dt = getattr(obj, "dtype", None)
        if dt is None:
            continue
        try:
            return np.dtype(dt).type
        except TypeError:
            pass
    name = os.environ.get("PHEASY_SM_DTYPE")
    if name is not None and str(name).strip().lower() in ("float32", "f32", "single"):
        return np.float32
    return np.float64


def _resident_value_dtype(op, torch):
    """The dtype the resident operator's stored data actually carries.

    Every vector handed to a resident matvec has to use it: the factors are
    float32 whenever PHEASY_SM_DTYPE=float32 (the production setting), and a
    float64 operand turns the sparse product into a mixed-dtype cuSPARSE SpMM,
    which is rejected with "matA (CUDA_R_32F) and matB (CUDA_R_64F) with
    different value types is not supported" -- the failure that kept resident RFE
    from ever starting, and earlier took out resident ridge.  Three separate
    omissions of this same rule have been found (GpuSubsetOperator._value_dtype,
    the ridge Augmented wrapper, and a hard-coded float64 target here), so it
    lives in one place now.
    """
    return getattr(op, "_value_dtype", None) or torch.float64


def _lsmr_tol(name, default, A=None):
    """atol/btol, raised to the floor the working precision can actually reach.

    scipy's LSMR stops when normar <= atol*normA*normr (and normr <= btol*normb).
    With PHEASY_SM_DTYPE=float32 the acting matrix carries ~1.2e-7 relative
    precision, so a requested 1e-8 can never be met: every solve burns its whole
    maxiter budget with istop=7, and the runtime stops depending on alpha or on
    the condition number at all.  Measured on the c6.5/c3=4.5 dataset: a ridge
    solve at alpha=1e-8 and one at alpha=1.127e2 (augmented condition number
    ~10, where the linear algebra says ~70 iterations) took the same order of
    time.  Raise the request to the floor and say so, rather than silently
    running thousands of useless iterations.
    """
    want = float(os.environ.get(name, str(default)))
    dt = _array_precision(A)
    floor = 10.0 * float(np.finfo(dt).eps)
    if want < floor:
        warnings.warn(
            "[lsmr] %s=%g is below the %s reachable floor %.3g; raised to the "
            "floor -- a tighter request only makes the stopping test "
            "unreachable and burns the iteration budget."
            % (name, want, np.dtype(dt).name, floor), stacklevel=2)
        return floor
    return want

try:
    from sparse_dot_mkl import dot_product_mkl as _mkl_dot
except Exception:                                   # pragma: no cover
    _mkl_dot = None


def _sp_mv(M, v):
    """Sparse matvec, MKL-multithreaded when sparse_dot_mkl is installed.

    scipy's CSR matvec is single-threaded and holds the GIL -- the reason the
    LSMR fit only showed ~1.9 cores. sparse_dot_mkl wraps mkl_sparse_?_mv
    (multithreaded, GIL-releasing) over the same CSR layout; a 1-D right-hand
    side is reshaped to a column and raveled back. Falls back to scipy's plain
    matmul (and to the native matmul for non-sparse M) when unavailable.
    """
    if _mkl_dot is not None and sp.issparse(M) and getattr(M, "_mkl_ok", True):
        try:
            out = _mkl_dot(M, np.ascontiguousarray(v).reshape(-1, 1), cast=False)
            return np.asarray(out).ravel()
        except Exception as e:
            M._mkl_ok = False
            print("[sp_mv] MKL unavailable for %s dtype=%s idx=%s (%s); "
                  "falling back to scipy"
                  % (M.shape, M.dtype, M.indices.dtype, e), flush=True)
    return M @ v


__all__ = ["Optimizer", "TwoLevelSM"]


def _avail_cores():
    """Usable CPU count for THIS job (affinity/cgroup aware), not the node's.

    os.cpu_count() reports the node's physical core count, which on a shared
    Slurm box overstates what --cpus-per-task actually granted. Prefer the
    process's CPU affinity; PHEASY_MAX_CORES overrides both (the shell sets it
    to $NCPU so the code never guesses the node width).
    """
    n = os.environ.get("PHEASY_MAX_CORES")
    if n:
        return max(1, int(n))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _resolve_n_jobs(method=None, default=-1):
    """Unified outer-parallelism resolver.

    Precedence: PHEASY_<METHOD>_N_JOBS > PHEASY_N_JOBS > 'default'.
    0 means serial, a value < 0 means "all available cores", and a positive
    value is clamped to [1, _avail_cores()]. Method names use '-' -> '_'
    (e.g. "RFE-OLS-TSQR" -> "RFE_OLS_TSQR").
    """
    raw = None
    if method:
        raw = os.environ.get("PHEASY_%s_N_JOBS" % method.upper().replace("-", "_"))
    if raw in (None, ""):
        raw = os.environ.get("PHEASY_N_JOBS")
    n = int(raw) if raw not in (None, "") else int(default)
    if n == 0:
        return 1          # 0 = serial (old behaviour + joblib convention)
    n_cpu = _avail_cores()
    return max(1, min(n if n > 0 else n_cpu, n_cpu))


@contextlib.contextmanager
def _blas_limit(n_outer):
    """Cap each worker's BLAS threads when the OUTER loop already runs n_outer.

    Without this, K outer folds x many inner BLAS threads oversubscribe the box
    and can be slower than serial. threadpool_limits sets the limit for the
    current process (all joblib threads see it); a missing threadpoolctl
    degrades to a no-op instead of raising.
    """
    per = max(1, _avail_cores() // max(1, n_outer))
    # Import OUTSIDE the yield scope: a `yield` inside `try` means any
    # ImportError raised by the *body* (the code inside `with _blas_limit`)
    # would be caught here and trigger a second yield ->
    # "RuntimeError: generator didn't stop after throw()", destroying the
    # original error. Deciding the import up front keeps body exceptions intact.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None
    if threadpool_limits is None:
        yield
    else:
        with threadpool_limits(limits=per):
            yield


def _gpu():
    """Lazily return the GPU backend module (None when unavailable/disabled)."""
    try:
        from . import gpu_backend
    except (ImportError, ValueError):
        try:
            from pheasy_gpu.core import gpu_backend
        except Exception:
            return None
    except Exception:
        return None
    if not gpu_backend.enabled():
        return None
    return gpu_backend


def _gpu_footprint_ok(gb, n, m):
    """True when the ~4x dense-solve footprint of an n x m matrix fits VRAM.

    The SVD path (lstsq) peaks at ~A + U + Vh + cuSOLVER workspace ~= 4x the
    A footprint (GPU.md measures ~3.2 GB for the 25515x3678 SM whose A alone
    is 0.75 GB), so estimate that worst case, not just A.
    """
    footprint = 4 * n * m * 8
    avail = gb.available_memory_bytes()
    if avail is None:
        return True
    frac = float(os.environ.get("PHEASY_GPU_MEM_FRACTION", "0.8"))
    return footprint <= avail * frac


def _gpu_required():
    """True when the execution context demands GPU and no explicit CPU override.

    Delegates the env mapping to gpu_backend.gpu_mode_from_env(), so
    PHEASY_USE_GPU=1 counts as "required" here EXACTLY as it does in the backend
    (and as README.md/GPU.md promise).  Reading only PHEASY_GPU_MODE left this
    False for PHEASY_USE_GPU=1 -- which the shipped fc-fit submit template
    exports -- and that silently disarmed every fail-closed guard below (they
    are no-ops unless the mode is required) and the resident default dispatch.

    PHEASY_GPU_MODE=required fails closed, but a per-instance use_gpu=False is an
    explicit CPU choice (set via set_gpu_mode(False) for the duration of fit) that
    overrides the ambient required mode. CPU baselines and explicit CPU fits must
    keep working under a required environment.
    """
    try:
        from . import gpu_backend as _gb
    except Exception:
        return False
    # An invalid PHEASY_GPU_MODE raises here on purpose: a typo must not read as
    # "not required" and silently turn the guarantees off.
    if _gb.gpu_mode_from_env() != "required":
        return False
    return _gb.get_gpu_mode() is not False


def _gpu_dense(A):
    """True when A should be solved on the GPU (dense, or sparse small enough to densify).

    A dense ndarray is accepted only when A plus its p x p Gram fit in
    PHEASY_GPU_MEM_FRACTION (default 0.8) of the free VRAM; this is the OOM
    fallback -- an oversized dense system degrades to the CPU instead of
    crashing the run. A sparse container is densified only when it is cheap
    enough on the host (PHEASY_MAX_DENSE / host-RAM budget) AND the same VRAM
    footprint gate passes.
    """
    gb = _gpu()
    if gb is None:
        return False
    if isinstance(A, np.ndarray):
        ok = _gpu_footprint_ok(gb, *A.shape)
        if not ok and _gpu_required():
            raise RuntimeError("GPU dense solve exceeds the configured VRAM budget; use a resident/iterative GPU path or explicit CPU mode")
        return ok
    if sp.issparse(A):
        # _should_densify_sparse only checks the HOST budget; the densified
        # matrix still has to fit VRAM, so run the same gate as the ndarray path.
        ok = _should_densify_sparse(A) and _gpu_footprint_ok(gb, *A.shape)
        if not ok and _gpu_required() and _should_densify_sparse(A):
            raise RuntimeError("GPU sparse densification exceeds the configured VRAM budget; use an iterative GPU path or explicit CPU mode")
        return ok
    return False


def _to_dense_f64(A):
    """Return A as a dense float64 ndarray (C-contiguous)."""
    if sp.issparse(A):
        return np.ascontiguousarray(A.toarray(), dtype=np.float64)
    if isinstance(A, np.ndarray):
        return np.ascontiguousarray(A, dtype=np.float64)
    if hasattr(A, "matvec"):  # LinearOperator (e.g. TwoLevelSM)
        n = A.shape[1]
        return np.ascontiguousarray(A @ np.eye(n, dtype=np.float64), dtype=np.float64)
    return np.ascontiguousarray(A, dtype=np.float64)


def _is_linear_operator(A):
    return (not isinstance(A, np.ndarray)) and (not sp.issparse(A)) and hasattr(A, "matvec")


def _col_norms(A):
    """Exact ||A[:, j]|| in float64."""
    if isinstance(A, TwoLevelSM):
        return A.col_norms()
    if sp.issparse(A):
        sq = np.asarray(A.multiply(A).sum(axis=0)).ravel()
        return np.sqrt(sq.astype(np.float64))
    if isinstance(A, np.ndarray):
        A64 = A.astype(np.float64, copy=False)
        return np.sqrt(np.einsum("ij,ij->j", A64, A64))
    n = A.shape[1]
    norms = np.zeros(n, dtype=np.float64)
    block = 64
    for j0 in range(0, n, block):
        j1 = min(j0 + block, n)
        I = np.zeros((n, j1 - j0), dtype=np.float64)
        I[j0:j1, :] = np.eye(j1 - j0, dtype=np.float64)
        col = A @ I
        norms[j0:j1] = np.sqrt(np.einsum("ij,ij->j", col, col))
    return norms


def derive_alpha_grid(A, y, nalpha=100, decades=4.0, standardize=False,
                     mu_shift=0.0):
    """Derive a LASSO/ALASSO alpha grid from the data.

    alpha_max = max_j |X_j^T y| / n  is the smallest alpha for which the LASSO
    solution is all zeros (the KKT threshold, matching sklearn's convention).
    The grid spans ``[alpha_max * 10**-decades, alpha_max]``.  When
    ``standardize`` is True the columns are first scaled to unit L2 norm (the
    same scaling the Optimizer applies), so the returned grid lives in the
    standardized space.

    Memory efficient: chunked accumulation for dense (mmap-friendly) input,
    sparse matvec for sparse / LinearOperator input.

    Returns a float64 array of alpha VALUES.
    """
    n = A.shape[0]
    p = A.shape[1]
    y64 = np.asarray(y, dtype=np.float64).ravel()
    g = np.zeros(p, dtype=np.float64)

    if sp.issparse(A) or _is_linear_operator(A):
        g = np.asarray(A.T @ y64).ravel().astype(np.float64)
        if standardize:
            cn = _col_norms(A)
            cn = np.where(cn < 1e-30, 1.0, cn)
            g = g / cn
    else:
        s2 = np.zeros(p, dtype=np.float64)
        blk = max(1, int(2e8 // max(p, 1)))
        for i0 in range(0, n, blk):
            B = np.asarray(A[i0:i0 + blk], dtype=np.float64)
            g += B.T @ y64[i0:i0 + B.shape[0]]
            if standardize:
                s2 += (B * B).sum(axis=0)
        if standardize:
            cn = np.sqrt(s2)
            cn = np.where(cn < 1e-30, 1.0, cn)
            g = g / cn

    a_max = float(np.abs(g).max()) / n
    a_max *= 10.0 ** float(mu_shift)
    if not np.isfinite(a_max) or a_max <= 0:
        raise ValueError("alpha_max = %r, invalid" % a_max)
    a_min = a_max * 10.0 ** (-float(decades))
    return np.logspace(np.log10(a_min), np.log10(a_max), nalpha)


def _make_cv_splits(n_samples, cv, random_state=None, group_size=None):
    """Return a list of (train_idx, val_idx) index arrays.

    Identical to core/gpu_backend._make_cv_splits (GPU and CPU CV must agree).
    """
    global _WARNED_UNGROUPED_CV, _WARNED_NO_SEED
    if cv is None or cv <= 1:
        cv = min(3, n_samples)
    cv = int(cv)
    if group_size and group_size > 1:
        if n_samples % group_size == 0:
            groups = np.arange(n_samples) // group_size
            n_groups = int(groups[-1]) + 1
            if n_groups >= 2:
                # Clamp cv to the number of configurations. A row-based KFold here
                # would leak rows of the same configuration across train/val folds,
                # silently biasing the CV estimate (FIX P23).
                eff_cv = int(min(cv, n_groups))
                gkf = GroupKFold(n_splits=eff_cv)
                return list(gkf.split(np.zeros(n_samples, dtype=np.int8),
                                      np.zeros(n_samples, dtype=np.int8), groups))
        else:
            # Grouped CV is impossible when the group size does not tile the row
            # count.  Falling back SILENTLY was the dangerous part: rows of one
            # configuration then sit in both folds, which leaks and biases the
            # selected alpha toward 0 (measured: 0 % -> 91.7 % of validation rows
            # sharing a configuration with training).
            warnings.warn(
                "PHEASY_CV_GROUP_SIZE=%s does not divide n_samples=%d: grouped "
                "cross-validation is impossible, so this fit falls back to "
                "shuffled ROW-based KFold.  Rows of one configuration then appear "
                "in both folds, which leaks information and biases alpha*/ridge "
                "toward 0.  Fix PHEASY_CV_GROUP_SIZE (it must be 3*natom and must "
                "divide the row count)." % (group_size, n_samples),
                RuntimeWarning, stacklevel=2)
    # no group info (or a single configuration): fall back to row-based KFold
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


def _iterative_solver_info(result, solver):
    """Expose stopping diagnostics and warn when an iterative solve stops early."""
    if isinstance(result, dict):
        return dict(result, solver=solver, converged=bool(result.get("converged", False)),
                    itn=int(result.get("n_iter", result.get("itn", 0))))
    istop = int(result[1])
    info = {"solver": solver, "istop": istop, "itn": int(result[2]),
            "normr": float(result[3]),
            "normar": float(result[7] if solver == "LSQR" else result[4]),
            "conda": float(result[6]),
            "converged": istop in (0, 1, 2, 4, 5)}
    if not info["converged"]:
        reason = ("iteration limit reached" if istop == 7 else
                  "condition limit reached" if istop in (3, 6) else
                  "unexpected stopping status")
        warnings.warn(
            "%s did not converge: %s (istop=%d, iterations=%d, "
            "normr=%.6e, normar=%.6e). Check the fit residual and solver "
            "tolerances/iteration limit before using these coefficients."
            % (solver, reason, istop, info["itn"], info["normr"], info["normar"]),
            RuntimeWarning, stacklevel=3)
    return info


def _solve_sparse_lsqr(A, y, info=None):
    """Iterative least squares (LSQR) for sparse / LinearOperator input.

    Only needs matvec / rmatvec, so peak memory is ~O(n_features) instead of
    densifying the sensing matrix (which for e.g. 685968 x 51590 would be
    ~280 GB). This is the same Krylov approach used by symfc / phonopy.
    """
    from scipy.sparse.linalg import lsqr as _sp_lsqr
    atol = float(_lsmr_tol("PHEASY_LSQR_ATOL", 1e-8))
    btol = float(_lsmr_tol("PHEASY_LSQR_BTOL", 1e-8))
    iter_lim = int(os.environ.get("PHEASY_LSQR_MAXITER", "5000"))
    y64 = np.asarray(y, dtype=np.float64).ravel()
    res = _sp_lsqr(A, y64, atol=atol, btol=btol, iter_lim=iter_lim)
    diagnostics = _iterative_solver_info(res, "LSQR")
    if info is not None:
        info.update(diagnostics)
    return np.asarray(res[0], dtype=np.float64)


def _available_memory_bytes():
    """Available physical RAM in bytes (None if undetectable).

    [FIX P26] Reads the OS's available-memory estimate (not total) so the
    dense/iterative dispatch reflects what the node can actually hand out right
    now, not just a fixed element budget.
    """
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def _should_densify_sparse(A):
    """True when densifying a sparse matrix is cheap enough (memory budget).

    A sparse container that is actually 100% dense (e.g. SM = SM_prime @ NS for
    small systems) is densified so the faster SVD/QR solvers run; genuinely
    sparse / huge matrices stay sparse and use the iterative LSQR solver.

    [FIX P26] now memory-aware: besides the PHEASY_MAX_DENSE element cap, the
    float64 footprint must fit in PHEASY_SOLVER_MEM_FRACTION of the available
    RAM (default 0.25). This keeps a large system on the iterative path even
    when its element count is small on paper but the node is already loaded.
    """
    n, m = A.shape
    max_dense = int(os.environ.get("PHEASY_MAX_DENSE", "200000000"))
    if (n * m) > max_dense:
        return False
    avail = _available_memory_bytes()
    if avail is not None:
        frac = float(os.environ.get("PHEASY_SOLVER_MEM_FRACTION", "0.25"))
        if (n * m * 8) > avail * frac:
            return False
    return True


def _resident_lasso_requested():
    """Explicit opt-in for the resident two-level LASSO/ALASSO backend.

    The initial development spelling (PHEASY_GPU_TWOLEVEL_LASSO) is an alias.
    This returns True only when the user explicitly requested the resident path;
    required-GPU-mode defaulting is handled separately by
    _resident_lasso_active so dense/sparse inputs keep their own GPU dispatch.
    """
    return os.environ.get("PHEASY_GPU_LASSO_RESIDENT",
                          os.environ.get("PHEASY_GPU_TWOLEVEL_LASSO", "0")).lower() in ("1", "true", "yes", "on")


def _resident_twolevel_input(A):
    return isinstance(A, TwoLevelSM) or hasattr(A, "_twolevel_base")


def _gpu_sm_explicit():
    """True when the narrower GPU-SM (sparse matvec) path was explicitly enabled."""
    return os.environ.get("PHEASY_GPU_SM", "0").lower() in ("1", "true", "yes", "on")


def _resident_default():
    """Production default selects a resident GPU path: required GPU mode without an
    explicit GPU-SM override. PHEASY_GPU_SM=1 is an explicit, narrower GPU path and
    must keep its own dispatch instead of being shadowed by the resident default."""
    return _gpu_required() and not _gpu_sm_explicit()


def _resident_lasso_explicitly_disabled():
    """True when the caller explicitly turned the resident backend OFF.

    Required GPU mode makes the resident path the default, but an explicit
    PHEASY_GPU_LASSO_RESIDENT=0 (which the shipped submit template exports, via
    ${...:-0}) must win: otherwise a deliberate opt-out is silently ignored and a
    run that cannot afford the resident factors is forced onto them.
    """
    for _key in ("PHEASY_GPU_LASSO_RESIDENT", "PHEASY_GPU_TWOLEVEL_LASSO"):
        raw = os.environ.get(_key)
        if raw is not None and raw.strip().lower() in ("0", "false", "no", "off"):
            return True
    return False


def _resident_lasso_active(A):
    """True when the resident two-level LASSO/ALASSO backend should run for A.

    Resident dispatch only applies to TwoLevelSM input. It activates on an
    explicit PHEASY_GPU_LASSO_RESIDENT=1, or by default in required GPU mode
    (where every supported main solve must run on the GPU) unless an explicit
    GPU-SM path was requested. Dense and ordinary sparse inputs are excluded
    here so they keep their own dense-GPU dispatch.
    """
    if _resident_lasso_explicitly_disabled():
        return False
    return _resident_twolevel_input(A) and (_resident_lasso_requested() or _resident_default())


_WARNED_UNGROUPED_CV = False
_WARNED_NO_SEED = False      # _make_cv_splits: non-reproducible shuffled KFold


def _lasso_backend(A):
    """Choose the LASSO/ALASSO backend: "dense" (sklearn) or "iterative" (FISTA).

    sklearn's LassoCV / RidgeCV only accept a materialized array, so any
    LinearOperator -- and any sparse matrix too big to densify -- must go
    through the matvec-only FISTA solver instead. This is the same dispatch
    policy _solve_lstsq already uses for OLS.
    """
    if _resident_lasso_active(A):
        # Pre-flight the footprint BEFORE the expensive path.  The resident
        # backend must hold both factors on one device; catching it here costs
        # seconds, while catching it inside the operator costs the factor load
        # (measured: 2 min into the fit, after a 3.6 GB SM_prime read).
        from . import gpu_backend as _gbp
        try:
            peak, budget, _free, _frac = _gbp.resident_twolevel_estimate(A)
        except Exception:   # never let the pre-flight itself become the bug
            peak = budget = None
        if peak is not None and budget is not None and peak > budget:
            # Fail closed by default: the documented contract is that an enabled
            # GPU path raises rather than quietly running something else (see
            # GPU.md).  PHEASY_GPU_RESIDENT_FALLBACK=1 opts into a substitute,
            # but the substitute must be named accurately: the fall-through
            # below is the plain FISTA backend, which is a GPU solve ONLY when
            # PHEASY_GPU_SM=1 shards the SM_prime matvec onto CUDA.  The old
            # message announced the GPU-SM path unconditionally, so a required
            # -mode run could return an unmarked CPU result.
            if os.environ.get("PHEASY_GPU_RESIDENT_FALLBACK", "0").lower() in ("1", "true", "yes", "on"):
                gpu_sm = _gpu_sm_explicit()
                if _gpu_required() and not gpu_sm:
                    raise MemoryError(_gbp.resident_twolevel_error_message(peak, budget))
                warnings.warn(
                    "Resident LASSO does not fit this device (%d > %d bytes); "
                    "PHEASY_GPU_RESIDENT_FALLBACK=1 switches to the %s."
                    % (peak, budget,
                       "two-level GPU-SM matvec path (SM_prime sharded over "
                       "PHEASY_GPU_SM_NGPU cards)" if gpu_sm else
                       "CPU FISTA 'iterative' path -- set PHEASY_GPU_SM=1 to "
                       "keep the solve on the GPU"),
                    RuntimeWarning, stacklevel=2)
            else:
                raise MemoryError(_gbp.resident_twolevel_error_message(peak, budget))
        else:
            # Selection is independent of availability: execution must fail closed.
            return "gpu_resident"
    if _is_linear_operator(A):
        return "iterative"
    if sp.issparse(A) and not _should_densify_sparse(A):
        return "iterative"
    if _gpu_dense(A) and os.environ.get("PHEASY_GPU_LASSO", "1").lower() not in ("0", "false", "no", "off"):
        return "gpu"
    return "dense"


def _solve_lstsq(A, y, driver="gelsd"):
    """Ordinary least squares: SVD for dense/small, LSQR for sparse/huge."""
    if _is_linear_operator(A):
        return _solve_sparse_lsqr(A, y)
    if _gpu_dense(A):
        return np.asarray(_gpu().lstsq(_to_dense_f64(A), y), dtype=np.float64)
    if sp.issparse(A) and not _should_densify_sparse(A):
        return _solve_sparse_lsqr(A, y)
    A64 = _to_dense_f64(A)
    y64 = np.asarray(y, dtype=np.float64).ravel()
    try:
        coef, *_ = spla.lstsq(A64, y64, cond=None, lapack_driver=driver)
    except Exception:
        coef, *_ = spla.lstsq(A64, y64, cond=None, lapack_driver="gelsd")
    return np.asarray(coef, dtype=np.float64)


def _tsqr_qless(A, y, block_rows=40000, diag_floor=1e-12):
    """[FIX P09/P24] Q-less tall-skinny QR least squares via a binary TREE.

    Level 0 QR-factors each row block independently, then the R factors are
    paired and re-QR'd in a binary tree (log-depth).  This is the classic TSQR
    of Demmel et al. and is numerically more stable than sequentially re-QRing
    one growing [R; A_i] stack (the previous implementation), which lets
    rounding error accumulate along the chain for ill-conditioned matrices.

    Peak memory is O(ceil(log2(nblocks)) * n^2 + block_rows * n): same-level
    R factors are merged on the fly, so at most one R per tree level is alive.
    Never a full Q or a full copy of A.

    Returns (coef, rank_ok, cond_estimate).
    """
    m, n = A.shape
    y64 = np.asarray(y, dtype=np.float64).ravel()
    # [FIX P36] no longer force block_rows >= n+1: the wide-matrix branch in
    # the final solve handles level-0 R factors that are wider than tall (which
    # the tree reduction accumulates to full rank). Letting the user pick a
    # small block_rows shrinks the block densification buffer from
    # block_rows*n*8 to blk*n*8 (e.g. cutoff 5.5: 21 GB -> 1.7 GB at blk=4000).
    # [FIX P36c] block-height guardrail: below ~n/4 the QR-call overhead grows
    # without any memory savings (the final n^2 R dominates the peak), so clamp
    # the effective height to at least min(n, 2048) and say so.  Correctness is
    # unaffected -- this only keeps users out of the counterproductive range.
    _blk = int(block_rows) if block_rows else 0
    _floor = min(n, 2048)
    if 0 < _blk < _floor and _blk < n // 4:
        print("[optimizer] WARNING: TSQR block_rows=%d << n=%d; effective height "
              "raised to %d (tiny blocks add QR-call overhead, not memory "
              "savings)." % (_blk, n, _floor), flush=True)
    block_rows = max(_blk, _floor)

    # ---- level 0: independent QR of each block --------------------------------
    # [FIX P36] streaming binary-tree TSQR: merge same-level R factors on the
    # fly, so at most ceil(log2(nblocks))+1 R matrices are alive at any time
    # (peak O(ceil(log2(nblocks))*n^2 + block_rows*n)) instead of materialising
    # every level-0 R before reducing (O(nblocks*n^2), which for wide/small-
    # block problems exceeds even O(m*n)). Same binary tree as the batch
    # version, just a different traversal order -- numerically equivalent.
    stack = []  # list of (level, R, z)
    for i0 in range(0, m, block_rows):
        i1 = min(i0 + block_rows, m)
        blk = A[i0:i1]
        blk = np.asarray(blk.toarray() if sp.issparse(blk) else blk, dtype=np.float64)
        Q, R = spla.qr(blk, mode="economic", check_finite=False)
        z = np.asarray(Q.T @ y64[i0:i1], dtype=np.float64)
        del Q, blk
        lvl = 0
        while stack and stack[-1][0] == lvl:   # merge same-level on the fly
            _l0, R0, z0 = stack.pop()
            M = np.vstack([R0, R])
            b = np.concatenate([z0, z])
            Q, R = spla.qr(M, mode="economic", check_finite=False)
            z = np.asarray(Q.T @ b, dtype=np.float64)
            del Q, M, b, R0, z0
            lvl += 1
        stack.append((lvl, R, z))
    while len(stack) > 1:                       # finish the residual tree
        l1, R1, z1 = stack.pop()
        l0, R0, z0 = stack.pop()
        M = np.vstack([R0, R1])
        b = np.concatenate([z0, z1])
        Q, R = spla.qr(M, mode="economic", check_finite=False)
        z = np.asarray(Q.T @ b, dtype=np.float64)
        stack.append((max(l0, l1) + 1, R, z))
    R = stack[0][1]
    z = stack[0][2]
    wide = R.shape[0] < R.shape[1]
    diag = np.abs(np.diag(R))
    dmax = float(diag.max()) if diag.size else 0.0
    dmin = float(diag.min()) if diag.size else 0.0
    if wide:
        # [FIX P36] diag of a wide R does not reflect conditioning.
        cond = np.nan
    else:
        cond = (dmax / dmin) if dmin > 0 else np.inf
    if dmax == 0.0 or (not wide and dmin <= diag_floor * max(dmax, 1.0)):
        return None, False, cond
    if wide:
        # [FIX P35] underdetermined (m < n): the reduced R is (m, n) and not
        # square, so the diag-based rank criterion is meaningless and
        # solve_triangular would raise. min-norm lstsq == gelsd on the
        # original A (e.g. c3=7.0 with a small NDATA, or small block_rows).
        coef = spla.lstsq(R, z, check_finite=False)[0]
    else:
        coef = spla.solve_triangular(R, z, lower=False, check_finite=False)
    return np.asarray(coef, dtype=np.float64), True, cond


def _solve_qr(A, y, block_rows=None, diag_floor=1e-12):
    """OLS via tall-skinny QR (Householder QR + triangular solve).

    Numerically stable for full-column-rank matrices. Sparse / LinearOperator
    input falls back to LSQR (scipy has no sparse QR least-squares driver; LSQR
    is a Golub-Kahan bidiagonalization, QR-like, and memory efficient).

    [FIX P09] ``block_rows`` now actually streams the factorization instead of
    being an ignored constructor argument; ``diag_floor`` guards the triangular
    solve against a rank-deficient R.
    """
    if _is_linear_operator(A):
        return _solve_sparse_lsqr(A, y)
    if sp.issparse(A) and not _should_densify_sparse(A):
        return _solve_sparse_lsqr(A, y)
    if block_rows and A.shape[0] > int(block_rows):
        if (os.environ.get("PHEASY_GPU_TSQR", "0").lower() in ("1", "true", "yes", "on")
                and _gpu_dense(A)):
            try:
                coef, _gpu_info = _gpu().gpu_tsqr(_to_dense_f64(A), y, int(block_rows), diag_floor)
                return np.asarray(coef, dtype=np.float64)
            except (RuntimeError, MemoryError, np.linalg.LinAlgError) as exc:
                if _gpu_required():
                    raise RuntimeError("GPU TSQR failed with fallback disabled: %s" % exc) from exc
                # Preserve CPU TSQR/SVD semantics only in auto mode.
                pass
        # [FIX P35] _tsqr_qless can raise (e.g. wide matrices); catch so the
        # SVD fallback actually runs instead of propagating the exception.
        try:
            coef, ok, _cond = _tsqr_qless(A, y, block_rows, diag_floor)
        except Exception:
            ok = False
        if ok:
            return coef
        return _solve_lstsq(A, y)     # rank deficient / wide -> SVD
    if _gpu_dense(A):
        return np.asarray(_gpu().qr_solve(_to_dense_f64(A), y), dtype=np.float64)
    A64 = _to_dense_f64(A)
    y64 = np.asarray(y, dtype=np.float64).ravel()
    Q, R = spla.qr(A64, mode="economic", check_finite=False)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return _solve_lstsq(A, y)
    dmax = float(diag.max())
    if dmax == 0.0:
        return _solve_lstsq(A, y)
    # rank threshold relative to the largest R diagonal (proxy for largest
    # singular value); fall back to SVD when (nearly) rank deficient.
    tol = np.finfo(float).eps * max(A64.shape) * dmax
    if float(diag.min()) <= tol:
        return _solve_lstsq(A, y)
    if R.shape[0] == R.shape[1]:
        coef = spla.solve_triangular(R, Q.T @ y64, lower=False, check_finite=False)
    else:
        # [FIX P35] wide / underdetermined (n_rows < n_features): the economic
        # QR gives a non-square R, so solve_triangular fails.  The min-norm
        # least-squares solution of R x = Q^T y equals gelsd on the original
        # A, which is what the other solvers return for the underdetermined
        # case (e.g. c3=7.0 with a small NDATA).
        coef = spla.lstsq(R, Q.T @ y64, check_finite=False)[0]
    return np.asarray(coef, dtype=np.float64)


def _make_masked_op(A, row_idx, col_idx):
    """LinearOperator for A[row_idx][:, col_idx] without materializing A."""
    n_rows_full, n_cols_full = A.shape
    n_rows = n_rows_full if row_idx is None else len(row_idx)
    n_cols = len(col_idx)
    dt = np.dtype(A.dtype) if hasattr(A, "dtype") else np.dtype(np.float64)

    def mv(v):
        v = np.asarray(v, dtype=dt).ravel()
        v_full = np.zeros(n_cols_full, dtype=dt)
        v_full[col_idx] = v
        out = np.asarray(A @ v_full).ravel()
        return out if row_idx is None else out[row_idx]

    def rmv(u):
        u = np.asarray(u, dtype=dt).ravel()
        if row_idx is not None:
            u_full = np.zeros(n_rows_full, dtype=dt)
            u_full[row_idx] = u
        else:
            u_full = u
        return np.asarray(A.T @ u_full).ravel()[col_idx]

    return LinearOperator((n_rows, n_cols), matvec=mv, rmatvec=rmv, dtype=dt)


def _soft_threshold(x, thr):
    """Elementwise soft-thresholding: sign(x) * max(|x| - thr, 0)."""
    x = np.asarray(x)
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


def _estimate_lipschitz(A, power_iters=15):
    """Estimate L = ||A||_2^2 (top eigenvalue of A^T A) by power iteration.

    Only needs matvec/rmatvec, so it works for dense, sparse and
    LinearOperator (TwoLevelSM) alike. L is the Lipschitz constant of the
    LASSO smooth part.
    """
    n = A.shape[1]
    v = np.random.RandomState(0).randn(n)
    v = v / (np.linalg.norm(v) + 1e-300)
    for _ in range(power_iters):
        u = np.asarray(A @ v, dtype=np.float64).ravel()
        v = np.asarray(A.T @ u, dtype=np.float64).ravel()
        vn = np.linalg.norm(v)
        if vn < 1e-30:
            break
        v = v / vn
    u = np.asarray(A @ v, dtype=np.float64).ravel()
    L = max(float(np.dot(u, u)), 1e-12)
    # [FIX P29] power iteration approaches lambda_max from BELOW, so the bare
    # estimate under-shoots L and makes step = 1/L too large: FISTA can become
    # non-monotonic or even diverge when the spectral gap is small.  A small
    # safety margin keeps the step conservative (override via env if needed).
    safety = float(os.environ.get("PHEASY_FISTA_LIPSCHITZ_SAFETY", "1.02"))
    return L * safety


def _top_eigval(G):
    """Largest eigenvalue of a symmetric PSD matrix (exact LAPACK).

    [FIX P34] ||A||^2 = lambda_max(A^T A) exactly for the Gram path, replacing
    the power-iteration estimate (and its safety factor) with a tighter step.
    """
    G = np.asarray(G, dtype=np.float64)
    n = G.shape[0]
    if n == 0:
        return 0.0
    try:
        return float(spla.eigvalsh(G, subset_by_index=(n - 1, n - 1))[0])
    except TypeError:
        # older scipy without subset_by_index: full eigendecomposition fallback
        return float(spla.eigvalsh(G)[-1])


def _gram_smprime(SM_prime, block_rows=2000):
    """P = SM_prime^T SM_prime via blocked densification + BLAS gemm.

    [FIX P34] SM_prime (n x mid) may be huge and sparse; densifying it all at
    once costs n*mid*8 bytes. Processing row blocks keeps peak memory at
    block_rows*mid*8 and accumulates the mid x mid Gram with BLAS.
    """
    n, mid = SM_prime.shape
    P = np.zeros((mid, mid), dtype=np.float64)
    for i0 in range(0, n, block_rows):
        B = np.asarray(SM_prime[i0:i0 + block_rows].toarray(), dtype=np.float64)
        P += B.T @ B
    return P


def _gram_budget_ok(A, max_gb=None):
    """[FIX P34] True if the Gram matrices fit the memory budget.

    For TwoLevelSM the dominant intermediate is P = SM_prime^T SM_prime
    (mid x mid), which can exceed G (n_features x n_features) when the null
    space projects a lot of columns away. Default 4 GB ~ 23000 dof.
    """
    max_gb = float(os.environ.get("PHEASY_GRAM_MAX_GB", "4")) if max_gb is None else float(max_gb)
    mid = getattr(A, "SM_prime", None)
    peak_dim = mid.shape[1] if mid is not None else A.shape[1]
    return peak_dim * peak_dim * 8.0 / 1e9 <= max_gb


def _compute_gram(A, y):
    """Precompute G = A^T A (n_features x n_features) and b = A^T y.

    [FIX P34] For TwoLevelSM the Gram is built factorized (P = SM_prime^T
    SM_prime, then G = NS^T P NS) without ever materializing the full SM.
    Returns (G, b, P): P is the SM_prime Gram kept so folds can use the cheap
    P_full - P_va identity instead of recomputing per fold.
    """
    n, m = A.shape
    y64 = np.asarray(y, dtype=np.float64).ravel()
    b = np.asarray(A.T @ y64, dtype=np.float64).ravel()
    P = None
    if hasattr(A, "SM_prime"):  # TwoLevelSM
        P = _gram_smprime(A.SM_prime)
        G = np.asarray(A.NS.T @ (P @ A.NS), dtype=np.float64)
    elif sp.issparse(A):
        G = np.asarray((A.T @ A).toarray(), dtype=np.float64)
    elif _is_linear_operator(A):
        # [FIX P34] build G column-block-wise so a bare LinearOperator does NOT
        # materialize the full dense SM (n_rows x n_features).  Peak memory is
        # n_rows x blk instead.
        G = np.zeros((m, m), dtype=np.float64)
        blk = int(os.environ.get("PHEASY_GRAM_BLOCK", "64"))
        for j0 in range(0, m, blk):
            j1 = min(j0 + blk, m)
            I_blk = np.zeros((m, j1 - j0), dtype=np.float64)
            I_blk[j0:j1, :] = np.eye(j1 - j0, dtype=np.float64)
            A_blk = np.asarray(A @ I_blk, dtype=np.float64)   # n_rows x blk
            G[:, j0:j1] = np.asarray(A.T @ A_blk, dtype=np.float64)
    else:
        A64 = np.asarray(A, dtype=np.float64)
        G = A64.T @ A64
    return G, b, P


def _fista_lasso(A, y, alpha, x0=None, max_iter=3000, tol=1e-7,
                 lipschitz=None, penalty_weights=None, _info=None,
                 gram=None, n_samples=None, warn_nonconvergence=True):
    """Solve min 0.5||A x - y||^2 + alpha * sum_j w_j |x_j| via FISTA.

    [FIX P26] Matvec-only LASSO so LASSO / ALASSO can run on a TwoLevelSM /
    LinearOperator and on genuinely-huge sparse matrices without ever
    building the dense sensing matrix; peak memory is ~O(n_features).
    Warm-started from x0 when given (the alpha path is warm-started from
    the previous alpha).

    penalty_weights (w_j) implements the adaptive-LASSO penalty directly in
    the ORIGINAL column space (per-coordinate soft-threshold) instead of
    column-scaling A by 1/w. That keeps the Lipschitz constant at ||A||^2
    rather than ||A / w||^2, so the ALASSO adaptive scaling no longer slows
    FISTA down.
    """
    n = A.shape[1]
    if n_samples is None:
        n_samples = A.shape[0]
    y64 = np.asarray(y, dtype=np.float64).ravel()
    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64).copy()

    if alpha <= 0:
        # [FIX P34] alpha<=0 is never produced by derive_alpha_grid, so this only
        # matters for direct API use. It still solves A x = y via LSQR (not the
        # Gram normal equation G x = b, which squares the condition number).
        coef = _solve_sparse_lsqr(A, y64)
        rhs = np.asarray(A.T @ y64).ravel()
        gradient = np.asarray(A.T @ (np.asarray(A @ coef).ravel() - y64)).ravel()
        scale = max(float(np.max(np.abs(rhs), initial=0.0)), np.finfo(float).tiny)
        kkt = float(np.max(np.abs(gradient), initial=0.0)) / scale
        converged = bool(np.isfinite(kkt) and kkt <= tol)
        if _info is not None:
            _info.update(n_iter=0, converged=converged, kkt_relative=kkt)
        if not converged:
            warnings.warn("FISTA zero-alpha least-squares result lacks stationarity: relative KKT=%g" % kkt,
                          RuntimeWarning, stacklevel=2)
        return coef

    if gram is not None:
        G, b = gram
    else:
        G = b = None
    if lipschitz is None:
        lipschitz = _top_eigval(G) if gram is not None else _estimate_lipschitz(A)
    step = 1.0 / max(float(lipschitz), 1e-12)
    # sklearn LassoCV convention is (1/(2 n_samples))||Ax-y||^2 + alpha||x||_1,
    # which equals 0.5||Ax-y||^2 + (n_samples*alpha)||x||_1 -- so the L1 weight
    # in the proximal step is scaled by n_samples to match the alpha grid.
    thr = float(alpha) * n_samples * step
    if penalty_weights is not None:
        thr = thr * np.asarray(penalty_weights, dtype=np.float64).ravel()

    # A small update is not an optimality certificate on ill-conditioned
    # columns. Check the L1 KKT residual at the actual iterate, not momentum z.
    rhs = np.asarray(b if gram is not None else A.T @ y64, dtype=np.float64).ravel()
    kkt_scale = max(float(np.max(np.abs(rhs), initial=0.0)), np.finfo(float).tiny)
    penalty = float(alpha) * n_samples
    if penalty_weights is not None:
        penalty = penalty * np.asarray(penalty_weights, dtype=np.float64).ravel()

    def kkt_relative(coef):
        gradient = (np.asarray(G @ coef).ravel() - b if gram is not None else
                    np.asarray(A.T @ (np.asarray(A @ coef).ravel() - y64)).ravel())
        violation = np.where(coef != 0, np.abs(gradient + penalty * np.sign(coef)),
                             np.maximum(np.abs(gradient) - penalty, 0.0))
        return float(np.max(violation, initial=0.0)) / kkt_scale

    converged = False
    kkt = float("inf")
    z = x.copy()          # momentum point y_0 = x_0
    t = 1.0
    x_prev = x.copy()
    n_iter = 0
    for it in range(int(max_iter)):
        n_iter = it + 1
        if gram is not None:
            # [FIX P34] gradient from the precomputed Gram: A^T(A z - y) = G z - b
            grad = np.asarray(G @ z, dtype=np.float64).ravel() - b
        else:
            Az = np.asarray(A @ z, dtype=np.float64).ravel()
            grad = np.asarray(A.T @ (Az - y64), dtype=np.float64).ravel()
        x_new = _soft_threshold(z - step * grad, thr)

        # FISTA with adaptive restart (O'Donoghue & Candes 2015): reset the
        # momentum when it points against the last step, which restores (near)
        # linear convergence on ill-conditioned problems like the ALASSO
        # adaptive-scaled matrix.
        if float(np.dot(z - x_new, x_new - x)) > 0.0:
            z = x_new
            t = 1.0
        else:
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = x_new + ((t - 1.0) / t_new) * (x_new - x)
            t = t_new
        x = x_new

        if it % 20 == 19:
            dx = float(np.linalg.norm(x - x_prev))
            if dx <= tol * max(1.0, float(np.linalg.norm(x))):
                kkt = kkt_relative(x)
                if np.isfinite(kkt) and kkt <= tol:
                    converged = True
                    break
            x_prev = x.copy()
    if not converged:
        kkt = kkt_relative(x)
        converged = bool(np.isfinite(kkt) and kkt <= tol)
    if _info is not None:
        _info.update(n_iter=n_iter, converged=converged, kkt_relative=kkt)
    if not converged and warn_nonconvergence:
        # CV folds pass warn_nonconvergence=False and report ONE aggregated line
        # instead: a ranking-only solve that stops at the cap is not the same
        # problem as an uncertified final refit, and 36 identical tracebacks
        # buried the real diagnostics.
        import warnings
        warnings.warn("FISTA did not converge: iterations=%d, relative KKT=%g, tol=%g"
                      % (n_iter, kkt, tol), RuntimeWarning, stacklevel=2)
    return x


def _scale_operator(A, w):
    """LinearOperator for the column-scaled A[:, j] / w[j], no materialization."""
    inv_w = (1.0 / np.asarray(w, dtype=np.float64)).ravel()

    def mv(v):
        v = np.asarray(v, dtype=np.float64).ravel()
        return np.asarray(A @ (v * inv_w), dtype=np.float64).ravel()

    def rmv(u):
        u = np.asarray(u, dtype=np.float64).ravel()
        return (np.asarray(A.T @ u, dtype=np.float64).ravel()) * inv_w

    result = LinearOperator(A.shape, matvec=mv, rmatvec=rmv, dtype=np.float64)
    if _resident_twolevel_input(A):
        # Preserve known column-scaling provenance, not arbitrary operators.
        result._twolevel_base = getattr(A, "_twolevel_base", A)
        result._twolevel_scale = np.asarray(w, dtype=np.float64).ravel() * getattr(A, "_twolevel_scale", 1.0)
    return result


def _row_slice_op(A, rows):
    """LinearOperator for A[rows, :] without materializing A (used by CV folds).

    [FIX P30] buffers are float64 so the FISTA gradient stays in float64 even
    when the underlying operator is float32 (the matvec itself still follows the
    operator's dtype, which is PHEASY_SM_DTYPE).
    """
    n = A.shape[1]
    dt = np.dtype(np.float64)
    rows = np.asarray(rows, dtype=np.intp)

    def mv(v):
        v = np.asarray(v, dtype=dt).ravel()
        return np.asarray(A @ v, dtype=dt).ravel()[rows]

    def rmv(u):
        u = np.asarray(u, dtype=dt).ravel()
        u_full = np.zeros(A.shape[0], dtype=dt)
        np.add.at(u_full, rows, u)  # repeated selected rows contribute additively
        return np.asarray(A.T @ u_full, dtype=dt).ravel()

    return LinearOperator((len(rows), n), matvec=mv, rmatvec=rmv, dtype=dt)


def _row_slice(A, rows):
    """A[rows, :] for dense / sparse / LinearOperator.

    [FIX P30] TwoLevelSM gets a true row-slice (slices SM_prime) so CV-fold
    matvecs cost O(nnz(SM_prime[rows])) instead of the full O(nnz(SM_prime)).
    """
    if hasattr(A, "_twolevel_base"):
        return _scale_operator(A._twolevel_base.row_slice(rows), A._twolevel_scale)
    if hasattr(A, "row_slice"):     # TwoLevelSM
        return A.row_slice(rows)
    if _is_linear_operator(A):
        return _row_slice_op(A, rows)
    return A[rows]


def _ridge_solve(A, y, alpha, x0=None):
    """min ||A x - y||^2 + alpha||x||^2 for dense / sparse / LinearOperator.

    x0 is a warm-start for the iterative (LinearOperator) branch: the alpha
    path walks large->small, so neighbouring alphas have nearly identical
    solutions and LSMR converges in a fraction of the cold-start iterations.
    """
    y64 = np.asarray(y, dtype=np.float64).ravel()
    if _is_linear_operator(A):
        # Narrow opt-in: keep TwoLevel factors resident and solve the augmented
        # ridge system with GPU CGLS. Any setup/kernel failure deliberately
        # falls through to the established CPU LSMR path.
        resident = (os.environ.get("PHEASY_GPU_RIDGE_RESIDENT", "1" if _resident_default() else "0").lower()
                    in ("1", "true", "yes", "on"))
        if resident and (hasattr(A, "SM_prime") or hasattr(A, "_twolevel_base")):
            try:
                from . import gpu_backend as _gb
                if _gb.enabled() and _gb.available():
                    # Build the resident operator ONCE per operator object, not once
                    # per (alpha, fold).  The CV sweep calls _ridge_solve
                    # len(alphas) x len(folds) times -- 21 on the c6.5/c3=4.5 fit --
                    # and every build uploads the whole factor set, so a per-call
                    # build makes the resident path slower than CPU LSMR no matter
                    # how fast CGLS is.  The operator is intentionally NOT closed
                    # here: it lives as long as the fold operator that produced it
                    # (one per CV fold), which bounds the residency at len(folds)
                    # replicas instead of len(alphas) x len(folds) uploads.
                    #
                    # device_ids spreads the factors over the selected cards.  The
                    # sharded layout is the one the matvec/rmatvec/col-norm gates
                    # cover; omitting it silently pinned resident RIDGE to one card
                    # while PHEASY_GPU_SM used three.
                    gpu_op = getattr(A, "_gpu_ridge_op", None)
                    if gpu_op is None:
                        gpu_op = _gb.GpuTwoLevelOperator(
                            A, device_ids=_gb.resident_device_ids())
                        try:
                            A._gpu_ridge_op = gpu_op
                        except Exception:
                            pass          # read-only operator: fall back to rebuild
                    coef, info = _gb.iterative_ridge(
                        gpu_op, y64, alpha,
                        atol=float(_lsmr_tol("PHEASY_LSQR_ATOL", 1e-8, A)),
                        btol=float(_lsmr_tol("PHEASY_LSQR_BTOL", 1e-8, A)),
                        maxiter=int(os.environ.get("PHEASY_LSQR_MAXITER", "5000")))
                    A._gpu_solver_info = _iterative_solver_info(info, "GPU CGLS-RIDGE")
                    return np.asarray(coef, dtype=np.float64)
            except Exception as exc:
                if _gpu_required():
                    raise RuntimeError("GPU Ridge solve failed with fallback disabled: %s" % exc) from exc
                pass
        n = A.shape[1]
        sqrt_a = float(np.sqrt(alpha)) if alpha > 0 else 0.0
        if sqrt_a > 0:
            def mv_aug(v):
                v = np.asarray(v, dtype=np.float64).ravel()
                return np.concatenate(
                    [np.asarray(A @ v, dtype=np.float64).ravel(), sqrt_a * v])

            def rmv_aug(u):
                u = np.asarray(u, dtype=np.float64).ravel()
                return (np.asarray(A.rmatvec(u[: A.shape[0]]), dtype=np.float64).ravel()
                        + sqrt_a * u[A.shape[0]:])

            op = LinearOperator((A.shape[0] + n, n), matvec=mv_aug,
                                rmatvec=rmv_aug, dtype=np.float64)
            y_aug = np.concatenate([y64, np.zeros(n)])
        else:
            op, y_aug = A, y64
        atol = float(_lsmr_tol("PHEASY_LSQR_ATOL", 1e-8))
        btol = float(_lsmr_tol("PHEASY_LSQR_BTOL", 1e-8))
        maxiter = int(os.environ.get("PHEASY_LSQR_MAXITER", "5000"))
        import time as _tr
        _t_r = _tr.time()
        res = _lsmr(op, y_aug, atol=atol, btol=btol, maxiter=maxiter, x0=x0)
        _iterative_solver_info(res, "LSMR")
        # scipy returns (x, istop, itn, normr, normar, norma, conda, normx).
        # istop is the whole diagnosis: 1/2 = a stopping test was met, 7 = the
        # iteration cap was hit (i.e. alpha never entered the solve and the cost
        # is independent of alpha), 5 = conda overflowed.
        print("[RIDGE-LSMR] alpha=%.6e istop=%d iters=%d normr=%.3e normar=%.3e "
              "norma=%.3e conda=%.3e warm=%s %.2fs"
              % (float(alpha), int(res[1]), int(res[2]), float(res[3]),
                 float(res[4]), float(res[5]), float(res[6]),
                 "yes" if x0 is not None else "no", _tr.time() - _t_r), flush=True)
        return np.asarray(res[0], dtype=np.float64)
    if _gpu_dense(A):
        return np.asarray(_gpu().ridge_solve(_to_dense_f64(A), y64, alpha), dtype=np.float64)
    if alpha > 0:
        ridge = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
        ridge.fit(A, y64)
        return ridge.coef_
    return _solve_lstsq(A, y64)



def _solve_subset(A, y, row_idx, col_idx, ridge_alpha=0.0, qr=False,
                  lsmr_atol=None, lsmr_btol=None, lsmr_maxiter=None,
                  block_rows=None, diag_floor=1e-12, column_scale=None,
                  diag_sink=None, diag_scope="full"):
    """Solve min ||A[row_idx][:, col_idx] x - y[row_idx]||^2 (+ optional ridge).

    [FIX P09] the lsmr_* / block_rows / diag_floor knobs are threaded through
    from the estimator instead of being silently dropped on the floor.
    diag_sink (a list) collects the LSMR stopping diagnostics: RFE used to
    throw them away, so a subset solve that hit its iteration limit still
    produced a coefficient vector with nothing recording that fact.
    """
    y_sub = np.asarray(y, dtype=np.float64).ravel()
    if row_idx is not None:
        y_sub = y_sub[row_idx]
    if _is_linear_operator(A):
        # LSMR on a masked operator: no materialization, memory ~O(n_features).
        op = _make_masked_op(A, row_idx, col_idx)
        n = len(col_idx)
        scale = np.ones(n, dtype=np.float64)
        if column_scale is not None:
            scale = np.asarray(column_scale, dtype=np.float64)[col_idx]
            if not np.isfinite(scale).all() or np.any(scale < 0):
                raise ValueError("column_scale must be finite and nonnegative")
            scale = np.where(scale < 1e-30, 1.0, scale)
            op = _scale_operator(op, scale)
        atol = float(os.environ.get("PHEASY_LSQR_ATOL", str(
            lsmr_atol if lsmr_atol is not None else 1e-8)))
        btol = float(os.environ.get("PHEASY_LSQR_BTOL", str(
            lsmr_btol if lsmr_btol is not None else 1e-8)))
        maxiter = int(os.environ.get("PHEASY_LSQR_MAXITER", str(
            lsmr_maxiter if lsmr_maxiter is not None else 5000)))
        if ridge_alpha > 0:
            sqrt_a = float(np.sqrt(ridge_alpha)) / scale
            base_op = op  # capture before reassignment below (closure safety)

            def mv_aug(v):
                v = np.asarray(v, dtype=np.float64).ravel()
                return np.concatenate([np.asarray(base_op @ v).ravel(), sqrt_a * v])

            def rmv_aug(u):
                u = np.asarray(u, dtype=np.float64).ravel()
                return (np.asarray(base_op.rmatvec(u[: base_op.shape[0]])).ravel()
                        + sqrt_a * u[base_op.shape[0]:])

            op = LinearOperator((base_op.shape[0] + n, n), matvec=mv_aug,
                                rmatvec=rmv_aug, dtype=np.float64)
            y_sub = np.concatenate([y_sub, np.zeros(n)])
        res = _lsmr(op, y_sub, atol=atol, btol=btol, maxiter=maxiter)
        _subset_info = _iterative_solver_info(res, "LSMR")
        if diag_sink is not None:
            diag_sink.append(dict(_subset_info, fit_scope=diag_scope,
                                  n_features=int(len(col_idx))))
        return np.asarray(res[0], dtype=np.float64) / scale

    A_sub = A[:, col_idx]
    if row_idx is not None:
        A_sub = A_sub[row_idx]
    if ridge_alpha > 0:
        # Dense subset ridge solves use the CUDA backend when the block fits.
        if _gpu_dense(A_sub):
            return np.asarray(_gpu().ridge_solve(_to_dense_f64(A_sub), y_sub, ridge_alpha), dtype=np.float64)
        ridge = Ridge(alpha=ridge_alpha, fit_intercept=False, solver="lsqr")
        ridge.fit(A_sub, y_sub)
        return ridge.coef_
    if _gpu_dense(A_sub):
        # RFE ranks features by |coef|*||col||; QR (gels) is backward-stable for
        # full-rank subsets and ~50x faster than the SVD on the 3090. The SVD
        # fallback inside qr_solve catches rank-deficient subsets.
        return np.asarray(_gpu().qr_solve(_to_dense_f64(A_sub), y_sub), dtype=np.float64)
    if qr:
        # [FIX P45] TSQR only pays off when the dense block does not fit in
        # memory. On a small matrix (the common case) the whole thing is one
        # block, so the tree-reduction pays all of its bookkeeping overhead for
        # zero gain AND parallelizes worse than a single LAPACK gelsd. Judge by
        # BYTES (not rows): the same row count is a very different footprint at
        # p=200 vs p=4000. PHEASY_TSQR_FORCE=1 keeps the QR path for A/B checks.
        _bytes = A_sub.shape[0] * A_sub.shape[1] * 8
        _thr = float(os.environ.get("PHEASY_TSQR_MIN_BYTES", "8e9"))
        if _bytes <= _thr and os.environ.get("PHEASY_TSQR_FORCE", "0") != "1":
            return _solve_lstsq(A_sub, y_sub)
        return _solve_qr(A_sub, y_sub, block_rows=block_rows,
                         diag_floor=diag_floor)
    return _solve_lstsq(A_sub, y_sub)


def _predict_subset(A, col_idx, row_idx, coef):
    """A[row_idx][:, col_idx] @ coef."""
    if _is_linear_operator(A):
        return np.asarray(_make_masked_op(A, row_idx, col_idx) @ coef).ravel()
    A_sub = A[:, col_idx]
    if row_idx is not None:
        A_sub = A_sub[row_idx]
    # CV prediction is also accelerated for dense blocks that fit on CUDA.
    if _gpu_dense(A_sub):
        return np.asarray(_gpu().predict(_to_dense_f64(A_sub), np.asarray(coef, dtype=np.float64)), dtype=np.float64).ravel()
    return np.asarray(A_sub @ coef).ravel()


def _cv_rmse(A, y, idx, solve, splits, n_jobs=1, predict=None, score=None):
    """K-fold CV RMSE (mean, standard error, per-fold) for active columns idx.

    [FIX P45] folds are independent, so parallelize them with THREADS (not the
    default loky processes): each solve slices A[:, col_idx][row_idx] and a
    process worker would copy that block N ways (OOM on many-core hosts);
    threads share A, and the time is spent in LAPACK/scipy where the GIL is
    released anyway.
    """
    y64 = np.asarray(y, dtype=np.float64).ravel()

    def _one(tr, va):
        coef = solve(idx, tr)
        if score is not None:
            return score(idx, va, coef)
        prediction = predict(idx, va, coef) if predict is not None else _predict_subset(A, idx, va, coef)
        err = prediction - y64[va]
        return float(np.sqrt(np.mean(err * err)))

    if n_jobs > 1 and len(splits) > 1:
        from joblib import Parallel, delayed
        with _blas_limit(min(n_jobs, len(splits))):
            fold = np.asarray(Parallel(n_jobs=min(n_jobs, len(splits)),
                                       prefer="threads")(
                delayed(_one)(tr, va) for tr, va in splits), dtype=np.float64)
    else:
        fold = np.asarray([_one(tr, va) for tr, va in splits], dtype=np.float64)

    mean = float(fold.mean())
    se = float(fold.std(ddof=1) / np.sqrt(len(fold))) if len(fold) > 1 else 0.0
    return mean, se, fold


def _select_1se(history):
    """Select (n_active, mean, se) by the one-standard-error rule (or argmin)."""
    use_1se = os.environ.get("PHEASY_RFE_1SE", "1").lower() in ("1", "true", "yes")
    if not history:
        raise ValueError("empty RFE history")
    best_idx = int(np.argmin([h[1] for h in history]))
    if not use_1se:
        return history[best_idx]
    thr = history[best_idx][1] + history[best_idx][2]
    cands = [h for h in history if h[1] <= thr]
    return min(cands, key=lambda h: h[0])


class TwoLevelSM(LinearOperator):
    """Behave like SM = SM_prime @ NS without materializing the product.

    matvec:  SM @ v   = SM_prime @ (NS @ v)
    rmatvec: SM.T @ u = NS.T @ (SM_prime.T @ u)
    """

    def __init__(self, SM_prime, NS, dtype=None):
        self.SM_prime = SM_prime
        self.NS = NS
        dt = dtype if dtype is not None else SM_prime.dtype
        self._dt = dt
        # Lazily-built CSR transposes. scipy's SM_prime.T is a CSC *view* that
        # does scatter-write in matvec; a real CSR transpose is read-sequential
        # and MKL-friendly. PHEASY_TWOLEVEL_CACHE_T=0 disables the cache (CV
        # under tight memory) and reverts to the CSC view.
        # [X1] read the env default BEFORE the GPU branch: when GPU SpMV is
        # active we force the cache off, and a mid-fit GPU failure falls back
        # to _sp_mv on SM_primeT without materializing the 60.5 GB transpose.
        self._cache_T = os.environ.get("PHEASY_TWOLEVEL_CACHE_T", "1").lower() \
            not in ("0", "false", "off")
        self._SMpT = None
        self._NST = None
        # [GPU-SM] optional cuSPARSE SpMV for the SM_prime half (row-split
        # across PHEASY_GPU_SM_NGPU devices). NS stays on the CPU. Falls
        # back to the scipy path silently when unavailable.
        self._gpu_mv = None
        if (os.environ.get("PHEASY_GPU_SM", "0").lower() in ("1", "true", "yes")
                and not _resident_lasso_requested()):
            # A GPU allocation failure must not trigger a second, potentially
            # huge host CSR transpose allocation on the first CPU rmatvec.
            self._cache_T = False
            try:
                try:
                    from . import gpu_backend as _gb
                except (ImportError, ValueError):
                    from pheasy_gpu.core import gpu_backend as _gb
                self._gpu_mv = _gb.GpuSparseMV(SM_prime)
                print("[GPU-SM] SpMV on %d device(s), dtype=%s" % (
                    len(self._gpu_mv._devs), SM_prime.dtype), flush=True)
            except Exception as _e:
                if os.environ.get("PHEASY_GPU_FALLBACK", "0").lower() in ("0", "false", "off", "no"):
                    raise RuntimeError("GPU SM initialization failed with fallback disabled: %s" % _e) from _e
                print("[GPU-SM] SpMV unavailable (%s); using CPU" % _e, flush=True)
                self._disable_gpu()
        super().__init__(np.dtype(dt), (SM_prime.shape[0], NS.shape[1]))

    def _disable_gpu(self):
        gpu_mv, self._gpu_mv = self._gpu_mv, None
        self._cache_T = False
        if gpu_mv is not None:
            try:
                gpu_mv.close()
            except Exception as exc:
                print("[GPU-SM] resource cleanup failed (%s)" % exc, flush=True)

    @property
    def SM_primeT(self):
        if not self._cache_T:
            return self.SM_prime.T
        if self._SMpT is None:
            self._SMpT = self.SM_prime.T.tocsr()
        return self._SMpT

    @property
    def NST(self):
        if not self._cache_T:
            return self.NS.T
        if self._NST is None:
            _nsT = self.NS.T
            self._NST = _nsT.tocsr() if hasattr(_nsT, "tocsr") else _nsT
        return self._NST

    def _matvec(self, v):
        v = np.ascontiguousarray(v, dtype=self._dt)
        t = _sp_mv(self.NS, v)
        if self._gpu_mv is not None:
            try:
                return self._gpu_mv.matvec(t)
            except Exception as _e:
                if _gpu_required() or os.environ.get("PHEASY_GPU_FALLBACK", "0").lower() in ("0", "false", "off", "no"):
                    raise RuntimeError("GPU SM matvec failed with fallback disabled: %s" % _e) from _e
                print("[GPU-SM] matvec failed (%s); disabling GPU, CPU fallback" % _e, flush=True)
                self._disable_gpu()
        return _sp_mv(self.SM_prime, np.ascontiguousarray(t, dtype=self._dt))

    def _rmatvec(self, u):
        u = np.ascontiguousarray(u, dtype=self._dt)
        if self._gpu_mv is not None:
            try:
                t = self._gpu_mv.rmatvec(u)
                return _sp_mv(self.NST, np.ascontiguousarray(t, dtype=self._dt))
            except Exception as _e:
                if _gpu_required() or os.environ.get("PHEASY_GPU_FALLBACK", "0").lower() in ("0", "false", "off", "no"):
                    raise RuntimeError("GPU SM rmatvec failed with fallback disabled: %s" % _e) from _e
                print("[GPU-SM] rmatvec failed (%s); disabling GPU, CPU fallback" % _e, flush=True)
                self._disable_gpu()
        t = _sp_mv(self.SM_primeT, u)
        return _sp_mv(self.NST, np.ascontiguousarray(t, dtype=self._dt))

    def col_norms(self, block_rows=None):
        """Exact column norms from bounded row blocks of SM_prime @ NS.

        The generic LinearOperator implementation needs one full SpMV per
        column. A sparse row-block product computes all columns together and
        discards each block after accumulating its squared entries in float64.
        The default 64 MiB block budget allows 24 bytes per possible output
        entry (float64 values, sparse indices, and the squaring temporary),
        plus the sliced input block. NS is shared when already float64.
        """
        n_rows, n_cols = self.shape
        squares = np.zeros(n_cols, dtype=np.float64)
        if not n_rows or not n_cols:
            return squares
        budget = max(1, int(os.environ.get("PHEASY_COL_NORM_BLOCK_BYTES", "67108864")))
        max_rows = max(1, budget // (24 * n_cols))
        if block_rows is not None:
            max_rows = min(max_rows, max(1, int(block_rows)))
        ns64 = self.NS.astype(np.float64, copy=False)
        prime = self.SM_prime
        i0 = 0
        while i0 < n_rows:
            i1 = min(n_rows, i0 + max_rows)
            if sp.isspmatrix_csr(prime):
                # A very wide SM_prime may dominate the sliced input memory.
                # Account for both its original and float64 working copy.
                while i1 > i0 + 1:
                    input_nnz = int(prime.indptr[i1] - prime.indptr[i0])
                    if 24 * ((i1 - i0) * n_cols + input_nnz) <= budget:
                        break
                    i1 = i0 + max(1, (i1 - i0) // 2)
            product = prime[i0:i1].astype(np.float64, copy=False) @ ns64
            if sp.issparse(product):
                product = product.tocsr()
                product.sum_duplicates()
                squares += np.bincount(product.indices,
                                       weights=np.square(product.data),
                                       minlength=n_cols)
            else:
                product = np.asarray(product, dtype=np.float64)
                squares += np.einsum("ij,ij->j", product, product)
            del product
            i0 = i1
        return np.sqrt(squares)

    def row_slice(self, rows):
        """[FIX P30] TwoLevelSM for A[rows, :] by slicing SM_prime only.

        This is O(nnz(SM_prime[rows])) per matvec instead of the full
        O(nnz(SM_prime)) that the generic _row_slice_op wrapper pays, so a
        K-fold CV costs ~1x the full problem rather than ~Kx.

        NS is unchanged by slicing, so the child shares the parent's cached
        NST (built once) instead of rebuilding a redundant copy per fold.
        SM_primeT stays per-child because each fold slices different rows.
        """
        if self._gpu_mv is not None:
            # CV must not allocate another copy of resident GPU factors.
            # The view keeps the parent alive and trades extra SpMV work for
            # bounded device memory. Normalize slices/masks to explicit rows.
            selected = np.arange(self.shape[0], dtype=np.intp)[rows]
            return _row_slice_op(self, selected)
        child = TwoLevelSM(self.SM_prime[rows], self.NS, dtype=self._dt)
        if self._cache_T:
            self.NST                     # force-build the shared NS transpose
            child._NST = self._NST
        return child

    def to_dense(self):
        return _to_dense_f64(self)



def _reselect_alpha(model, A, y, sample_weight=None, grid_diag=None):
    """[FIX P10] Re-pick alpha from the CV path and refit if it changed.

    Two problems with sklearn's plain ``argmin`` here:

    1. When coordinate descent stops early (a loose ``--tol`` is scaled by
       ``||y||^2``, so 1e-3 is very loose), the low-alpha end of the path
       returns literally the same solution and the CV curve goes flat.  Since
       ``alphas_`` is sorted descending, ``argmin`` then picks the *most*
       regularized member of the tie -- which is how LASSO ends up with force
       constants ~10% too small.  Ties are now broken toward the smallest alpha
       and a warning is printed, because a flat tail means "not converged".
    2. ``PHEASY_LASSO_1SE=1`` optionally applies the one-standard-error rule
       (largest alpha within 1 SE of the best), matching what RFE already does.
    """
    alphas = np.asarray(model.alphas_, dtype=np.float64)
    mse = np.asarray(model.mse_path_, dtype=np.float64)
    if alphas.size < 2 or mse.ndim != 2:
        return
    mean = mse.mean(axis=1)
    best = float(mean.min())
    rtol = float(os.environ.get("PHEASY_LASSO_TIE_RTOL", "1e-9"))
    tied = np.flatnonzero(mean <= best * (1.0 + rtol) + 1e-300)
    if os.environ.get("PHEASY_LASSO_1SE", "0").lower() in ("1", "true", "yes"):
        k = int(np.argmin(mean))
        se = float(mse[k].std(ddof=1) / np.sqrt(mse.shape[1])) if mse.shape[1] > 1 else 0.0
        cand = np.flatnonzero(mean <= best + se)
        new_alpha = float(alphas[cand].max())
    else:
        new_alpha = float(alphas[tied].min())
    # [FIX P43] a flat CV tail is only a CONVERGENCE problem if coordinate
    # descent actually hit its iteration cap; otherwise it converged and the tail
    # is genuinely flat (the alphas simply do not separate).
    # [FIX P43/P44] sklearn's LassoCV.n_iter_ is the FINAL refit's iteration
    # count at the chosen alpha*, NOT the CV-path fits that produced the tail --
    # so it cannot certify that the tail is genuine. Only a hit cap
    # (n_iter >= max_iter) is assertable; otherwise we state the path is not
    # ruled out and ask for a tighter --tol to confirm (the flat tail often
    # disappears once tol is tightened).
    _n_iter = int(np.max(np.atleast_1d(model.n_iter_)))
    _hit_cap = _n_iter >= int(model.max_iter)
    if tied.size > 1:
        if _hit_cap:
            print("[CV] WARNING: %d alphas tie at CV MSE %.6e (%.3e ... %.3e). "
                  "Coordinate descent hit max_iter (%d); lower --tol (1e-6) "
                  "and/or raise --max_iter, otherwise the fit is over-regularized."
                  % (tied.size, best, float(alphas[tied].min()),
                     float(alphas[tied].max()), _n_iter), flush=True)
        else:
            print("[CV] WARNING: %d alphas tie at CV MSE %.6e (%.3e ... %.3e). "
                  "The final refit converged in %d iters, but sklearn does not "
                  "expose the CV-path iteration counts, so a tolerance-limited "
                  "path is not ruled out -- re-run with a tighter --tol to "
                  "confirm." % (tied.size, best, float(alphas[tied].min()),
                                float(alphas[tied].max()), _n_iter), flush=True)
    # [FIX P39/P41/P42/P43] alpha* pinned to the grid MINIMUM has FOUR causes, in
    # priority order: (1) the whole grid is below the weighted KKT threshold -- a
    # GRID-scale problem (manual --no-alpha_auto); (2) a flat tail AND CD hit
    # max_iter -- a CONVERGENCE problem; (3) a flat tail but CD converged -- alpha*
    # is just not well-determined; (4) the curve genuinely still falls -- a model-
    # density conclusion (treat as unregularized / compare OLS). Stash the flags so
    # run_pheasy can defer instead of re-asserting a cause it cannot see.
    model._alpha_at_min = new_alpha <= float(alphas.min()) * (1.0 + 1e-12)
    model._alpha_at_min_flat = (
        model._alpha_at_min and tied.size > 1
        and float(alphas[tied].min()) <= float(alphas.min()) * (1.0 + 1e-12))
    model._alpha_at_min_hitcap = model._alpha_at_min and _hit_cap
    if model._alpha_at_min:
        if grid_diag:
            print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM; %s"
                  % (new_alpha, grid_diag), flush=True)
        elif model._alpha_at_min_hitcap:
            print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM via the "
                  "selected CV candidate, but CD hit max_iter; this is a "
                  "CONVERGENCE problem (lower --tol (1e-6) and/or raise "
                  "--max_iter), not a model-density conclusion." % new_alpha,
                  flush=True)
        elif model._alpha_at_min_flat:
            print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM on a flat CV "
                  "tail; the final refit converged in %d iters, but sklearn does "
                  "not expose the CV-path iteration counts, so a tolerance-limited "
                  "path is not ruled out -- re-run with a tighter --tol to confirm."
                  % (new_alpha, _n_iter), flush=True)
        else:
            print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM; the CV curve "
                  "is still falling at the low end, so widening the grid only pushes "
                  "alpha* toward OLS. Treat this fit as effectively unregularized "
                  "(compare with OLS/RFE)." % new_alpha, flush=True)
    if new_alpha == float(model.alpha_):
        return
    print("[CV] alpha reselected: %.6e -> %.6e" % (float(model.alpha_), new_alpha),
          flush=True)
    from sklearn.linear_model import Lasso as _Lasso
    est = _Lasso(alpha=new_alpha, fit_intercept=model.fit_intercept,
                 max_iter=model.max_iter, tol=model.tol,
                 selection=getattr(model, "selection", "cyclic"),
                 random_state=getattr(model, "random_state", None))
    est.fit(A, y, sample_weight=sample_weight)  # [FIX] keep sample_weight in the refit
    model.alpha_ = new_alpha
    model.coef_ = est.coef_
    model.intercept_ = est.intercept_
    model.n_iter_ = int(np.max(np.atleast_1d(est.n_iter_)))


class _OLSModel:
    def __init__(self, coef, n_iter=None, alpha=None):
        self.coef_ = np.asarray(coef)
        self.intercept_ = 0.0
        self.n_features_in_ = self.coef_.shape[0]
        self.n_iter_ = n_iter
        # [FIX P47] RIDGE stores an _OLSModel (no sklearn model), but
        # run_pheasy reads alpha_ for the grid-edge flag. Expose it here as a
        # defensive mirror of o.results["alpha"] (the formal exit).
        self.alpha_ = alpha

    def predict(self, A):
        return np.asarray(A @ self.coef_).ravel()


def _lasso_n_jobs(A):
    """[FIX P24] Cap sklearn LassoCV process parallelism by matrix size.

    LassoCV(n_jobs=N) fans the (alpha, fold) grid out to N loky *processes*;
    each worker copies the centered training fold of the dense design matrix,
    so N=-1 (one worker per core) on a many-core host blows up memory and the
    OOM killer SIGTERMs the run. Default to a memory-aware, bounded worker count.
    """
    n = _resolve_n_jobs("LASSO")
    if n == 1:
        return 1
    n_cpu = _avail_cores()
    n = min(n, n_cpu)
    try:
        per_worker = int(A.shape[0]) * int(A.shape[1]) * 8
        budget = int(float(os.environ.get("PHEASY_LASSO_MEM_GB", "6"))) * 2 ** 30
        cap = max(1, int(budget // max(per_worker, 1)))
    except Exception:
        cap = 4
    # [FIX P24b] the 16-worker ceiling was hardcoded; it underuses a many-core
    # box. Keep a memory cap but let the ceiling be raised explicitly.
    _hi = int(os.environ.get("PHEASY_LASSO_MAX_WORKERS", "16"))
    return max(1, min(n, cap, _hi))


def _predict_rows(A, coef, rows):
    """A[rows] @ coef for dense / sparse / LinearOperator (one matvec)."""
    return np.asarray(A @ coef, dtype=np.float64).ravel()[rows]


class _LassoCVIterative:
    """LASSO over an alpha grid with grouped CV via the matvec-only FISTA solver.

    [FIX P26] Drop-in for _LassoCVModel when the design matrix is a
    LinearOperator or a sparse matrix too large to densify (see
    _lasso_backend). Never materializes the sensing matrix; peak memory is
    ~O(n_features). The alpha path is walked large->small and warm-started.

    NOTE: this is a memory-for-time trade-off. FISTA converges O(1/k^2) and each
    TwoLevelSM matvec is two sparse multiplies, so it can be 10-100x slower than
    the dense sklearn path. It is auto-selected only when the dense SM does not
    fit in memory (_lasso_backend); PHEASY_LASSO_TWOLEVEL stays off by default.
    """
    def __init__(self, alphas, cv, tol, max_iter, rand_seed, n_jobs=None,
                 fit_intercept=False, group_size=None, selection="cyclic",
                 penalty_weights=None, grid_diag=None):
        # [FIX P40] sort ascending: the alpha walk assumes smallest-first and
        # the grid-MINIMUM edge check (best_i == 0) depends on it. Callers pass
        # logspace/derive grids that are ascending, but be explicit rather than
        # relying on that contract.
        self.alphas = np.sort(np.asarray(alphas, dtype=np.float64))
        self.cv = cv
        self.tol = tol
        self.max_iter = int(max_iter)
        self.rand_seed = rand_seed
        # [FIX P45] thread-fold parallelism, resolved independently of the
        # dense sklearn _lasso_n_jobs memory cap (threads share A, so there is
        # no per-worker matrix copy to budget against).
        self.n_jobs = _resolve_n_jobs("LASSO", n_jobs if n_jobs is not None else -1)
        self.fit_intercept = fit_intercept
        self.group_size = group_size
        self.selection = selection
        self.penalty_weights = penalty_weights
        # [FIX P44] manual-grid scale-mismatch diagnosis (message string from
        # _AdaptiveLassoCV.fit); used to give the GRID, not the data, as the
        # cause of a pinned alpha*.
        self.grid_diag = grid_diag
        self._lipschitz = None

    def fit(self, A, y, sample_weight=None):
        y64 = np.asarray(y, dtype=np.float64).ravel()
        if self.fit_intercept:
            raise NotImplementedError("fit_intercept is not supported by iterative LASSO")
        if sample_weight is not None:
            raise NotImplementedError("sample weights are not supported by iterative LASSO")
        splits = _make_cv_splits(A.shape[0], self.cv, self.rand_seed,
                                 self.group_size)
        n_alphas = len(self.alphas)

        # [FIX P34] build the Gram A^T A (and per-fold variants) when it fits the
        # memory budget, then FISTA runs a dense n_features x n_features matvec
        # per iteration instead of two sparse multiplies over the full SM. On a
        # budget miss (or a build error) fall back to the matvec path.
        use_gram = False
        gram_full = None
        gram_folds = None
        lip_full = None
        lip_folds = None
        A_va_list = None
        if _gram_budget_ok(A):
            try:
                G_full, b_full, _P = _compute_gram(A, y64)
                gram_full = (G_full, b_full)
                lip_full = _top_eigval(G_full)
                gram_folds = []
                lip_folds = []
                # [FIX P34] A_va_list holds the K disjoint VALIDATION slices
                # (~1x SM_prime total, not the (K-1)x of the P33 train-fold
                # cache); it pays for the per-fold prediction A[va] @ coef.
                # G_tr = G_full - G_va: safe for K>=3; for K=2 / LOOCV the
                # cancellation (G_va ~ G_full) can leave G_tr slightly non-PSD
                # (harmless to eigvalsh lambda_max, but noted).
                A_va_list = []
                for _tr, va in splits:
                    A_va = _row_slice(A, va)
                    G_va, b_va, _ = _compute_gram(A_va, y64[va])
                    G_tr = G_full - G_va
                    gram_folds.append((G_tr, b_full - b_va))
                    lip_folds.append(_top_eigval(G_tr))
                    A_va_list.append(A_va)
                use_gram = True
                print("[optimizer] Gram path: G=%dx%d built (PHEASY_GRAM_MAX_GB=%s); "
                      "per-iter cost is a dense matvec, no full-SM multiply."
                      % (G_full.shape[0], G_full.shape[1],
                         os.environ.get("PHEASY_GRAM_MAX_GB", "4")), flush=True)
            except Exception as _e:
                print("[optimizer] WARNING: Gram build failed (%s); falling back "
                      "to matvec FISTA." % _e, flush=True)
                use_gram = False
        if use_gram:
            self._lipschitz = lip_full
            self._gram = gram_full
        else:
            self._lipschitz = _estimate_lipschitz(A)
            self._gram = None

        # CV folds only need to RANK the alphas, so a loose tolerance and a low
        # iteration budget suffice; the final refit uses the caller's tight tol.
        # [FIX P32] the knobs are exposed because a flat CV tail (see the tie
        # warning below) is fixed by tightening these, not by --tol (which only
        # affects the final refit).
        cv_tol = float(os.environ.get(
            "PHEASY_CV_TOL", str(float(self.tol))))
        cv_max_iter = int(os.environ.get(
            "PHEASY_CV_MAX_ITER", str(int(self.max_iter))))

        # [FIX P33] hoist the per-fold row slices out of the alpha loop so the
        # CSR / TwoLevelSM slicing is done once instead of n_alphas times.  Off
        # by default: caching K folds keeps ~(K-1)x SM_prime resident; enable on
        # big-but-not-huge systems via PHEASY_CV_CACHE_FOLDS=1.
        _cache_folds = os.environ.get("PHEASY_CV_CACHE_FOLDS", "0").lower() in ("1", "true", "yes")
        A_folds = None
        if _cache_folds:
            A_folds = [_row_slice(A, tr) if _is_linear_operator(A) else A[tr]
                       for tr, _ in splits]

        mse_path = np.zeros((n_alphas, len(splits)))
        x_folds = [None] * len(splits)   # [FIX P28] per-fold warm-start
        x_full = None                    # full-data warm-start for the alpha path
        best_i = 0
        best_mean = float("inf")
        best_x = None
        # [FIX P43] track the max FISTA iterations used by the CV solves so a flat
        # tail can be attributed to hitting cv_max_iter (convergence) vs a genuinely
        # flat curve (converged).
        _cv_info = {}
        _cv_max_n_iter = 0
        # [D1] "hit the cap" must mean "hit the cap WITHOUT converging" (the
        # periodic FISTA check can let a converged fit report n_iter == cap).
        _cv_hit_cap = False
        _cv_max_kkt = 0.0

        n_jobs = min(self.n_jobs, len(splits))
        for a_i in range(n_alphas - 1, -1, -1):  # descending: large alpha first
            alpha = float(self.alphas[a_i])

            # [FIX P45] the alpha path is a warm-start chain (each fold's
            # solution seeds the next alpha), so only FOLDS -- never alphas --
            # may run in parallel. Each fold reads/writes its own x_folds[k] and
            # its own _info dict, so the workers share no mutable state.
            def _fold_fit(k):
                tr, va = splits[k]
                info = {}
                if use_gram:
                    # [FIX P34] the Gram encodes A[tr], so pass the TRAIN fold's
                    # row count: the L1 threshold is (n_tr * alpha), not
                    # (n_full * alpha) -- otherwise the fold fits at ~(K/(K-1))x
                    # the intended effective alpha.
                    coef = _fista_lasso(A, y64, alpha, x0=x_folds[k],
                                        max_iter=cv_max_iter, tol=cv_tol,
                                        lipschitz=lip_folds[k],
                                        penalty_weights=self.penalty_weights,
                                        gram=gram_folds[k],
                                        n_samples=len(tr), _info=info,
                                        warn_nonconvergence=False)
                    pred = np.asarray(A_va_list[k] @ coef, dtype=np.float64).ravel()
                else:
                    if A_folds is not None:
                        A_tr = A_folds[k]
                    else:
                        A_tr = _row_slice(A, tr) if _is_linear_operator(A) else A[tr]
                    # [FIX P28] warm-start from THIS fold's previous-alpha solution,
                    # not the full-data solution: with a loose CV budget the warm
                    # start does not fully wash out, so x_full would leak the
                    # validation rows into the fold fit and bias CV low.
                    coef = _fista_lasso(A_tr, y64[tr], alpha, x0=x_folds[k],
                                        max_iter=cv_max_iter, tol=cv_tol,
                                        lipschitz=self._lipschitz,
                                        penalty_weights=self.penalty_weights,
                                        _info=info, warn_nonconvergence=False)
                    pred = _predict_rows(A, coef, va)
                err = pred - y64[va]
                return (float(np.mean(err * err)), coef, int(info.get("n_iter", 0)),
                        bool(info.get("converged", False)),
                        float(info.get("kkt_relative", float("nan"))))

            if n_jobs > 1:
                from joblib import Parallel, delayed
                with _blas_limit(n_jobs):
                    results = Parallel(n_jobs=n_jobs, prefer="threads")(
                        delayed(_fold_fit)(k) for k in range(len(splits)))
            else:
                results = [_fold_fit(k) for k in range(len(splits))]
            fold_mse = np.zeros(len(splits))
            for k, (mse_k, coef, n_it, conv_k, kkt_k) in enumerate(results):
                fold_mse[k] = mse_k
                x_folds[k] = coef
                _cv_max_n_iter = max(_cv_max_n_iter, n_it)
                if np.isfinite(kkt_k):
                    _cv_max_kkt = max(_cv_max_kkt, kkt_k)
                if n_it >= cv_max_iter and not conv_k:
                    _cv_hit_cap = True
            mse_path[a_i] = fold_mse
            mean = float(fold_mse.mean())
            # warm-start the next (smaller) alpha from this alpha's full fit
            if use_gram:
                x_full = _fista_lasso(A, y64, alpha, x0=x_full,
                                      max_iter=cv_max_iter, tol=cv_tol,
                                      lipschitz=lip_full,
                                      penalty_weights=self.penalty_weights,
                                      gram=gram_full, _info=_cv_info,
                                      warn_nonconvergence=False)
            else:
                x_full = _fista_lasso(A, y64, alpha, x0=x_full,
                                      max_iter=cv_max_iter, tol=cv_tol,
                                      lipschitz=self._lipschitz,
                                      penalty_weights=self.penalty_weights,
                                      _info=_cv_info, warn_nonconvergence=False)
            # The full-data warm-start solve also reports n_iter; it feeds the
            # "max iterations" message but must NOT set the hit-cap flag: that
            # flag is the CONVERGENCE diagnosis for the CV solves themselves.
            _cv_max_n_iter = max(_cv_max_n_iter, _cv_info.get("n_iter", 0))
            # [FIX P27] <= (not <) so a tie picks the SMALLEST alpha (the one
            # seen LAST in the descending walk), matching _reselect_alpha's
            # tie-break toward the least-regularized member.
            if mean <= best_mean:
                best_mean = mean
                best_i = a_i
                best_x = x_full.copy()

        # [CV-budget] ONE aggregated line instead of a warning per fold: the CV
        # folds only RANK the alphas, so a fold that stops at the cap is not an
        # uncertified fit -- but the reader does need to know the ranking was
        # computed at a loose tolerance (PHEASY_CV_TOL) or a small cap.
        if _cv_hit_cap:
            print("[CV] ranking budget: a fold hit cv_max_iter (%d) with relative "
                  "KKT up to %.3e > PHEASY_CV_TOL=%.1e. CV only ranks alphas "
                  "(differences between alphas are far larger than this gap), so "
                  "alpha* is not affected by itself; raise PHEASY_CV_TOL toward "
                  "1e-3 (the GPU backends' default) or raise PHEASY_CV_MAX_ITER "
                  "to certify the ranking." % (cv_max_iter, _cv_max_kkt, cv_tol),
                  flush=True)
        # [FIX P27] port _reselect_alpha's tie warning: a flat CV tail means the
        # loose CV solver did not separate the alphas and the choice is suspect.
        mean_path = mse_path.mean(axis=1)
        # PHEASY_LASSO_1SE: one-standard-error rule (largest alpha within 1 SE of
        # the CV minimum), matching _reselect_alpha and GpuLassoCV.  This class is
        # the backend auto-selected for TwoLevelSM / LinearOperator / large sparse
        # input, and it used to ignore the knob entirely (only _reselect_alpha,
        # the sklearn path, read it) -- so the documented "all backends" rule was
        # a silent no-op on the production path.
        if os.environ.get("PHEASY_LASSO_1SE", "0").lower() in ("1", "true", "yes"):
            _se = (float(mse_path[best_i].std(ddof=1) / np.sqrt(mse_path.shape[1]))
                   if mse_path.shape[1] > 1 else 0.0)
            _cand = np.flatnonzero(mean_path <= mean_path[best_i] + _se)
            _new_i = int(_cand[np.argmax(self.alphas[_cand])])
            if _new_i != best_i:
                print("[CV] PHEASY_LASSO_1SE: alpha* %.6e -> %.6e (1 SE = %.3e above "
                      "the CV minimum)" % (float(self.alphas[best_i]),
                                           float(self.alphas[_new_i]), _se), flush=True)
            best_i = _new_i
            best_x = None          # the new alpha needs its own warm start
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
        # [FIX P39/P41/P42/P43/P44] best_i == 0 is the SMALLEST alpha (the grid
        # walks descending). Four causes, in priority order: (1) manual-grid scale
        # mismatch -- a GRID-scale problem; (2) flat tail AND FISTA hit cv_max_iter
        # -- CONVERGENCE; (3) flat tail but FISTA converged -- alpha* not
        # well-determined; (4) the curve still falls -- model-density (treat as
        # unregularized / compare OLS). NOTE: unlike the sklearn path (whose n_iter_
        # is the final refit count and cannot certify convergence), _cv_max_n_iter
        # HERE measures the actual CV-path FISTA iterations, so the 'FISTA already
        # converged' branch is a real assertion.
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
        # final refit at the chosen alpha, warm-started from the path. Cap the
        # iteration budget (FISTA is O(1/k^2), and the scaled ALASSO matrix is
        # more ill-conditioned); 5000 warm-started steps already reach ~1e-5.
        _finfo = {"n_iter": 0}
        self.coef_ = _fista_lasso(A, y64, self.alpha_, x0=best_x,
                                  max_iter=self.max_iter,
                                  tol=float(self.tol),
                                  lipschitz=lip_full if use_gram else self._lipschitz,
                                  penalty_weights=self.penalty_weights,
                                  _info=_finfo,
                                  gram=gram_full if use_gram else None)
        self.intercept_ = 0.0
        self.alphas_ = self.alphas
        self.mse_path_ = mse_path
        self.n_iter_ = int(_finfo.get("n_iter", 0))
        # Name the backend explicitly: the iterative LASSO is the CPU FISTA
        # solver unless PHEASY_GPU_SM shards the SM_prime matvec onto CUDA.  The
        # key used to be absent, so Optimizer._fit_impl never set
        # results["execution_backend"] for this path and a CPU solve was
        # indistinguishable from a GPU one in the result record.
        self.regularized_solver_info_ = dict(
            _finfo, solver="FISTA",
            stage="regularized_refit_before_debias", tol=float(self.tol),
            backend=("gpu_sm_spmv" if _gpu_sm_explicit() else "cpu_iterative_fista"))
        self.n_features_in_ = A.shape[1]
        return self

    def predict(self, A):
        pred = np.asarray(A @ self.coef_).ravel()
        if self.fit_intercept:
            pred = pred + self.intercept_
        return pred

class _LassoCVModel:
    """Thin wrapper around sklearn LassoCV with correct alpha grid and grouped CV."""

    def __init__(self, alphas, cv, tol, max_iter, rand_seed, n_jobs,
                 fit_intercept=False, group_size=None, selection="cyclic"):
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.cv = cv
        self.tol = tol
        self.max_iter = max_iter
        self.rand_seed = rand_seed
        self.n_jobs = n_jobs
        self.fit_intercept = fit_intercept
        self.group_size = group_size
        self.selection = selection

    def fit(self, A, y, sample_weight=None):
        if _lasso_backend(A) == "gpu_resident":
            from . import gpu_backend
            it = gpu_backend.GpuTwoLevelLassoCV(
                self.alphas, self.cv, self.tol, self.max_iter, self.rand_seed,
                fit_intercept=self.fit_intercept, group_size=self.group_size)
            it.fit(A, y, sample_weight=sample_weight)
            self.model_ = it
            for name in ("coef_", "intercept_", "alpha_", "alphas_", "mse_path_",
                         "n_iter_", "regularized_solver_info_", "n_features_in_",
                         "_alpha_at_min", "_alpha_at_min_flat", "_alpha_at_min_hitcap"):
                setattr(self, name, getattr(it, name))
            return self
        if _lasso_backend(A) == "iterative":
            it = _LassoCVIterative(
                self.alphas, self.cv, self.tol, self.max_iter, self.rand_seed,
                None, fit_intercept=self.fit_intercept,
                group_size=self.group_size, selection=self.selection)
            it.fit(A, y, sample_weight=sample_weight)
            self.model_ = it
            self.coef_ = it.coef_
            self.intercept_ = it.intercept_
            self.alpha_ = it.alpha_
            self.alphas_ = it.alphas_
            self.mse_path_ = it.mse_path_
            self.n_iter_ = it.n_iter_
            self.regularized_solver_info_ = dict(it.regularized_solver_info_)
            self.n_features_in_ = it.n_features_in_
            self._alpha_at_min = getattr(it, "_alpha_at_min", False)
            self._alpha_at_min_flat = getattr(it, "_alpha_at_min_flat", False)
            self._alpha_at_min_hitcap = getattr(it, "_alpha_at_min_hitcap", False)
            return self

        if _lasso_backend(A) == "gpu":
            gb = _gpu()
            it = gb.GpuLassoCV(
                self.alphas, self.cv, self.tol, self.max_iter, self.rand_seed,
                self.n_jobs, fit_intercept=self.fit_intercept,
                group_size=self.group_size, selection=self.selection)
            it.fit(_to_dense_f64(A), y, sample_weight=sample_weight)
            self.model_ = it
            self.coef_ = it.coef_
            self.intercept_ = it.intercept_
            self.alpha_ = it.alpha_
            self.alphas_ = it.alphas_
            self.mse_path_ = it.mse_path_
            self.n_iter_ = it.n_iter_
            self.regularized_solver_info_ = dict(it.regularized_solver_info_)
            self.n_features_in_ = it.n_features_in_
            self._alpha_at_min = getattr(it, "_alpha_at_min", False)
            self._alpha_at_min_flat = getattr(it, "_alpha_at_min_flat", False)
            self._alpha_at_min_hitcap = getattr(it, "_alpha_at_min_hitcap", False)
            return self

        n_samples = A.shape[0]
        splits = _make_cv_splits(n_samples, self.cv, self.rand_seed, self.group_size)
        model = LassoCV(
            alphas=self.alphas,
            cv=splits,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
            random_state=self.rand_seed,
            selection=self.selection,
            n_jobs=self.n_jobs,
        )
        model.fit(A, y, sample_weight=sample_weight)
        self.model_ = model
        _reselect_alpha(model, A, y, sample_weight=sample_weight)  # [FIX P10]
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        self.alpha_ = model.alpha_
        self.alphas_ = np.asarray(model.alphas_)
        self.mse_path_ = np.asarray(model.mse_path_)
        self.n_iter_ = int(model.n_iter_)
        self.n_features_in_ = A.shape[1]
        self._alpha_at_min = getattr(model, "_alpha_at_min", False)
        self._alpha_at_min_flat = getattr(model, "_alpha_at_min_flat", False)
        self._alpha_at_min_hitcap = getattr(model, "_alpha_at_min_hitcap", False)
        return self

    def predict(self, A):
        pred = np.asarray(A @ self.coef_).ravel()
        if self.fit_intercept:
            pred = pred + self.intercept_
        return pred


class _AdaptiveLassoCV(_LassoCVModel):
    """Adaptive LASSO (Zou 2006).

    Stage 1: ridge initial estimate beta0.
    Stage 2: weights w_j = 1/(|beta0_j| + eps)^gamma, then column-scaled LASSO.
    """

    def __init__(self, *args, gamma=1.0, init_alpha=1e-3, eps=1e-8,
                 nalpha=None, decades=4.0, alpha_auto=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = float(gamma)
        self.init_alpha = float(init_alpha)
        self.eps = float(eps)
        self.nalpha = int(nalpha) if nalpha else None
        self.decades = float(decades)
        # [FIX P38] the weighted-space grid is an AUTO-mode behavior: it must
        # not clobber a user-supplied manual --mu_min/--mu_max grid. run_pheasy
        # only derives the LASSO grid when --alpha_auto is on; ALASSO must
        # respect the same flag instead of unconditionally overriding.
        self.alpha_auto = bool(alpha_auto)
        self._weights = None

    def _initial_estimate(self, A, y):
        # [FIX P26] _ridge_solve handles dense / sparse / LinearOperator, so the
        # adaptive weights are available on the two-level operator too (LSMR on
        # the augmented system for operators, cholesky/svd for dense).
        return _ridge_solve(A, y, self.init_alpha)

    def fit(self, A, y, sample_weight=None):
        if _resident_lasso_requested():
            raise NotImplementedError("Resident GPU ALASSO is dispatched by Optimizer for TwoLevelSM input")
        if sample_weight is not None:
            raise NotImplementedError("ALASSO sample weights are not supported end-to-end: the adaptive pilot is unweighted")
        n_samples = A.shape[0]
        beta0 = self._initial_estimate(A, y)
        self._weights = 1.0 / (np.abs(beta0) + self.eps) ** self.gamma
        # [FIX] eps-floor fraction: the fraction of pilot coefficients at/below
        # eps. Underdetermined ridge pilots can either FLATTEN (weights ~uniform)
        # or SATURATE at the 1/eps ceiling -- two opposite failure modes that a
        # single weight-dispersion number cannot separate. 0.0 = flattened,
        # ~1.0 = most weights pinned at the 1/eps ceiling.
        self._beta0_floor = float(np.mean(np.abs(beta0) < self.eps))

        # [FIX P35] derive the alpha grid in the WEIGHTED space:
        # (A/w)^T y = (A^T y)/w, so alpha_max = max_j |(A^T y)_j / w_j| / n.
        # This replaces the unweighted grid + hardcoded mu_shift=-2 and is the
        # right grid for BOTH the scaled (LassoCV on A/w) and the penalized
        # (FISTA with per-coordinate penalty w_j) ALASSO forms.
        _wgrid = os.environ.get("PHEASY_ALASSO_WEIGHTED_GRID", "1").lower() in ("1", "true", "yes")
        # [FIX P38/P39/P40] three grid modes, so the logging and the matvec count
        # are each honest:
        #   weighted-auto: _wgrid & alpha_auto      -> derive the weighted KKT grid
        #                  here (one rmatvec).
        #   mu_shift-auto: (not _wgrid) & alpha_auto -> run_pheasy already derived a
        #                  mu_shift grid (one rmatvec THERE); do NOT redo it here.
        #   manual:        not alpha_auto           -> user --mu_min/--mu_max grid,
        #                  used AS-IS (diagnostic rmatvec only).
        _override = bool(self.nalpha and _wgrid and self.alpha_auto)
        _manual = bool(self.nalpha and not self.alpha_auto)
        _mu_shift = bool(self.nalpha and self.alpha_auto and not _wgrid)
        _a_uw = _a_max = 0.0
        _grid_diag = None   # [FIX P44] manual-grid scale-mismatch diagnosis string
        if _override:
            # [FIX P39/P40] one rmatvec for BOTH the weighted KKT threshold and the
            # unweighted one, kept in A's dtype so a float32 sensing matrix is not
            # silently promoted to float64 (which doubles peak memory). NOTE: a
            # float32 matvec accumulates in float32 (~1e-5 relative at n~1e5 rows);
            # acceptable because these thresholds only set grid ENDPOINTS.
            _A_dt = A.dtype if hasattr(A, "dtype") else np.float64
            _g_raw = np.abs(np.asarray(
                A.T @ np.asarray(y, dtype=np.float64).ravel().astype(_A_dt, copy=False),
                dtype=np.float64)).ravel()
            _a_uw = float(_g_raw.max()) / n_samples
            _g = _g_raw / np.maximum(self._weights, 1e-300)
            _a_max = float(_g.max()) / n_samples
        elif _manual and self.alphas.size > 1:
            # [FIX P40] manual grid diagnostic: one rmatvec (guarded by
            # PHEASY_ALASSO_GRID_DIAG) to quantify how far the UNWEIGHTED scale is
            # from the weighted KKT threshold. Default ON here because this is the
            # one non-auto path most likely to be mis-scaled.
            _diag = os.environ.get("PHEASY_ALASSO_GRID_DIAG", "1").lower() in ("1", "true", "yes")
            if _diag:
                _A_dt = A.dtype if hasattr(A, "dtype") else np.float64
                _g_raw = np.abs(np.asarray(
                    A.T @ np.asarray(y, dtype=np.float64).ravel().astype(_A_dt, copy=False),
                    dtype=np.float64)).ravel()
                _a_max = float((_g_raw / np.maximum(self._weights, 1e-300)).max()) / n_samples

        if _override:
            if _a_max > 0 and np.isfinite(_a_max) and _a_uw > 0 and np.isfinite(_a_uw):
                # [FIX P37] the weighted grid must also reach the UNDER-regularized
                # regime. The 4-decade span below the weighted KKT threshold clips
                # the CV optimum whenever the true model is dense (measured on
                # MnIn2Se4 c3=7.0 n=45: nnz 546 / rel_err 6.5% with the 4-decade
                # span vs nnz 3317 / rel_err 0.55% once the bottom reaches the
                # unweighted-threshold scale; the CV optimum for a dense model sits
                # far below the weighted threshold). Anchor the top at the weighted
                # KKT threshold and the bottom at 10^-max(decades, 6) below the
                # MIN of the two thresholds.
                # Only extend for overdetermined problems (n_rows > n_cols): an
                # underdetermined system genuinely needs regularization, its CV
                # optimum is interior, and the extra low-alpha tail just makes
                # coordinate descent crawl (MnIn2Se4 c3=7.0 n=4: 11s -> 649s for a
                # result within 5% of the 4-decade one).
                if n_samples > A.shape[1]:
                    _hi = _a_max
                    _lo = min(_a_max, _a_uw) * 10.0 ** -max(self.decades, 6.0)
                else:
                    _hi = _a_max
                    _lo = _a_max * 10.0 ** -self.decades
                if _lo > 0 and np.isfinite(_lo) and _hi > _lo:
                    # [FIX P38/P39/P40] density is anchored to the user's per-decade
                    # count (PHEASY_ALPHA_PER_DECADE, default (nmu-1)/4 matching the
                    # historical 4-decade grid), NOT to --alpha_decades: decades
                    # controls the SPAN while density stays fixed, so widening the
                    # grid (to chase a low alpha*) no longer coarsens the step.
                    _span = np.log10(_hi / _lo)
                    _per_dec = float(os.environ.get(
                        "PHEASY_ALPHA_PER_DECADE", str((self.nalpha - 1) / 4.0)))
                    _n = 1 + int(np.ceil(_span * _per_dec))
                    _n = max(_n, self.nalpha)
                    # [FIX P40] hard safety cap (PHEASY_ALPHA_NMAX); warn if it
                    # clamps so --nmu is not silently ignored.
                    _nmax = int(os.environ.get("PHEASY_ALPHA_NMAX", "200"))
                    if _n > _nmax:
                        print("[ALASSO] grid density capped at %d alphas "
                              "(PHEASY_ALPHA_NMAX; computed %d)."
                              % (_nmax, _n), flush=True)
                        _n = _nmax
                    self.alphas = np.logspace(np.log10(_lo), np.log10(_hi), _n)
                    print("[ALASSO] weighted-space alpha grid: [%.3e .. %.3e] "
                          "(%d alphas, %.2f decades, step %.2fx)"
                          % (_lo, _hi, _n, _span,
                             10.0 ** (_span / max(_n - 1, 1))), flush=True)
        elif _manual and self.alphas.size > 1:
            # [FIX P40/P44/P45] manual grid (--no-alpha_auto): used AS-IS, but
            # ALWAYS report the weighted scale so even a well-scaled manual grid
            # leaves an auditable record, and diagnose the two real mismatch modes:
            # (a) the ENTIRE grid is below the weighted KKT threshold (no point
            # regularizes); (b) the grid BOTTOM is far above where the auto grid
            # starts (never reaches the under-regularized regime). Both pin alpha*
            # to the bottom; whether the fit is actually over-regularized depends
            # on the data (the [1e-2,1e2] case can recover the true support).
            _lo_m = float(self.alphas.min())
            _hi_m = float(self.alphas.max())
            if _a_max > 0:
                _a_uw = float(_g_raw.max()) / n_samples
                _auto_lo = min(_a_max, _a_uw) * 10.0 ** -max(self.decades, 6.0)
                if _hi_m < _a_max:
                    _grid_diag = ("the ENTIRE grid lies below the weighted KKT "
                                  "threshold (%.3e): no grid point regularizes at "
                                  "all -- the GRID, not the data, pins alpha*. Drop "
                                  "--no-alpha_auto and use the weighted auto grid."
                                  % _a_max)
                elif _lo_m > _auto_lo * 10.0:
                    _grid_diag = ("the grid BOTTOM (%.3e) is %.1f decades ABOVE "
                                  "where the auto grid starts (%.3e): alpha* is "
                                  "pinned by the grid bottom rather than chosen by "
                                  "CV; the fit MAY be over-regularized (MnIn2Se4 "
                                  "c3=5.2 n45 on this path: nnz 375 vs 1224 on the "
                                  "auto grid). Drop --no-alpha_auto and use the "
                                  "weighted auto grid."
                                  % (_lo_m, np.log10(_lo_m / _auto_lo), _auto_lo))
                else:
                    _grid_diag = None
                print("[ALASSO] manual alpha grid [%.3e .. %.3e] used AS-IS; "
                      "weighted KKT threshold %.3e, auto grid would start at "
                      "%.3e.%s" % (_lo_m, _hi_m, _a_max, _auto_lo,
                                   (" " + _grid_diag) if _grid_diag else ""),
                      flush=True)
            else:
                print("[ALASSO] manual alpha grid [%.3e .. %.3e] used AS-IS "
                      "(scale diagnostic off: PHEASY_ALASSO_GRID_DIAG=0)."
                      % (_lo_m, _hi_m), flush=True)
        elif _mu_shift and self.alphas.size > 1:
            # [FIX P40] mu_shift-auto grid (PHEASY_ALASSO_WEIGHTED_GRID=0 with
            # alpha_auto): run_pheasy already did the rmatvec to derive it, so no
            # matvec here -- a terse note that this is the fallback path.
            print("[ALASSO] mu_shift-auto alpha grid [%.3e .. %.3e] (fallback "
                  "path, PHEASY_ALASSO_WEIGHTED_GRID=0)."
                  % (float(self.alphas.min()), float(self.alphas.max())), flush=True)

        if _lasso_backend(A) == "iterative":
            # [FIX P26] penalized form: pass per-coordinate weights w_j and
            # fit on the ORIGINAL columns (no column-scaling), so the FISTA
            # Lipschitz stays ||A||^2 and convergence is as fast as plain
            # LASSO. Column-scaling by 1/w would inflate the Lipschitz to
            # ||A / w||^2 and make FISTA crawl on the ill-conditioned matrix.
            it = _LassoCVIterative(
                self.alphas, self.cv, self.tol, self.max_iter, self.rand_seed,
                None, fit_intercept=self.fit_intercept,
                group_size=self.group_size, selection=self.selection,
                penalty_weights=self._weights, grid_diag=_grid_diag)
            it.fit(A, y, sample_weight=sample_weight)
            self.model_ = it
            self.coef_ = it.coef_
            self.intercept_ = it.intercept_ if self.fit_intercept else 0.0
            self.alpha_ = it.alpha_
            self.alphas_ = it.alphas_
            self.mse_path_ = it.mse_path_
            self.n_iter_ = it.n_iter_
            self.regularized_solver_info_ = dict(it.regularized_solver_info_)
            self.n_features_in_ = A.shape[1]
            self._alpha_at_min = getattr(it, "_alpha_at_min", False)
            self._alpha_at_min_flat = getattr(it, "_alpha_at_min_flat", False)
            self._alpha_at_min_hitcap = getattr(it, "_alpha_at_min_hitcap", False)
            return self

        if _lasso_backend(A) == "gpu":
            gb = _gpu()
            it = gb.GpuLassoCV(
                self.alphas, self.cv, self.tol, self.max_iter, self.rand_seed,
                self.n_jobs, fit_intercept=self.fit_intercept,
                group_size=self.group_size, selection=self.selection,
                penalty_weights=self._weights, grid_diag=_grid_diag)
            it.fit(_to_dense_f64(A), y, sample_weight=sample_weight)
            self.model_ = it
            self.coef_ = it.coef_
            self.intercept_ = it.intercept_ if self.fit_intercept else 0.0
            self.alpha_ = it.alpha_
            self.alphas_ = it.alphas_
            self.mse_path_ = it.mse_path_
            self.n_iter_ = it.n_iter_
            self.regularized_solver_info_ = dict(it.regularized_solver_info_)
            self.n_features_in_ = it.n_features_in_
            self._alpha_at_min = getattr(it, "_alpha_at_min", False)
            self._alpha_at_min_flat = getattr(it, "_alpha_at_min_flat", False)
            self._alpha_at_min_hitcap = getattr(it, "_alpha_at_min_hitcap", False)
            return self

        A_scaled = _scale_columns(A, self._weights)
        splits = _make_cv_splits(n_samples, self.cv, self.rand_seed, self.group_size)
        model = LassoCV(
            alphas=self.alphas,
            cv=splits,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
            random_state=self.rand_seed,
            selection=self.selection,
            n_jobs=self.n_jobs,
        )
        model.fit(A_scaled, y, sample_weight=sample_weight)
        _reselect_alpha(model, A_scaled, y, sample_weight=sample_weight,
                       grid_diag=_grid_diag)  # [FIX P23/P43/P44]
        self.model_ = model
        self.coef_ = model.coef_ / self._weights
        self.intercept_ = model.intercept_ if self.fit_intercept else 0.0
        self.alpha_ = model.alpha_
        self.alphas_ = np.asarray(model.alphas_)
        self.mse_path_ = np.asarray(model.mse_path_)
        self.n_iter_ = int(model.n_iter_)
        self.n_features_in_ = A.shape[1]
        self._alpha_at_min = getattr(model, "_alpha_at_min", False)
        self._alpha_at_min_flat = getattr(model, "_alpha_at_min_flat", False)
        self._alpha_at_min_hitcap = getattr(model, "_alpha_at_min_hitcap", False)
        return self

    def predict(self, A):
        pred = np.asarray(A @ self.coef_).ravel()
        if self.fit_intercept:
            pred = pred + self.intercept_
        return pred


def _scale_columns(A, w):
    """Column-scaled copy of A: A[:, j] / w[j] (dense or sparse).

    [FIX P26] A LinearOperator (e.g. TwoLevelSM) is wrapped by _scale_operator
    instead of being materialized, so LASSO / ALASSO can run the matvec-only
    FISTA solver and keep the two-level memory optimization (sklearn's
    coordinate descent cannot consume an operator, so _LassoCVModel routes
    it to the iterative backend).
    """
    inv_w = 1.0 / np.asarray(w, dtype=np.float64)
    if sp.issparse(A):
        return A.astype(np.float64).multiply(inv_w[None, :]).tocsr()
    if _is_linear_operator(A):
        # [FIX P26] wrap instead of materializing, so LASSO/ALASSO run the
        # matvec-only FISTA solver and keep the two-level memory optimization.
        return _scale_operator(A, w)
    A64 = _to_dense_f64(A)
    # A[:, j] * (1/w[j]) == A[:, j] / w[j]
    return A64 * inv_w[None, :]


class _RFECVBase:
    """Recursive feature elimination with cross-validated feature count.

    Uses scale-invariant importance |coef_j| * ||A[:, j]|| so that 2nd and 3rd
    order force-constant columns (different physical units) are ranked by their
    actual contribution to the fit.
    """

    def __init__(self, step=0.1, cv=5, min_features=1, n_jobs=None,
                 verbose=False, random_state=None, solver="lstsq", ridge_alpha=0.0,
                 patience=5, lsmr_maxiter=5000, lsmr_atol=1e-8, lsmr_btol=1e-8,
                 block_rows=None, diag_floor=1e-12):
        self.step = float(step)
        self.cv = int(cv)
        self.min_features = int(min_features)
        self.n_jobs = _resolve_n_jobs("RFE", n_jobs if n_jobs is not None else -1)
        self.verbose = verbose
        self.random_state = random_state
        self.ridge_alpha = float(ridge_alpha)
        self._solver_name = solver
        # [FIX P09] previously accepted-and-ignored constructor arguments
        self.patience = int(patience)
        self.lsmr_maxiter = int(lsmr_maxiter)
        self.lsmr_atol = float(lsmr_atol)
        self.lsmr_btol = float(lsmr_btol)
        self.block_rows = None if block_rows is None else int(block_rows)
        self.diag_floor = float(diag_floor)
        # [FIX P35] feature-count selection criterion. Default "cv" (CV + 1-SE).
        # RFE-OLS-TSQR overrides this from PHEASY_TSQR_CRITERION=bic|cv so it
        # becomes a genuinely independent method (BIC) instead of a numerically
        # identical twin of RFE.
        self._criterion = "cv"

    def _cv_group_size(self, n_samples):
        raw = os.environ.get("PHEASY_CV_GROUP_SIZE")
        gs = int(raw or 0)
        if gs > 1 and n_samples % gs == 0:
            return gs
        if not raw:
            global _WARNED_UNGROUPED_CV
            if not _WARNED_UNGROUPED_CV:
                _WARNED_UNGROUPED_CV = True
                warnings.warn(
                    "PHEASY_CV_GROUP_SIZE is not set: the cross-validation is "
                    "ungrouped, so the 3*natom rows of one configuration can land "
                    "in both folds.  That leaks information, biases the selected "
                    "alpha/ridge toward 0 and inflates the reported CV score.  Set "
                    "PHEASY_CV_GROUP_SIZE=3*natom (the pheasy CLI now does this "
                    "automatically when the variable is unset).",
                    RuntimeWarning, stacklevel=2)
        return None

    def _bic_n_eff(self, n_samples):
        """[FIX P35] effective independent observations for BIC.

        Force components in one configuration are highly correlated (3*natoms
        rows per config) -- the same reason the CV is grouped.  BIC's n should
        therefore be the CONFIGURATION count, not the raw row count, or the
        k*ln(n) penalty is off by ln(3*natoms) and the fit term is inflated by
        3*natoms. PHEASY_BIC_N_EFF=samples falls back to raw rows.
        """
        gs = self._cv_group_size(n_samples)
        if (os.environ.get("PHEASY_BIC_N_EFF", "groups").lower() != "samples"
                and gs and gs > 1 and n_samples % gs == 0):
            return n_samples // gs, gs
        return n_samples, gs

    def fit(self, A, y, sample_weight=None):
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = A.shape
        self.n_features_in_ = n_features

        col_norms = _col_norms(A)
        # Reuse training norms as a right preconditioner, not as a change to
        # feature ranking or ridge objective. Independent test data is not used;
        # CV folds share these training-pool norms as a numerical preconditioner.
        use_scaling = os.environ.get("PHEASY_RFE_JACOBI", "0").lower() in ("1", "true", "yes")
        _qr = self._solver_name == "qr"

        splits = _make_cv_splits(n_samples, self.cv, self.random_state,
                                 self._cv_group_size(n_samples))
        index_cache_bytes = 8 * (sum(len(tr) + len(va) for tr, va in splits) + n_features)
        gpu_subset_solves = 0
        resident_A = resident_y = resident_backend = None
        resident_reason = None
        resident_operator = False
        iterative_diagnostics = []
        resident_subset_builds = 0
        if os.environ.get("PHEASY_GPU_RFE_RESIDENT", "1" if _resident_default() else "0").lower() in ("1", "true", "yes", "on"):
            if isinstance(A, np.ndarray) and self.n_jobs == 1:
                resident_backend = _gpu()
                if resident_backend is not None:
                    import torch
                    fraction = float(os.environ.get("PHEASY_GPU_MEM_FRACTION", "0.8"))
                    if not 0 < fraction <= 1:
                        raise ValueError("PHEASY_GPU_MEM_FRACTION must be in (0, 1]")
                    budget = resident_backend.available_memory_bytes() * fraction
                    needed = 8 * (9 * A.size + 16 * n_features**2 + 4 * n_samples) + 64 * 1024**2 + index_cache_bytes
                    if needed <= budget:
                        try:
                            resident_A = resident_backend._to_torch(A, torch.float64)
                            resident_y = resident_backend._to_torch(y, torch.float64)
                        except (RuntimeError, MemoryError) as exc:
                            resident_A = resident_y = None
                            resident_reason = "resident RFE upload failed: %s: %s" % (type(exc).__name__, exc)
                    else:
                        resident_reason = "resident RFE workspace exceeds memory budget"
                else:
                    resident_reason = "CUDA disabled or unavailable"
            elif self.n_jobs == 1 and (sp.issparse(A) or _resident_twolevel_input(A)):
                resident_backend = _gpu()
                if resident_backend is None:
                    resident_reason = "CUDA disabled or unavailable"
                else:
                    import torch
                    try:
                        adapter = (resident_backend.GpuCSRResidentOperator if sp.issparse(A)
                                   else resident_backend.GpuTwoLevelOperator)
                        resident_A = adapter(
                            A, extra_workspace_bytes=index_cache_bytes,
                            **({"device_ids": resident_backend.resident_device_ids()}
                               if adapter is not resident_backend.GpuCSRResidentOperator
                               else {}))
                        # Follow the FACTOR dtype, not a literal.  This adapter keeps
                        # the operator's own precision (float32 when PHEASY_SM_DTYPE
                        # is float32, the production setting), so a hard-coded
                        # float64 target turns the setup probe below into
                        # torch.sparse.mm(float32_csr, float64_dense) and cuSPARSE
                        # rejects it:
                        #   matA (CUDA_R_32F) and matB (CUDA_R_64F) with different
                        #   value types is not supported
                        # That is why resident RFE never started at any scale.  Same
                        # omission as GpuSubsetOperator._value_dtype (7190139) and the
                        # ridge Augmented wrapper (694e1fd).
                        _vd = getattr(resident_A, "_value_dtype", None) or torch.float64
                        resident_y = resident_backend._to_torch(y, _vd)
                        # Probe sparse kernels during setup, before any elimination.
                        resident_A.rmatvec(resident_A.matvec(resident_y.new_zeros(n_features)))
                        resident_operator = True
                    except (RuntimeError, MemoryError, ValueError, TypeError, NotImplementedError) as exc:
                        if resident_A is not None:
                            resident_A.close()
                        resident_A = resident_y = None
                        resident_reason = "resident RFE setup failed: %s: %s" % (type(exc).__name__, exc)
            else:
                resident_reason = "resident RFE requires dense, scipy sparse or TwoLevel input and n_jobs=1"

        # A resident RFE that cannot be set up must fail loudly under GPU-required
        # mode rather than quietly produce cpu_rfe coefficients.  The old guard
        # deliberately excluded the "resident RFE requires ..." reasons -- n_jobs=1
        # being the common one -- so a misconfigured run reported
        # execution_backend="cpu_rfe" with no diagnostic anywhere, and the only way
        # to notice was to read the solver field in the results dict.
        _rfe_resident_requested = (
            os.environ.get("PHEASY_GPU_RFE_RESIDENT", "").lower()
            in ("1", "true", "yes", "on") or _gpu_required())
        if resident_reason is not None and _is_linear_operator(A) and _rfe_resident_requested:
            raise RuntimeError(
                "GPU RFE resident solve failed with fallback disabled: %s "
                "(set PHEASY_GPU_RFE_RESIDENT=0 to allow the CPU solver)"
                % resident_reason)

        full_fit_coef = None
        resident_norms = None
        gpu_importance_rounds = 0
        row_index_cache = {}
        cached_columns = cached_column_tensor = cached_subset = None
        column_index_uploads = 0

        def resident_columns(cols):
            nonlocal cached_columns, cached_column_tensor, cached_subset, column_index_uploads
            if cached_columns is None or not np.array_equal(cached_columns, cols):
                # Invalidate before allocation; never expose a new key with old data.
                cached_columns = cached_column_tensor = cached_subset = None
                new_columns = np.array(cols, copy=True)
                new_tensor = torch.as_tensor(cols, dtype=torch.long, device=resident_A.device)
                new_subset = new_tensor if resident_operator else resident_A.index_select(1, new_tensor)
                cached_columns, cached_column_tensor, cached_subset = new_columns, new_tensor, new_subset
                column_index_uploads += 1
            return cached_subset

        def resident_rows(rows):
            # Cache owns the original array too, preventing Python id reuse.
            key = id(rows)
            if key not in row_index_cache:
                row_index_cache[key] = (rows, torch.as_tensor(rows, dtype=torch.long, device=resident_A.device))
            return row_index_cache[key][1]

        def resident_column_norms():
            nonlocal resident_norms
            if resident_norms is None:
                resident_norms = torch.as_tensor(col_norms, dtype=_resident_value_dtype(resident_A, torch), device=resident_A.device)
            return resident_norms.index_select(0, cached_column_tensor)

        def solve(col_idx, row_idx=None, download=True):
            nonlocal gpu_subset_solves, full_fit_coef, resident_subset_builds
            if resident_operator:
                columns = resident_columns(col_idx)
                rows = None if row_idx is None else resident_rows(row_idx)
                coef, info = resident_backend.solve_resident_subset(
                    resident_A, resident_y, columns, rows,
                    column_scale=resident_column_norms() if use_scaling and _is_linear_operator(A) else None,
                    ridge_alpha=self.ridge_alpha,
                    atol=float(os.environ.get("PHEASY_LSQR_ATOL", str(self.lsmr_atol))),
                    btol=float(os.environ.get("PHEASY_LSQR_BTOL", str(self.lsmr_btol))),
                    maxiter=int(os.environ.get("PHEASY_LSQR_MAXITER", str(self.lsmr_maxiter))))
                iterative_diagnostics.append(dict(info, n_features=len(col_idx),
                                                 n_samples=n_samples if row_idx is None else len(row_idx),
                                                 fit_scope="full" if row_idx is None else "fold"))
                resident_subset_builds += 1
                gpu_subset_solves += 1
                if row_idx is not None:
                    return coef
                full_fit_coef = coef
                return resident_backend._to_numpy(coef, np.float64) if download else None
            if resident_A is not None:
                subset = resident_columns(col_idx)
                target = resident_y
                if row_idx is not None:
                    rows = resident_rows(row_idx)
                    subset = subset.index_select(0, rows)
                    target = target.index_select(0, rows)
                if self.ridge_alpha > 0:
                    coef = resident_backend._ridge_solve_tensor(subset, target, self.ridge_alpha)
                else:
                    coef = resident_backend._qr_solve_tensor(subset, target)
                gpu_subset_solves += 1
                # Training-fold coefficients feed CUDA scoring directly.
                if row_idx is not None:
                    return coef
                full_fit_coef = coef
                return resident_backend._to_numpy(coef, np.float64) if download else None
            if not _is_linear_operator(A):
                _probe = A[:, col_idx]
                if row_idx is not None:
                    _probe = _probe[row_idx]
                if _gpu_dense(_probe):
                    gpu_subset_solves += 1
            return _solve_subset(A, y, row_idx, col_idx,
                                 self.ridge_alpha, _qr,
                                 lsmr_atol=self.lsmr_atol,
                                 lsmr_btol=self.lsmr_btol,
                                 lsmr_maxiter=self.lsmr_maxiter,
                                 block_rows=self.block_rows,
                                 diag_floor=self.diag_floor,
                                 column_scale=col_norms if use_scaling else None,
                                 diag_sink=iterative_diagnostics,
                                 diag_scope="full" if row_idx is None else "fold")

        def operator_prediction(cols, rows, coef):
            nonlocal resident_subset_builds
            view = resident_backend.GpuSubsetOperator(
                resident_A, resident_columns(cols), None if rows is None else resident_rows(rows))
            resident_subset_builds += 1
            return view.matvec(torch.as_tensor(coef, dtype=_resident_value_dtype(resident_A, torch), device=resident_A.device))

        def predict_subset(cols, rows, coef):
            if resident_operator:
                return resident_backend._to_numpy(operator_prediction(cols, rows, coef), np.float64)
            if resident_A is None:
                return _predict_subset(A, cols, rows, coef)
            subset = resident_columns(cols)
            if rows is not None:
                ri = resident_rows(rows)
                subset = subset.index_select(0, ri)
            ct = torch.as_tensor(coef, dtype=_resident_value_dtype(resident_A, torch), device=resident_A.device)
            return resident_backend._to_numpy(subset @ ct, np.float64)

        def residual_squares(cols, rows, coef):
            if resident_operator:
                target = resident_y if rows is None else resident_y.index_select(0, resident_rows(rows))
                return (operator_prediction(cols, rows, coef) - target).square()
            subset = resident_columns(cols)
            target = resident_y
            if rows is not None:
                ri = resident_rows(rows)
                subset = subset.index_select(0, ri)
                target = target.index_select(0, ri)
            ct = torch.as_tensor(coef, dtype=_resident_value_dtype(resident_A, torch), device=resident_A.device)
            return (subset @ ct - target).square()

        def score_subset(cols, rows, coef):
            return float(residual_squares(cols, rows, coef).mean().sqrt().item())


        # [FIX P09] constructor value is the default; env var is an override
        patience = int(os.environ.get("PHEASY_RFE_PATIENCE", str(self.patience)))
        criterion = getattr(self, "_criterion", "cv")
        gpu_ranking_rounds = 0
        active = np.ones(n_features, dtype=bool)
        history = []      # (n_active, cv_mean, cv_se, support)
        history_bic = []  # [FIX P35] (n_active, bic, support) for TSQR BIC

        if self.verbose:
            print(f"[RFE] START n_features={n_features}, step={self.step:.2f}, "
                  f"cv={self.cv}, min_features={self.min_features}, "
                  f"patience={patience}, "
                  f"solver={self._solver_name}, ridge_alpha={self.ridge_alpha:.2e}",
                  flush=True)

        round_num = 0
        best_cv = float("inf")
        best_bic = float("inf")
        no_improve = 0
        while True:
            idx = np.where(active)[0]
            n_active = len(idx)
            # Keep the initial no-elimination fast path, but evaluate the
            # minimum support reached by elimination before selecting a model.
            if n_active <= self.min_features and round_num == 0:
                break

            defer_download = (resident_A is not None
                              and os.environ.get("PHEASY_GPU_RFE_RANKING", "0").lower() in ("1", "true", "yes", "on"))
            coef_active = solve(idx, download=not defer_download)
            if self.verbose:
                nonzero_count = (int(torch.count_nonzero(full_fit_coef).item()) if coef_active is None
                                 else int(np.count_nonzero(coef_active)))
            if criterion in ("bic", "aic"):
                # [FIX P35] IC mode: no CV needed for stopping (skips the K-fold
                # sub-solves -> ~5x faster); CV is computed once at the end for
                # the report only. Loop driven by the criterion's own patience.
                if resident_A is not None:
                    _rss = float(residual_squares(idx, None, full_fit_coef).sum().item())
                else:
                    _pred = predict_subset(idx, None, coef_active)
                    _rss = float(np.sum((_pred - y) ** 2))
                _rss = max(_rss, np.finfo(float).tiny * n_samples)
                _n_eff, _gs = self._bic_n_eff(n_samples)
                # dimension-consistent: fit term and penalty share n_eff
                # [FIX P36] k+1 counts sigma^2 as a parameter (constant offset;
                # argmin over k is unchanged but the reported absolute value is
                # the textbook AIC/BIC).
                _loglik = _n_eff * np.log(_rss / _n_eff)
                _pen = ((n_active + 1) * np.log(_n_eff) if criterion == "bic"
                        else 2 * (n_active + 1))
                _ic = _loglik + _pen
                history_bic.append((n_active, _ic, active.copy()))
                cv_mean, cv_se = 0.0, 0.0
                if self.verbose:
                    print(f"[RFE] Round {round_num:3d}: n_active={n_active:5d}  "
                          f"{criterion.upper()}={_ic:.2e} (n_eff={_n_eff})  "
                          f"nonzero={nonzero_count}", flush=True)
                if _ic < best_bic:
                    best_bic = _ic
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                cv_mean, cv_se, _ = _cv_rmse(A, y, idx, solve, splits,
                                                n_jobs=self.n_jobs, predict=predict_subset,
                                                score=score_subset if resident_A is not None else None)
                history.append((n_active, cv_mean, cv_se, active.copy()))
                if self.verbose:
                    print(f"[RFE] Round {round_num:3d}: n_active={n_active:5d}  "
                          f"CV_RMSE={cv_mean:.6e} (+-{cv_se:.2e})  "
                          f"nonzero={nonzero_count}", flush=True)
                if cv_mean < best_cv:
                    best_cv = cv_mean
                    no_improve = 0
                else:
                    no_improve += 1
            if no_improve >= patience:
                if self.verbose:
                    print(f"[RFE] {criterion.upper()} 连续 {patience} 轮无改善, 提前停止.",
                          flush=True)
                break

            if n_active <= self.min_features:
                break

            imp = None
            n_remove = max(1, int(round(n_active * self.step)))
            n_remove = min(n_remove, n_active - self.min_features)
            rank_backend = _gpu() if os.environ.get("PHEASY_GPU_RFE_RANKING", "0").lower() in ("1", "true", "yes", "on") else None
            if rank_backend is not None:
                import torch
                if resident_A is not None:
                    resident_columns(idx)
                    norms = resident_column_norms()
                    importance = full_fit_coef.abs() * norms
                    gpu_importance_rounds += 1
                else:
                    if coef_active is None:
                        coef_active = resident_backend._to_numpy(full_fit_coef, np.float64)
                    imp = np.abs(coef_active) * col_norms[idx]
                    importance = torch.as_tensor(imp, dtype=torch.float64, device=rank_backend.device())
                ordered, order = torch.sort(importance)
                # NumPy quicksort tie order is not stable. Preserve it exactly
                # on tied/nonfinite inputs rather than silently changing support.
                unambiguous = torch.isfinite(ordered).all() & ~torch.any(ordered[1:] == ordered[:-1])
                if bool(unambiguous.item()):
                    remove_local = order[:n_remove].cpu().numpy()
                    gpu_ranking_rounds += 1
                else:
                    if coef_active is None:
                        coef_active = resident_backend._to_numpy(full_fit_coef, np.float64)
                    imp = np.abs(coef_active) * col_norms[idx]
                    remove_local = np.argsort(imp)[:n_remove]
            else:
                if coef_active is None:
                    coef_active = resident_backend._to_numpy(full_fit_coef, np.float64)
                imp = np.abs(coef_active) * col_norms[idx]
                remove_local = np.argsort(imp)[:n_remove]
            active[idx[remove_local]] = False
            round_num += 1

        # --- selection (each criterion prints its own summary, once) ---
        if criterion in ("bic", "aic") and history_bic:
            best_round = int(np.argmin([h[1] for h in history_bic]))
            n_best = history_bic[best_round][0]
            best_support = history_bic[best_round][2]
            # CV once for the selected round (report only)
            _sel = np.where(best_support)[0]
            best_mean, best_se, _ = _cv_rmse(A, y, _sel, solve, splits,
                                                n_jobs=self.n_jobs, predict=predict_subset,
                                                score=score_subset if resident_A is not None else None)
            if self.verbose:
                print("[RFE] %s min: n_active=%d, %s=%.2e" % (criterion.upper(), n_best, criterion.upper(), history_bic[best_round][1]), flush=True)
                print("[RFE] selected: n_active=%d, CV_RMSE=%.6e (+-%.2e)" % (n_best, best_mean, best_se), flush=True)
        elif history:
            n_best, best_mean, best_se, best_support = _select_1se(history)
            if self.verbose:
                argmin = history[int(np.argmin([h[1] for h in history]))]
                print(f"[RFE] argmin: n_active={argmin[0]}, CV={argmin[1]:.6e}", flush=True)
                print(f"[RFE] selected: n_active={n_best}, CV={best_mean:.6e} (+-{best_se:.2e})", flush=True)
        else:
            # min_features >= n_features: nothing was eliminated, keep all.
            n_best, best_mean, best_se = n_features, 0.0, 0.0
            best_support = np.ones(n_features, dtype=bool)
        best_idx = np.where(best_support)[0]

        coef_final = solve(best_idx)
        coef_full = np.zeros(n_features, dtype=np.float64)
        coef_full[best_idx] = coef_final

        self.coef_ = coef_full
        self.intercept_ = 0.0
        self.support_ = best_support
        self.n_iter_ = round_num
        # A run that broke out before the first CV evaluation has NO CV score.
        # It used to publish best_mean = 0.0, i.e. a fabricated PERFECT CV RMSE
        # ("- RMSE_CV: 0.0 eV/A" in the log, rmse_path_mean = 0.0 in the metrics)
        # for a fit in which no cross-validation ever ran -- reachable whenever
        # n_features <= min_features (RFE-OLS-TSQR defaults to min_features=100).
        # Report it as not-evaluated instead of as a perfect score.
        self.cv_evaluated = bool(round_num > 0)
        self.best_rmse_cv_ = float(best_mean) if self.cv_evaluated else float("nan")
        self.ridge_alpha = self.ridge_alpha
        self.alphas_ = np.array([self.ridge_alpha])
        self.mse_path_ = np.array([[self.best_rmse_cv_ ** 2]])
        self.backend_metadata_ = {
            "subset_solver": "gpu_resident_iterative" if resident_operator else ("gpu_dense" if gpu_subset_solves else "cpu"),
            "iterative_diagnostics": iterative_diagnostics,
            "resident_input_kind": ("csr" if sp.issparse(A) else "twolevel") if resident_operator else ("dense" if resident_A is not None else None),
            "gpu_subset_solves": int(gpu_subset_solves),
            "jacobi_applied": bool(use_scaling and _is_linear_operator(A)),
            "resident_subset_inputs": resident_A is not None,
            "resident_index_cache_budget_bytes": index_cache_bytes,
            "resident_row_index_uploads": len(row_index_cache),
            "resident_column_index_uploads": column_index_uploads,
            "resident_subset_builds": resident_subset_builds if resident_operator else column_index_uploads,
            "cv_fold_scoring": "gpu" if resident_A is not None else "cpu",
            "resident_fallback_reason": resident_reason,
            "gpu_prediction": bool(resident_A is not None or ((not _is_linear_operator(A)) and _gpu_dense(A[:, best_idx]))),
            "gpu_importance_rounds": gpu_importance_rounds,
            "gpu_ranking_rounds": gpu_ranking_rounds,
            "ranking": "gpu_with_cpu_tie_fallback" if gpu_ranking_rounds else "cpu",
            "orchestration": "cpu",
            "postprocessing": "cpu",
        }
        return self

    def predict(self, A):
        return np.asarray(A @ self.coef_).ravel()


class PheasyRFECV(_RFECVBase):
    """RFE with an OLS (optionally ridge-regularized) base estimator."""

    def __init__(self, step=0.05, cv=5, ridge_alpha=0.0, lsmr_maxiter=3000,   # [FIX P21]
                 lsmr_atol=1e-8, lsmr_btol=1e-8, n_jobs=None, min_features=1,
                 verbose=True, random_state=None, patience=5):
        # [FIX P09] lsmr_* used to be dropped here
        super().__init__(step=step, cv=cv, min_features=min_features, n_jobs=n_jobs,
                         verbose=verbose, random_state=random_state,
                         solver="lstsq", ridge_alpha=ridge_alpha,
                         patience=patience, lsmr_maxiter=lsmr_maxiter,
                         lsmr_atol=lsmr_atol, lsmr_btol=lsmr_btol)


class PheasyRFE_OLS_TSQR(_RFECVBase):
    """RFE with a strict OLS base estimator solved by Q-less tall-skinny QR.

    [FIX P09] ``patience``, ``block_rows`` and ``diag_floor`` are now honoured:
    the base solve streams the factorization block by block (see
    ``_tsqr_qless``) instead of running a plain dense QR on the whole matrix.
    The former ``recalibrate`` argument is gone: it never had an effect, and
    the base estimator re-solves exactly every round, so there is nothing to
    recalibrate.
    """

    def __init__(self, step=0.05, patience=5, min_features=100, block_rows=40000,
                 diag_floor=1e-12, cv=5, verbose=True,
                 random_state=None, n_jobs=None):
        super().__init__(step=step, cv=cv, min_features=min_features, n_jobs=n_jobs,
                         verbose=verbose, random_state=random_state,
                         solver="qr", ridge_alpha=0.0, patience=patience,
                         block_rows=block_rows, diag_floor=diag_floor)
        # [FIX P35] BIC/AIC are genuinely independent stopping rules (the
        # CV+1-SE path makes RFE and RFE-OLS-TSQR numerically identical).
        # NOTE: with the grouped-CV n_eff (configuration count, often tens),
        # BIC's k*ln(n_eff) penalty is heavy and picks over-sparse models
        # (e.g. MnIn2Se4 n_eff=45 -> 21 features, CV_RMSE 78x worse than CV's
        # 1238). CV remains the recommended default; BIC/AIC are sensible only
        # when n_eff is large (hundreds+).
        _crit = os.environ.get("PHEASY_TSQR_CRITERION", "cv").lower()
        self._criterion = _crit if _crit in ("bic", "aic") else "cv"


# backward-compatible aliases
CelerLassoCV = _LassoCVModel
CelerALassoCV = _AdaptiveLassoCV
_LsmrOLSResult = _OLSModel


class Optimizer(object):
    """Interatomic force constant optimizer.

    Supported methods: ols, lasso, alasso, rfe, rfe-ols-tsqr (rfe_tsqr) and
    the legacy ridge.
    """

    def __init__(
        self,
        method="ols",
        nalpha=100,
        alpha_min=-6,
        alpha_max=-2,
        alpha=None,
        cv=5,
        tol=1e-4,
        max_iter=20000,
        rand_seed=None,
        standardize=False,
        fit_intercept=False,
        alpha_auto=True,
        decades=4.0,
        use_gpu=None,
    ):
        self._method = method
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._nalpha = nalpha
        self._cv = cv
        self._tol = tol
        self._max_iter = max_iter
        self._rand_seed = rand_seed
        self._standardize = standardize
        self._fit_intercept = fit_intercept
        self._alpha_auto = bool(alpha_auto)
        # An explicit alpha= grid is a request to fit AT those alphas.  The
        # ALASSO auto-grid used to overwrite it silently (the guard only covered
        # the alpha_auto=False spelling), so the returned coefficients solved a
        # different regularization problem than the one asked for.
        self._alpha_user_supplied = alpha is not None
        self._decades = float(decades)

        if alpha is not None:
            self._alpha = np.asarray(alpha, dtype=np.float64)
        else:
            # alpha_min/alpha_max are POWERS OF 10 (exponents), matching the
            # pheasy CLI (--mu_min/--alpha_min, default -6/-2):
            #   alphas = 10^alpha_min ... 10^alpha_max
            self._alpha = np.logspace(alpha_min, alpha_max, nalpha)

        # Store the override on the instance. The process-global mode is only
        # touched inside fit() (and restored afterwards), so constructing an
        # Optimizer never clobbers a gb.set_gpu_mode(False) the caller set
        # earlier, and "build N optimizers, then fit each" keeps each
        # instance's own choice.
        self._use_gpu = use_gpu

        self._group_size = None
        self._results = {}
        self._metrics = {}

    def _ols_lsmr(self, X, y, atol=1e-8, btol=1e-8, maxiter=5000):
        # X is the operator in hand, so the floor reflects the precision its data
        # actually carries -- not the PHEASY_SM_DTYPE proxy, which production can
        # leave unset while SM_prime is still float32.
        atol = float(_lsmr_tol("PHEASY_OLS_ATOL", atol, X))
        btol = float(_lsmr_tol("PHEASY_OLS_BTOL", btol, X))
        maxiter = int(os.environ.get("PHEASY_OLS_MAXITER", str(maxiter)))
        ridge = float(os.environ.get("PHEASY_OLS_RIDGE", "0"))
        # Jacobi (exact column scaling) is a pure change of variables for OLS: it
        # cannot change the solution, only whether LSMR converges.  It used to be
        # opt-in, which is how a 113794-unknown fit burned 16 minutes and was then
        # REFUSED by the acceptance gate at istop=7 (measured column-norm spread
        # on that system: 94x).  Default it ON for matrix-free input -- a
        # LinearOperator / TwoLevelSM means the sensing matrix is never
        # materialised, i.e. the regime where a stalled LSMR is expensive -- and
        # keep PHEASY_OLS_JACOBI=0/1 as the explicit override.  Decided here, at
        # the top, because the resident-GPU branch below must skip itself when
        # Jacobi is wanted (it implements neither ridge nor Jacobi).
        _jacobi_env = os.environ.get("PHEASY_OLS_JACOBI")
        if _jacobi_env is None:
            _use_jacobi = _is_linear_operator(X)
            if _use_jacobi:
                print("[OLS] Jacobi preconditioning ENABLED by default for matrix-free "
                      "input (exact change of variables; PHEASY_OLS_JACOBI=0 disables)",
                      flush=True)
        else:
            _use_jacobi = _jacobi_env.lower() in ("1", "true", "yes")
        # Resident (device-side) OLS: single card, or SHARDED across the devices
        # named by PHEASY_GPU_DEVICES / PHEASY_GPU_NGPU / PHEASY_GPU_SM_DEVICES.
        # Explicitly selecting a device list counts as a request for it.
        _res_env = os.environ.get("PHEASY_GPU_OLS_RESIDENT")
        if _res_env is None:
            _want_resident = bool(_resident_default()) or bool(
                os.environ.get("PHEASY_GPU_DEVICES", "").strip()
                or os.environ.get("PHEASY_GPU_NGPU", "").strip()
                or os.environ.get("PHEASY_GPU_SM_DEVICES", "").strip())
        else:
            _want_resident = _res_env.lower() in ("1", "true", "yes", "on")
        if _want_resident and (hasattr(X, "SM_prime") or hasattr(X, "_twolevel_base")):
            try:
                from . import gpu_backend as _gb
                jacobi = _use_jacobi
                # Jacobi is now implemented ON the resident operator rather than
                # used as a reason to skip it.  Jacobi is an exact change of
                # variables: solve (A D) z = y with D = diag(1/||A[:,j]||), then
                # x = D z.  GpuTwoLevelOperator.matvec already computes
                # A (v / scale) and rmatvec computes (A^T u) / scale, i.e. it IS
                # the scaled operator once scale holds the column norms -- so the
                # GPU CGLS runs on the same well-conditioned system the CPU path
                # builds, and only the back-transform x = z / scale is added here.
                # Skipping instead (the old behaviour) left the resident OLS
                # unreachable under the default configuration, because Jacobi
                # defaults ON for matrix-free input.
                if ridge > 0:
                    self._ols_gpu_fallback_reason = "Resident OLS does not implement the ridge option; preserving CPU semantics"
                if _gb.enabled() and ridge <= 0:
                    _dev_ids = _gb.resident_device_ids()
                    base = getattr(X, "_twolevel_base", X)
                    gpu_op = _gb.GpuTwoLevelOperator(base, device_ids=_dev_ids)
                    _coln = None
                    try:
                        if jacobi:
                            _coln = np.asarray(X.col_norms(), dtype=np.float64).ravel()
                            if _coln.shape != (base.shape[1],) or not np.all(np.isfinite(_coln)) or np.any(_coln <= 0):
                                raise ValueError("Jacobi column norms must be finite, positive and complete")
                            gpu_op.scale = gpu_op.torch.as_tensor(
                                _coln, dtype=gpu_op._value_dtype, device=gpu_op.device)
                        coef, info = _gb.iterative_lstsq(gpu_op, y, atol=atol, btol=btol, maxiter=maxiter)
                    finally:
                        gpu_op.close()
                    if _coln is not None:
                        coef = np.asarray(coef, dtype=np.float64).ravel() / _coln
                    if isinstance(info, dict):
                        info = dict(info, jacobi_applied=bool(jacobi))
                    self._ols_lsmr_info = info
                    return coef
            except Exception as exc:
                self._ols_gpu_fallback_reason = "%s: %s" % (type(exc).__name__, exc)
                if _gpu_required():
                    raise RuntimeError("GPU OLS solve failed with fallback disabled: %s" % exc) from exc
        """OLS via LSMR (iterative; sparse and LinearOperator safe).

        [P2] PHEASY_OLS_JACOBI=1 applies a Jacobi (column-scaling)
        preconditioner: solve (A D) z = y with D = diag(1/||A[:,j]||), then
        x = D z. Cuts the iteration count on ill-conditioned columns (Si
        617,818x col-span: 27 -> 13 iters, same residual). Cost: one col-norm
        pass via X.col_norms() -- TwoLevelSM accumulates exact norms from
        bounded sparse row-block products. Off by default.
        """

        n_samples = X.shape[0]
        damp = float(np.sqrt(ridge * n_samples)) if ridge > 0 else 0.0
        y_in = np.asarray(y, dtype=np.float64).ravel()
        if _use_jacobi:
            print("[OLS] Computing exact column norms for Jacobi scaling", flush=True)
            cn = (X.col_norms() if hasattr(X, "col_norms") else _col_norms(X))
            cn = np.where(np.asarray(cn, dtype=np.float64) < 1e-30, 1.0, cn)
            print("[OLS] Column norms ready: min=%.6g max=%.6g; starting LSMR"
                  % (cn.min(), cn.max()), flush=True)
            X_s = _scale_operator(X, cn)
            if damp > 0:
                # x = z / cn: the ridge penalty must remain damp*||x||,
                # hence the augmented block is damp*diag(1/cn), not I.
                penalty = damp / cn

                def mv_ridge(v):
                    v = np.asarray(v, dtype=np.float64).ravel()
                    return np.concatenate([np.asarray(X_s @ v).ravel(),
                                           penalty * v])

                def rmv_ridge(u):
                    u = np.asarray(u, dtype=np.float64).ravel()
                    return (np.asarray(X_s.T @ u[:n_samples]).ravel()
                            + penalty * u[n_samples:])

                aug = LinearOperator((n_samples + X.shape[1], X.shape[1]),
                                     matvec=mv_ridge, rmatvec=rmv_ridge,
                                     dtype=np.float64)
                y_aug = np.concatenate([y_in, np.zeros(X.shape[1])])
                result = _lsmr(aug, y_aug, atol=atol, btol=btol,
                               maxiter=maxiter)
            else:
                result = _lsmr(X_s, y_in, atol=atol, btol=btol,
                               maxiter=maxiter)
            coef = np.asarray(result[0], dtype=np.float64) / cn
            self._ols_lsmr_info = _iterative_solver_info(result, "LSMR")
            return coef
        result = _lsmr(X, y_in, damp=damp, atol=atol, btol=btol, maxiter=maxiter)
        coef = np.asarray(result[0], dtype=np.float64)
        self._ols_lsmr_info = _iterative_solver_info(result, "LSMR")
        return coef

    def fit(self, A, F, weights=None):
        # Apply the per-instance GPU override only for the duration of this fit,
        # then restore the process-global mode. Constructing must NOT touch the
        # global mode (a gb.set_gpu_mode(False) set by the caller survives an
        # Optimizer(use_gpu=None) construction), and a batch of optimizers built
        # first then fit one-by-one each sees its own override.
        _gb_mod = None
        _prev_mode = None
        _have_prev = False
        try:
            from . import gpu_backend as _gb_mod
            _prev_mode = _gb_mod.get_gpu_mode()
            _have_prev = True
        except Exception:
            pass
        if _gb_mod is not None and self._use_gpu is not None:
            _gb_mod.set_gpu_mode(bool(self._use_gpu))
        try:
            return self._fit_impl(A, F, weights)
        finally:
            if _gb_mod is not None and _have_prev:
                _gb_mod.set_gpu_mode(_prev_mode)

    def _fit_impl(self, A, F, weights=None):
        self._results.pop("regularized_solver_info", None)
        self._results.pop("execution_backend", None)
        self._results.pop("postfit_backend", None)
        self._results.pop("backend_metadata", None)
        self._results.pop("fallback_reason", None)
        method = self._method.upper().replace("_", "-")
        if method in ("RFE-OLS-TSQR", "RFE-TSQR"):
            method = "RFE-OLS-TSQR"
        elif method == "RFECV":
            method = "RFE"

        F = np.asarray(F)
        if F.ndim == 2:
            F = F.ravel()
        F64 = np.asarray(F, dtype=np.float64).ravel()

        if _is_linear_operator(A) and method in ("RIDGE", "LASSO", "ALASSO"):
            if self._fit_intercept:
                raise NotImplementedError("fit_intercept is not supported for operator penalized fits; use a supported dense path or fit_intercept=False")
            if weights is not None:
                raise NotImplementedError("sample weights are not supported for operator penalized fits; weights must be None")

        if weights is not None and method == "LASSO" and self._debias_enabled():
            raise NotImplementedError("weighted LASSO debias is not supported; set PHEASY_LASSO_DEBIAS=0 for supported dense weighted fitting")

        resident_lasso = _resident_lasso_active(A)
        if (_resident_lasso_requested() and not _resident_twolevel_input(A)
                and method in ("LASSO", "ALASSO")):
            raise NotImplementedError("Resident GPU LASSO/ALASSO requires TwoLevelSM input")
        if resident_lasso and method in ("LASSO", "ALASSO"):
            from . import gpu_backend as resident_gb
            if self._use_gpu is False or not resident_gb.enabled() or not resident_gb.available():
                raise RuntimeError("Resident two-level LASSO requires enabled CUDA; no CPU fallback")

        self._group_size = self._detect_group_size(A.shape[0])

        # Column standardization (unit L2 norm) for the scale-sensitive
        # penalized methods; coefficients are un-scaled after fitting.
        col_scale = None
        A_fit = A
        if self._standardize and method in ("LASSO", "ALASSO", "RIDGE") and not (resident_lasso and method in ("LASSO", "ALASSO")):
            col_scale = _col_norms(A)
            col_scale = np.where(col_scale < 1e-30, 1.0, col_scale)
            A_fit = _scale_columns(A, col_scale)
        elif self._standardize:
            # Do not drop a requested knob silently: standardization is a MODEL
            # choice for the penalized methods (it makes the L1/Ridge penalty
            # fair per column, and alpha lives in standardized units), but it is
            # only a reparametrization for OLS/RFE -- the OLS solution is
            # invariant to column scaling, so the solve keeps the raw columns and
            # uses PHEASY_OLS_JACOBI for conditioning instead.
            print("[note] --std has no effect for %s: column standardization is a "
                  "model choice only for LASSO/ALASSO/RIDGE. The %s solve is "
                  "invariant to column scaling; its preconditioner is "
                  "PHEASY_OLS_JACOBI (default on for matrix-free input)."
                  % (method, method), flush=True)

        if method == "OLS":
            coef, n_iter = self._fit_ols(A, F64)
            self._model = _OLSModel(coef, n_iter=n_iter)
        elif method in ("LASSO", "ALASSO") and resident_lasso:
            self._model = resident_gb.GpuTwoLevelLassoCV(
                self._alpha, self._cv, self._tol, self._max_iter, self._rand_seed,
                group_size=self._group_size, standardize=self._standardize,
                adaptive=(method == "ALASSO"),
                gamma=float(os.environ.get("PHEASY_ALASSO_GAMMA", "1.0")),
                init_alpha=float(os.environ.get("PHEASY_ALASSO_RIDGE_ALPHA", "1e-3")),
                eps=float(os.environ.get("PHEASY_ALASSO_EPS", "1e-8")),
                nalpha=self._nalpha, decades=self._decades,
                alpha_auto=self._alpha_auto and not self._alpha_user_supplied)
            self._model.fit(A, F64, sample_weight=weights,
                            retain_operator=self._debias_enabled())
            coef = self._model.coef_
            # Backend returns physical coefficients. Keep A_fit unscaled so
            # the existing optional CPU debias operates in physical coordinates.
            self._results["execution_backend"] = "gpu_twolevel_resident"
            # postfit_backend is set after the optional debias (which now runs on the
            # retained resident operator); do not pre-declare it here.
            print("[optimizer] gpu_resident %s complete" % method, flush=True)
        elif method == "LASSO":
            self._model = _LassoCVModel(
                self._alpha, self._cv, self._tol, self._max_iter, self._rand_seed,
                _lasso_n_jobs(A),
                fit_intercept=self._fit_intercept, group_size=self._group_size)
            self._model.fit(A_fit, F64, sample_weight=weights)
            coef = self._model.coef_
        elif method == "ALASSO":
            self._model = _AdaptiveLassoCV(
                self._alpha, self._cv, self._tol, self._max_iter, self._rand_seed,
                _lasso_n_jobs(A),
                fit_intercept=self._fit_intercept, group_size=self._group_size,
                gamma=float(os.environ.get("PHEASY_ALASSO_GAMMA", "1.0")),
                init_alpha=float(os.environ.get("PHEASY_ALASSO_RIDGE_ALPHA", "1e-3")),
                eps=float(os.environ.get("PHEASY_ALASSO_EPS", "1e-8")),
                nalpha=self._nalpha,
                decades=self._decades,
                alpha_auto=self._alpha_auto and not self._alpha_user_supplied)
            self._model.fit(A_fit, F64, sample_weight=weights)
            coef = self._model.coef_
        elif method == "RFE":
            self._model = PheasyRFECV(
                step=float(os.environ.get("PHEASY_RFE_STEP", "0.05")),  # [FIX P21]
                cv=self._cv,
                ridge_alpha=float(os.environ.get("PHEASY_RFE_RIDGE_ALPHA", "0")),
                n_jobs=None,  # resolved via PHEASY_RFE_N_JOBS / PHEASY_N_JOBS
                min_features=int(os.environ.get("PHEASY_RFE_MIN_FEATURES", "1")),
                patience=int(os.environ.get("PHEASY_RFE_PATIENCE", "5")),
                lsmr_maxiter=int(os.environ.get("PHEASY_LSQR_MAXITER", "5000")),
                lsmr_atol=float(_lsmr_tol("PHEASY_LSQR_ATOL", 1e-8)),
                lsmr_btol=float(_lsmr_tol("PHEASY_LSQR_BTOL", 1e-8)),
                verbose=True, random_state=self._rand_seed)
            self._model.fit(A, F64, sample_weight=weights)
            coef = self._model.coef_
        elif method == "RFE-OLS-TSQR":
            # [FIX P13] default matches the class default (100), not 1
            self._model = PheasyRFE_OLS_TSQR(
                # [FIX P21] PHEASY_TSQR_STEP overrides, else PHEASY_RFE_STEP
                step=float(os.environ.get(
                    "PHEASY_TSQR_STEP",
                    os.environ.get("PHEASY_RFE_STEP", "0.05"))),
                cv=self._cv,
                min_features=int(os.environ.get("PHEASY_TSQR_MIN_FEATURES", "100")),
                block_rows=int(os.environ.get("PHEASY_TSQR_BLOCK_ROWS", "40000")),
                diag_floor=float(os.environ.get("PHEASY_TSQR_DIAG_FLOOR", "1e-12")),
                patience=int(os.environ.get("PHEASY_RFE_PATIENCE", "5")),
                verbose=True, random_state=self._rand_seed)
            self._model.fit(A, F64, sample_weight=weights)
            coef = self._model.coef_
        elif method == "RIDGE":
            alphas = np.sort(np.asarray(self._alpha, dtype=np.float64))[::-1]
            if _is_linear_operator(A_fit):
                # [FIX P26/P45] ridge CV over the alpha grid on the two-level
                # operator via _ridge_solve (LSMR on the augmented system).
                # (1) row slices are hoisted out of the alpha loop; (2) the
                # alpha path walks large->small with LSMR warm-start; (3) folds
                # are parallelized with threads (alpha stays serial).
                splits = _make_cv_splits(A.shape[0], self._cv, self._rand_seed,
                                         self._group_size)
                n_jobs = _resolve_n_jobs("RIDGE")
                A_tr_list = [_row_slice(A_fit, tr) for tr, _ in splits]
                y_tr_list = [F64[tr] for tr, _ in splits]
                warm = [None] * len(splits)

                def _fold_rmse(a, k):
                    c = _ridge_solve(A_tr_list[k], y_tr_list[k], a, x0=warm[k])
                    warm[k] = c
                    va = splits[k][1]
                    return float(np.mean(
                        (_predict_rows(A_fit, c, va) - F64[va]) ** 2))

                mse_path = np.zeros((len(alphas), len(splits)))
                parallel = n_jobs > 1 and len(splits) > 1
                n_workers = min(n_jobs, len(splits))
                if parallel:
                    from joblib import Parallel, delayed
                # [FIX P45b] enter _blas_limit ONCE for the whole alpha sweep
                # instead of per alpha (threadpool_limits walks the loaded BLAS
                # libraries on every entry/exit).
                ctx = _blas_limit(n_workers) if parallel else contextlib.nullcontext()
                # Per-alpha progress.  Ridge on a matrix-free two-level operator
                # is a long, completely silent loop (one linear solve per fold,
                # x folds x alphas); a fit that printed nothing for hours could
                # only be monitored by guessing, and an alpha grid anchored in
                # the wrong decade could not be spotted before the run ended.
                # A per-alpha line costs one print per fold-solve.
                import time as _time
                _t_alpha0 = _time.time()
                with ctx:
                    for j, a in enumerate(alphas):
                        _t_one = _time.time()
                        if parallel:
                            errs = Parallel(n_jobs=n_workers, prefer="threads")(
                                delayed(_fold_rmse)(a, k)
                                for k in range(len(splits)))
                        else:
                            errs = [_fold_rmse(a, k) for k in range(len(splits))]
                        mse_path[j] = errs
                        # ||c|| rides along for free: _fold_rmse already kept the
                        # fold coefficients in warm[], and printing their norms
                        # gives the ||c(alpha)|| half of the L-curve without
                        # solving anything extra.  Without it the ridge knee can
                        # only be read off the CV curve, which -- as measured on
                        # the c6.5/c3=4.5 fit -- has no turning point at all,
                        # while the actual tolerance drifts 7.7x looser as alpha
                        # falls (LSMR's normA is a partial bidiagonalisation sum
                        # that loses orthogonality in float32, so istop=2 is not
                        # the certificate it looks like).
                        print("[RIDGE-CV] alpha %d/%d = %.3e | fold_mse %s | mean %.6e"
                              " | ||c|| %s | %.1fs (total %.1fs)"
                              % (j + 1, len(alphas), float(a),
                                 " ".join("%.6e" % float(e) for e in errs),
                                 float(np.mean(errs)),
                                 " ".join("n/a" if c is None else "%.4e" % float(np.linalg.norm(c))
                                          for c in warm),
                                 _time.time() - _t_one,
                                 _time.time() - _t_alpha0), flush=True)
                best_alpha = float(alphas[int(np.argmin(mse_path.mean(axis=1)))])
                # L-curve.  The CV curve says which alpha predicts best; it does
                # not say whether the alpha grid was anchored anywhere near the
                # region where alpha changes the solution at all -- the first
                # RIDGE attempt on this dataset swept [1e-16,1e-8] and its
                # strongest point had not converged after 32 minutes, i.e. the
                # whole grid sat inside "indistinguishable from OLS".
                # ||c(alpha)|| and the full-data residual expose that directly.
                # Note this is NOT free in this code path: each alpha solves on
                # the CV folds only, so this is an extra warm-started full-data
                # refit per alpha (the alpha path walks large->small, so the
                # previous solution is a good x0).  Gate it with
                # PHEASY_RIDGE_LCURVE=0 if the extra solves are not wanted; the
                # best alpha's coefficient is reused for the final model.
                _lc = {}
                if os.environ.get("PHEASY_RIDGE_LCURVE", "1").lower() not in ("0", "false", "no", "off"):
                    _x0 = None
                    print("[RIDGE-LC] alpha | ||c|| | ||Xc-y|| | time", flush=True)
                    for _a in alphas:
                        _tl = _time.time()
                        _c = _ridge_solve(A_fit, F64, _a, x0=_x0)
                        _x0 = _c
                        _lc[float(_a)] = _c
                        _res = np.asarray(A_fit @ _c, dtype=np.float64).ravel() - F64
                        print("[RIDGE-LC] %.6e | %.6e | %.6e | %.1fs"
                              % (float(_a), float(np.linalg.norm(_c)),
                                 float(np.linalg.norm(_res)),
                                 _time.time() - _tl), flush=True)
                coef = _lc.get(best_alpha)
                if coef is None:
                    coef = _ridge_solve(A_fit, F64, best_alpha)
                self._model = _OLSModel(coef, alpha=best_alpha)
                self._results["alpha"] = best_alpha
                self._results["mse_path"] = mse_path
                ridge_info = getattr(A_fit, "_gpu_solver_info", None)
                if ridge_info is not None:
                    self._results["execution_backend"] = "gpu_twolevel_ridge_resident"
                    self._results["regularized_solver_info"] = dict(ridge_info)
                    self._results["postfit_backend"] = "cpu_metrics"
            else:
                # [FIX P46] dense ridge CV: grouped CV (NOT leave-one-out GCV),
                # closed-form via one economic SVD per fold. The old
                # RidgeCV(cv=None) / GpuRidgeCV ran LOO GCV, which ignored --cv
                # and PHEASY_CV_GROUP_SIZE and leaked the other 3N-1 rows of the
                # same configuration into training (biasing alpha* toward 0).
                if sp.issparse(A_fit):
                    A_dense = _to_dense_f64(A_fit)
                else:
                    A_dense = np.ascontiguousarray(A_fit, dtype=np.float64)
                splits = _make_cv_splits(A_dense.shape[0], self._cv,
                                         self._rand_seed, self._group_size)
                if self._fit_intercept:
                    # grouped CV still fixes the leak; sklearn RidgeCV handles
                    # the intercept the closed form below does not (the CLI
                    # leaves fit_intercept=False).
                    self._model = RidgeCV(alphas=self._alpha, fit_intercept=True,
                                          cv=splits)
                    self._model.fit(A_dense, F64, sample_weight=weights)
                    coef = self._model.coef_
                    self._results["alpha"] = float(self._model.alpha_)
                elif _gpu_dense(A_fit):
                    # [FIX P47] weighted ridge == unweighted ridge on
                    # sqrt(weights)-scaled rows; scale on the host, then grouped
                    # CV on the GPU (one torch SVD per fold).
                    # Reuse A_dense (already densified just above) instead of a
                    # second _to_dense_f64() copy; the *sw scaling allocates a
                    # fresh array, so the caller's A_fit is never mutated.
                    A_g = A_dense
                    y_g = F64
                    if weights is not None:
                        sw = np.sqrt(np.asarray(weights, dtype=np.float64).ravel())
                        A_g = A_dense * sw[:, None]
                        y_g = F64 * sw
                    self._model = _gpu().GpuRidgeCV(
                        alphas=self._alpha, cv=self._cv,
                        rand_seed=self._rand_seed, group_size=self._group_size)
                    self._model.fit(A_g, y_g)
                    coef = self._model.coef_
                    self._results["alpha"] = float(self._model.alpha_)
                    self._results["mse_path"] = np.asarray(self._model.mse_path_)
                else:
                    # [FIX P47] weighted ridge == unweighted ridge on
                    # sqrt(weights)-scaled rows; exact, and the CV folds then
                    # need no per-fold reweighting.
                    y_dense = F64
                    if weights is not None:
                        sw = np.sqrt(np.asarray(weights, dtype=np.float64).ravel())
                        A_dense = A_dense * sw[:, None]
                        y_dense = y_dense * sw
                    # [FIX P47] mse_path keeps the SAME (n_alphas, n_folds)
                    # shape as the operator branch (per-fold values), so the two
                    # paths agree and downstream can read per-fold RMSE.
                    mse_path = np.zeros((len(alphas), len(splits)), dtype=np.float64)
                    for k, (tr, va) in enumerate(splits):
                        U, s, Vt = np.linalg.svd(A_dense[tr], full_matrices=False)
                        Uty = U.T @ y_dense[tr]
                        AvV = A_dense[va] @ Vt.T
                        for j, a in enumerate(alphas):
                            pred = AvV @ ((s / (s ** 2 + a)) * Uty)
                            mse_path[j, k] = float(np.mean((pred - y_dense[va]) ** 2))
                    best_alpha = float(alphas[int(np.argmin(mse_path.mean(axis=1)))])
                    # final refit at the selected alpha (closed form, full data)
                    U, s, Vt = np.linalg.svd(A_dense, full_matrices=False)
                    coef = Vt.T @ ((s / (s ** 2 + best_alpha)) * (U.T @ y_dense))
                    self._model = _OLSModel(coef, alpha=best_alpha)
                    self._results["alpha"] = best_alpha
                    self._results["mse_path"] = mse_path
        else:
            raise ValueError(
                "Unknown linear model for fitting force constants: {} ".format(self._method)
                + "(expected OLS, LASSO, ALASSO, RFE, RFE-OLS-TSQR, RIDGE)")

        coef = np.asarray(coef, dtype=np.float64)

        # Relaxed LASSO / debias: L1 selects the support, then an unbiased OLS
        # refit on that support removes the L1 shrinkage bias.  This is the
        # standard practical way to obtain physical force constants from a
        # LASSO fit (Meinshausen 2007; used by ALAMODE/phono3py).  Default on;
        # disable with PHEASY_LASSO_DEBIAS=0.
        self._results.pop("pre_debias_coef", None)
        if method in ("LASSO", "ALASSO") and self._debias_enabled():
            # Preserve physical-coordinate coefficients for paired evaluation.
            self._results["pre_debias_coef"] = (
                coef / col_scale if col_scale is not None else coef.copy())
            # [FIX P34] propagate the model's Gram (built on A_fit) so _debias
            # can solve G[sup,sup] x = b[sup] instead of re-solving the OLS.
            self._gram = getattr(self._model, "_gram", None)
            coef = self._debias(A_fit, F64, coef)

        # un-scale standardized coefficients back to the original column scale
        if col_scale is not None:
            coef = coef / col_scale

        # [FIX P11] the hard threshold used to hit every method, including OLS
        # and RFE, where zeroing tiny-but-real coefficients is not wanted.
        # Default: only the L1 methods (whose exact zeros are the point).
        _default_tol = "1e-12" if method in ("LASSO", "ALASSO") else "0"
        _zero_tol = float(os.environ.get("PHEASY_COEF_ZERO_TOL", _default_tol))
        if _zero_tol > 0:
            coef = np.where(np.abs(coef) < _zero_tol, 0.0, coef)
        self._results["coef"] = coef
        self._model.coef_ = self._results["coef"]

        if method in ("LASSO", "ALASSO"):
            self._results["alpha"] = float(self._model.alpha_)
            self._results["n_iter"] = int(self._model.n_iter_)
            info = getattr(self._model, "regularized_solver_info_", None)
            if info is not None:
                self._results["regularized_solver_info"] = dict(info)
                backend = str(info.get("backend", ""))
                if backend:
                    self._results["execution_backend"] = backend
                    if self._debias_enabled():
                        db = getattr(self, "_debias_backend", "unknown")
                        self._results["debias_backend"] = db
                        self._results["postfit_backend"] = db + "_and_cpu_metrics"
                    else:
                        self._results["debias_backend"] = "disabled"
                        self._results["postfit_backend"] = "cpu_metrics"
            alpha_idx = int(np.argmin(np.abs(self._model.alphas_ - self._model.alpha_)))
            self._metrics["mse_path"] = np.asarray(self._model.mse_path_[alpha_idx])
            self._metrics["mse_path_mean"] = float(np.mean(self._metrics["mse_path"]))
            self._metrics["rmse_path"] = np.sqrt(self._metrics["mse_path"])
            self._metrics["rmse_path_mean"] = float(np.mean(self._metrics["rmse_path"]))
            self._metrics["n_features"] = self._model.n_features_in_
            self._metrics["n_featrues"] = self._metrics["n_features"]  # [FIX P12] deprecated alias
        elif method in ("RFE", "RFE-OLS-TSQR"):
            self._results["alpha"] = float(getattr(self._model, "ridge_alpha", 0.0))
            self._results["n_iter"] = int(getattr(self._model, "n_iter_", 0))
            # cv_evaluated=False means no CV ran at all: keep the metric NaN
            # rather than the fabricated 0.0 (a "perfect" score nobody measured).
            _cv_done = bool(getattr(self._model, "cv_evaluated", True))
            bcv = float(getattr(self._model, "best_rmse_cv_", 0.0)) if _cv_done else float("nan")
            if not _cv_done:
                self._metrics["cv_evaluated"] = False
            self._metrics["mse_path"] = np.array([bcv ** 2])
            self._metrics["mse_path_mean"] = bcv ** 2
            self._metrics["rmse_path"] = np.array([bcv])
            self._metrics["rmse_path_mean"] = bcv
            self._metrics["n_features"] = self._model.n_features_in_
            self._metrics["n_featrues"] = self._metrics["n_features"]  # [FIX P12] deprecated alias
            rfe_meta = getattr(self._model, "backend_metadata_", None)
            if rfe_meta is not None:
                self._results["execution_backend"] = "gpu_rfe_resident_iterative" if rfe_meta.get("subset_solver") == "gpu_resident_iterative" else ("gpu_rfe_subsets" if rfe_meta.get("subset_solver") == "gpu_dense" else "cpu_rfe")
                self._results["postfit_backend"] = "cpu_orchestration_and_metrics"
                self._results["backend_metadata"] = dict(rfe_meta)
        elif method == "RIDGE":
            ridge_info = getattr(A_fit, "_gpu_solver_info", None)
            if ridge_info is None:
                ridge_info = getattr(self._model, "regularized_solver_info_", None)
            if ridge_info is not None:
                backend = str(ridge_info.get("backend", "gpu_twolevel_ridge_resident"))
                self._results["execution_backend"] = backend
                self._results["regularized_solver_info"] = dict(ridge_info)
                self._results["postfit_backend"] = "cpu_metrics"
            # [FIX P47] alpha_ is now set on every RIDGE model (_OLSModel,
            # GpuRidgeCV, sklearn RidgeCV); _results["alpha"] is the fallback.
            self._results["alpha"] = float(getattr(
                self._model, "alpha_", self._results.get("alpha", 0.0)))
        elif method == "OLS":
            self._results["n_iter"] = getattr(self._model, "n_iter_", None)
            if self._ols_lsmr_info is not None:
                self._results["solver_info"] = dict(self._ols_lsmr_info)
                if str(self._ols_lsmr_info.get("backend", "")).startswith("gpu"):
                    self._results["execution_backend"] = str(self._ols_lsmr_info["backend"])
            else:
                self._results.pop("solver_info", None)
                # Label query only: _gpu_dense() raises under required GPU mode
                # when the matrix does not fit VRAM, and an OLS fit that already
                # completed must not fail while REPORTING which backend ran.
                try:
                    _dense_gpu = bool(_gpu_dense(A))
                except Exception as _label_exc:
                    _dense_gpu = False
                    self._results["backend_label_error"] = str(_label_exc)
                if _dense_gpu:
                    self._results["execution_backend"] = "gpu_dense"
            if getattr(self, "_ols_gpu_fallback_reason", None):
                self._results["fallback_reason"] = self._ols_gpu_fallback_reason
                self._results["execution_backend"] = "cpu_lsmr"

        # A returned coefficient vector is not automatically a certified fit.
        _solver_infos = []
        for _key in ("solver_info", "regularized_solver_info"):
            _value = self._results.get(_key)
            if isinstance(_value, dict):
                _solver_infos.append(_value)
        # RFE records its subset solves in backend_metadata; without this the
        # LSMR diagnostics never reached the acceptance check, so a subset solve
        # that hit its iteration limit still reported fit_accepted=True (the
        # resident GPU path raises for exactly the same condition).  Fold-level
        # diagnostics do not veto the fit; the FULL-data solve does.
        _meta = self._results.get("backend_metadata")
        if isinstance(_meta, dict):
            for _diag in _meta.get("iterative_diagnostics") or ():
                if isinstance(_diag, dict) and _diag.get("fit_scope", "full") == "full":
                    _solver_infos.append(_diag)
        _nonconverged = [x for x in _solver_infos if x.get("converged") is False]
        self._results["fit_accepted"] = not _nonconverged
        self._results["status"] = ("fit_returned" if not _nonconverged
                                     else "fit_returned_not_accepted")

        F_pred = np.asarray(self.predict(A)).ravel()
        eps = np.finfo(F64.dtype).eps
        F_err = np.abs(F_pred - F64)
        F_re = F_err / np.maximum(np.abs(F64), eps)

        self._metrics["re"] = float(np.sqrt(np.dot(F_err, F_err) / np.dot(F64, F64)))
        self._metrics["r2_score"] = float(r2_score(F64, F_pred, sample_weight=weights))
        self._metrics["mae"] = float(mean_absolute_error(F64, F_pred, sample_weight=weights))
        self._metrics["mape"] = float(mean_absolute_percentage_error(F64, F_pred, sample_weight=weights))
        self._metrics["mse"] = float(mean_squared_error(F64, F_pred, sample_weight=weights))
        self._metrics["rmse"] = float(np.sqrt(self._metrics["mse"]))
        self._metrics["mspe"] = float(np.average(np.square(F_re), weights=weights, axis=0))
        self._metrics["rmspe"] = float(np.sqrt(self._metrics["mspe"]))
        return self

    def _fit_ols(self, A, F):
        self._ols_lsmr_info = None
        self._ols_gpu_fallback_reason = None
        streamed = sp.issparse(A) or (hasattr(A, "SM_prime") and hasattr(A, "NS"))
        if streamed and os.environ.get("PHEASY_GPU_TSQR", "0").lower() in ("1", "true", "yes", "on"):
            from . import gpu_backend as gb
            # Preserve the operator ridge/Jacobi options handled by LSMR.
            if gb.enabled() and float(os.environ.get("PHEASY_OLS_RIDGE", "0")) == 0 and os.environ.get("PHEASY_OLS_JACOBI", "0").lower() not in ("1", "true", "yes"):
                try:
                    coef, info = gb.gpu_tsqr(A, F, block_rows=int(os.environ.get("PHEASY_TSQR_BLOCK_ROWS", "40000")))
                    self._ols_lsmr_info = info
                    return coef, None
                except (MemoryError, RuntimeError, np.linalg.LinAlgError) as exc:
                    if _gpu_required():
                        raise RuntimeError("GPU TSQR OLS solve failed with fallback disabled: %s" % exc) from exc
                    self._results["fallback_reason"] = "GPU TSQR: %s: %s" % (type(exc).__name__, exc)
                    # Keep sparse fallback matrix-free even for small inputs.
                    if sp.issparse(A):
                        self._ols_lsmr_info = {}
                        coef = _solve_sparse_lsqr(A, F, info=self._ols_lsmr_info)
                        self._results["execution_backend"] = "cpu_lsqr"
                        return coef, self._ols_lsmr_info["itn"]
                    self._results["execution_backend"] = "cpu_lsmr"
        if _is_linear_operator(A):
            # LSMR only needs matvec/rmatvec; the two-level operator stays sparse.
            coef = self._ols_lsmr(A, F, maxiter=self._max_iter)
            n_iter = self._ols_lsmr_info.get("itn")
            return coef, n_iter
        if sp.issparse(A) and not _should_densify_sparse(A):
            self._ols_lsmr_info = {}
            coef = _solve_sparse_lsqr(A, F, info=self._ols_lsmr_info)
            return coef, self._ols_lsmr_info["itn"]
        # dense, or a sparse container: _solve_lstsq densifies small/sparse-but-
        # dense matrices (fast SVD) and only uses iterative LSQR for genuinely
        # huge sparse matrices (FIX: previously everything sparse went to LSMR).
        coef = _solve_lstsq(A, F, driver="gelsd")
        return coef, None

    @staticmethod
    def _debias_enabled():
        return os.environ.get("PHEASY_LASSO_DEBIAS", "1").lower() in ("1", "true", "yes")

    def _debias(self, A, y, coef):
        """OLS refit on the nonzero support (relaxed LASSO)."""
        sup = np.flatnonzero(np.abs(coef) > 0)
        # Full support still has L1 shrinkage and must be refitted.
        # Only empty support has no least-squares problem to solve.
        if sup.size == 0:
            self._debias_backend = "skipped"
            return coef
        gram = getattr(self, "_gram", None)
        # Relaxed-LASSO debias is an explicitly-declared post-fit stage; a CPU
        # support refit here is permitted (and reported via postfit_backend), not
        # a silent solver fallback. It never changes which backend ran the main
        # regularized solve.
        if gram is not None:
            # [FIX P34] OLS on the support via the Gram: G[sup,sup] x = b[sup]
            # is a |sup| x |sup| dense solve, far cheaper than re-solving the
            # full least-squares problem against the operator.
            self._debias_backend = "cpu_gram_solve"
            G, b = gram
            Gss = G[np.ix_(sup, sup)]
            bs = b[sup]
            try:
                coef_sub = np.linalg.solve(Gss, bs)
            except np.linalg.LinAlgError:
                coef_sub = _solve_lstsq(Gss, bs)
            new = np.zeros_like(coef)
            new[sup] = coef_sub
            r_new = float(np.linalg.norm(np.asarray(A @ new).ravel() - y))
            r_old = float(np.linalg.norm(np.asarray(A @ coef).ravel() - y))
            if r_new <= r_old:
                return new
            return coef
        if _is_linear_operator(A):
            # [FIX P26] column-slice via a masked operator + LSMR, so the
            # relaxed-LASSO debias is no longer skipped on the two-level
            # operator (the L1 shrinkage bias is removed there too).
            if self._resident_debias_available():
                coef_sub = self._debias_resident_gpu(y, sup)
            else:
                self._debias_backend = "cpu_lsmr"
                op = _make_masked_op(A, None, sup)
                coef_sub = _solve_sparse_lsqr(op, y)
            new = np.zeros_like(coef)
            new[sup] = coef_sub
            r_new = float(np.linalg.norm(np.asarray(A @ new).ravel() - y))
            r_old = float(np.linalg.norm(np.asarray(A @ coef).ravel() - y))
            if r_new <= r_old:
                return new
            return coef
        A_sub = A[:, sup]
        self._debias_backend = ("gpu_dense_lstsq" if _gpu_dense(A_sub) else "cpu_dense_lstsq")
        coef_sub = _solve_lstsq(A_sub, y)
        new = np.zeros_like(coef)
        new[sup] = coef_sub
        # keep only if the residual does not increase (guards against a
        # support that is inconsistent with the data scale)
        r_new = float(np.linalg.norm(np.asarray(A @ new).ravel() - y))
        r_old = float(np.linalg.norm(np.asarray(A @ coef).ravel() - y))
        if r_new <= r_old:
            return new
        return coef

    def _resident_debias_available(self):
        """True when the resident operator is retained and CUDA is on.

        The retained operator is only usable for a GPU support refit on real CUDA;
        under torch-CPU emulation (tests) or an explicit PHEASY_GPU_DEBIAS=0 the
        ordinary CPU LSQR path keeps running so the post-fit semantics are unchanged.
        """
        res_op = getattr(getattr(self, "_model", None), "_operator", None)
        if res_op is None:
            return False
        dev = getattr(res_op, "device", None)
        if dev is None or str(dev).split(":")[0] != "cuda":
            return False
        if os.environ.get("PHEASY_GPU_DEBIAS", "1").lower() in ("0", "false", "no", "off"):
            return False
        return True

    def _debias_resident_gpu(self, y, sup):
        """Support OLS refit on the retained resident operator (GPU CGLS).

        solve_resident_subset solves against the normalized resident operator, so
        its coefficients carry the column scale; divide by column_scale_ to return
        the physical-coordinate support coefficients that match the CPU LSQR path.
        """
        from . import gpu_backend as gb
        res_op = self._model._operator
        try:
            coef_sub, _info = gb.solve_resident_subset(res_op, y, sup, raise_on_nonconvergence=False)
            coef_sub = gb._to_numpy(coef_sub, np.float64)
            scale = getattr(self._model, "column_scale_", None)
            if scale is not None:
                coef_sub = coef_sub / np.asarray(scale, dtype=np.float64)[sup]
        finally:
            res_op.close()
            self._model._operator = None
        self._debias_backend = "gpu_cgls"
        return coef_sub

    @staticmethod
    def _detect_group_size(n_samples):
        gs = int(os.environ.get("PHEASY_CV_GROUP_SIZE", "0"))
        if gs > 1 and n_samples % gs == 0:
            return gs
        return None

    def predict(self, A):
        return self._model.predict(A)

    @property
    def results(self):
        return self._results

    @property
    def metrics(self):
        return self._metrics

    @property
    def model(self):
        return self._model

    def get_paras(self):
        return self._model.coef_

    def __repr__(self):
        return "<Optimizer method={}>".format(self._method)
