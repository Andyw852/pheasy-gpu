"""Classes and functions for force constant regression.

Implements the force-constant fitting methods exposed by pheasy:
OLS, RFE / RFE-OLS (recursive feature elimination with an OLS base
estimator; "RFE" and "RFE-OLS" are aliases), RFE-OLS-TSQR (RFE_TSQR),
LASSO, ALASSO (adaptive LASSO), ARDR (automatic relevance determination
regression) and the legacy RIDGE method.  The public entry point is the
Optimizer class.

References:
  H. Zou, "The Adaptive Lasso and Its Oracle Properties", JASA 101 (2006).
  F. Eriksson et al., Adv. Theory Simul. 2 (2019) (hiphive).
  J. Demmel et al., SIAM J. Sci. Comput. 34 (2012) A206 (TSQR).
  E. Fransson, F. Eriksson, P. Erhart, npj Comput. Mater. 6, 135 (2020)
    (OLS / LASSO / RFE-OLS / ARDR comparison for force-constant models;
    ARDR = scikit-learn ARDRegression, threshold_lambda = 1e4).
  D. J. C. MacKay, "Bayesian interpolation", Neural Comput. 4 (1992) 415.
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

    The factors are read through the _twolevel_base chain FIRST, because the
    wrappers that carry a row slice or a column scaling are declared float64
    whatever they multiply: _scale_columns -> _scale_operator builds a
    _CustomLinearOperator(..., dtype=np.float64) that keeps its two-level
    provenance only in _twolevel_base, and _row_slice of it (every CV fold)
    inherits that.  Testing A.dtype before the factors therefore reported
    float64 for a float32 operator, _lsmr_tol saw 1e-8 as reachable and left it
    alone -- the exact failure the floor was written to prevent.  Measured on
    the c7 resident-ridge operator: atol 1.192e-06 for the bare TwoLevelSM but
    1e-08 once --std wrapped it in _scale_operator, i.e. the resident CGLS was
    asked for a tolerance five orders below what float32 can represent.
    """
    # Explicit provenance wins over inference: the wrappers below record what
    # they multiply in _data_dtype, because a wrapper that declares
    # dtype=float64 around float32 factors (see _scale_operator, _row_slice_op)
    # is otherwise indistinguishable from a genuine float64 operator.
    hint = getattr(A, "_data_dtype", None)
    if hint is not None:
        try:
            return np.dtype(hint).type
        except TypeError:
            pass
    seen = set()
    node = A
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        dt = getattr(getattr(node, "SM_prime", None), "dtype", None)
        if dt is not None:
            try:
                return np.dtype(dt).type
            except TypeError:
                pass
        node = getattr(node, "_twolevel_base", None)
    dt = getattr(A, "dtype", None)
    if dt is not None:
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


# ---------------------------------------------------------------------------
# [FIX P46] LASSO auto-grid: span floor + per-decade density.
#
# The grid TOP is principled and system independent: alpha_max = max_j|X_j^T y|/n
# is the exact KKT threshold (verified to 1e-9 against the definition on
# synthetic sparse/half/dense problems).  The grid BOTTOM used to be a bare
# --alpha_decades = 4, and that truncates cross-validation whenever the data is
# high-SNR, because the CV curve then keeps falling below everything the grid
# reaches and alpha* is pinned to the bottom edge -- i.e. the penalty is chosen
# by the GRID, not by the data.
#
# Measured on Mg8C120 (454656 x 69487, c2=7/c3=4, 296 configs):
#     4-decade grid  -> alpha*=3.846e-08 (1.2x above the bottom), re 7.52%,
#                       grouped holdout relL2 7.54%   (the shipped v4 fit)
#     6-decade grid  -> alpha*=2.305e-10 (still the bottom),  re 1.34%,
#                       grouped holdout relL2 1.65%; the same command with the
#                       same data: 4dec 2.43% vs 6dec 1.32% on 237/59 configs
#     OLS            ->                                     re 1.39%,
#                       grouped holdout relL2 1.23%
# The synthetic study (tmp/alpha_grid_snr.py) shows the regime boundary: with a
# genuinely sparse truth (20 of 300 coefficients) the CV minimum is INTERIOR at
# 4 decades, so the floor below is a no-op there; with a dense truth and low
# noise (the Mg8C120 regime) alpha* pins at the bottom for 4, 6, 8 AND 10
# decades and the LASSO solution equals OLS to the last digit -- which is what
# the relaxed/L1-free refit in Optimizer.fit is for.
#
# ALASSO already floors its OVERDETERMINED grid at 6 decades ([FIX P37]); LASSO
# did not, so the two paths disagreed on the same data.  Keep the floor to
# overdetermined problems: an underdetermined system genuinely needs
# regularization, its CV optimum is interior, and a long low-alpha tail only
# makes the solver crawl.
_LASSO_GRID_MIN_DECADES = 6.0


def lasso_grid_floor_enabled():
    """PHEASY_LASSO_GRID_FLOOR=0 restores the historical bare --alpha_decades."""
    return os.environ.get("PHEASY_LASSO_GRID_FLOOR", "1").lower() in ("1", "true", "yes")


def lasso_grid_min_decades(n_samples, n_features, decades):
    """Effective decades below alpha_max for an auto-derived LASSO grid."""
    dec = float(decades)
    if not lasso_grid_floor_enabled():
        return dec
    if int(n_samples) <= int(n_features):
        return dec
    floor = float(os.environ.get("PHEASY_LASSO_GRID_MIN_DECADES",
                                 str(_LASSO_GRID_MIN_DECADES)))
    if not np.isfinite(floor) or floor <= dec:
        return dec
    return floor


def lasso_alpha_grid(lo, hi, nalpha):
    """Log grid lo..hi whose DENSITY does not drop when the span widens.

    Widening the span must not coarsen the step (a 6-decade span at a fixed 20
    points is 2x coarser per step than 4 decades, which is how the v4 fit ended
    up with alpha* between two grid points).  Density follows
    PHEASY_ALPHA_PER_DECADE (default (nalpha-1)/4, i.e. the historical 4-decade
    grid's density) and the count is capped by PHEASY_ALPHA_NMAX.
    """
    lo = float(lo)
    hi = float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= lo:
        raise ValueError("invalid alpha grid bounds [%r, %r]" % (lo, hi))
    span = float(np.log10(hi / lo))
    per_dec = float(os.environ.get(
        "PHEASY_ALPHA_PER_DECADE", str(max((int(nalpha) - 1) / 4.0, 1.0))))
    n = 1 + int(np.ceil(span * per_dec))
    n = max(n, int(nalpha))
    nmax = int(os.environ.get("PHEASY_ALPHA_NMAX", "200"))
    if nmax > 0 and n > nmax:
        n = nmax
    return np.logspace(np.log10(lo), np.log10(hi), n)


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
    # [FIX P46] span floor + span-independent density (see the block above).
    _dec = lasso_grid_min_decades(n, p, decades)
    if _dec > float(decades):
        print("[alpha_auto] alpha grid widened from %.1f to %.1f decades below the "
              "KKT threshold (overdetermined %d x %d; PHEASY_LASSO_GRID_FLOOR=0 "
              "restores the old span). A 4-decade grid pins alpha* to its own "
              "bottom on high-SNR data: Mg8C120 re 7.52%% at 4 decades vs 1.34%% "
              "at 6 with the identical command."
              % (float(decades), _dec, n, p), flush=True)
    a_min = a_max * 10.0 ** (-_dec)
    return lasso_alpha_grid(a_min, a_max, nalpha)


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
    # Pass the matrix: the floor is a property of the STORED data, so a sparse
    # float32 sensing matrix cannot support the 1e-8 default and every solve on
    # it would burn its whole iter_lim with istop=7 instead of converging.
    atol = float(_lsmr_tol("PHEASY_LSQR_ATOL", 1e-8, A))
    btol = float(_lsmr_tol("PHEASY_LSQR_BTOL", 1e-8, A))
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
            # [FIX R3] The resident operator tries a FULL per-card replica first
            # and downgrades to ONE sharded operator ("CV folds run sequentially")
            # when a replica does not fit, so the pre-flight must describe the
            # sharded footprint too.  Without n_shards a large third-order fit is
            # judged against the single-card number and sent to CPU even when the
            # sharded operator fits: measured at c3=5.0, 88029528920 bytes was
            # compared against 23643429273, while the sharded 22540422422 bytes
            # would have fit that same budget.
            try:
                _n_shards = max(1, len(_gbp._resident_cv_devices()))
            except Exception:
                _n_shards = 1
            peak, budget, _free, _frac = _gbp.resident_twolevel_estimate(
                A, n_shards=_n_shards)
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

    result = LinearOperator((n_rows, n_cols), matvec=mv, rmatvec=rmv, dtype=dt)
    # The masked view multiplies exactly the same stored factors, so it carries
    # the same reachable precision -- otherwise _lsmr_tol reads the ambient
    # dtype of this wrapper and the RFE subset solves lose the floor.
    result._data_dtype = np.dtype(_array_precision(A))
    return result


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


def _gram_smprime(SM_prime, block_rows=None):
    """P = SM_prime^T SM_prime via blocked densification + BLAS gemm.

    [FIX P34] SM_prime (n x mid) may be huge and sparse; densifying it all at
    once costs n*mid*8 bytes. Processing row blocks keeps peak memory at
    block_rows*mid*8 and accumulates the mid x mid Gram with BLAS.

    block_rows trades memory for P traffic: each block rewrites the whole
    mid x mid P (2 * mid^2 * 8 bytes).  At mid=90108 that is 130 GB per block,
    so the default 2000 gives 154 blocks = ~20 TB of traffic (measured 740 s
    per block on the MgC 2+3 build, i.e. ~32 h).  PHEASY_GRAM_SMPRIME_BLOCK_ROWS
    raises it (20000 -> 15 blocks, ~3 h of traffic, at the cost of a
    block_rows x mid float64 buffer).
    """
    import time as _t
    if block_rows is None:
        block_rows = int(os.environ.get("PHEASY_GRAM_SMPRIME_BLOCK_ROWS", "2000"))
    n, mid = SM_prime.shape
    P = np.zeros((mid, mid), dtype=np.float64)
    _nblk = (n + block_rows - 1) // block_rows
    _t0 = _t.time()
    _tlast = _t0
    _used_syrk = False
    try:
        from scipy.linalg import blas as _blas
    except Exception:
        _blas = None
    for _b, i0 in enumerate(range(0, n, block_rows)):
        B = np.asarray(SM_prime[i0:i0 + block_rows].toarray(), dtype=np.float64)
        if _blas is not None and hasattr(_blas, "dsyrk"):
            # P += B^T B in place.  The naive form materializes a second
            # mid x mid array (65 GB at mid=90108) on top of P, i.e. ~130 GB
            # peak, which segfaulted the 2+3 build; dsyrk accumulates into P.
            P = _blas.dsyrk(1.0, B, beta=1.0, c=P, trans=1, lower=0, overwrite_c=1)
            _used_syrk = True
        else:
            P += B.T @ B
        _now = _t.time()
        if _b % 20 == 0 or _now - _tlast > 60.0 or _b == _nblk - 1:
            print("[gram] P = SM_prime^T SM_prime: block %d/%d (%.1f%%) "
                  "%.0fs elapsed, %.0fs since last report"
                  % (_b + 1, _nblk, 100.0 * (_b + 1) / _nblk, _now - _t0,
                     _now - _tlast), flush=True)
            _tlast = _now
    if _used_syrk:
        # SYRK fills ONE triangle only (upper with lower=0); the other stays
        # zero, so P was silently triangular and G = NS^T P NS indefinite.
        # alpha*G + diag(lambda) then failed Cholesky and fell back to the
        # ~100x slower pinvh (and the ARD trajectory was wrong).  Mirror the
        # strict upper triangle into the lower to make P symmetric.
        P += np.triu(P, 1).T
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
        # [FIX P40] G = NS^T P NS computed in column blocks so the mid x p
        # intermediate (P @ NS) is never fully materialized.  At c3=3.8 that
        # intermediate is 78930 x 60238 x 8 = 38 GB; materializing it on top of
        # P (49.8 GB) and G (29 GB) peaked at ~117 GB and was OOM-killed on the
        # shared box.  Blocking caps the peak at P + mid x blk + G.
        _NS = A.NS
        _p = _NS.shape[1]
        _blk = int(os.environ.get("PHEASY_GRAM_PROJECT_BLOCK", "1024"))
        _blk = max(1, min(_blk, _p))
        G = np.zeros((_p, _p), dtype=np.float64)
        for _j0 in range(0, _p, _blk):
            _j1 = min(_j0 + _blk, _p)
            _Tblk = np.asarray(P @ _NS[:, _j0:_j1], dtype=np.float64)
            G[:, _j0:_j1] = np.asarray(_NS.T @ _Tblk, dtype=np.float64)
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


def _compute_gram_blockwise(A, y, blk=None):
    """G = A^T A and b = A^T y built column-block-wise, without forming P.

    For a TwoLevelSM, _compute_gram first builds P = SM_prime^T SM_prime
    (mid x mid) and then G = NS^T P NS.  When mid is large, P dominates the
    peak memory (and can exceed G itself), even though the null-space
    projection makes G much smaller.  This loop touches only n x blk and
    p x blk temporaries:

        SM_blk    = A @ I_blk        (n x blk)
        G[:, blk] = A^T @ SM_blk     (p x blk)

    so the peak is G + n*blk.  Cost is one TwoLevelSM matvec pair per block;
    PHEASY_GRAM_BLOCK (default 64) trades memory for Python/matvec overhead.
    """
    n, m = A.shape
    y64 = np.asarray(y, dtype=np.float64).ravel()
    b = np.asarray(A.T @ y64, dtype=np.float64).ravel()
    blk = int(os.environ.get("PHEASY_GRAM_BLOCK", "64")) if blk is None else int(blk)
    blk = max(1, blk)
    G = np.zeros((m, m), dtype=np.float64)
    for j0 in range(0, m, blk):
        j1 = min(j0 + blk, m)
        I_blk = np.zeros((m, j1 - j0), dtype=np.float64)
        I_blk[j0:j1, :] = np.eye(j1 - j0, dtype=np.float64)
        A_blk = np.asarray(A @ I_blk, dtype=np.float64)     # n x blk
        G[:, j0:j1] = np.asarray(A.T @ A_blk, dtype=np.float64)
    return G, b


def _build_gram_matrix(A, y64, budget_gb=None, force_block=False):
    """Build (G = X^T X, b = X^T y) by the cheapest route that fits the budget.

    For two-level input the factorized P = SM_prime^T SM_prime route needs far
    fewer matvecs, but P is mid x mid and dominates when the null space
    projects many columns away.  When P exceeds budget_gb (or force_block is
    set) this falls back to _compute_gram_blockwise, whose peak is G + n*blk.
    Column-scaled operator wrappers are handled by scaling G and b.

    Returns (G, b, how) with how in {"P", "block", "dense", "sparse"}.
    """
    base = getattr(A, "_twolevel_base", None)
    scale = getattr(A, "_twolevel_scale", None)
    if budget_gb is None:
        budget_gb = float(os.environ.get(
            "PHEASY_ARDR_GRAM_MAX_GB", os.environ.get("PHEASY_GRAM_MAX_GB", "4")))
    _no_p_env = os.environ.get("PHEASY_ARDR_GRAM_NO_P")
    _force = force_block or (_no_p_env is not None
                             and _no_p_env.lower() in ("1", "true", "yes", "on"))
    p_base = base if base is not None else A
    can_p = hasattr(p_base, "SM_prime") and not _force
    if can_p and not _gram_budget_ok(p_base, max_gb=budget_gb):
        print("[gram] P = SM_prime^T SM_prime (mid^2) exceeds "
              "PHEASY_ARDR_GRAM_MAX_GB=%.1f; using the P-free block Gram"
              % budget_gb, flush=True)
        can_p = False
    if can_p and os.environ.get("PHEASY_ARDR_GRAM_COST_HEURISTIC", "0").lower() in (
            "1", "true", "yes", "on"):
        # Opt-in cost heuristic.  In theory the P route needs n*mid^2 flops and
        # the block route ~2*nnz*p, but measured on the MgC operators the block
        # route is SLOWER (its sparse multi-column products run at a few
        # GFLOPS: p=7918/mid=8361 block Gram took 489 s vs ~195 s for P), so P
        # stays the default when it fits memory.
        _mid = getattr(p_base, "SM_prime", None)
        try:
            _npt = float(A.shape[0]); _ppt = float(A.shape[1])
            if (_mid is not None
                    and _npt * float(_mid.shape[1]) ** 2 > 4.0 * float(_mid.nnz) * _ppt):
                print("[gram] block Gram is >4x cheaper than P "
                      "(n*mid^2=%.3g vs 2*nnz*p=%.3g); using the P-free block Gram"
                      % (_npt * float(_mid.shape[1]) ** 2,
                         2.0 * float(_mid.nnz) * _ppt), flush=True)
                can_p = False
        except Exception:
            pass
    if can_p:
        G, b, _P = _compute_gram(p_base, y64)
        if scale is not None:
            d = 1.0 / np.asarray(scale, dtype=np.float64).ravel()
            G = G * d[:, None] * d[None, :]
            b = b * d
        return G, b, "P"
    p = int(A.shape[1])
    if p * p * 8.0 / 1e9 > budget_gb:
        raise NotImplementedError(
            "the design Gram G = p x p (p=%d, %.2f GB) exceeds "
            "PHEASY_ARDR_GRAM_MAX_GB=%.1f; raise the budget only if the machine "
            "can genuinely hold it." % (p, p * p * 8.0 / 1e9, budget_gb))
    if isinstance(A, np.ndarray) or sp.issparse(A):
        # A concrete matrix: one BLAS gemm beats the block loop.
        G, b, _P = _compute_gram(A, y64)
        return G, b, "dense" if isinstance(A, np.ndarray) else "sparse"
    G, b = _compute_gram_blockwise(A, y64)
    return G, b, "block"


def _fista_lasso(A, y, alpha, x0=None, max_iter=3000, tol=1e-7,
                 lipschitz=None, penalty_weights=None, _info=None,
                 gram=None, n_samples=None, warn_nonconvergence=True,
                 auto_floor=None):
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
    # --- measured-floor valve (same idiom as _fista_twolevel / the CGLS one) --
    # This is the path the LASSO takes whenever the resident factors do not fit
    # the device (PHEASY_GPU_RESIDENT_FALLBACK): the matvecs are still the
    # sharded GPU ones, so the certificate is limited by the same float32
    # operator, and a tolerance below that floor can never fire.
    try:
        from .gpu_backend import (_fista_floor_cfg as _floor_cfg,
                                  _FISTA_STALL_MARGIN as _stall_margin,
                                  _FISTA_KKT_EVERY as _kkt_every)
    except Exception:                                   # pragma: no cover
        _floor_cfg = lambda auto_floor=None, max_iter=None: (False, 3, 1e-2, 0.10)  # noqa: E731
        _stall_margin = 0.05
        _kkt_every = 20
    _valve, _stall_limit, _floor_max, _floor_margin = _floor_cfg(auto_floor,
                                                                max_iter)
    tol_requested = float(tol)
    tol_eff = float(tol)
    stop_reason = "iteration_limit"
    best_kkt = None
    best_x = None
    stalls = 0
    measured_floor = None
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
            small_step = dx <= tol_eff * max(1.0, float(np.linalg.norm(x)))
            if small_step or _valve:
                kkt = kkt_relative(x)
                if np.isfinite(kkt) and kkt <= tol_eff:
                    converged = True
                    stop_reason = ("converged" if tol_eff <= tol_requested
                                   else "converged_on_raised_tol")
                    break
                if _valve and not (np.isfinite(step) and step > 0.0):
                    # Solver-health guard, BEFORE the floor valve: a zero step
                    # (nonfinite or overflowed Lipschitz estimate) freezes the
                    # iterate, which leaves the certificate just as flat as a
                    # precision floor does.  Never certify that.
                    warnings.warn(
                        "FISTA step size is %g (lipschitz estimate %r): a zero or "
                        "nonfinite step cannot move the iterate, so the flat "
                        "certificate at iteration %d is a step-size failure, not a "
                        "precision floor.  Stopping WITHOUT certifying; re-estimate "
                        "the Lipschitz constant (PHEASY_FISTA_LIPSCHITZ_SAFETY / "
                        "PHEASY_LIPSCHITZ_POWER_P) or check the operator."
                        % (step, lipschitz, n_iter), RuntimeWarning, stacklevel=2)
                    stop_reason = "step_size_diverged"
                    break
                if _valve:
                    # Certificate stopped improving: measure the floor instead
                    # of assuming the requested tolerance is reachable.
                    if best_kkt is None or kkt < best_kkt * (1.0 - _stall_margin):
                        best_kkt = kkt
                        best_x = x.copy()
                        stalls = 0
                    else:
                        stalls += 1
                    if stalls >= _stall_limit:
                        measured_floor = best_kkt
                        if measured_floor <= _floor_max:
                            accepted = min(measured_floor * (1.0 + _floor_margin),
                                           _floor_max)
                            if accepted > tol_eff:
                                warnings.warn(
                                    "FISTA cannot reach tol=%g (operator dtype %s): "
                                    "the relative KKT certificate improved by less "
                                    "than %g%% in every %d-iteration check for the "
                                    "last %d iterations (best %g at iteration %d).  "
                                    "The request is below what this precision can "
                                    "express, so the run certifies the BEST iterate "
                                    "at tol_effective=%g instead of spending the "
                                    "remaining %d of max_iter=%d iterations on a "
                                    "stopping test it cannot satisfy.  Request "
                                    "--tol >= %g, or shorten the window with "
                                    "PHEASY_FISTA_STALL_POINTS, to make the request "
                                    "honest."
                                    % (tol_requested, _array_precision(A),
                                       100.0 * _stall_margin, _kkt_every,
                                       int(_stall_limit) * _kkt_every, measured_floor,
                                       n_iter, accepted, int(max_iter) - n_iter,
                                       int(max_iter), accepted),
                                    RuntimeWarning, stacklevel=2)
                                tol_eff = accepted
                            x = best_x
                            kkt = measured_floor
                            stop_reason = "converged_measured_floor"
                            break
                        warnings.warn(
                            "FISTA stalled at relative KKT=%g, past "
                            "PHEASY_FISTA_FLOOR_MAX=%g: this precision cannot certify "
                            "any meaningful tolerance here, so the run stops at "
                            "iteration %d instead of grinding to max_iter=%d.  Re-run "
                            "with --tol >= %g or fix the conditioning."
                            % (measured_floor, _floor_max, n_iter, int(max_iter),
                               measured_floor * 1.05), RuntimeWarning, stacklevel=2)
                        x = best_x
                        kkt = measured_floor
                        stop_reason = "stall_above_floor"
                        break
            x_prev = x.copy()
    if stop_reason not in ("converged", "converged_on_raised_tol"):
        kkt = kkt_relative(x)
        if stop_reason == "converged_measured_floor":
            converged = bool(np.isfinite(kkt) and kkt <= tol_eff)
            if not converged:
                stop_reason = "stall_above_floor"
        else:
            converged = bool(np.isfinite(kkt) and kkt <= tol)
    if _info is not None:
        _info.update(n_iter=n_iter, converged=converged, kkt_relative=kkt,
                     stop_reason=stop_reason, tol_requested=tol_requested,
                     tol_effective=tol_eff, measured_floor=measured_floor,
                     stall_points=int(_stall_limit))
    if not converged and warn_nonconvergence:
        # CV folds pass warn_nonconvergence=False and report ONE aggregated line
        # instead: a ranking-only solve that stops at the cap is not the same
        # problem as an uncertified final refit, and 36 identical tracebacks
        # buried the real diagnostics.
        # NOTE: deliberately no function-local "import warnings" here.  A local
        # import binds the name for the WHOLE function, so every warnings.warn
        # ABOVE this line (the FISTA floor / measured-floor warnings) raised
        # UnboundLocalError: cannot access local variable "warnings" -- measured
        # by dev.test_optimizer_large_regressions' FISTA floor case, and the
        # same bug fires on any CPU-FISTA LASSO/ALASSO run that stalls.  This
        # function uses the module-level import at the top of the file.
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
    result._data_dtype = np.dtype(_array_precision(A))
    # MOVE any resident operator the wrapped operator owns to this wrapper.
    # It is the same factors, and leaving it behind made the wrapper -- the
    # operator the RIDGE CV folds are sliced from -- upload them again, once for
    # the wrapper and once per fold.  _resident_ridge_op() re-applies the column
    # scaling (input_scale) whenever it hands the op out, so a moved op stays
    # consistent; the source simply rebuilds if anything asks it again.
    _cached_op = getattr(A, "_gpu_ridge_op", None)
    if _cached_op is not None:
        try:
            del A._gpu_ridge_op
        except Exception:
            pass
        try:
            result._gpu_ridge_op = _cached_op
        except Exception:
            pass
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

    result = LinearOperator((len(rows), n), matvec=mv, rmatvec=rmv, dtype=dt)
    result._data_dtype = np.dtype(_array_precision(A))
    return result


class _ResidentRowView(object):
    """A[rows, :] as a VIEW on an already-uploaded resident operator.

    The CV folds used to build their own GpuTwoLevelOperator from the sliced
    host factors: one factor upload plus one host csr_tocsc transpose each
    (both inside GpuTwoLevelOperator.__init__).  On the MgC production operator
    that is ~3 x (137 s upload + transpose) per RIDGE fit, which outweighs every
    GPU kernel in the fit and is why the folds looked "host bound" after the
    column norms moved to the GPU.  The factors the parent already has on the
    device cover exactly the same matrix, so a fold is a row SELECTION over
    them -- GpuSubsetOperator, the same view the resident RFE subsets use.

    matvec/rmatvec return numpy on purpose: the host CV prediction path calls
    np.asarray() on their result, and a CUDA tensor there fails.
    """

    def __init__(self, parent, resident, rows):
        self._parent = parent
        self._resident = resident
        self._rows = np.asarray(rows, dtype=np.intp)
        self._cols = np.arange(int(parent.shape[1]), dtype=np.intp)
        self.shape = (int(self._rows.size), int(parent.shape[1]))
        self.dtype = np.dtype(np.float64)
        self._view = None

    @property
    def _gpu_ridge_parent(self):
        # the host parent rides along so the tolerance floor still reads the
        # FACTOR dtype (float32) instead of this view's float64 shell.
        return (self._resident, self._cols, self._rows, self._parent)

    @property
    def _data_dtype(self):
        # [FIX P50c 2026-09-25] This view multiplies the PARENT's stored
        # factors, so it must report THEIR precision.  Without this it was a
        # float64 shell wrapped around a float32 operator, and the tolerance
        # floor (_array_precision -> the FISTA tol_effective / _lsmr_tol) read
        # float64: a request of 1e-7 looked reachable and the run kept
        # iterating past what float32 can express.  Same provenance idiom as
        # _row_slice_op / _scale_operator, which record _data_dtype for exactly
        # this reason (see _array_precision's docstring).
        return np.dtype(_array_precision(self._parent))

    def _subset(self):

        if self._view is None:
            from . import gpu_backend as _gb
            self._view = _gb.GpuSubsetOperator(self._resident, self._cols,
                                               self._rows)
        return self._view

    def _to_device(self, v):
        r = self._resident
        return r.torch.as_tensor(np.asarray(v, dtype=np.float64).ravel(),
                                 dtype=r._value_dtype, device=r.device)

    def matvec(self, v):
        out = self._subset().matvec(self._to_device(v))
        return np.asarray(out.detach().cpu().numpy(), dtype=np.float64).ravel()

    def rmatvec(self, u):
        out = self._subset().rmatvec(self._to_device(u))
        return np.asarray(out.detach().cpu().numpy(), dtype=np.float64).ravel()

    def __matmul__(self, x):
        """Operator product: the view must behave like every other operator here.

        [FIX P50 2026-09-17] _is_linear_operator duck-types on matvec, so this
        view IS an operator as far as the dispatchers are concerned -- but FISTA
        and the masked/scaled wrappers multiply through the @ operator, not
        through matvec.  A view that exposed only matvec/rmatvec would therefore
        pass dispatch and fail one line later (A @ z / A.T @ u).  2-D operands
        take the column path so the Gram builder can still multiply blocks.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            return self.matvec(arr)
        if arr.ndim == 2:
            return np.column_stack([self.matvec(arr[:, j])
                                    for j in range(arr.shape[1])])
        raise ValueError("operator product expects a 1-D or 2-D operand, got %d-D"
                         % arr.ndim)

    dot = __matmul__

    def row_slice(self, rows):
        # rows index THIS view; the resident op indexes the parent.
        return _ResidentRowView(self._parent, self._resident,
                                self._rows[np.asarray(rows, dtype=np.intp)])

    def transpose(self):
        return _ResidentRowAdjoint(self)

    @property
    def T(self):
        """Transpose of the view, as an operator with matvec/rmatvec swapped.

        [FIX P50 2026-09-17] This view is NOT a scipy LinearOperator -- matvec
        and rmatvec are plain methods that return numpy -- so "A.T" used to be
        an AttributeError.  That stayed invisible while the only consumer was
        _ridge_solve, which recognises the view through _gpu_ridge_parent, but
        _row_slice hands the view to EVERY caller that slices rows of an
        operator carrying a cached _gpu_ridge_op, and FISTA is one of them:
        _lasso_backend returns "iterative" (FISTA) whenever the resident
        two-level LASSO is not active (e.g. PHEASY_GPU_LASSO_RESIDENT=false
        with PHEASY_GPU_SM=1), and _fista_lasso needs the transpose for the KKT
        scale (A.T @ y), the gradient (A.T @ (Az - y)) and the Lipschitz power
        iteration (A.T @ u).  Measured on the Mg2C60 2x2x2 fit (2026-09-17):
        after a RIDGE CV had cached the resident op, the LASSO fit died with
        AttributeError: a _ResidentRowView has no attribute "T".

        The adjoint is a dedicated class built from the swapped BOUND methods.
        It must never be a closure over self.T (or over a name later rebound to
        the transpose): that is the recursion b6b7f92 had to fix in
        _solve_subset.  Only ONE definition of __matmul__ / T may exist in this
        class body -- a second one silently shadows the first.
        """
        return _ResidentRowAdjoint(self)


class _ResidentRowAdjoint(object):
    """Adjoint of a _ResidentRowView: V.T @ u evaluates V.rmatvec(u).

    Kept as its own class (rather than a scipy LinearOperator over bound
    methods) so .T round-trips: (V.T).T is V and (V.T).transpose() is V.

    [FIX P50b 2026-09-25] The operator protocol has to be COMPLETE on both
    sides.  A later duplicate __matmul__ on _ResidentRowView had been reduced
    to `self.matvec(other)`, dropping the 2-D column path, and this class only
    forwarded 1-D operands.  Both are reachable from the Gram builder:
    _compute_gram does `A @ I_blk` and `A.T @ A_blk` with 2-D blocks, and the
    FISTA Gram path wraps that call in try/except, so a cached-resident view
    silently fell back to the slow matvec loop ("Gram build failed") instead of
    building G.  Measured on the MgC resident view: `view @ I_blk` raised
    "matmul: Input operand 1 has a mismatch in its core dimension 0".

    _data_dtype keeps the provenance idiom of _row_slice_op/_scale_operator:
    the adjoint multiplies the SAME stored factors, so it must not look like a
    fresh float64 operator to the tolerance floor (_array_precision).
    """

    def __init__(self, view):
        self._view = view
        self.shape = (int(view.shape[1]), int(view.shape[0]))
        self.dtype = getattr(view, "dtype", np.dtype(np.float64))

    @property
    def _data_dtype(self):
        return np.dtype(_array_precision(self._view))

    def __matmul__(self, other):
        arr = np.asarray(other, dtype=np.float64)
        if arr.ndim == 1:
            return self._view.rmatvec(arr)
        if arr.ndim == 2:
            return np.column_stack([self._view.rmatvec(arr[:, j])
                                    for j in range(arr.shape[1])])
        raise ValueError("operator product expects a 1-D or 2-D operand, got %d-D"
                         % arr.ndim)

    dot = __matmul__

    def matvec(self, other):
        return self._view.rmatvec(other)

    def rmatvec(self, other):
        return self._view.matvec(other)

    def transpose(self):
        return self._view

    @property
    def T(self):
        return self._view

def _row_slice(A, rows):
    """A[rows, :] for dense / sparse / LinearOperator.

    [FIX P30] TwoLevelSM gets a true row-slice (slices SM_prime) so CV-fold
    matvecs cost O(nnz(SM_prime[rows])) instead of the full O(nnz(SM_prime)).

    When the operator already owns an uploaded resident op, slice as a VIEW over
    it instead (see _ResidentRowView): re-slicing the host factors and uploading
    them again per fold was the dominant cost of a resident RIDGE CV.
    """
    resident = getattr(A, "_gpu_ridge_op", None)
    if resident is not None:
        return _ResidentRowView(A, resident, rows)
    if hasattr(A, "_twolevel_base"):
        return _scale_operator(A._twolevel_base.row_slice(rows), A._twolevel_scale)
    if hasattr(A, "row_slice"):     # TwoLevelSM
        return A.row_slice(rows)
    if _is_linear_operator(A):
        return _row_slice_op(A, rows)
    return A[rows]


def _resident_ridge_op(A):
    """The resident operator for A: cached, or freshly uploaded (base + scaling).

    Single place that knows how a column-scaled two-level operator maps onto
    GpuTwoLevelOperator: hand it the FACTOR carrier (a _CustomLinearOperator
    wrapper has no SM_prime) and express the scaling as input_scale, which the
    matvec divides by.  Passing the wrapper raised inside the constructor and the
    surrounding except swallowed it, so every --std fold silently ran on CPU.
    input_scale is (re)applied on every call so a moved/cached op stays correct.
    """
    from . import gpu_backend as _gb
    op = getattr(A, "_gpu_ridge_op", None)
    if op is None:
        base = getattr(A, "_twolevel_base", A)
        # Hand over the canonical host factors (one tocsr copy plus one full
        # transpose, built once and kept while host RAM allows): from RAW
        # factors __init__ pays one O(nnz) column-slice transpose PER SHARD for
        # the adjoint.  Measured on the MgC operator (2 shards): 456.9 s of a
        # 1298.8 s full-scale RIDGE fit, i.e. 35% of the fit in host transposes.
        op = _gb.GpuTwoLevelOperator(
            base, device_ids=_gb.resident_device_ids(),
            host_factors=_gb.twolevel_host_factors(base))
        try:
            A._gpu_ridge_op = op
        except Exception:
            pass          # read-only operator: fall back to rebuild
    _w = getattr(A, "_twolevel_scale", None)
    if _w is not None:
        op.input_scale = op.torch.as_tensor(
            np.asarray(_w, dtype=np.float64),
            dtype=op._value_dtype, device=op.device)
    return op


def _ridge_solve(A, y, alpha, x0=None):
    """min ||A x - y||^2 + alpha||x||^2 for dense / sparse / LinearOperator.

    x0 is a warm-start for the iterative (LinearOperator) branch: the alpha
    path walks large->small, so neighbouring alphas have nearly identical
    solutions and LSMR converges in a fraction of the cold-start iterations.
    """
    y64 = np.asarray(y, dtype=np.float64).ravel()
    _parent_view = getattr(A, "_gpu_ridge_parent", None)
    if _parent_view is not None:
        # Fold solve THROUGH the parent's already-uploaded factors: see
        # _ResidentRowView.  No per-fold upload, no per-fold host transpose.
        _resident, _cols, _rows, _host = _parent_view
        try:
            from . import gpu_backend as _gb
            _view = _gb.GpuSubsetOperator(_resident, _cols, _rows)
            _yt = _resident.torch.as_tensor(
                y64, dtype=_view._value_dtype, device=_view.device)
            _coef, _info = _gb._iterative_ridge_tensor(
                _view, _yt, float(alpha),
                atol=float(_lsmr_tol("PHEASY_LSQR_ATOL", 1e-8, _host)),
                btol=float(_lsmr_tol("PHEASY_LSQR_BTOL", 1e-8, _host)),
                maxiter=int(os.environ.get("PHEASY_LSQR_MAXITER", "5000")),
                x0=x0)
            _info = _iterative_solver_info(_info, "GPU CGLS-RIDGE")
            _info["alpha"] = float(alpha)
            try:
                A._gpu_solver_info = _info
                A._ridge_solver_info = _info
            except Exception:
                pass
            return np.asarray(_coef.detach().cpu().numpy(),
                              dtype=np.float64).ravel()
        except Exception as _e:
            if _gpu_required():
                raise RuntimeError("GPU ridge fold solve failed with fallback "
                                   "disabled: %s" % _e) from _e
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
                    # Hand GpuTwoLevelOperator the FACTOR carrier, not the wrapper.
                    # A column-scaled operator (--std -> _scale_columns ->
                    # _scale_operator) exposes only _twolevel_base and _twolevel_scale,
                    # so passing the wrapper raised inside the constructor, the
                    # enclosing except swallowed it and EVERY --std CV fold ran the
                    # CPU LSMR branch in silence -- the opposite of the fail-closed
                    # contract.  The wrapper's effective operator is exactly the base
                    # with that column scaling, which GpuTwoLevelOperator expresses as
                    # input_scale (matvec divides by scale*input_scale).
                    gpu_op = _resident_ridge_op(A)
                    # x0 is the previous alpha's solution.  The resident branch
                    # used to drop it while the CV folds and the L-curve walk kept
                    # computing and passing one, so the warm start the CPU branch
                    # documents was silently absent on the GPU path.
                    coef, info = _gb.iterative_ridge(
                        gpu_op, y64, alpha,
                        atol=float(_lsmr_tol("PHEASY_LSQR_ATOL", 1e-8, A)),
                        btol=float(_lsmr_tol("PHEASY_LSQR_BTOL", 1e-8, A)),
                        maxiter=int(os.environ.get("PHEASY_LSQR_MAXITER", "5000")),
                        x0=x0)
                    info = _iterative_solver_info(info, "GPU CGLS-RIDGE")
                    # alpha rides along so a reader can tell which solve of this
                    # operator a certificate belongs to (the L-curve walk leaves
                    # one per alpha on the same object).
                    info["alpha"] = float(alpha)
                    A._gpu_solver_info = info
                    A._ridge_solver_info = info
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
        # The floor is a property of the DATA, not of the arithmetic: this branch
        # runs in float64 but on whatever factors the operator holds, and for a
        # float32 two-level operator the achievable relative normal residual is
        # bounded by eps32, not eps64.  Without A the floor never applied here and
        # the 1e-8 default was simply unreachable, which is what turns into
        # istop=7 at the iteration cap.
        atol = float(_lsmr_tol("PHEASY_LSQR_ATOL", 1e-8, A))
        btol = float(_lsmr_tol("PHEASY_LSQR_BTOL", 1e-8, A))
        maxiter = int(os.environ.get("PHEASY_LSQR_MAXITER", "5000"))
        import time as _tr
        _t_r = _tr.time()
        res = _lsmr(op, y_aug, atol=atol, btol=btol, maxiter=maxiter, x0=x0)
        # Certify the CPU branch too.  The return value used to be discarded, so a
        # RIDGE fit that reached its iteration cap on CPU reported
        # fit_accepted=True while the identical solve on GPU raised.  The alpha is
        # recorded with it so a caller can tell which solve a certificate belongs
        # to (the L-curve walk leaves several on the same operator object).
        A._ridge_solver_info = dict(_iterative_solver_info(res, "LSMR"),
                                    alpha=float(alpha), backend="cpu_lsmr_ridge")
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
        # Route through _lsmr_tol WITH the operator: reading the environment
        # here directly was the one remaining path where the request never met
        # the precision floor, so a float32-factored TwoLevelSM got a 1e-8 it
        # cannot reach and every full-support RFE subset solve ran to istop=7.
        # op is still the un-augmented operator at this point, which is the one
        # whose stored precision bounds the solve.
        atol = float(_lsmr_tol("PHEASY_LSQR_ATOL",
                               (lsmr_atol if lsmr_atol is not None else 1e-8), op))
        btol = float(_lsmr_tol("PHEASY_LSQR_BTOL",
                               (lsmr_btol if lsmr_btol is not None else 1e-8), op))
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

    def _matmat(self, X):
        """SM @ X for a dense k-column block.

        Without this, scipy's LinearOperator falls back to k separate _matvec
        calls, which is what made the P-free Gram build on the MgC operators
        take minutes (each call rebuilds t = NS @ v).  One sparse product per
        factor with a multi-column RHS is the efficient form.
        """
        X = np.ascontiguousarray(X, dtype=self._dt)
        t = np.asarray(self.NS @ X, dtype=self._dt)
        return np.asarray(self.SM_prime @ t, dtype=self._dt)

    def _rmatmat(self, U):
        """SM.T @ U for a dense k-column block."""
        U = np.ascontiguousarray(U, dtype=self._dt)
        t = np.asarray(self.SM_primeT @ U, dtype=self._dt)
        return np.asarray(self.NST @ t, dtype=self._dt)

    def _col_norms_on_gpu(self, gpu_op=None):
        """Exact column norms from the resident CUDA operator, or None.

        The CPU loop below is one bounded sparse-sparse product per row block.
        On the MgC operator (36864x69487) it dominated the whole fit: profiled
        at 125.5 s of RIDGE's 154.4 s and 197.9 s of OLS's 219.9 s (81% and
        90%), while the actual iterations cost 4.9 s and 16.2 s.  The resident
        operator already implements the same computation as blocked basis
        matvecs (GpuTwoLevelOperator.col_norms), so use it.

        The operator built here is CACHED as self._gpu_ridge_op, which is the
        same slot the resident solvers look in -- so the factor upload is paid
        once and the solve that follows reuses it.  Passing gpu_op (the OLS
        Jacobi path already holds one) avoids even that.
        """
        if getattr(self, "_twolevel_scale", None) is not None:
            return None          # scaled wrapper: keep the CPU semantics
        try:
            from . import gpu_backend as _gb
        except Exception:
            return None
        try:
            # Probing the mode must never become the failure: under required GPU
            # mode without CUDA, enabled() raises, and letting that escape here
            # replaced the solver's own (correct, tested) error surface with a
            # bare "GPU mode is required but CUDA is unavailable".
            if not (_gb.enabled() and _gb.available()):
                return None
        except Exception:
            return None
        op = gpu_op if gpu_op is not None else getattr(self, "_gpu_ridge_op", None)
        if op is None:
            try:
                # Same canonical host factors as _resident_ridge_op: this call is
                # usually the FIRST resident construction in a fit, so it is the
                # one that pays the host passes (see twolevel_host_factors).
                op = _gb.GpuTwoLevelOperator(
                    self, device_ids=_gb.resident_device_ids(),
                    host_factors=_gb.twolevel_host_factors(self))
            except Exception as _e:
                # [FIX colnorms-A] The GPU path here only accelerates a
                # preprocessing step; the solve itself still runs on the GPU.
                # Aborting the whole fit because this optional resident operator
                # does not fit (it is budgeted as a per-CARD replica while the
                # sharded two-level factors already occupy that card) made every
                # large third-order fit die under required GPU mode.  Use the
                # bounded CPU loop this method exists to accelerate instead.
                print("[optimizer] GPU column norms unavailable (%s); using the "
                      "bounded CPU column-norm loop instead" % _e, flush=True)
                return None
            try:
                self._gpu_ridge_op = op
            except Exception:
                pass
        try:
            norms = op.col_norms()
            return np.asarray(norms.detach().cpu().numpy(), dtype=np.float64).ravel()
        except Exception as _e:
            # [FIX colnorms-B] Same rationale as [FIX colnorms-A]: an
            # accelerator that cannot be allocated must not abort the fit.
            print("[optimizer] GPU column norms unavailable (%s); using the "
                  "bounded CPU column-norm loop instead" % _e, flush=True)
            return None

    def col_norms(self, block_rows=None, gpu_op=None):
        """Exact column norms from bounded row blocks of SM_prime @ NS.

        The generic LinearOperator implementation needs one full SpMV per
        column. A sparse row-block product computes all columns together and
        discards each block after accumulating its squared entries in float64.
        The default 64 MiB block budget allows 24 bytes per possible output
        entry (float64 values, sparse indices, and the squaring temporary),
        plus the sliced input block. NS is shared when already float64.

        The resident CUDA operator is used when it is available (see
        _col_norms_on_gpu); the CPU loop below stays as the fallback and as the
        definition of the result.
        """
        cached = getattr(self, "_col_norms_cache", None)
        if cached is not None and gpu_op is None:
            return cached.copy()
        norms = self._col_norms_on_gpu(gpu_op)
        if norms is not None and norms.shape == (self.shape[1],):
            try:
                self._col_norms_cache = norms
            except Exception:
                pass
            return norms.copy()
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
    """A[rows] @ coef for dense / sparse / LinearOperator.

    Slicing FIRST is not a detail: the grouped CV calls this once per
    (alpha, fold), and a full-row matvec on a two-level operator costs
    O(nnz(SM_prime)) however small the validation fold is.  On the MgC
    operator (454656 rows) the uncounted CV path dominated the host time that
    the wall/CPU/GPU audit attributed to RIDGE.  row_slice exists on TwoLevelSM
    (and on the resident views, which slice SM_prime / mask rows on device), so
    use it when present and keep the full matvec as the fallback.
    """
    rows = np.asarray(rows, dtype=np.intp)
    slicer = getattr(A, "row_slice", None)
    if callable(slicer):
        try:
            return np.asarray(slicer(rows) @ coef, dtype=np.float64).ravel()
        except Exception:
            pass          # keep the exact old behaviour as the fallback
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
                                        warn_nonconvergence=False,
                                        auto_floor=False)
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
                                        _info=info, warn_nonconvergence=False,
                                        auto_floor=False)
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
                                      warn_nonconvergence=False,
                                      auto_floor=False)
            else:
                x_full = _fista_lasso(A, y64, alpha, x0=x_full,
                                      max_iter=cv_max_iter, tol=cv_tol,
                                      lipschitz=self._lipschitz,
                                      penalty_weights=self.penalty_weights,
                                      _info=_cv_info, warn_nonconvergence=False,
                                      auto_floor=False)
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
        # This backend iterated too, and it used to be the only LASSO path that
        # published no certificate at all: the FISTA backends (resident, GPU
        # Gram, CPU iterative) all report converged/kkt_relative and the
        # acceptance gate vetoes on it, while a coordinate descent that stopped
        # at max_iter still read as fit_accepted=True.  sklearn's own signal is
        # n_iter_ == max_iter plus a ConvergenceWarning -- a warning is not a
        # gate.  sklearn optimizes over the alpha path and reports n_iter_ for
        # the SELECTED alpha, which is the solve whose coefficients are kept.
        _hit_cap = bool(self.n_iter_ >= int(self.max_iter))
        self.regularized_solver_info_ = {
            "solver": "sklearn LassoCV (coordinate descent)",
            "backend": "cpu_dense_coordinate_descent",
            "converged": not _hit_cap,
            "stop_reason": "iteration_limit" if _hit_cap else "converged",
            "n_iter": self.n_iter_,
            "maxiter": int(self.max_iter),
            "tol": float(self.tol),
        }
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
        # Same certificate as the plain dense LASSO, and for the same reason: the
        # adaptive branch runs its OWN LassoCV on the weighted matrix and used to
        # publish nothing at all -- measured on c7 it reported
        # regularized_solver_info={} and execution_backend=None, so the whole
        # adaptive solve was invisible to the acceptance gate while the resident
        # FISTA equivalent vetoed on convergence.
        _hit_cap = bool(self.n_iter_ >= int(self.max_iter))
        self.regularized_solver_info_ = {
            "solver": "sklearn LassoCV on the weighted matrix (coordinate descent)",
            "backend": "cpu_dense_coordinate_descent",
            "converged": not _hit_cap,
            "stop_reason": "iteration_limit" if _hit_cap else "converged",
            "n_iter": self.n_iter_,
            "maxiter": int(self.max_iter),
            "tol": float(self.tol),
            "weight_dispersion": float(np.std(np.log(np.maximum(self._weights,
                                                             1e-300)))),
            "beta0_floor_fraction": float(getattr(self, "_beta0_floor", float("nan"))),
        }
        return self

    def predict(self, A):
        pred = np.asarray(A @ self.coef_).ravel()
        if self.fit_intercept:
            pred = pred + self.intercept_
        return pred


def _ardr_evidence_gram(G, b, yty, n_samples, y_var, threshold_lambda=1e4,
                        max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6,
                        lambda_1=1e-6, lambda_2=1e-6, gpu=None, verbose=False,
                        checkpoint=None):
    """ARD evidence maximization driven by the Gram matrix alone.

    sklearn.linear_model.ARDRegression's update touches only G = X^T X,
    b = X^T y and yty = y^T y: _update_sigma inverts
    lambda_keep * I + alpha * X_keep^T X_keep, update_coeff multiplies that
    inverse by X_keep^T y, and sse = ||y - X coef||^2 is the identical Gram
    quadratic form.  This function reproduces that loop exactly, so the
    n_samples x n_features design matrix never has to be materialized -- which
    is what lets ARDR run on a large sensing matrix whose rows do not fit.

    Only the linear-algebra backend differs from sklearn: scipy.linalg.pinvh
    on the CPU (the same routine sklearn calls) and torch Cholesky when a GPU
    backend is supplied.  The hard limit that remains is the p x p Gram itself:
    O(p^2) memory and O(p^3) per evidence iteration.

    Returns (coef, alpha, lambda_, n_iter, stop_reason, sse).
    """
    import numpy as np
    p = int(G.shape[0])
    n = int(n_samples)
    yty = float(yty)
    y_var = float(y_var)
    tiny = float(np.finfo(np.float64).tiny)
    eps = float(np.finfo(np.float64).eps)
    thr = float(threshold_lambda)
    # Gram-form sse suffers catastrophic cancellation when the fit is
    # near-perfect (many features, few samples): yty and 2 b.c - c^T G c cancel
    # to below machine precision, sse clamps to tiny, and alpha = n/sse explodes
    # (observed on the 2+3 c3=3.5 N=40 fit).  Floor it at a relative residual of
    # 1e-6, which is below any physical force noise and only guards the update.
    _sse_floor = max(tiny, 1e-12 * float(yty))
    if p == 0:
        return (np.zeros(0, dtype=np.float64), 1.0, np.zeros(0, dtype=np.float64),
                0, "pruned_empty", 0.0)

    def _reason(converged, exhausted, keep_any):
        if converged:
            return "converged"
        if not keep_any:
            return "pruned_empty"
        if exhausted:
            return "iteration_limit"
        return "pruned_empty"

    if gpu is not None:
        import torch
        dev = gpu.device()
        Gt = gpu._to_torch(np.asarray(G, dtype=np.float64), torch.float64)
        bt = gpu._to_torch(np.asarray(b, dtype=np.float64), torch.float64)
        lam = torch.ones(p, dtype=torch.float64, device=dev)
        keep = torch.ones(p, dtype=torch.bool, device=dev)
        coef = torch.zeros(p, dtype=torch.float64, device=dev)
        alpha_ = 1.0 / (y_var + eps)
        coef_old = None
        converged = False
        exhausted = False
        n_iter = 0
        for it in range(int(max_iter)):
            idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                n_iter = it
                break
            Gk = Gt.index_select(0, idx).index_select(1, idx)
            A = alpha_ * Gk
            A.diagonal().add_(lam.index_select(0, idx))
            try:
                Ainv = torch.cholesky_inverse(torch.linalg.cholesky(A))
            except Exception:
                Ainv = torch.linalg.pinv(A)
            lam_k = lam.index_select(0, idx)
            bk = bt.index_select(0, idx)
            ck = alpha_ * (Ainv @ bk)
            sse = max(yty - 2.0 * float(ck @ bk) + float(ck @ (Gk @ ck)), _sse_floor)
            gamma = 1.0 - lam_k * Ainv.diagonal()
            lam = lam.index_copy(0, idx, (gamma + 2.0 * lambda_1) / (ck * ck + 2.0 * lambda_2))
            alpha_ = (n - float(gamma.sum()) + 2.0 * alpha_1) / (sse + 2.0 * alpha_2)
            coef = coef * keep
            coef = coef.index_copy(0, idx, ck)
            keep = lam < thr
            coef = coef * keep
            n_iter = it + 1
            if verbose:
                print("[ARDR]   iter %d: active=%d sse=%.6e alpha=%.4e"
                      % (it, int(idx.numel()), sse, alpha_), flush=True)
            if it > 0 and float((coef_old - coef).abs().sum()) < float(tol):
                converged = True
                break
            coef_old = coef.clone()
            if not bool(keep.any()):
                break
        else:
            exhausted = True
        if bool(keep.any()):
            idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
            Gk = Gt.index_select(0, idx).index_select(1, idx)
            A = alpha_ * Gk
            A.diagonal().add_(lam.index_select(0, idx))
            try:
                Ainv = torch.cholesky_inverse(torch.linalg.cholesky(A))
            except Exception:
                Ainv = torch.linalg.pinv(A)
            bk = bt.index_select(0, idx)
            ck = alpha_ * (Ainv @ bk)
            coef = torch.zeros(p, dtype=torch.float64, device=dev).index_copy(0, idx, ck)
            sse = max(yty - 2.0 * float(ck @ bk) + float(ck @ (Gk @ ck)), _sse_floor)
        else:
            coef = torch.zeros(p, dtype=torch.float64, device=dev)
            sse = yty
        return (gpu._to_numpy(coef, np.float64), float(alpha_),
                gpu._to_numpy(lam, np.float64), int(n_iter),
                _reason(converged, exhausted, bool(keep.any())), float(sse))

    _force_pinvh = os.environ.get("PHEASY_ARDR_PINVH", "0").lower() in ("1", "true", "yes", "on")

    def _sigma_diag_and_solve(A, bk):
        """(diag(A^-1), A^-1 bk) for symmetric PD A.

        alpha*G + diag(lambda) is PD (lambda > 0), so Cholesky is exact and is
        ~100x faster than pinvh's eigendecomposition (measured p=5000: 0.4s vs
        41.6s; p=20000: 16s vs ~44 min).  diag(A^-1) via LAPACK dpotri is then
        ~4x faster than solve_triangular(L, I) (p=20000: 30s vs 120s) and uses
        half the workspace.  sklearn uses pinvh, so PHEASY_ARDR_PINVH=1 restores
        that path for exact parity checks.
        """
        if not _force_pinvh:
            try:
                c, lower = spla.cho_factor(A, check_finite=False)
                sigma_bk = spla.cho_solve((c, lower), bk, check_finite=False)
                Ainv, info = spla.lapack.dpotri(c, lower=lower, overwrite_c=1)
                if info == 0:
                    return np.diag(Ainv), sigma_bk
            except (spla.LinAlgError, ValueError):
                pass
        sigma = spla.pinvh(A, check_finite=False)
        return np.diag(sigma), sigma @ bk

    alpha_ = 1.0 / (y_var + eps)
    lam = np.ones(p, dtype=np.float64)
    keep = np.ones(p, dtype=bool)
    coef = np.zeros(p, dtype=np.float64)
    coef_old = None
    converged = False
    exhausted = False
    n_iter = 0
    start_it = 0
    if checkpoint and os.path.exists(checkpoint):
        # Resume a long run across process restarts.  Only the O(p) state is
        # saved (coef, lambda, alpha), never the p x p Gram.
        _z = np.load(checkpoint, allow_pickle=True)
        coef = np.asarray(_z["coef"], dtype=np.float64)
        lam = np.asarray(_z["lam"], dtype=np.float64)
        alpha_ = float(_z["alpha_"])
        if "coef_old" in _z.files:
            coef_old = np.asarray(_z["coef_old"], dtype=np.float64)
        start_it = int(_z["it"]) + 1
        keep = lam < thr
        print("[ARDR] resumed from %s at iteration %d (active=%d)"
              % (checkpoint, start_it, int(keep.sum())), flush=True)
    for it in range(start_it, int(max_iter)):
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            n_iter = it
            break
        Gk = np.asarray(G[np.ix_(idx, idx)], dtype=np.float64)
        A = Gk
        A *= alpha_
        A[np.diag_indices_from(A)] += lam[idx]
        lam_k = lam[idx]
        bk = np.asarray(b[idx], dtype=np.float64)
        diag_sigma, sigma_bk = _sigma_diag_and_solve(A, bk)
        ck = alpha_ * sigma_bk
        # sse = yty - 2 ck.bk + ck.(Gk @ ck) == yty - ck.bk - alpha * lam.sigma^2
        # (Gk @ ck = bk - diag(lam) @ sigma_bk), avoids keeping the original Gk alive
        # after the in-place scale, halving the p x p peak for the full-active phase.
        sse = max(yty - float(ck @ bk) - alpha_ * float(lam_k @ (sigma_bk ** 2)), _sse_floor)
        gamma = 1.0 - lam_k * diag_sigma
        lam[idx] = (gamma + 2.0 * lambda_1) / (ck * ck + 2.0 * lambda_2)
        alpha_ = (n - float(gamma.sum()) + 2.0 * alpha_1) / (sse + 2.0 * alpha_2)
        coef[:] = 0.0
        coef[idx] = ck
        keep = lam < thr
        coef[~keep] = 0.0
        n_iter = it + 1
        if verbose:
            print("[ARDR]   iter %d: active=%d sse=%.6e alpha=%.4e"
                  % (it, int(idx.size), sse, alpha_), flush=True)
        if it > 0 and float(np.sum(np.abs(coef_old - coef))) < float(tol):
            converged = True
            break
        coef_old = coef.copy()
        if not keep.any():
            break
        if checkpoint:
            np.savez(checkpoint, coef=coef, lam=lam, alpha_=alpha_,
                     coef_old=coef_old, it=it)
    else:
        exhausted = True
    if keep.any():
        idx = np.flatnonzero(keep)
        Gk = np.asarray(G[np.ix_(idx, idx)], dtype=np.float64)
        A = Gk
        A *= alpha_
        A[np.diag_indices_from(A)] += lam[idx]
        bk = np.asarray(b[idx], dtype=np.float64)
        diag_sigma, sigma_bk = _sigma_diag_and_solve(A, bk)
        ck = alpha_ * sigma_bk
        coef[:] = 0.0
        coef[idx] = ck
        sse = max(yty - float(ck @ bk) - alpha_ * float(lam[idx] @ (sigma_bk ** 2)), _sse_floor)
    else:
        coef[:] = 0.0
        sse = yty
    return (coef, float(alpha_), lam, int(n_iter),
            _reason(converged, exhausted, bool(keep.any())), float(sse))


class _ARDRModel:
    """Automatic Relevance Determination Regression (ARDR).

    Bayesian linear regression with one precision hyperparameter per
    coefficient.  Evidence maximization drives the precision of irrelevant
    coefficients above 'threshold_lambda'; those coefficients are then pruned
    to exactly zero, so the model selects features without an explicit
    elimination schedule (MacKay, Neural Comput. 4 (1992) 415; Tipping 2001).

    This is a grouped-CV wrapper around scikit-learn's ARDRegression.
    Fransson, Eriksson & Erhart, npj Comput. Mater. 6, 135 (2020) compared
    ARDR against OLS / LASSO / RFE-OLS for force-constant models using exactly
    the scikit-learn implementation with pruning threshold lambda_t = 1e4
    (scikit-learn's default threshold_lambda); that is the default here.

    Two fit paths share the same evidence loop:

    * dense / sparse input (default) -- sklearn ARDRegression, which needs the
      full X;
    * matrix-free Gram mode (automatic for a TwoLevelSM / LinearOperator, or
      PHEASY_ARDR_GRAM=1) -- the update runs directly on G = X^T X, b = X^T y
      and y^T y, so the n_samples x n_features design matrix is never built.
      The O(p^2) Gram and the O(p^3) per-iteration solve remain: that is ARD's
      hard wall, the unfavourable scaling reported in the reference paper.
      PHEASY_ARDR_GPU=1 moves that solve to torch Cholesky on the device.
    """

    def __init__(self, threshold_lambda=1e4, thresholds=None, cv=5,
                 max_iter=300, tol=1e-3, fit_intercept=False, rand_seed=None,
                 group_size=None, alpha_1=1e-6, alpha_2=1e-6,
                 lambda_1=1e-6, lambda_2=1e-6, n_jobs=None):
        if thresholds is None:
            self.thresholds = np.asarray([float(threshold_lambda)], dtype=np.float64)
        else:
            self.thresholds = np.asarray(list(thresholds), dtype=np.float64)
            if self.thresholds.size == 0:
                raise ValueError("ARDR thresholds must be non-empty")
        self.cv = int(cv)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.fit_intercept = bool(fit_intercept)
        self.rand_seed = rand_seed
        self.group_size = group_size
        self.alpha_1 = float(alpha_1)
        self.alpha_2 = float(alpha_2)
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.n_jobs = n_jobs
        # CV of the full-data fit is always evaluated for the report; when more
        # than one threshold is supplied the CV also SELECTS the threshold.
        self.threshold_ = float(self.thresholds[0])
        self.mse_path_ = None
        self.best_rmse_cv_ = float("nan")

    def _new_model(self, threshold):
        import inspect
        from sklearn.linear_model import ARDRegression
        kwargs = dict(
            tol=self.tol,
            alpha_1=self.alpha_1, alpha_2=self.alpha_2,
            lambda_1=self.lambda_1, lambda_2=self.lambda_2,
            threshold_lambda=float(threshold),
            fit_intercept=self.fit_intercept, copy_X=True)
        # scikit-learn >= 1.3 names the iteration cap 'max_iter'; older
        # releases (the pyproject floor is 1.1) name it 'n_iter'.
        if "max_iter" in inspect.signature(ARDRegression.__init__).parameters:
            kwargs["max_iter"] = self.max_iter
        else:
            kwargs["n_iter"] = self.max_iter
        return ARDRegression(**kwargs)

    def fit(self, A, y, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError("ARDR does not support sample weights")
        y64 = np.asarray(y, dtype=np.float64).ravel()
        # Matrix-free Gram mode: the ARD update only needs X^T X, X^T y and
        # y^T y, so a TwoLevelSM / LinearOperator can be fitted without ever
        # materializing the n_samples x n_features design matrix.
        _gram_env = os.environ.get("PHEASY_ARDR_GRAM")
        if _gram_env is None:
            gram_mode = _is_linear_operator(A)
        else:
            gram_mode = _gram_env.lower() in ("1", "true", "yes", "on")
        if _is_linear_operator(A) and not gram_mode:
            raise NotImplementedError(
                "ARDR received a matrix-free LinearOperator/TwoLevelSM with "
                "PHEASY_ARDR_GRAM=0. Enable the Gram path (the default) or pass "
                "the dense/sparse sensing matrix.")
        if gram_mode:
            return self._fit_gram(A, y64)
        A64 = _to_dense_f64(A)
        n_samples, n_features = A64.shape

        # Memory gate: ARDRegression owns a dense copy of X, the
        # n_features x n_features posterior covariance and its Cholesky
        # factors.  Fail loudly (and name the knob) instead of dying in the
        # allocator or thrashing swap.
        per_feat = 8.0 * n_samples * n_features + 24.0 * n_features ** 2
        budget = float(os.environ.get("PHEASY_ARDR_MAX_GB", "16")) * 1e9
        max_features = int(os.environ.get("PHEASY_ARDR_MAX_FEATURES", "0") or 0)
        if max_features > 0 and n_features > max_features:
            raise NotImplementedError(
                "ARDR: n_features=%d exceeds PHEASY_ARDR_MAX_FEATURES=%d. ARD "
                "regression is O(n_features^3) in time and O(n_features^2) in "
                "memory; raise the cap explicitly if the machine can take it."
                % (n_features, max_features))
        if per_feat > budget:
            raise NotImplementedError(
                "ARDR estimated workspace %.2f GB exceeds PHEASY_ARDR_MAX_GB=%.1f "
                "(n_samples=%d, n_features=%d). ARD regression does not scale to "
                "this design matrix; use RIDGE/LASSO/ALASSO or an explicit cutoff."
                % (per_feat / 1e9, budget / 1e9, n_samples, n_features))

        splits = _make_cv_splits(n_samples, self.cv, self.rand_seed, self.group_size)
        thresholds = self.thresholds
        mse_path = np.full((thresholds.size, len(splits)), np.nan, dtype=np.float64)
        with warnings.catch_warnings():
            # ARDRegression emits a ConvergenceWarning when it hits the
            # iteration cap; that is recorded as converged=False below instead
            # of surfacing as a warning per fold.
            warnings.simplefilter("ignore")
            for j, thr in enumerate(thresholds):
                for k, (tr, va) in enumerate(splits):
                    model = self._new_model(thr)
                    model.fit(A64[tr], y64[tr])
                    pred = np.asarray(A64[va] @ model.coef_).ravel()
                    if self.fit_intercept:
                        pred = pred + model.intercept_
                    mse_path[j, k] = float(np.mean((pred - y64[va]) ** 2))
            best_idx = int(np.argmin(mse_path.mean(axis=1)))
            self.threshold_ = float(thresholds[best_idx])
            model = self._new_model(self.threshold_)
            model.fit(A64, y64)

        self.model_ = model
        self.coef_ = np.asarray(model.coef_, dtype=np.float64)
        self.intercept_ = float(model.intercept_)
        self.alpha_ = float(model.alpha_)
        self.lambda_ = np.asarray(model.lambda_, dtype=np.float64)
        self.sigma_ = np.asarray(model.sigma_, dtype=np.float64)
        self.n_iter_ = int(model.n_iter_)
        self.n_features_in_ = int(n_features)
        self.mse_path_ = mse_path
        self.best_index_ = best_idx
        self.best_rmse_cv_ = float(np.sqrt(mse_path[best_idx].mean()))
        self.cv_evaluated = True
        hit_cap = self.n_iter_ >= self.max_iter
        self.regularized_solver_info_ = {
            "solver": "sklearn ARDRegression (evidence maximization)",
            "backend": "cpu_dense_ardr",
            "converged": not hit_cap,
            "stop_reason": "iteration_limit" if hit_cap else "converged",
            "n_iter": self.n_iter_,
            "maxiter": int(self.max_iter),
            "tol": float(self.tol),
            "threshold_lambda": self.threshold_,
        }
        return self

    def _fit_gram(self, A, y64):
        """Matrix-free ARD fit driven by the sensing matrix's Gram.

        Runs the exact sklearn ARD evidence loop on G = X^T X, b = X^T y and
        yty = y^T y, so no n_samples x n_features matrix is ever allocated.
        The remaining cost is the p x p Gram (O(p^2) memory) and the O(p^3)
        per-iteration solve; a GPU backend moves that solve to torch Cholesky.
        """
        if self.thresholds.size != 1:
            warnings.warn(
                "ARDR Gram mode fits one pruning threshold (the paper's "
                "lambda_t = 1e4); ignoring PHEASY_ARDR_THRESHOLDS and using "
                "%.4g." % float(self.thresholds[0]), RuntimeWarning, stacklevel=3)
        thr = float(self.thresholds[0])
        base = getattr(A, "_twolevel_base", None)
        scale = getattr(A, "_twolevel_scale", None)
        _gram_gb = float(os.environ.get(
            "PHEASY_ARDR_GRAM_MAX_GB", os.environ.get("PHEASY_GRAM_MAX_GB", "4")))
        print("[ARDR] building the design Gram (matrix-free; "
              "PHEASY_ARDR_GRAM_MAX_GB=%.1f)..." % _gram_gb, flush=True)
        import time as _t
        _t_gram0 = _t.time()
        G, b, _gram_how = _build_gram_matrix(A, y64, budget_gb=_gram_gb)
        G = np.asarray(G, dtype=np.float64)
        print("[ARDR] design Gram ready: %dx%d (%.2f GB) in %.1fs"
              % (G.shape[0], G.shape[1], G.nbytes / 1e9, _t.time() - _t_gram0),
              flush=True)
        n_samples = int(y64.shape[0])
        yty = float(y64 @ y64)
        y_var = float(np.var(y64))

        _gpu_env = os.environ.get("PHEASY_ARDR_GPU")
        if _gpu_env is not None:
            want_gpu = _gpu_env.lower() in ("1", "true", "yes", "on")
        else:
            want_gpu = _gpu() is not None
        gb = _gpu() if want_gpu else None
        if want_gpu and gb is None and _gpu_required():
            raise RuntimeError(
                "ARDR was asked for the GPU backend (or GPU is required) but "
                "CUDA is unavailable")
        p = int(G.shape[0])
        if gb is not None:
            print("[ARDR] matrix-free Gram fit on GPU: G=%dx%d (%.2f GB)"
                  % (p, p, G.nbytes / 1e9), flush=True)
        else:
            print("[ARDR] matrix-free Gram fit on CPU: G=%dx%d (%.2f GB); each "
                  "evidence iteration solves a p x p system (O(p^3))"
                  % (p, p, G.nbytes / 1e9), flush=True)

        # sklearn's 300-iteration default is tuned for modest p.  On a large
        # p the evidence path can still be pruning a handful of coefficients
        # per step when the cap is reached, so Gram mode defaults higher
        # (PHEASY_ARDR_GRAM_MAX_ITER); the dense path is unchanged.
        _gram_max_iter = int(os.environ.get(
            "PHEASY_ARDR_GRAM_MAX_ITER", str(max(self.max_iter, 1000))))
        _gram_tol = float(os.environ.get("PHEASY_ARDR_GRAM_TOL", str(self.tol)))
        coef, alpha_, lam, n_iter, stop_reason, sse = _ardr_evidence_gram(
            G, b, yty, n_samples, y_var, threshold_lambda=thr,
            max_iter=_gram_max_iter, tol=_gram_tol, alpha_1=self.alpha_1,
            alpha_2=self.alpha_2, lambda_1=self.lambda_1, lambda_2=self.lambda_2,
            gpu=gb,
            verbose=os.environ.get("PHEASY_ARDR_VERBOSE", "1").lower()
            in ("1", "true", "yes", "on"))

        self.model_ = None
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = 0.0
        self.alpha_ = float(alpha_)
        self.lambda_ = np.asarray(lam, dtype=np.float64)
        self.sigma_ = np.zeros((0, 0), dtype=np.float64)
        self.n_iter_ = int(n_iter)
        self.n_features_in_ = p
        self.threshold_ = thr
        self.mse_path_ = np.array(
            [[max(sse, 0.0) / max(n_samples, 1)]], dtype=np.float64)
        self.best_index_ = 0
        self.best_rmse_cv_ = float(np.sqrt(self.mse_path_[0, 0]))
        # Gram mode has no hold-out folds: a per-fold Gram costs folds x p^2,
        # and the reference paper uses one fixed threshold. The reported score
        # is therefore the in-sample RMSE and cv_evaluated says so.
        self.cv_evaluated = False
        self.backend_ = "gpu_gram_ardr" if gb is not None else "cpu_gram_ardr"
        self.regularized_solver_info_ = {
            "solver": "ARD evidence maximization on the Gram (matrix-free)",
            "backend": self.backend_,
            "converged": stop_reason in ("converged", "pruned_empty"),
            "stop_reason": stop_reason,
            "n_iter": self.n_iter_,
            "maxiter": int(_gram_max_iter),
            "tol": float(self.tol),
            "threshold_lambda": thr,
            "cv_skipped": "gram_mode",
            "gram_construction": _gram_how,
            "n_features": p,
            "n_samples": n_samples,
        }
        return self

    def predict(self, A):
        pred = np.asarray(A @ self.coef_).ravel()
        if self.fit_intercept:
            pred = pred + self.intercept_
        return pred


class _RVMModel:
    """Result holder for the fast marginal-likelihood RVM fit."""

    def __init__(self, coef, active, alpha, beta, n_iter, n_features, rss,
                 n_samples, converged):
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = 0.0
        self.n_features_in_ = int(n_features)
        self.n_iter_ = int(n_iter)
        self.active_ = np.asarray(active, dtype=np.int64)
        self.alpha_ = np.asarray(alpha, dtype=np.float64)
        self.beta_ = float(beta)
        self.cv_evaluated = False
        self.mse_path_ = np.array(
            [[max(float(rss), 0.0) / max(int(n_samples), 1)]], dtype=np.float64)
        self.best_index_ = 0
        self.best_rmse_cv_ = float(np.sqrt(self.mse_path_[0, 0]))
        self.regularized_solver_info_ = {
            "solver": "fast marginal-likelihood RVM (Tipping & Faul 2003)",
            "backend": "cpu_gram_rvm",
            "converged": bool(converged),
            "stop_reason": "evidence_maximum" if converged else "step_limit",
            "n_iter": int(n_iter),
            "n_active": int(self.active_.size),
            "beta": float(beta),
            "cv_skipped": "gram_mode",
        }

    def predict(self, A):
        return np.asarray(A @ self.coef_).ravel()


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


def _materialize_columns(A, col_idx):
    """[PATCH rfe-final-tsqr-v2] A[:, col_idx] as scipy sparse / ndarray, or None."""
    if isinstance(A, TwoLevelSM):
        NSs = A.NS[:, col_idx]
        if not sp.issparse(NSs):
            NSs = sp.csr_matrix(NSs)
        return (A.SM_prime @ NSs).tocsr()
    if sp.issparse(A):
        return A[:, col_idx].tocsr()
    if isinstance(A, np.ndarray):
        return A[:, col_idx]
    return None


class _RowBlockProduct:
    """[PATCH rfe-final-tsqr-v2] Streaming row blocks of SM_prime @ NS_sub.

    A CSR A_sub for a 30k-feature support is ~38 GB and holding it while the
    block QR allocates its own dense blocks OOM-killed the box.  This view
    computes each requested row block on demand so the full product is never
    resident.
    """

    def __init__(self, prime, ns):
        self.prime = prime
        self.ns = ns
        self.shape = (int(prime.shape[0]), int(ns.shape[1]))

    def __getitem__(self, sl):
        return self.prime[sl] @ self.ns


def _rfe_final_refit_exact(A, y, best_idx, *, block_rows=None, diag_floor=1e-12,
                           iterative_diagnostics=None, n_samples=None):
    """[PATCH rfe-final-tsqr-v3] Exact OLS on the selected support.

    Default method "gram" accumulates the normal equations G = A^T A in float64
    over row blocks, streaming from SM_prime @ NS, so the peak is ~G + one dense
    block (~25 GB for 30k features).  The Q-less TSQR needs the R factors plus a
    merged Q (~120 GB) and OOM-killed the box at this size; it is still available
    via PHEASY_RFE_FINAL_METHOD=tsqr.  Both are unregularized OLS (no ridge bias).
    Returns the coefficient vector or None (caller falls back to LSMR).
    """
    y64 = np.asarray(y, dtype=np.float64).ravel()
    m = int(A.shape[0])
    n = int(len(best_idx))
    blk = int(block_rows) if block_rows else int(
        os.environ.get("PHEASY_RFE_FINAL_BLOCK_ROWS", "20000"))
    method = os.environ.get("PHEASY_RFE_FINAL_METHOD", "gram").lower()

    # --- row-block source: sparse slice getter -------------------------------
    if isinstance(A, TwoLevelSM):
        try:
            NSs = A.NS[:, best_idx]
            if not sp.issparse(NSs):
                NSs = sp.csr_matrix(NSs)
        except Exception as exc:
            print("[RFE] final exact refit: NS slice failed (%s); LSMR fallback"
                  % type(exc).__name__, flush=True)
            return None
        prime = A.SM_prime
        def _rows(i0, i1):
            return prime[i0:i1] @ NSs
    else:
        try:
            A_full = _materialize_columns(A, best_idx)
        except Exception as exc:
            print("[RFE] final exact refit: materialization failed (%s); LSMR fallback"
                  % type(exc).__name__, flush=True)
            return None
        if A_full is None:
            return None
        def _rows(i0, i1):
            return A_full[i0:i1]

    if method == "tsqr":
        print("[RFE] final exact refit: streaming TSQR (%d x %d), blocks of %d"
              % (m, n, blk), flush=True)
        try:
            class _GetterBlocks(object):
                def __init__(self, getter, shape):
                    self._getter = getter
                    self.shape = shape

                def __getitem__(self, sl):
                    return self._getter(sl.start, sl.stop)
            coef, ok, _cond = _tsqr_qless(_GetterBlocks(_rows, (m, n)), y64, blk, diag_floor)
            if (not ok) or coef is None:
                print("[RFE] final exact refit: TSQR rank-deficient; LSMR fallback",
                      flush=True)
                return None
        except Exception as exc:
            print("[RFE] final exact refit: TSQR failed (%s); LSMR fallback"
                  % type(exc).__name__, flush=True)
            return None
    else:
        print("[RFE] final exact refit: Gram normal equations (%d x %d), blocks of %d"
              % (m, n, blk), flush=True)
        G = np.zeros((n, n), dtype=np.float64)
        rhs = np.zeros(n, dtype=np.float64)
        try:
            for i0 in range(0, m, blk):
                i1 = min(i0 + blk, m)
                B = _rows(i0, i1)
                Bd = np.asarray(B.toarray() if sp.issparse(B) else B, dtype=np.float64)
                G += Bd.T @ Bd
                rhs += Bd.T @ y64[i0:i1]
                del Bd, B
        except MemoryError:
            print("[RFE] final exact refit: Gram accumulation OOM; LSMR fallback",
                  flush=True)
            return None
        ridge = float(os.environ.get("PHEASY_RFE_FINAL_RIDGE", "0"))
        if ridge > 0:
            G.flat[:: n + 1] += ridge
        try:
            L = np.linalg.cholesky(G)
            from scipy.linalg import solve_triangular as _stri
            z = _stri(L, rhs, lower=True, check_finite=False)
            coef = _stri(L.T, z, lower=False, check_finite=False)
            _resid = float(np.linalg.norm(G @ coef - rhs))
            print("[RFE] final exact refit: Cholesky ok (||Gx-b||=%.3e)" % _resid,
                  flush=True)
        except np.linalg.LinAlgError:
            print("[RFE] final exact refit: Gram not PD; lstsq(rcond=1e-12)", flush=True)
            coef = np.linalg.lstsq(G, rhs, rcond=1e-12)[0]
    if iterative_diagnostics is not None:
        iterative_diagnostics.append(dict(
            solver="Gram-Cholesky" if method != "tsqr" else "TSQR",
            converged=True, stop_reason="qr_exact" if method == "tsqr" else "normal_equations",
            fit_scope="full", n_features=n,
            n_samples=int(m if n_samples is None else n_samples)))
    return np.asarray(coef, dtype=np.float64)


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
        _rfe_resident_on = os.environ.get(
            "PHEASY_GPU_RFE_RESIDENT",
            "1" if _resident_default() else "0").lower() in ("1", "true", "yes", "on")
        if _rfe_resident_on and _gpu_required() and self.n_jobs != 1:
            # The resident RFE operator is not fork-safe, so the outer loop has to
            # be serial -- but in REQUIRED GPU mode the resident solve is the
            # requirement and outer parallelism is only a CPU-side optimization.
            # Refusing outright made "GPU required + RFE" a hard failure for every
            # caller with a generic thread count: measured with the shipped
            # fit_3090.sh, which exports PHEASY_N_JOBS=8, and the documented escape
            # hatch was to give up the GPU entirely.  Serialize and say so; the
            # per-subset solves stay on the GPU either way.
            warnings.warn(
                "Resident RFE needs a serial outer loop; PHEASY_RFE_N_JOBS/"
                "PHEASY_N_JOBS requested n_jobs=%d. Running the RFE rounds "
                "serially so every subset solve stays on the GPU (set "
                "PHEASY_GPU_RFE_RESIDENT=0 to use the CPU solver with the "
                "requested parallelism)." % int(self.n_jobs),
                RuntimeWarning, stacklevel=3)
            self.n_jobs = 1
        if _rfe_resident_on:
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

        def solve(col_idx, row_idx=None, download=True, scope=None):
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
                    maxiter=int(os.environ.get("PHEASY_LSQR_MAXITER", str(self.lsmr_maxiter))),
                    # RFE's subset solves are for RANKING, not for delivering
                    # coefficients: on the production operator the first round lands
                    # on a measured precision floor above the certifiable cap, and
                    # refusing there aborted the whole fit.  Accept the measured
                    # floor (recorded per solve in info["floor_accepted"] /
                    # "floor_note", and counted in backend_metadata_) while an
                    # exhausted budget with no floor evidence still raises.
                    # PHEASY_RFE_RANKING_FLOOR=0 restores the strict fail-closed path.
                    accept_measured_floor=os.environ.get(
                        "PHEASY_RFE_RANKING_FLOOR", "1").lower() in ("1", "true", "yes", "on"))
                iterative_diagnostics.append(dict(info, n_features=len(col_idx),
                                                 n_samples=n_samples if row_idx is None else len(row_idx),
                                                 fit_scope=(scope if scope is not None
                                                            else ("full" if row_idx is None else "fold"))))
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
                                 diag_scope=(scope if scope is not None
                                             else ("full" if row_idx is None else "fold")))

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
        # [PATCH rfe-final-tsqr-v2] Optional support override: skip the rounds and
        # run only the exact final refit on a support saved by a previous run.
        _sup_npy = os.environ.get("PHEASY_RFE_SUPPORT_NPY")
        _support_override = None
        if _sup_npy:
            try:
                _support_override = np.asarray(np.load(_sup_npy), dtype=np.int64).ravel()
                print("[RFE] support override loaded: %d features from %s"
                      % (_support_override.size, _sup_npy), flush=True)
            except Exception as _exc:
                print("[RFE] support override load failed (%s); running rounds"
                      % type(_exc).__name__, flush=True)
                _support_override = None
        while _support_override is None:
            idx = np.where(active)[0]
            n_active = len(idx)
            # Keep the initial no-elimination fast path, but evaluate the
            # minimum support reached by elimination before selecting a model.
            if n_active <= self.min_features and round_num == 0:
                break

            defer_download = (resident_A is not None
                              and os.environ.get("PHEASY_GPU_RFE_RANKING", "0").lower() in ("1", "true", "yes", "on"))
            coef_active = solve(idx, download=not defer_download, scope="ranking")
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
        if _support_override is not None:
            best_support = np.zeros(n_features, dtype=bool)
            best_support[_support_override] = True
            n_best = int(_support_override.size)
            best_mean, best_se = float("nan"), 0.0
            if self.verbose:
                print("[RFE] support override: n_active=%d (rounds skipped)" % n_best,
                      flush=True)
        elif criterion in ("bic", "aic") and history_bic:
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

        # [PATCH rfe-final-tsqr] Deliver exact OLS coefficients on the selected
        # support.  Every round's full-data LSMR solve is tagged "ranking" above,
        # so only this final solve vetoes the acceptance gate; solving it exactly
        # with TSQR makes the delivered force constants certifiable.
        coef_final = None
        if (_is_linear_operator(A)
                and os.environ.get("PHEASY_RFE_FINAL_TSQR", "1").lower()
                in ("1", "true", "yes", "on")):
            try:
                np.save(os.path.join(".", "rfe_support.npy"), best_idx)
            except Exception:
                pass
            coef_final = _rfe_final_refit_exact(
                A, y, best_idx, block_rows=self.block_rows, diag_floor=self.diag_floor,
                iterative_diagnostics=iterative_diagnostics, n_samples=n_samples)
        if coef_final is None:
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

    Supported methods: ols, lasso, alasso, rfe / rfe-ols (aliases: RFE with an
    OLS base estimator), rfe-ols-tsqr (rfe_tsqr), ardr (automatic relevance
    determination regression) and the legacy ridge.
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
                    # Reuse the resident operator this host operator already owns.
                    # The column-norm pass caches one, and so does resident RIDGE,
                    # so a fit sequence (RIDGE then OLS, or the same operator
                    # fitted twice) used to upload the factors again: 41.5 s at the
                    # MgC slice size and ~206 s at full size for nothing.
                    gpu_op = getattr(base, "_gpu_ridge_op", None)
                    _owns_op = gpu_op is None
                    if _owns_op:
                        gpu_op = _gb.GpuTwoLevelOperator(
                            base, device_ids=_dev_ids,
                            host_factors=_gb.twolevel_host_factors(base))
                    _coln = None
                    try:
                        if jacobi:
                            # Hand the already-built resident operator over so the
                            # norms are computed ON the GPU with no second factor
                            # upload: this pass was 90% of the OLS fit (measured
                            # 197.9 s of 219.9 s on the MgC operator).
                            _coln = np.asarray(
                                X.col_norms(gpu_op=gpu_op) if isinstance(X, TwoLevelSM)
                                else X.col_norms(), dtype=np.float64).ravel()
                            if _coln.shape != (base.shape[1],) or not np.all(np.isfinite(_coln)) or np.any(_coln <= 0):
                                raise ValueError("Jacobi column norms must be finite, positive and complete")
                            gpu_op.scale = gpu_op.torch.as_tensor(
                                _coln, dtype=gpu_op._value_dtype, device=gpu_op.device)
                        coef, info = _gb.iterative_lstsq(gpu_op, y, atol=atol, btol=btol, maxiter=maxiter)
                    finally:
                        if _owns_op:
                            gpu_op.close()
                        else:
                            # Jacobi scaled this shared operator; put the scale
                            # back so a later reuse (resident RIDGE, or the
                            # column-norm pass) does not inherit it.
                            try:
                                gpu_op.scale = gpu_op.torch.ones_like(gpu_op.scale)
                            except Exception:
                                pass
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
        elif method in ("RFE-OLS", "RFE-OLS-CV"):
            # RFE's base estimator IS OLS; "RFE-OLS" is the explicit spelling
            # of the same method (Fransson et al. 2020 call it RFE-OLS).
            method = "RFE"
        elif method in ("ARD", "ARD-REGRESSION"):
            method = "ARDR"
        elif method in ("FAST-RVM", "FASTRVM", "RVM-FAST"):
            method = "RVM"
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
        if self._standardize and method in ("LASSO", "ALASSO", "RIDGE", "ARDR", "RVM") and not (resident_lasso and method in ("LASSO", "ALASSO")):
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
            try:
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
            except (getattr(resident_gb, "ResidentFootprintError", MemoryError),
                    MemoryError) as _e:
                # [FIX resident-fallback] The resident backend asked for a
                # footprint it cannot get.  Do not abort the fit: fall through to
                # the iterative two-level backend, which is exactly the path
                # _lasso_backend already selected when its own pre-flight saw the
                # same overflow (and which keeps the solve on the GPU).
                print("[optimizer] resident %s backend does not fit this device "
                      "set (%s); falling back to the iterative FISTA backend"
                      % (method, _e), flush=True)
                self._model = _LassoCVModel(
                    self._alpha, self._cv, self._tol, self._max_iter, self._rand_seed,
                    _lasso_n_jobs(A),
                    fit_intercept=self._fit_intercept, group_size=self._group_size)
                self._model.fit(A_fit, F64, sample_weight=weights)
                coef = self._model.coef_
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
        elif method == "ARDR":
            _thr_env = os.environ.get("PHEASY_ARDR_THRESHOLDS", "")
            _thresholds = [float(t) for t in _thr_env.split(",") if t.strip()] or None
            self._model = _ARDRModel(
                threshold_lambda=float(os.environ.get("PHEASY_ARDR_THRESHOLD", "1e4")),
                thresholds=_thresholds,
                cv=self._cv,
                max_iter=int(os.environ.get("PHEASY_ARDR_MAX_ITER", "300")),
                tol=self._tol,
                fit_intercept=self._fit_intercept,
                rand_seed=self._rand_seed,
                group_size=self._group_size,
                n_jobs=None,
            )
            self._model.fit(A_fit, F64, sample_weight=weights)
            coef = self._model.coef_
        elif method == "RVM":
            from .fast_rvm import fast_rvm as _fast_rvm
            import time as _t
            _rvm_budget = float(os.environ.get(
                "PHEASY_ARDR_GRAM_MAX_GB", os.environ.get("PHEASY_GRAM_MAX_GB", "4")))
            print("[RVM] building the design Gram (matrix-free; budget %.1f GB)..."
                  % _rvm_budget, flush=True)
            _t0 = _t.time()
            G_rvm, b_rvm, _rvm_how = _build_gram_matrix(A_fit, F64, budget_gb=_rvm_budget)
            print("[RVM] design Gram %dx%d (%.2f GB, %s) in %.1fs"
                  % (G_rvm.shape[0], G_rvm.shape[1], G_rvm.nbytes / 1e9, _rvm_how,
                     _t.time() - _t0), flush=True)
            _rvm_beta_env = os.environ.get("PHEASY_RVM_BETA")
            _rvm_beta = float(_rvm_beta_env) if _rvm_beta_env not in (None, "") else None
            _rvm_max_steps = os.environ.get("PHEASY_RVM_MAX_STEPS")
            _res = _fast_rvm(
                G_rvm, b_rvm, float(F64 @ F64), F64.shape[0],
                y_var=float(np.var(F64)), beta=_rvm_beta,
                beta_iters=int(os.environ.get("PHEASY_RVM_BETA_ITERS", "10")),
                tol=float(os.environ.get("PHEASY_RVM_TOL", "1e-6")),
                max_steps=int(_rvm_max_steps) if _rvm_max_steps else None,
                add_batch=int(os.environ.get("PHEASY_RVM_ADD_BATCH", "1")),
                prune_threshold=float(os.environ.get("PHEASY_RVM_THRESHOLD", "1e4")),
                verbose=os.environ.get("PHEASY_RVM_VERBOSE", "1").lower()
                in ("1", "true", "yes", "on"))
            _c_rvm = np.asarray(_res["coef"], dtype=np.float64)
            _rss_rvm = (float(F64 @ F64) - 2.0 * float(_c_rvm @ b_rvm)
                        + float(_c_rvm @ (G_rvm @ _c_rvm)))
            self._model = _RVMModel(_c_rvm, _res["active"], _res["alpha"],
                                    _res["beta"], _res["n_steps"], A.shape[1],
                                    _rss_rvm, F64.shape[0], _res["converged"])
            self._results["rvm_gram"] = _rvm_how
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
                # No operator exists yet here -- RFE builds each subset later -- so
                # the precision floor CANNOT be applied at this point.  Calling
                # _lsmr_tol without one made _array_precision fall back to
                # PHEASY_SM_DTYPE and bake the AMBIENT precision into this default,
                # which then survived into _solve_subset and over-raised the
                # tolerance of every float64 subset in a process that happened to
                # set the smoke-test variable.  The floor is applied where the
                # operator is in hand.
                lsmr_atol=float(os.environ.get("PHEASY_LSQR_ATOL", 1e-8)),
                lsmr_btol=float(os.environ.get("PHEASY_LSQR_BTOL", 1e-8)),
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
                # Pre-warm the resident operator for the SCALED parent before any
                # fold is sliced.  _row_slice turns a fold into a row VIEW over this
                # operator when it exists, which is what removes one factor upload
                # plus one host csr_tocsc transpose per fold (measured at NCONF=6:
                # GpuTwoLevelOperator.__init__ 5 calls -> 1).  A failure here is not
                # fatal: the folds then slice host factors exactly as before.
                if _resident_default() or os.environ.get(
                        "PHEASY_GPU_RIDGE_RESIDENT", "").lower() in (
                            "1", "true", "yes", "on"):
                    try:
                        from . import gpu_backend as _gb_pw
                        if _gb_pw.enabled() and _gb_pw.available():
                            _resident_ridge_op(A_fit)
                    except Exception:
                        if _gpu_required():
                            raise
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
                # Solver certificates are kept per alpha: A_fit carries only the
                # LAST solve, and the L-curve walk ends on the smallest alpha, so
                # reporting A_fit._gpu_solver_info described a fit that was thrown
                # away while the delivered coefficients came from best_alpha.
                _lc_info = {}
                if os.environ.get("PHEASY_RIDGE_LCURVE", "1").lower() not in ("0", "false", "no", "off"):
                    _x0 = None
                    print("[RIDGE-LC] alpha | ||c|| | ||Xc-y|| | time", flush=True)
                    for _a in alphas:
                        _tl = _time.time()
                        _c = _ridge_solve(A_fit, F64, _a, x0=_x0)
                        _c_info = getattr(A_fit, "_ridge_solver_info", None)
                        _lc_info[float(_a)] = (dict(_c_info, alpha=float(_a)) if _c_info
                                               else (dict(A_fit._gpu_solver_info)
                                                     if getattr(A_fit, "_gpu_solver_info", None)
                                                     else None))
                        _x0 = _c
                        _lc[float(_a)] = _c
                        # Residual for the L-curve line.  This is a diagnostic
                        # print, and on a two-level operator the host matvec costs
                        # O(nnz(SM_prime)) per alpha: two orders of magnitude more
                        # than the print is worth at full size (measured on the MgC
                        # operator: 640 s -> 127 s for the whole RIDGE fit once the
                        # folds went resident, with the L-curve host matvecs and
                        # factor uploads left as the remaining host share).  Use the
                        # resident operator when one exists; it is the same matvec.
                        _op = getattr(A_fit, "_gpu_ridge_op", None)
                        if _op is not None:
                            try:
                                _t = _op.torch.as_tensor(
                                    np.asarray(_c, dtype=np.float64),
                                    dtype=_op._value_dtype, device=_op.device)
                                _pred = np.asarray(
                                    _op.matvec(_t).detach().cpu().numpy(),
                                    dtype=np.float64).ravel()
                                _res = _pred - F64
                            except Exception:
                                _res = np.asarray(A_fit @ _c,
                                                  dtype=np.float64).ravel() - F64
                        else:
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
                # The certificate must describe the solve whose coefficients
                # are returned (best_alpha), never whichever alpha ran last.
                ridge_info = _lc_info.get(float(best_alpha))
                if ridge_info is None:
                    # Only the last solve of this operator is still on it, so it
                    # is usable only when it names the alpha being delivered.
                    _last = getattr(A_fit, "_ridge_solver_info", None)
                    if _last is None:
                        _last = getattr(A_fit, "_gpu_solver_info", None)
                    if (_last is not None
                            and float(_last.get("alpha", best_alpha)) == float(best_alpha)):
                        ridge_info = dict(_last)
                if ridge_info is not None:
                    # Label from the certificate itself: this branch runs whichever
                    # iterative solver _ridge_solve selected, so a hard-coded GPU
                    # name would mislabel the CPU LSMR fallback.
                    self._results["execution_backend"] = str(
                        ridge_info.get("backend", "gpu_twolevel_ridge_resident"))
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
                + "(expected OLS, LASSO, ALASSO, RVM, ARDR, RFE, RFE-OLS, RFE-OLS-TSQR, RIDGE)")

        coef = np.asarray(coef, dtype=np.float64)

        # Relaxed LASSO / debias: L1 selects the support, then an unbiased OLS
        # refit on that support removes the L1 shrinkage bias.  This is the
        # standard practical way to obtain physical force constants from a
        # LASSO fit (Meinshausen 2007; used by ALAMODE/phono3py).  Default on;
        # disable with PHEASY_LASSO_DEBIAS=0.
        self._results.pop("pre_debias_coef", None)
        # [FIX P46] alpha* pinned at the grid bottom => the CV curve never turned
        # up, i.e. the data supports no sparsity.  Ship the relaxed refit and say
        # so in the manifest instead of silently returning the shrunk support
        # (this is how the delivered Mg8C120 v4 fit lost 6x generalization:
        # debias was off AND the 4-decade grid pinned alpha* at its bottom).
        _alpha_at_edge = bool(getattr(self._model, "_alpha_at_min", False))
        self._results["alpha_at_grid_edge"] = _alpha_at_edge
        if _alpha_at_edge:
            self._results["sparsity_supported"] = False
        _edge_relaxed = _alpha_at_edge and self._alpha_edge_relaxed_enabled()
        if _edge_relaxed and not self._debias_enabled():
            self._results["debias_forced_reason"] = (
                "alpha* sits at the grid bottom (CV curve still falling): no "
                "sparsity is supported, so the L1 shrinkage bias is removed by "
                "the relaxed refit even though PHEASY_LASSO_DEBIAS=0.")
        if method in ("LASSO", "ALASSO") and (self._debias_enabled() or _edge_relaxed):
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
            if not info:
                # Every LASSO/ALASSO backend is supposed to certify its own
                # solve.  Silence must not read as a clean bill of health: name
                # the model that published nothing so the gap is auditable
                # instead of showing up as a missing execution_backend.
                self._results["regularized_solver_info_missing"] = type(
                    self._model).__name__
            if info is not None:
                self._results["regularized_solver_info"] = dict(info)
                backend = str(info.get("backend", ""))
                if backend:
                    self._results["execution_backend"] = backend
                    if self._debias_enabled() or _edge_relaxed:
                        db = getattr(self, "_debias_backend", "unknown")
                        self._results["debias_backend"] = db
                        self._results["postfit_backend"] = db + "_and_cpu_metrics"
                        # A declared post-fit stage that did not take effect must
                        # be visible: the residual guard can reject the refit, and
                        # before this the results still named the backend as if it
                        # had run.
                        if getattr(self, "_debias_accepted", None) is not None:
                            self._results["debias_accepted"] = bool(self._debias_accepted)
                            self._results["debias_stage"] = getattr(
                                self, "_debias_stage", None)
                            _r = getattr(self, "_debias_residuals", None)
                            if _r is not None:
                                self._results["debias_residual_refit"] = _r[0]
                                self._results["debias_residual_shrunk"] = _r[1]
                            if not self._debias_accepted:
                                self._results["debias_fallback_reason"] = (
                                    "support refit rejected: residual %.6e > %.6e, "
                                    "keeping the L1 coefficients" % (_r[0], _r[1]))
                        _di = getattr(self, "_debias_solver_info", None)
                        if isinstance(_di, dict):
                            self._results["debias_solver_info"] = dict(_di)
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
        elif method == "ARDR":
            self._results["n_iter"] = int(getattr(self._model, "n_iter_", 0))
            # 'alpha' is the ARD noise precision (1/sigma^2), not a penalty.
            self._results["alpha"] = float(getattr(self._model, "alpha_", 0.0))
            self._results["ardr_threshold"] = float(
                getattr(self._model, "threshold_", float("nan")))
            _mse = np.asarray(self._model.mse_path_, dtype=np.float64)
            _best = int(getattr(self._model, "best_index_", 0))
            self._metrics["mse_path"] = _mse[_best]
            self._metrics["mse_path_mean"] = float(np.mean(_mse[_best]))
            self._metrics["rmse_path"] = np.sqrt(_mse[_best])
            self._metrics["rmse_path_mean"] = float(
                getattr(self._model, "best_rmse_cv_",
                        np.sqrt(self._metrics["mse_path_mean"])))
            self._metrics["n_features"] = self._model.n_features_in_
            self._metrics["n_featrues"] = self._metrics["n_features"]
            self._results["cv_evaluated"] = bool(
                getattr(self._model, "cv_evaluated", True))
            _ardr_info = getattr(self._model, "regularized_solver_info_", None)
            if _ardr_info:
                self._results["regularized_solver_info"] = dict(_ardr_info)
                self._results["execution_backend"] = str(
                    _ardr_info.get("backend", "cpu_dense_ardr"))
                self._results["postfit_backend"] = "cpu_metrics"
        elif method == "RVM":
            self._results["n_iter"] = int(getattr(self._model, "n_iter_", 0))
            self._results["alpha"] = float(getattr(self._model, "beta_", 0.0))
            self._results["rvm_beta"] = float(getattr(self._model, "beta_", float("nan")))
            self._results["rvm_active"] = int(len(getattr(self._model, "active_", [])))
            _mse = np.asarray(self._model.mse_path_, dtype=np.float64)
            self._metrics["mse_path"] = _mse[0]
            self._metrics["mse_path_mean"] = float(np.mean(_mse[0]))
            self._metrics["rmse_path"] = np.sqrt(_mse[0])
            self._metrics["rmse_path_mean"] = float(getattr(
                self._model, "best_rmse_cv_", np.sqrt(self._metrics["mse_path_mean"])))
            self._metrics["n_features"] = self._model.n_features_in_
            self._metrics["n_featrues"] = self._metrics["n_features"]
            self._results["cv_evaluated"] = False
            _rvm_info = getattr(self._model, "regularized_solver_info_", None)
            if _rvm_info:
                self._results["regularized_solver_info"] = dict(_rvm_info)
                self._results["execution_backend"] = str(
                    _rvm_info.get("backend", "cpu_gram_rvm"))
                self._results["postfit_backend"] = "cpu_metrics"
        elif method == "RIDGE":
            # _ridge_solver_info is written by BOTH iterative branches of
            # _ridge_solve (resident CGLS and CPU LSMR); _gpu_solver_info is the
            # older GPU-only name, kept as a fallback for operators solved before
            # this change.
            ridge_info = getattr(A_fit, "_ridge_solver_info", None)
            if ridge_info is None:
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

    @staticmethod
    def _alpha_edge_relaxed_enabled():
        """[FIX P46] Relaxed refit forced when alpha* sits at the grid bottom.

        A pinned alpha* means the CV curve never turned up inside the grid: the
        data supports no sparsity, and the L1 coefficients then carry the full
        shrinkage bias (Mg8C120 v4: ||x|| 608 vs 2363 unbiased, re 7.5% vs 1.3%).
        The relaxed (unpenalized-on-support) refit is the generalizing answer, so
        it is enabled even when the wrapper set PHEASY_LASSO_DEBIAS=0 -- which is
        exactly how the shipped v4 fit lost it.  PHEASY_LASSO_EDGE_RELAXED=0
        disables.
        """
        return os.environ.get("PHEASY_LASSO_EDGE_RELAXED", "1").lower() in ("1", "true", "yes")

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
            self._record_debias(r_new, r_old, "gram")
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
            self._record_debias(r_new, r_old, "operator")
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
        self._record_debias(r_new, r_old, "dense")
        if r_new <= r_old:
            return new
        return coef

    def _record_debias(self, r_new, r_old, stage):
        """Say whether the relaxed-LASSO refit was actually kept.

        The residual guard is right to drop a refit whose residual grew -- that
        is what stops a corrupted support solve from shipping -- but dropping it
        silently means the declared stage vanishes: on c7 float32 the pristine
        CGLS debias came back `invalid_search_direction`, the guard rejected it,
        and the results still read debias_backend=gpu_cgls with no hint that the
        L1 shrinkage had NOT been removed.  Record the decision and its numbers.
        """
        self._debias_accepted = bool(r_new <= r_old)
        self._debias_residuals = (float(r_new), float(r_old))
        self._debias_stage = stage

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
            coef_sub, _info = gb.solve_resident_subset(res_op, y, sup,
                                                        raise_on_nonconvergence=False)
            # Keep the debias certificate.  It was discarded, and the residual
            # guard below then silently dropped the whole stage when the solve
            # came back corrupted -- on c7 float32 that made the declared
            # relaxed-LASSO refit a no-op with nothing in the results saying so.
            self._debias_solver_info = dict(_info) if isinstance(_info, dict) else _info
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
