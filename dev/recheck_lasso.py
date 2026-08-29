#!/usr/bin/env python3
"""Re-verify the LASSO/ALASSO GPU path after the Lipschitz fix (54d9683).

Three levels, cheapest first -- run them in order and stop at the first failure.

  1. --level lipschitz   no data, no GPU-heavy work. Asserts _power_lipschitz(G)
                         == eigvalsh(G)[-1]. This is the direct proof that the
                         lambda_max^2 bug is gone; if it fails nothing else matters.
  2. --level synthetic   no data dir. GpuLassoCV vs sklearn LassoCV on a fixed-seed
                         problem: alpha_, nnz, coef rel diff. This is the bit-level
                         reference the GPU.md "~1e-4 agreement" claim needs.
  3. --level real DIR    the c7 scan directory. Optimizer LASSO/ALASSO with
                         PHEASY_GPU_LASSO=1 (GPU FISTA) vs 0 (sklearn CD), keeping
                         the [CV] WARNING lines that holdout_eval.py discards.

Usage on the GPU box:

    CUDA_VISIBLE_DEVICES=6 PHEASY_USE_GPU=1 python dev/recheck_lasso.py --level all \
        --data-dir $REMOTE/pheasy/MnIn2Se4_c7 2>&1 | tee recheck.out

Exit code is 0 only if every level that ran passed its threshold.
"""
import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

FAILED = []


def _ok(cond, label, detail=""):
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label,
                           ("  -- " + detail) if detail else ""), flush=True)
    if not cond:
        FAILED.append(label)
    return cond


# ---------------------------------------------------------------------------
# level 1: the Lipschitz constant itself
# ---------------------------------------------------------------------------
def level_lipschitz():
    print("\n=== level 1: _power_lipschitz == lambda_max(G) ===", flush=True)
    from pheasy_gpu.core import gpu_backend as gb
    if not gb.enabled():
        print("  SKIP: GPU backend not enabled (PHEASY_USE_GPU / torch.cuda)", flush=True)
        return
    import torch
    rng = np.random.default_rng(0)
    for (n, p) in [(400, 80), (2000, 300)]:
        A = rng.standard_normal((n, p)) * 3.0
        G = A.T @ A
        ref = float(np.linalg.eigvalsh(G)[-1])
        got = float(gb._power_lipschitz(
            torch.as_tensor(G, dtype=torch.float64, device=gb.device())))
        rel = abs(got - ref) / ref
        # the old bug returned lambda_max^2; flag it explicitly so a regression
        # is unmistakable rather than just "a large number".
        squared = abs(got - ref * ref) / (ref * ref) < 1e-6
        _ok(rel < 1e-10 and not squared,
            "n=%d p=%d" % (n, p),
            "lambda_max=%.6e got=%.6e rel=%.2e%s"
            % (ref, got, rel, "  <-- REGRESSED TO lambda_max^2" if squared else ""))


# ---------------------------------------------------------------------------
# level 2: GpuLassoCV vs sklearn LassoCV on a synthetic problem
# ---------------------------------------------------------------------------
def level_synthetic():
    print("\n=== level 2: GpuLassoCV vs sklearn LassoCV (synthetic) ===", flush=True)
    from pheasy_gpu.core import gpu_backend as gb
    from sklearn.linear_model import LassoCV
    if not gb.enabled():
        print("  SKIP: GPU backend not enabled", flush=True)
        return

    rng = np.random.default_rng(0)
    n, p, k = 2000, 400, 100
    A = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    beta[k:] = 0.0
    y = A @ beta + 0.1 * rng.standard_normal(n)
    cn = np.sqrt((A * A).sum(axis=0))
    cn = np.where(cn < 1e-30, 1.0, cn)
    As = A / cn
    alphas = np.logspace(-6, -1, 12)

    # tol tight on BOTH sides: a loose sklearn tol was the original source of
    # the flat-CV-tail confusion, and a loose comparison would hide a stalled
    # FISTA rather than expose it.
    lcv = LassoCV(alphas=alphas, cv=5, fit_intercept=False, tol=1e-8,
                  max_iter=200000, random_state=0)
    lcv.fit(As, y)
    gcv = gb.GpuLassoCV(alphas, cv=5, tol=1e-8, max_iter=200000, rand_seed=0)
    gcv.fit(As, y)

    nz_s = int(np.count_nonzero(lcv.coef_))
    nz_g = int(np.count_nonzero(gcv.coef_))
    denom = max(float(np.linalg.norm(lcv.coef_)), 1e-30)
    cdiff = float(np.linalg.norm(lcv.coef_ - gcv.coef_) / denom)
    adiff = abs(lcv.alpha_ - gcv.alpha_) / max(float(lcv.alpha_), 1e-30)

    print("  sklearn: alpha=%.6e  nnz=%d" % (lcv.alpha_, nz_s), flush=True)
    print("  gpu    : alpha=%.6e  nnz=%d  n_iter=%d"
          % (gcv.alpha_, nz_g, int(gcv.n_iter_)), flush=True)
    _ok(adiff < 1e-12, "same alpha* grid point", "rel diff %.2e" % adiff)
    _ok(cdiff < 1e-3, "coef agreement", "rel diff %.3e (was ~3e-5 pre-bug)" % cdiff)
    # the stalled-FISTA signature: support blows up toward dense.
    _ok(abs(nz_g - nz_s) <= max(5, 0.05 * nz_s),
        "support size", "sklearn %d vs gpu %d (of p=%d)" % (nz_s, nz_g, p))


# ---------------------------------------------------------------------------
# level 3: real sensing matrix, GPU FISTA vs sklearn CD through the Optimizer
# ---------------------------------------------------------------------------
def _load(d, n_configs, rows_per_config):
    from scipy import sparse as sp
    from pheasy_gpu.core import gpu_backend as gb
    n_rows = n_configs * rows_per_config
    sm = sp.load_npz(os.path.join(d, "sm_prime.npz"))
    nsh = sp.load_npz(os.path.join(d, "ns_harm.npz"))
    nsa = sp.load_npz(os.path.join(d, "ns_anharm3.npz"))
    t0 = time.time()
    SM = gb.load_sensing_matrix(sm, nsh, nsa, n_rows, dtype=np.float64)
    fm = np.load(os.path.join(d, "fm1d.npz"))
    key = "F" if "F" in fm else [k for k in fm.files if not k.startswith("_")][0]
    F = np.asarray(fm[key], dtype=np.float64).ravel()[:n_rows]
    print("  loaded SM %s F %s (%.1fs)" % (SM.shape, F.shape, time.time() - t0),
          flush=True)
    return SM, F


def _fit(method, SM, F, gpu_lasso, rows_per_config):
    """Fit once; return (coef, alpha_, n_iter_, captured stdout, seconds)."""
    from pheasy_gpu.core.optimizer import Optimizer, derive_alpha_grid
    os.environ["PHEASY_GPU_LASSO"] = "1" if gpu_lasso else "0"
    os.environ["PHEASY_CV_GROUP_SIZE"] = str(rows_per_config)
    kw = dict(nalpha=20, cv=5, tol=1e-6, max_iter=20000, alpha_auto=True,
              decades=4.0, rand_seed=0, standardize=True)
    if method == "LASSO":
        kw["alpha"] = derive_alpha_grid(SM, F, nalpha=20, decades=4.0,
                                        standardize=True)
    t0 = time.time()
    o = Optimizer(method, **kw)
    buf = io.StringIO()
    # capture, but -- unlike holdout_eval.py -- do NOT discard: the [CV] WARNING
    # tie/flat-tail lines are the whole point of this comparison.
    with contextlib.redirect_stdout(buf):
        o.fit(SM, F)
    return (np.asarray(o.results["coef"], dtype=np.float64),
            float(o._model.alpha_), int(getattr(o._model, "n_iter_", -1)),
            buf.getvalue(), time.time() - t0)


def level_real(data_dir, n_configs, rows_per_config, methods):
    print("\n=== level 3: %s on %s (n_configs=%d) ==="
          % ("/".join(methods), data_dir, n_configs), flush=True)
    SM, F = _load(data_dir, n_configs, rows_per_config)

    for m in methods:
        print("\n-- %s --" % m, flush=True)
        rows = {}
        for label, gpu in (("gpu-fista", True), ("sklearn-cd", False)):
            c, a, nit, out, sec = _fit(m, SM, F, gpu, rows_per_config)
            ties = [ln for ln in out.splitlines() if "alphas tie" in ln]
            atmin = [ln for ln in out.splitlines() if "grid MINIMUM" in ln]
            rows[label] = (c, a, nit, ties, atmin)
            print("  %-10s alpha=%.6e  nnz=%d  n_iter=%d  (%.1fs)"
                  % (label, a, int(np.count_nonzero(c)), nit, sec), flush=True)
            for ln in ties + atmin:
                print("      | " + ln.strip(), flush=True)

        cg, ag, _, tg, _ = rows["gpu-fista"]
        cs, asl, _, ts, _ = rows["sklearn-cd"]
        adiff = abs(ag - asl) / max(asl, 1e-30)
        cdiff = float(np.linalg.norm(cg - cs) / max(float(np.linalg.norm(cs)), 1e-30))
        _ok(adiff < 1e-12, "%s: same alpha*" % m, "rel diff %.2e" % adiff)
        _ok(cdiff < 1e-3, "%s: coef agreement" % m, "rel diff %.3e" % cdiff)
        # Not a hard failure: a genuinely flat tail can survive the fix. But if
        # the GPU side ties and the sklearn side does not, FISTA is still short.
        if tg and not ts:
            _ok(False, "%s: GPU-only tie warning" % m,
                "FISTA still not converging within cv_max_iter")
        elif tg and ts:
            print("  [note] %s: both backends report a tie -- genuinely flat tail"
                  % m, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", default="all",
                    choices=["lipschitz", "synthetic", "real", "all"])
    ap.add_argument("--data-dir", default=None,
                    help="c7 scan dir (sm_prime.npz / ns_harm.npz / "
                         "ns_anharm3.npz / fm1d.npz); required for --level real")
    ap.add_argument("--n-configs", type=int, default=8)
    ap.add_argument("--rows-per-config", type=int, default=567)
    ap.add_argument("--methods", default="LASSO ALASSO",
                    help="space-separated, same convention as holdout_eval.py")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pheasy_gpu.core import gpu_backend as gb
    print("torch available=%s  gpu enabled=%s" % (gb.available(), gb.enabled()),
          flush=True)

    if args.level in ("lipschitz", "all"):
        level_lipschitz()
    if args.level in ("synthetic", "all"):
        level_synthetic()
    if args.level in ("real", "all"):
        if args.data_dir:
            level_real(args.data_dir, args.n_configs, args.rows_per_config,
                       args.methods.split())
        elif args.level == "real":
            ap.error("--level real needs --data-dir")
        else:
            print("\n=== level 3: SKIP (no --data-dir) ===", flush=True)

    print("\n%s" % ("FAILED: " + ", ".join(FAILED) if FAILED else "ALL PASS"),
          flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
