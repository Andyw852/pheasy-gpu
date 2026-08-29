#!/usr/bin/env python3
"""Fixed-alpha FISTA vs sklearn-CD scan (no CV): coef rel diff vs nnz.

Settles whether the ~3.5e-4 LASSO FISTA-vs-CD coefficient difference on the c7
SM (n=8) is a conditioning artifact of the near-unregularized regime, or a sign
that FISTA's tolerance is too loose. By dropping alpha SELECTION (CV) and
comparing the two solvers at fixed alphas, we read off the rel-diff-vs-nnz
curve: ~1e-7 in the sparse regime degrading monotonically toward ~1e-4 as
nnz -> 100% is the "conditioning" signature.

Unlike recheck_lasso.py level 3, this does NOT run CV: each alpha is solved once
by FISTA (GpuLassoCV with a single alpha, cv=2, full-data final refit) and once
by sklearn CD (Lasso), on the same standardized A. sklearn's ConvergenceWarning
is captured (it goes to stderr via warnings, invisible to redirect_stdout), so an
unconverged CD baseline is flagged instead of silently trusted.

Usage on the GPU box:

    CUDA_VISIBLE_DEVICES=5 PHEASY_USE_GPU=1 python dev/alpha_scan.py \
        --data-dir $REMOTE/pheasy/MnIn2Se4_c7
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="c7 scan dir (sm_prime.npz / ns_harm.npz / "
                         "ns_anharm3.npz / fm1d.npz)")
    ap.add_argument("--n-configs", type=int, default=8)
    ap.add_argument("--rows-per-config", type=int, default=567)
    ap.add_argument("--alphas", default="1e-3,1e-4,1e-5,1e-6,1e-7,1e-8",
                    help="comma-separated alphas, descending (sparse -> dense)")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=200000)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scipy import sparse as sp
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso
    from pheasy_gpu.core import gpu_backend as gb

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    n_rows = args.n_configs * args.rows_per_config
    sm = sp.load_npz(os.path.join(args.data_dir, "sm_prime.npz"))
    nsh = sp.load_npz(os.path.join(args.data_dir, "ns_harm.npz"))
    nsa = sp.load_npz(os.path.join(args.data_dir, "ns_anharm3.npz"))
    t0 = time.time()
    SM = gb.load_sensing_matrix(sm, nsh, nsa, n_rows, dtype=np.float64)
    fm = np.load(os.path.join(args.data_dir, "fm1d.npz"))
    key = "F" if "F" in fm else [k for k in fm.files if not k.startswith("_")][0]
    F = np.asarray(fm[key], dtype=np.float64).ravel()[:n_rows]
    print("loaded SM %s F %s (%.1fs)" % (SM.shape, F.shape, time.time() - t0),
          flush=True)

    # column standardization (unit L2 norm), matching Optimizer(standardize=True)
    cn = np.sqrt((SM * SM).sum(axis=0))
    cn = np.where(cn < 1e-30, 1.0, cn)
    As = SM / cn
    p = As.shape[1]

    print("%-11s %8s %8s %10s  %s" % ("alpha", "nnz", "nnz%", "rel diff", "CD"),
          flush=True)
    for a in alphas:
        g = gb.GpuLassoCV([a], cv=2, tol=args.tol, max_iter=args.max_iter,
                          rand_seed=0).fit(As, F)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = Lasso(alpha=a, fit_intercept=False, tol=args.tol,
                      max_iter=args.max_iter).fit(As, F)
        conv = [x for x in w if issubclass(x.category, ConvergenceWarning)]
        nnz = int(np.count_nonzero(s.coef_))
        denom = max(float(np.linalg.norm(s.coef_)), 1e-30)
        rel = float(np.linalg.norm(g.coef_ - s.coef_) / denom)
        note = "n_iter=%d" % int(s.n_iter_)
        if conv:
            note += "  <-- ConvergenceWarning (CD not converged)"
        print("%-11.3e %8d %7.1f%% %10.3e  %s" % (a, nnz, 100.0 * nnz / p, rel, note),
              flush=True)


if __name__ == "__main__":
    main()
