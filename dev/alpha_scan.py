#!/usr/bin/env python3
"""Fixed-alpha FISTA vs sklearn-CD scan (no CV): solver difference vs nnz.

Reads off |FISTA - CD| / |CD| at fixed alphas so alpha SELECTION (CV) cannot
hide the solver comparison. The main column is the DIFFERENCE between the two
solvers, not an error vs truth; pass --reference to also fit a tol=1e-12 CD
reference and report err_fista / err_cd (each solver's error vs that reference,
which separates "FISTA is off" from "CD is off").

Note the FISTA side runs through GpuLassoCV, whose final full-data refit is
capped at max_iter=5000 / tol=1e-7 (shared verbatim with the CPU
_LassoCVIterative), so a printed FISTA n_iter of 5000 at dense alphas means the
FISTA side stopped on the iteration cap, not on tolerance.

Usage on the GPU box:

    CUDA_VISIBLE_DEVICES=5 PHEASY_USE_GPU=1 python dev/alpha_scan.py --data-dir $REMOTE/pheasy/MnIn2Se4_c7
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
    ap.add_argument("--reference", action="store_true",
                    help="also fit a tol=1e-12 / max_iter=1e6 CD reference and "
                         "report err_fista / err_cd (error vs that reference)")
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

    # column standardization (unit L2 norm), matching Optimizer(standardize=True).
    # np.linalg.norm avoids the transient full copy that (SM*SM) would make.
    cn = np.linalg.norm(SM, axis=0)
    cn = np.where(cn < 1e-30, 1.0, cn)
    As = SM / cn
    p = As.shape[1]

    header = "%-11s %8s %8s %12s  %-30s" % ("alpha", "nnz", "nnz%", "|F-C|/|C|",
                                             "iters")
    if args.reference:
        header += "  %10s %10s" % ("err_fista", "err_cd")
    print(header, flush=True)
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
        note = "CD=%d/FISTA=%d" % (int(s.n_iter_), int(g.n_iter_))
        if conv:
            note += " CD-ConvWarning"
        if int(g.n_iter_) >= 5000:
            note += " FISTA-cap"
        row = "%-11.3e %8d %7.1f%% %12.3e  %-30s" % (a, nnz, 100.0 * nnz / p,
                                                      rel, note)
        if args.reference:
            with warnings.catch_warnings(record=True) as w2:
                warnings.simplefilter("always")
                ref = Lasso(alpha=a, fit_intercept=False, tol=1e-12,
                            max_iter=10 ** 6).fit(As, F)
            rconv = [x for x in w2 if issubclass(x.category, ConvergenceWarning)]
            rden = max(float(np.linalg.norm(ref.coef_)), 1e-30)
            err_f = float(np.linalg.norm(g.coef_ - ref.coef_) / rden)
            err_c = float(np.linalg.norm(s.coef_ - ref.coef_) / rden)
            row += "  %10.3e %10.3e" % (err_f, err_c)
            if rconv:
                row += " ref-hit-cap"
        print(row, flush=True)


if __name__ == "__main__":
    main()
