#!/usr/bin/env python3
"""gpu_test.py -- verify pheasy-gpu against the CPU path and time both.

Usage (on the GPU box, from the data dir or with --data):
    python dev/gpu_test.py <data_dir> --n-configs 8 --methods OLS RIDGE LASSO ALASSO
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pheasy_gpu.core import gpu_backend as gb


def _check_lstsq():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((500, 80))
    y = rng.standard_normal(500)
    gpu = gb.lstsq(A, y)
    cpu, *_ = np.linalg.lstsq(A, y, rcond=None)
    err = float(np.linalg.norm(gpu - cpu))
    print("[lstsq] GPU vs numpy.linalg.lstsq coef diff = %.3e" % err)
    assert err < 1e-6, "lstsq mismatch"


def _check_ridge():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((500, 80))
    y = rng.standard_normal(500)
    for a in (1e-2, 1.0, 10.0):
        g = gb.ridge_solve(A, y, a)
        from sklearn.linear_model import Ridge
        c = Ridge(alpha=a, fit_intercept=False, solver="svd").fit(A, y).coef_
        err = float(np.linalg.norm(g - c))
        print("[ridge a=%.2g] GPU vs sklearn coef diff = %.3e" % (a, err))
        assert err < 1e-6, "ridge mismatch"


def _load(d, n_configs, rpc):
    from scipy import sparse as sp
    sm = sp.load_npz(os.path.join(d, "sm_prime.npz"))
    nsh = sp.load_npz(os.path.join(d, "ns_harm.npz"))
    nsa = sp.load_npz(os.path.join(d, "ns_anharm3.npz"))
    t0 = time.time()
    SM = gb.load_sensing_matrix(sm, nsh, nsa, n_configs * rpc, dtype=np.float64)
    print("[load] SM %s (%.1fs)" % (SM.shape, time.time() - t0))
    fm = np.load(os.path.join(d, "fm1d.npz"))
    key = "F" if "F" in fm else list(fm.keys())[0]
    F = np.asarray(fm[key], dtype=np.float64).ravel()[: n_configs * rpc]
    return SM, F


def _fit(method, A, y, use_gpu):
    from pheasy_gpu.core.optimizer import Optimizer
    os.environ["PHEASY_CV_GROUP_SIZE"] = "567"
    kw = dict(nalpha=20, cv=5, tol=1e-6, max_iter=20000, alpha_auto=True,
              decades=4.0, rand_seed=0, standardize=(method in ("LASSO", "ALASSO", "RIDGE")))
    if method == "LASSO":
        from pheasy_gpu.core.optimizer import derive_alpha_grid
        kw["alpha"] = derive_alpha_grid(A, y, nalpha=20, decades=4.0, standardize=True)
    if method == "RIDGE":
        kw["alpha"] = np.logspace(-6, 4, 50)
        kw["alpha_auto"] = False
    o = Optimizer(method, use_gpu=use_gpu, **kw)
    o.fit(A, y)
    return o.results["coef"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--n-configs", type=int, default=8)
    ap.add_argument("--rows-per-config", type=int, default=567)
    ap.add_argument("--methods", default="OLS RIDGE LASSO ALASSO")
    args = ap.parse_args()

    print("GPU available:", gb.available(), "| enabled:", gb.enabled())
    if gb.enabled():
        import torch
        print("GPU device:", torch.cuda.get_device_name(gb.device()))
    _check_lstsq()
    _check_ridge()

    SM, F = _load(args.data_dir, args.n_configs, args.rows_per_config)
    for m in args.methods.split():
        t0 = time.time(); c_gpu = _fit(m, SM, F, True); t_gpu = time.time() - t0
        t0 = time.time(); c_cpu = _fit(m, SM, F, False); t_cpu = time.time() - t0
        d = float(np.linalg.norm(c_gpu - c_cpu)) / max(float(np.linalg.norm(c_cpu)), 1e-30)
        nnz = int(np.count_nonzero(c_gpu))
        print("%-8s rel coef diff=%.3e  nnz=%d  GPU %.1fs  CPU %.1fs  speedup %.1fx"
              % (m, d, nnz, t_gpu, t_cpu, t_cpu / max(t_gpu, 1e-9)))


if __name__ == "__main__":
    main()

