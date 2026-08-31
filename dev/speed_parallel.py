"""Speed benchmark: fitting wall-time vs parallelization backend (same 3090).

Backends (OpenBLAS left at host default 56 threads for every CPU run):
  gpu       -- torch cuBLAS (use_gpu=True)
  cpu_blas  -- single process, n_jobs=1 (OpenBLAS multithreaded only)
  cpu_njobs -- sklearn loky processes, n_jobs=NPROC (CV grid fanned out)

n_jobs only applies to the CV-grid methods (LASSO / ALASSO / RFE); OLS and
RIDGE are a single SVD so n_jobs is inert and they run gpu + cpu_blas only.
Times are fit-only (SM loaded once up front, stdout silenced); the GPU is
warmed up (one OLS) before timed runs. Launch with PHEASY_USE_GPU=1 so the SM
load itself uses the GPU; per-fit use_gpu then scopes each backend.
"""
import os, time, io, argparse, contextlib
import numpy as np
from scipy import sparse as sp
from pheasy_gpu.core import gpu_backend as gb
from pheasy_gpu.core.optimizer import Optimizer, derive_alpha_grid

GROUP = 567
FIT_KW = dict(nalpha=20, cv=5, tol=1e-6, max_iter=20000, alpha_auto=True, decades=4.0)
CV_METHODS = {"LASSO", "ALASSO", "RFE"}

def load(data_dir, n):
    sm = sp.load_npz(os.path.join(data_dir, "sm_prime.npz"))
    nsh = sp.load_npz(os.path.join(data_dir, "ns_harm.npz"))
    nsa = sp.load_npz(os.path.join(data_dir, "ns_anharm3.npz"))
    SM = gb.load_sensing_matrix(sm, nsh, nsa, n * GROUP, dtype=np.float64)
    fm = np.load(os.path.join(data_dir, "fm1d.npz"))
    key = [k for k in fm if not k.startswith("_")][0]
    F = np.asarray(fm[key], dtype=np.float64).ravel()[:n * GROUP]
    return SM, F

def fit(method, use_gpu, n_jobs, SM, F):
    os.environ["PHEASY_N_JOBS"] = str(n_jobs)
    os.environ["PHEASY_RFE_STEP"] = "0.3"
    os.environ["PHEASY_RFE_MIN_FEATURES"] = "100"
    kw = dict(FIT_KW, rand_seed=0, standardize=True)
    if method == "LASSO":
        kw["alpha"] = derive_alpha_grid(SM, F, nalpha=20, decades=4.0, standardize=True)
    t0 = time.time()
    o = Optimizer(method, use_gpu=use_gpu, **kw)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        o.fit(SM, F)
    dt = time.time() - t0
    return dt, int(np.count_nonzero(o.results["coef"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--skip-rfe", action="store_true")
    args = ap.parse_args()

    print("loading SM (n=%d) ..." % args.n, flush=True)
    SM, F = load(args.data_dir, args.n)
    print("SM %s F %s" % (SM.shape, F.shape), flush=True)

    if gb.enabled():
        fit("OLS", True, 1, SM, F)
        print("gpu warmup done", flush=True)

    all_backends = [
        ("gpu",       True,  1),
        ("cpu_blas",  False, 1),
        ("cpu_njobs", False, args.n_jobs),
    ]
    methods = ["OLS", "RIDGE", "LASSO", "ALASSO"]
    if not args.skip_rfe:
        methods.append("RFE")

    print("%-8s %-10s %10s %8s" % ("method", "backend", "time_s", "nnz"), flush=True)
    for m in methods:
        backends = all_backends if m in CV_METHODS else all_backends[:2]
        for name, use_gpu, n_jobs in backends:
            try:
                dt, nnz = fit(m, use_gpu, n_jobs, SM, F)
                print("%-8s %-10s %10.1f %8d" % (m, name, dt, nnz), flush=True)
            except Exception as e:
                print("%-8s %-10s FAILED %r" % (m, name, e), flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
