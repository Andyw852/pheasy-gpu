"""Memory benchmark: per-method GPU VRAM peak + host RSS, c7 SM n=8.

Complement to dev/speed_parallel.py. Loads the SM once, then fits each method
on the GPU backend and reports torch.cuda.max_memory_allocated (the allocator
peak, i.e. what a fit actually touches) plus max_memory_reserved (cache) and
the process high-water host RSS. Launch with PHEASY_USE_GPU=1 and a free GPU.
"""
import os, time, io, contextlib, argparse, resource
import numpy as np, torch
from scipy import sparse as sp
from pheasy_gpu.core import gpu_backend as gb
from pheasy_gpu.core.optimizer import Optimizer, derive_alpha_grid

GROUP = 567
FIT_KW = dict(nalpha=20, cv=5, tol=1e-6, max_iter=20000, alpha_auto=True, decades=4.0)

def rss_mb():
    # ru_maxrss is in KiB on Linux (high-water mark of the whole process)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    d = args.data_dir
    n = args.n

    sm = sp.load_npz(os.path.join(d, "sm_prime.npz"))
    nsh = sp.load_npz(os.path.join(d, "ns_harm.npz"))
    nsa = sp.load_npz(os.path.join(d, "ns_anharm3.npz"))
    torch.cuda.reset_peak_memory_stats()
    SM = gb.load_sensing_matrix(sm, nsh, nsa, n * GROUP, dtype=np.float64)
    print("[load] SM %s | gpu_peak=%.0f MB reserved=%.0f MB host_rss=%.0f MB" % (
        SM.shape, torch.cuda.max_memory_allocated() / 1e6,
        torch.cuda.memory_reserved() / 1e6, rss_mb()), flush=True)

    fm = np.load(os.path.join(d, "fm1d.npz"))
    key = [k for k in fm if not k.startswith("_")][0]
    F = np.asarray(fm[key], dtype=np.float64).ravel()[:n * GROUP]

    print("%-8s %12s %10s %10s %6s" % ("method", "gpu_peak_MB", "reserved_MB", "host_rss_MB", "nnz"), flush=True)
    for m in ["OLS", "RIDGE", "LASSO", "ALASSO", "RFE"]:
        kw = dict(FIT_KW, rand_seed=0, standardize=True)
        if m == "LASSO":
            kw["alpha"] = derive_alpha_grid(SM, F, nalpha=20, decades=4.0, standardize=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        o = Optimizer(m, use_gpu=True, **kw)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            o.fit(SM, F)
        print("%-8s %12.0f %10.0f %10.0f %6d" % (
            m, torch.cuda.max_memory_allocated() / 1e6,
            torch.cuda.memory_reserved() / 1e6, rss_mb(),
            int(np.count_nonzero(o.results["coef"]))), flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
