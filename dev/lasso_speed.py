import os, time, io, numpy as np
os.environ["PHEASY_CV_GROUP_SIZE"] = "567"
from pheasy_gpu.core import gpu_backend as gb
from pheasy_gpu.core.optimizer import Optimizer, derive_alpha_grid
from scipy import sparse as sp
import contextlib

d = "<DATA_DIR>/MnIn2Se4_c7"  # set to your c7 scan directory
sm = sp.load_npz(os.path.join(d, "sm_prime.npz"))
nsh = sp.load_npz(os.path.join(d, "ns_harm.npz"))
nsa = sp.load_npz(os.path.join(d, "ns_anharm3.npz"))
SM = gb.load_sensing_matrix(sm, nsh, nsa, 8*567, dtype=np.float64)
fm = np.load(os.path.join(d, "fm1d.npz"))
key = [k for k in fm if not k.startswith("_")][0]
F = np.asarray(fm[key], dtype=np.float64).ravel()[:8*567]

FIT_KW = dict(nalpha=20, cv=5, tol=1e-6, max_iter=20000, alpha_auto=True, decades=4.0)

def run(method, use_gpu):
    kw = dict(FIT_KW, rand_seed=0, standardize=True)
    if method == "LASSO":
        kw["alpha"] = derive_alpha_grid(SM, F, nalpha=20, decades=4.0, standardize=True)
    t0 = time.time()
    o = Optimizer(method, use_gpu=use_gpu, **kw)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        o.fit(SM, F)
    c = o.results["coef"]
    print("%-6s use_gpu=%s: alpha=%.6e  nnz=%d  %.1fs" % (method, use_gpu, float(o._model.alpha_), int(np.count_nonzero(c)), time.time()-t0), flush=True)
    return c

c_l = run("LASSO", True)
c_a = run("ALASSO", True)
print("LASSO nnz=%d  ALASSO nnz=%d" % (np.count_nonzero(c_l), np.count_nonzero(c_a)), flush=True)
print("DONE", flush=True)

