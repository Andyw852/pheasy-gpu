import os, time, numpy as np, torch
from pheasy_gpu.core import gpu_backend as gb
from pheasy_gpu.core.optimizer import Optimizer
from scipy import sparse as sp

d = "<DATA_DIR>/MnIn2Se4_c7"  # set to your c7 scan directory
sm = sp.load_npz(os.path.join(d, "sm_prime.npz"))
nsh = sp.load_npz(os.path.join(d, "ns_harm.npz"))
nsa = sp.load_npz(os.path.join(d, "ns_anharm3.npz"))

def mb(x): return "%.0f MB" % (x / 1e6)

# ---- full-size (n=45) memory profile ----
torch.cuda.reset_peak_memory_stats()
SM = gb.load_sensing_matrix(sm, nsh, nsa, 25515, dtype=np.float64)
print("[n=45] SM %s | peak %.0f MB | current %.0f MB" % (SM.shape, torch.cuda.max_memory_allocated()/1e6, torch.cuda.memory_allocated()/1e6), flush=True)

At = torch.as_tensor(np.ascontiguousarray(SM), dtype=torch.float64, device="cuda:0")
print("[n=45] At tensor: %.0f MB" % (At.numel()*8/1e6), flush=True)

torch.cuda.reset_peak_memory_stats()
U,S,Vh = torch.linalg.svd(At, full_matrices=False)
print("[n=45] SVD(OLS/RIDGE): peak %.0f MB  (U=%.0f Vh=%.0f)" % (torch.cuda.max_memory_allocated()/1e6, U.numel()*8/1e6, Vh.numel()*8/1e6), flush=True)
del U, S, Vh; torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()
yt = torch.randn(25515, dtype=torch.float64, device="cuda:0")
coef = torch.linalg.lstsq(At, yt, driver="gels").solution
print("[n=45] QR(gels, RFE): peak %.0f MB" % (torch.cuda.max_memory_allocated()/1e6), flush=True)
del coef, yt; torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()
G = At.T @ At
print("[n=45] Gram(FISTA LASSO/ALASSO): peak %.0f MB  (G=%.0f)" % (torch.cuda.max_memory_allocated()/1e6, G.numel()*8/1e6), flush=True)
del G, At, SM; torch.cuda.empty_cache()

# ---- RFE-OLS-TSQR smoke test (n=8) ----
os.environ["PHEASY_CV_GROUP_SIZE"] = "567"
SM = gb.load_sensing_matrix(sm, nsh, nsa, 8*567, dtype=np.float64)
fm = np.load(os.path.join(d, "fm1d.npz"))
key = [k for k in fm if not k.startswith("_")][0]
F = np.asarray(fm[key], dtype=np.float64).ravel()[:8*567]

t0 = time.time()
o = Optimizer("RFE-OLS-TSQR", cv=3, tol=1e-5, max_iter=2000, rand_seed=0, standardize=False, use_gpu=True)
o.fit(SM, F)
c = o.results["coef"]
print("[TSQR] nnz=%d/%d  %.1fs" % (int(np.count_nonzero(c)), len(c), time.time()-t0), flush=True)
print("DONE", flush=True)

