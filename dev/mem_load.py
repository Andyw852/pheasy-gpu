import os, time, numpy as np, torch
from pheasy_gpu.core import gpu_backend as gb
from scipy import sparse as sp

d = "<DATA_DIR>/MnIn2Se4_c7"  # set to your c7 scan directory
sm = sp.load_npz(os.path.join(d, "sm_prime.npz"))
nsh = sp.load_npz(os.path.join(d, "ns_harm.npz"))
nsa = sp.load_npz(os.path.join(d, "ns_anharm3.npz"))

t0 = time.time()
torch.cuda.reset_peak_memory_stats()
SM = gb.load_sensing_matrix(sm, nsh, nsa, 25515, dtype=np.float64)
print("[n=45 CSR] SM %s | peak %.0f MB | %.1fs" % (SM.shape, torch.cuda.max_memory_allocated()/1e6, time.time()-t0), flush=True)
print("DONE", flush=True)

