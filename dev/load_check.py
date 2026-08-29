import os, time, numpy as np, torch
from pheasy_gpu.core import gpu_backend as gb
from scipy import sparse as sp

d = "<DATA_DIR>/MnIn2Se4_c7"  # set to your c7 scan directory
sm = sp.load_npz(os.path.join(d, "sm_prime.npz"))
nsh = sp.load_npz(os.path.join(d, "ns_harm.npz"))
nsa = sp.load_npz(os.path.join(d, "ns_anharm3.npz"))

n = 8*567
SM_gpu = gb.load_sensing_matrix(sm, nsh, nsa, n, dtype=np.float64)
nsh2 = nsh.toarray(); nsa2 = nsa.toarray()
nh,mh = nsh2.shape; na,ma = nsa2.shape
NS = np.zeros((nh+na, mh+ma)); NS[:nh,:mh]=nsh2; NS[nh:,mh:]=nsa2
SM_cpu = (sm[:n] @ NS)   # slice BEFORE the multiply
err = float(np.linalg.norm(SM_gpu - SM_cpu) / np.linalg.norm(SM_cpu))
print("CSR vs scipy rel diff = %.3e" % err, flush=True)
print("OK" if err < 1e-10 else "MISMATCH", flush=True)

