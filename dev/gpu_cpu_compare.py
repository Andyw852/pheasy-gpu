"""GPU lstsq vs CPU gelsd comparison on C60Mg2 (full 52283 free IFCs).

Builds the dense sensing matrix for the first NCONF configurations,
SM = SM_prime[:NROW] @ NS_full, then solves OLS both ways:
  - GPU: gpu_backend.lstsq  (torch.linalg.svd, min-norm, float64)
  - CPU: scipy.linalg.lstsq(driver="gelsd", cond=None)  (min-norm, float64)

Reports wall time and the coefficient agreement, so we can see the GPU
speedup and confirm the GPU solve reproduces the CPU gelsd reference.

Run from the C60Mg2_test/ directory (needs sm_prime.npz, ns_harm.npz,
ns_anharm3.npz, force_matrix.pkl).
"""
import os
import pickle
import time

import numpy as np
import scipy.sparse as sp
from scipy import linalg

NCONF = 7
NATOM = 496
NROW = NCONF * NATOM * 3          # 3 force components per atom

# single GPU for the dense SVD (the free ones are 1,2,4,5,6)
os.environ["PHEASY_USE_GPU"] = "1"
os.environ.setdefault("PHEASY_GPU_DEVICE", "1")

from pheasy_gpu.core import gpu_backend as gb   # noqa: E402


def main():
    t0 = time.time()

    # 1. sensing matrix + null space (block-diagonal, same as the fit)
    SM_prime = sp.load_npz("sm_prime.npz")
    ns_harm = sp.load_npz("ns_harm.npz")
    ns_anharm3 = sp.load_npz("ns_anharm3.npz")
    NS_full = sp.block_diag([ns_harm, ns_anharm3]).tocsc()
    print("SM_prime %s nnz=%d | NS_full %s" % (SM_prime.shape, SM_prime.nnz,
                                               NS_full.shape), flush=True)

    # 2. force targets for the first NCONF configs
    with open("force_matrix.pkl", "rb") as f:
        fmat = pickle.load(f)
    F = np.asarray(fmat[:NCONF], dtype=np.float64).reshape(-1)
    assert F.shape[0] == NROW, F.shape
    print("F shape", F.shape, flush=True)

    # 3. dense SM = SM_prime[:NROW] @ NS_full
    sm_blk = SM_prime[:NROW].tocsr()
    SM_sp = sm_blk @ NS_full                     # (NROW, 52283) sparse
    print("SM_sp nnz=%d (%.1f%% dense)" % (SM_sp.nnz,
                                           100.0 * SM_sp.nnz / (NROW * 52283)),
          flush=True)
    SM_dense = SM_sp.toarray()
    print("SM_dense %s (%.2f GB)" % (SM_dense.shape, SM_dense.nbytes / 1e9),
          flush=True)
    print("load+dense time %.1f s" % (time.time() - t0), flush=True)

    # 4. GPU solve
    dev = int(os.environ.get("PHEASY_GPU_DEVICE", "1"))
    free_gb = (gb._device_free_bytes(dev) or 0) / 1e9
    print("GPU %d free %.2f GB before solve" % (dev, free_gb), flush=True)
    t1 = time.time()
    coef_gpu = gb.lstsq(SM_dense, F)
    t_gpu = time.time() - t1
    print("GPU lstsq %.2f s  coef[:4]=%s" % (t_gpu, coef_gpu[:4]), flush=True)

    # 5. CPU solve (gelsd reference)
    t2 = time.time()
    coef_cpu, _, rank, _sv = linalg.lstsq(SM_dense, F, cond=None,
                                          lapack_driver="gelsd")
    t_cpu = time.time() - t2
    print("CPU gelsd %.2f s  coef[:4]=%s  rank=%d" % (t_cpu, coef_cpu[:4],
                                                      rank), flush=True)

    # 6. agreement
    diff = np.abs(coef_gpu - coef_cpu)
    rel = np.linalg.norm(diff) / max(np.linalg.norm(coef_cpu), 1e-30)
    print("max|diff| = %.3e   rel_norm_diff = %.3e" % (diff.max(), rel),
          flush=True)
    print("CPU/GPU time ratio = %.2fx" % (t_cpu / max(t_gpu, 1e-9)), flush=True)


if __name__ == "__main__":
    main()
