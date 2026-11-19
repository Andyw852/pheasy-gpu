"""并行 CPU 列范数：与 TwoLevelSM.col_norms 的 float64 定义逐字一致，
只在行块之间并行（行块互相独立，因此结果与串行完全相同到求和顺序）。

用法: python colnorm_parallel.py <case_dir> <out.npy> [workers]
"""
import os, sys, time
import numpy as np
import scipy.sparse as sp

CASE = sys.argv[1]
OUT = sys.argv[2]
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 32
BLOCK_BYTES = 64 << 20


def load_prime(case):
    """Load sm_prime.npz as CSR, narrowing the COLUMN indices to int32.

    The npz stores int64 indices (the narrow int32 layout is only used by the
    fit's loader).  At c3=5.0 that is 60 GB of index data on top of the 30 GB of
    values, so the load alone would need ~91 GB; the column indices are below
    2**31 by construction (they index <=136,251 columns), so int32 is exact and
    halves the load.  The indptr stays int64: nnz=7.5e9 does not fit int32.
    """
    z = np.load(os.path.join(case, "sm_prime.npz"))
    shape = tuple(int(x) for x in z["shape"])
    idx = z["indices"]
    if idx.dtype == np.int64 and idx.size and int(idx.max()) < 2**31:
        idx = idx.astype(np.int32)
    return sp.csr_matrix((z["data"], idx, z["indptr"]), shape=shape)


def load_ns(case):
    blocks = []
    for f in ("ns_harm.npz", "ns_anharm3.npz"):
        z = np.load(os.path.join(case, f))
        sh = tuple(int(x) for x in z["shape"])
        blocks.append(sp.coo_matrix((z["data"], (z["row"], z["col"])), shape=sh).tocsr())
    return sp.block_diag(blocks, format="csr")


PRIME = None
NS = None
NROWS = 0
NCOLS = 0
MAXROWS = 0


def _init(prime, ns, nrows, ncols, maxrows):
    global PRIME, NS, NROWS, NCOLS, MAXROWS
    PRIME, NS, NROWS, NCOLS, MAXROWS = prime, ns, nrows, ncols, maxrows


def _chunk(i0):
    """算 [i0, i1) 行块的平方和（与 CPU 定义同一算式）。"""
    i1 = min(NROWS, i0 + MAXROWS)
    while i1 > i0 + 1:
        input_nnz = int(PRIME.indptr[i1] - PRIME.indptr[i0])
        if 24 * ((i1 - i0) * NCOLS + input_nnz) <= BLOCK_BYTES:
            break
        i1 = i0 + max(1, (i1 - i0) // 2)
    prod = PRIME[i0:i1].astype(np.float64, copy=False) @ NS
    sq = np.zeros(NCOLS, dtype=np.float64)
    if sp.issparse(prod):
        prod = prod.tocsr()
        prod.sum_duplicates()
        sq += np.bincount(prod.indices, weights=np.square(prod.data), minlength=NCOLS)
    else:
        prod = np.asarray(prod, dtype=np.float64)
        sq += np.einsum("ij,ij->j", prod, prod)
    return sq


def main():
    import multiprocessing as mp
    t0 = time.time()
    print("[colnorm] 载入 sm_prime ...", flush=True)
    prime = load_prime(CASE)
    print("  SM_prime %s nnz=%d (%.0fs)" % (prime.shape, prime.nnz, time.time() - t0), flush=True)
    ns = load_ns(CASE)
    print("  NS %s nnz=%d" % (ns.shape, ns.nnz), flush=True)
    n_rows, n_cols = prime.shape[0], ns.shape[1]
    max_rows = max(1, BLOCK_BYTES // (24 * n_cols))
    starts = list(range(0, n_rows, max_rows))
    print("  %d 个行块, %d workers" % (len(starts), WORKERS), flush=True)

    ctx = mp.get_context("fork")
    acc = np.zeros(n_cols, dtype=np.float64)
    t1 = time.time()
    with ctx.Pool(WORKERS, initializer=_init,
                  initargs=(prime, ns, n_rows, n_cols, max_rows)) as pool:
        done = 0
        for part in pool.imap_unordered(_chunk, starts, chunksize=1):
            acc += part
            done += 1
            if done % max(1, len(starts) // 10) == 0:
                el = time.time() - t1
                print("  %d/%d 块  %.0fs  预计剩余 %.0fs"
                      % (done, len(starts), el, el / done * (len(starts) - done)), flush=True)
    norms = np.sqrt(acc)
    np.save(OUT, norms)
    print("[colnorm] 完成 %.1f 分钟 -> %s  (%d 个值)"
          % ((time.time() - t0) / 60, OUT, norms.size), flush=True)
    print("  范围 [%.4e, %.4e]" % (norms.min(), norms.max()), flush=True)


if __name__ == "__main__":
    main()
