#!/usr/bin/env python3
"""Real-GPU check of GpuSparseMV with more row shards than cards.

The n_shards > n_cards path (row_shard_count, commit 8b1e331) is covered only by
the fake-torch tests in dev/test_gpu_memory_regressions.py.  Run this once on
real cards before trusting a fit on it: it builds GpuSparseMV on the given
devices with the given shard count and compares matvec / rmatvec against a
float64 scipy reference computed in row chunks (no full float64 copy).

Always pass the card list (--devices).  Without an explicit list GpuSparseMV's
auto filter sizes every card for a STORED transpose (2 x SM / n_gpu): for the
c3=4.5 matrix on 2 cards that is 35.9 GiB per card, so no 24 GB card qualifies
and construction fails with "no usable CUDA device" before the sharding code
runs.  With --devices the per-block VRAM pre-flight is the only guard.

Examples (run from the repository root):

  # c3=4.5 matrix, 2 cards, 4 shards, low-mem layout (~19.7 GiB per card)
  python dev/check_gpu_sm_shards.py --npz $CASE/sm_prime.npz \\
      --devices 0,1 --shards 4 --lowmem

  # synthetic smoke test, 2 cards, 4 shards, both layouts
  python dev/check_gpu_sm_shards.py --synthetic 400000x20000 --density 2e-3 \\
      --devices 0,1 --shards 4
  python dev/check_gpu_sm_shards.py --synthetic 400000x20000 --density 2e-3 \\
      --devices 0,1 --shards 4 --lowmem

Exit status 0 when both relative errors are below --rtol, 1 otherwise.
"""
import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_ROOT = Path(__file__).resolve().parents[1]


def _backend():
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from core import gpu_backend
    return gpu_backend


@contextlib.contextmanager
def _env(**values):
    """Set (str) or unset (None) environment variables for the block."""
    saved = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _chunks(sm, max_nnz):
    """Row ranges of sm holding at most ~max_nnz nonzeros each."""
    indptr = sm.indptr
    r0, n = 0, sm.shape[0]
    while r0 < n:
        target = int(indptr[r0]) + int(max_nnz)
        r1 = int(np.searchsorted(indptr, target, side="right")) - 1
        r1 = min(n, max(r1, r0 + 1))
        yield r0, r1
        r0 = r1


def reference(sm, x, u, max_nnz=1 << 26):
    """float64 SM @ x and SM.T @ u, one row chunk at a time."""
    y = np.empty(sm.shape[0], dtype=np.float64)
    z = np.zeros(sm.shape[1], dtype=np.float64)
    for r0, r1 in _chunks(sm, max_nnz):
        blk = sm[r0:r1].astype(np.float64)
        y[r0:r1] = blk @ x
        z += blk.T @ u[r0:r1]
    return y, z


def _rel(a, b):
    den = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b)) / (den if den > 0 else 1.0)


def check(sm, devices, shards=None, lowmem=False, seed=0, backend=None,
          reps=1, log=print):
    """Build GpuSparseMV(sm) on `devices`, compare matvec/rmatvec to scipy.

    Returns a dict with the shard layout, the two relative errors and the
    per-call wall times.  `backend` defaults to core.gpu_backend (tests pass
    the module they patched).
    """
    gb = backend or _backend()
    devices = [int(d) for d in devices]
    env = {
        "PHEASY_GPU_SM_SHARDS": shards if shards else None,
        "PHEASY_GPU_SM_LOWMEM": "1" if lowmem else "0",
        "PHEASY_GPU_SM_DEVICES": ",".join(str(d) for d in devices),
    }
    with _env(**env):
        derived = gb.row_shard_count(int(sm.nnz), len(devices))
        t0 = time.time()
        op = gb.GpuSparseMV(sm, n_gpu=len(devices), device_ids=devices)
        t_build = time.time() - t0
    try:
        rs = [int(r) for r in op._rs]
        row_nnz = [int(sm.indptr[b] - sm.indptr[a]) for a, b in zip(rs[:-1], rs[1:])]
        log("[check] %d shard(s) on cards %s (row_shard_count would give %d), "
            "layout=%s, build %.1fs"
            % (op._n_shards, op._block_devs, derived,
               "low-mem" if lowmem else "stored-transpose", t_build))
        log("[check] per-shard row nnz %s (max %d, int32 limit %d)"
            % (row_nnz, max(row_nnz), gb._CUDA_INT32_LIMIT))
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(sm.shape[1])
        u = rng.standard_normal(sm.shape[0])
        t0 = time.time()
        y_ref, z_ref = reference(sm, x, u)
        log("[check] scipy float64 reference %.1fs" % (time.time() - t0))
        t_mv, t_rmv = [], []
        for _ in range(max(1, int(reps))):
            t0 = time.time()
            y = op.matvec(x)
            t_mv.append(time.time() - t0)
            t0 = time.time()
            z = op.rmatvec(u)
            t_rmv.append(time.time() - t0)
        result = {
            "n_shards": int(op._n_shards),
            "block_devs": list(op._block_devs),
            "row_nnz": row_nnz,
            "matvec_rel_err": _rel(y, y_ref),
            "rmatvec_rel_err": _rel(z, z_ref),
            "matvec_s": min(t_mv),
            "rmatvec_s": min(t_rmv),
        }
        log("[check] matvec  rel err %.3e  (%.3fs)"
            % (result["matvec_rel_err"], result["matvec_s"]))
        log("[check] rmatvec rel err %.3e  (%.3fs)"
            % (result["rmatvec_rel_err"], result["rmatvec_s"]))
        return result
    finally:
        op.close()


def _load(args):
    if args.npz:
        gb = _backend()
        return gb._sparse_io().load_csr(args.npz)
    rows, cols = (int(v) for v in args.synthetic.lower().split("x"))
    rng = np.random.default_rng(args.seed)
    sm = sp.random(rows, cols, density=args.density, format="csr",
                   dtype=np.float64, random_state=rng)
    return sm.astype(np.dtype(args.dtype))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", help="sm_prime.npz to load (core.sparse_io.load_csr)")
    src.add_argument("--synthetic", metavar="ROWSxCOLS",
                     help="random CSR instead of a real matrix")
    p.add_argument("--density", type=float, default=1e-3)
    p.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    p.add_argument("--devices", required=True,
                   help="comma-separated CUDA indices, e.g. 0,1 (required)")
    p.add_argument("--shards", type=int, default=0,
                   help="force PHEASY_GPU_SM_SHARDS (0 = row_shard_count)")
    p.add_argument("--lowmem", action="store_true",
                   help="PHEASY_GPU_SM_LOWMEM=1 (row blocks only)")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rtol", type=float, default=0.0,
                   help="pass threshold (default 1e-4 for float32, 1e-12 for float64); "
                        "a misplaced or dropped shard gives an O(1) error")
    args = p.parse_args(argv)

    t0 = time.time()
    sm = _load(args)
    print("[check] matrix %s nnz=%d dtype=%s (%.1fs)"
          % (sm.shape, sm.nnz, sm.dtype, time.time() - t0), flush=True)
    rtol = args.rtol or (1e-4 if sm.dtype == np.float32 else 1e-12)
    res = check(sm, [d for d in args.devices.split(",") if d.strip()],
                shards=args.shards or None, lowmem=args.lowmem,
                seed=args.seed, reps=args.reps,
                log=lambda s: print(s, flush=True))
    ok = res["matvec_rel_err"] <= rtol and res["rmatvec_rel_err"] <= rtol
    print("[check] %s (rtol %.1e)" % ("PASS" if ok else "FAIL", rtol), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
