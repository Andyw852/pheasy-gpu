#!/usr/bin/env python3
"""Phase-level benchmark for pheasy-gpu on the real c2.6 TwoLevelSM data.

Measures the host-side cost of the remaining non-GPU phases so GPU-migration
decisions are driven by measured numbers, not guesswork:

  * alpha-grid   (derive_alpha_grid: A.T@y rmatvec + column norms)
  * column normalization / standardization (_col_norms + _scale_columns)
  * metrics      (one matvec + MSE/RMSE/MAE/R2 reductions)
  * RFE ranking  (|coef| + argsort)

Measured on c2.6 (99792 x 20125, nnz ~22M): the only host-side O(nnz) pass is
column norms (~3s); the rmatvec is ~0.02s, metrics ~0.02s and RFE ranking
~0.003s. Column norms are already computed on GPU inside the resident solver
(op.normalize()), so metrics and RFE ranking are the only remaining candidates
and their GPU round-trip would cost more than it saves.

Usage:
  python dev/benchmark_phases.py [--data-dir DIR] [--rows N]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse as sp


def load_c26(data_dir):
    spz = np.load(os.path.join(data_dir, "sm_prime.npz"))
    nsz = np.load(os.path.join(data_dir, "ns_harm.npz"))
    fmz = np.load(os.path.join(data_dir, "fm1d.npz"))
    sm = sp.csr_matrix((spz["data"], spz["indices"], spz["indptr"]),
                       shape=tuple(spz["shape"]))
    ns = sp.coo_matrix((nsz["data"], (nsz["row"], nsz["col"])),
                       shape=tuple(nsz["shape"])).tocsr()
    F = np.asarray(fmz["F"], dtype=np.float64).ravel()
    return sm, ns, F


def bench_phases(SM, F):
    from core.optimizer import derive_alpha_grid, _col_norms, _scale_columns

    out = {}

    t0 = time.perf_counter()
    derive_alpha_grid(SM, F, nalpha=20, decades=4.0, standardize=True)
    out["alpha_grid_std"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cn = _col_norms(SM)
    cn = np.where(cn < 1e-30, 1.0, cn)
    _scale_columns(SM, cn)
    out["standardize"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _col_norms(SM)
    out["col_norms_only"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    np.asarray(SM.T @ F).ravel()
    out["at_y_rmatvec"] = time.perf_counter() - t0

    rng = np.random.default_rng(0)
    coef = rng.normal(size=SM.shape[1])
    t0 = time.perf_counter()
    pred = np.asarray(SM @ coef).ravel()
    resid = pred - F
    mse = float(np.mean(resid ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(resid)))
    r2 = float(1 - np.sum(resid ** 2) / np.sum((F - F.mean()) ** 2))
    out["metrics"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    np.argsort(np.abs(coef))[::-1]
    out["rfe_ranking"] = time.perf_counter() - t0

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="tmp/resident_lasso_inputs_c2_6")
    ap.add_argument("--rows", type=int, default=0, help="0 = full matrix")
    args = ap.parse_args()

    os.environ.setdefault("PHEASY_MAX_CORES",
                          os.environ.get("PHEASY_MAX_CORES", "16"))

    sm, ns, F = load_c26(args.data_dir)
    if args.rows:
        sm = sm[:args.rows]
        F = F[:args.rows]
    from core.optimizer import TwoLevelSM
    SM = TwoLevelSM(sm, ns)
    print("TwoLevelSM shape=%s nnz(SM_prime)=%d" % (SM.shape, sm.nnz))

    res = bench_phases(SM, F)
    print()
    print("== host-side phase cost (seconds) ==")
    for k, v in res.items():
        print("  %-16s %.4fs" % (k, v))


if __name__ == "__main__":
    main()
