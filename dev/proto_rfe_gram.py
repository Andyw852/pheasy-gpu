#!/usr/bin/env python3
"""Phase-0 correctness gate for the streaming RFE-OLS Gram engine.

SPEC_gpu_sm_streaming.md, phase 0: a synthetic two-level operator (SM_prime @ NS,
the production layout: a protected "harmonic" block with ~40x column norms and
an eliminable "anharmonic" block with a sparse truth), 5 grouped folds, the
production RFE settings (step 0.10, patience 5, min_features 1), run through
PheasyRFECV once per engine:

  lsmr         the iterative path the c3=4.5 fit uses today (Gram engines off,
               LSMR tolerances tightened so it converges)
  gram_ram     the existing in-RAM float64 engine (_RFEGramCV)
  stream_ram   PHEASY_RFE_GRAM_CV=1, RAM store, float64
  stream_disk  PHEASY_RFE_GRAM_CV=1, .npy memmap store, float64
  stream_1buf  PHEASY_RFE_GRAM_CV=1, one round buffer, no total slot
  stream_gpu   PHEASY_RFE_GRAM_CV=1 with GPU accumulation (CUDA if present,
               else the same torch code on the CPU device)

Gate (exit 1 on failure): every engine reproduces the reference's round
sequence (n_active), per-round CV_RMSE and final coefficients to --tol relative
(default 1e-6), and the same final support.  The reference is gram_ram; lsmr is
compared to it as well, which is the spec's "equality against the current LSMR
loop".  The default operator is float64: the gate tests the algorithm, and a
float32 LSMR stops at its precision floor far above 1e-6.

  python dev/proto_rfe_gram.py                       # p = 2000 (spec size)
  python dev/proto_rfe_gram.py --p3 300 --p2 60      # quick
  python dev/proto_rfe_gram.py --modes gram_ram,stream_disk
"""
import argparse
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
import warnings

import numpy as np
import scipy.sparse as sp

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from core import optimizer as opt  # noqa: E402

ALL_MODES = ("lsmr", "gram_ram", "stream_ram", "stream_disk", "stream_1buf", "stream_gpu")


def make_problem(seed=0, n_cfg=100, rows_per=120, m=None, p2=400, p3=1600, k3=160,
                 noise=0.02, dtype=np.float64):
    """TwoLevelSM(SM_prime, NS) with a protected FC2-like block.

    Returns (A, y, protected_indices, group_size).  Rows come in groups of
    rows_per (one "configuration"); the CV is grouped on them.
    """
    r = np.random.default_rng(seed)
    p = p2 + p3
    m = int(m or (3 * p) // 2)
    n = n_cfg * rows_per
    prime = sp.random(n, m, density=min(1.0, 40.0 / m), random_state=r,
                      format="csr", dtype=np.float64)
    ns = (sp.random(m, p, density=min(1.0, 3.0 / m), random_state=r, format="csr")
          + sp.eye(m, p, format="csr")).tocsr()
    scale = np.ones(p)
    scale[:p2] = 40.0                       # FC2-like norm spread
    ns = (ns @ sp.diags(scale)).tocsr()
    c = np.zeros(p)
    c[:p2] = r.standard_normal(p2) * 0.5 / 40.0
    nz = p2 + r.choice(p3, min(k3, p3), replace=False)
    c[nz] = r.standard_normal(nz.size) * np.geomspace(5, 0.01, nz.size)
    A = opt.TwoLevelSM(prime.astype(dtype), ns.astype(np.float64))
    y = np.asarray(A @ c, dtype=np.float64)
    y = y + noise * np.std(y) * r.standard_normal(n)
    return A, y, np.arange(p2), rows_per


def mode_env(mode, group_size):
    env = {"PHEASY_CV_GROUP_SIZE": str(group_size), "PHEASY_GPU_SM": "0",
           "PHEASY_RFE_GRAM_CV": "0"}
    env["PHEASY_GPU_MODE"] = "auto" if mode == "stream_gpu" else "cpu"
    if mode == "lsmr":
        env.update(PHEASY_RFE_GRAM_GB="0", PHEASY_LSQR_ATOL="1e-14",
                   PHEASY_LSQR_BTOL="1e-14", PHEASY_LSQR_MAXITER="100000",
                   PHEASY_GPU_RFE_RESIDENT="0")
    elif mode == "gram_ram":
        env.update(PHEASY_RFE_GRAM_GB="1000")
    else:
        env.update(PHEASY_RFE_GRAM_CV="1", PHEASY_RFE_GRAM_STORE="ram",
                   PHEASY_RFE_GRAM_DTYPE="float64")
        if mode == "stream_disk":
            env.update(PHEASY_RFE_GRAM_STORE="disk")
        elif mode == "stream_1buf":
            env.update(PHEASY_RFE_GRAM_ROUND_BUFFERS="1", PHEASY_RFE_GRAM_TOTAL="0")
        elif mode == "stream_gpu":
            env.update(PHEASY_RFE_GRAM_GPU="1")
            try:
                import torch
                if not torch.cuda.is_available():
                    env.update(PHEASY_RFE_GRAM_GPU_DEVICES="cpu",
                               PHEASY_RFE_GRAM_GPU_GB="2")
            except ImportError:
                pass
    return env


def run(mode, A, y, protected, group_size, extra_env=None, verbose=False):
    """Fit PheasyRFECV with one engine; returns rounds, support, coef, metadata."""
    env = mode_env(mode, group_size)
    env.update(extra_env or {})
    model = opt.PheasyRFECV(step=0.10, cv=5, n_jobs=1, verbose=True, random_state=0,
                            patience=5, min_features=1, protected=protected,
                            lsmr_maxiter=100000, lsmr_atol=1e-14, lsmr_btol=1e-14)
    buf = io.StringIO()
    old = os.getcwd()
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="proto_rfe_") as tmp:
        os.chdir(tmp)                      # fit() writes rfe_support.npy (+ the store)
        try:
            with patch.dict(os.environ, env), warnings.catch_warnings(), \
                    contextlib.redirect_stdout(buf):
                warnings.simplefilter("ignore")
                model.fit(A, y)
            leftovers = sorted(os.listdir(os.path.join(tmp, "rfe_gram_store"))) \
                if os.path.isdir(os.path.join(tmp, "rfe_gram_store")) else []
        finally:
            os.chdir(old)
    log = buf.getvalue()
    if verbose:
        print(log)
    # full-precision per-round CV (the log prints 7 digits, too coarse for 1e-6)
    rounds = [(n_act, cv) for n_act, cv, _se in model.rfe_history_]
    return {"mode": mode, "rounds": rounds, "support": model.support_.copy(),
            "coef": model.coef_.copy(), "meta": model.backend_metadata_,
            "seconds": time.time() - t0, "log": log, "store_leftovers": leftovers}


def compare(ref, got):
    """Max relative differences of got against ref (None when the rounds differ)."""
    same_rounds = [r[0] for r in got["rounds"]] == [r[0] for r in ref["rounds"]]
    cv = None
    if same_rounds and ref["rounds"]:
        a = np.array([r[1] for r in got["rounds"]])
        b = np.array([r[1] for r in ref["rounds"]])
        cv = float(np.max(np.abs(a - b) / np.abs(b)))
    scale = float(np.abs(ref["coef"]).max()) or 1.0
    return {"same_rounds": same_rounds, "cv_rel": cv,
            "coef_rel": float(np.abs(got["coef"] - ref["coef"]).max() / scale),
            "same_support": bool(np.array_equal(got["support"], ref["support"]))}


def gate(results, tol):
    ref = results["gram_ram"] if "gram_ram" in results else next(iter(results.values()))
    ok = True
    rows = []
    for mode, res in results.items():
        c = compare(ref, res)
        passed = (c["same_rounds"] and c["same_support"]
                  and (c["cv_rel"] is None or c["cv_rel"] <= tol)
                  and c["coef_rel"] <= tol and not res["store_leftovers"])
        ok &= passed
        rows.append((mode, res, c, passed))
    return ok, ref["mode"], rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--p2", type=int, default=400, help="protected (harmonic) columns")
    ap.add_argument("--p3", type=int, default=1600, help="eliminable (anharmonic) columns")
    ap.add_argument("--cfg", type=int, default=100, help="configurations")
    ap.add_argument("--rows", type=int, default=120, help="rows per configuration")
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = sorted(set(modes) - set(ALL_MODES))
    if unknown:
        ap.error("unknown modes: %s" % ", ".join(unknown))
    A, y, protected, g = make_problem(seed=args.seed, n_cfg=args.cfg, rows_per=args.rows,
                                      p2=args.p2, p3=args.p3, dtype=np.dtype(args.dtype))
    print("[proto] operator %s (%d protected), %d rows in groups of %d, dtype %s"
          % (A.shape, protected.size, A.shape[0], g, args.dtype), flush=True)
    results = {}
    for mode in modes:
        try:
            results[mode] = run(mode, A, y, protected, g, verbose=args.verbose)
        except Exception as exc:              # report, then fail the gate
            print("[proto] %-12s FAILED: %s: %s" % (mode, type(exc).__name__, exc), flush=True)
            return 1
        res = results[mode]
        print("[proto] %-12s %2d rounds, n_active %s -> %d selected, %.1fs"
              % (mode, len(res["rounds"]), res["rounds"][0][0] if res["rounds"] else "-",
                 int(res["support"].sum()), res["seconds"]), flush=True)
    ok, ref, rows = gate(results, args.tol)
    print("\n[proto] against %s (tol %.0e):" % (ref, args.tol))
    print("  %-12s %-7s %-8s %-11s %-11s %s" % ("mode", "rounds", "support", "CV_RMSE", "coef", ""))
    for mode, res, c, passed in rows:
        print("  %-12s %-7s %-8s %-11s %-11s %s" % (
            mode, "same" if c["same_rounds"] else "DIFF",
            "same" if c["same_support"] else "DIFF",
            "%.1e" % c["cv_rel"] if c["cv_rel"] is not None else "-",
            "%.1e" % c["coef_rel"], "ok" if passed else "FAIL" + (
                " (store files left: %s)" % res["store_leftovers"] if res["store_leftovers"] else "")))
    print("\n[proto] %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
