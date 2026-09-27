#!/usr/bin/env python3
"""Validate the matrix-free Gram ARDR (large systems).

Three checks:

1. EQUIVALENCE: _ardr_evidence_gram(G, b, yty) reproduces
   sklearn.linear_model.ARDRegression bit-for-bit on the same problem (both
   call scipy.linalg.pinvh), so the Gram reformulation does not change the
   method.
2. TWO-LEVEL CONSISTENCY: fitting Optimizer('ARDR') on a TwoLevelSM operator
   (matrix-free Gram path) gives the same coefficients as fitting the explicit
   dense SM.
3. LARGE SYSTEM: run ARDR on a design matrix that is never materialized --
   n samples x p features with n * p far above host RAM -- and show the
   sparse support is recovered. This is the path that lets ARDR run without
   the n_samples x n_features dense matrix; the p x p Gram remains, which is
   ARD's hard O(p^2) / O(p^3) wall.

Usage:
    python dev/test_ardr_gram.py
    python dev/test_ardr_gram.py --p 10000 --mid 10000 --n 200000
    python dev/test_ardr_gram.py --skip-large
"""
import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]


def bootstrap():
    tmp = Path(tempfile.mkdtemp(prefix="pheasy_ardr_gram_"))
    pkg = tmp / "pheasy_gpu"
    pkg.mkdir(parents=True)
    pkg.joinpath("__init__.py").write_text(
        "from pathlib import Path\n"
        "_root = Path(%r)\n" % str(ROOT)
        + "__path__ = [str(_root)]\n"
        "exec(compile((_root / '__init__.py').read_text(), str(_root / '__init__.py'), 'exec'))\n"
    )
    sys.path.insert(0, str(tmp))
    return tmp


def problem(rng, n, p, s, noise=0.01):
    X = rng.standard_normal((n, p))
    if p > 60:
        X[:, 60:] += 0.6 * X[:, :p - 60]
    true = np.zeros(p)
    true[rng.choice(p, s, replace=False)] = rng.uniform(1.0, 3.0, s)
    y = X @ true + noise * rng.standard_normal(n)
    return X, true, y


def check_equivalence(rng):
    from sklearn.linear_model import ARDRegression
    from pheasy_gpu.core.optimizer import _ardr_evidence_gram
    n, p, s = 800, 200, 20
    X, true, y = problem(rng, n, p, s)
    kw = dict(threshold_lambda=1e4, max_iter=300, tol=1e-3, fit_intercept=False)
    ref = ARDRegression(**kw).fit(X, y)
    G = X.T @ X
    b = X.T @ y
    coef, alpha_, lam, n_iter, reason, sse = _ardr_evidence_gram(
        G, b, float(y @ y), n, float(np.var(y)), max_iter=300, tol=1e-3)
    dcoef = float(np.max(np.abs(coef - ref.coef_))
                  / max(float(np.max(np.abs(ref.coef_))), 1e-30))
    dlam = float(np.max(np.abs(lam - ref.lambda_))
                 / max(float(np.max(np.abs(ref.lambda_))), 1e-30))
    nz_ref, nz_gram = int(np.count_nonzero(ref.coef_)), int(np.count_nonzero(coef))
    print("[equiv] sklearn n_iter=%d nz=%d | gram n_iter=%d nz=%d reason=%s"
          % (ref.n_iter_, nz_ref, n_iter, nz_gram, reason))
    print("[equiv] rel coef=%.3e  rel lambda=%.3e  alpha %.6e vs %.6e"
          % (dcoef, dlam, alpha_, ref.alpha_))
    assert n_iter == ref.n_iter_, (n_iter, ref.n_iter_)
    assert nz_gram == nz_ref, (nz_gram, nz_ref)
    assert dcoef < 1e-8, dcoef
    assert dlam < 1e-8, dlam
    assert abs(alpha_ - ref.alpha_) <= 1e-6 * abs(ref.alpha_) + 1e-300
    return True


def check_twolevel_consistency(rng):
    from pheasy_gpu.core.optimizer import Optimizer, TwoLevelSM
    n, mid, p, s = 20000, 600, 400, 25
    SM_prime = sp.random(n, mid, density=max(2.0 / mid, 0.004), format="csr",
                         random_state=1).astype(np.float64)
    NS = sp.random(mid, p, density=max(2.0 / mid, 0.01), format="csr",
                   random_state=2).astype(np.float64)
    true = np.zeros(p)
    true[rng.choice(p, s, replace=False)] = rng.uniform(1.0, 3.0, s)
    y = np.asarray(TwoLevelSM(SM_prime, NS) @ true, dtype=np.float64)
    y = y + 0.01 * rng.standard_normal(n)
    op = TwoLevelSM(SM_prime, NS)
    opt_op = Optimizer("ARDR", cv=5, rand_seed=0, standardize=False)
    opt_op.fit(op, y)
    dense = np.asarray((SM_prime @ NS).toarray(), dtype=np.float64)
    opt_dense = Optimizer("ARDR", cv=5, rand_seed=0, standardize=False)
    opt_dense.fit(dense, y)
    c_op = np.asarray(opt_op.results["coef"], dtype=np.float64)
    c_de = np.asarray(opt_dense.results["coef"], dtype=np.float64)
    rel = float(np.max(np.abs(c_op - c_de)) / max(float(np.max(np.abs(c_de))), 1e-30))
    backend = opt_op.results.get("execution_backend")
    print("[twolevel] dense vs operator rel coef=%.3e  backend=%s  nz=%d/%d"
          % (rel, backend, np.count_nonzero(c_op), np.count_nonzero(c_de)))
    assert rel < 1e-6, rel
    assert backend == "cpu_gram_ardr", backend
    return True


def check_no_p(rng):
    """The P-free block Gram must reproduce the factorized P route exactly."""
    from pheasy_gpu.core.optimizer import Optimizer, TwoLevelSM
    n, mid, p, s = 20000, 2000, 400, 30
    SM_prime = sp.random(n, mid, density=max(8.0 / mid, 0.004), format="csr",
                         random_state=11).astype(np.float64)
    NS = sp.random(mid, p, density=max(6.0 / mid, 0.01), format="csr",
                   random_state=12).astype(np.float64)
    true = np.zeros(p)
    true[rng.choice(p, s, replace=False)] = rng.uniform(1.0, 3.0, s)
    op = TwoLevelSM(SM_prime, NS)
    y = np.asarray(op @ true, dtype=np.float64) + 0.01 * rng.standard_normal(n)

    def fit(env):
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            o = Optimizer("ARDR", cv=5, rand_seed=0, standardize=False)
            o.fit(op, y)
            return (np.asarray(o.results["coef"], dtype=np.float64),
                    o.results.get("regularized_solver_info") or {})
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    for k in ("PHEASY_ARDR_GRAM_NO_P", "PHEASY_ARDR_GRAM_MAX_GB"):
        os.environ.pop(k, None)
    c_p, info_p = fit({})
    c_b, info_b = fit({"PHEASY_ARDR_GRAM_NO_P": "1"})
    # A budget too small for P (mid^2) but large enough for G (p^2) must fall back.
    c_f, info_f = fit({"PHEASY_ARDR_GRAM_MAX_GB": "0.01"})
    rel_b = float(np.max(np.abs(c_p - c_b)) / max(float(np.max(np.abs(c_p))), 1e-30))
    rel_f = float(np.max(np.abs(c_p - c_f)) / max(float(np.max(np.abs(c_p))), 1e-30))
    print("[no-P] construction P=%s block=%s auto=%s | rel block=%.3e auto=%.3e"
          % (info_p.get("gram_construction"), info_b.get("gram_construction"),
             info_f.get("gram_construction"), rel_b, rel_f))
    assert info_p.get("gram_construction") == "P"
    assert info_b.get("gram_construction") == "block"
    assert info_f.get("gram_construction") == "block"
    assert rel_b < 1e-8 and rel_f < 1e-8, (rel_b, rel_f)
    return True


def check_large(rng, n, mid, p, s=300):
    from pheasy_gpu.core.optimizer import Optimizer, TwoLevelSM
    nnz_row = max(8, min(20, mid))
    SM_prime = sp.random(n, mid, density=nnz_row / float(mid), format="csr",
                         random_state=3).astype(np.float64)
    NNZ_NS = max(2, min(6, mid))
    NS = sp.random(mid, p, density=NNZ_NS / float(mid), format="csr",
                   random_state=4).astype(np.float64)
    op = TwoLevelSM(SM_prime, NS)
    true = np.zeros(p)
    true[rng.choice(p, s, replace=False)] = rng.uniform(1.0, 3.0, s)
    t0 = time.time()
    y = np.asarray(op @ true, dtype=np.float64)
    y = y + 0.01 * rng.standard_normal(n)
    dense_gb = n * p * 8 / 1e9
    gram_gb = p * p * 8 / 1e9
    print("[large] n=%d mid=%d p=%d nnz(SM_prime)=%d nnz(NS)=%d" % (n, mid, p, SM_prime.nnz, NS.nnz))
    print("[large] explicit SM would be %.1f GB; Gram G is %.2f GB" % (dense_gb, gram_gb))
    os.environ["PHEASY_ARDR_MAX_ITER"] = os.environ.get("PHEASY_ARDR_MAX_ITER", "40")
    opt = Optimizer("ARDR", cv=5, rand_seed=0, standardize=False)
    opt.fit(op, y)
    c = np.asarray(opt.results["coef"], dtype=np.float64)
    info = opt.results.get("regularized_solver_info") or {}
    nz = int(np.count_nonzero(c))
    true_sup = np.flatnonzero(true)
    recovered = int(np.count_nonzero(c[true_sup]))
    t = time.time() - t0
    print("[large] backend=%s iters=%s reason=%s nz=%d/%d (true %d, recovered %d) "
          "rmse=%.3e  %.0fs"
          % (opt.results.get("execution_backend"), info.get("n_iter"),
             info.get("stop_reason"), nz, p, s, recovered, opt.metrics["rmse"], t))
    assert opt.results.get("execution_backend") == "cpu_gram_ardr"
    assert nz < p // 4, nz
    assert recovered >= int(0.9 * s), (recovered, s)
    assert np.isfinite(opt.metrics["rmse"]) and opt.metrics["rmse"] > 0
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150000)
    ap.add_argument("--mid", type=int, default=5000)
    ap.add_argument("--p", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-large", action="store_true")
    args = ap.parse_args()
    tmp = bootstrap()
    try:
        rng = np.random.default_rng(args.seed)
        check_equivalence(rng)
        check_twolevel_consistency(rng)
        check_no_p(rng)
        if not args.skip_large:
            check_large(rng, args.n, args.mid, args.p)
        print("ALL ARDR GRAM CHECKS PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
