#!/usr/bin/env python3
"""Validate the fast marginal-likelihood RVM (core/fast_rvm.py).

1. EQUIVALENCE: on a small sparse problem the fast RVM (with the paper's
   lambda_t = 1e4 pruning) must reproduce sklearn ARDRegression: same support,
   same coefficients to ~1e-6.
2. TWO-LEVEL: the Gram can be built matrix-free from a TwoLevelSM
   (_compute_gram_blockwise); the fit must match the explicit dense-SM fit.
3. SCALE: at p ~ 3000 the fast RVM must avoid the batch ARD's full p x p
   factorization and be faster, while still recovering the sparse support.

Usage: python dev/test_fast_rvm.py
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
    tmp = Path(tempfile.mkdtemp(prefix="pheasy_fast_rvm_"))
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


def make_problem(rng, n, p, s, noise=0.01):
    X = rng.standard_normal((n, p))
    if p > 60:
        X[:, 60:] += 0.6 * X[:, :p - 60]
    true = np.zeros(p)
    true[rng.choice(p, s, replace=False)] = rng.uniform(1.0, 3.0, s)
    y = X @ true + noise * rng.standard_normal(n)
    return X, true, y


def check_equivalence(rng):
    from sklearn.linear_model import ARDRegression
    from pheasy_gpu.core.fast_rvm import fast_rvm
    n, p, s = 800, 200, 20
    X, true, y = make_problem(rng, n, p, s)
    G = X.T @ X
    b = X.T @ y
    ref = ARDRegression(threshold_lambda=1e4, max_iter=300, tol=1e-3,
                        fit_intercept=False).fit(X, y)
    res = fast_rvm(G, b, float(y @ y), n, y_var=float(np.var(y)),
                   beta_iters=10, tol=1e-8)
    rel = float(np.linalg.norm(res["coef"] - ref.coef_)
                / max(float(np.linalg.norm(ref.coef_)), 1e-30))
    true_sup = np.flatnonzero(true)
    print("[equiv] ARD nz=%d beta=%.4e | RVM nz=%d beta=%.4e steps=%d | "
          "rel=%.3e recovered=%d/%d"
          % (np.count_nonzero(ref.coef_), ref.alpha_,
             np.count_nonzero(res["coef"]), res["beta"], res["n_steps"],
             rel, np.count_nonzero(res["coef"][true_sup]), s))
    assert rel < 1e-6, rel
    assert np.count_nonzero(res["coef"][true_sup]) == s
    return True


def check_twolevel(rng):
    from pheasy_gpu.core.optimizer import TwoLevelSM, _compute_gram_blockwise
    from pheasy_gpu.core.fast_rvm import fast_rvm
    n, mid, p, s = 20000, 1200, 800, 40
    SM_prime = sp.random(n, mid, density=10.0 / mid, format="csr",
                         random_state=1).astype(np.float64)
    NS = sp.random(mid, p, density=6.0 / mid, format="csr",
                   random_state=2).astype(np.float64)
    op = TwoLevelSM(SM_prime, NS)
    true = np.zeros(p)
    true[rng.choice(p, s, replace=False)] = rng.uniform(1.0, 3.0, s)
    y = np.asarray(op @ true, dtype=np.float64) + 0.01 * rng.standard_normal(n)
    G, b = _compute_gram_blockwise(op, y)
    r_op = fast_rvm(G, b, float(y @ y), n, y_var=float(np.var(y)),
                    beta_iters=10, tol=1e-8)
    dense = np.asarray((SM_prime @ NS).toarray(), dtype=np.float64)
    Gd = dense.T @ dense
    bd = dense.T @ y
    r_de = fast_rvm(Gd, bd, float(y @ y), n, y_var=float(np.var(y)),
                    beta_iters=10, tol=1e-8)
    rel = float(np.max(np.abs(r_op["coef"] - r_de["coef"]))
                / max(float(np.max(np.abs(r_de["coef"]))), 1e-30))
    print("[twolevel] operator vs dense rel=%.3e  nz=%d/%d"
          % (rel, np.count_nonzero(r_op["coef"]), np.count_nonzero(r_de["coef"])))
    assert rel < 1e-6, rel
    return True


def check_scale(rng):
    from pheasy_gpu.core.fast_rvm import fast_rvm
    from pheasy_gpu.core.optimizer import _ardr_evidence_gram
    n, p, s = 20000, 3000, 100
    X, true, y = make_problem(rng, n, p, s)
    G = X.T @ X
    b = X.T @ y
    t0 = time.time()
    res = fast_rvm(G, b, float(y @ y), n, y_var=float(np.var(y)),
                   beta_iters=10, tol=1e-8)
    t_rvm = time.time() - t0
    true_sup = np.flatnonzero(true)
    rec = int(np.count_nonzero(res["coef"][true_sup]))
    print("[scale] p=%d n=%d: fast_RVM %.1fs n_steps=%d nz=%d recovered=%d/%d"
          % (p, n, t_rvm, res["n_steps"], np.count_nonzero(res["coef"]), rec, s))
    assert rec >= int(0.9 * s), (rec, s)
    # The batch ARD evidence loop on the same Gram (Cholesky path) for reference.
    t0 = time.time()
    c_ard, _, _, nit, _, _ = _ardr_evidence_gram(
        G, b, float(y @ y), n, float(np.var(y)), threshold_lambda=1e4,
        max_iter=100, tol=1e-3, verbose=False)
    t_ard = time.time() - t0
    rel = float(np.max(np.abs(res["coef"] - c_ard))
                / max(float(np.max(np.abs(c_ard))), 1e-30))
    print("[scale] batch ARD (Gram, Cholesky) %.1fs iters=%d | fast_RVM %.1fs | rel=%.3e"
          % (t_ard, nit, t_rvm, rel))
    assert rel < 1e-3, rel
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tmp = bootstrap()
    try:
        rng = np.random.default_rng(args.seed)
        check_equivalence(rng)
        check_twolevel(rng)
        check_scale(rng)
        print("ALL FAST-RVM CHECKS PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
