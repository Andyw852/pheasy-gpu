#!/usr/bin/env python3
"""End-to-end smoke test for an installed pheasy-gpu package.

Runs without touching the checkout's source tree: it imports `pheasy_gpu` from
site-packages (editable or wheel) and fits a small two-level OLS problem, then
checks the public acceptance metadata. Use it after `pip install -e .` to
confirm the console entry point and public API are usable.

    python dev/smoke_installed.py            # uses the ambient interpreter
    /path/to/venv/bin/python dev/smoke_installed.py
"""
import sys

import numpy as np
import scipy.sparse as sp

import pheasy_gpu
from pheasy_gpu.core import gpu_backend as gb
from pheasy_gpu.core.optimizer import Optimizer, TwoLevelSM


def main():
    print("pheasy_gpu package:", pheasy_gpu.__file__)
    print("GPU mode:", gb.gpu_mode_from_env())

    rng = np.random.default_rng(7)
    A = rng.standard_normal((80, 10))
    x_true = rng.standard_normal(10)
    y = A @ x_true + 1e-4 * rng.standard_normal(80)
    op = TwoLevelSM(sp.csr_matrix(A), sp.eye(10, format="csr"))

    opt = Optimizer("OLS", max_iter=2000)
    opt.fit(op, y)

    assert opt.results.get("fit_accepted") is True, opt.results.get("status")
    assert opt.results.get("status") == "fit_returned"
    assert opt.results["coef"].shape == (10,)
    print("fit_accepted:", opt.results["fit_accepted"])
    print("status:", opt.results["status"])
    print("rmse:", opt.metrics["rmse"])
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
