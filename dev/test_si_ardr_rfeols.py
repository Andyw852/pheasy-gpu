#!/usr/bin/env python3
"""Si benchmark for RFE-OLS and ARDR (Fransson, Eriksson & Erhart 2020).

Reproduces, on a Si dataset built and solved locally, the central observation
of E. Fransson, F. Eriksson, P. Erhart, npj Comput. Mater. 6, 135 (2020):
as the force-constant cutoff -- and therefore the number of free parameters --
grows, plain OLS keeps every feature and its held-out error grows, while
RFE-OLS and ARDR keep the selected feature count bounded and the held-out
error flat.

Dataset
-------
A 3x3x3 Si supercell (54 atoms) with one atom removed (53 atoms).  The vacancy
breaks the crystal symmetry, so the number of free 2nd-order IFC parameters
grows with the cutoff exactly as in the paper's low-symmetry Ta-vacancy models.
The ground truth is a nearest-neighbour spring model generated directly in real
space (never F = SM @ coef); a small Gaussian force noise floor is added, and
validation is scored against the NOISELESS forces on configurations the fit
never saw.

The sensing matrices are built by the production pheasy-gpu CLI (cluster ->
constraints -> sensing); the fits then call the Optimizer API directly.  No GPU
is required.

Usage:
    python dev/test_si_ardr_rfeols.py
    python dev/test_si_ardr_rfeols.py --cutoffs 3.0 4.5 5.5 6.5 \
        --ntrain 24 --ntest 36 --noise 0.02 --workdir tmp/si_ardr_test
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms as ASEAtoms
from ase.build import bulk
from ase.io import write

ROOT = Path(__file__).resolve().parents[1]


def checkout_bootstrap(workdir):
    """Expose the flat checkout as pheasy_gpu without touching the environment."""
    bootstrap = workdir / "_bootstrap"
    package = bootstrap / "pheasy_gpu"
    package.mkdir(parents=True, exist_ok=True)
    package.joinpath("__init__.py").write_text(
        "from pathlib import Path\n"
        "_root = Path(%r)\n" % str(ROOT)
        + "__path__ = [str(_root)]\n"
        "exec(compile((_root / '__init__.py').read_text(), str(_root / '__init__.py'), 'exec'))\n"
    )
    return bootstrap


def load_csr(path):
    import scipy.sparse as sp
    z = np.load(path)
    return sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                         shape=tuple(int(x) for x in z["shape"]))


def load_coo(path):
    import scipy.sparse as sp
    z = np.load(path)
    return sp.coo_matrix((z["data"], (z["row"], z["col"])),
                         shape=tuple(int(x) for x in z["shape"])).tocsr()


def load_null_space(work):
    import scipy.sparse as sp
    blocks = [load_coo(work / fn) for fn in ("ns_harm.npz", "ns_anharm3.npz")
              if (work / fn).exists()]
    if len(blocks) == 1:
        return blocks[0]
    return sp.block_diag(blocks).tocsr()


def build_dataset(work, ntrain, ntest, noise, seed):
    sys.path.insert(0, str(work / "_bootstrap"))
    from pheasy_gpu.structure.atoms import Atoms, create_supercell

    primitive = bulk("Si", "diamond", a=5.43)
    scell = create_supercell(Atoms(aseatoms=primitive), np.array([3, 3, 3]))
    geometry = ASEAtoms(numbers=scell.numbers, positions=scell.positions,
                        cell=scell.cell, pbc=True)
    del geometry[1]                                   # one vacancy: low symmetry
    natoms = len(geometry)

    distances = geometry.get_all_distances(mic=True)
    nearest = float(distances[distances > 1e-8].min())
    neighbors = (distances > 1e-8) & (distances < 1.05 * nearest)
    # The perfect diamond lattice has NN degree 4; removing one atom leaves the
    # four atoms around the vacancy with degree 3, which is exactly the
    # low-symmetry defect cell we want.  The model is still a valid harmonic
    # IFC (symmetric, isotropic bonds, ASR on every site).
    phi = -neighbors[:, :, None, None].astype(float) * np.eye(3)[None, None]
    phi[np.arange(natoms), np.arange(natoms)] = (
        neighbors.sum(axis=1)[:, None, None] * np.eye(3))
    np.testing.assert_array_equal(phi.sum(axis=1), 0)

    rng = np.random.default_rng(seed)
    nconf = ntrain + ntest
    u = rng.normal(0, 0.02, (nconf, natoms, 3))
    f_clean = -np.einsum("ijab,njb->nia", phi, u, optimize=True)
    f = f_clean + noise * np.std(f_clean) * rng.standard_normal(f_clean.shape)

    write(work / "POSCAR", geometry, format="vasp", direct=True)
    write(work / "SPOSCAR", geometry, format="vasp", direct=True)
    for name, arr in (("disp_matrix.pkl", u),
                      ("force_matrix.pkl", f),
                      ("force_clean.pkl", f_clean)):
        with (work / name).open("wb") as stream:
            pickle.dump(arr, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return natoms, nearest, nconf


def run_sensing(work, c2, nconf, bootstrap, jobs):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PHEASY_")}
    env.update({"PYTHONPATH": str(bootstrap), "PHEASY_USE_GPU": "0",
                "PHEASY_N_JOBS": str(jobs), "PHEASY_SM_DTYPE": "float64",
                "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4"})
    base = [sys.executable, "-m", "pheasy_gpu.run_pheasy", "--dim", "1", "1", "1",
            "-w", "2", "--c2", str(c2), "--eps", "0.001"]
    for name, flags in (("cluster", ["-s"]), ("constraints", ["-c"]),
                        ("sensing", ["-d", "--ndata", str(nconf), "--disp_file"])):
        with (work / (name + ".log")).open("w") as stream:
            result = subprocess.run(base + flags, cwd=work, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            tail = (work / (name + ".log")).read_text().splitlines()[-25:]
            raise RuntimeError("CLI %s failed (c2=%s)\n%s" % (name, c2, "\n".join(tail)))
    smp = load_csr(work / "sm_prime.npz")
    ns = load_null_space(work)
    sm = (smp @ ns).tocsr()
    # The fit's feature count is the number of free IFC parameters (NS columns),
    # not SM_prime's cluster columns.
    return sm, ns.shape[1]


def fit_methods(sm_train, f_train, methods, seed):
    from pheasy_gpu.core.optimizer import Optimizer
    rows = []
    for method in methods:
        t0 = time.time()
        os.environ["PHEASY_RFE_STEP"] = os.environ.get("PHEASY_RFE_STEP", "0.1")
        os.environ["PHEASY_TSQR_STEP"] = os.environ.get("PHEASY_TSQR_STEP", "0.1")
        opt = Optimizer(method, nalpha=40, alpha_min=-10, alpha_max=-3.5, cv=5,
                        max_iter=100000, rand_seed=seed, standardize=False)
        try:
            opt.fit(sm_train, f_train)
        except Exception as exc:
            rows.append({"method": method, "ok": False,
                         "error": "%s: %s" % (type(exc).__name__, exc),
                         "seconds": time.time() - t0})
            continue
        coef = opt.results["coef"]
        rows.append({"method": method, "ok": True,
                     "nonzero": int(np.count_nonzero(coef)),
                     "n_features": int(coef.shape[0]),
                     "rmse_train": float(opt.metrics["rmse"]),
                     "rmse_cv": float(opt.metrics.get("rmse_path_mean", float("nan"))),
                     "fit_accepted": bool(opt.results.get("fit_accepted", True)),
                     "seconds": time.time() - t0,
                     "coef": coef})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoffs", type=float, nargs="+",
                    default=[3.0, 4.5, 5.5, 6.5])
    ap.add_argument("--ntrain", type=int, default=24)
    ap.add_argument("--ntest", type=int, default=36)
    ap.add_argument("--noise", type=float, default=0.02,
                    help="Gaussian force noise as a fraction of the force RMS")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--methods", nargs="+",
                    default=["OLS", "LASSO", "RFE-OLS", "RFE-OLS-TSQR", "ARDR"])
    ap.add_argument("--workdir", type=Path, default=ROOT / "tmp" / "si_ardr_rfeols")
    args = ap.parse_args()

    work = args.workdir.resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    bootstrap = checkout_bootstrap(work)

    natoms, nearest, nconf = build_dataset(work, args.ntrain, args.ntest,
                                           args.noise, args.seed)
    from pheasy_gpu.structure.atoms import Atoms  # noqa: F401  (bootstrap check)
    rows_per_conf = 3 * natoms
    train_rows = np.arange(args.ntrain * rows_per_conf)
    test_rows = np.arange(args.ntrain * rows_per_conf, nconf * rows_per_conf)
    f_clean = np.load(work / "force_clean.pkl", allow_pickle=True).ravel()
    f_full = np.load(work / "force_matrix.pkl", allow_pickle=True).ravel()
    f_train = np.asarray(f_full, dtype=np.float64)[train_rows]

    print("Si vacancy: natoms=%d nearest=%.3f A  nconf=%d (%d train / %d test)  "
          "noise=%.1f%%" % (natoms, nearest, nconf, args.ntrain, args.ntest,
                            100.0 * args.noise))
    print("=" * 108)
    header = ("%-13s %6s %8s %8s %10s %10s %10s %9s %8s"
              % ("cutoff(A)", "freePar", "method", "nonzero", "rmse_tr",
                 "rmse_cv", "rmse_test", "rel_test", "sec"))
    print(header)
    print("-" * 108)

    report = {"natoms": natoms, "nearest_A": nearest, "ntrain": args.ntrain,
              "ntest": args.ntest, "noise": args.noise, "seed": args.seed,
              "rows": []}
    for c2 in args.cutoffs:
        t0 = time.time()
        sm, free_par = run_sensing(work, c2, nconf, bootstrap, args.jobs)
        sm_train = sm[train_rows].tocsr()
        sm_test = sm[test_rows].tocsr()
        f_test_clean = np.asarray(f_clean, dtype=np.float64)[test_rows]
        os.environ["PHEASY_CV_GROUP_SIZE"] = str(3 * natoms)
        fits = fit_methods(sm_train, f_train, args.methods, args.seed)
        for row in fits:
            if not row["ok"]:
                print("%-13.1f %6d %-13s   FAILED: %s"
                      % (c2, free_par, row["method"], row["error"]))
                report["rows"].append({"cutoff": c2, "free_params": free_par, **row})
                continue
            pred = np.asarray(sm_test @ row.pop("coef")).ravel()
            rmse_test = float(np.sqrt(np.mean((pred - f_test_clean) ** 2)))
            rel_test = rmse_test / float(np.sqrt(np.mean(f_test_clean ** 2)))
            entry = {"cutoff": c2, "free_params": free_par, "rmse_test": rmse_test,
                     "rel_test": rel_test, "sensing_seconds": time.time() - t0, **row}
            report["rows"].append(entry)
            print("%-13.1f %6d %-13s %8d %10.3e %10.3e %10.3e %9.2e %8.1f"
                  % (c2, free_par, row["method"], row["nonzero"],
                     row["rmse_train"], row["rmse_cv"], rmse_test, rel_test,
                     row["seconds"]))
        (work / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print("=" * 108)
    print("report: %s" % (work / "report.json"))

    # --- the paper's central qualitative checks -------------------------------
    largest = args.cutoffs[-1]
    at_largest = {r["method"]: r for r in report["rows"]
                  if r["cutoff"] == largest and r.get("ok")}
    checks = []
    if "OLS" in at_largest and "ARDR" in at_largest:
        checks.append(("ARDR selects fewer features than OLS at %.1f A" % largest,
                       at_largest["ARDR"]["nonzero"] < at_largest["OLS"]["nonzero"]))
    if "OLS" in at_largest and "RFE-OLS" in at_largest:
        checks.append(("RFE-OLS selects fewer features than OLS at %.1f A" % largest,
                       at_largest["RFE-OLS"]["nonzero"] < at_largest["OLS"]["nonzero"]))
    for name in ("RFE-OLS", "ARDR"):
        if name in at_largest:
            checks.append(("%s held-out error stays within 2x OLS at %.1f A"
                           % (name, largest),
                           at_largest[name]["rmse_test"]
                           <= 2.0 * at_largest["OLS"]["rmse_test"]))
    # Feature-count growth from the smallest to the largest cutoff, relative to
    # OLS.  The paper's point is not that RFE-OLS/ARDR are perfectly flat but
    # that their retained-feature count grows far more slowly than OLS's.
    ols_lo = [r for r in report["rows"] if r["cutoff"] == args.cutoffs[0]
              and r["method"] == "OLS" and r.get("ok")]
    ols_hi = [r for r in report["rows"] if r["cutoff"] == largest
              and r["method"] == "OLS" and r.get("ok")]
    if ols_lo and ols_hi and ols_lo[0]["nonzero"]:
        ols_growth = ols_hi[0]["nonzero"] / float(ols_lo[0]["nonzero"])
        for name in ("RFE-OLS", "ARDR"):
            lo = [r for r in report["rows"] if r["cutoff"] == args.cutoffs[0]
                  and r["method"] == name and r.get("ok")]
            hi = [r for r in report["rows"] if r["cutoff"] == largest
                  and r["method"] == name and r.get("ok")]
            if lo and hi:
                growth = hi[0]["nonzero"] / float(max(lo[0]["nonzero"], 1))
                checks.append(("%s feature count grows at most half as fast as OLS "
                               "from %.1f to %.1f A (%.1fx vs %.1fx)"
                               % (name, args.cutoffs[0], largest, growth, ols_growth),
                               growth <= 0.5 * ols_growth))
    print("checks:")
    for label, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
    # A run in which every fit raised must not read as "all checks passed".
    n_ok = sum(1 for r in report["rows"] if r.get("ok"))
    checks.append(("at least one fit succeeded for every method",
                   n_ok >= len(args.methods) * len(args.cutoffs)))
    # Report the just-appended check too.
    print("  [%s] %s" % ("PASS" if checks[-1][1] else "FAIL", checks[-1][0]))
    report["checks"] = [{"label": l, "ok": bool(o)} for l, o in checks]
    (work / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    if not all(ok for _, ok in checks):
        print("SOME CHECKS FAILED (see the table above)")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
