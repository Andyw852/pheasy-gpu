#!/usr/bin/env python3
"""--fix_fc2 through the CLI: fixing the FC2 of a joint fit reproduces it.

Si 2x2x2, synthetic forces.  A joint fit writes fc2.hdf5; that file, copied to
fc.hdf5, is read back with --fix_fc2 --fc2_fmt PHONOPY at the same cutoffs.
The FC3 block of a joint least-squares solution is the optimum given its FC2,
so the fixed-FC2 fit must reproduce the joint force RMSE, and its
re_total_forces must equal the joint re.  This pins:

  * [FIX FIXFC2-SPP]   _digest's _sp_p was an unbound closure variable;
  * [FIX FIXFC2-DTYPE] the fix branch hard-coded a float32 SM against the
                       PHEASY_SM_DTYPE FM (assert_uniform_dtype);
  * [FIX FIXFC2-TOTAL] the relative error and the alignment gate are taken on
                       the total forces (the anharmonic residual alone can be
                       noise: here re ~ 0.98 for a correct fit).

Runs the four CLI stages twice per case (~10-20 s each).

    python -m unittest dev.test_fix_fc2_cli
"""
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

try:
    from ase.build import bulk
    from ase import Atoms as ASEAtoms
    from ase.io import write
except ImportError:                      # pragma: no cover
    bulk = None


def _bootstrap(workdir):
    package = workdir / "bootstrap" / "pheasy_gpu"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text(
        "from pathlib import Path\n"
        f"_root = Path({str(ROOT)!r})\n"
        "__path__ = [str(_root)]\n"
        "exec(compile((_root / '__init__.py').read_text(), str(_root / '__init__.py'), 'exec'))\n")
    return workdir / "bootstrap"


@unittest.skipIf(bulk is None, "ase is required")
class FixFc2CliTests(unittest.TestCase):
    def _case(self, method, dtype):
        work = Path(tempfile.mkdtemp(prefix="fixfc2_"))
        boot = _bootstrap(work)
        sys.path.insert(0, str(boot))
        try:
            from pheasy_gpu.structure.atoms import Atoms, create_supercell
        finally:
            sys.path.remove(str(boot))
        prim = bulk("Si", "diamond", a=5.43)
        scell = create_supercell(Atoms(aseatoms=prim), np.array([2, 2, 2]))
        geo = ASEAtoms(numbers=scell.numbers, positions=scell.positions,
                       cell=scell.cell, pbc=True)
        d = geo.get_all_distances(mic=True)
        nearest = float(d[d > 1e-8].min())
        n = len(scell)
        k = np.zeros((n, n))
        k[(d > 1e-8) & (d < 1.05 * nearest)] = 1.0
        k[(d > 1.05 * nearest) & (d < 1.7 * nearest)] = 0.01
        phi = -k[:, :, None, None] * np.eye(3)
        phi[np.arange(n), np.arange(n)] = k.sum(axis=1)[:, None, None] * np.eye(3)
        rng = np.random.default_rng(20261027)
        ncfg = 16
        u = rng.normal(0, 0.02, (ncfg, n, 3))
        f = -np.einsum("ijab,njb->nia", phi, u)
        for i in range(n):              # a term FC3 cannot represent: residual ~ noise
            for j in range(n):
                if k[i, j] == 1.0:
                    du = u[:, i] - u[:, j]
                    f[:, i] -= 0.5 * du * np.abs(du)
        f += 1e-5 * rng.normal(size=f.shape)
        env = {kk: v for kk, v in os.environ.items() if not kk.startswith("PHEASY_")}
        env.update({"PYTHONPATH": str(boot), "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1", "PHEASY_N_JOBS": "1",
                    "PHEASY_SM_DTYPE": dtype, "PHEASY_USE_GPU": "0"})
        base = [sys.executable, "-m", "pheasy_gpu.run_pheasy", "--dim", "2", "2", "2",
                "-w", "3", "--c2", "%.4f" % (2.2 * nearest),
                "--c3", "%.4f" % (1.05 * nearest), "--eps", "0.001"]

        def run(name, extra):
            dd = work / name
            dd.mkdir()
            write(dd / "POSCAR", prim, format="vasp", direct=True)
            write(dd / "SPOSCAR", scell, format="vasp", direct=True)
            for fn, data in (("disp_matrix.pkl", u), ("force_matrix.pkl", f)):
                with (dd / fn).open("wb") as s:
                    pickle.dump(data, s, protocol=pickle.HIGHEST_PROTOCOL)
            stages = [["-s"], ["-c"], ["-d", "--ndata", str(ncfg), "--disp_file"],
                      ["-f", "--ndata", str(ncfg), "--disp_file", "--full_ifc", "-l",
                       method, "--hdf5"] + extra]
            log = ""
            for flags in stages:
                if extra and flags[0] == "-f":
                    shutil.copy(work / "joint" / "fc2.hdf5", dd / "fc.hdf5")
                r = subprocess.run(base + flags, cwd=dd, env=env, capture_output=True,
                                   text=True)
                log += r.stdout + r.stderr
                self.assertEqual(r.returncode, 0, "\n".join(log.splitlines()[-25:]))
            return json.load(open(dd / "fit_manifest.json")), log

        joint, _ = run("joint", [])
        fixed, log = run("fixed", ["--fix_fc2", "--fc2_fmt", "PHONOPY"])
        return joint, fixed, log

    def _check(self, joint, fixed, log, rtol, sparse):
        jm, fm = joint["metrics"], fixed["metrics"]
        self.assertTrue(fixed["results"]["fit_accepted"])
        if sparse:      # the FC2 protection of the sparse methods stands down
            self.assertIn("[HARM_DENSE] ignored: FIX_FC2", log)
        self.assertIn("alignment check OK", log)
        self.assertGreater(fm["re"], 0.5)              # residual is mostly noise ...
        self.assertAlmostEqual(fm["rmse"] / jm["rmse"], 1.0, delta=rtol)
        self.assertAlmostEqual(fm["re_total_forces"] / jm["re"], 1.0, delta=rtol)

    def test_ols_float64_reproduces_the_joint_fit(self):
        self._check(*self._case("OLS", "float64"), rtol=1e-9, sparse=False)

    def test_rfe_ols_float32(self):
        self._check(*self._case("RFE-OLS", "float32"), rtol=1e-4, sparse=True)


if __name__ == "__main__":
    unittest.main()
