#!/usr/bin/env python3
"""[FIX WS-IMAGES] the minimum-image search on a skewed supercell basis.

NeighborList.wigner_seitz searched only the 27 translations {-1,0,1}^3 of the
supercell basis.  For a rhombohedral primitive cell repeated N x N x 1 the
basis is far from reduced and the nearest image needs larger coefficients, so
cutoff selection, the Born-Huang / Huang constraints (--rasr BH/H/BHH, built
from these r_ij) and the ShengBTE fc3 export used farther images.

* the search agrees with a brute-force search on a skewed and a reduced cell,
  and reproduces the old search (same images, same order) where it was right;
* a NeighborList pickled by the old search is not reused;
* a cache sidecar from before the fix stays valid only for unaffected cells;
* end to end: a central-spring FC2 model satisfies BH and Huang exactly, so a
  --rasr BHH fit must reproduce the unconstrained fit in both settings of the
  same structure (the skewed one failed before the fix).

    python -m unittest dev.test_ws_images
"""
import itertools
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

try:
    from ase import Atoms as ASEAtoms
    from ase.io import write
except ImportError:                      # pragma: no cover
    ASEAtoms = None


def _bootstrap(workdir):
    package = workdir / "bootstrap" / "pheasy_gpu"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text(
        "from pathlib import Path\n"
        f"_root = Path({str(ROOT)!r})\n"
        "__path__ = [str(_root)]\n"
        "exec(compile((_root / '__init__.py').read_text(), str(_root / '__init__.py'), 'exec'))\n")
    return workdir / "bootstrap"


def _hex(a, c):
    A = np.array([a, 0.0, 0.0])
    B = np.array([-a / 2, np.sqrt(3) * a / 2, 0.0])
    C = np.array([0.0, 0.0, c])
    R1 = 2 / 3 * A + 1 / 3 * B + 1 / 3 * C
    R2 = -1 / 3 * A + 1 / 3 * B + 1 / 3 * C
    R3 = -1 / 3 * A - 2 / 3 * B + 1 / 3 * C
    return A, B, C, R1, R2, R3


def _layered(setting, a=3.0, c=24.0, xs=(0.1, -0.1)):
    """Same R-3m-like crystal in the symmetric rhombohedral setting (skewed when
    repeated N x N x 1) or the hexagonal-like setting (reduced)."""
    A, B, C, R1, R2, R3 = _hex(a, c)
    prim = np.array([R1, R2, R3]) if setting == "rhombohedral" else np.array([A, B, R1])
    cart = np.array([x * C for x in xs])
    frac = np.linalg.solve(prim.T, cart.T).T % 1.0
    return ASEAtoms("Si%d" % len(xs), scaled_positions=frac, cell=prim, pbc=True)


def _brute(cell, x1, x2, N=6):
    t = np.array(list(itertools.product(range(-N, N + 1), repeat=3)), float)
    v = (x2 + t - x1) @ cell
    return np.linalg.norm(v, axis=1), t


@unittest.skipIf(ASEAtoms is None, "ase is required")
class WignerSeitzSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="ws_images_"))
        sys.path.insert(0, str(_bootstrap(cls.work)))
        from pheasy_gpu.structure import atoms as at
        cls.at = at

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.work / "bootstrap"))

    def _nl(self, setting, dim=(4, 4, 1), xs=(0.0, 0.11, -0.11, 0.23, -0.23, 0.36, -0.36),
            c=40.0, a=4.0):
        sc = self.at.create_supercell(self.at.Atoms(aseatoms=_layered(setting, a, c, xs)),
                                      np.array(dim))
        sc.set_pmap()
        return sc, self.at.NeighborList(sc)

    def _old_search(self, nl, x1, x2):
        t = np.array(list(itertools.product(np.arange(-1.0, 2.0), repeat=3)))
        d = np.linalg.norm(np.dot(nl._cell.T, (x2 + t - x1).T), axis=0)
        sel = np.where(abs(d - d.min()) < nl._eps)[0]
        return d.min(), t[sel]

    def test_skewed_basis_matches_brute_force(self):
        sc, nl = self._nl("rhombohedral")
        cell = sc.cell.real
        missed_by_old = 0
        for i in sc.pmap:
            for j in range(len(sc)):
                x1, x2 = sc.scaled_positions[i], sc.scaled_positions[j]
                w, dmin, images, offsets = nl.wigner_seitz(x1, x2)
                d, t = _brute(cell, x1, x2)
                self.assertAlmostEqual(dmin, d.min(), places=9)
                self.assertEqual(w, int(np.sum(abs(d - d.min()) < nl._eps)))
                np.testing.assert_allclose(np.linalg.norm((images - x1) @ cell, axis=1),
                                           dmin, atol=1e-9)
                if self._old_search(nl, x1, x2)[0] > d.min() + 1e-9:
                    missed_by_old += 1
        self.assertGreater(missed_by_old, 0)              # the bug is real here ...
        self.assertGreater(nl.ws_extended_pairs, 0)       # ... and it is reported

    def test_reduced_basis_reproduces_the_old_search(self):
        sc, nl = self._nl("hexagonal")
        self.assertEqual(nl.ws_extended_pairs, 0)
        for i in sc.pmap:
            for j in range(len(sc)):
                x1, x2 = sc.scaled_positions[i], sc.scaled_positions[j]
                _, dmin, _, offsets = nl.wigner_seitz(x1, x2)
                old_d, old_t = self._old_search(nl, x1, x2)
                self.assertEqual(dmin, old_d)
                np.testing.assert_array_equal(offsets, old_t)

    def test_old_pickle_is_rebuilt(self):
        sc, nl = self._nl("hexagonal")
        self.assertTrue(nl.matches(sc))
        del nl._ws_algo                                    # as pickled before the fix
        self.assertFalse(nl.matches(sc))

    def test_cache_sidecar_from_before_the_fix(self):
        sys.path.insert(0, str(self.work / "bootstrap"))
        try:
            from pheasy_gpu.run_pheasy import WorkFlow
        finally:
            sys.path.remove(str(self.work / "bootstrap"))
        meta = self.work / "cs.pkl.meta.json"
        old = {"max_order": 3, "poscar": "x"}              # no ws_extended_pairs key
        meta.write_text(json.dumps(old))

        class Fake(object):
            def __init__(self, n):
                self._n = n

            def _cache_fingerprint(self):
                return dict(old, ws_extended_pairs=self._n)
        self.assertTrue(WorkFlow._check_cache_fingerprint(Fake(0), str(self.work / "cs.pkl"),
                                                          "cluster space"))
        with patch.dict(os.environ, {"PHEASY_ALLOW_STALE_CACHE": "0"}), \
                self.assertRaises(RuntimeError):
            WorkFlow._check_cache_fingerprint(Fake(12), str(self.work / "cs.pkl"),
                                              "cluster space")


@unittest.skipIf(ASEAtoms is None, "ase is required")
class RotationalInvarianceSkewedCellCliTests(unittest.TestCase):
    """--rasr BHH on a central-spring FC2 model, both settings, through the CLI."""

    def _case(self, setting, rasr):
        work = Path(tempfile.mkdtemp(prefix="ws_rasr_%s_" % setting))
        boot = _bootstrap(work)
        sys.path.insert(0, str(boot))
        try:
            from pheasy_gpu.structure.atoms import Atoms, create_supercell
        finally:
            sys.path.remove(str(boot))
        # 7 atoms on the 3-fold axis, c = 40 A: in the rhombohedral setting the
        # 4x4x1 basis is 54/54/13.5 A at 17 deg and the old search mis-imaged
        # pairs from 4.6 A up; springs reach 6 A (< 13.5/2, one image per pair)
        prim = _layered(setting, a=4.0, c=40.0,
                        xs=(0.0, 0.11, -0.11, 0.23, -0.23, 0.36, -0.36))
        dim = np.array([4, 4, 1])
        scell = create_supercell(Atoms(aseatoms=prim), dim)
        cell = scell.cell.real
        n = len(scell)
        rc = 6.0                     # < half the shortest supercell translation
        phi = np.zeros((n, n, 3, 3))
        for i in range(n):
            for j in range(n):
                d, t = _brute(cell, scell.scaled_positions[i], scell.scaled_positions[j], N=5)
                for v in ((scell.scaled_positions[j] + t[d < rc] - scell.scaled_positions[i])
                          @ cell):
                    r = np.linalg.norm(v)
                    if r < 1e-8:
                        continue
                    e = v / r
                    phi[i, j] -= np.exp(-(r - 3.0)) * np.outer(e, e)
            phi[i, i] -= phi[i].sum(axis=0)
        rng = np.random.default_rng(20261110)
        ncfg = 10
        u = rng.normal(0, 0.02, (ncfg, n, 3))
        f = -np.einsum("ijab,njb->nia", phi, u) + 1e-7 * rng.normal(size=(ncfg, n, 3))
        env = {k: v for k, v in os.environ.items() if not k.startswith("PHEASY_")}
        env.update({"PYTHONPATH": str(boot), "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1", "PHEASY_N_JOBS": "1",
                    "PHEASY_SM_DTYPE": "float64", "PHEASY_USE_GPU": "0"})
        geo = ASEAtoms(numbers=scell.numbers, positions=scell.positions, cell=cell, pbc=True)
        write(work / "POSCAR", prim, format="vasp", direct=True)
        write(work / "SPOSCAR", geo, format="vasp", direct=True)
        for fn, data in (("disp_matrix.pkl", u), ("force_matrix.pkl", f)):
            with (work / fn).open("wb") as s:
                pickle.dump(data, s, protocol=pickle.HIGHEST_PROTOCOL)
        base = [sys.executable, "-m", "pheasy_gpu.run_pheasy", "--dim", "4", "4", "1",
                "-w", "2", "--c2", "100", "--eps", "0.001"]
        extra = ["--rasr", rasr] if rasr else []
        log = ""
        for flags in (["-s"], ["-c"] + extra, ["-d", "--ndata", str(ncfg), "--disp_file"],
                      ["-f", "--ndata", str(ncfg), "--disp_file", "--full_ifc", "-l", "OLS",
                       "--hdf5"] + extra):
            r = subprocess.run(base + flags, cwd=work, env=env, capture_output=True, text=True)
            log += r.stdout + r.stderr
            self.assertEqual(r.returncode, 0, "%s setting, --rasr %s, stage %s:\n%s"
                             % (setting, rasr, flags[0], "\n".join(log.splitlines()[-8:])))
        with open(work / "fit_manifest.json") as fh:
            return json.load(fh), log

    def test_bhh_reproduces_the_unconstrained_fit_in_both_settings(self):
        runs = {setting: (self._case(setting, None), self._case(setting, "BHH"))
                for setting in ("hexagonal", "rhombohedral")}
        for setting, ((free, _), (bhh, log)) in runs.items():
            self.assertIn("Imposing rotational invariance and equilibrium conditions", log)
            self.assertLess(free["metrics"]["re"], 1e-4, setting)
            # the true FC2 satisfies BH and Huang: imposing them costs nothing
            self.assertLess(bhh["metrics"]["re"], 1e-4,
                            "%s: --rasr BHH re=%.3e (unconstrained %.3e)"
                            % (setting, bhh["metrics"]["re"], free["metrics"]["re"]))
        for setting, (_, (bhh, log)) in runs.items():
            extended = bhh["geometry"]["ws_extended_pairs"]
            if setting == "rhombohedral":
                self.assertGreater(extended, 0)
                self.assertIn("[WS-IMAGES]", log)
            else:
                self.assertEqual(extended, 0)

if __name__ == "__main__":
    unittest.main()
