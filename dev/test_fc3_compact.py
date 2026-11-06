#!/usr/bin/env python3
"""[FIX FC3-COMPACT] --fc3_compact: phono3py compact fc3.hdf5 with p2s_map.

The full fc3.hdf5 is natoms^3*27*8 bytes in memory while pheasy writes it and
while phono3py reads it (41 GB for a 576-atom supercell, although gzip makes
the file small).  phono3py works natively on the compact
(n_patom, natoms, natoms, 3, 3, 3) array.  This test runs the CLI once with
--full_ifc and once with --full_ifc --fc3_compact on Si 2x2x2 (synthetic
forces with a cubic term) and checks:

  * the compact array is exactly the full one restricted to the primitive
    atoms (fc3_full[p2s_map]), and fc2 stays full under --full_ifc;
  * the p2s_map dataset equals phono3py's own primitive.p2s_map and pheasy's
    supercell is phono3py's supercell, atom by atom;
  * phono3py gives the same RTA kappa from the compact and from the full fc3;
  * a different primitive table is REFUSED through p2s_map instead of being
    read silently wrong.

    python -m unittest dev.test_fc3_compact
"""
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

try:
    import h5py
    from ase.build import bulk
    from ase import Atoms as ASEAtoms
    from ase.io import write
    import phono3py
    from phono3py.file_IO import read_fc3_from_hdf5
    from phonopy.structure.atoms import PhonopyAtoms
except ImportError:                      # pragma: no cover
    phono3py = None


def _bootstrap(workdir):
    package = workdir / "bootstrap" / "pheasy_gpu"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text(
        "from pathlib import Path\n"
        f"_root = Path({str(ROOT)!r})\n"
        "__path__ = [str(_root)]\n"
        "exec(compile((_root / '__init__.py').read_text(), str(_root / '__init__.py'), 'exec'))\n")
    return workdir / "bootstrap"


@unittest.skipIf(phono3py is None, "phono3py, ase and h5py are required")
class Fc3CompactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = Path(tempfile.mkdtemp(prefix="fc3compact_"))
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
        k[(d > 1.05 * nearest) & (d < 1.7 * nearest)] = 0.1
        phi = -k[:, :, None, None] * np.eye(3)
        phi[np.arange(n), np.arange(n)] = k.sum(axis=1)[:, None, None] * np.eye(3)
        rng = np.random.default_rng(7)
        ncfg = 16
        u = rng.normal(0, 0.03, (ncfg, n, 3))
        f = -np.einsum("ijab,njb->nia", phi, u)
        for i in range(n):                  # a cubic pair term: nonzero fc3
            for j in range(n):
                if k[i, j] == 1.0:
                    du = u[:, i] - u[:, j]
                    f[:, i] -= 2.0 * du * du
        f += 1e-5 * rng.normal(size=f.shape)
        env = {kk: v for kk, v in os.environ.items() if not kk.startswith("PHEASY_")}
        env.update({"PYTHONPATH": str(boot), "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1", "PHEASY_N_JOBS": "1",
                    "PHEASY_SM_DTYPE": "float64", "PHEASY_USE_GPU": "0"})
        base = [sys.executable, "-m", "pheasy_gpu.run_pheasy", "--dim", "2", "2", "2",
                "-w", "3", "--c2", "%.4f" % (2.2 * nearest),
                "--c3", "%.4f" % (1.05 * nearest), "--eps", "0.001"]
        cls.dirs = {}
        for name, extra in (("full", []), ("compact", ["--fc3_compact"])):
            dd = work / name
            dd.mkdir()
            write(dd / "POSCAR", prim, format="vasp", direct=True)
            write(dd / "SPOSCAR", scell, format="vasp", direct=True)
            for fn, data in (("disp_matrix.pkl", u), ("force_matrix.pkl", f)):
                with (dd / fn).open("wb") as s:
                    pickle.dump(data, s, protocol=pickle.HIGHEST_PROTOCOL)
            for flags in (["-s"], ["-c"], ["-d", "--ndata", str(ncfg), "--disp_file"],
                          ["-f", "--ndata", str(ncfg), "--disp_file", "--full_ifc",
                           "-l", "OLS", "--hdf5"] + extra):
                r = subprocess.run(base + flags, cwd=dd, env=env, capture_output=True,
                                   text=True)
                if r.returncode != 0:
                    raise RuntimeError("\n".join((r.stdout + r.stderr).splitlines()[-25:]))
            cls.dirs[name] = dd
        cls.prim, cls.scell_pos, cls.natoms = prim, scell.positions, n

    def _ph3(self, unitcell=None, smat=(2, 2, 2), pmat=None):
        a = self.prim if unitcell is None else unitcell
        uc = PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=a.cell[:],
                          scaled_positions=a.get_scaled_positions())
        return phono3py.Phono3py(uc, supercell_matrix=np.diag(smat) if np.ndim(smat) == 1
                                 else smat, primitive_matrix=pmat, log_level=0)

    def test_compact_is_the_full_tensor_on_the_primitive_atoms(self):
        full = read_fc3_from_hdf5(self.dirs["full"] / "fc3.hdf5")
        comp = read_fc3_from_hdf5(self.dirs["compact"] / "fc3.hdf5")
        with h5py.File(self.dirs["compact"] / "fc3.hdf5", "r") as fd:
            p2s = fd["p2s_map"][:]
        self.assertEqual(full.shape, (self.natoms,) * 3 + (3, 3, 3))
        self.assertEqual(comp.shape, (len(p2s), self.natoms, self.natoms, 3, 3, 3))
        self.assertGreater(np.abs(full).max(), 1e-3)          # a real fc3
        np.testing.assert_allclose(comp, full[p2s], rtol=0, atol=1e-12)
        with h5py.File(self.dirs["compact"] / "fc2.hdf5", "r") as fd:   # fc2 stays full
            self.assertEqual(fd["fc2"].shape, (self.natoms, self.natoms, 3, 3))

    def test_p2s_map_and_supercell_match_phono3py(self):
        ph3 = self._ph3()
        with h5py.File(self.dirs["compact"] / "fc3.hdf5", "r") as fd:
            p2s = fd["p2s_map"][:]
        np.testing.assert_array_equal(p2s, ph3.primitive.p2s_map)
        np.testing.assert_allclose(ph3.supercell.positions, self.scell_pos, atol=1e-8)
        # the reader's own check passes with phono3py's table
        read_fc3_from_hdf5(self.dirs["compact"] / "fc3.hdf5", p2s_map=ph3.primitive.p2s_map)

    def test_phono3py_kappa_is_identical(self):
        kap = {}
        for name in ("full", "compact"):
            ph3 = self._ph3()
            with h5py.File(self.dirs[name] / "fc2.hdf5", "r") as fd:
                ph3.fc2 = fd["fc2"][:]
            ph3.fc3 = read_fc3_from_hdf5(self.dirs[name] / "fc3.hdf5")
            ph3.mesh_numbers = [5, 5, 5]
            ph3.init_phph_interaction()
            ph3.run_thermal_conductivity(temperatures=[300], log_level=0)
            kap[name] = np.asarray(ph3.thermal_conductivity.kappa)[0, 0]
        self.assertTrue(np.all(np.isfinite(kap["full"])) and kap["full"][0] > 0, kap)
        np.testing.assert_allclose(kap["compact"], kap["full"], rtol=1e-8)

    def test_wrong_primitive_table_is_refused(self):
        # a phono3py run whose primitive cell differs (e.g. primitive_matrix
        # 'auto' on a non-primitive unit cell) has another p2s_map; the file's
        # table makes the reader refuse instead of indexing the wrong atoms
        ph3 = self._ph3()
        wrong = np.asarray(ph3.primitive.p2s_map)[::-1].copy()
        with self.assertRaises(Exception):
            read_fc3_from_hdf5(self.dirs["compact"] / "fc3.hdf5", p2s_map=wrong)


if __name__ == "__main__":
    unittest.main()
