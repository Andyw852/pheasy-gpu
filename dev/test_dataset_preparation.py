#!/usr/bin/env python3
"""Regression checks for coordinate alignment and the GPU wrapper entrypoint."""
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from ase import Atoms
from ase.io import write


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_dataset", ROOT / "tools/prepare_dataset.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class DatasetPreparationTests(unittest.TestCase):
    def test_fractional_alignment_reorders_forces_and_removes_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            atoms = Atoms("MgC", cell=[[4, 0, 0], [1.1, 3, 0], [0.4, 0.6, 5]],
                          scaled_positions=[[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]], pbc=True)
            write(folder / "SPOSCAR", atoms, format="vasp", direct=True)
            ref = atoms.get_scaled_positions()
            delta = np.array([[[-0.15, 0, 0], [0, 0.02, 0]]])
            d = np.concatenate([(ref + delta) % 1, ref[None]])[:, [1, 0]]
            expected_forces = np.array([[[1., 2, 3], [4, 5, 6]]])
            residual = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            f = np.concatenate([expected_forces + residual, residual[None]])[:, [1, 0]]
            np.save(folder / "d.npy", d)
            np.save(folder / "f.npy", f)
            subprocess.run([sys.executable, str(ROOT / "tools/prepare_dataset.py"),
                            "SPOSCAR", "d.npy", "f.npy", "--frac", "--align-reference"],
                           cwd=folder, check=True, capture_output=True, text=True)
            with (folder / "disp_matrix.pkl").open("rb") as stream:
                actual_u = pickle.load(stream)
            with (folder / "force_matrix.pkl").open("rb") as stream:
                actual_f = pickle.load(stream)
            np.testing.assert_allclose(actual_u, delta @ atoms.cell, atol=1e-14)
            np.testing.assert_allclose(actual_f, expected_forces, atol=1e-14)
            report = json.loads((folder / "dataset_alignment.json").read_text())
            self.assertEqual(report["target_to_source_permutation"], [1, 0])
            self.assertEqual(report["reference_frame"], 1)

    def test_alignment_rejects_different_reference_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "SPOSCAR"
            atoms = Atoms("CC", cell=np.eye(3) * 4,
                          scaled_positions=[[0, 0, 0], [.5, .5, .5]], pbc=True)
            write(path, atoms, format="vasp", direct=True)
            ref = atoms.get_scaled_positions()
            ref[0, 0] += .01
            with self.assertRaisesRegex(ValueError, "does not match"):
                prepare.align_fractional_reference(ref, path)

    def test_wrapper_selects_gpu_entrypoint_and_accepts_explicit_path(self):
        for override in (False, True):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                bindir = folder / "bin"
                bindir.mkdir()
                executable = bindir / ("custom gpu" if override else "pheasy-gpu")
                executable.write_text("#!/bin/bash\n"
                                      "printf '%s\\n' \"$*\" >> calls.txt\n"
                                      "touch cs.pkl ns_harm.npz sm_prime.npz\n")
                executable.chmod(0o755)
                poison = bindir / "pheasy"
                poison.write_text("#!/bin/bash\nexit 99\n")
                poison.chmod(0o755)
                atoms = Atoms("C", cell=np.eye(3) * 4, positions=[[0, 0, 0]], pbc=True)
                for name in ("POSCAR", "SPOSCAR"):
                    write(folder / name, atoms, format="vasp", direct=True)
                for name in ("disp_matrix.pkl", "force_matrix.pkl"):
                    with (folder / name).open("wb") as stream:
                        pickle.dump(np.zeros((3, 1, 3)), stream)
                env = os.environ.copy()
                env["PATH"] = os.pathsep.join([str(bindir), str(Path(sys.executable).parent), env["PATH"]])
                env.pop("PHEASY_EXECUTABLE", None)
                if override:
                    env["PHEASY_EXECUTABLE"] = str(executable)
                subprocess.run(["bash", str(ROOT / "pheasy_fit.sh"), "FIT_METHOD=OLS", "NCPU=1"],
                               cwd=folder, env=env, check=True, capture_output=True, text=True)
                calls = (folder / "calls.txt").read_text()
                self.assertIn(" -s ", calls)
                self.assertIn(" -f ", calls)


if __name__ == "__main__":
    unittest.main()
