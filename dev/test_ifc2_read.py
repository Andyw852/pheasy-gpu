#!/usr/bin/env python3
"""HDF5 force constants survive file closure and accept both valid layouts."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pheasy_gpu', ROOT / '__init__.py',
                                             submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules['pheasy_gpu'] = package
spec.loader.exec_module(package)
from pheasy_gpu.interface.phono23py import read_ifc2


class ReadFc2Test(unittest.TestCase):
    def test_full_compact_and_alternative_keys(self):
        scell = SimpleNamespace(get_global_number_of_atoms=lambda: 4,
                                get_number_of_atoms_unit_cell=lambda: 2,
                                supercell=np.array([2, 1, 1]), pmap=np.array([0, 2]))
        full = np.arange(4 * 4 * 9, dtype=float).reshape(4, 4, 3, 3)
        compact = full[scell.pmap]
        clusters = [[SimpleNamespace(atom_index=[0, 1])],
                    [SimpleNamespace(atom_index=[2, 3])]]
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'fc2.hdf5')
            for key in ('fc2', 'force_constants', 'force_constants_2nd'):
                for values in (full, compact):
                    with self.subTest(key=key, shape=values.shape):
                        with h5py.File(path, 'w') as stream:
                            stream[key] = values
                        result = read_ifc2(scell, clusters, path)
                        self.assertIsInstance(result, np.ndarray)
                        np.testing.assert_array_equal(result, compact)
                        np.testing.assert_array_equal(read_ifc2(scell, clusters, path, full=False),
                                                      compact[[0, 1], [1, 3]])
            with h5py.File(path, 'w') as stream:
                stream['fc2'] = np.zeros((2, 4))
            with self.assertRaises(RuntimeError):
                read_ifc2(scell, clusters, path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
