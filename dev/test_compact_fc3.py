#!/usr/bin/env python3
"""Compact fc3 equals a full reference slice without allocating the full tensor."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pheasy_gpu', ROOT / '__init__.py',
                                             submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules['pheasy_gpu'] = package
spec.loader.exec_module(package)
from pheasy_gpu.interface import phono23py


class Cluster:
    def __init__(self, indices):
        self.atom_index = indices

    def get_crotation_tensor(self):
        return np.eye(27)


class CompactFc3Test(unittest.TestCase):
    def test_compact_equals_reordered_full_slice_and_never_allocates_full(self):
        scell = SimpleNamespace(get_global_number_of_atoms=lambda: 4, pmap=np.array([2, 0]))
        clusters = [[None, Cluster([0, 1, 2])], [None, Cluster([2, 2, 3])],
                    [None, Cluster([1, 1, 1])]]
        phi = np.arange(81, dtype=float).reshape(3, 3, 3, 3)
        original = np.zeros
        allocated = []

        def checked_zeros(shape, *args, **kwargs):
            if len(np.atleast_1d(shape)) == 6:
                allocated.append(tuple(shape))
                self.assertNotEqual(tuple(shape), (4, 4, 4, 3, 3, 3))
            return original(shape, *args, **kwargs)

        previous = Path.cwd()
        with tempfile.TemporaryDirectory(prefix='pheasy_compact_fc3_') as tmp:
            try:
                os.chdir(tmp)
                phono23py.write_ifc3(phi, scell, clusters, full=True)
                with h5py.File('fc3.hdf5') as fd:
                    reference = fd['fc3'][:][scell.pmap]
                with patch.object(phono23py.np, 'zeros', side_effect=checked_zeros):
                    phono23py.write_ifc3(phi, scell, clusters, full=False)
                with h5py.File('fc3.hdf5') as fd:
                    actual = fd['fc3'][:]
                np.testing.assert_array_equal(actual, reference)
                self.assertEqual(allocated, [(2, 4, 4, 3, 3, 3)])
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main(verbosity=2)
