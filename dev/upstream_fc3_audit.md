# Upstream pheasy audit (fc3 synthetic cross-check)

## Purpose
When anyone touches optimizer.py / symmetry_constraints.py (the fit & constraint
layers that were rewritten 224->2577 / 687->1086 lines vs upstream), run this to
prove the rewrite reproduces upstream numerically on the SAME data/model.

## Setup (isolated venv, must not collide with pheasy_gpu)
```bash
python -m venv /tmp/pheasy_up        # or anywhere; keep separate from pheasy-gpu envs
/tmp/pheasy_up/bin/pip install pheasy==0.0.2 numpy scipy scikit-learn spglib phonopy ase h5py pyyaml
# numpy>=2 removed np.math.factorial: patch upstream (matches our fork's fix)
sed -i 's/np\.math\.factorial/math.factorial/' /tmp/pheasy_up/lib/python*/site-packages/pheasy/core/cluster_orbit.py
sed -i '/^import itertools$/a import math' /tmp/pheasy_up/lib/python*/site-packages/pheasy/core/cluster_orbit.py
```

## Inputs (shared, produced once)
- POSCAR (unit cell) + disp_matrix.pkl / force_matrix.pkl (synthetic: independent
  real-space F = -Phi2 u - 0.5 Phi3:u u; Phi2/Phi3 = symmetry-consistent fc from a
  real fc2/fc3 truncated inside cutoff; displacements large enough that 3rd-order
  is >10% of the force).
- Phi3_known.npy / Phi2_known.npy

## Run both
```bash
# upstream (isolated venv)
/tmp/pheasy_up/bin/python -c "import sys; sys.argv=['pheasy','--dim','4','4','4','-w','3',
  '-c','-d','-f','--c2','6.0','--c3','6.0','--disp_file','-l','OLS','--hdf5','--full_ifc',
  '--eps','0.001']; from pheasy.run_pheasy import main; main()"
# pheasy-gpu (its own env)
pheasy-gpu --dim 4 4 4 -w 3 -c -d -f --c2 6.0 --c3 6.0 --disp_file -l OLS --hdf5 --full_ifc --eps 0.001
```

## Assert (on the SAME known Phi3)
- rel fit error equal to ~0.1%
- fc3 recovery on on-site / two-equal / distinct triplets identical (measured
  1e-5 / 5e-4 / 8.4% for BOTH; the 8.4% distinct and the non-machine on-site/
  two-equal are UPSTREAM behavior, not introduced by the rewrite)
- phi vectors agree to ~1e-5 relative (NS subspaces verified identical: QR
  orthonormal bases have cross-Gram singular values = 1.00000000, stacked rank
  equals rank)

## Measured result (2026-09-05, Si 4x4x4, c2=c3=6.0, 30 configs, u rms 0.06 A)
| metric | upstream pheasy 0.0.2 | pheasy-gpu |
|---|---|---|
| rel fit err | 5.477e-4 | 5.456e-4 |
| fc3 on-site | 1e-5 | 1e-5 |
| fc3 two-equal | 5.1e-4 | 5e-4 |
| fc3 distinct | 8.43% | 8.4% |
| phi vs upstream | - | max rel 2.2e-5 |

Conclusion: the fit/constraint rewrite is numerically equivalent to upstream;
kappa-scale deviations are data/口径, not rewrite artifacts.