# Mg4C60 phonon eigenvectors + group velocities (8x8x8 irreducible q-points)

Phonon **eigenvectors** and **group velocities** on the same 150 irreducible
q-points used by the Wigner kappa calculations in ../../kappa/, computed from the
same harmonic force constants (FC2 md5 313d0f61).

## Files

| file | q range | content |
|---|---|---|
| mg4_eigenvectors_part1of3.npz | 0-49 | qpoint, frequency, eigenvectors, group_velocity, species_weight |
| mg4_eigenvectors_part2of3.npz | 50-99 | same |
| mg4_eigenvectors_part3of3.npz | 100-149 | same |
| manifest.json | - | q ranges, byte sizes, md5 of each part |

Each part is < 60 MB (GitHub single-file limit), and together they cover all
150 q-points x 384 branches.

## Arrays (in each part)

| key | shape | dtype | unit | meaning |
|---|---|---|---|---|
| q_index | (nq_part,) | int32 | - | index of the q-point in the 8x8x8 grid list (0..149) |
| qpoint | (nq_part, 3) | float32 | fractional | q-point coordinates |
| frequency | (nq_part, 384) | float32 | THz | phonon frequencies (ascending branch order) |
| eigenvectors | (nq_part, 384, 384) | complex64 | - | phonon eigenvectors, shape (nq, nband, 3*natom); normalized so sum_i |e_i|^2 = 1 |
| group_velocity | (nq_part, 384, 3) | float32 | THz*A (= A/ps) | d(omega)/dq |
| species_weight | (nq_part, 384, 2) | float32 | - | eigenvector weight on [Mg, C] sublattices (sums to 1) |
| mesh | (3,) | int32 | - | [8, 8, 8] |
| species_names | (2,) | str | - | ['Mg', 'C'] |

## How it was produced

    from phonopy import Phonopy
    from phonopy.interface.vasp import read_vasp
    from phono3py.file_IO import read_fc2_from_hdf5

    uc = read_vasp("POSCAR")
    ph = Phonopy(uc, supercell_matrix=[[2,0,0],[0,2,0],[0,0,1]], primitive_matrix="P")
    ph.force_constants = read_fc2_from_hdf5("fc2.hdf5")
    ph.run_qpoints(qpoints, with_eigenvectors=True, with_group_velocities=True)
    # ph.qpoints.frequencies, ph.qpoints.eigenvectors, ph.group_velocity.group_velocities

Run on the 3090 cluster node (SLURM job 7926, mace-gpu env), 359 s for all
150 q-points. Frequencies agree with the Wigner kappa output to 1.8e-7 THz.

## Conventions / caveats

* branch order = ascending frequency (same order as the frequency array);
* eigenvectors are complex and **phase-gauge dependent** (phonopy picks a
  deterministic phase per eigenvector, but any physical quantity must be built
  from |e|^2 or from gauge-invariant combinations);
* the eigenvector basis is the primitive cell (128 atoms -> 384 degrees of
  freedom), matching the primitive cell used for the Wigner kappa runs;
* the 3.0 A FC3 case shares the same FC2, so its eigenvectors are identical to
  these (harmonic properties do not depend on FC3).
