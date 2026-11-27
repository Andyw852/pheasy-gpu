# Mode-resolved phonon scattering data (Mg4C60)

Per-mode anharmonic and isotope scattering data extracted from the **full-mesh
Wigner runs** that produced the kappa values in ../kappa/. They are the raw
material for scattering-rate / mean-free-path figures.

## Files

| file | content |
|---|---|
| scattering_modes_c3_3p50.npz | per-mode arrays, 3.5 A case (MAIN value) |
| scattering_modes_c3_3p00.npz | per-mode arrays, 3.0 A case |
| scattering_modes_c3_3p50.csv | derived per-mode quantities for the 3.5 A case |
| fig_scattering_c3_3p50.png / .pdf | three-panel figure (rates + mean free path) |
| ../../scripts/plot_scattering.py | the script that produces the figure and the CSV |
| scattering_section.html | ready-to-paste HTML section for a report (Chinese) |
| insert_scattering_section.py | helper that inserts the section into an existing report.html |

## Arrays in the .npz

| key | shape | unit | meaning |
|---|---|---|---|
| frequency | (150, 384) | THz | phonon frequency at each irreducible q-point / branch |
| gamma | (1, 150, 384) | THz | **ph-ph** linewidth (imaginary self-energy) at 300 K |
| gamma_isotope | (150, 384) | THz | **natural-isotope** linewidth |
| heat_capacity | (1, 150, 384) | eV/K | mode heat capacity at 300 K |
| mode_kappa_P_RTA | (1, 150, 384, 6) | W/(m.K) | mode-resolved particle kappa (Voigt xx,yy,zz,yz,xz,xy) |
| group_velocity | (150, 384, 3) | THz.A = A/ps | d(omega)/dq (recomputed from FC2 with phonopy) |
| weight | (150,) | - | k-star weight of each irreducible q-point (sum = 512) |
| grid_point | (150,) | - | index of the q-point in the 8x8x8 grid |
| qpoint | (150, 3) | fractional | q-point coordinates |
| temperature | (1,) | K | 300 |
| mesh | (3,) | - | 8 8 8 |

## Conventions

* phono3py stores the linewidth gamma in **THz**; the mode lifetime is
  tau = 1/(2*gamma), which gives tau in **ps** (1/THz = 1 ps).
* scattering rate: tau^-1 = 2*gamma (THz).
* group velocity is in THz.A = A/ps; mean free path = |v| * tau (A), divided by
  10 for nm.
* the figure drops the Gamma point and modes with frequency < 0.1 THz.
* kappa_C (coherent) is handled separately and is not part of these mode arrays.

## Where the data come from

| case | source hdf5 | fc2 | fc3 |
|---|---|---|---|
| 3.5 A | 3090: Mg4C60_8cube_b3_wigner/kappa-m888.hdf5 | 313d0f61 | 6683f98e |
| 3.0 A | jzzn: Mg4C60_wigner_b2/kappa-m888.hdf5 | 313d0f61 | d78fbaf2 |

Both runs: phono3py 3.24.0, Wigner RTA, 8x8x8 mesh, 150 irreducible q-points,
384 branches (128-atom primitive cell), 300 K, isotope scattering on.

The two cases share the same FC2 (313d0f61), hence identical phonon frequencies
(the stored spectra agree to 5e-7 THz); only FC3 differs, so the scattering
rates are directly comparable.

Group velocities are not stored in the phono3py output. They were recomputed from
FC2 on the 3090 cluster node (SLURM job 7747, mace-gpu env, phonopy, ~97 s); the
frequencies used for the recomputation agree with the Wigner output to
5.7e-7 THz. (An independent run on a different machine gives the same velocities
to within 1.2e-10 A/ps.)

## eigenvectors/

eigenvectors/ holds the phonon **eigenvectors** and **group velocities** on the
same 150 irreducible q-points (3 npz parts < 60 MB each + manifest.json + README),
computed from the same FC2 with phonopy. See eigenvectors/README.md.

## Using the section in a report

The folder also contains a self-contained HTML section and a small helper:

    python insert_scattering_section.py /path/to/report_folder            # dry run
    python insert_scattering_section.py /path/to/report_folder --write    # apply

It copies the figure into <report>/figures/ and replaces the section that starts
at the first <h2> containing "声子散射"/"散射"/"scattering" (or appends before
</body> if no such heading exists), keeping a report.html.bak backup.

## Reproducing the figure

    python plot_scattering.py scattering_modes_c3_3p50.npz -o fig_scattering_c3_3p50

(needs numpy, matplotlib; the npz already contains the group velocities, so no
phonopy/phono3py call is required).
