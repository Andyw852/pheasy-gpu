# Mg4C60 (C120Mg8) — force-constant fits at 3.0 Å / 3.5 Å and Wigner κ sensitivity

This folder archives the **pheasy force-constant fits** behind the lattice-thermal-conductivity
study of Mg4C60 (C120Mg8), and documents how the choice of the **second-order (FC2) and
third-order (FC3) fitting protocol** propagates into the Wigner κ.

All five fits share **one and the same** force/displacement dataset. Only the FC3 neighbour
cutoff and the treatment of FC2 differ. The point of the archive is therefore: *same data,
same material, same κ engine — what changes when the fit protocol changes?*

---

## 1. Dataset (shared by all fits)

| item | value |
|---|---|
| supercell | 2×2×1, **512 atoms** |
| frames | 297 stored (`dataset_disps.npy`, `dataset_forces.npy`, shape `(297, 512, 3)`); **296 used** in the fits |
| equilibrium force subtracted | yes (`subtract_equilibrium = true`, `eq_force_max = 0.2`) |
| coordinates | fractional |
| FC3 cutoff candidates | 3.0 / 3.5 / 4.0 / 4.5 Å (shells at 1.4511 / 1.6065 / 2.4723 / 5.4953 Å) |
| fit engine | `pheasy-gpu`, method **RFE-OLS** (recursive feature elimination + OLS) |
| harmonic block | `HARM_DENSE = 1` → all **74017** harmonic columns always retained |
| md5 `dataset_disps.npy` | `edf0c4fa92ff19fb5c2ceded9dc79e51` |
| md5 `dataset_forces.npy` | `9037c5972b3fa8287ebe05fe902b5e6f` |

Files: `00_shared_inputs/` (POSCAR, SPOSCAR, SPOSCAR.dataset, both dataset arrays, `fc_dataset.json`).

---

## 2. Two fitting protocols

**(a) Joint refit — FC2 and FC3 solved simultaneously.**
The design matrix is the block `[A_harm | A_anharm]` and the least-squares problem
`F = A_harm·x_FC2 + A_anharm·x_FC3` is solved **in one shot**; RFE prunes only the anharmonic
block (harmonic columns are never pruned because of `HARM_DENSE=1`).

* 3.0 Å → 74017 + 10540 = **84557** free IFCs
* 3.5 Å → 74017 + 16311 = **90328** free IFCs

**(b) Frozen FC2 — only FC3 is fitted** (`fix_fc2`).
FC2 is read from file (here: the FC2 produced by fit **01**, md5 `313d0f61…`) and subtracted
from the forces; then only the anharmonic block is solved:
`F - F_FC2 = A_anharm · x_FC3`.
This is much cheaper (10540 / 16311 unknowns instead of ~90000) but it *pins* the 2nd-order
force constants at the 3.0 Å solution, so the 3.5 Å run no longer has a self-consistent FC2.

---

## 3. The five fits

| folder | c3 cutoff | protocol | free IFCs | selected support | fc2 md5 | fc3 md5 | train RMSE (eV/Å) | CV RMSE |
|---|---|---|---|---|---|---|---|---|
| `01_3p0A_joint_7683` | 3.0 Å | joint | 84557 | 7683 / 10540 anharm (total 81700) | `313d0f61…` | `fed9fc93…` | 0.0066888 | — |
| `02_3p5A_joint_82686_exactgram` | 3.5 Å | joint | 90328 | 82686 / 90328 (8669 / 16311 anharm) | `dd90672d…` | `45594abf…` | **0.0066560** | 0.0102976 |
| `03_3p5A_joint_82686_old` | 3.5 Å | joint | 90328 | 82686 / 90328 | `f19f63f0…` | `0ec23e7d…` | 0.0066578 | — |
| `04_3p0A_frozenFC2_7360` | 3.0 Å | frozen FC2 | 10540 | 7360 / 10540 | `313d0f61…` | `d78fbaf2…` | 0.0066978 | 0.0069095 |
| `05_3p5A_frozenFC2_9765` | 3.5 Å | frozen FC2 | 16311 | 9765 / 16311 | `313d0f61…` | `6683f98e…` | 0.0065826 | 0.0068391 |

Notes:

* **02 vs 03** are two independent RFE runs of the *same* recipe (3.5 Å joint, CV+1SE rule).
  They select the same number of features (82686) but the supports differ by **117 / 82686
  entries (99.86 % overlap)**; the resulting FCs differ (fc2 and fc3 md5 above), while the
  training RMSE differs by only −0.027 %.
* `pheasy_relative_error` is **not comparable** between the joint and frozen-FC2 groups — the
  normalisation denominator differs (≈0.0077 vs ≈0.090). Compare RMSE instead, or
  `pheasy_re_total_forces` (0.00771 for 04, 0.00758 for 05).
* `rfe_support.npy` holds the **selected feature indices**: for joint fits it is
  `74017 harmonic + N_anharm` (81700 for 01, 82686 for 02/03); for frozen-FC2 fits it is
  `N_anharm` only (7360 / 9765).
* In 04/05 the frozen second-order FC is stored as **`fc.hdf5`** (original filename); its md5
  equals `01_3p0A_joint_7683/fc2.hdf5` — i.e. FC2 is literally the 3.0 Å joint solution.
* All fits pass the phonon-stability gate (smallest frequency ≈ −5e-7 THz at Γ, "noise" class).

---

## 4. Lattice thermal conductivity (Wigner κ = κ_P + κ_C)

Wigner **κ_P (particle) + κ_C (coherent)** from phono3py 3.24.0:
q-grid **8×8×8**, **300 K**, **including isotope scattering**, 150 irreducible q-points.
Values in W m⁻¹ K⁻¹.

| case | cutoff | κ_P | κ_C | κ_total |
|---|---|---|---|---|
| 01 joint 7683 | 3.0 Å | 1.5217 | 0.5416 | **2.0633** |
| 02 joint 82686 (exactgram) | 3.5 Å | 1.2355 | 0.5465 | **1.7821** |
| 03 joint 82686 (old) | 3.5 Å | 1.1853 | 0.5475 | **1.7328** |
| 04 frozen FC2, 7360 | 3.0 Å | 1.4406 | 0.5424 | **1.9830** |
| 05 frozen FC2, 9765 | 3.5 Å | 1.3783 | 0.5458 | **1.9240** |

### Cutoff effect depends entirely on the protocol

| protocol | 3.0 Å | 3.5 Å | Δκ_P | Δκ_C | Δκ_total |
|---|---|---|---|---|---|
| **A** both cutoffs fully joint (3.5 Å = exactgram) | 2.0633 | 1.7821 | **−18.80 %** | +0.92 % | **−13.63 %** |
| **A′** both cutoffs fully joint (3.5 Å = old) | 2.0633 | 1.7328 | −22.10 % | +1.09 % | −16.02 % |
| **B** identical frozen FC2, only FC3 refit | 1.9830 | 1.9240 | −4.32 % | +0.61 % | −2.97 % |

Route A, component by component:

| quantity | 3.0 Å joint | 3.5 Å joint (exactgram) | change |
|---|---|---|---|
| κ_P mean | 1.521700 | 1.235547 | −18.80 % |
| κ_P xx / yy / zz | 0.842917 / 2.035415 / 1.686768 | 0.630795 / 1.742413 / 1.333434 | −25.17 / −14.40 / −20.95 % |
| κ_P principal 1/2/3 | 0.713 / 1.817 / 2.035 | 0.509 / 1.456 / 1.742 | −28.63 / −19.89 / −14.40 % |
| κ_C mean | 0.541578 | 0.546538 | +0.92 % |
| κ_C xx / yy / zz | 0.235127 / 0.829248 / 0.560360 | 0.241716 / 0.830911 / 0.566987 | +2.80 / +0.20 / +1.18 % |
| κ_total mean | 2.063278 | 1.782085 | −13.63 % |
| κ_total xx / yy / zz | 1.078044 / 2.864663 / 2.247128 | 0.872511 / 2.573324 / 1.900421 | −19.07 / −10.17 / −15.43 % |
| κ_C share of κ_total | 26.3 % | 30.7 % | +4.4 pts |

---

## 5. Findings

1. **κ_C is remarkably robust.** Across all five fits (two cutoffs × two protocols) κ_C spans
   0.5416–0.5475 W m⁻¹ K⁻¹, i.e. **≤1.1 %** peak-to-peak. The coherent channel is essentially
   fixed by FC2 + crystal structure.
2. **κ_P carries the entire variation.** It moves by −4.3 % (protocol B) to −18.8 %
   (protocol A) when going 3.0 Å → 3.5 Å, and by +4.2 % on an independent same-recipe refit at
   fixed cutoff.
3. **Fit reproducibility floor ≈ 2.8 % on κ_total.** Comparing 02 (exactgram) with 03 (old):
   κ_P **+4.24 %**, κ_C **−0.17 %**, κ_total **+2.84 %**, from an FC2 that differs by only
   0.0276 % in Frobenius norm and an FC3 support that differs by 117/82686 entries. The κ_P
   change is **systematic** (all 25 q-point chunks shift in the same direction, +1.4 %…+12.5 %,
   while κ_C stays within ±0.7 %), so it is a genuine FC3 effect, not numerical noise.
   A linear extrapolation from the support-overlap fraction predicted only ~0.05 % — it
   underestimates the true sensitivity by ~60×.
4. **"Cutoff convergence" is protocol-dependent.** Under the frozen-FC2 protocol the
   3.0 → 3.5 Å change is only −3.0 %, which looks converged; under a self-consistent joint
   refit at both cutoffs it is −13.6 % and still decreasing. The frozen route suppresses the
   cutoff dependence by construction, because FC2 is held at the 3.0 Å solution.
5. **Practical consequence.** κ_total of this material must always be quoted together with the
   FC protocol: the same data set yields 1.73–2.06 W m⁻¹ K⁻¹ depending on how FC2/FC3 are fitted.
   Within one protocol, the fit-to-fit reproducibility is ≈3 %, which should be treated as a
   floor on any claimed precision.

---

## 6. What is deliberately NOT included

| file | size | why |
|---|---|---|
| `cs.pkl` | 117–123 MB | cache; exceeds GitHub's 100 MB per-file hard limit and is fully regenerable |
| `FORCE_CONSTANTS`, `FORCE_CONSTANTS_2ND`, `FORCE_CONSTANTS_3RD` | 15–63 MB | text export of the same tensors already stored in `fc2.hdf5` / `fc3.hdf5` |
| `fm1d.npz`, `force_matrix.pkl`, `disp_matrix.pkl`, `neighbor_list.pkl`, `phi*.npz`, `sm_*.npz` | — | intermediate caches of the fitter |
| κ output (phono3py `kappa-*.hdf5`) | GB-scale | can be regenerated from the FC files above |

If the caches or the κ outputs are needed, they can be provided through a release asset instead
of git history.

---

## 7. Reproducing

Each fit folder contains the exact driver/scheduler scripts used (`submit.sh`,
`run_joint_exactgram.sh`, `run_b1_formal.sh`, `run_b3.sh`), the merged configuration
(`fit_config.json`), the fitter's environment dump where available (`pheasy_env.txt`), the
command line (`pheasy_f_cmdline.txt`), and the full fit log. The essential switches are:

    # joint (01 / 02 / 03): FC2+FC3 in one least squares, RFE on the anharmonic block only
    PHEASY_C3_CUTOFF=3.5  HARM_DENSE=1  PHEASY_RFE_1SE=1  PHEASY_EXACT_GRAM_ALL=1  pheasy-gpu ...

    # frozen FC2 (04 / 05): subtract FC2, solve FC3 only
    fix_fc2=fc.hdf5  PHEASY_C3_CUTOFF=3.5  PHEASY_RFE_1SE=1  pheasy-gpu ...

The κ values above were produced from these `fc2.hdf5` + `fc3.hdf5` pairs with phono3py 3.24.0
(Wigner RTA, 8×8×8, 300 K, isotope on). When a q-grid subset is passed via `grid_points`,
`conductivity.set_kappa_at_sigmas()` must be called explicitly, otherwise κ comes back as zero.
