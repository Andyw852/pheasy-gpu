# Third-order force constants of Mg4C60 — cutoff truncation vs the acoustic sum rule

This note documents an experiment that separates two things which are easy to
confuse when a force-constant **cutoff** is changed:

1. the *physical* effect of restricting FC3 to shorter-range triplets, and
2. the *artefact* introduced when a fitted FC3 is simply **zeroed** beyond a
   distance cutoff without re-imposing the constraints that the fitter had built in.

The second effect turns out to be the dominant one for this material, because it
breaks the **acoustic sum rule (ASR)** of the third-order force constants.

---

## 1. What the acoustic sum rule means for FC3

FC2 and FC3 are the second and third derivatives of the total energy with respect
to atomic displacements,

    Phi2(i,j)       = d2E / du_i du_j
    Phi3(i,j,k)     = d3E / du_i du_j du_k

Translational invariance means that displacing **all** atoms by the same vector
must not produce any force. Differentiating that statement twice more gives the
third-order acoustic sum rule:

    sum_j Phi3(i,j,k) = 0    for every (i,k)

and, through permutation symmetry, the sums over i and over k vanish as well.
Equivalently: in reciprocal space the FC3 matrix element must vanish as the
phonon wavevectors go to zero, because a q = 0 acoustic phonon is just a rigid
translation of the crystal.

The pheasy RFE-OLS fit **enforces this constraint**, so the fitted FC3 satisfies it
to machine precision. Measured on the 3.0 A joint fit (c3_3p00_exact):

    max | sum_j Phi3(i,j,k) | = 4.2e-12        (i.e. exactly zero)

---

## 2. Why truncating a fitted FC3 breaks it

The truncation used here takes the 3.5 A joint fit (Mg4C60_joint_3p50_exactgram,
fc2 dd90672d, fc3 45594abf) and sets to zero every triplet (i,j,k) for which any
of the three pair distances exceeds 3.0 A (minimum-image distance in the 512-atom
2x2x1 supercell; nearest pair 1.368 A).

| quantity | value |
|---|---|
| nonzero triplets before truncation | 73328 |
| removed (31.36 %) | 22992 |
| kept | **50336** |
| kept set vs the 3.0 A fit's own triplet set | **identical, 0 missing** |
| max pair distance of kept triplets | 2.9966 A |
| coefficient change vs the 3.0 A fit on the common support | 2.03 % (Frobenius) |
| **max abs(sum_j Phi3) after truncation** | **3.139**  -> ASR violated |

The removed long-range triplets carried a net, non-cancelling contribution to the
sum over j, so the remaining tensor no longer integrates to zero.

---

## 3. Repairing it: projection onto ASR + permutation symmetry

phono3py provides exactly the two projections needed
(phono3py.phonon3.fc3):

    from phono3py.phonon3.fc3 import (set_translational_invariance_fc3,
                                      set_permutation_symmetry_fc3)
    set_translational_invariance_fc3(fc3)   # enforces sum_i = sum_j = sum_k = 0
    set_permutation_symmetry_fc3(fc3)       # restores full index permutation symmetry

Iterating the two to convergence took **one** iteration here:

| quantity | value |
|---|---|
| max abs(sum_j Phi3) after projection | **1.6e-12** (all three indices) |
| **relative change of FC3** (Frobenius) | **0.0959 %** |
| max abs(Phi3) entry | 177.85 (unchanged) |
| sparsity afterwards | 99.6 % nonzero — the correction spreads a tiny uniform component over all triplets, so the tensor becomes dense (2.18 GB) |

---

## 4. Measured impact — one q-point chunk (chunk 3, N = 17)

Because a full 25-chunk kappa run costs ~12 h, the effect was first measured on a
single chunk (the same 6 irreducible q-points in all three runs):

| FC3 used | kappa_P | kappa_C | kappa_TOT |
|---|---|---|---|
| 3.5 A joint, full (no truncation) | **3.6191** | 0.5359 | 4.1550 |
| 3.5 A joint, truncated to 3.0 A | **1.3758** (-62.0 %) | 0.5390 | 1.9148 |
| truncated **+ ASR projection** | **2.7551** (-23.9 %) | 0.5357 | 3.2908 |

* kappa_C is unaffected (0.5357-0.5390, +-0.3 %): the coherent channel is fixed by FC2.
* The ASR projection — which changes FC3 by only **0.096 %** in Frobenius norm —
  **recovers 61.5 % of the truncation-induced drop in kappa_P**
  ((2.7551-1.3758) / (3.6191-1.3758) = 0.615).
* The remaining -23.9 % is the genuine "shorter FC3 range" effect, in as much as a
  zeroed fitted tensor can represent it at all.

---

## 5. Why 0.1 % of FC3 can move kappa by tens of percent

The scattering matrix element is the Fourier transform of FC3 over the supercell,

    Phi3(q,q',q'') = sum_{i,j,k} Phi3(i,j,k) exp[i(q.r_i + q'.r_j + q''.r_k)]

For the **long-wavelength acoustic modes** the phases vary slowly across the
supercell, so the sum is dominated by the lowest moments of Phi3 — precisely the
quantities that the ASR makes vanish. An ASR-violating FC3 has a spurious
"uniform" component that does not vanish as q -> 0, and therefore produces
anomalously strong scattering of exactly the modes with the longest lifetimes,
which carry most of kappa_P. Since kappa_P is proportional to 1/gamma and
gamma is proportional to |matrix element|^2, a small constraint violation is
amplified into tens of percent of kappa.

This also explains the earlier observation that two independent joint fits of the
same data (same support size, 117/82686 features different, FC2 differing by
0.028 % in norm) give kappa_TOT differing by 2.84 %: this quantity is simply very
sensitive to small, low-q-affecting perturbations of FC3.

---

## 6. Consequence for the 3.0 A / 3.5 A comparison

* **The naive truncated value kappa_TOT = 1.5176 (kappa_P = 0.9720) must not be quoted
  as "the 3.0 A result".** About 60 % of its deviation from the 3.5 A value is an
  ASR artefact.
* **The clean way to ask "what does a 3.0 A FC3 give?" is to refit** FC3 with
  c3_cutoff = 3.0 while keeping everything else identical; the fitter then
  re-optimises the short-range coefficients *under* the ASR constraint. That run
  is archived in 07_refit30_c3_3p0/.
* Even the ASR-projected truncation remains an approximation: the retained
  short-range coefficients were optimised in the presence of the long-range terms
  that were later removed, so they are not a re-fit.

Full 25-chunk runs with the ASR-projected FC3 and with the refit FC3 were launched
to complete the comparison; their merged kappa JSONs will be added here.

---

## 7. Files

06_trunc30_kappa/
* fc3_trunc30.hdf5 — the sparse truncated FC3 (fc3 dataset, 512x512x512x3x3x3,
  gzip; md5 85f165e391eb13ebdb524a313258dfa2). This is the *input* to the ASR projection.
* kappa_trunc30_merged.json — 25-chunk Wigner kappa for the truncated FC3
  (no ASR projection), i.e. the artefact-affected number.
* kappa_trunc30_asr_chunk03.json — chunk-3 pilot: all three FC3 variants on the
  same 6 q-points (this note's table).
* provenance.json — exact recipe, distances, validation counters, md5s.

07_refit30_c3_3p0/
* fc3.hdf5 — FC3 refitted with c3_cutoff = 3.0 while FC2 is frozen to the
  3.5 A joint FC2 (dd90672d, i.e. file 02_3p5A_joint_82686_exactgram/fc2.hdf5).
  RFE-OLS, CV=5, step 0.05, 1SE rule -> 7360 of 10540 anharmonic terms kept;
  train RMSE 0.0067372 (0.12 meV/A above the 3.5 A reference, as expected for a
  restricted model); phonons stable (min frequency -0.000 THz).
* fit_config.json, fit_metrics.json, fit_manifest.json, rfe_support.npy,
  fit.log, post.log, phonon_summary.json, fc_fit_summary.json.

### Reproducing the ASR-projected FC3 (the 2.18 GB dense file is not stored)

    import h5py, numpy as np
    from phono3py.phonon3.fc3 import (set_translational_invariance_fc3,
                                      set_permutation_symmetry_fc3)
    with h5py.File("fc3_trunc30.hdf5") as f:
        fc3 = np.array(f["fc3"][:], dtype="double")
    for _ in range(2):
        set_translational_invariance_fc3(fc3)
        set_permutation_symmetry_fc3(fc3)     # residual -> 1.6e-12
    # write fc3 with the same layout (chunks (32,32,64,1,1,1), gzip) if needed
