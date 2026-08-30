# pheasy-gpu: GPU-accelerated force-constant fitting

pheasy-gpu is a drop-in CUDA (PyTorch) backend for pheasy. It keeps the exact
same control flow -- grouped cross-validation, alpha grids, standardization,
relaxed-LASSO debias, recursive feature elimination -- and moves only the heavy
dense linear algebra onto the GPU. The original pheasy is untouched; this is a
separate package named `pheasy_gpu` so both can be installed side by side.

## What is accelerated

| Method | CPU (scipy/sklearn) | GPU (torch/cuSOLVER) |
|---|---|---|
| OLS | `scipy.linalg.lstsq` (gelsd SVD) | `torch.linalg.svd` + rcond solve |
| RIDGE | `sklearn.RidgeCV` (generalized CV) | SVD hat-matrix GCV (identical alpha) |
| LASSO | `sklearn.LassoCV` (coordinate descent) | Gram-based FISTA (GPU by default) |
| ALASSO | ridge pilot + `LassoCV` on scaled cols | ridge pilot + FISTA with per-column weights |
| RFE | `scipy.linalg.lstsq` per subset | rank-checked QR + `gels` per subset, SVD fallback |
| SM loading | `sm_prime @ NS` (sparse) on CPU | `torch.sparse.mm` on GPU (**holdout_eval only**) |

**LASSO/ALASSO default to the GPU Gram-based FISTA.** FISTA solves the exact
same convex problem as sklearn's coordinate descent and, once the per-iteration
host sync is amortised (`PHEASY_FISTA_RESTART_EVERY`, default 5), is several
times faster on the 3090 for the `holdout_eval` matrices. The FISTA Lipschitz
constant is `lambda_max(G)`, computed exactly with `torch.linalg.eigvalsh`
(matching the CPU Gram path); an earlier power-iteration version returned
`lambda_max^2`, shrinking the step by ~lambda_max and stalling FISTA inside
`cv_max_iter` (dense, non-sparse results). Re-checked on the 3090
(`dev/recheck_lasso.py`): GpuLassoCV coef agrees with sklearn LassoCV to ~4e-9
(synthetic); on the c7 SM (n=8) alpha_/support are identical and the raw FISTA
coef (debias off) agrees to ~3.5e-4 / 5.9e-6. Set
`PHEASY_GPU_LASSO=0` to force the
sklearn coordinate-descent path when you want bit-identical LASSO against the
original pheasy.

**RFE uses QR for the per-subset solves**, not the SVD. QR is
backward-stable for the full-rank subsets and ~50x faster on the 3090 than the
FP64 SVD. `qr_solve` factorizes R only (`mode="r"`, no Q materialization),
checks its diagonal (the same threshold as `_solve_qr`), falls back to the SVD
for rank-deficient or wide subsets, then runs the fast cuSOLVER `gels` solve;
CUDA `gels` silently returns NaN/inf on rank-deficient inputs, so the earlier
try/except fallback never fired. Verified `qr_solve` vs
`numpy.linalg.lstsq` to ~5e-15 (full rank).

**`RFE-OLS-TSQR` (alias `RFE-TSQR`) is also GPU-accelerated.** Its dense
subset solves go through the same `qr_solve` path as RFE (the Q-less tall-skinny
blocked QR is a memory optimisation for CPU tall matrices; on the GPU the same
rank-checked QR + `gels` path is used). Its BIC/AIC stopping rule is opt-in via
`PHEASY_TSQR_CRITERION=bic`; the default is `cv`, which makes it select the same
support as RFE. Verified on n=8: ~72 s, nnz=2092.

## Usage

Activation (in priority order):

1. `Optimizer(..., use_gpu=True/False)` -- explicit process-wide override
   (last-created Optimizer wins; there is no per-instance isolation).
2. `PHEASY_USE_GPU` env var: `0`/`off` forces CPU, `1`/`on` forces GPU,
   unset -> auto (GPU when `torch.cuda.is_available()`).

```bash
# auto (uses GPU when available)
python holdout_eval.py <data_dir> --methods OLS RIDGE LASSO ALASSO RFE \
    --n-configs 24 --n-splits 5

# force GPU (fail loudly if unavailable: set CUDA_VISIBLE_DEVICES first)
CUDA_VISIBLE_DEVICES=6 PHEASY_USE_GPU=1 python holdout_eval.py <data_dir> ...

# force CPU (A/B baseline)
PHEASY_USE_GPU=0 python holdout_eval.py <data_dir> ...
```

## Tuning knobs

* `PHEASY_GPU_DEVICE` -- CUDA device index (default `0` / first visible
  device); read fresh on every call (no caching).
* `PHEASY_GPU_MEM_FRACTION` -- fraction of free VRAM a dense solve may occupy
  before it falls back to the CPU (default 0.8).
* `PHEASY_FISTA_RESTART_EVERY` -- how often (iterations) the FISTA adaptive-
  restart overshoot check syncs to the host (default 5). Higher = fewer syncs,
  marginally less-frequent restarts; the fixed point is unchanged.
* `PHEASY_GPU_LASSO` -- `0` routes LASSO/ALASSO to the sklearn coordinate-
  descent path (bit-identical to original pheasy); unset/`1` uses the GPU FISTA
  (default).
* `PHEASY_LASSO_1SE` -- `1` applies the one-standard-error rule to LASSO
  alpha selection on all backends (dense, iterative, and GPU).

## Measured performance (RTX 3090, c7 sensing matrix 25515x6588)

`holdout_eval` n=8, one split, single free 3090 (CUDA_VISIBLE_DEVICES=6):

| step | CPU | GPU |
|---|---|---|
| SM load (25515x6588 @ 6588x3678) | ~1421 s | ~28 s |
| OLS (4536x3678, gelsd vs SVD) | ~600+ s | ~21 s |
| RIDGE (50-alpha GCV) | ~1700 s (n=24) | ~16 s |
| LASSO (20-alpha grouped CV) | ~939 s | ~40 s |
| ALASSO (20-alpha weighted CV) | ~940 s | ~35 s |
| RFE (step 0.05, 3-fold) | SVD per subset (hours) | ~99 s (QR; pre-rank-check) |

At the full n=24 scale the CPU baseline measured ~1804 s (OLS) and ~1715 s
(RIDGE) per fold on the shared box; the GPU SVD for 13608x3678 is ~28 s, so
the end-to-end holdout drops from hours to minutes.

> The RFE timing predates the R-only rank-check pass (re-run before quoting).
> LASSO/ALASSO were re-measured after the fixes: 38/41 s GPU FISTA vs 418/306 s
> sklearn CD on the c7 SM (n=8); see the recheck note in "Verified numerics".

Verified numerics (GPU vs CPU):
* `lstsq` vs `numpy.linalg.lstsq`: ~1e-15
* `ridge_solve` vs `sklearn.Ridge`: ~1e-15
* `GpuRidgeCV.alpha_` vs `sklearn.RidgeCV.alpha_`: exact match
* `qr_solve` vs `numpy.linalg.lstsq`: ~5e-15
* `GpuLassoCV.alpha_` vs `sklearn.LassoCV.alpha_`: identical; coef rel diff
  ~4e-9 (synthetic), ~3.5e-4 / 5.9e-6 (c7 SM, n=8, debias off, tol=1e-6) after
  the fix -- the 3.5e-4 is a tol artifact, see the fixed-alpha scan below.

Recheck after the Lipschitz fix (`dev/recheck_lasso.py`, RTX 3090):
* level 1: `_power_lipschitz == lambda_max(G)` to ~1e-15 (no lambda_max^2).
* level 2 (synthetic): alpha_ identical, nnz identical, coef rel diff 3.9e-9.
* level 3 (c7 SM, n=8): alpha_ and support identical for LASSO (7.3236e-08,
  nnz 3398) and ALASSO (7.3236e-10, nnz 3102). Raw FISTA-vs-CD coefficients
  (debias OFF): LASSO coef rel diff 3.5e-4, ALASSO 5.9e-6. With debias ON
  (the default) the delivered relaxed-LASSO coefficients are bit-identical
  because both backends run the same numpy OLS refit on the same support.
* level 3 (c7 SM, n=8): both backends print the same "grid MINIMUM" warning at
  `decades=4.0` (the holdout_eval default): alpha_ sits at the grid lower bound
  and the CV curve is still falling, so nnz is near-dense (~92%). This is a
  grid/data property of the holdout config, not a FISTA bug; `run_pheasy` can
  widen the grid via `PHEASY_ALPHA_DECADES`.
* Fixed-alpha scan (`dev/alpha_scan.py`, no CV, tol=1e-8 CD / 1e-7 FISTA):
  LASSO |FISTA-CD|/|CD| vs nnz on the c7 SM degrades monotonically with
  conditioning -- 1.8e-10 (1.5% nnz) -> 8e-9 (12%) -> 1.2e-7 (57%) -> 4.5e-7
  (90%) -> 1.25e-6 (98.5%). Neither solver hit its iteration cap: sklearn CD
  n_iter <=1438 (no ConvergenceWarning) and FISTA n_iter <=600 (final-refit cap
  5000), so the curve is a genuine solver difference, not a cap artifact. A
  tol=1e-12 CD reference is a certified KKT point (KKT violation 1.7e-10 /
  1.4e-9 / 1.3e-8 at 57% / 90% / 98.5% nnz) but, with lam_min(A.T A / n) =
  6.7e-7, its own coefficient error is only bounded to sqrt(2*gap/mu) = 2.6e-4,
  so the err_fista/err_cd distances to it (9.1e-8/2.8e-8, 3.4e-7/4.7e-7,
  2.5e-7/1.1e-6) sit below its resolution limit, not certified distances to the
  true minimizer. The reference is itself CD, so err_cd is if anything
  understated by correlated error (same sweep/active-set as the loose CD), and
  err_cd still exceeds err_fista at 98.5% -- so the |F-C| gap is the vector
  difference of two comparable stopping-criterion errors, not a one-sided FISTA
  error. The level-3 LASSO 3.5e-4 is therefore the recheck's tol=1e-6 stopping
  early in the near-OLS regime, not a conditioning floor.

## Memory (RTX 3090, full n=45 dense SM 25515x3678)

Peak GPU memory by operation:

| operation | peak |
|---|---|
| SM load (`sm_prime @ NS`, CSR sparse.mm) | ~2.3 GB |
| dense SM tensor (float64) | 0.75 GB |
| OLS / RIDGE SVD (`U`, `S`, `Vh` + workspace) | ~3.2 GB |
| RFE / TSQR QR (rank-check + `gels` + workspace) | ~2.4 GB (re-measure) |
| LASSO / ALASSO FISTA (per-fold Gram + A_va copies) | ~0.5-1.5 GB (see note) |

The 24 GB 3090 is therefore far from memory-bound at n=45 (peak ~3.2 GB); the
binding constraint is FP64 compute (the SVD), not memory. Memory notes:

* **LASSO/ALASSO work on the Gram (`p x p` = 108 MB at p=3678), but
  `GpuLassoCV` holds one Gram per CV fold plus one `A_va` copy per fold** --
  5-fold p=3678 is ~540 MB of Grams alone, so a single-Gram figure understates
  the CV peak.
* **Loading uses CSR** (not COO): ~5x faster and ~half the sparse-tensor memory,
  and it slices `sm_prime[:n_rows]` before multiplying.
* **The backend is uniformly float64.** A `PHEASY_GPU_DTYPE=float32` mode was
  removed because it only affected a few entry points and silently mixed
  precisions.
* The dense SM (`n x p`) materialisation is inherent to the dense path; the
  `TwoLevelSM` LinearOperator path (still CPU) avoids it for very large systems.

## Correctness check (recommended)

The LASSO/ALASSO numbers below depend on three defaults documented nowhere
else; set these first to reproduce them:

| knob | default | effect |
|---|---|---|
| `PHEASY_CV_TOL` | `max(tol, 1e-3)` | the CV path runs at 1e-3, not the caller's `tol` |
| `PHEASY_CV_MAX_ITER` | `min(max_iter, 800)` | the CV path is capped at 800 iterations |
| `PHEASY_LASSO_DEBIAS` | `1` | `results["coef"]` is the OLS refit on the support, not the raw FISTA/CD solution |

Run a small case twice and diff:

```bash
PHEASY_USE_GPU=1 python holdout_eval.py <data_dir> --methods OLS LASSO --n-configs 6 \
    --n-splits 3 --seed 0 > gpu.out
PHEASY_USE_GPU=0 python holdout_eval.py <data_dir> --methods OLS LASSO --n-configs 6 \
    --n-splits 3 --seed 0 > cpu.out
diff <(grep -E "OLS|LASSO" gpu.out) <(grep -E "OLS|LASSO" cpu.out)
```

OLS should match to ~1e-8 (identical SVD). LASSO/ALASSO match to ~1e-9..1e-7 in
the sparse regime (nnz < 60%), degrading to ~1e-6 as nnz -> 100%. That is the
joint relaxation of the two solvers' stopping criteria in the near-OLS regime,
not a GPU-specific limit: the CPU `_LassoCVIterative` shares the same tol/iter
caps as `GpuLassoCV`. The fixed-alpha scan (`dev/alpha_scan.py`) measured
LASSO |FISTA-CD|/|CD| 1.8e-10 -> 1.25e-6 as nnz ran 1.5% -> 98.5% on the c7 SM.
The recheck's raw (debias-off) LASSO 3.5e-4 at ~92% nnz is a tol=1e-6 artifact,
not a conditioning floor. To force the identical sklearn solver for a bit-exact
LASSO diff, add `PHEASY_GPU_LASSO=0` to both runs.

## Limitations

* The two-level `TwoLevelSM` LinearOperator path (used by `run_pheasy` for very
  large systems that never materialize the dense sensing matrix) still runs its
  LSMR/FISTA matvecs on the CPU.
* **GPU SM loading is wired into `holdout_eval.py` only.** The `pheasy-gpu`
  CLI (`run_pheasy.py`) still assembles `SM_prime @ NS` with scipy on the CPU;
  the "SM load 1421s -> 28s" row above applies to `holdout_eval`, not to the
  CLI. The CLI *does* get GPU-accelerated fitting (OLS/LASSO/ALASSO/RIDGE/RFE
  solves) once SM is assembled.
* `torch.linalg.lstsq` on CUDA only exposes `driver="gels"`, so `lstsq()` here
  reimplements the SVD (gelsd) solve with `torch.linalg.svd` -- numerically
  equivalent to scipy, at a small constant-factor cost.
* FP64 throughput on a consumer 3090 is ~1/64 of FP32, but the dense fits here
  are still 10-60x faster than the single-machine CPU LAPACK path.

