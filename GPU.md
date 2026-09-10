# pheasy-gpu: GPU-accelerated force-constant fitting

pheasy-gpu is a drop-in CUDA (PyTorch) backend for pheasy. It keeps the exact
same control flow -- grouped cross-validation, alpha grids, standardization,
relaxed-LASSO debias, recursive feature elimination -- and moves dense linear
algebra and optional two-level sparse matrix-vector products onto GPUs. This is a
separate package named `pheasy_gpu` so both can be installed side by side.

## Convergence diagnostics and current operator limits

FISTA checks L1 KKT stationarity at the actual coefficient iterate before
accepting a small-step stop. The relative residual is normalized by
`max(abs(A.T @ y))`; nonconvergence emits `RuntimeWarning`. Final refits honor
the requested `max_iter` and `tol`; CV has separate `PHEASY_CV_MAX_ITER` and
`PHEASY_CV_TOL` settings. A small update or a low training error alone is not
a convergence certificate.

FISTA-backed LASSO/ALASSO expose `Optimizer.results["regularized_solver_info"]`
with `converged`, `kkt_relative`, `n_iter`, and `tol`. Its stage is explicitly
`regularized_refit_before_debias`: it certifies neither every CV candidate nor
the post-debias/thresholded output. Dense sklearn paths retain sklearn warnings
and do not invent a FISTA certificate.

Operator RIDGE/LASSO/ALASSO currently reject `fit_intercept=True` or non-None
`weights` with `NotImplementedError`; iterative LASSO also rejects these
options when selected for a large sparse input. Previously these options could
be ignored. Default force-constant fits (no intercept, no weights) are unchanged.
Full weighted/intercept operator fitting is not implemented. ALASSO also
rejects sample weights on dense inputs because its adaptive pilot is unweighted.
Weighted LASSO with debias enabled is rejected; use `PHEASY_LASSO_DEBIAS=0`
with a supported dense backend for weighted LASSO. These fail-fast checks
prevent partially weighted fits, not implement full weighted fitting.

Mg2C60 c2=7.0/c3=4.5 validation is complete for ordinary RFE, RIDGE,
LASSO and ALASSO on a fixed 80/20 configuration split (float64). Local
regressions pass 29/29; CUDA checks pass 18/18, including full-support debias.
LASSO/ALASSO default to debias (`PHEASY_LASSO_DEBIAS=1`): refit OLS on
selected support, including full support; only empty support is skipped.
`results["pre_debias_coef"]` preserves physical-coordinate coefficients before
debias (without the final output threshold). Operator debias uses LSQR, with
`PHEASY_LSQR_ATOL`, `PHEASY_LSQR_BTOL`, `PHEASY_LSQR_MAXITER`.
Debias is not always better: LASSO holdout RMSE fell 30.20%, while same-run
ALASSO rose 4.98%. These are single-split development results, not a new blind
test or proof of universal method superiority. Timing and memory comparisons
are recorded in `tmp/other_methods_20260907/METHOD_RESOURCE_COMPARISON_zh.md`.

## Opt-in GPU-resident two-level LASSO

`PHEASY_GPU_LASSO_RESIDENT=1` selects the experimental resident backend for
`Optimizer("LASSO")` or `Optimizer("ALASSO")` with `TwoLevelSM` input. Unlike `PHEASY_GPU_SM=1`, which
only accelerates `SM_prime` multiplication and returns vectors to the host,
this backend uploads both `SM_prime` and `NS` (and their transposes), and keeps
normalization, residuals, gradients, soft thresholding, momentum, training-row
masks, validation MSE and KKT calculations on one CUDA device in float64.
No dense sensing matrix or Gram matrix is formed. Small scalar synchronizations
for line search, convergence and logging remain; this is not a CPU-free program.

The resident LASSO/ALASSO path has passed the local single-card CUDA acceptance suite and, as of
2026-09, a real multi-card hardware run on a 6x RTX 3090 host (Torch 2.6.0+cu124, two devices
pinned via `CUDA_VISIBLE_DEVICES`) covering 1-GPU, 2-GPU, 3-GPU, and 6-GPU CV. The 6-GPU run also passed the same 24-check
acceptance matrix; an explicit request
without CUDA, with an unsupported input/method, or with insufficient memory
fails rather than silently claiming GPU execution after a CPU fallback.
`CUDA_VISIBLE_DEVICES` still controls available devices; no GPU allocation is
requested by the solver itself. Resident CV supports dynamic fold scheduling
through `PHEASY_GPU_DEVICES` and `PHEASY_GPU_NGPU`; every selected GPU must
fit a complete factor replica. Hardware acceptance must be reported separately
from CPU-emulated scheduler tests.

### GPU acceleration status (current)

Dense OLS/Ridge and opt-in dense streamed TSQR use CUDA float64 kernels.

`PHEASY_GPU_RFE_RESIDENT=1` retains dense matrices or sparse/TwoLevel factors and targets across RFE subset solves and CV predictions. It supports OLS and positive-alpha Ridge subsets and requires `n_jobs=1` (the existing Jacobi option applies only to operator inputs; `jacobi_applied` reports actual use), with a conservative workspace budget checked before upload. Metadata exposes `resident_subset_inputs` and `resident_fallback_reason`. Dense fits cache CV row indices and the current support matrix on CUDA; sparse/TwoLevel fits reuse factors through vector scatter/gather views without constructing subset matrices. Their solver is CGLS (`gpu_rfe_resident_iterative`), with per-solve convergence diagnostics and `PHEASY_LSQR_ATOL/BTOL/MAXITER` controls; nonconvergence aborts before elimination. `resident_row_index_uploads`, `resident_column_index_uploads`, and `resident_subset_builds` count this work. The pre-upload budget includes the support matrix and index caches (`resident_index_cache_budget_bytes`). An allocation failure during recursion currently aborts the fit; only initial input-upload failures use the fallback path. Training-fold coefficients and validation residuals stay on CUDA; only per-fold RMSE scalars are downloaded. BIC/AIC residual sums are reduced on CUDA and only RSS scalars are downloaded. With GPU ranking enabled, full-fit coefficients stay on CUDA until a tied/nonfinite importance requires NumPy fallback or the final public coefficient vector is produced. Verbose logs download only the nonzero count. With GPU ranking disabled, coefficients are downloaded each round for CPU importance calculation. Column norms are computed on CPU, uploaded once, then reused for CUDA importance and operator Jacobi scaling. Fold aggregation, support updates, patience and selection remain CPU. Metadata exposes `cv_fold_scoring="gpu"`. This is input residency, not a fully resident RFE loop. The existing validator supports `--benchmark-ridge-rfe --benchmark-rows 6000 --benchmark-columns 256` to compare CPU, legacy GPU and resident-input GPU fits. Add `--benchmark-input csr|twolevel` for sparse fixtures (10% density; TwoLevel uses identity NS). On the local RTX 4060, a 6000×256 CSR fixture gave CPU 0.3193 s versus resident GPU 0.5210 s median over three fits, with relative prediction difference 4.06e-10 (`tmp/csr_bench130.json`). A smaller 1200×96 TwoLevel fixture was also slower on GPU. These synthetic results do not establish a sparse speedup; keep residency opt-in and measure the actual workload.

A matched 6000x256 sweep on a confirmed-idle RTX 3090 (Torch 2.6.0+cu124; all six devices
verified at 0% utilization before and after; median of three warm fits) gives the clearest
current picture:

| input | CPU | GPU (non-resident) | GPU resident |
| --- | --- | --- | --- |
| dense | 0.5322 s | 0.3381 s | **0.1621 s** |
| csr, 10% density | 0.5534 s | **0.2829 s** | 0.3392 s |
| TwoLevel, identity NS | 0.2097 s | 0.2328 s | 0.5264 s |
| TwoLevel, sparse-mixed NS | 0.4458 s | 0.4475 s | 0.7916 s |

Dense residency is a real win on this hardware (3.3x vs CPU, and 2.1x vs the non-resident GPU
path, so residency itself -- not merely "using the GPU" -- is what pays). Sparse CSR is ~2x faster
than CPU on the GPU, but residency is ~20% *slower* than the simpler non-resident GPU path: CGLS
iteration and per-solve setup cost more than the subset transfers it avoids at this size. TwoLevel
is the worst case because each `matvec` is two sparse products, so CGLS needs far more work per
solved subset; resident TwoLevel is slower than CPU here and should stay opportunistic. Do not
extrapolate any of these figures to larger problems in either direction -- re-measure the actual
workload. Prediction agreement with CPU stayed at 1e-15 (dense/CSR) and ~4e-10 (CGLS paths).
`PHEASY_GPU_OLS_RESIDENT=1` enables CUDA-resident CGLS for `TwoLevelSM` OLS.
`PHEASY_GPU_LASSO_RESIDENT=1` enables CUDA-resident LASSO and ALASSO: adaptive pilot, weights, weighted FISTA, CV, refit, and KKT run on CUDA.
`PHEASY_GPU_TSQR=1` enables bounded binary-tree TSQR for oversized dense tall full-rank systems; CPU TSQR remains the default. `PHEASY_GPU_RIDGE_RESIDENT=1` enables augmented GPU CGLS for TwoLevel/operator Ridge, with CPU metrics and fallback. Dense RFE subset solves/predictions use CUDA when memory allows, while RFE orchestration, support selection, and postprocessing remain CPU. `PHEASY_GPU_RFE_RANKING=1` opts into CUDA importance sorting; with resident RFE it also computes importance from CUDA coefficients (`gpu_importance_rounds`); otherwise importance calculation remains CPU. Column norms are prepared on CPU and support updates remain CPU. Tied/nonfinite importance falls back to NumPy to preserve its exact ordering. Metadata reports `gpu_ranking_rounds`. This is not a resident RFE loop or a demonstrated speedup. For public OLS, the same `PHEASY_GPU_TSQR=1` flag also enables sparse/TwoLevel streamed TSQR, with `PHEASY_TSQR_BLOCK_ROWS` (default 40000, at least the column count). CPU code assembles and expands each bounded row block; CUDA performs QR, tree reduction and triangular solve. Memory/rank failures fall back to matrix-free LSQR/LSMR with `fallback_reason`; operator ridge/Jacobi options retain their existing path. RFE sparse/TwoLevel residency uses the separate `PHEASY_GPU_RFE_RESIDENT=1` CGLS path, not streamed TSQR. TSQR still requires an O(n_columns^2) dense R factor and workspace.

**Scope:** file I/O, automatic alpha-grid preprocessing (including its own column
norm calculation), CV split construction and final host metrics remain on CPU.
The resident solver performs its own standardization on GPU; this does not
move the earlier CLI alpha-grid preparation to GPU.
The existing optional LASSO OLS-debias stage also remains on CPU and is reported
separately as `postfit_backend`. Setting `PHEASY_LASSO_DEBIAS=0` isolates pure
LASSO for solver validation but changes the delivered estimator relative to
a debiased fit; never present that comparison as an identical full pipeline.
Neither a completed CUDA kernel nor a generated IFC certifies CV convergence
or dynamical stability. Check per-fold and final KKT records.

Tests: `dev/test_gpu_twolevel_lasso.py` covers dispatch, numerical comparison,
normalization and device residency. Torch-CPU emulation and skipped CUDA tests
are not evidence of GPU execution. `dev/validate_resident_lasso.py` reads existing
cache files without modifying them and writes a new exclusive output directory.
Real-data full-fit speed and convergence are not established merely by these
interfaces existing; report the measured device, precision, CV limits and
postprocessing scope with every benchmark.

Initial real-cache validation (RTX 4060, Torch 2.5.1+cu121): a 99792×20125
effective operator with 20 alphas and five row-wise folds took 510.70 s including
CPU automatic-grid preparation, with debias explicitly disabled. Final refit
converged in 4180 iterations (relative KKT 9.38e-7, target 1e-6), but 57/100
CV candidates failed the historical 800-iteration / 1e-3 criteria. Therefore
the complete run is **FAIL**, not an accepted alpha-selection result. Peak
PyTorch allocation was 1,770,686,976 bytes. This is not a speedup measurement
against a matched CPU pipeline. A stricter 20000-iteration / 1e-6 CV run is not part of the local acceptance evidence;
these historical directories are retained as diagnostic artifacts only. They must
not be interpreted as an accepted real-data result or a speed benchmark.

Accepted real-cache dual-GPU run (2026-09, 6x RTX 3090 host, Torch 2.6.0+cu124): the same
99792x20125 effective operator, `--devices 0,1 --ngpu 2 --debias 0` with the strict CV
settings (`--cv-tol 1e-6 --cv-max-iter 20000`), returned **status PASS** with every CV
convergence record and the final refit accepted. Five folds were split across `cuda:0` and
`cuda:1` (61 and 40 solver records respectively), so this is genuine two-card CV, not a
single card with a second visible. Total fit time was 552.29 s. Note the caveats that ship
in that run's own `result.json`: the CPU-derived automatic alpha grid is inside the timing,
CUDA residency covers the solver stage rather than every pipeline operation, allocator peaks
are not total device memory, and the strict CV defaults differ from the historical 1e-3/800
criteria, so this is a correctness acceptance and **not** a matched CPU/GPU speed benchmark.

### Reproduce resident acceptance tests

Use a CUDA-enabled Python environment with this checkout as the working directory:

```bash
# Small numerical/device tests; CUDA skips do not certify a GPU pass.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python dev/test_gpu_twolevel_lasso.py

# Independent CPU/GPU full synthetic comparison, default CPU debias preserved.
CUDA_VISIBLE_DEVICES=0 python dev/validate_resident_lasso.py --synthetic --cpu-reference \
  --output tmp/resident-synthetic-NEW

# Full real-cache model-selection acceptance; explicitly isolate pure LASSO.
# DATA contains sm_prime.npz, ns_harm.npz and fm1d.npz.
# This is a real-data run, not a local synthetic acceptance; do not run without DATA.
CUDA_VISIBLE_DEVICES=0 python dev/validate_resident_lasso.py DATA --debias 0 \
  --cv-tol 1e-6 --cv-max-iter 20000 --output tmp/resident-real-NEW
```

### Reproduce the two-card hardware acceptance

```bash
# Full acceptance matrix (dense + rank-deficient, all five methods, resident dense/CSR/
# TwoLevel) plus 1-GPU and 2-GPU CV. Pins two devices and asserts the resident backend
# actually dispatched, so a silent CPU fallback fails the run instead of passing.
CUDA_VISIBLE_DEVICES=0,1 python dev/validate_gpu_backends.py --devices 0,1 \
  --json tmp/gpu_backend_validation_twogpu.json

# Direct two-card CV hardware test (requires >= 2 visible devices).
CUDA_VISIBLE_DEVICES=0,1 python -m unittest dev.test_multigpu_cuda_hw
```

On a host where the package is also installed under its distribution name (so both a flat
`core` and a `pheasy_gpu.core` namespace resolve), the unit tests that monkeypatch
`core.gpu_backend` will miss the module object the optimizer imports and report phantom
failures. Alias the namespaces before running them (`sys.modules["pheasy_gpu"] = core`,
`sys.modules["pheasy_gpu.core"] = core`) or import the package the same way the tests do.


Output directories must not already exist. The script validates all 100 CV
convergence records and the final refit after allowing the complete path to
finish. The default random-row split preserves the historical comparison but
is **not** independent configuration-level generalization evidence.
Use `--group-size` deliberately for a different, grouped validation design.
To enable the solver in an existing LASSO command, prefix it with
`PHEASY_GPU_LASSO_RESIDENT=1`; existing debias behavior is preserved unless
explicitly changed. Select a free allocated device using `CUDA_VISIBLE_DEVICES`
and logical `PHEASY_GPU_DEVICE=0`, rather than copying a physical device index
into both variables. No new remote deployment is implied by these examples.

## What is accelerated

| Method | CPU (scipy/sklearn) | GPU (torch/cuSOLVER) |
|---|---|---|
| OLS | `scipy.linalg.lstsq` (gelsd SVD) | `torch.linalg.svd` + rcond solve |
| RIDGE | grouped K-fold CV (closed form) / operator LSMR | dense grouped SVD CV; opt-in TwoLevel/operator augmented CGLS |
| LASSO | `sklearn.LassoCV` (coordinate descent) | Gram-based FISTA (GPU by default) |
| ALASSO | ridge pilot + `LassoCV` on scaled cols | ridge pilot + FISTA with per-column weights |
| RFE | `scipy.linalg.lstsq` per subset | dense: rank-checked QR + `gels` per subset, SVD fallback; outer CV/orchestration remains CPU |
| SM loading | `sm_prime @ NS` (sparse) on CPU | `torch.sparse.mm` on GPU (**holdout_eval only**) |
| TwoLevelSM | two sparse matvecs on CPU | `PHEASY_GPU_SM=1`: matvec split across GPUs; opt-in resident OLS/Ridge/LASSO/ALASSO use CUDA CGLS/FISTA without densifying |

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

**Dense RFE uses QR for the per-subset solves**, not the SVD. QR is
backward-stable for the full-rank subsets; a historical 3090 measurement reported
~50x faster than the FP64 SVD, but this is not a current benchmark. `qr_solve` factorizes R only (`mode="r"`, no Q materialization),
checks its diagonal (the same threshold as `_solve_qr`), falls back to the SVD
for rank-deficient or wide subsets, then runs the fast cuSOLVER `gels` solve;
CUDA `gels` silently returns NaN/inf on rank-deficient inputs, so the earlier
try/except fallback never fired. Verified `qr_solve` vs
`numpy.linalg.lstsq` to ~5e-15 (full rank).

**`RFE-OLS-TSQR` (alias `RFE-TSQR`) has a GPU dense subset-solve path.** Its dense
subset solves go through the same `qr_solve` path as RFE (the Q-less tall-skinny
blocked QR is a memory optimisation for CPU tall matrices; on the GPU the same
rank-checked QR + `gels` path is used). Its BIC/AIC stopping rule is opt-in via
`PHEASY_TSQR_CRITERION=bic`; the default is `cv`, which makes it select the same
support as RFE. The historical n=8 timing (~72 s, nnz=2092) is retained only as
non-current diagnostic evidence, not as a current performance benchmark.

## Usage

Activation (in priority order):

1. `Optimizer(..., use_gpu=True/False)` -- override applied during that
   instance's fit; the preceding backend mode is restored afterwards.
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

### Large matrices and multiple GPUs

After preparing the data and building the cluster space, null space, and
`sm_prime.npz`, run inside a scheduler allocation with six visible GPUs:

```bash
PHEASY_USE_GPU=1 PHEASY_GPU_SM=1 PHEASY_GPU_SM_NGPU=6 \
PHEASY_GPU_SM_DEVICES=0,1,2,3,4,5 PHEASY_TWOLEVEL_CACHE_T=0 \
PHEASY_SM_DTYPE=float64 PHEASY_OLS_TWOLEVEL=1 \
pheasy-gpu --dim 2 2 2 -w 3 --c2 7 --c3 4.5 -f --ndata 699 -l OLS --hdf5
```

Device numbers are relative to `CUDA_VISIBLE_DEVICES`. `PHEASY_USE_GPU`
controls dense solves; `PHEASY_GPU_SM` separately enables the two-level sparse
backend. The complete sparse matrix still resides in host RAM. GPUs store
row blocks and transpose blocks; this does not pool their memory into a
single dense allocation. Actual block budgets and int32 index limits are
checked before upload. Ordinary fitting reports a CPU fallback if a GPU
cannot be used; `dev/validate_large_fit.py` treats a fallback as a failed test.

`RFE-OLS-TSQR` can use `PHEASY_RFE_TWOLEVEL=1` to avoid constructing the dense
product. Its sparse/TwoLevel subset solves use **LSMR**, not a distributed QR;
dense subsets use GPU QR when the dense GPU gate passes. Dense TSQR retains an
O(p²) factor and is unsuitable when that factor and its workspace exceed RAM. Grouped RFE may require many complete iterative
solves, so first measure a representative configuration subset. `Optimizer.results`
records `execution_backend`, `backend_metadata`, and `postfit_backend` so callers
can distinguish GPU subset algebra from CPU RFE control and metrics.

OLS exposes iterative stopping diagnostics in `Optimizer.results["solver_info"]`.
LSQR/LSMR warn on iteration/condition limits. Check `converged` before accepting
the coefficients. `PHEASY_OLS_MAXITER`, `PHEASY_OLS_ATOL`, and
`PHEASY_OLS_BTOL` configure two-level OLS; RFE uses `PHEASY_LSQR_*`.
Exact two-level column norms use bounded row products; their temporary budget
defaults to 64 MiB (`PHEASY_COL_NORM_BLOCK_BYTES`).
`PHEASY_OLS_JACOBI=1` uses these norms to scale columns before LSMR and
returns coefficients in the original units. This can help when column scales
differ substantially. It preserves the least-squares objective (and the
original ridge penalty when enabled); for nonunique unregularized solutions,
column scaling can change which minimum-residual coefficient vector is chosen.
Explicitly requested preconditioning propagates preparation/solver errors
instead of silently repeating the solve without scaling.

For fractional-coordinate arrays whose atom order differs from SPOSCAR:

```bash
python tools/prepare_dataset.py SPOSCAR dataset_disps.npy dataset_forces.npy \
    --frac --align-reference
```

This applies the same geometrically verified permutation to coordinates and
forces, and writes `dataset_alignment.json`. The reference frame must describe
the same structure as SPOSCAR. Compact fc3 output allocates only primitive
representatives on its first atom axis; `--full_ifc` explicitly requests the
larger complete tensor.

### Dense solver controls

* `PHEASY_GPU_DEVICE` -- CUDA device index (default `0` / first visible
  device); read fresh on every call (no caching).
* `PHEASY_GPU_MEM_FRACTION` -- fraction of free VRAM a dense solve may occupy
  before it falls back to the CPU (default 0.8). The returned backend diagnostics
  identify whether a fallback occurred; explicit resident requests fail closed.
* `PHEASY_FISTA_RESTART_EVERY` -- how often (iterations) the FISTA adaptive-
  restart overshoot check syncs to the host (default 5). Higher = fewer syncs,
  marginally less-frequent restarts; the fixed point is unchanged.
* `PHEASY_GPU_LASSO` -- `0` routes LASSO/ALASSO to the sklearn coordinate-
  descent path (bit-identical to original pheasy); unset/`1` uses the GPU FISTA
  (default).
* `PHEASY_LASSO_1SE` -- `1` applies the one-standard-error rule to LASSO
  alpha selection on all backends (dense, iterative, and GPU).

## Historical measured performance (RTX 3090, c7 sensing matrix 25515x6588)

Historical `holdout_eval` n=8, one split, single free 3090 (CUDA_VISIBLE_DEVICES=6); these figures are not current hardware acceptance:

| step | CPU | GPU |
|---|---|---|
| SM load (25515x6588 @ 6588x3678) | ~1421 s | ~28 s |
| OLS (4536x3678, gelsd vs SVD) | ~600+ s | ~21 s |
| RIDGE (50-alpha grouped CV) | ~1700 s (n=24) | ~16 s |
| LASSO (20-alpha grouped CV) | ~939 s | ~40 s |
| ALASSO (20-alpha weighted CV) | ~940 s | ~35 s |
| RFE (step 0.05, 3-fold) | SVD per subset (hours) | ~99 s (QR; pre-rank-check) |

At the full n=24 scale the CPU baseline measured ~1804 s (OLS) and ~1715 s
(RIDGE) per fold on the shared box; the GPU SVD for 13608x3678 is ~28 s, so
the end-to-end holdout drops from hours to minutes.

> The RFE timing predates the R-only rank-check pass (re-run before quoting).
> LASSO/ALASSO were re-measured after the fixes: 38/41 s GPU FISTA vs 418/306 s
> sklearn CD on the c7 SM (n=8); see the recheck note in "Verified numerics".
>
> RIDGE switched from leave-one-out GCV to grouped K-fold CV (P46) -- the same
> grouped splits as LASSO/ALASSO/RFE. The ~1700 s / ~16 s figures above were
> measured with the old GCV (one SVD); grouped CV costs K SVDs per fit, so
> re-measure before quoting RIDGE timing.

Verified numerics (GPU vs CPU):
* `lstsq` vs `numpy.linalg.lstsq`: ~1e-15
* `ridge_solve` vs `sklearn.Ridge`: ~1e-15
* `GpuRidgeCV` grouped-CV == CPU grouped-CV closed form: ~1e-15 (both now
  grouped K-fold CV, not GCV)
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
* The dense SM (`n x p`) materialisation is inherent to the dense path;
  `TwoLevelSM` avoids it and optionally distributes sparse matvecs over GPUs.

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

* Legacy `PHEASY_GPU_SM=1` accelerates sparse matvecs while the solver iteration and NS multiplication remain on the CPU. The separate resident OLS/Ridge/LASSO/ALASSO flags keep their supported solver vectors and factors on CUDA.
* **GPU SM loading is wired into `holdout_eval.py` only.** The `pheasy-gpu`
  CLI (`run_pheasy.py`) still assembles `SM_prime @ NS` with scipy on the CPU;
  the "SM load 1421s -> 28s" row above applies to `holdout_eval`, not to the
  CLI. Once SM is assembled, its supported dense and resident fitting paths can use
  GPU acceleration (OLS/LASSO/ALASSO/RIDGE, and dense RFE subset solves); RFE
  orchestration/postprocessing and optional debias remain CPU.
* `torch.linalg.lstsq` on CUDA only exposes `driver="gels"`, so `lstsq()` here
  reimplements the SVD (gelsd) solve with `torch.linalg.svd` -- numerically
  equivalent to scipy, at a small constant-factor cost.
* FP64 throughput on a consumer 3090 is ~1/64 of FP32. The 10-60x dense-fit
  comparison is historical and workload-specific, not a current benchmark or a
  guarantee for grouped Ridge/RFE end-to-end pipelines.



## Data preparation gotcha: supercell atom order

pheasy's `create_supercell` orders supercell atoms as per-primitive-atom blocks
(all images of primitive atom 0, then all of atom 1, ...), each block in the
`ndindex(*dim[::-1])` translation order. ASE's `Atoms.repeat()` interleaves per
image. Displacement/force data prepared with ASE's ordering scrambles every
atom except the first (identical in both: the origin image).

Symptom: the on-site IFC fits (it only involves atom 0) but the fit residual is
~50% with per-config correlation ~0.8 instead of ~0.999. C60Mg2's box data is
aligned (corr 0.9997); a fresh Si test hit this (54% -> 0.13% after
regenerating the data in the pheasy order).

Verify any new dataset with the per-config residual check
(`SM[cfg rows] @ coef` vs the config's forces): corr < 0.99 is a red flag.


## Methodology lessons (hard-won, 2026-09)

* **Synthetic-data self-consistency is a blind spot.** `F = SM @ coef` is in
  the column space BY CONSTRUCTION, so lstsq always recovers it to ~1e-15 --
  it never tests the displacement/force READ-IN path (config order, atom
  order, units). A real misalignment shows as "on-site IFC fits but the
  residual is ~50% with per-config corr ~0.8" -- the supercell atom-order
  trap above. The post-fit corr check in run_pheasy.py flags it on the first
  fit (worst corr < 0.99 = warning; the check itself failing = UNVERIFIED).
* **Three-way controlled experiment (serial / per-config / chunked) is the
  only way to separate dispatch overhead from kernel cost.** Without it, the
  COO build's 13.4x per-config would have been misread as 1.0x wall-clock (the
  per-config joblib dispatch re-pickled CS_full per task and ate the win).
* **An unconverged Krylov solve amplifies ~1e-14 matvec perturbations to
  ~1e-3.** Truncated-iteration comparisons are meaningless; any ~1e-12-level
  verification must first let LSMR converge (or use P2 Jacobi,
  PHEASY_OLS_JACOBI=1).
* **Reference-solution certification bounds.** A KKT residual certifies to
  ~1e-9, but the coef error is only bounded to ~2.6e-4 -- do not quote
  tighter agreement than the certification warrants.
* **SIGTERM (exit 143) means someone ran kill -15, NOT GPU contention.** On
  this shared box, another user's job (caier/zls gmx mdrun etc.) starting on
  a card my process uses SIGTERMs it (device faults give Xid/segfaults
  instead). Verified with a fully-detached (setsid) test: busy card died 143,
  the free-card control finished exit 0. Use only free cards
  (PHEASY_GPU_SM_DEVICES + the free-VRAM filter in _pick_devices).
* **A 60 GB SM needs ~140 GB RSS to build the GPU blocks** (per-block column
  slices) or ~185 GB with the full-transpose path -- the kernel OOM killer
  (exit 137) strikes on the shared box unless the construction is memory-lean
  (see GpuSparseMV __init__).

## RFE operator preconditioning

`PHEASY_RFE_JACOBI=1` enables right column scaling for operator RFE subset
LSMR solves (default off). `PHEASY_OLS_JACOBI` only controls standalone OLS,
not RFE. RFE reuses its training-matrix column norms across subsets; these
are not exact per-fold norms. Coefficients are mapped back before ranking
and prediction, and ridge augmentation retains the original coefficient
penalty. No dense sensing matrix is formed. In rank-deficient problems,
right scaling can select a different nonunique solution; evaluate held-out
predictions as well as solver convergence. Small regression tests cover
scaling, masked rows/columns, ridge, and estimator integration. Real Mg2C60
RFE validation passed with this option: 13 LSMR sub-solves converged;
26,141/52,283 coefficients selected and fixed-split holdout RMSE improved
2.62% versus OLS. This is one split, not a general performance guarantee.

## Null-space construction profile: order-3 translational invariance (C60Mg2, 2026-09)

The `-c` null-space step was the largest remaining lever (~41 min on C60Mg2,
699-config system). cProfile (full run under profiler 56.6 min) splits it:

* **Order-3 translational invariance (TI): ~25 min (60%).** The pure-Python
  loop in `build_translational_invariance` does |asr reps| x |orbits| x
  orbit-size inner tests = 1192 x 2061 x ~32 = **77.6M `_diff_cluster`
  calls**, ~70% of the phase in Counter machinery (`_diff_cluster` cum ~1600 s
  of the 3398 s whole-run profile; 99.77% of inner tests are wasted).
* **Order-3 ASR sparse elimination: ~13.6 min (33%).** 32184 constraint rows
  over ns whose nnz grows 55k -> 4.5M.
* Order-2 all phases ~1.3 min; isotropy ~26 s.

**Fix (TI): inverted, hash-narrowed construction.** A rep L matches an image
I iff the rep atom MULTISET is contained in I with exactly one extra atom,
which is equivalent to L being one of the <= order sub-multisets of I. Each
image is narrowed by a hash to <= order candidates, and **every candidate is
then confirmed by the original `_diff_cluster`** -- matches are exactly the
old ones by construction (no false positives, no misses).

Hash keys are the canonical multiset form `tuple(sorted(...))` -- the same
semantics as `_diff_cluster`'s Counter and the orbit dedup key. This matters
beyond order 3: a `frozenset` key is injective only on 2-multisets (the
order-2 reps that back order 3); from order 4 up, lower reps like
(a,a,b)/(a,b,b) collide on {a,b} and an image like (a,a,b,b) would have its two
different 3-sub-multisets merged by a frozenset `seen` set, silently dropping
one rep. Multiset keys keep the fast path correct for every order >= 3.

Measured (order 3, C60Mg2): cons output bitwise-identical to the old builder
(max abs diff 0.0, full 1192-block comparison; nnz 153590 both), full order-3
TI ~7-10 s vs ~25 min, whole `-c` 41 min -> 15:44. Order-2 path untouched.

One intentional numeric difference vs the old builder: `np.nonzero` keeps only
numerically non-zero entries, while the old COO accumulation could keep
structural zeros. Currently irrelevant (eliminated/skipped counts and
max|C_old @ ns_new| match), but if a future pivot search iterates stored
entries instead of `toarray()`, the nnz difference would change pivot choice --
worth remembering before touching `_eliminate_row`.

**Validation: `max|C@ns|` is self-referential -- do not use it alone.** It
uses the NEW constraint matrix C; a fast path that silently dropped rows would
still pass at ~1e-15. The discriminating test is the old constraints against
the new null space: capture the OLD TI blocks once (isolated run, ~10 min,
`cons_old.pkl`), then check `max|C_old @ ns_new|` ~1e-15 plus `p`
(n_free count). Measured on C60Mg2 order 3: `max|C_old @ ns_new| = 1.7e-15`,
`p = 52283` (10252 + 42031) unchanged.

**Synthetic harmonic recovery is the strongest regression for the fit chain.**
`dev/synth_harmonic_regression.py` generates forces with an INDEPENDENT real-space
path (`F = -Phi @ u`, never `SM @ coef`, which is in the column space by
construction) from a symmetry-consistent Phi2 truncated inside the cutoff, fits
order-2 with pheasy, and asserts the recovered fc2 equals Phi to machine
precision (measured rel 1.2e-15 on Si 4x4x4, 40 configs). It locks SM
construction + indexing + force read-in + OLS + ASR-as-constraint + fc2 write
in one shot. Run it before touching any of that chain (and on C60Mg2 after the
c2=7.0 shell-band question is settled).

**Elimination (13.6 min) is format-independent and deferred.** Standalone
re-run with the real data: CSC 1008 s vs CSR 1028 s -- not a sparse
format-conversion problem (the profile's 900 s of `csc_tocsr` lives in the
old TI loop's per-match coo churn, removed by the fix). 24493 of the 32184
rows (76%) are linearly dependent after earlier eliminations yet each pays a
`row @ ns`; the 7691 real eliminations are rank-1 sparse updates on growing
nnz, i.e. sparse rank-revealing -- speeding it up means a rewrite (sparse QR /
SuiteSparseQR). Ceiling is ~17% of the whole 41+34+32 min pipeline and the
risk is asymmetric (it is the one place a wrong change silently invalidates
all produced fc2/fc3), so it stays a known bottleneck.
