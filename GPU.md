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
the requested `max_iter` and `tol`. CV defaults to those same values but can be
overridden independently with `PHEASY_CV_MAX_ITER` / `PHEASY_CV_TOL`. OLS LSMR
iteration limit follows `Optimizer(max_iter=...)` (`PHEASY_OLS_MAXITER` still
overrides); its `atol/btol` remain separate `PHEASY_OLS_ATOL/BTOL` knobs because
an LSMR residual tolerance is a different quantity from FISTA's relative KKT
`tol`. A small update or a low training error alone is not a convergence certificate.

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

As of 2026-09-11 the full GPU acceptance is re-verified on a 3x RTX 3090 host
(Torch 2.6.0+cu124, physical GPUs 1/4/5): `dev/validate_gpu_backends.py` passes
24/24 checks (all six methods on dense, CSR and two-level inputs, 1-GPU and
3-GPU, well-conditioned and rank-deficient; GPU matvec/adjoint match CPU to
~1e-16, every method records `gpu_spmv_calls > 0`), `test_multigpu_cuda_hw.py`
passes, `dev/test_cli_harmonic_chain.py --gpu` recovers the true Si fc2 to
~4e-16 relative error with a real `gpu_backend.lstsq` call on `cuda:0`, and the
`dev/` unittest suite runs green (17 files, 124 tests collected; the 3 cases that
need a live device report `CUDA unavailable` on a CPU-only host). Re-record this
count on the GPU host whenever `dev/` changes: the suite does not run in CI, so
a stale count is worse than no count.
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
fit a complete factor replica. Note the asymmetry: an explicit
`PHEASY_GPU_DEVICES` list is a HARD requirement — the resident path does not
filter it by free memory or by queryability, so one dead or busy entry aborts
the whole fit (`Cannot query resident CUDA memory budget`), while the GPU-SM
path would simply have skipped that card through `_multi_gpu_devices`.
Prefer `PHEASY_GPU_NGPU`, which starts from `PHEASY_GPU_DEVICE` and picks
other visible cards, when the device list is not known to be uniformly healthy.
Hardware acceptance must be reported separately
from CPU-emulated scheduler tests.

### GPU acceleration status (current)

Dense OLS/Ridge and opt-in dense streamed TSQR use CUDA float64 kernels.

`PHEASY_GPU_RFE_RESIDENT=1` retains dense matrices or sparse/TwoLevel factors and targets across RFE subset solves and CV predictions. It supports OLS and positive-alpha Ridge subsets and requires `n_jobs=1` (the existing Jacobi option applies only to operator inputs; `jacobi_applied` reports actual use), with a conservative workspace budget checked before upload. Metadata exposes `resident_subset_inputs` and `resident_fallback_reason`. Dense fits cache CV row indices and the current support matrix on CUDA; sparse/TwoLevel fits reuse factors through vector scatter/gather views without constructing subset matrices. Their solver is CGLS (`gpu_rfe_resident_iterative`), with per-solve convergence diagnostics and `PHEASY_LSQR_ATOL/BTOL/MAXITER` controls; nonconvergence aborts before elimination. `resident_row_index_uploads`, `resident_column_index_uploads`, and `resident_subset_builds` count this work. The pre-upload budget includes the support matrix and index caches (`resident_index_cache_budget_bytes`). An allocation failure during recursion currently aborts the fit; only initial input-upload failures use the fallback path. Training-fold coefficients and validation residuals stay on CUDA; only per-fold RMSE scalars are downloaded. BIC/AIC residual sums are reduced on CUDA and only RSS scalars are downloaded. With GPU ranking enabled, full-fit coefficients stay on CUDA until a tied/nonfinite importance requires NumPy fallback or the final public coefficient vector is produced. Verbose logs download only the nonzero count. With GPU ranking disabled, coefficients are downloaded each round for CPU importance calculation. Column norms are computed on CPU, uploaded once, then reused for CUDA importance and operator Jacobi scaling. Fold aggregation, support updates, patience and selection remain CPU. Metadata exposes `cv_fold_scoring="gpu"`. This is input residency, not a fully resident RFE loop. The existing validator supports `--benchmark-ridge-rfe --benchmark-rows 6000 --benchmark-columns 256` to compare CPU, legacy GPU and resident-input GPU fits. Add `--benchmark-input csr|twolevel` for sparse fixtures (10% density; TwoLevel uses identity NS). On the local RTX 4060, a 6000×256 CSR fixture gave CPU 0.3193 s versus resident GPU 0.5210 s median over three fits, with relative prediction difference 4.06e-10 (`tmp/csr_bench130.json`). A smaller 1200×96 TwoLevel fixture was also slower on GPU. These synthetic results do not establish a sparse speedup; keep residency opt-in and measure the actual workload.

A matched 6000x256 sweep on a confirmed-idle RTX 3090 (Torch 2.6.0+cu124; all six devices
verified at 0% utilization before and after; median of three warm fits) gives the clearest
current picture:

| input | CPU | GPU (non-resident) | GPU resident | required env (reconstructed from the code paths -- the original run did not record it) |
| --- | --- | --- | --- | --- |
| dense | 0.5322 s | 0.3381 s | **0.1621 s** | CPU: none; non-resident: `PHEASY_USE_GPU=1`; resident: `PHEASY_GPU_RFE_RESIDENT=1` |
| csr, 10% density | 0.5534 s | **0.2829 s** | 0.3392 s | same as dense |
| TwoLevel, identity NS | 0.2097 s | 0.2328 s | 0.5264 s | CPU: none; non-resident: `PHEASY_USE_GPU=1`; **resident: `PHEASY_GPU_OLS_RESIDENT=1` AND `PHEASY_OLS_JACOBI=0`** |
| TwoLevel, sparse-mixed NS | 0.4458 s | 0.4475 s | 0.7916 s | same as TwoLevel, identity NS |

**Read the env column, not just the timings.** As of `2c71bac` the TwoLevel resident column IS
reachable under the default configuration: Jacobi is applied ON the resident operator (the column
norms become `GpuTwoLevelOperator.scale`, so CGLS solves the same scaled system the CPU path
builds and only `x = z / scale` is added), so Jacobi no longer disqualifies the path. Only a
ridge request (`PHEASY_OLS_RIDGE>0`) still falls back, and `fallback_reason` names the ridge
option. The measured row above was taken with `PHEASY_OLS_JACOBI=0`, which is now a recorded
detail of that run rather than a precondition. The default behaviour is pinned by
`dev/test_gpu_operator_ridge.py::test_matrix_free_default_reaches_resident_ols`.

That also invalidates the old reading of the harness: `dev/validate_large_fit.py` sets
`PHEASY_OLS_JACOBI='1'`, which used to force **CPU LSMR**. It does not do so automatically any
more, so a green light from it is evidence for neither path until it is re-measured.

Dense residency is a real win on this hardware (3.3x vs CPU, and 2.1x vs the non-resident GPU
path, so residency itself -- not merely "using the GPU" -- is what pays). Sparse CSR is ~2x faster
than CPU on the GPU, but residency is ~20% *slower* than the simpler non-resident GPU path: CGLS
iteration and per-solve setup cost more than the subset transfers it avoids at this size. TwoLevel
is the worst case because each `matvec` is two sparse products, so CGLS needs far more work per
solved subset; resident TwoLevel is slower than CPU here and should stay opportunistic. Do not
extrapolate any of these figures to larger problems in either direction -- re-measure the actual
workload. Prediction agreement with CPU stayed at 1e-15 (dense/CSR) and ~4e-10 (CGLS paths).
`PHEASY_GPU_OLS_RESIDENT=1` enables CUDA-resident CGLS for `TwoLevelSM` OLS.

CGLS stopping tests use `normar <= atol * norma * normr` or
`normr <= btol * normb + atol * norma * normx`. `norma` is a deterministic
10-step matrix-free spectral estimate of the effective operator, not SciPy's
iterative norm estimate or a certified upper bound. Identical CPU/GPU iteration
counts are not guaranteed; CGLS does not implement SciPy's independent `conlim`.
The requested `atol/btol` are raised to the floor the working precision can
reach -- `eps * ||A|| / sqrt(alpha)` for a ridge-augmented operator, capped at
1e-3 -- and the applied values are reported as `atol_effective`/`btol_effective`
alongside `tolerance_floor`, so a manifest never has to hide a loosened test.
`fit_manifest.json` serializes `Optimizer.results` (so the whole solver
certificate rides along, including the probe fields below) and now records EVERY
`PHEASY_*` setting, not just the GPU/OLS/LSQR/CV prefix whitelist it used to
carry: a manifest that omits `PHEASY_CGLS_PROBE_START` /
`PHEASY_FISTA_AUTO_FLOOR` / `PHEASY_HOST_FACTOR_CACHE` cannot explain why a fit
stopped or how much host memory it intended to use.
Before returning, the solver recomputes `b - A x` and its adjoint product on
CUDA. A verdict from the cheap recurrence residual is only accepted if that
recomputation agrees; otherwise the recurrence is replaced by the honest
residual and the iteration continues, which is recorded as
`honest_residual_check=recurrence_replaced`. Diagnostics expose the norm
estimate, criterion and residual certificate; a recomputation that cannot
confirm the recurrence reports `true_residual_check_failed`, and an iterate
whose honest residual runs past `10 * ||b||` reports `residual_growth`.
`converged=True` additionally requires `normr <= normb`, because `x = 0`
already achieves `||b||` and the `||x||`-scaled branch above is unbounded in
`||x||`. Small residuals do not
certify small coefficient errors on ill-conditioned systems. Existing hardware
timings predate this repair unless explicitly noted. Synthetic scaling and CPU
LSMR comparisons do not replace the requested Si real-data acceptance.

Subset and augmented operators cache their own norm estimate, never the base
operator estimate. This avoids repeated queries on the same object; a newly
constructed RFE subset or new Ridge alpha still incurs an independent estimate.
No across-alpha speedup is claimed. Replica scale assignment explicitly clears
the replica cache. Final verification alone accepting a fit is reported as
`converged_on_true_residual`. The follow-up cache regressions and actual two-card
CV test passed on RTX 3090 (18 tests); this does not establish a throughput gain.
A one-status-transfer CGLS experiment passed correctness tests but was reverted:
on RTX 3090 it was 15.7–16.8% slower than 574f64a in eight synthetic inner-solver
cases (dense and 8% CSR, 1024x128 and 8192x1024, fixed/converged runs; two warmups
and seven alternating timed pairs). Fewer synchronizations did not yield lower
wall time. The original per-iteration checks remain; ten-iteration batching is
not implemented. Dense/small CSR paired coefficients were bitwise identical;
medium CSR varied within both versions, so cross-version differences cannot be
attributed solely to the change. Raw timings and measured sources are retained
in `tmp/cgls_compare_574f64a/`. These are not full-material fit benchmarks.
A separate periodic-only check prototype was also rejected: a NumPy logic
counterexample with an ill-conditioned noisy system stopped at iteration 54
with per-step checks but at 160 with checks every ten steps. Both passed the
final residual criterion. The normal-residual stopping predicate is not
monotone, so missing a crossing does not bound the extra work to nine steps.
This is logic-level evidence, not a CUDA timing. Any future batching must
preserve the first accepted iterate on device or explicitly document changed
stopping semantics; fewer host checks alone are insufficient.
Subsequent isolated RTX 3090 measurements of this periodic-only prototype showed
3.05–3.23x speedups in eight synthetic timing cases (two warmups and seven
alternating pairs), but confirmed the semantic problem: the noisy ill-conditioned
CUDA case stopped at 77 versus 160 iterations and had relative coefficient L2
difference 13.2661. Ordinary converged timing cases differed by about 1e-7 with
extra iterations. This experiment is not integrated or enabled by a product
flag. Results are retained in `tmp/cgls_batched_prototype/cuda_results.json`;
these timings are not a claim of equivalent-estimator or material-fit speedup.

The precision floor and the honest re-check are NOT the rejected periodic-check
prototype: the cheap test is still evaluated on every iteration and still stops
at the first crossing, so no crossing is skipped, and the first accepted iterate
is still the one delivered. What changed is what happens AT that crossing -- it
is confirmed on the recomputed residual, and a disagreement restarts the
recurrence from the honest residual instead of returning
`true_residual_check_failed`. The extra cost is two matvecs per accepted stop,
plus the same pair every `PHEASY_CGLS_VERIFY_EVERY` (default 50) iterations
once the recurrence has been caught lying. Measured on the c7 operator
(25515x3678, float32 factors, one RTX 3090) through `_ridge_solve`:

- without `--std`, pristine code at `PHEASY_LSQR_ATOL=1e-8` stopped with
  `true_residual_check_failed` at alpha=1e-2/1e-4/1e-6 (itn 104/569/907) while
  the delivered `normr` was already 2.973078 / 0.837251 / 0.168144, i.e.
  identical to the float64 solve -- this is the shipped symptom, reproduced;
- with `--std`, pristine code ran the whole grid into
  `invalid_search_direction` with `normr` 2.8e17-1.2e19 and `normx` inf, because
  `_scale_columns` -> `_scale_operator` declares dtype float64 and
  `_array_precision` read that wrapper instead of the float32 factors, so the
  1e-8 request survived;
- with the repair, both configurations converge in 3-516 iterations, `normr`
  matches the float64 run digit for digit, and the relative coefficient
  difference against float64 is 1.7e-7 to 2.7e-5.

The recurrence residual was measured optimistic against the recomputed one by 8x
at alpha=1e-2, 48x at 1e-4 and 370x at 1e-6, and the true achievable relative
normal residual was 9.5e-7 / 7.1e-6 / 5.1e-5 -- 2.5x to 7x below the
`eps * ||A|| / sqrt(alpha)` floor, which is why that floor is used as the
acceptance bound rather than a tighter fit to the measurements. One shipped c7
CV sweep reported 17 `true_residual_check_failed` and 4 `iteration_limit`
entries; the coefficients behind them were correct. Artifacts, logs and the A/B
driver: `tmp/cert_ab_20260915/`.

Full C60Mg2 column-norm audit (699 displaced configurations, 496 atoms, cutoffs
7.0/4.5 Angstrom): the unscaled cached effective operator has shape
1,040,112 x 52,283. All rows were used. Float64 column norms range from
0.0356650027857755 to 3.7195257256162937, a max/min span of 104.2906332563;
zero/nonfinite counts are both zero. The diagonal of A.T A therefore has a span
of about 1.0877e4, not 104.29. This is neither a condition-number estimate nor a
Jacobi speedup measurement; it does not establish the previously quoted 12%
iteration saving on full data. The 60.49 GB sparse prime factor exceeds one
24 GiB RTX 3090, so this audit used the existing bounded CPU column-norm method
on the GPU host (1846.85 seconds for norms, 1985.31 seconds including loading).
It is not a CUDA timing. Detailed local evidence is retained in
`tmp/full_c60mg2_norms_summary_20260910.json` and
`tmp/full_c60mg2_norms_20260910.jsonl`. The separate 99,792 x 20,125 cache was
not used for this full-data measurement.

Full-data FP32 SpMV capacity was subsequently verified on three RTX 3090s
(physical devices 1, 4, 5). All six resident CSR blocks held float32 values;
peak allocated memory was 17.46/19.86/19.11 GiB. Against the original FP64
prime and NS before rounding, one seeded effective-operator forward/adjoint
probe measured relative errors 1.28248e-7 / 3.58492e-7. Evidence is retained in
`tmp/full_fp32_3gpu/capacity_result.json`. This establishes capacity and this
operator check only, not convergence, coefficient accuracy, or a speedup.
The current path is mixed precision: FP32 GPU sparse products with host-side
NS and solver work; it is not end-to-end FP32 fitting.

A full-row three-GPU bounded six-method diagnostic was completed in
`tmp/full_fp32_3gpu/allmethods_run_1789064024/`. OLS, RIDGE, LASSO, ALASSO,
RFE, and RFE-OLS-TSQR all returned coefficients and original-FP64 residual
certificates with 18/84/169/186/116/116 shared SpMV calls respectively;
all were deliberately marked `fit_accepted=false` because the pilot capped
inner solves at eight iterations. OLS, RIDGE, ALASSO, RFE, and RFE-OLS-TSQR
showed recorded nonconvergence; LASSO reached its bounded FISTA KKT flag but
was not accepted as a converged material fit.

The paired full-data OLS Jacobi diagnostic is in
`tmp/full_fp32_3gpu/jacobi_run_1789065172/`. Both arms used 200 LSMR
iterations and 401 GPU calls. Jacobi-off took 13.6034 s and had original-FP64
RMSE 0.0733105, relative residual 0.0812069, normal residual 2.95262;
Jacobi-on took 13.7883 s and had RMSE 0.0410126, relative residual 0.0454301,
normal residual 18.5756. Neither arm converged (`istop=7`), so this is a
traceable bounded numerical comparison, not a claimed speedup or accepted fit.
The coefficient/residual tradeoff and approximately 1.36% slower wall time
mean no semantics-preserving Jacobi benefit is established by this run.

When both arms converge, Jacobi column-scaling is semantics-preserving: a
regression test (`dev/test_jacobi_semantics.py`) verifies that
`PHEASY_OLS_JACOBI=0/1` recover the same dense-SVD solution to 1e-6 on a small
two-level problem and both report `fit_accepted=true`. The full-data bounded
run above did not converge on either arm, so it establishes the traceable
comparison, not a production speedup.
GPU-backed TwoLevelSM row slices now use a view of the parent upload, avoiding
a second factor allocation per CV fold. Each view still multiplies the full
parent operator; this trades extra SpMV work for bounded device memory. CPU
row slices retain their physical sparse slicing. Repeated selected rows
accumulate in the adjoint.

`PHEASY_GPU_MODE=auto|cpu|required` is the unified dispatch contract. `auto` uses CUDA when available and preserves the legacy CPU fallback for compatibility; `cpu` disables GPU; and `required` is the production default: every supported main solve (OLS/RIDGE/LASSO/ALASSO/RFE and the resident TSQR) runs on the GPU by default, and any GPU runtime failure raises instead of silently returning a CPU result. Concretely, under `required` the GPU-SM matvec/adjoint and the RIDGE-CV fold handlers raise even when `PHEASY_GPU_FALLBACK=1`, and an over-budget resident LASSO is refused unless `PHEASY_GPU_SM=1` actually supplies the substitute GPU path (`PHEASY_GPU_RESIDENT_FALLBACK=1` alone is not enough). Legacy `PHEASY_USE_GPU=1/0` maps to `required/cpu`. A per-instance `use_gpu=False` is an explicit CPU choice that still works under `required` (e.g. a CPU baseline in the validator).

`PHEASY_GPU_FALLBACK=0` (now the default) makes an enabled GPU sparse-matvec path fail closed on any forward/adjoint CUDA error, and a GPU RIDGE-CV fold OOM raises instead of running a CPU SVD; `PHEASY_GPU_FALLBACK=1` restores the legacy CPU continuation for compatibility. Production runs should use `PHEASY_GPU_MODE=required PHEASY_GPU_FALLBACK=0` and inspect `gpu_call_delta`/backend metadata.

`PHEASY_GPU_LASSO_RESIDENT=1` enables CUDA-resident LASSO and ALASSO: adaptive pilot, weights, weighted FISTA, CV, refit, and KKT run on CUDA.
`PHEASY_GPU_TSQR=1` enables bounded binary-tree TSQR for oversized dense tall full-rank systems; CPU TSQR remains the default. `PHEASY_GPU_RIDGE_RESIDENT=1` enables augmented GPU CGLS for TwoLevel/operator Ridge, with CPU metrics and fallback. Dense RFE subset solves/predictions use CUDA when memory allows, while RFE orchestration, support selection, and postprocessing remain CPU. `PHEASY_GPU_RFE_RANKING=1` opts into CUDA importance sorting; with resident RFE it also computes importance from CUDA coefficients (`gpu_importance_rounds`); otherwise importance calculation remains CPU. Column norms are prepared on CPU and support updates remain CPU. Tied/nonfinite importance falls back to NumPy to preserve its exact ordering. Metadata reports `gpu_ranking_rounds`. This is not a resident RFE loop or a demonstrated speedup. For public OLS, the same `PHEASY_GPU_TSQR=1` flag also enables sparse/TwoLevel streamed TSQR, with `PHEASY_TSQR_BLOCK_ROWS` (default 40000, at least the column count). CPU code assembles and expands each bounded row block; CUDA performs QR, tree reduction and triangular solve. Memory/rank failures fall back to matrix-free LSQR/LSMR with `fallback_reason`; operator ridge/Jacobi options retain their existing path. RFE sparse/TwoLevel residency uses the separate `PHEASY_GPU_RFE_RESIDENT=1` CGLS path, not streamed TSQR. TSQR still requires an O(n_columns^2) dense R factor and workspace.

**Scope:** file I/O, CV split construction and final host metrics remain on CPU.
The resident solver performs its own standardization on GPU and, when
`alpha_auto=True`, derives its own KKT-threshold alpha grid on the retained
operator (`op.rmatvec`) for both LASSO and ALASSO, matching
`derive_alpha_grid(standardize=True)` to machine precision; the earlier CLI
alpha-grid preprocessing is therefore redundant for the resident path. Exact
TwoLevelSM column norms (`_col_norms` + `_scale_columns`) go to the GPU whenever a
resident operator is available, and the operator that computes them is CACHED on
the host operator, so the norm pass and the solve that follows share one upload
(`TwoLevelSM._col_norms_on_gpu`; measured at NCONF=24, 36864x69487: the CPU pass
was 125.5 s of RIDGE's 154.4 s and 197.9 s of OLS's 219.9 s -- 81% and 90% -- while
the iterations cost 4.9 s and 16.2 s; at NCONF=296 the same pass is one 456.9 s
upload plus a 153.7 s kernel). Only dense/sklearn-style standardization on
non-TwoLevel operators stays a CPU BLAS step. A real-data benchmark
(`dev/benchmark_phases.py`, 99792 x 20125 c2.6 operator) measures column norms
at ~2.6s as the only host-side O(nnz) pass, with the rmatvec ~0.02s, metrics
~0.02s and RFE ranking ~0.002s; those three are too cheap for a GPU round-trip,
so metrics/RFE stay host-side by design. The resident GPU column norms run in
bounded column blocks whose size is ADAPTIVE (`_norm_block_budget`): a quarter of
free VRAM, capped at 8 GiB and never below a 512 MiB floor
(`PHEASY_GPU_NORM_WORKSPACE_MB` pins it instead); the earlier 64-column cap made
them ~3.5x slower than CPU.
The optional LASSO OLS-debias stage is reported separately as `debias_backend`
and `postfit_backend`. Dense and densified-sparse debias solve the support
least-squares on the GPU (`debias_backend="gpu_dense_lstsq"` via `gb.lstsq`);
the resident TwoLevel path solves it with GPU CGLS on the retained operator
(`debias_backend="gpu_cgls"`); `PHEASY_GPU_DEBIAS=0` falls back to CPU LSQR
(`debias_backend="cpu_lsmr"`). Setting `PHEASY_LASSO_DEBIAS=0` isolates pure
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

**Run the suite with a CLEAN PHEASY environment.** Measured on the production tree
(2026-09-17): with only `CUDA_VISIBLE_DEVICES`, `PHEASY_N_JOBS` and `PYTHONPATH`
exported, the suite is **172 tests, OK (1 skip)**. Exporting `PHEASY_USE_GPU=1` (or
`PHEASY_GPU_MODE=required`) for the suite run flips SIX tests into failures --
three that assert a silent CPU FALLBACK and three RFE mock tests that count
`qr_solve`/`_solve_subset` calls on the pre-residency path -- because required mode
is fail-closed by design and routes those solves through the resident operator.
Those six are a harness-environment artifact, not code flakiness and not a
regression: they pass in the clean environment, and the failure set no longer
rotates between runs. One genuine staleness was fixed at the same time:
`test_optimizer_standardization_uses_resident_math_and_physical_coefficients`
asserted the soft-threshold result under a one-point alpha grid, which the LASSO
edge-relaxed valve (P46) now overrides; the test pins
`PHEASY_LASSO_EDGE_RELAXED=0` and documents why.

**Current measurement (2026-09-25, after the P50/P50b/P50c view fixes).** Same clean
environment, one process per module, every `dev/test_*.py` in the tree:

| mode | modules | ran | failures | errors | skipped |
| --- | --- | --- | --- | --- | --- |
| real CUDA (RTX 3090) | 23 | **206** | 0 | 0 | 0 |
| `CUDA_VISIBLE_DEVICES=` (CPU) | 23 | **206** | 1 | 1 | 63 |

**Real CUDA is fully green (2026-09-25).** The 6F+1E reported in the P50 recert
were all in the RFE / operator-ridge area and were stale tests, not code bugs:
they predated `[PATCH rfe-final-tsqr]` (`PHEASY_RFE_FINAL_TSQR`, default **on**),
which solves the FINAL selected support exactly (Gram normal equations, or
streamed Q-less TSQR) and so changes four things the tests had pinned:
(1) `min_features == n_features` no longer runs an elimination round, so
`test_rfe_jacobi_reaches_subset_solver` never reached the iterative subset solver
(the fix is `min_features=12`, which exercises 24 real subset solves);
(2) the refit appends a diagnostics entry with a different schema
(`solver="Gram-Cholesky"/"TSQR"`, no `itn`/`normar`), so the CGLS schema checks in
`test_gpu_rfe` now select the `solver == "GPU CGLS"` entries;
(3) the refit densifies bounded ROW BLOCKS of the operator, so
`test_public_twolevel_grouped_cv_and_ic_without_densification` now asserts "only
`_rfe_final_refit_exact` may densify, and only in `block_rows`-sized row blocks"
(call-site checked) instead of "nothing may densify", and compares the delivered
coefficients against exact OLS on the SAME support because the CPU reference is
solved with `PHEASY_RFE_RIDGE_ALPHA=.2` (a ridge solve) while the delivered vector
is unbiased OLS;
(4) the final coefficients come from the host-side refit, so
`test_resident_ranking_downloads_coefficients_only_for_final_output` now pins
"ranking never downloads per round" (`count <= 1` and `< gpu_ranking_rounds`)
rather than an exact `1`.
`test_gpu_operator_ridge::test_extra_workspace_rejected_before_any_upload` was
separately flaky (passed alone, failed in the module) because its budget came from
`_device_available_bytes`, i.e. from however much VRAM earlier tests left
allocated; the test now pins that function, making the pre-flight verdict a pure
function of the matrix. The 11 view/transpose tests
(`dev/test_resident_row_view_transpose.py`) pass in both modes.

In `CUDA_VISIBLE_DEVICES=` mode two CUDA-requiring tests fail because they have no
skip guard and assert CUDA-runtime behaviour:
`test_gpu_operator_ridge::test_resident_ols_honors_limits_and_ridge_option` sees
the fallback reason
`"GPU mode is required but CUDA is unavailable"` instead of
`"does not implement the ridge option"`, and
`test_optimizer_large_regressions::test_gpu_construction_failure_does_not_cache_host_transpose`
raises `RuntimeError: CUDA unavailable`. Both pass on real CUDA; they are a
harness-environment artifact, not code flakiness.

Two modules report no unittest summary by design: `test_cli_harmonic_chain` (a CLI
driver, exit 0) and `test_fit_methods` (needs `sm_dense.npy` in the cwd -- a data
fixture that lives in the material directories, so it exits 1 with
`FileNotFoundError` in a bare checkout, in BOTH modes and before these fixes as
well). `PYTHONPATH=.` is required: `python dev/test_x.py` puts `dev/` on
`sys.path`, not the checkout root, and the older modules rely on the root being
importable. (The 2026-09-17 baseline of 204 ran / 0F on the then-current file is
kept in git history; the tree has since gained the ARDR work above.)

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

### Several fits at once: `fit_scripts/fit_3090_parallel.sh`

The measured answer to "how do I use the other cards" is one fit PER CARD, not one
fit split across cards (see "Sharding is a memory tool" above: RIDGE cv=5 took
606.8 s on one card and 713.5 s on three, and each resident fit needs ~17-19 GiB of
a 24 GiB card). `fit_scripts/fit_3090_parallel.sh` is the template for that (the fit
wrappers live in `fit_scripts/`; the shared driver `pheasy_fit.sh` stays at the
repository root because `scan_methods.sh` and both wrappers reference it by name --
see `fit_scripts/README.md`):

```bash
# in a material directory (POSCAR / SPOSCAR / disp_matrix.pkl / force_matrix.pkl)
bash fit_scripts/fit_3090_parallel.sh FIT_METHODS="RIDGE OLS LASSO ALASSO" NDATA=296 NCPU=8 \
     DEVICES="0 1 2 3" MIN_FREE_MIB=19000 OUTROOT=$PWD/parallel_fits
```

It assigns each method to a card that has `MIN_FREE_MIB` free (waiting instead of
overloading one), gives every method its own `$OUTROOT/<method>/` with its own
`fc*.hdf5` / `fit_manifest.json`, and prints a summary table read straight from
the manifests (accepted, backend, `stop_reason`, rmse, and the CGLS probe fields).
Keys it does not recognise are forwarded to `pheasy_fit.sh`; `PHEASY_SRC=<tree>`
makes it run that source tree instead of the installed `pheasy-gpu` (through a
one-line shim plus a `pheasy_gpu` link on `PYTHONPATH`).

**Inputs are copied, never symlinked -- and that is not a style preference.** The
fit stage REWRITES `SPOSCAR` and `phi.npz` in its working directory (see
`_content_sig` in `run_pheasy.py`: "-s rewrites SPOSCAR"), so symlinking them into
a run directory writes THROUGH into the material directory. An earlier version of
this template did exactly that and overwrote a user material`s `SPOSCAR` (recovered
byte-exactly from `SPOSCAR.orig` against the sha256 prefix recorded in
`cs.pkl.meta.json`). The rule is now size-based: everything under 8 MiB is copied,
only the GB-scale read-only caches (`sm_prime.npz`, `ns_*.npz`, the displacement and
force matrices) are linked, and a stale symlink left by an earlier run is removed
before the rule is applied.

Verified end to end against the real pipeline (Mg8C120 material, NDATA=24,
FIT_METHODS="OLS RIDGE", two cards): the scheduler gave each method its own card,
the OLS method finished in 211 s accepted (gpu_resident_iterative,
stop_reason=converged, rmse 8.768e-05), and its fit_manifest.json carried the whole
CGLS certificate -- probe_start 8000, probe_count, stall_floor, stall_iteration,
criterion_value, best_criterion -- plus 9 PHEASY_* environment entries. The hashes
of SPOSCAR, POSCAR and phi.npz in the material directory were identical before and
after, which is the regression guard for the copy-dont-link rule above.

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

RIDGE exposes the certificate of the solve whose coefficients are returned, in
`Optimizer.results["regularized_solver_info"]`, for BOTH of its iterative
branches. The resident branch writes `backend="gpu_resident_iterative"`; the
matrix-free CPU fallback writes `backend="cpu_lsmr_ridge"` and its LSMR `istop`.
Both carry the `alpha` they solved, because the L-curve walk leaves one
certificate per alpha on the same operator object and only the last one is still
readable from it -- reporting that one described a fit that had been discarded.
The CPU fallback used to discard its own `_iterative_solver_info(res, "LSMR")`
return value entirely, so a RIDGE solve that hit its iteration cap reported
`fit_accepted=True` on CPU while the identical solve raised on GPU; it now
participates in the acceptance gate, and it passes the operator to `_lsmr_tol` so
the data-precision floor applies there too (it previously omitted `A` and asked
float32-factored operators for an unreachable 1e-8).

`_ridge_solve` honours `x0` on the resident branch as well. The CV folds (one
warm start per fold) and the L-curve walk (large alpha to small) both compute and
pass a warm start, but the resident branch had no `x0` parameter and silently
dropped it, so the warm start the CPU branch documents existed only on CPU. It is
validated before use (right length, all finite) and dropped with a warning
otherwise, since a warm start is an optimization rather than a contract. Measured
on the c7 operator with `--std`, walking 1e2/1/1e-2/1e-4/1e-6: iterations
3/14/56/61/55 cold versus 3/14/55/49/17 warm, i.e. 189 -> 138 total and 3.2x
fewer at the smallest alpha, with `normr` identical and `normx` agreeing to
1e-6 relative.

**The float32 CV does not change model selection (measured).** Running the same
RIDGE fit twice on the MgC slice (NCONF=24, 36864x69487, CV=5, identical alpha grid
and y), once with the resident float32 CGLS and once with the old float64 CPU LSMR
(`PHEASY_GPU_RIDGE_RESIDENT=0`), the CV selected the SAME alpha (7.196857e-04) and
the fold-MSE path leading to it was identical value for value
(0.740712 0.763326 0.764898 0.764351 0.763039 0.676107 ...), so the precision of the
CV solver is not a model-selection risk here and no precision knob is needed to
guard it. What the precision does change is the FINAL coefficients: the float32
resident solve returned rmse 8.930536e-03 / r2 0.999895362, the float64 CPU LSMR
2.711818e-03 / r2 0.999990352 -- 3.3x lower rmse for 918.6 s against 57.4 s (16x).
A reader who needs the lower-residual coefficient vector should therefore request
the float64 path for the final solve, not for the CV.

The same floor now applies on the two remaining iterative entry points, both of
which used to omit the matrix:

* `_solve_sparse_lsqr` passed no matrix to `_lsmr_tol`, so `_array_precision` fell
  back to `PHEASY_SM_DTYPE` and the AMBIENT variable decided the tolerance of the
  matrix actually in hand.  With `PHEASY_SM_DTYPE=float32` a float64 sparse solve
  inherited the float32 floor and stopped ~190 iterations early for ~100x the
  coefficient error (measured on a 3000x400 float32/f64 pair: itn 510 / rel err
  1.1e-5 correct versus itn 322 / rel err 1.1e-3), while a float32 matrix got the
  floor only by coincidence.
* resident `iterative_lstsq` carries no penalty, so there is no `sigma_min` bound
  and no condition number to build a floor from.  It now falls back to the same
  one-decade-above-eps rule `_lsmr_tol` uses (`_CGLS_UNPENALIZED_COND`), so the
  CGLS and LSMR paths agree on what the stored precision can reach;
  `dev/test_cgls_tolerance_floor.py` asserts that agreement numerically.

The ALASSO adaptive pilot is where the second point mattered.  It ran
`iterative_lstsq` at `min(self.tol, 1e-8)` -- unreachable on a float32 operator --
and nothing consumed its `converged` verdict, so a destroyed pilot silently became
meaningless weights.  Measured on c7 (float32, one RTX 3090): the pristine pilot
ended `invalid_search_direction` with `normr` 1.0e19 and `||c||` 4.0e19, which
drives every weight `(|c|+eps)^-gamma` to ~1e-19 -- a numerically ZERO adaptive
penalty, i.e. an ALASSO that had degenerated into truncated OLS (the fit reported
3678 nonzeros, no sparsification at all).  The pilot now has its own accuracy
budget, `PHEASY_ALASSO_PILOT_TOL` (default 1e-5), deliberately independent of the
FISTA tolerance, and its certificate is kept in `model.pilot_info_`.  At 1e-5 on
c7 it converges in 126 iterations (0.3s versus 32.7s at the floor), its weight
vector is within 0.64% of the fully-iterated solution, and the resulting fit is
identical to the one from that solution (alpha 2.4754e-05, rmse 4.5076e-02,
r2 0.911658, 79 nonzeros).  At 1e-4 the peak weight difference is already 62%,
because the smallest coefficients -- the ones that set the top of the weight range
-- are where the relative error concentrates.  This changes what an ALASSO fit on
float32 data DELIVERS: the penalty is no longer numerically zero and the solution
is genuinely sparse (c7, 5-point 2-decade auto grid: 1159 -> 79 nonzeros, r2
0.938 -> 0.912).  That is the intended behaviour being restored, but an earlier
float32 ALASSO result should be re-run rather than trusted.

Two more sites made the same ambient-precision mistake, and one wrapper chain
had no precision provenance at all:

* `_solve_subset` computed its tolerances from `PHEASY_LSQR_ATOL` / the model's
  `lsmr_*` DIRECTLY rather than through `_lsmr_tol`, so the subset LSMR -- the
  CPU matrix-free path every non-resident RFE round uses -- never met the floor.
  It now passes the operator it is about to solve.
* `_scale_operator` and `_row_slice_op` declare `dtype=float64` around whatever
  they wrap, and `_make_masked_op` reported only the container dtype, so a
  wrapped float32 operator was indistinguishable from a genuine float64 one.
  All three now record the precision of what they multiply in `_data_dtype`,
  which `_array_precision` prefers over inference.  Measured on c7: the masked
  view and its column-scaled wrapper reported float64 before, float32 after.
* The RFE model is built before any subset exists, and calling `_lsmr_tol` there
  without an operator baked `PHEASY_SM_DTYPE` into the model default `lsmr_atol`
  (1e-8 -> 1.19e-6 whenever that variable was set), which then over-raised the
  tolerance of every later FLOAT64 subset in the same process.  The default is
  now the plain requested value, and the floor is applied where the operator is
  known.

Scope, honestly: on c7 the CPU subset LSMR was NOT producing garbage.  scipy's
LSMR runs in float64 over float32-stored factors and reached `normar` 8.9e-8, so
it converged (istop=2) even at the un-floored 1e-8 -- the recurrence-drift
failure mode is specific to the resident float32 CGLS.  What this round changes
on c7 is cost and reporting: the full-support subset solve stops after 614
iterations instead of 917 (column-scaled: 76 instead of 98) with the same
coefficients (`||c||` 105.7802 vs 105.7811, relative difference 9e-6), and the
raise to the floor is reported instead of silent.  What it changes in general is
that a float64 subset can no longer inherit a float32 tolerance from the
environment.

**A measured floor, because the analytic one is not always reachable.** The
floor above is an estimate; when it is below what the arithmetic can actually
reach, the honest test never fires and the solve grinds to `maxiter`. The CGLS
now tracks the best honest iterate by the primary criterion and stops when it
stops improving (`PHEASY_CGLS_STALL_POINTS`, default 5 verifications, each
`PHEASY_CGLS_VERIFY_EVERY` = 50 iterations apart). It then delivers THAT iterate
and reports `stop_reason=precision_floor`, `tolerance_floor_measured`, and an
`atol_effective` raised to the measured value plus `_CGLS_FLOOR_MARGIN`. The
measured floor is capped at `_CGLS_FLOOR_MAX` exactly like the analytic one: past
it the verdict is `converged=False` / `precision_floor`, i.e. a refusal, not a
certificate.

**And a floor that is REPORTED even when nothing fires.** The valve above only
ran inside the honest re-verification, which the cheap recurrence test has to
trigger first. A solve that never triggers it -- measured on the production MgC
OLS (454656x69487, float32) -- therefore had no floor detector at all: 20000
iterations reached a criterion of 1.33e-03 against an analytic floor of 1.19e-06
(three orders below what the arithmetic can express, and the best value of the
whole run: 5.31e-03 at 10000 iterations, 1.33e-03 at 20000, 2.02e-03 at 40000),
and the fit was refused as a bare `iteration_limit` with no number to act on. The
CGLS now also probes honestly and unconditionally from
`probe_start = min(_CGLS_PROBE_WINDOW_MAX, max(maxiter // 2, verify_every))`
(`PHEASY_CGLS_PROBE_START` overrides it; `_CGLS_PROBE_WINDOW_MAX` = 8000, the same
"never before half the budget" rule FISTA needed). Two outcomes, both honest: a
floor inside `_CGLS_FLOOR_MAX` is promoted and certified exactly as above, and a
floor ABOVE it makes the verdict `stall_above_floor` -- a refusal that names the
value, carries it in `stall_floor` / `stall_iteration` / `best_criterion` /
`criterion_value`, and tells the caller which tolerance would certify. The probe
never ends the run on an above-cap floor: the recurrence is still creeping down
there, so stopping would deliver a worse iterate than the remaining budget reaches.
It also hands back the best iterate it saw when that beats the loop`s last one.

What that is worth on the production problem (Mg8C120, 454656x69487, float32,
default tolerances, PHEASY_OLS_MAXITER=20000): on TWO shards, where the cheap test
never fires, the fit used to grind all 20000 iterations and come back
`fit_accepted=False` / `iteration_limit` in 792.3 s (rmse 8.6662e-03); with the
probe it stops at 10750 iterations, certifies against the measured floor
(`stop_reason=precision_floor`, `atol_effective` 9.381e-04, inside the cap) and is
ACCEPTED in 497.8 s with rmse 9.7488e-03 / r2 0.999874 -- 1.6x faster and finally
trustworthy, at the cost of the 12 % residual difference that stopping at the floor
rather than past it makes. On ONE card the same fit already reached the floor
(10200 iterations, accepted) before this change, so the probe's contribution is
exactly the configuration whose recurrence test never fires. RIDGE with the production CV count on the same card and operator is
unchanged in verdict (accepted, alpha 1.0e-06, rmse 1.7697e-02, r2 0.999585) and
takes 606.8 s at cv=5.

**Do not reach for a loose `atol` as the lever.** The criterion is RELATIVE
(`||A^T r|| <= atol * ||A|| * ||r||`), so it is also satisfied while `||r||` is
still large: measured on that same fit, `PHEASY_OLS_ATOL=3e-3` certified after 45
iterations -- `converged=True`, 208 s -- but delivered rmse 1.8832e-01 / r2
0.952971, twenty times worse than the run above. The measured floor the probe
reports is the trustworthy lever; a tolerance chosen by hand from it is not.

The margins are not cosmetic. On the c7 operator three recomputations of
`||A^T (y - A x)||` for the SAME x gave 2.715779e-06 / 2.767306e-06 /
2.832976e-06 -- a +-2% spread from the sparse-kernel accumulation order in
float32. A 1% margin therefore put the verdict inside the noise: measured,
`atol_effective` 7.825285e-06 against a re-derived `normar` 3.242940e-06 missed
the test by 1.25% and the fit was rejected for no reason. `_CGLS_STALL_MARGIN`
(0.05, what counts as progress) and `_CGLS_FLOOR_MARGIN` (0.10, headroom on the
certified floor) both sit above that noise.

Scope, honestly: this fixes PLATEAUS, not slow convergence. On c7 the resident
OLS solve went from `stop_reason=iteration_limit`, `itn=20000`, 331.5s,
`fit_accepted=False` to `precision_floor`, `itn=436` (and 486/1557 on other runs),
`converged=true`, `fit_accepted=true`, with `normr`/`normx` unchanged, and the
unpenalized ALASSO pilot from 20000 iterations / 32.7s / `iteration_limit` to 536
iterations / 0.9s / certified (its weights unchanged at 0.1273 .. 3.17e5). Four
synthetic float32 fixtures with 1e-6..1e-9 column spreads still end at
`iteration_limit`: their criterion keeps creeping down by more than the progress
margin, so there is no plateau to certify and the refusal is correct.

**A measured floor for FISTA, with a window long enough to mean something.**
The CGLS floor above had no FISTA counterpart: a tolerance below what the
float32 operator can express made the relative KKT certificate of
`_fista_twolevel` (resident GPU FISTA) and of `_fista_lasso` (the two-level / CPU
FISTA the resident fallback uses) unreachable, so the run always ground to
`max_iter` and was then rejected by the acceptance gate. Measured on the Mg8C120
c2=7 A / c3=4 A LASSO (296 configs, 69487 free IFCs, float32 `SM_prime`): the
default request of 1e-4 reached 2.566e-4 after 20000 iterations (3364 s) and
2.523e-4 after 60000 iterations (2759 s) -- 1.7 % of certificate movement for 3x
the budget, both runs ending in `fit_returned_not_accepted`.

Both FISTA loops now keep the best iterate by the certificate (checked every 20
iterations) and stop when it stops improving, delivering THAT iterate certified
at the measured floor plus 10 % headroom. `regularized_solver_info` carries
`stop_reason` (`converged`, `converged_on_raised_tol`,
`converged_measured_floor`, `stall_above_floor`, `iteration_limit`),
`tol_requested`, `tol_effective`, `measured_floor` and `stall_points`, and
`fit_accepted` follows `converged` exactly as before. CV folds and the alpha-path
walk pass the valve OFF: there the iteration cap is the design (see
`_cv_hit_cap`), not waste.

The window is the load-bearing part, and the first version got it wrong. FISTA's
tail is SUBLINEAR, not flat: on this operator the certificate runs 9.90e-04 at
iteration 80, 2.98e-04 at 7160 and 2.52e-04 at 60000. A 3-check (60-iteration)
window declared a "floor" of 9.90e-04 at iteration 80 of a tol=1e-6 run and
delivered `||x||=460.4` / `nnz 37285` against the converged run's `608.4` /
25847 -- a certificate plateau is not a precision floor. The window is now
`min(8000, max(200, max_iter // 2))` iterations (`PHEASY_FISTA_STALL_POINTS`
overrides it), so the valve cannot fire before half the budget is spent, and
never before 8000 iterations unless the budget is smaller than that.

Verification (RTX 3090, `tmp/test_fista_valve.py` and
`tmp/test_fista_valve_cpu.py`, both PASS): with the valve off an unreachable tol
still grinds to `max_iter` (`iteration_limit`, rejected); with it on the run
stops at the measured floor (synthetic CPU fixture: 180 of 3000 iterations, best
`kkt` 1.503e-07, `tol_effective` 1.654e-07, `converged=True`) and its
coefficients differ from the valve-off solution by 2.5e-07 relative; a 3-point
window fires an order of magnitude earlier than the default one (180 versus 1760
iterations), which is the regression test for the premature stop above; and
`PHEASY_FISTA_FLOOR_MAX=1e-15` still refuses (`stall_above_floor`,
`converged=False`) instead of certifying.

On the production Mg8C120 LASSO (296 configs, resident float32, tol=1e-4,
max_iter=20000) the valve did NOT fire: the certificate descended sublinearly to
2.161e-4 with no plateau (it was still improving at the cap), so
`stop_reason=iteration_limit`, `fit_accepted=False` -- exactly like the earlier
1e-4 runs. That is correct: the consecutive-stall window is only a certificate
for a genuine plateau, and this operator's certificate does not plateau (the
CGLS fixtures with the same behaviour also correctly refuse). The lever that
made the accepted v4 fit work is the tolerance: certifying at `--tol 3e-4`
reached 2.98e-4 in 7160 iterations and was accepted -- the honest way to handle
a reachable-but-sublinear target rather than an unreachable precision floor.

**A frozen iterate is not a precision floor (float32 backtracking pathology).**
The valve above decides from the certificate alone, and a *frozen* solver keeps
that certificate perfectly flat too. The Mg8C120 `exp_std` run (LASSO, `--std`,
tol=3e-4, max_iter=60000) exposed exactly that: its manifest reports
`lipschitz=Infinity`, `lipschitz_inflations=292`, `n_iter=60000`,
`kkt_relative=3.81e-4` -- a *worse* certificate than the same operator reached
in 7160 iterations (2.98e-4) on the accepted fit, and rejected. The cause is the
step-size backtracking test: it compared the objective against a `1e-9`
*relative* slack, but a float32 objective over a ~2e5-scale residual cannot
resolve differences below ~1e-7 relative, so noise-level increases counted as
real increases and doubled `L` on every check until float32 overflowed it to
`inf`; `1/L` is then 0 and the iterate never moves again.

Two changes, in `_fista_twolevel` and (for the slack) `_fista_gram`:

- the monotonicity slack is dtype-scaled -- `_FISTA_F_REFINE * eps(dtype)`
  (8 eps: ~1e-6 relative in float32, ~1.8e-15 in float64) instead of a fixed
  `1e-9`, so rounding noise no longer triggers backtracking;
- a health guard ordered *before* the floor valve stops the run with
  `stop_reason=step_size_diverged`, `converged=False` when `L` is non-finite or
  the inflation count reaches `PHEASY_FISTA_MAX_INFLATIONS` (default 40). It
  never certifies: a step-size failure must not be reported as
  `converged_measured_floor`. `_fista_lasso` (no backtracking of its own) gets
  the same refusal for a zero or non-finite step, which is how a broken
  Lipschitz estimate freezes the CPU path.

Verification (`tmp/test_fista_guard.py`, PASS): with
`PHEASY_FISTA_MAX_INFLATIONS=1` and an `L` two times too small the run stops at
iteration 40 with `step_size_diverged`, `converged=False`, `measured_floor=None`
(never certified); the pre-existing non-finite-certificate `RuntimeError` still
fires first for an extreme underestimate (`L/50`); under the default budget the
same `L/2` start self-heals in one inflation and converges, so the wider slack
does not weaken step-size control; and a sound `L` now reports 0 inflations.

Production validation of that fix (`tmp/exp_std_fix_verify.sh`, the exact
`exp_std` recipe rerun in a fresh directory, same `alpha=2.305464e-08`):

| | old (broken) | new (fixed) |
|---|---|---|
| `lipschitz_inflations` | 292 | **0** |
| `lipschitz` | `Infinity` | 6743.02 |
| `n_iter` | 60000 (cap) | **18500** |
| `fit_accepted` | False (rejected) | **True** |
| `stop_reason` | `iteration_limit` | `converged_measured_floor` |
| `kkt_relative` | 3.813e-4 | 3.793e-4 |
| `re` / `r2_score` | 0.022151 / 0.999509 | 0.022711 / 0.999484 |

So the noise-slack change removes the freeze (292 inflations -> 0), the same
operator now converges in **18500 instead of 60000** iterations, and the fitted
constants are unchanged in quality (`re` 0.0221 -> 0.0227, `r2` identical to 4
decimals). The measured-floor valve is what ended it at 18500 rather than the
60000 cap: this is the waste-avoidance the valve was written for, on a real
production fit rather than a synthetic fixture.

**Grouped by-config holdout through the resident paths** (18 configs / 2 splits,
c7, float32, one RTX 3090; `tmp/cert_ab_20260915/holdout_twolevel.py`). This is
the generalisation evidence for the repair, and it has to be run on a TwoLevel
operator: `holdout_eval.py` densifies SM, so it exercises the dense closed-form
RIDGE and the sklearn/FISTA dense LASSO, neither of which this repair touches.

| method | relL2 before | relL2 after | nnz before | nnz after |
| --- | --- | --- | --- | --- |
| RIDGE | 6.93e+18 (`fit_accepted=False`) | 1.18e-02 (accepted) | 3678 | 3678 |
| ALASSO | 8.27e-02 | 7.49e-02 | 3158 | 588 |
| OLS | 1.1760e-02 (`fit_accepted=False`) | 1.1760e-02 (accepted) | 3678 | 3678 |

The RIDGE row is the whole point of the repair: pristine resident RIDGE on
float32 delivers a destroyed coefficient vector (relL2 6.9e18, the `normx=inf`
explosion) on HELD-OUT configurations, and the repaired one delivers 1.18e-02.
The ALASSO row answers the sparsity question the adaptive-pilot fix raised: it is
both sparser (588 versus 3158 nonzeros) and better on unseen data (7.49e-02
versus 8.27e-02), so restoring the pilot did not cost generalisation. The OLS row
is the certificate-only case: identical predictions, `fit_accepted` flips from
False to True.

**Every method on c7 at float32, before and after** (resident paths, one
RTX 3090, `tmp/cert_ab_20260915/sweep_methods.py`; OLS `maxiter=20000`, RIDGE a
10-point 1e-6..1e4 grid, LASSO/ALASSO the auto grid with `nalpha=6, decades=3`,
`cv=2`):

| method | before | after |
| --- | --- | --- |
| OLS | `fit_returned_not_accepted`, `true_residual_check_failed`, itn 86 | accepted, `precision_floor`, itn 786, rmse unchanged |
| RIDGE | `fit_returned_not_accepted`, `invalid_search_direction`, itn 682, rmse 9.462e-04 | accepted, `converged`, itn 3, rmse 8.202e-04 |
| LASSO | accepted, rmse 5.901e-03, 2637 nnz | accepted, rmse 3.587e-03, 2637 nnz |
| ALASSO | accepted, alpha* 8.54e10, rmse 2.186e-02, 2418 nnz | accepted, alpha* 2.48e-06, rmse 1.633e-02, 266 nnz |

Two methods were being rejected outright and are now accepted. The LASSO and
ALASSO rows improve for a reason worth stating separately, because it is a third
symptom of the same root cause rather than a consequence of the FISTA solver
changing: **the relaxed-LASSO debias stage was silently a no-op**. Its support
refit is guarded by "keep the new coefficients only if the residual did not
grow", which is right -- but on float32 the resident CGLS refit came back
`invalid_search_direction`, the guard rejected it, and the results still reported
`debias_backend=gpu_cgls` with nothing recording that the L1 shrinkage had NOT
been removed. Measured on c7 LASSO: pre-debias residual 0.942593, final residual
0.942593 (identical -- the refit never took effect). With the repair the refit
converges (`precision_floor`, 298 iterations), the residual drops to 0.572931,
and it is kept: training rmse 5.901e-03 -> 3.587e-03. ALASSO moves the same way.
The results now carry `debias_accepted`, `debias_stage`,
`debias_residual_refit`/`debias_residual_shrunk`, `debias_solver_info` and, when
the guard does reject a refit, `debias_fallback_reason`, so a declared post-fit
stage can no longer vanish without saying so.

**Every LASSO/ALASSO backend now certifies its own solve.** The dispatch is
`_lasso_backend(A)`: a TwoLevel/LinearOperator with the resident path enabled ->
GPU FISTA resident; any other LinearOperator, or a scipy sparse matrix past the
densify budget -> the FISTA "iterative" backend (GPU-SM when sharded); a dense
matrix that fits CUDA -> the GPU Gram FISTA; and everything else (dense, or
sparse within the densify budget) -> sklearn coordinate descent. The FISTA
backends always published `converged`/`kkt_relative` and the gate vetoed on them;
the two sklearn branches published NOTHING -- the plain one set no
`regularized_solver_info_` at all, and the adaptive one (which runs its own
LassoCV on the weighted matrix) reported `{}` with `execution_backend=None`, so a
whole adaptive solve was invisible to the acceptance gate. Both now publish
`backend="cpu_dense_coordinate_descent"` with `converged = (n_iter_ < max_iter)`:
sklearn signals the cap with `n_iter_ == max_iter` plus a ConvergenceWarning, and
a warning is not a gate. Note what "sparse" means here: a scipy sparse input
within the densify budget is densified and follows the sklearn route, and it
reproduces the dense numbers exactly -- input sparsity selects the backend, and
is unrelated to whether the LASSO SOLUTION is sparse.

Measured on c7 (9 configs / 5103 rows, float32, one RTX 3090, IDENTICAL alpha
grid on both sides so only the solver differs; `tmp/cert_ab_20260915/
dense_vs_resident.py`):

| method | backend | alpha* | nnz | rmse | wall | certificate |
| --- | --- | --- | --- | --- | --- | --- |
| LASSO | dense sklearn CD | 1e-06 | 2065 | 4.3698e-03 | 298.9s | converged (n_iter 53) |
| LASSO | resident GPU FISTA | 1e-06 | 2065 | 4.3698e-03 | 4.4s | converged (n_iter 120) |
| ALASSO | dense sklearn CD | 1e-06 | 336 | 1.2249e-02 | 15.2s | converged (n_iter 17) |
| ALASSO | resident GPU FISTA | 1e-06 | 338 | 1.2206e-02 | 4.1s | converged (n_iter 40) |

The LASSO coefficients agree to 1.7e-06 relative and ALASSO to 3.4e-02 (336
versus 338 nonzeros), so the dense backend is not a worse answer -- it is the same
convex problem solved by coordinate descent on the CPU, at ~68x the wall time for
LASSO. An earlier comparison that did NOT pin the grid differed by 2.8e-01
relative; that was the grid, not the solver.

That missing-certificate observation is RESOLVED, and it was a harness artifact
rather than a code path: two drivers wrote the same log filename, one of them
before the adaptive certificate existed, so the `{}` line came from the older
overlay. Re-measured on the REAL c7 rows with sparse input and the auto grid at
2 / 3 / 4 / 6 / 9 configurations: `_AdaptiveLassoCV` publishes the full
certificate at every size (`tmp/cert_ab_20260915/bisect_cert.log`). The
`regularized_solver_info_missing = <model class>` marker stays as the audit trail
for any future silence.

The dense chain is numerically untouched by this repair, measured rather than
inferred (c7, 9 configs, shared manual grid, `PHEASY_USE_GPU=0`, sklearn
coordinate descent): pristine LASSO reported `execution_backend=None` with
`regularized_solver_info={}` and rmse 4.3698e-03 / r2 0.99919 / 2065 nonzeros in
300.3s; the repaired code returns the SAME rmse, r2, alpha* and nonzero count in
298.9s and additionally publishes its certificate. Only the certificate changed.

**Which backend every method uses on c7 with GPU required** (25515x3678, float32,
`PHEASY_USE_GPU=1`, `tmp/cert_ab_20260915/gpu_audit.py`):

| method | execution_backend | subset/debias | stays on CPU |
| --- | --- | --- | --- |
| OLS | `gpu_twolevel_tsqr` | - | metrics |
| RIDGE | `gpu_resident_iterative` | - | metrics |
| LASSO | `gpu_twolevel_resident` | debias `gpu_cgls` | metrics |
| ALASSO | `gpu_twolevel_resident` | debias `gpu_cgls` | metrics |
| RFE | `gpu_rfe_resident_iterative` | `gpu_resident_iterative` | orchestration, ranking |
| RFE-OLS-TSQR | `gpu_rfe_resident_iterative` | `gpu_resident_iterative` | orchestration, ranking |

Everything that SOLVES is on the GPU for all six. What is still CPU by design: the
postfit metrics (RMSE / R^2 / MAE, host numpy, O(n)) and RFE orchestration plus
importance ranking -- the latter becomes GPU with `PHEASY_GPU_RFE_RANKING=1`.

The all-GPU recipe is `PHEASY_GPU_MODE=required` (or `PHEASY_USE_GPU=1`), which is
what makes the resident paths the DEFAULT via `_resident_default()`;
`PHEASY_GPU_FALLBACK=0` makes a resident setup failure raise instead of quietly
running something else. `PHEASY_GPU_LASSO_RESIDENT`, `PHEASY_GPU_RIDGE_RESIDENT`,
`PHEASY_GPU_OLS_RESIDENT` and `PHEASY_GPU_RFE_RESIDENT` all default to the
resident path in required mode; `PHEASY_GPU_DEBIAS` (default 1) keeps the
relaxed-LASSO refit on the GPU; `PHEASY_GPU_TSQR=1` selects streamed GPU TSQR for
oversized dense OLS; `PHEASY_GPU_LASSO=1` (default) sends a GPU-eligible DENSE
LASSO to the Gram FISTA instead of sklearn.

One footgun removed: resident RFE needs a serial outer loop (the retained operator
is not fork-safe), and it used to REFUSE whenever `n_jobs != 1`. The shipped
`fit_3090.sh` exports `PHEASY_N_JOBS=8`, so "GPU required + RFE" was a hard
RuntimeError there and the documented escape hatch was to give up the GPU. In
required mode the resident solve is the requirement and outer parallelism is only
a CPU-side optimization, so it now serializes with a RuntimeWarning and keeps
every subset solve on the GPU -- verified with `PHEASY_N_JOBS=8` and no RFE
override: backend `gpu_rfe_resident_iterative`, `subset_solver`
`gpu_resident_iterative`.

OLS exposes iterative stopping diagnostics in `Optimizer.results["solver_info"]`.
LSQR/LSMR warn on iteration/condition limits. Check `converged` before accepting
the coefficients. `PHEASY_OLS_MAXITER`, `PHEASY_OLS_ATOL`, and
`PHEASY_OLS_BTOL` configure two-level OLS; RFE uses `PHEASY_LSQR_*`.
Exact two-level column norms use bounded row products. The CPU fallback's block
budget is `PHEASY_COL_NORM_BLOCK_BYTES` (64 MiB) and the GPU path's is adaptive
(see "Host cost is part of the GPU path" below).
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

### Host cost is part of the GPU path: fold views, GPU column norms, canonical factors

Moving the arithmetic to CUDA does not by itself make a fit GPU-bound. Measured at
production size (Mg8C120 c2=7 A / c3=4 A, 296 configs, 454656x69487, float32
`SM_prime` with nnz 1.127e9, two cards, shared box) a full-scale RIDGE fit took
1298.8 s of wall time at 21% mean device utilisation, because the host kept
REBUILDING and re-uploading the same factors. Three separate host passes were
involved; the instrumented operator counts below are exact, the times are wall
clock on a loaded shared box (so they are upper bounds), and the matvec/rmatvec
figures are launch time rather than kernel time, because the CGLS loop enqueues
and the device was idle most of the run.

| step | before | after |
| --- | --- | --- |
| `GpuTwoLevelOperator.__init__` per RIDGE fit (NCONF=6, serial) | 5 | **1** |
| `GpuTwoLevelOperator.__init__` per RIDGE fit (NCONF=296) | 3 built, then a further replica refused (free VRAM exhausted) | **1**, 456.9 s |
| host `csr_tocsc` transposes per fit | 11 calls / 4.95 s at NCONF=6 | none (fold views) |
| column-norm pass (NCONF=24) | CPU 125.5 s (RIDGE) / 197.9 s (OLS) | GPU kernel |
| fold-solve wall, NCONF=6 serial, 48 solves | 21.9 s | **6.1 s** |
| fit wall, NCONF=6 serial | 43.4 s | **17.7 s** |

**One resident operator per fit; folds are row views** (`_ResidentRowView`).
Every CV fold used to re-slice the host factors and upload its own
`GpuTwoLevelOperator`, which is one factor upload plus one host transpose PER
FOLD -- cost that grows with the fold count and dwarfs every kernel in the fit.
The operator the fit already has on the device covers exactly the same matrix, so
a fold is a row SELECTION over it: `GpuSubsetOperator`, the same view the resident
RFE subsets use. Results are unchanged (fold-view A/B at NCONF=6: RIDGE rmse
8.9125e-03..8.9563e-03, LASSO 2.698e-02..2.713e-02, ALASSO
1.0044e-01..1.0046e-01 in both arms). At NCONF=296 the difference is not a
speedup but a verdict: the per-fold path built 3 replicas (571.9 s), then REFUSED
the next one -- "Resident
two-level GPU estimate 6683335192 bytes exceeds budget 538842644", i.e. the fold
operators had consumed the free VRAM of both cards -- and the fit died at 787 s
with `fit_accepted` never computed, while the fold-view fit ran to completion in
1298.8 s (peak host RSS 20.48 GiB against the per-fold path's 33.79 GiB).

**A fold view is an operator only if it can be multiplied like one.**
`_ResidentRowView` exposes `matvec`/`rmatvec`, and `_is_linear_operator()`
duck-types on exactly that -- so every dispatcher treated it as an operator -- but
it was never a scipy `LinearOperator`, so it had no `.T` and no `@`. `_row_slice`
hands the view to EVERY consumer of a row slice whenever the operator carries a
cached `_gpu_ridge_op` (which the RIDGE CV writes), and only `_ridge_solve` knew how
to consume it (through `_gpu_ridge_parent`). The LASSO path does not: `_lasso_backend`
returns `"iterative"` (FISTA) whenever the resident two-level LASSO is not active
(e.g. `PHEASY_GPU_LASSO_RESIDENT=false` with `PHEASY_GPU_SM=1`), and FISTA needs the
transpose for the KKT scale (`A.T @ y`), the gradient (`A.T @ (Az - y)`) and the
Lipschitz power iteration (`A.T @ u`, `_estimate_lipschitz`). Measured 2026-09-17 on
the Mg2C60 2x2x2 fc2 fit (jobid 2119, 12 min, exit 1): the fit passed the dispatch
that had already classified the view as an operator and died one line later with
`AttributeError: '_ResidentRowView' object has no attribute 'T'`. **Adding `.T`
alone does not fix it**: the very next statement (`A @ z`) fails the same way because
`@` was never defined either, and the Gram builder multiplies 2-D blocks
(`A @ I_blk`, `A.T @ A_blk`) as well, which is the path that fit actually took
("Gram path: G=5x5 built"). Both are now defined; `.T` is a `LinearOperator` over the
swapped BOUND methods and must never be a closure over `self.T` or over a name later
rebound to the transpose -- that is the recursion commit b6b7f92 had to fix in
`_solve_subset`. The contract is pinned by `dev/test_resident_row_view_transpose.py`
(9 tests, CPU-only, no CUDA needed): the transpose must match the dense slice, must
not recurse, must keep the factor dtype provenance the floor reads, and -- the
end-to-end contract -- a cached resident op must be INVISIBLE in the numbers, i.e. the
LASSO coefficients are bitwise identical with and without the cache, while the backend
still reports `cpu_iterative_fista`.

*Lesson*: `_is_linear_operator()` duck-types on `matvec`, so a hand-rolled operator
that satisfies the dispatcher can still fail one line later. The interface the
consumers actually use is the union of `matvec` / `rmatvec` / `.T` / `@` / `row_slice`
-- not the subset the first consumer happened to need. The sibling wrappers
(`_row_slice_op`, `_make_masked_op`, `_scale_columns`, `TwoLevelSM.row_slice`) are all
real `LinearOperator`s; this one was the only hole.

**One definition per operator member -- a second silently shadows the first.**
`_ResidentRowView` ended up carrying TWO `__matmul__` and TWO `T`: the P50 block
above and a second, later block added independently for the Mg8C120 folds. Python
keeps the LAST definition, so the effective view had lost the 2-D column path
(`view @ I_blk`) and `_ResidentRowAdjoint` had no `_data_dtype`. The 2-D product is
exactly what the Gram builder uses (`A @ I_blk`, then `A.T @ A_blk`), and
`_fista_lasso` wraps the whole Gram build in `try/except` -- so a cached-resident
view did not crash, it printed "Gram build failed" and dropped to the slow matvec
FISTA: a silent performance/provenance regression behind a warning. Measured on the
MgC view: `view @ I_blk` raised "matmul: Input operand 1 has a mismatch in its core
dimension 0", `view @ B` and `view.T @ B` raised for every 2-D `B`, and
`_compute_gram(view, y)` raised where `_compute_gram(dense_slice, y)` worked.

The fix keeps ONE `__matmul__` (the 2-D one) and ONE `T` -- the `_ResidentRowAdjoint`
class, completed with the 2-D column path, a `dot` alias and the `_data_dtype`
provenance idiom shared with `_row_slice_op`/`_scale_operator`. A/B run on BOTH
trees: the duplicated block fails `dev/test_resident_row_view_transpose.py` with 3
errors (2-D product, `_compute_gram`, adjoint provenance); the fixed one passes
11/11 on CPU and on real CUDA. The lesson is the one the `T` docstring now states:
never leave a second definition of a dunder in the class body -- the dispatchers keep
calling `matvec`/`@`/`.T` while the effective operator quietly changes underneath.

**The view must report the FACTORS' precision, not its own float64 shell.**
`_ResidentRowView` is declared `dtype=float64` (it returns numpy float64 from
`matvec`), but it multiplies the PARENT's stored factors, which are float32 in
production. `_array_precision` therefore read float64 for every view, so the
tolerance floor (`_lsmr_tol`, and FISTA's `tol_effective`) judged a 1e-7 request
reachable on a float32 operator -- the exact failure the floor exists to prevent.
The view now defines `_data_dtype = _array_precision(self._parent)`, the same
provenance idiom `_row_slice_op` and `_scale_operator` already use, and the
adjoint inherits it. Measured on the MgC resident operator (float32 factors):
`_array_precision(view)` went from `float64` to `float32`. A/B on the 11-test
file: the pre-P50c file fails the provenance test, the fixed one passes.
**Column norms are computed where the factors already are.** `TwoLevelSM.col_norms`
used to run the exact CPU sparse-sparse product on every standardized RIDGE/OLS fit
-- 125.5 s of a 154.4 s RIDGE and 197.9 s of a 219.9 s OLS at NCONF=24, i.e. 81% and
90% of the fit in a pass whose iterations cost 4.9 s and 16.2 s. The resident
operator implements the same computation as blocked basis matvecs, and the operator
it builds is CACHED as the same slot the solver looks in, so the norms and the solve
share one upload. The block budget is adaptive (`_norm_block_budget`): a quarter of
free VRAM, capped at 8 GiB, floor 512 MiB. Isolated on the MgC operator the kernel
takes 12.5 s at the adaptive budget versus 23.0 s at the old 512 MB ceiling and
125.5 s on the CPU (NCONF=24); at NCONF=296 the same kernel is 153.7 s. The GPU and
CPU definitions of the norm agree to 1.7e-07 max relative difference, and a second
call is free (cached).

**The adjoint transpose is shared, not rebuilt per shard.** `GpuTwoLevelOperator`
built from raw factors transposes a fresh column slice per shard
(`prime[:, c0:c1].T`), and the LASSO path already avoided that by handing every
replica the canonical factors from `_canonical_twolevel_host()`. RIDGE, OLS and the
column-norm pass did not, so they paid it again: measured on the MgC factor
(nnz 1.127e9) the per-shard transposes cost 251.7 s for two shards, while one
canonical copy plus transpose costs 167.6 s (`.T.tocsr()` 155.2 s, the `tocsr` copy
16.8 s) and is then reused by every shard, by the solve that follows, and -- with
`twolevel_host_factors()` keeping it on the operator -- by any later operator over
the same factors (RIDGE then OLS, or a repeated fit). Retention is a HOST MEMORY
decision, not a speed one: the canonical set adds ~16.9 GiB for this operator, so it
is kept only while `MemAvailable` can afford twice that plus an 8 GiB headroom for
the fit itself (`_HOST_FACTOR_CACHE_HEADROOM`) AND the set is at least 256 MiB
(`_HOST_FACTOR_CACHE_MIN_BYTES`, below which rebuilding the transpose is cheaper
than holding it), and `PHEASY_HOST_FACTOR_CACHE=0` (or `=1`) overrides the
automatic verdict. A
single-shard build now uses the supplied adjoint instead of transposing a second
copy of `prime`.


**Production evidence, one table (Mg8C120, 296 configs, 454656x69487, float32,
`METHODS="RIDGE OLS"` in ONE process so the second fit reuses the first one`s
operator):**

| method | cards | wall | `__init__` (operator constructions) | verdict | rmse |
| --- | --- | --- | --- | --- | --- |
| RIDGE cv=5 | 1 | 606.8 s | 1 (9.6 s) | accepted, alpha 1.0e-06 | 1.7697e-02 |
| RIDGE cv=5 | 3 | 713.5 s | 1 (78.9 s) | accepted, alpha 1.0e-06 | 1.7653e-02 |
| OLS (default tol) | 1 | 470.0 s | 1 (9.5 s) | accepted, `precision_floor` @10800 | 9.7602e-03 |
| OLS (default tol) | 2 | 497.8 s | 1 (60.3 s) | accepted, `precision_floor` @10750 | 9.7488e-03 |
| OLS (default tol) | 3, after RIDGE | 492.8 s | 1 (71.5 s, NO new transpose) | accepted, `precision_floor` @10850 | 9.7650e-03 |
| OLS (default tol), pre-probe code | 2 | 792.3 s | 1 | REFUSED, `iteration_limit` @20000 | 8.6662e-03 |

Two things that table settles. First, the sharded layouts are SLOWER than one card
on this problem (RIDGE 713.5 s on 3 cards against 606.8 s on 1), which is the
sharding finding above restated at production size and with the configuration
`fit_3090.sh` ships. Second, the canonical factors are shared ACROSS methods: the
OLS fit that ran after RIDGE in the same process built its operator in 71.5 s with
no host transpose at all, against the ~215 s per-shard transpose work it would have
redone on its own.

Verified end to end at NCONF=296 on the same two cards, same y, new code against
the arm above: `GpuTwoLevelOperator.__init__` 456.9 s -> **42.1 s**, the whole
column-norm phase (canonical build + operator + kernel) 619.5 s -> **275.4 s**, the
fit 1298.8 s -> **505.8 s** (2.6x) with mean device utilisation 21% -> **61%**, and
the same model out (rmse 1.7661e-02 vs 1.7651e-02, r2 0.999586 vs 0.999587, same
alpha, 69487 nonzeros, `fit_accepted=True`); the 0.06% rmse difference is the
documented +-2% float32 kernel nondeterminism, not a changed solver. The price is
host RAM: peak RSS 20.48 -> 37.35 GiB while the canonical set is retained (switch
it off with `PHEASY_HOST_FACTOR_CACHE=0` on a memory-tight host). The single-card
path was verified separately at NCONF=24 (card 2, old vs new code): `__init__`
5.9 s -> 1.8 s, identical model (alpha, nnz, r2 0.999895, rmse within 0.3%),
`dev/test_resident_host_factors.py` (6 tests) pins the helper's contract.

What is still host-side by design: file I/O, the CV split construction, the final
metrics and (for RFE) the orchestration and ranking. Everything that solves is on
the GPU in all six methods.

**Sharding is a memory tool on this host, not a speed tool (measured).**
`torch.cuda.can_device_access_peer` is False for EVERY pair of cards here (no
NVLink, no PCIe peer access), so each cross-card tensor copy is staged through host
memory, and the sharded two-level matvec must do two of them per iteration
(broadcast the `mid` vector, collect the row blocks). Measured on the MgC operator
(454656x69487, nnz 1.127e9): the matvec+rmatvec pair costs 22.50 ms with 1 shard,
23.02 with 2, 24.15 with 3 and 24.73 with 4 -- more cards make an ITERATION
slightly SLOWER. The two half-SpMMs themselves do overlap (interleaved medians in
one process on idle cards: 5.45 ms for both halves against 10.60 ms for the same
work on one card), so the cost is the copies, which sit on the critical path: the
whole sharded matvec measures 10.90 ms against 10.61 ms for ONE card doing the
entire matrix, and a host five times faster would not change that. Issuing the
remote shard on its own stream with `non_blocking` copies was implemented,
measured at 10.93 ms against the plain loop unchanged 10.90 ms (bit-identical
output), and therefore NOT kept: the copies, not the stream semantics, are the
bottleneck. One card also already runs at ~850 GB/s, i.e. 91% of an RTX 3090s
peak, so there is nothing left for a second card to take. Practical rule: shard
when the factors do not fit in one cards VRAM; otherwise run ONE card per fit and
spread folds, alphas or methods across cards -- with fold views the folds of a
single fit are cheap to hand out, and that parallelism has no cross-card copy in
its inner loop at all.

### Dense solver controls

* `PHEASY_HOST_FACTOR_CACHE` -- `auto` (default), `0` or `1`: whether a fit keeps
  the canonical host factors (one copy plus the adjoint transpose, ~16.9 GiB on
  the MgC operator) on the operator so a later operator over the same factors
  uploads with no host pass at all. `auto` keeps them only when `MemAvailable` can
  afford twice what they add PLUS an 8 GiB headroom for the rest of the fit, and
  the set is at least 256 MiB -- so a memory-tight host, or one where the factors
  are small, simply rebuilds them instead of risking the run.
* `PHEASY_CGLS_PROBE_START` -- earliest iteration of the CGLS honest probe
  (default `min(8000, max(maxiter // 2, PHEASY_CGLS_VERIFY_EVERY))`). Set it above
  `maxiter` to switch the probe off entirely; it is what turns an unreachable
  tolerance into a reported `stall_floor` instead of a bare `iteration_limit`.
* `PHEASY_CGLS_STALL_POINTS` / `PHEASY_CGLS_VERIFY_EVERY` -- how many consecutive
  honest verifications must fail the 5 % progress test before the criterion counts
  as floored (default 5), and how many iterations apart they run (default 50).
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
* `PHEASY_FISTA_AUTO_FLOOR` -- `0` disables the FISTA measured-floor valve on
  the final refit (default `1`, on). CV folds and the alpha-path walk always run
  with the valve off: there the iteration cap is the design, not a failure.
* `PHEASY_FISTA_STALL_POINTS` -- consecutive KKT checks (20 iterations apart)
  that must fail the `_FISTA_STALL_MARGIN` (5 %) progress test before the
  certificate is declared floored (default 3).
* `PHEASY_FISTA_FLOOR_MAX` -- the largest measured floor that may still be
  certified (default 1e-2). A floor above it is a refusal
  (`stop_reason=stall_above_floor`), not a certificate.
* `PHEASY_FISTA_FLOOR_MARGIN` -- headroom added to the measured floor before
  certifying (default 0.10), because re-deriving the same residual from a
  float32 operator spreads +-2 % (see the CGLS floor note above).


## Measured results, Mg8C120 (2026-09-17, RTX 3090)

Material `userfit_Mg8C120_c2_7.0_c3_4.0` (c2=7.0, c3=4.0, dim 2 2 1, 512 atoms,
296 configurations), effective operator **454,656 x 69,487** at full size, float32
factors. Every number below was produced on that operator with the code in this tree;
host steps vary ~2.7x with box load while fit walls stay within ~3 %.

### Sparsity is a SMALL-DATA regime property, and the acceptance gate depends on it

Coefficient nnz from the same auto grid (6 decades, 12 alphas, cv=3):

| configurations | OLS nnz | LASSO nnz | LASSO alpha* |
| --- | --- | --- | --- |
| 24 | 69,487 (dense) | **11,549 (16.6 %)** | 5.43e-07 |
| 120 | 69,487 | **69,376 (99.8 %)** | 3.65e-10 |
| 296 | 69,487 | **68,955 (99.2 %)** | 2.31e-10 |

So the compressive-sensing claim holds only while the structure set is small; at 296
configurations CV drives alpha to ~1e-10 and LASSO is numerically dense. Grouped
holdout by configuration (train 197 / test 99, resident operator) says the same thing
about generalisation, and the L1 advantage inverts:

| configurations | method | held-out relative L2 | nnz | accepted |
| --- | --- | --- | --- | --- |
| 24 (earlier grouped holdout) | LASSO / ALASSO | ~1.1e-01 / 1.2e-01 | 13,219 / 6,162 | - |
| 24 (earlier grouped holdout) | OLS / RIDGE | ~5.66e-01 | dense | - |
| **296** | **OLS** | **1.4153e-02** | 69,487 | yes |
| **296** | LASSO | **1.7008e-02** | 69,356 | yes |

### Acceptance at production size (production CLI, one card)

| run | wall | accepted | backend | stop_reason | rmse |
| --- | --- | --- | --- | --- | --- |
| OLS @24 | 167 s | yes | gpu_resident_iterative | converged | 8.75e-05 |
| LASSO @24 | 243 s | yes | gpu_twolevel_resident | converged_measured_floor | 3.19e-02 |
| OLS @120 | 371 s | yes | gpu_resident_iterative | precision_floor | 8.40e-03 |
| LASSO @120 | 2723 s | yes | gpu_twolevel_resident | converged_measured_floor | 1.39e-02 |
| OLS @296 | 605 s | yes | gpu_resident_iterative | precision_floor | 9.83e-03 |
| LASSO @296 | 6866 s | yes | gpu_twolevel_resident | converged_measured_floor | 1.58e-02 |
| RIDGE cv=5 @296 (harness, quiet box) | 592 s | yes | gpu_resident_iterative | converged | 1.7691e-02 |
| OLS @296 (harness, quiet box) | 515-597 s | yes | gpu_resident_iterative | precision_floor / converged | 9.80e-03 |

The two LASSO@296 walls above were measured while a second LASSO job shared the card;
the harness run of the same size took 2746 s. The grid matters for the verdict too:
`ALPHA_DECADES=6 NMU=12 CV=3` returns `fit_returned_not_accepted` on 296 configurations
while the production grid (4 decades, 20 alphas, cv=5) is accepted at kkt 2.54e-04, so a
sparsity or accuracy claim has to name both the configuration count and the grid.

### Init chain, measured separately (the reason the adjoint cache exists)

| step | time |
| --- | --- |
| `sp.load_npz(sm_prime.npz)`, 9 GB, warm page cache | 8.6 s |
| canonical `tocsr(copy)+dedupe+sort` | 4.1 s |
| transpose to build the adjoint | **42.2 s** (up to 116.7 s cold/loaded) |
| write the adjoint cache (`.npz`, 8.40 GiB) | 11.6-14.9 s |
| load the adjoint from disk cache | 13.6-48.5 s |

Hence caching the adjoint is worth ~30-100 s per fit (measured fit walls: 597.4 s cold
vs 515.0 s warm) while caching the npz LOAD is not (8.6 s warm). The cache is opt-in
(`PHEASY_HT_CACHE_DIR`, plus `PHEASY_HT_CACHE_KEY`, which `run_pheasy.py` fills from the
sensing-matrix file identity) and costs 8.4 GiB per entry.

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
| `PHEASY_CV_TOL` | `max(tol, 1e-3)` | GPU CV paths run at 1e-3, not the caller's `tol` (the CPU `_LassoCVIterative` path instead uses `tol` itself) |
| `PHEASY_CV_MAX_ITER` | `min(max_iter, 400)` | resident two-level CV cap; the dense GPU `GpuLassoCV` uses `min(max_iter, 800)` and the CPU iterative path uses `max_iter` uncapped — the three backends do NOT share a CV budget, so a cross-backend alpha* comparison must set the env vars explicitly |
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
not a GPU-specific limit.  NOTE (verified by reading the defaults, not by a GPU
run): the CPU `_LassoCVIterative` does NOT share the GPU CV caps -- it uses the
caller's `tol`/`max_iter` (1e-4/20000 at the CLI defaults) while `GpuLassoCV` uses
`max(tol,1e-3)`/`min(max_iter,800)` and `GpuTwoLevelLassoCV` uses
`min(max_iter,400)`.  A CPU-vs-GPU alpha* comparison must set `PHEASY_CV_TOL` and
`PHEASY_CV_MAX_ITER` explicitly on both sides, or the two runs are not solving the
same CV problem. The fixed-alpha scan (`dev/alpha_scan.py`) measured
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
* **The c3=5.0 factors (42.84 GB) cost ~129 GB of host RAM before the 2026-09
  fix; the measured peak is now 66.76 GiB.** Three 42.84 GB copies made that
  peak: the raw `sm_prime.npz`, an ALREADY-CANONICAL `tocsr(copy=True)` copy,
  and an eager `prime.T.tocsr()`. `_canonical_csr_inplace` now keeps the
  loaded object when it is already canonical (`PHEASY_HOST_FACTOR_INPLACE=0`
  restores copying) and `_AdjointBlocks` holds `csr.T` as a scipy CSC VIEW
  with `resident_bytes == 0`, materialising only the per-shard row blocks
  (`PHEASY_TWOLEVEL_ADJOINT=lazy` default; `eager`/`auto` force). Measured on
  the real fit: `[mem] ... RSS=43.48 HWM=45.14 GiB` after the load and
  `HWM=66.76 GiB` after the first shard row+column pair (21.4 GB).
* **Split shards by NONZEROS, not by index count.** `np.linspace` boundaries
  put 65% of the c3=5.0 nonzeros (2.32e9 of 3.57e9) on shard 0, which needed
  **19.00 GiB** of CUDA CSR while the even-split estimate claimed 22.54 GB for a
  whole shard. `_balanced_shard_edges` (default on, `PHEASY_SHARD_BALANCE=0`
  reverts) gives `892432896+892432896` per shard, i.e. **14.63 GiB/card**, and
  the sharded resident operator now pre-flights every shard pair with
  `_cuda_spmv_block_budget` instead of trusting the average shard.
* **Index width is per BLOCK.** `resident_twolevel_estimate` and
  `GpuTwoLevelOperator.upload()` used to read int32-vs-int64 off the GLOBAL
  nnz, so a 3.57e9-nnz float32 fit was pre-flighted at 22.5e9 B/card (real
  16.9e9) and uploaded int64 indices for shards whose blocks hold 892M nonzeros
  (+3.57 GB of VRAM per card).
* **Column indices fit int32 even when indptr does not.** scipy's CSR loader
  derives ONE index dtype from max(nnz, n_rows), so the c3=5.0 sensing matrix
  (nnz 3.57e9, n_cols 265662) carries int64 column indices: 26.6 GiB of index
  buffers where 13.3 GiB suffice, held for the whole fit.  core/sparse_io.py
  streams the npz members and builds `indices int32 + indptr int64` with no
  full-size temporary (the peak is the destination arrays: 26.7 GiB instead of
  the 39.9 GiB scipy load); `PHEASY_SM_INDEX_DTYPE=int64` restores the old
  layout.  Mixing the widths is legal for every kernel the fit uses (matvec,
  slicing, transposes, tocsr, lsmr/lsqr, hstack/vstack, pickle round-trips) but
  `tocoo()`/`nonzero()`/`eliminate_zeros()` raise "Output dtype not compatible
  with inputs" -- they expand indptr through a compiled routine that wants one
  dtype; the GPU COO fallback in load_sensing_matrix therefore expands it with
  `_csr_major_indices()`.
* **scipy's canonical/sorted CHECK is what balloons on the mixed layout.**
  csr_has_sorted_indices()/csr_has_canonical_format() convert the indices to
  indptr's dtype internally, so the first query on an int32-indices/int64-indptr
  matrix allocates a FULL int64 copy of the indices: measured VmHWM 1.40 -> 2.76
  GiB on an 0.68 GiB-index fixture, and 26.8 -> 53.3 GiB on the real c3=5.0
  file.  sparse_io.canonical_format() therefore verifies in chunks (chunk-sized
  temporaries only) and caches the verdict through scipy's public
  has_canonical_format setter, so later queries are free; gpu_backend's
  _canonical_block()/_canonical_csr_inplace() call it instead of the property.
  sparse_io.canonicalize() widens int32 -> int64 before sort/dedupe (the C
  kernels require matching widths).  A NON-canonical stored CSR is refused by
  the loader (auto mode falls back to scipy's int64 load).
* **Real 42.84 GB file, load_csr vs scipy.sparse.load_npz (2026-09-23).**
  data sha256, data/indices/indptr sums and nnz IDENTICAL; buffers 42.84 ->
  28.56 GB (indices int64 -> int32); VmHWM 39.95 -> 26.83 GiB; 282 -> 330 s.
  With the sequential block build the whole host phase should now peak near
  43 GiB instead of the measured 60.11 GiB.
* **Build the row block, upload it, release it, THEN build the adjoint block.**
  Holding both through the upload measured +21.6 GiB on the real c3=5.0 factors
  (two 10.7 GB blocks) on top of the 43.48 GiB factor set, and the budget check
  used to materialise that pair just to measure it.  The per-shard pre-flight now
  counts nonzeros (`_cuda_spmv_block_budget_counts`) instead of weighing the
  blocks, so GpuSparseMV and the resident operator both keep ONE block alive:
  probe at 1/16 scale (two 0.62 GiB blocks) delta HWM 1.74 -> 1.12 GiB (1.55x),
  i.e. the real peak should land near 57 GiB instead of the measured 66.76 GiB.
  `SequentialBlockBuildTest` pins the build/upload call order -- the uploaded
  tensors look identical either way, so only the order catches a regression.

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
