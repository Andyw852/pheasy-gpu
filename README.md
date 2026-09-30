# pheasy-gpu

GPU-accelerated (CUDA / PyTorch) edition of pheasy. Same fitting methods
and control flow as pheasy; the heavy dense linear algebra (OLS, RIDGE, RFE,
sensing-matrix loading) runs on the GPU **when the input is dense**. On a
matrix-free `TwoLevelSM` input -- the production path for large cells -- the
resident OLS is reachable under the default configuration: Jacobi is applied on
the resident operator itself (the column norms become the operator's `scale`,
so CGLS solves the same scaled system the CPU path builds) and only a ridge
request (`PHEASY_OLS_RIDGE>0`) falls back to CPU LSMR. Note that GPU-resident
TwoLevel OLS still measured *slower* than CPU at the sizes tried. See
[`GPU.md`](GPU.md) for activation, the per-row env of every measured benchmark,
and correctness checks.

This is a separate package named `pheasy_gpu` and a separate console command
`pheasy-gpu`, so it can be installed alongside the original `pheasy` without
touching it.

Force-constant extraction from finite-displacement / AIMD data.

## Install

```bash
pip install -e .          # exposes the `pheasy-gpu` command
pip install -e '.[gpu]'   # + torch (CUDA), the GPU backend
pip install -e '.[fast]'  # + celer, a faster LASSO solver (CPU path)
```

## Verify the install

```bash
python dev/smoke_installed.py                      # ambient interpreter
/path/to/venv/bin/python dev/smoke_installed.py    # explicit env
pheasy-gpu --help                                  # console entry point
```

A clean, reproducible base install (no GPU extra) is:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/python dev/smoke_installed.py
```

For CUDA, add the `[gpu]` extra: `.venv/bin/python -m pip install -e '.[gpu]'`.
Each fit writes a `fit_manifest.json` recording the method, resolved GPU
environment variables, dependency versions, backend, convergence diagnostics and
`fit_accepted`/`status`; an unaccepted fit refuses to write force constants
unless `PHEASY_ALLOW_UNACCEPTED_FIT=1` is set.

## Fitting methods (`-l`)

| flag | method |
|---|---|
| `OLS` | ordinary least squares (LSMR / SVD) |
| `LASSO` | L1 with cross-validated alpha, then debias refit |
| `ALASSO` | adaptive LASSO (Zou 2006) |
| `RFE` | recursive feature elimination, OLS base, grouped CV |
| `RFE-OLS-TSQR` | RFE with a Q-less tall-skinny QR base solver |
| `RIDGE` | L2 with cross-validated alpha |

`PHEASY_HARM_DENSE=1` (or `HARM_DENSE=true` in `pheasy_fit.sh`) makes the
sparse methods (LASSO, ALASSO, RFE, RFE-OLS-TSQR, ARDR, RVM) act on the
anharmonic block only: the harmonic (FC2) columns carry no L1 penalty / a flat
prior / are never eliminated, and are refitted jointly.  The alpha grid is then
anchored at the anharmonic KKT threshold on the residual of the FC2-only OLS.
On the Gram paths (CPU and GPU dense FISTA) the harmonic block is eliminated
exactly (Schur complement) and the anharmonic block is Jacobi-scaled before
FISTA runs, and the matrix-free resident path iterates in Jacobi-scaled
coordinates; without this, raw FC2/FC3 column scales (~40x apart) make FISTA
crawl while its relative KKT already looks converged.  The reduced solves take
their step from an upper bound on lambda_max of the reduced Gram (CPU: Lanczos
above `PHEASY_LIPSCHITZ_LANCZOS_P`, default 1024 penalized columns, exact
eigvalsh below), and lambda_max of the unreduced Grams is no longer computed.
With `--fix_fc2` the harmonic block is not fitted at all and the flag is moot.
See `dev/test_harm_dense.py` for the contract.

## Cross-validation on large fits (alpha-resolved CV)

On large, overdetermined fits the LASSO/ALASSO CV used to put alpha* at the
bottom of *every* auto grid ("effectively unregularized"). That was an
optimiser artefact, not a property of the data. The FISTA CV solves stopped
on a relative KKT residual (`PHEASY_CV_TOL`, 1e-3 on the GPU paths) that is far
above the L1 penalty of the small alphas (`alpha/alpha_max` = 1e-6 at the
bottom of a 6-decade grid; float32 cannot even certify below ~1e-4). The CV
curve therefore followed the accumulated iteration count down the
warm-started path, not alpha. All FISTA backends (resident GPU, GPU Gram,
CPU) now:

* stop each (alpha, fold) solve at `min(tol, rho * alpha*n*q / max|A^T y|)`, a
  fraction `rho` of that alpha's own penalty level;
* count an alpha as *resolved* only when its KKT residual is below the
  penalty level, and exclude unresolved alphas from the selection;
* measure the alpha -> 0 end exactly (per-fold least squares). If that OLS
  limit cross-validates at least as well as every resolved alpha, the fit is
  returned as OLS with `alpha_opt = 0` and `cv_selected_ols_limit = true`.
  This is a certified conclusion (the data support no L1 penalty), not a
  pinned grid edge. `PHEASY_LASSO_1SE=1` picks the sparsest resolved alpha
  within one standard error instead.
* if the CV curve is still falling at the bottom of the grid, extend the grid
  downward (same log density, every fold warm-starts from its last grid
  solution) until the minimum is bracketed, the curve meets the OLS limit, or
  `PHEASY_CV_EXTEND_DECADES` is spent. An alpha* below the original grid that
  beats OLS is a regularized optimum; it is no longer reported as "effectively
  unregularized" and no longer fails the fit-quality gate. All four backends
  (GPU resident, GPU Gram, CPU FISTA, sklearn coordinate descent) do this.

The resident GPU path also iterates in Jacobi-scaled coordinates for
unstandardized fits (exact reparametrization).

CV folds are now contiguous blocks of configurations. GroupKFold dealt
configurations out round-robin, which puts the trajectory neighbours of every
validation frame into training: for MD-derived datasets that is a leak that
biases alpha*/ridge toward 0.

| variable | default | meaning |
|---|---|---|
| `PHEASY_CV_ALPHA_AWARE` | `1` | `0` restores the old fixed CV tolerance |
| `PHEASY_CV_ALPHA_TOL_RATIO` | `0.1` | `rho`: KKT residual as a fraction of the L1 penalty level |
| `PHEASY_CV_UNRESOLVED_STOP` | `2` | a fold stops descending after this many unresolved alphas in a row |
| `PHEASY_CV_OLS_REFERENCE` | `1` | compare the path with its exact OLS limit (auto grids only) |
| `PHEASY_CV_OLS_MAX_ITER` | `max(5*cv_cap, 2000)` | CGLS budget of the resident per-fold OLS reference |
| `PHEASY_CV_FOLD_MODE` | `contiguous` | `interleaved` = legacy sklearn GroupKFold |
| `PHEASY_CV_EXTEND_DECADES` | `6` | max decades the LASSO/ALASSO grid is extended below its minimum (`0` = off) |
| `PHEASY_CV_EXTEND_STEP` | `2` | decades added per extension round |
| `PHEASY_RESIDENT_JACOBI` | `1` | Jacobi-scaled resident FISTA for unstandardized fits |
| `PHEASY_RIDGE_ALPHA_AUTO` | `1` | RIDGE grid also spans `[lambda_max*10^-PHEASY_RIDGE_DECADES, lambda_max]` of A^T A |
| `PHEASY_RIDGE_DECADES` / `_PER_DECADE` / `_NMAX` | `8` / `2.5` / `40` | RIDGE auto-grid span / density / cap |
| `PHEASY_ARD_STD` | `unit_variance` | ARDR/RVM `--std` convention (`unit_norm` = legacy) |
| `PHEASY_RFE_JACOBI` | `1` for operators | Jacobi-preconditioned RFE subset solves |

Other method fixes in the same change:

* **RIDGE:** the fixed `--mu_min/--mu_max` grid (1e-6..1e-2) sits below the
  whole spectrum of `A^T A` on a large fit, so every alpha returned OLS. With
  `--alpha_auto` the grid is widened to the spectrum, and a grid-edge alpha* is
  reported.
* **ARDR/RVM:** `--std` scaled columns to unit L2 norm, which raises the
  absolute pruning threshold `lambda_t = 1e4` by a factor of n (the number of
  rows), so large fits pruned almost nothing. They now use unit variance, as
  in hiphive and Fransson et al. (2020).
* **RFE:** Jacobi scaling is now on by default for matrix-free input.

Memory and GPU placement:

* **GPU LASSO/ALASSO (dense Gram, `gpu_dense_fista`):** the CV now runs fold by
  fold, so only one fold's training Gram, validation rows and (with
  `HARM_DENSE`) Schur reduction are on the device at a time. Previously all K
  folds, plus a second copy of A, were resident together. A LASSO-specific VRAM
  pre-flight that counts the Grams replaces the `4*n*p` estimate. Under
  `PHEASY_GPU_MODE=required` it fails fast; otherwise it falls back to the CPU
  path, where a mid-fit CUDA OOM used to happen.
* **Host copies:** dense GPU solves no longer widen a float32 sensing matrix to a
  float64 host copy before upload; they widen on the device.
  `GpuRidgeCV` no longer pre-slices every fold as float64 host copies.
* **RIDGE CV:** validation predictions go through the cached resident operator
  on the GPU. With `--std` they used to be a full host SpMV per (alpha, fold).
* **Resident LASSO:** the debias residuals are computed on the GPU. The post-fit
  prediction is cached, so `run_pheasy`'s alignment gate does not repeat a host
  SpMV. The retained operator is always released.
* **RFE final exact refit:** accumulates `A^T A` in place with `dsyrk`, removing
  an n x n temporary (7 GB at 30k features). It stays on the CPU on purpose,
  because consumer GPUs run float64 at 1/64 of their float32 rate.

## Typical workflow

```bash
python3 tools/prepare_dataset.py SPOSCAR dataset_disps.npy dataset_forces.npy
pheasy-gpu --dim 3 3 3 -w 3 -s --c3 5.2
pheasy-gpu --dim 3 3 3 -w 3 -c --c3 5.2
pheasy-gpu --dim 3 3 3 -w 3 -d --c3 5.2 --ndata 45 --disp_file
pheasy-gpu --dim 3 3 3 -w 3 -f --c3 5.2 --ndata 45 -l OLS --full_ifc --hdf5
```

## Notes

- The `pheasy-gpu` CLI has no `--use-gpu` flag: GPU activation is via the
  `PHEASY_GPU_MODE=auto|cpu|required` (`auto` is the default). Legacy
  `PHEASY_USE_GPU=1/0` maps to `required/cpu`. `required` is the production
  default: every supported main solve (OLS/RIDGE/LASSO/ALASSO/RFE) runs on the
  GPU by default, and any GPU runtime failure raises instead of silently
  returning a CPU result; a per-instance `use_gpu=False` still selects CPU
  explicitly. See `GPU.md`. `PHEASY_GPU_FALLBACK=0` (now the default) makes an
  enabled GPU sparse-matvec path fail closed on any mid-fit CUDA error; set
  `PHEASY_GPU_FALLBACK=1` to restore the legacy CPU continuation.
- LASSO / ALASSO need a tight tolerance. `--tol 1e-3` is *not* tight: sklearn
  scales it by `||y||^2`, coordinate descent stops early at small alpha, the CV
  curve goes flat and the fit ends up over-regularized. Use `--tol 1e-6`.
- `PHEASY_SM_DTYPE` (`float64` default) controls the precision of SM / NS / FM.
  All of them must agree, otherwise scipy silently upcasts.
- `tools/prepare_dataset.py` is a repository utility (not installed by pip);
  run it from the checkout, not from the installed package.
