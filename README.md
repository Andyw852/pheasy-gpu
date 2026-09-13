# pheasy-gpu

GPU-accelerated (CUDA / PyTorch) edition of pheasy. Same fitting methods
and control flow as pheasy; the heavy dense linear algebra (OLS, RIDGE, RFE,
sensing-matrix loading) runs on the GPU **when the input is dense**. On a
matrix-free `TwoLevelSM` input -- the production path for large cells --
the default configuration does *not* reach the GPU for OLS: Jacobi defaults ON
for matrix-free input and the resident OLS branch implements neither Jacobi nor
a ridge, so it skips itself and the fit lands on CPU LSMR. Set
`PHEASY_OLS_JACOBI=0` (together with `PHEASY_GPU_OLS_RESIDENT=1`) to reach it,
and note that GPU-resident TwoLevel OLS measured *slower* than CPU at the sizes
tried. See [`GPU.md`](GPU.md) for activation, the per-row env of every measured
benchmark, and correctness checks.

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
