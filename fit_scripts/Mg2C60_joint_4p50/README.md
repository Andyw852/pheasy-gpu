# Mg2C60 joint force-constant fit at c3 = 4.5 A (RTX 3090)

Operational scripts behind the production **RFE-OLS** fit of the Mg2C60 joint
dataset (2x2x2 supercell, 496 atoms, 699 displaced configurations) at a
third-order cutoff of 4.5 A.  Each file here encodes a failure that cost hours of
cluster time; see
[REPORT_gpu_sm_sharding_20261116.md](../../REPORT_gpu_sm_sharding_20261116.md)
for the code-side change they rely on.

**Status: finished 2026-11-19 02:43** (job 7630, 5 cards, stored-transpose
layout).  Selected 51,316 of 77,063 free IFCs, CV_RMSE 7.0384e-3, fit 14:24.

## Size of the problem

| quantity | value |
|---|---|
| sensing matrix `SM_prime` | `(1040112, 92079)`, float32 CSR |
| nonzeros | `4,816,356,048` |
| `sm_prime.npz` on disk | 57.8 GB |
| free force constants (HARM + ANHARM3) | 77,063 (35,032 protected + 42,031 eliminable) |
| RFE rounds / CV folds | 9 rounds, 5 folds |
| columns-norm vector | 77,063 float64 (`colnorms_4p50.npy`, 617 kB) |

## The layout that works: 5 cards, stored transpose

`2 x nnz x 8 B / n_cards` per card (row block + its transpose):

| cards | per card | verdict |
|---|---|---|
| 3 | 25.7 GB = 23.9 GiB + 10% | **impossible** on a 24 GB card (jobs 7619, 7620 died) |
| 4 | 19.3 GB = 18.0 GiB (+10% -> 19.8) | only if four cards are really empty |
| 5 | 15.4 GB = 14.4 GiB (+10% -> 15.8) | **the production run** (job 7630) |

The low-memory layout is **retired**: it does not lower that peak (torch copies
every block inside `R.t() @ u`, ~11.96 GiB at this cutoff) and it OOMed job
7620.  `PHEASY_GPU_SM_LOWMEM=1` now raises at construction.

## Files

| file | role |
|---|---|
| `submit_4p50_resume6.sh` | **the production recipe**: runtime card choice (5 cards >=17 GiB, else 4 >=21.5), prep -> `pheasy -f` -> post, fail-fast guards |
| `submit_colnorm.sh` | SLURM wrapper that precomputes the column norms on 32 CPUs |
| `colnorm_parallel.py` | the parallel column-norm pass (also used for c3=5.0; it narrows the npz column indices to int32) |
| `submit_4p50_stream.sh` | streaming mode (`PHEASY_RFE_GRAM_CV=1`), CPU only or 1 GPU (`STREAM_GPU=1`); see below |
| `submit_4p50_stream_cpu.sh` | streaming mode pinned to CPU with the c3=4.5 sizing (`GRAM_TOTAL=0`, 237.5 GB store); measured verdict below |
| `pick_gpus.py` | standalone helper printing visible GPU indices by measured free VRAM |

Everything else (`fc_fit_driver.py`, `fit_config.json`, `sm_prime.npz`,
`ns_*.npz`, `disp_matrix.pkl` ...) is produced by the fit pipeline itself.

## How to run

Copy the scripts into the case directory and submit from there (`CASE`
defaults to `$SLURM_SUBMIT_DIR`; `CONDA_SH` defaults to
`$HOME/miniconda3/etc/profile.d/conda.sh`).

```bash
# 0) after every -d pass that (re)wrote sm_prime.npz:
sbatch submit_colnorm.sh                  # ~2.7 min with 32 workers (vs ~9 h serial)

# 1) the fit itself
sbatch submit_4p50_resume6.sh             # prep + pheasy -f + post

# 2) inspect
grep -E "SpMV on|RFE. Round|optimality" fit.log
```

With `FCFIT_REUSE_SM=1` + `FCFIT_KEEP_CS=1` a restart after a crash costs
~40 s of probe + SM load instead of `-c` (37 min) + `-d` (4-5 h).

## Environment that matters

| variable | value here | why |
|---|---|---|
| `PHEASY_HARM_DENSE` | `1` | RFE-OLS keeps the 35,032 harmonic columns; they are never pruned |
| `PHEASY_SM_DTYPE` | `float32` | 38.5 GB instead of 77 GB for `SM_prime` |
| `PHEASY_GPU_SM` | `1` | cuSPARSE SpMV for the `SM_prime` half of the two-level operator |
| `PHEASY_GPU_SM_NGPU` / `_DEVICES` | 5 picked cards | explicit list skips the auto filter, which sizes every card for the worst case |
| `PHEASY_GPU_FALLBACK` | **`0`** | a GPU problem must abort, not silently run on one CPU core |
| `PHEASY_COL_NORMS_FILE` | `colnorms_4p50.npy` | skip the in-fit column-norm pass; exported **only** when `sm_prime.npz` is reused |
| `PHEASY_GPU_MEM_FRACTION` | `1.0` | the box is shared; the free-VRAM pre-flight is the real guard |
| `FCFIT_REUSE_SM` | **`1`** | otherwise the driver calls `_clear_pheasy_sm_cache()` before the `-s` probe and deletes `sm_prime.npz` |
| `FCFIT_KEEP_CS` | **`1`** | reuse `cs.pkl`/`ns_*.npz`; `-f` validates their fingerprints, a stale cache aborts |
| `PHEASY_TWOLEVEL_ADJOINT` | `lazy` | no eager 38.5 GB transpose copy |
| `PHEASY_CV_GROUP_SIZE` | `1488` | one displaced configuration (496 atoms x 3) per CV group |

## The rules learned the hard way

1. **Card count != shard count.** The CUDA CSR path stores int32 indices, so a
   row block must stay under `2**31-1` nonzeros. `nnz/2 = 2.41e9` on two cards
   is too big and the fit used to fall back to the CPU. `row_shard_count()`
   raises the block count independently of the cards, but the *footprint* is
   fixed at `2 x nnz x 8 B`: sharding cannot make 5 cards out of 4, and it
   cannot make the low-memory layout cheaper.
2. **`PHEASY_GPU_FALLBACK=0`.** With `1`, any GPU-SM failure degrades to the
   serial CPU matvec: ~1000x slower, no error beyond one line. A 4.5 A fit then
   needs weeks and looks like a slow normal run. `PHEASY_GPU_MODE=required` also
   aborts construction failures.
3. **Never trust SLURM GRES for card selection on a shared box.** Twice the job
   received cards another user's non-SLURM process was holding (16-19 GiB) while
   `squeue` showed them as ours. The script requests `--gres=gpu:5`, measures
   `torch.cuda.mem_get_info()` at start, keeps cards above the threshold and
   exits if too few qualify.
4. **`prep` can invalidate a 57.8 GB `sm_prime.npz` by rewriting
   `disp_matrix.pkl`** (atom-order bookkeeping). The script snapshots its md5
   before and after `prep` and only `touch`es `sm_prime.npz` when the bytes are
   identical. Never `touch` unconditionally: a sensing matrix in a different
   atom order is silently wrong (documented drift 6.9% -> 11%).
   `colnorms_4p50.npy` has the same problem one level down (the loader checks
   only its shape); the script exports `PHEASY_COL_NORMS_FILE` only when the
   matrix is reused and renames stale norms to `.stale`.
5. **The null-space step is the slow one at large cutoffs.** `-c` takes 37 min
   at c3=4.5 and **2:30 h** at c3=5.0. `FCFIT_KEEP_CS=1` is safe because
   `pheasy -f` validates the `cs.pkl`/`ns_*.npz` fingerprints itself, but it
   only works because `_clear_pheasy_cache()` now keeps `ns_*.npz` through the
   mandatory `-s` order probe.

## Streaming mode (0 or 1 GPU) -- measured, not recommended at this cutoff

`submit_4p50_stream.sh` (and its CPU-only variant) builds the five fold Grams
once and solves every RFE round exactly from them.  Measured on 2026-11-17/18
(job 7622, CPU only, `store=disk`, `PHEASY_RFE_GRAM_TOTAL=0`):

| stage | measured |
|---|---|
| Gram accumulation | **9.25 h** (33,307 s; 1.6-2.2 h per fold) |
| round 0 | **>6.65 h and unfinished** when stopped (RSS 194 GB, ~20 cores) |

The accumulation is 3x the design estimate because `PHEASY_CV_GROUP_SIZE=1488`
makes the folds non-contiguous, so the fold-by-fold accumulator needs one full
pass per fold (five passes) instead of "K folds in one pass".  Round 0 looks like
the `_spd_solve_inplace` eigendecomposition fallback (its pre-flight asks
`3 s^2 x 8 B = 142 GB`, matching the 194 GB peak) rather than the 26 min
Cholesky path.  Details and the follow-up list: SPEC_gpu_sm_streaming.md §10.
Use the resident 5-card path above; keep streaming for the no-GPU case and for
c3 <= 4.0.

## Timings measured on this box (RTX 3090, 24 GB, 56 CPU cores)

| stage | wall time | notes |
|---|---|---|
| `prep` | 3-20 s | reuses the pickles |
| `-s` cluster space (x2) | 1:11 each | |
| `-c` symmetry constraints + null space | **37 min** (c3=4.5), 2:30 h (c3=5.0) | skipped by `FCFIT_KEEP_CS=1` when the cache is fresh |
| `-d` sensing matrix rebuild | **3:20-4:40** | skipped when `FCFIT_REUSE_SM=1` lets the cache be reused |
| `-f` RFE-OLS, 14 rounds | **14:24** | GPU SpMV ~30 min of it; Gram/Cholesky host work dominates |
| GPU SpMV upload (5 cards) | 5-10 min | 38.5 GB over PCIe |
| column norms (32 CPU workers) | 2.7 min | |

## Operational gotchas

* **SLURM snapshots the batch script at submit time** - editing a pending job's
  script has no effect; `scancel` + `sbatch` instead.
* **Do not `setsid` the fit child**: SLURM then TERMs the child on job cleanup
  (rc=143). A plain `&` + `wait` is correct here.
* **`-d` progress lines are block-buffered** through the driver's pipe, so the
  only live signal during a rebuild is the child's RSS growth (~83 MB per
  configuration). The same buffering hides `[RFE]` lines inside a running job.
* **One physical card (`84:00.0`) is dead**: `nvidia-smi` fails globally,
  per-card queries via `torch` still work, and other users' jobs appear on cards
  GRES calls free.
* **c3=5.0 does not fit this box** (120.7 GB of stored transposes vs ~117 GB of
  usable VRAM) and its low-memory variant OOMed too; it runs on A800
  (`fit_scripts/Mg2C60_joint_5p00/submit_5p00_a800.sh`, 3 cards).
