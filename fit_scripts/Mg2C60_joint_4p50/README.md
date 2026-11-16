# Mg2C60 joint force-constant fit at c3 = 4.5 A (RTX 3090)

Operational scripts behind one production **RFE-OLS** fit of the Mg2C60 joint
dataset (2x2x2 supercell, 496 atoms, 699 displaced configurations) at a
third-order cutoff of 4.5 A, plus the two helper passes that make it feasible on
consumer cards. They are archived here because every one of them encodes a
failure that cost hours of cluster time; see
[REPORT_gpu_sm_sharding_20261116.md](../../REPORT_gpu_sm_sharding_20261116.md)
for the code-side change they rely on.

## Size of the problem

| quantity | value |
|---|---|
| sensing matrix `SM_prime` | `(1040112, 92079)`, float32 CSR |
| nonzeros | `4,816,356,048` |
| `sm_prime.npz` on disk | 57.8 GB |
| free force constants (HARM + ANHARM3) | 77,063 (35,032 protected + 42,031 eliminable) |
| RFE rounds / CV folds | 9 rounds, 5 folds |
| columns-norm vector | 77,063 float64 (`colnorms_4p50.npy`, 617 kB) |

## Files

| file | role |
|---|---|
| `submit_4p50_resume5.sh` | the fit job: prep -> `pheasy -f` -> post, with GPU card selection and fail-fast guards |
| `pick_gpus.py` | standalone helper: prints the visible GPU indices ordered by **measured** free VRAM (freest first), for setups where the resident budget comes from `device[0]`. `submit_4p50_resume5.sh` does **not** call it; it has its own inline selector with the 15 GiB threshold |
| `submit_colnorm.sh` | SLURM wrapper that precomputes the column norms on 32 CPUs |
| `submit_4p50_stream.sh` | the same fit in **streaming mode** (`PHEASY_RFE_GRAM_CV=1`): no resident SpMV, CPU only or 1 GPU (`STREAM_GPU=1`); see "Streaming mode" below. Not yet run on the cluster |
| `colnorm_parallel.py` | the actual parallel column-norm pass. Same per-block float64 arithmetic as the serial definition, but the blocks are summed in completion order (`imap_unordered`), so it matches to float64 rounding, not bit for bit |

Everything else (`fc_fit_driver.py`, `fit_config.json`, `sm_prime.npz`,
`ns_*.npz`, `disp_matrix.pkl` ...) is produced by the fit pipeline itself and
is not archived here.

## How to run

Copy the scripts into the case directory and submit from there (`CASE`
defaults to `$SLURM_SUBMIT_DIR`; `CONDA_SH` defaults to
`$HOME/miniconda3/etc/profile.d/conda.sh`).

```bash
# 0) after every -d pass that (re)wrote sm_prime.npz:
sbatch submit_colnorm.sh                  # ~2.7 min with 32 workers (vs ~9 h serial)
                                          # -> colnorms_4p50.npy

# 1) the fit itself
sbatch submit_4p50_resume5.sh             # prep + pheasy -f + post

# 2) inspect
grep -E "SpMV on|RFE. Round|optimality" fit.log
```

The script reuses an existing `sm_prime.npz` instead of rebuilding it (see rule 4
below), so a restart after a crash costs ~40 min (`-c`) instead of ~5 h.

## Environment that matters

| variable | value here | why |
|---|---|---|
| `PHEASY_HARM_DENSE` | `1` | RFE-OLS keeps the 35,032 harmonic columns; they are never pruned |
| `PHEASY_SM_DTYPE` | `float32` | 38.5 GB instead of 77 GB for `SM_prime` |
| `PHEASY_GPU_SM` | `1` | cuSPARSE SpMV for the `SM_prime` half of the two-level operator |
| `PHEASY_GPU_SM_LOWMEM` | `1` | row blocks only; the adjoint is the zero-copy CSC view (half the VRAM) |
| `PHEASY_GPU_SM_NGPU` | `3` | see rule 1: at this cutoff >=3 cards keeps one block per card under 2^31 |
| `PHEASY_GPU_SM_DEVICES` | the 3 picked cards | explicit list: skips `GpuSparseMV`'s auto filter, which sizes cards for a stored transpose (2 x SM / n_gpu) and is too strict for the low-mem layout |
| `PHEASY_GPU_FALLBACK` | **`0`** | rule 2: a GPU problem must abort, not silently run on one CPU core |
| `PHEASY_COL_NORMS_FILE` | `colnorms_4p50.npy` | skip the in-fit column-norm pass; exported **only** when `sm_prime.npz` is reused (rule 4) |
| `PHEASY_GPU_MEM_FRACTION` | `1.0` | the box is shared; the free-VRAM pre-flight is the real guard |
| `FCFIT_REUSE_SM` | **`1`** | without it the driver calls `_clear_pheasy_sm_cache()` **before** the `-s` probe and deletes `sm_prime.npz`, so the md5/touch guard never applies and `-d` runs again (rule 4) |
| `PHEASY_TWOLEVEL_ADJOINT` | `lazy` | no eager 38.5 GB transpose copy |
| `PHEASY_CV_GROUP_SIZE` | `1488` | one displaced configuration (496 atoms x 3) per CV group |

## The four rules learned the hard way

1. **Card count != shard count.** The CUDA CSR path stores int32 indices, so a
   row block must stay under `2**31-1` nonzeros. `nnz/2 = 2.41e9` on two cards
   is too big and the fit used to fall back to the CPU. With the
   `row_shard_count()` fix, 2 cards get 4 shards of 1.20e9; at this cutoff 3
   cards already give 1.60e9 per block, i.e. the simple one-block-per-card
   layout works. Memory is unchanged either way (same nonzeros, regrouped).
   Two 24 GB cards are only enough with `PHEASY_GPU_SM_LOWMEM=1` (19.7 GiB per
   card; the stored-transpose layout doubles it) **and** an explicit
   `PHEASY_GPU_SM_DEVICES` (the auto filter wants 35.9 GiB per card at 2
   cards). `submit_4p50_resume5.sh` always takes 3 cards, so it runs the
   one-block-per-card path, not the new one; validate that with
   `dev/check_gpu_sm_shards.py` first.
2. **`PHEASY_GPU_FALLBACK=0`.** With `1`, any GPU-SM construction failure
   degrades to the serial CPU matvec: ~1000x slower, no error, no warning beyond
   one line. A 4.5 A fit then needs weeks and looks like a slow normal run.
3. **Never trust SLURM GRES for card selection on a shared box.** Twice the job
   received cards that another user's non-SLURM process was already holding
   (16-19 GiB each) while `squeue` showed them as allocated to us. The script
   therefore requests `--gres=gpu:5`, measures `torch.cuda.mem_get_info()` at
   start, keeps only cards with `>=15 GiB` free, and exits if fewer than 3
   qualify. Required headroom per card (low-mem layout: 4 B value + 4 B int32
   index per nonzero) is `1.1 * (nnz / n_cards) * 8 B` -- per **card**, not per
   shard: a card holding k shards needs k times the per-shard figure. Here
   that is 13.2 GiB at 3 cards (hence the 15 GiB threshold) and 19.7 GiB at
   2 cards; the stored-transpose layout needs twice as much.
4. **`prep` can invalidate a 57.8 GB `sm_prime.npz` by rewriting
   `disp_matrix.pkl`** (atom-order bookkeeping), after which the driver
   correctly refuses the cache and rebuilds (\~4-5 h). The script therefore
   snapshots the md5 of `disp_matrix.pkl` before and after `prep`, and only
   `touch`es `sm_prime.npz` when the bytes are identical. Never bypass this by
   `touch`ing unconditionally: a sensing matrix in a different atom order is
   silently wrong (documented drift 6.9% -> 11% in an earlier job).
   **The guard is only reachable with `FCFIT_REUSE_SM=1`**: without it the
   driver wipes `sm_prime.npz` before the `-s` probe, which is exactly what
   happened to the 2026-11-16 production run (job 7619 rebuilt at 20:34 after
   the earlier job had already written the matrix at 19:33).
   `colnorms_4p50.npy` has the same problem one level down: the loader checks
   only its shape, so norms of an old matrix would be accepted. The script
   exports `PHEASY_COL_NORMS_FILE` only when `sm_prime.npz` is reused, and
   moves the file to `colnorms_4p50.npy.stale` when a rebuild is coming;
   rerun `submit_colnorm.sh` after the rebuild.

## Streaming mode (0 or 1 GPU)

`submit_4p50_stream.sh` runs the same prep -> fit -> post with
`PHEASY_RFE_GRAM_CV=1` (GPU.md "Streaming exact RFE-OLS").  The fit builds the
five fold Grams of the 77,063 features once, then solves every RFE round
exactly from them.  None of the four rules above about the resident operator
applies: there are no shards and no `GpuSparseMV`, and the column norms come
from the Grams.  What it needs instead:

* **disk**: ~242 GB free under `PHEASY_RFE_GRAM_DIR` (default
  `$CASE/rfe_gram_store`) for five 47.5 GB float64 fold Grams; the files are
  deleted when the fit ends (`PHEASY_RFE_GRAM_KEEP=1` keeps them);
* **RAM**: `--mem=200G`.  The engine plans with 0.8 of the RAM available when it
  starts and prints the plan; a plan that does not fit raises before allocating;
* **time** (estimated, not measured): ~3 h of accumulation on the host
  (`STREAM_GPU=1` moves it to the freest card) plus ~2.2 h of round
  Cholesky, instead of 10-14 h on 3 resident cards.

Its rounds are exact least squares, so CV_RMSE will **not** match the
resident run's LSMR iterates to 1e-4: those stop at the 5000-iteration cap.
Compare supports and kappa, and treat the streaming run as the reference.

## Timings measured on this box (RTX 3090, 24 GB, 56 CPU cores)

| stage | wall time | notes |
|---|---|---|
| `prep` | 3-20 s | reuses the pickles |
| `-s` cluster space (x2) | 1:11 each | |
| `-c` symmetry constraints + null space | **37 min** | always recomputed; only `-f` reads the `ns_*.npz` cache |
| `-d` sensing matrix rebuild | **3:20-4:40** | skipped when rule 4 lets the cache be reused |
| `-f` RFE-OLS, 9 rounds | **10-14 h** | GPU SpMV is only ~30 min of it; the Gram/Cholesky host work dominates |
| GPU SpMV upload (3 cards, low-mem) | 2-4 min | 38.5 GB over PCIe |
| column norms (32 CPU workers) | 2.7 min | |

## Operational gotchas

* **SLURM snapshots the batch script at submit time** - editing a pending job's
  script has no effect; `scancel` + `sbatch` instead.
* **Do not `setsid` the fit child**: SLURM then TERMs the child on job cleanup
  (rc=143). A plain `&` + `wait` is correct here.
* **`-d` progress lines are block-buffered** through the driver's pipe: the
  `Sensing matrix: N/699` lines appear in one batch, so during the rebuild the
  only live signal is the child's RSS growth (~83 MB per configuration).
* **Two GPU cards of this host can be dead or stolen**: physical `84:00.0` is
  broken (`nvidia-smi` fails globally, per-card queries via `torch` still
  work), and other users' jobs appear on cards GRES calls free.
* The `--gres=gpu:5` request in `submit_4p50_resume5.sh` is deliberate: it makes
  all five cards visible so the script can pick the three genuinely free ones.
  It does not mean five cards are used.
