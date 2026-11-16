# GPU-SM row split beyond the card count (int32 CUDA CSR limit)

Date: 2026-11-16
Scope: `core/gpu_backend.py` (`GpuSparseMV`), `dev/test_gpu_memory_regressions.py`;
follow-up: `core/optimizer.py` (required mode at construction),
`dev/check_gpu_sm_shards.py` (real-GPU check) -- see section 9
Status: implemented, CPU-only regression tests pass; **no production run yet with
more shards than cards** (see "Validation status" at the end).

---

## 1. Symptom

Two different failures of the same fit (Mg2C60 joint RFE-OLS, 2x2x2 supercell,
496 atoms, 699 configurations, `--c3 4.5`). The sensing matrix is
`SM_prime (1040112 x 92079)`, `nnz = 4,816,356,048`, float32.

**(a) 2 GPUs -- int32 refusal, then a silent CPU downgrade**

```
[GPU-SM] SpMV unavailable (CUDA CSR block shape=(520046, 92079)
         nnz=2408182532 exceeds the int32 index range; use more GPU blocks or
         the CPU sparse path); using CPU
```

The fit then ran the TwoLevelSM matvec on **one CPU core** (`pcpu = 106%`) and
produced no RFE round in 4.6 h, while the same stage takes ~1 h/round at c3=4.0
with GPU SpMV. Estimated time to finish on that path: weeks.

**(b) 4 GPUs -- memory pre-flight refusal, then the same silent downgrade**

```
[GPU-SM] SpMV unavailable (CUDA device 2: actual CSR row/transpose blocks need
         9.87 GiB including workspace, only 4.92 GiB usable; nonzeros may be
         unevenly distributed across devices. Use more/free GPUs or the CPU
         sparse path); using CPU
```

Here the cause was not pheasy: two of the four cards SLURM had handed out were
actually occupied by another user's non-SLURM processes (16-19 GiB each), so the
per-card pre-flight in `GpuSparseMV.__init__` correctly refused. (SLURM GRES on
this shared box does not see non-SLURM processes.)

The dangerous part of both cases is that the downgrade is **silent**: the fit
keeps running and merely becomes ~1000x slower. `PHEASY_GPU_MODE=required` does
not stop it because the fallback is gated on a *different* knob,
`PHEASY_GPU_FALLBACK` (default `0` = raise; the production script had set it to
`1`). See section 6. Only the construction branch had this gap: `_matvec` /
`_rmatvec` already raised under required mode. Fixed in the follow-up (section 9):
`TwoLevelSM.__init__` now aborts when the mode is required, whatever
`PHEASY_GPU_FALLBACK` says.

---

## 2. Root cause

`GpuSparseMV` splits `SM_prime` into one row block **per card**:

```python
G = len(devs)
self._rs = _row_shard_edges(sm_prime, G)      # G shards
self._cs = _col_shard_edges(sm_prime, G)
```

`_check_cuda_csr_indices()` then rejects a block with `nnz > 2**31 - 1`, because
the CUDA CSR path stores both row pointers and column indices as **int32**
(half the VRAM of int64, and the reason a 38.5 GB float32 matrix fits on
consumer cards at all). With 2 cards each block carries `nnz/2 = 2.41e9`
nonzeros, i.e. above the limit -> refusal.

The card count and the required shard count were conflated. They are different
quantities:

* shard count is bounded below by the int32 representation limit;
* card count is bounded by hardware.

---

## 3. Change

```python
_CUDA_INT32_LIMIT = 2 ** 31 - 1
_CUDA_SHARD_SAFE_NNZ = int(_CUDA_INT32_LIMIT * 0.90)   # 10% headroom

def row_shard_count(nnz, n_dev):
    forced = os.environ.get("PHEASY_GPU_SM_SHARDS", "").strip()
    if forced:
        return max(1, int(forced))
    n_dev = max(1, int(n_dev))
    need = -(-int(nnz) // _CUDA_SHARD_SAFE_NNZ)        # ceil
    need = max(need, n_dev)
    return int(np.ceil(need / float(n_dev))) * n_dev   # multiple of n_dev
```

`GpuSparseMV` now builds `n_shards = row_shard_count(nnz, len(devs))` blocks and
maps them round-robin onto the cards:

```python
self._n_shards   = n_shards
self._block_devs = [devs[i % G] for i in range(n_shards)]
self._rs = _row_shard_edges(sm_prime, n_shards)
self._cs = _col_shard_edges(sm_prime, n_shards)
```

and the three places that used `self._devs[i]` (upload loop, `_mv_blocks`,
`_rmatvec_from_row_blocks`) now use `self._block_devs[i]`.

Why this is safe without touching the kernels:

* `matvec` concatenates per-block results **in row order**, which is independent
  of which card produced them;
* `rmatvec` either concatenates the stored transpose blocks in column order
  (default layout) or, with `PHEASY_GPU_SM_LOWMEM=1`, **sums**
  `R_i.t() @ u[r0:r1]` over blocks -- placement-independent either way;
* the per-block memory pre-flight still runs before each upload and re-reads the
  card's free VRAM, so a second block on the same card sees the already-reduced
  headroom;
* nothing about the numerics changes: the same rows, the same values, the same
  float32 CSR, only more, smaller kernel launches.

### 3.1 The 10% headroom

`_balanced_shard_edges()` equalises nonzeros per shard, but boundaries fall on
row indices, so a shard can end up slightly above `nnz/n_shards` -- by at most
one row (<= 92,079 nonzeros here; one column, <= 1,040,112, for the transpose
blocks). 0.90 (about 2.1e8 nonzeros of slack) is therefore generous for the
balanced edges. It does **not** cover `PHEASY_SHARD_BALANCE=0`, where linspace
edges put 65% of the c3=5.0 nonzeros on shard 0 (GPU.md); in that case
`_check_cuda_csr_indices()` still refuses the block rather than wrapping.

### 3.2 Shard counts

```
n_shards(nnz, n_dev)                          n_dev=1  2   3   4   5
c3=4.5 Mg2C60 joint (1040112 x 92079), 4.816e9      3  4   3   4   5
c3=5.0 (454656 x 265662), 3.570e9 [GPU.md]          2  2   3   4   5
hypothetical 6.0e9                                  4  4   6   4   5
```

Note `n_dev=3 -> 3` at c3=4.5: 1.605e9 per block is already under the limit, so
no extra shards are added; `n_dev=2 -> 4` (two blocks per card). At 6.0e9,
`n_dev=3` gives **6** shards (the minimum of 4 rounded up to a multiple of 3).
The measured c3=5.0 matrix (a smaller dataset than the 699-configuration joint
set) fits one block per card from 2 cards up. The first version of this table
gave a c3=5.0 row of `4 4 3 4 5` for "nnz~6.0e9": that figure has no measured
source in the repository, and its `n_dev=3` cell was wrong. All three rows are
now asserted in `test_row_shard_count_covers_the_int32_index_limit`. Small
matrices keep exactly one block per card, i.e. the previous behaviour.

### 3.3 Per-card footprint is unchanged

Total resident nonzeros do not depend on the split, only their grouping does.
With the low-memory layout (row blocks only; the adjoint is the zero-copy CSC
view of the same buffers) a card holding `k` shards needs

```
k * (nnz/n_shards) * (4 B values + 4 B int32 indices) * 1.10 (workspace)
```

For c3=4.5 on 2 cards: `2 * 1.204e9 * 8 B = 19.27 GB` = 17.9 GiB, +10% = 19.7 GiB
against 23.18 GiB free on a 24 GB RTX 3090. On 3 cards: 12.8 GB (12.0 GiB)
resident per card, 13.2 GiB with the workspace -- comfortable.

Both figures are for the low-mem layout. The default stored-transpose layout
holds a column block next to each row block and needs twice as much (about
39 GiB per card on 2 cards -- does not fit). And the 2-card case needs an
explicit `PHEASY_GPU_SM_DEVICES`: without it `GpuSparseMV._pick_devices()`
filters cards by `2 * SM / n_gpu` (it assumes the stored transpose), i.e.
35.9 GiB per card at 2 cards, so no 24 GB card qualifies and construction fails
with "no usable CUDA device" before the sharding code runs.

---

## 4. Backward compatibility

When `nnz / n_dev < 0.9 * (2**31 - 1)` the helper returns `n_dev`, so
`_block_devs == devs` and every loop degenerates to the previous code. The
production paths already validated before this change (c3=4.0 on 5 cards,
c3=4.5 on 3 cards) therefore take the identical route.

The class attribute `self._devs` is kept (callers and `[GPU-SM] SpMV on %d
device(s)` use it); `self._n_shards` / `self._block_devs` are new.

---

## 5. Tests

`dev/test_gpu_memory_regressions.py` -- CPU-only, scipy-backed fake torch, no
CUDA needed. 11 tests, all pass (`python3 dev/test_gpu_memory_regressions.py`);
10 in the original commit, plus `test_real_gpu_check_script_logic` from the
follow-up.

New:

* `test_row_shard_count_covers_the_int32_index_limit` -- every cell of the
  table in 3.2, the multiple-of-`n_dev` invariant, the *average* shard under
  the 0.9 target (actual edges are not built here), the small-matrix case, and
  the `PHEASY_GPU_SM_SHARDS` override.
* `test_more_shards_than_cards_match_cpu` -- 2 cards, 4 forced shards: asserts
  `_block_devs == [0, 1, 0, 1]`, 4 blocks, and `matvec`/`rmatvec` equal to scipy
  `A @ x` / `A.T @ u` to 1e-13.
* `test_low_mem_more_shards_than_cards_match_cpu` -- same with
  `PHEASY_GPU_SM_LOWMEM=1` (asserts no stored transpose, `_T == []`).

The fake torch's `_SparseTensor` gained `.t()` (mirrors torch's zero-copy CSC
view) so the low-mem adjoint is exercisable without CUDA.

Also run: `dev/test_resident_footprint_estimate.py` (11) and
`dev/test_colnorm_failfast.py` (7) -- 18 tests, all pass. (The first version of
this report said 28; that was the total including the 10 above.)

Follow-up: `dev/test_gpu_failure_semantics.py` gained
`ConstructionFailurePolicyTest` (3 tests; 6 in the file): a `GpuSparseMV` that
raises inside `TwoLevelSM.__init__` must abort under `PHEASY_GPU_MODE=required`
or `PHEASY_USE_GPU=1` even with `PHEASY_GPU_FALLBACK=1`, abort with
`PHEASY_GPU_FALLBACK=0`, and fall back to the CPU only for an explicit
fallback outside required mode. The required-mode test fails on the previous
`optimizer.py`. Run it from the repository root with `PYTHONPATH=.`.

---

## 6. Operational guidance (not code, but learned the hard way here)

1. **`PHEASY_GPU_FALLBACK=0` in production.** With `1`, any GPU-SM construction
   failure silently reverts to the serial CPU matvec. For c3=4.5 that is weeks of
   single-core time and it looks like a normal (slow) run in the log. With `0`
   the fit aborts with the actual reason.
2. **Do not trust SLURM GRES for card selection on a shared box.** Two cards
   allocated to a job were occupied by another user's non-SLURM `campaign.py`
   (16-19 GiB each). Measure `torch.cuda.mem_get_info()` at job start, drop cards
   below the required per-card footprint, and abort (or wait) when too few
   remain. The requirement per **card** (low-mem layout) is
   `free_i >= 1.1 * (nnz/n_cards) * 8 B`, i.e. `k * 1.1 * (nnz/n_shards) * 8 B`
   for a card holding `k` shards (twice that with the stored transpose). The
   first version wrote `nnz/n_shards`, which is the per-shard figure and
   understates the 2-card / 4-shard need by 2x (9.9 vs 19.7 GiB).
3. **Shard count for a given card count** can be read off `row_shard_count`
   (table in 3.2). To stay on the one-block-per-card path: c3=4.5 >= 3 cards;
   the measured c3=5.0 matrix (3.57e9) >= 2 cards; a 6.0e9 matrix >= 4 cards
   (3 cards would give 6 shards).
4. Existing scripts that printed `[GPU-SM] SpMV on N device(s)` still work; with
   more shards than cards the log line reports the *card* count, and the
   `_mem_trace` line reports `<shard>/<n_shards> on gpu<id>`.

---

## 7. Validation status

* CPU/fake-torch regression tests: pass (section 5).
* Production single-shard-per-card runs (what everyone uses today): unchanged by
  construction. The c3=4.0 (5 cards) / c3=4.5 (3 cards) runs in flight are on
  build `73cc175`, i.e. **before** this change; they validate the old route,
  which is the route the new code takes for those card counts.
* **Not yet exercised on real GPUs: `n_shards > n_cards`** (the 2-card, 4-shard
  configuration). It is the first thing to verify on a free machine, with
  `dev/check_gpu_sm_shards.py` (follow-up). It builds `GpuSparseMV` on the
  listed cards, prints the shard layout and per-shard nonzeros, and compares
  `matvec`/`rmatvec` against a float64 scipy reference computed in row chunks.
  The card list is mandatory: see 3.3 for why the auto filter would refuse two
  24 GB cards.

  ```bash
  # synthetic smoke test first (seconds), then the real matrix (~19.7 GiB/card)
  python dev/check_gpu_sm_shards.py --synthetic 400000x20000 --density 2e-3 \
      --devices 0,1 --shards 4 --lowmem
  python dev/check_gpu_sm_shards.py --npz $CASE/sm_prime.npz \
      --devices 0,1 --shards 4 --lowmem
  ```

  Exit status 0 = both relative errors below 1e-4 (float32); a misplaced or
  dropped shard gives an O(1) error. Its comparison logic runs under the fake
  torch in `test_real_gpu_check_script_logic`, including a broken shard that
  must be flagged.

  Until that is done, stay on the one-block-per-card card counts of 6.3, where
  the new code path is identical to the old one.

## 8. Known limits left as-is

* `GpuTwoLevelOperator` (the resident LASSO/OLS operator) still uses one shard per
  card; it is not used when `PHEASY_GPU_OLS_RESIDENT=0` / `PHEASY_GPU_LASSO_RESIDENT=0`
  and was deliberately not touched here.
* `sensing_matrix_on_gpu()` keeps its whole-matrix `nnz >= 2**31` guard: that
  helper materialises `SM @ NS` on one device and cannot be sharded the same way.

---

## 9. Revisions (follow-up commit)

Corrections after review of the first version:

* 3.2: c3=5.0 row replaced (measured 3.57e9 matrix, plus the 6.0e9 case with
  its `n_dev=3` cell corrected from 3 to 6); the `row_shard_count` docstring no
  longer claims that "at c3=5.0 even four are not enough".
* 3.3 / 6.2: the per-card headroom is `nnz/n_cards`, not `nnz/n_shards`; the
  2-card c3=4.5 figure holds only for the low-mem layout with an explicit
  `PHEASY_GPU_SM_DEVICES`; 3 cards are 12.0 GiB resident (13.2 GiB with
  workspace), not ~13.4 GiB.
* 5: the two extra test files hold 18 tests, not 28.
* 7: the in-flight runs are on `73cc175`; the placeholder recipe is replaced by
  `dev/check_gpu_sm_shards.py`.
* Code: `TwoLevelSM.__init__` honours required GPU mode (section 1).
* `fit_scripts/Mg2C60_joint_4p50/`: per-card formula and 2-card conditions in
  the README; the submit script no longer exports `PHEASY_COL_NORMS_FILE` when
  `sm_prime.npz` will be rebuilt, replaces its `optimizer.py` md5 pin with
  feature checks, and takes `CASE` / `CONDA_SH` instead of a hard-coded home
  directory.
* Not fixable without rewriting published history: the `8b1e331` commit
  message says "CSP .t() view" where it means the CSC view.
