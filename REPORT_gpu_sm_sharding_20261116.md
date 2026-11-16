# GPU-SM row split beyond the card count (int32 CUDA CSR limit)

Date: 2026-11-16
Scope: `core/gpu_backend.py` (`GpuSparseMV`), `dev/test_gpu_memory_regressions.py`
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
`1`). See section 6.

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
* `rmatvec` **sums** `R_i.t() @ u[r0:r1]` over blocks, also placement-independent;
* the per-block memory pre-flight still runs before each upload and re-reads the
  card's free VRAM, so a second block on the same card sees the already-reduced
  headroom;
* nothing about the numerics changes: the same rows, the same values, the same
  float32 CSR, only more, smaller kernel launches.

### 3.1 The 10% headroom

`_balanced_shard_edges()` equalises nonzeros per shard, but boundaries fall on
row indices, so a shard can end up slightly above `nnz/n_shards`. 0.90 leaves
room for that rounding without wasting shards for realistic matrices.

### 3.2 Shard counts for the two production cutoffs

```
n_shards(nnz, n_dev)          n_dev=1  2   3   4   5
c3=4.5   nnz=4.816e9            3     4   3   4   5
c3=5.0   nnz~6.0e9              4     4   3   4   5
```

Note `n_dev=3 -> 3` at c3=4.5: 1.605e9 per block is already under the limit, so
no extra shards are added; `n_dev=2 -> 4` (two blocks per card). Small matrices
keep exactly one block per card, i.e. the previous behaviour.

### 3.3 Per-card footprint is unchanged

Total resident nonzeros do not depend on the split, only their grouping does.
With the low-memory layout (row blocks only; the adjoint is the zero-copy CSC
view of the same buffers) a card holding `k` shards needs

```
k * (nnz/n_shards) * (4 B values + 4 B int32 indices) * 1.10 (workspace)
```

For c3=4.5 on 2 cards: `2 * 1.204e9 * 8 B = 19.27 GB` = 17.9 GiB, +10% = 19.7 GiB
against 23.18 GiB free on a 24 GB RTX 3090. On 3 cards: 12.8 GB/card (~13.4 GiB)
-- comfortable.

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
CUDA needed. 10 tests, all pass (`python3 dev/test_gpu_memory_regressions.py`).

New:

* `test_row_shard_count_covers_the_int32_index_limit` -- the table in 3.2, the
  multiple-of-`n_dev` invariant, `per-shard nnz <= 2**31-1`, the small-matrix
  case, and the `PHEASY_GPU_SM_SHARDS` override.
* `test_more_shards_than_cards_match_cpu` -- 2 cards, 4 forced shards: asserts
  `_block_devs == [0, 1, 0, 1]`, 4 blocks, and `matvec`/`rmatvec` equal to scipy
  `A @ x` / `A.T @ u` to 1e-13.
* `test_low_mem_more_shards_than_cards_match_cpu` -- same with
  `PHEASY_GPU_SM_LOWMEM=1` (asserts no stored transpose, `_T == []`).

The fake torch's `_SparseTensor` gained `.t()` (mirrors torch's zero-copy CSC
view) so the low-mem adjoint is exercisable without CUDA.

Also run: `dev/test_resident_footprint_estimate.py`, `dev/test_colnorm_failfast.py`
-- 28 tests, all pass (same commit window).

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
   remain. The requirement is
   `free_i >= 1.1 * (nnz/n_shards) * 8 B`.
3. **Shard count for a given card count** can be read off `row_shard_count`;
   for c3=4.5 use >=3 cards, for c3=5.0 use >=4.
4. Existing scripts that printed `[GPU-SM] SpMV on N device(s)` still work; with
   more shards than cards the log line reports the *card* count, and the
   `_mem_trace` line reports `<shard>/<n_shards> on gpu<id>`.

---

## 7. Validation status

* CPU/fake-torch regression tests: pass (section 5).
* Production single-shard-per-card runs (what everyone uses today): unchanged by
  construction, and c3=4.0 (5 cards) / c3=4.5 (3 cards) runs are in flight on
  the same commit.
* **Not yet exercised on real GPUs: `n_shards > n_cards`** (the 2-card, 4-shard
  configuration). It is the first thing to verify on a free machine:

  ```bash
  PHEASY_GPU_SM_SHARDS=4 PHEASY_GPU_SM_NGPU=2 PHEASY_GPU_SM_LOWMEM=1 \
      python -c "..."   # build GpuSparseMV on a real SM_prime and compare one
                        # matvec/rmatvec against scipy
  ```

  Until that is done, prefer >=3 cards at c3=4.5 and >=4 at c3=5.0, where the
  new code path is identical to the old one.

## 8. Known limits left as-is

* `GpuTwoLevelOperator` (the resident LASSO/OLS operator) still uses one shard per
  card; it is not used when `PHEASY_GPU_OLS_RESIDENT=0` / `PHEASY_GPU_LASSO_RESIDENT=0`
  and was deliberately not touched here.
* `sensing_matrix_on_gpu()` keeps its whole-matrix `nnz >= 2**31` guard: that
  helper materialises `SM @ NS` on one device and cannot be sharded the same way.
