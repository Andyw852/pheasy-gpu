# Streaming (batched) execution for the c3 >= 4.5 RFE-OLS fit — specification for review

Status: phases 0-3 implemented behind `PHEASY_RFE_GRAM_CV=1` (default off);
phase 4 (c3=4.5 on the cluster) not run.  See section 9 for what was built,
where it deviates from this proposal, and the corrections to sections 2-6.
Audience: whoever implements/reviews the streaming path in `pheasy-gpu`.
Scope: `core/optimizer.py` (RFE-OLS / TwoLevelSM / Gram engine),
`core/gpu_backend.py` (`GpuSparseMV`), plus one dev harness.

---

## 1. Requirement

Run the Mg2C60 c3=4.5 (and later c3=5.0) RFE-OLS fit **without holding the whole
sensing matrix resident in VRAM**, so the job no longer needs 2-5 GPUs:

* must run with **1 card** (a session may only get one), and ideally **0 cards**
  (CPU-only) at some predictable slowdown;
* must **not** silently degrade to the serial CPU SpMV path;
* must produce the same force constants as the resident run to a stated
  tolerance (numbers, tolerance policy and acceptance tests in section 6);
* runtime should stay in the same order as the resident path (overnight, not
  weeks).

"Streaming" here means **one pass over the 38.5 GB matrix per accumulated
quantity**, *not* one host-to-device upload per matvec (see section 3).

---

## 2. What exists today

`SM_prime` for c3=4.5 Mg2C60 joint: shape `(1,040,112 x 92,079)`, dtype
float32 CSR, **nnz = 4,816,356,048**; 38.54 GB in memory, 57.80 GB as
`sm_prime.npz`. Feature count after symmetry/null-space reduction: p = 77,063
(35,032 harmonic columns are protected, 42,031 anharmonic ones are eliminated
by RFE).

`GpuSparseMV` (`core/gpu_backend.py`) shards the rows and keeps **all shards
resident**; since `row_shard_count()` the shard count can exceed the card count
(`_block_devs`), but every block must still fit at the same time:

| cards | shards | nnz/block | blocks/card | need/card (low-mem) | feasible |
|---|---|---|---|---|---|
| 1 | 3 | 1.605e9 | 3 | 39.49 GiB | no (card has 23.18 GiB free) |
| 2 | 4 | 1.204e9 | 2 | 19.75 GiB | yes, tight |
| 3 | 3 | 1.605e9 | 1 | 13.16 GiB | yes (current production run) |
| 4-5 | 4-5 | 1.20e9 / 0.96e9 | 1 | 9.87 / 7.90 GiB | yes, unnecessary |

Measured costs of the resident path (RTX 3090, 24 GB; host 56 cores / 251 GB):

| item | value | source |
|---|---|---|
| GPU SpMV pair (matvec + rmatvec), 2 shards | 0.0016 s + 0.0051 s | measured |
| LSMR iterations actually used | **5000 per fold** (`istop=7`, the cap) | c3=4.5 log |
| matvec pairs over one fit (9 rounds x 5 folds x 5000) | ~450,000 | derivation |
| GPU SpMV share of the wall time | ~30 min of 10-14 h | measurement |
| dominant cost | host-side CV solves / Gram / Cholesky | measurement |
| one full host->device pass of the matrix | **~1.75 s** | bandwidth estimate: 38.5 GB / ~22 GB/s PCIe |
| `-c` symmetry constraints + null space | 37 min, pure CPU | measured |
| `-d` sensing-matrix build | 3.3-4.7 h, pure CPU/RAM | measured |

---

## 3. Why the naive streaming design fails

Streaming the matrix for **every matvec** costs one full pass per call:

```
450,000 matvec pairs x 1.75 s = 787,500 s = 9.1 days
```

So "upload block i, compute, upload block i+1" inside `matvec` is hopeless for
this operator at this cutoff, unless the number of matvecs drops by 2-3 orders
of magnitude. Any streaming design must therefore **change what is streamed**:
stream the matrix once and accumulate something small enough to iterate on.

---

## 4. Design space

### D1 — streamed matvec + fewer iterations (keep LSMR)

Keep `GpuSparseMV` semantics but stage each shard from host memory on demand,
and cut the pass count with warm starts across RFE rounds (the active set only
shrinks) plus a Jacobi preconditioner.

* passes: `fits x iterations x 2`. For one night (20 h = 72,000 s) the budget is
  ~41,000 passes, i.e. **~200 iterations per fit** for 45 fits.
* today's LSMR hits 5000 -> needs a 25x reduction. Warm start may give part of
  it; not guaranteed.
* numerics: same solver, different stopping -> results differ slightly; the
  approximation is *not* controllable by the user.
* memory: only one shard staged -> a few GiB, works on 1 card.
* **Verdict: attractive only as an incremental optimisation, not as the
  streaming answer.**

### D2 — one streaming pass to per-fold Grams, then dense solves (RECOMMENDED)

The RFE-OLS criterion is exactly the OLS CV-RMSE of the current active set, and
RFE's importance is the (scaled) OLS solution. Both are **functions of the
normal-equation sums only**:

```
for each validation fold k:   G_k = A_k^T A_k,  b_k = A_k^T y_k,  yy_k = ||y_k||^2
train Grams:                  G_train_k = sum_{j != k} G_j
CV-RMSE for a candidate set S:
    beta   = solve(G_train_k[S,S], b_train_k[S])
    SSE    = yy_val_k - 2 beta^T b_k[S] + beta^T G_k[S,S] beta
    RMSE_k = sqrt(SSE / n_val_k)
```

No matrix access is needed after the pass: **all 9 RFE rounds x 5 folds become
dense p_S x p_S solves.** Pass count drops from ~450,000 to **1**.

Cost of that one pass: accumulating K fold Grams costs `n x p^2` multiply-adds
= 1.04e6 x 5.94e9 = **6.2e15 flops** at p = 77,063 (2.2e15 at the final
p = 45,986, which the current code already does in ~1 h on 32 CPU cores with
`dsyrk`).

| place the Gram is accumulated | throughput | one pass | note |
|---|---|---|---|
| host, float64 `dsyrk` (32 cores) | ~600 GFLOPS (inferred from the ~1 h final refit) | **~2.9 h** | already implemented |
| GPU, float32 cuBLAS | ~15 TFLOPS | **~7 min** | needs the Gram on device |
| GPU, float64 cuBLAS | ~0.5 TFLOPS (3090) | ~3.4 h | pointless |

Memory: one Gram is `p^2 x 8 B = 47.5 GB` (float64) or `23.7 GB` (float32);
K = 5 validation folds -> 237 GB / 118 GB if all are kept. Options, in order of
increasing implementation effort:

1. **float32 Grams + float64 Jacobi-scaled solve**, keep the K validation Grams
   (118 GB) and stream the matrix from `sm_prime.npz` on disk (memmap row
   blocks) so the 38.5 GB host array is not needed: total ~120-130 GB of the
   node's 251 GB -> fits with `--mem=200G`. Solve path already exists
   (`_gram_solve`).
2. **column-panel the Gram** (build `G[:, :p/2]` in pass 1a, `G[p/2:, p/2:]` in
   pass 1b, exploiting symmetry): 2 passes, ~59 GB float32, no disk needed.
3. **disk-backed Grams** (`np.savez`/memmap, 118 GB on the 296 GB free disk),
   paging in only `G[S,S]` per round: smallest RAM, ~1 extra minute per round.
4. **GPU-resident float32 Gram** (23.7 GB): fits one 24 GB card only with the
   block buffer squeezed; 2 cards are comfortable and the pass drops to minutes.

### D3 — hybrid residency + streamed tail

Keep the largest shard(s) resident on the cards we have and stream the rest per
matvec. Per-pass cost scales with the streamed fraction:

```
1 card holding 40% -> 0.6 x 1.75 s = 1.05 s per matvec pair -> 450,000 = 5.5 days
```

Only viable together with D1's iteration reduction. **Verdict: not worth its
complexity once D2 exists.**

---

## 5. Recommended implementation plan

Target: **D2 with a host fallback and an optional GPU Gram**, keeping the
resident path untouched and selected by env knob.

Phase 0 — prototype the RFE-from-Gram loop offline (no cluster time)
* new dev harness `dev/proto_rfe_gram.py`: synthetic operator with known
  coefficients, p = 2,000, 5 folds, 9 rounds.
* reimplement the round loop: for each round, for each fold, `_gram_solve` on
  `G_train_k[S,S]`; compute CV-RMSE by the SSE identity; rank by
  scale-invariant importance (exactly the quantity used today).
* assert equality against the current LSMR-based loop to `1e-6` relative on the
  synthetic problem. This is the correctness gate for everything after.

Phase 1 — productionize the round loop
* reuse `_accumulate_fold_grams(A, y, cols, splits)` (already implemented) and
  `_gram_solve(build, rhs)` (already implemented, Jacobi-scaled Cholesky with an
  `eigh` min-norm fallback);
* reuse `_rfe_gram_budget_bytes()` / `PHEASY_RFE_GRAM_GB` for the memory guard;
* new knobs: `PHEASY_RFE_GRAM_CV=1` (enable), `PHEASY_RFE_GRAM_DTYPE=float32|float64`,
  `PHEASY_RFE_GRAM_STORE=ram|disk`, `PHEASY_RFE_GRAM_GPU=0|1`;
* keep the LSMR path as default until section 6 passes.

Phase 2 — the streaming pass
* row source: memmap `sm_prime.npz` blocks (`_rfe_row_source` already yields
  column-sliced row blocks from the operator; add a disk-backed source that does
  not materialise the whole matrix);
* block size from a budget (current default: `PHEASY_RFE_FINAL_BLOCK_ROWS=20000`,
  i.e. `20,000 x p x 8 B` = 12.3 GB at p=77k -> must be tuned down for float32
  panel builds);
* guards: refuse (loudly) if the requested Gram set plus the row block does not
  fit; never fall back to the serial matvec path silently.

Phase 3 — optional GPU Gram (cuBLAS, float32)
* split the Gram by columns across the cards that are present, or keep it on one
  if it fits; the row block is streamed to each card once;
* expected pass time ~7 min at 15 TFLOPS, i.e. the whole fit becomes
  `-c` (37 min) + one Gram pass + ~9 x 5 dense solves.

Phase 4 — end-to-end validation on c3=4.5
* compare against the resident 3-card result (the run in flight):
  per-round CV_RMSE, final support, final `fc2.hdf5`/`fc3.hdf5`, and kappa
  after the phono3py step.

Estimated totals at c3=4.5 (p=77,063):

| configuration | matrix passes | Gram accumulation | CV solves | fit total |
|---|---|---|---|---|
| resident 3 cards (today) | 450,000 matvec pairs | - | LSMR, 10-14 h | **10-14 h** |
| D2 host float32, 5 folds, matrix in RAM (118 GB Grams + 38.5 GB matrix) | 1 | ~1.5-3 h `ssyrk` | minutes | **~2-3.5 h** |
| D2 host float64, 3 folds (71 GB Grams, needs a CV-scheme decision) | 1 | ~2.9 h `dsyrk` | minutes | **~3-4 h** |
| D2 GPU float32 Gram, 2 cards (Gram 23.7 GB, split or single) | 1 = 1.75 s of PCIe | **~7 min** (15 TFLOPS) | minutes | **~10-15 min** |

Two things fall out of this table:

* the pass count, not the card count, was the problem: one pass costs 1.75 s of
  PCIe, while 450,000 passes cost 9.1 days -- which is why the naive design was
  hopeless and why D2 makes card count almost irrelevant;
* with D2 the fit stops being GPU-bound at all: the wall time becomes `-c`
  (37 min) plus, when the matrix has to be built, the `-d` pass (3.3-4.7 h,
  CPU/RAM only).

---

## 6. Acceptance criteria

1. **Numerics.** On the synthetic harness, per-round CV_RMSE and the final
   coefficient vector must match the current LSMR path to `1e-6` relative; on
   c3=4.5, per-round CV_RMSE must match the resident run to `1e-4` relative and
   the final `fc2`/`fc3` must give the same kappa within `1e-3` relative.
   A different elimination *order* in the last rounds is acceptable only if the
   final support and kappa meet those bounds — record it in the log.
2. **No silent downgrade.** Any failure in the streaming/Gram path raises with
   the reason; `PHEASY_GPU_MODE=required` semantics stay as fixed in `6c3b013`.
   The serial CPU matvec path must be unreachable without an explicit opt-in.
3. **Memory guards.** Refuse before allocating: the check must use the same
   budget helpers (`_rfe_gram_budget_bytes`, `_available_memory_bytes`) and
   print what it decided (Gram size, dtype, store, blocks, per-card bytes).
4. **Reproducibility.** Same seed/splits as today (`PHEASY_CV_GROUP_SIZE=1488`,
   `PHEASY_RFE_SEED`) and the same `step=0.10 / patience=5 / min_features=1`;
   the CV splits come from `_make_cv_splits` unchanged.
5. **Regression suite.** The four existing GPU test files keep passing
   (`test_gpu_memory_regressions.py` 11, `test_gpu_failure_semantics.py` 6,
   `test_resident_footprint_estimate.py` 11, `test_colnorm_failfast.py` 7) plus
   new tests for the Gram-CV loop (synthetic equality, memory guard, disk store
   round-trip, and a "dropped row block" test that must be caught).
6. **Docs.** `GPU.md` (dispatch contract), the toolkit README, and the report
   get a section each; no claim of "bit-identical" results.

---

## 7. Decisions needed from the maintainer

* Is changing the CV solver from LSMR (tolerance-limited) to exact normal
  equations acceptable, given that all cutoffs are re-run together? If kappa is
  the acceptance quantity, the cross-cutoff comparison stays consistent as long
  as every cutoff uses the same mode.
* float32 Gram (23.7 GB, 2x faster, Jacobi-scaled solve) or float64 (47.5 GB,
  safe)? The current final refit uses float64 `dsyrk`; mixing dtypes between the
  round solves and the final refit would need a documented reason.
* Keep 5 folds (118-237 GB of Grams) or reduce to 3 folds under the streaming
  mode (71-142 GB) with a documented comparability note?
* Where should the Gram live by default: host RAM, disk memmap, or GPU? The
  recommendation above is "whatever fits, GPU first, then host float32, then
  disk".
* Should `GpuTwoLevelOperator` (resident LASSO/OLS) get the same treatment, or
  stay one shard per card?

---

## 8. Pointers (verified in the current tree)

* `core/gpu_backend.py`: `GpuSparseMV` (~line 2459), `row_shard_count()`,
  `_block_devs`, `_mv_blocks()`, `_rmatvec_from_row_blocks()`; the existing
  streaming example to copy the pattern from is `col_norms_streaming()` (~line 215).
* `core/optimizer.py`: `TwoLevelSM` (~3200, GPU-SM construction + required-mode
  rule), `_accumulate_fold_grams()` (5586), `_gram_cv_folds()` (4970),
  `_gram_solve()` (5521), `_rfe_row_source()`, `_rfe_gram_budget_bytes()` (5570),
  `_exact_gram_budget_bytes()` / `PHEASY_EXACT_GRAM_GB` (5646), the final exact
  refit's blocked Gram (`[RFE] final exact refit: Gram normal equations`, ~6189),
  RFE round loop, `PHEASY_LSQR_MAXITER` (default 5000).
* Dev harness to model after: `dev/check_gpu_sm_shards.py` (streamed reference
  comparison), `dev/test_gpu_memory_regressions.py` (fake-torch pattern).
* The streaming work must not disturb the resident path; `6c3b013` is the
  current `main` (see `REPORT_gpu_sm_sharding_20261116.md`,
  `REVIEW_NOTE_gpu_sm_sharding_zh.md`).

---

## 9. Implementation status (phases 0-3)

Opt-in through `PHEASY_RFE_GRAM_CV=1`; without it the RFE path is unchanged.
User documentation: GPU.md "Streaming exact RFE-OLS"; knobs also in README.md.

| phase | where | state |
|---|---|---|
| 0 prototype / gate | `dev/proto_rfe_gram.py` | passes at p = 2000 (below) |
| 1 round loop | `_RFEStreamGramCV`, `_GramStore`, `_spd_solve_inplace` in `core/optimizer.py`, wired into `_RFECVBase.fit` | done |
| 2 streaming pass + guards | `_rfe_stream_plan`, `_rfe_row_take`, fold-by-fold accumulation, row accounting | done (no memmap of `sm_prime.npz`, see below) |
| 3 GPU Gram | `GpuGramAccumulator` in `core/gpu_backend.py` | done, tested on the torch CPU device only |
| 4 c3=4.5 end to end | `fit_scripts/Mg2C60_joint_4p50/submit_4p50_stream.sh` | **not run** |

**Phase-0 gate** (`python dev/proto_rfe_gram.py`): a 12000 x 2000 two-level
operator with 400 protected columns at 40x norm, 5 grouped folds, step 0.10,
patience 5.  Every engine runs the same 23 rounds and selects the same 574
features:

| engine | CV_RMSE vs in-RAM engine | coef |
|---|---|---|
| LSMR (tolerances tightened, float64 operator) | 2.4e-13 | 1.3e-15 |
| stream, RAM store | 2.8e-13 | 3.7e-16 |
| stream, disk store | 2.8e-13 | 3.7e-16 |
| stream, one round buffer | 3.5e-13 | 3.7e-16 |
| stream, GPU accumulation (torch CPU device) | 1.6e-13 | 3.2e-16 |

The gate needs a float64 operator: a float32 LSMR stops at its precision floor,
far above 1e-6.  `dev/test_rfe_gram_stream.py` (16 tests) runs the same gate at
test size plus the guard cases of section 6: memory refusal before allocating,
disk round-trip, a dropped row block, and both float32 guards.

### Knobs (section 5's names, plus what the implementation needed)

`PHEASY_RFE_GRAM_CV`, `PHEASY_RFE_GRAM_DTYPE=float64|float32`,
`PHEASY_RFE_GRAM_STORE=auto|ram|disk`, `PHEASY_RFE_GRAM_GPU=0|1` as proposed.
Added: `PHEASY_RFE_GRAM_DIR` / `_KEEP`, `_TOTAL` (store `sum_k G_k` too),
`_ROUND_BUFFERS` (1 or 2 `s x s` buffers), `_BLOCK_GB` / `_BLOCK_ROWS`,
`_PANEL_GB`, `_THREADS`, `_F32_MAXERR`, and `PHEASY_RFE_GRAM_GPU_{DEVICES,DTYPE,
PANEL,GB,FRACTION,MAX_WAVES,FLUSH_ROWS}`.  `PHEASY_RFE_GRAM_GB` is the budget
of both engines; in streaming mode its default is 0.8 of the available RAM.

### Deviations from sections 4-5

* **Fold-by-fold accumulation.**  All K fold Grams cannot be in memory while
  one pass routes rows to them (237.5 GB at p = 77,063).  Each fold is
  therefore accumulated from its own rows into one float64 accumulator.  With
  the default contiguous CV folds each fold is one row range, and the K folds
  together are still one pass.
* **float64 is the default everywhere** (store, accumulation, GPU GEMM).
  float32 storage is accepted only under the two guards below; GPU float32
  GEMM flushes into a float64 host accumulator.
* **No memmap of `sm_prime.npz`.**  The driver already holds `SM_prime` (38.5
  GB) when the RFE starts.  The plan counts it, through the available-RAM
  budget, and the c3=4.5 plan fits `--mem=200G` with it in RAM.  Mapping
  the npz instead needs a driver change and uncompressed members.  That is
  left for when a cutoff stops fitting.

### Corrections to sections 2-6

1. **D2 already existed.**  `_RFEGramCV` ([FIX RFE-GRAM]) uses exactly the
   identities of section 4/D2, in float64 host RAM.  Its gate `8 (K+4) p^2 <=
   PHEASY_RFE_GRAM_GB` (default 16 GB) needs 427.6 GB at p = 77,063 and
   152.3 GB at the final ~46k, so it never engaged at c3=4.5.  The work was the
   memory layout, not a new solver.
2. **"CV solves: minutes" is wrong.**  A round is K+1 Cholesky factorizations
   of `s^3/3` flops: 25 min at round 0 (s = 77,063) at 600 GFLOPS.  With the
   support shrinking 77,063 -> 53,125 that is ~2.2 h over nine rounds.  The D2
   totals in section 5 understate the host runs by ~2 h.
3. **One pass is not 1.75 s of PCIe.**  What is streamed and multiplied are
   dense rows of `SM_prime @ NS`, 641 GB in float64 (321 GB float32), not the
   38.5 GB sparse `SM_prime`.  On the GPU that is ~30 s of PCIe per wave.
   Forming those rows on the host (sparse x sparse products, densified) is a
   cost nobody has measured.  It may dominate the accumulation.
4. **The memory tables omit the round buffers.**  A round needs one or two
   `s x s` float64 matrices: 47.5 GB each at round 0.  The "float32 Grams in
   RAM, ~120-130 GB" option is really 118.8 GB store + 47.5-95 GB buffers +
   38.5 GB `SM_prime`, i.e. 205-252 GB, which does not fit the 251 GB host
   with two buffers.  The float64 disk store fits: 97 GB host peak + 38.5 GB.
5. **"GPU-resident float32 Gram (23.7 GB)" is one of K.**  With fold-by-fold
   accumulation only one is resident at a time.  Only its upper triangle is
   needed (11.9 GB float32, 23.8 GB float64), so float64 fits one A800 or two
   24 GB cards in one pass, and one 24 GB card in 2-3.
6. **float32 Grams fail the CV tolerance of section 6 on high-SNR data.**  The
   validation SSE `||y_k||^2 - 2 x.b_k + x.G_k.x` cancels by a factor of
   `x.G_k.x / SSE_k`, which multiplies the 2^-24 storage rounding.  On the
   test problem (cond ~1e3, 1% noise) float32 storage gave 2e-7 coefficient
   error but 1.6e-4 CV_RMSE error.  The engine refuses float32 when that
   estimate, or `cond x 2^-24` from LAPACK `dpocon`, exceeds
   `PHEASY_RFE_GRAM_F32_MAXERR` (1e-4).  The same guards apply to float32
   GPU GEMM accumulation, with a larger, conservatively modelled entry error
   (~2.9e-6 at the c3=4.5 defaults; GPU.md).  That makes the "3090 float32
   GEMM, ~7 min" option refuse unless the tolerance is raised on purpose.
7. **"Per-round CV_RMSE within 1e-4 of the resident run" is not attainable.**
   The resident rounds are LSMR iterates at the 5000-iteration cap (istop=7),
   not least-squares solutions.  [FIX RFE-GRAM] measured the same support
   scoring CV_RMSE 2.505e-01 or 3.401e-01 depending only on the solver
   tolerance.  For phase 4, compare the final support, fc2/fc3 and kappa with
   the resident run, and take the streaming run as the reference for
   per-round CV.  A tolerance check needs an exact reference, e.g. the in-RAM
   engine on a support that fits it (`PHEASY_RFE_SUPPORT_NPY` +
   `PHEASY_RFE_SUPPORT_CV`).

### Section 7 decisions, as currently defaulted

1. LSMR -> exact: opt-in per run (`PHEASY_RFE_GRAM_CV=1`).
2. Gram dtype: float64; float32 behind the two guards.
3. Folds: unchanged (5, the same `_make_cv_splits`).
4. Gram location: `auto` = RAM if the plan fits, else disk; the GPU only
   accumulates, and only when asked.
5. `GpuTwoLevelOperator`: untouched (not used by this path).

---

## 10. First production attempt at c3=4.5 (2026-11-17/18): measurements, and why it was stopped

Job 7622, run CPU-only on the RTX-3090 box while another user held all five GPUs.
Configuration: p = 77,063, K = 5, `store=disk` float64, `PHEASY_RFE_GRAM_TOTAL=0`,
`PHEASY_RFE_GRAM_GB=120`, `--mem=180G`, 32 CPUs. The engine's plan line:

```
[RFE-GRAM-STREAM] p=77063 K=5 n=1040112: store=disk(258.8 GB free) dtype=float64
  total=off round-buffers=2; Grams 237.5 GB, host peak 97.0 GB of 120.0 GB budget
  (one float64 Gram = 47.5 GB, row blocks of 1622 rows = 1.00 GB);
  accumulation: host dsyrk (float64)
```

### Gram accumulation: 33,307.4 s (9.25 h), not the ~3 h the design assumed

| artifact | value |
|---|---|
| total, as logged | `per-fold float64 Grams (5 folds, 237.55 GB, streamed, store=disk) built in 33307.4s` |
| fold 4/5 | 208,320 rows in 129 blocks, 5,873.1 s |
| fold 5/5 | 206,832 rows in 128 blocks, 7,759.4 s |
| per-fold wall time from the store mtimes | 2.08 / 1.72 / 1.65 / 1.63 / 2.15 h |

**Root cause of the 3x underestimate: the folds are not contiguous row ranges.**
`PHEASY_CV_GROUP_SIZE=1488` makes the splits group-aware, so each fold is a
scattered set of row blocks.  The fold-by-fold accumulator design assumed
"each fold is one row range, and the K folds together are still one pass"
(section 9); with grouped splits the engine needs **one full pass per fold** —
five passes over 1,040,112 rows, i.e. 5 x (SpGEMM that forms 641 GB of dense
SM@NS rows + the GEMM), and the measured 1.6-2.2 h per fold is exactly that.
Section 9's "K folds together are one pass" holds only for contiguous splits.

### Round 0: stopped after 6.65 h with zero rounds completed

Round 0 (s = 77,063) started at 21:02 and was still running at 03:41 when the
job was stopped.  Observed during it: ~20 CPUs busy and RSS peaking at 194 GB.

Both numbers point at the `_spd_solve_inplace` **eigendecomposition fallback**
rather than its Cholesky path: the fallback pre-flights `3 s^2 x 8 B = 142 GB`
of workspace at this size, which together with the 47.5 GB fold Gram reproduces
the 194 GB peak, and dsyevd is highly parallel.  Cholesky for all five folds
would have been ~26 min in total.

The root cause is not confirmed: the `cholesky`/`eigh_min_norm` marker never
reached the log because the child's stdout is block-buffered through a pipe.
Candidates, in order:
1. a fold's train Gram has zero-variance / collinear columns (the round ridge is
   1e-11), so the Jacobi-scaled Cholesky fails on a genuinely non-PD system;
2. the upper-triangle store is not mirrored back before the solve.

### Cost of the attempt and state left behind

* 14 h wall clock (9.25 h accumulation + 6.65 h of round 0), no fc2/fc3.
* 237.5 GB of disk backed the store; it was removed after stopping (disk back
  to 239 GB free), `sm_prime.npz` (57.8 GB), `ns_*.npz` and the column norms
  were kept.
* The driver's `-d` reuse patch (fullerene-kappa-db `14dd7f6`) means a later
  attempt pays `-c` (39 min) + the RFE, not the 4 h rebuild.

### What to fix before trying streaming at this size again

1. **Report the solve method.** Print `method`/`rcond` per fold at round 0 (or
   flush the child's stdout) so the eigh fallback is visible immediately.
2. **Round cost.** 9 rounds at `O(s^3)` with a 142 GB eigh workspace is not
   viable; either make the round-0 system Cholesky-safe (ridge / column
   pruning) or accept a different elimination schedule.
3. **Fold layout.** Either make the streaming splits contiguous (documented
   comparability note required) or budget 5 passes, i.e. ~9 h for the
   accumulation at this p.
4. **GPU accumulation** is worth it only on big-FP64 parts: on A800 (9.7 TFLOPS
   FP64) 6.2e15 flops is ~11 min; on the 3090 (0.556 TFLOPS) it is the same
   ~3 h as the host dsyrk, and the 23.8 GB triangle does not fit one 24 GB card
   anyway (2-3 waves, with the host SpGEMM repeated per wave).
5. Streaming at this p therefore remains a **fallback for when no GPU is
   available**, not the production route: the resident GPU path (5 cards at
   c3=4.5) is still the fast one, once the cards are free.
