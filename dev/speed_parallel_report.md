# Fitting speed vs parallelization backend (RTX 3090, c7 SM n=8)

Reproducer: `dev/speed_parallel.py` (fit-only wall time, SM loaded once,
GPU warmed up with one OLS).

## Setup

- Data: MnIn2Se4 c7, n=8 configs -> SM 4536x3678 (FP64), force target F 4536.
- Host: 1x RTX 3090 (dedicated, GPU 4) + 2x Xeon E5-2680 v4 (56 cores),
  OpenBLAS 0.3.28. The box was NOT idle: load 44/56 at start, 56/56 by the
  end (other users' jobs), so CPU times are contended/inflated.
- FIT_KW: nalpha=20, cv=5, tol=1e-6, max_iter=20000, alpha_auto=True,
  decades=4.0, standardize=True. RFE step=0.3, min_features=100.

Backends (only n_jobs differs between the two CPU modes):

| backend | use_gpu | PHEASY_N_JOBS | OpenBLAS threads |
|---|---|---|---|
| gpu       | True  | 1  | 56 (numpy is minor here) |
| cpu_blas  | False | 1  | 56 (single process, BLAS threads) |
| cpu_njobs | False | 16 | 56 per loky worker (16x56 oversubscribed) |

## Results

| method | gpu | cpu_blas | cpu_njobs | GPU speedup (vs cpu_blas) | n_jobs speedup (blas/njobs) |
|---|---|---|---|---|---|
| OLS    | 23.1 s | 967.6 s  | n/a       | 42x  | n/a (single SVD) |
| RIDGE  | 23.3 s | 1338.3 s | n/a       | 57x  | n/a (single SVD) |
| LASSO  | 26.7 s | 1844.7 s | 1178.9 s  | 69x  | 1.56x |
| ALASSO | 32.6 s | 1623.4 s | 1048.3 s  | 50x  | 1.55x |
| RFE    | 172.2 s | >9720 s (killed) | n/a | >56x | n/a |

nnz (sanity): OLS/RIDGE 3678 (dense); LASSO 3398 (92%), ALASSO 3102 (84%),
RFE 2575 (selected support). The LASSO/ALASSO CV picked alpha at the grid
minimum (near-OLS regime, the known "alpha sits at grid MINIMUM" warning), so
these timings are the dense/near-OLS end of the spectrum.

## Findings

1. GPU (cuBLAS) is 42-69x faster than the best single-process CPU path
   (OpenBLAS 56 threads) for the dense fits; RFE is >56x (CPU impractical:
   killed after >2.7 h with the box saturated, vs 172 s on GPU). These are
   LOWER bounds - the CPU side ran under 44->56/56 load from other users.
2. Process parallelism (n_jobs=16) gives a further ~1.55x over single-process
   BLAS for the CV-grid methods (LASSO, ALASSO) even though each of the 16
   loky workers inherits 56 OpenBLAS threads (16x56 oversubscription): the CV
   (alpha x fold) grid fan-out outweighs the thread oversubscription.
3. n_jobs is inert for OLS/RIDGE (single SVD, no CV grid to fan out); their
   CPU parallelism comes only from OpenBLAS threads.
4. GPU total (5 methods) = 277.9 s (~4.6 min) vs CPU-BLAS total >15.6 h
   (OLS+RIDGE+LASSO+ALASSO = 96 min + RFE >2.7 h extrapolated over step).

## Caveats

- CPU timings are contended (shared box, 44->56/56 load); GPU timings are
  clean (dedicated device). Re-run on an idle box for absolute CPU numbers.
- RFE used step=0.3 / min_features=100 (not the default 0.05), so RFE timing
  is for ~6-11 elimination rounds, not the full default sweep.
- The dense near-OLS regime (alpha at grid minimum) makes LASSO/ALASSO do
  more CD/FISTA iterations than a well-regularized fit would; speed ranking is
  unaffected but absolute times are the slow end.
## Memory (c7 SM n=8, same config as the speed table)

Measured with `dev/mem_bench.py` (RTX 3090, `torch.cuda.max_memory_allocated`
peak per fit; host RSS = process high-water mark).

| method | GPU VRAM peak | host RSS (GPU backend) |
|---|---|---|
| SM load (CSR sparse.mm -> dense) | 579 MB | 1.76 GB |
| OLS   | 839 MB | 1.76 GB |
| RIDGE | 847 MB | 1.76 GB |
| LASSO | 1581 MB | 2.06 GB |
| ALASSO | 1581 MB | 2.07 GB |
| RFE   | 864 MB | 2.08 GB |

CPU RAM per backend (n=8, host side):

- cpu_blas (single process): ~1.5-2 GB = dense SM 133 MB + SVD workspace.
- cpu_njobs (16 loky workers): ~2-3 GB = 16 x ~107 MB training-fold copy +
  CD workspace. The memory-aware cap (`_lasso_n_jobs`) computes per_worker =
  4536x3678x8 = 133.5 MB against a 6 GB budget (PHEASY_LASSO_MEM_GB) -> cap 48,
  so n_jobs=16 is NOT memory-limited at n=8. At n=45 (per_worker 716 MiB) the
  same budget caps workers to 8 - the OOM guard the cap exists for.

Findings:

- Memory is never the binding constraint: 24 GB VRAM peaks at ~1.6 GB (n=8)
  and ~3.2 GB (n=45, GPU.md); the bottleneck is FP64 compute, not memory.
- LASSO/ALASSO are the heaviest (1581 MB) because GpuLassoCV holds one Gram
  (p x p = 108 MB) plus one A_va copy per CV fold: 5-fold = ~540 MB Grams +
  ~133 MB A_va, plus the 133 MB SM and FISTA workspace -> the 1.58 GB peak.
  OLS/RIDGE/RFE need only one SVD/QR workspace (~840-870 MB).
- n_jobs process parallelism costs host RAM ~linearly in worker count but stays
  bounded by the 6 GB budget; the GPU path spends its parallel budget in VRAM
  (a few hundred MB) instead, so it scales without RAM pressure.
- Note: mem_bench did not set PHEASY_RFE_STEP, so its RFE used the default
  step=0.05 (nnz 3153 vs 2575 at step=0.3); the VRAM peak (864 MB) is the
  round-0 full-feature QR and is step-independent.

