#!/bin/bash
# Mg2C60 joint RFE-OLS fit at cut3 = 4.5 A, STREAMING mode, CPU ONLY.
#
# Used when the GPUs are unavailable: PHEASY_RFE_GRAM_CV=1 replaces the resident
# GPU SpMV (3-5 cards) with one streaming pass that builds the 5 per-fold Grams
# fold by fold and answers every RFE round from them by dense solves.
#
# Sizing at c3=4.5 (p = 77,063, measured with the engine's own _rfe_stream_plan):
#   one float64 Gram 47.5 GB; store = 5 folds = 237.5 GB on disk
#   (PHEASY_RFE_GRAM_TOTAL=0 skips the extra sum_k G_k slot, which would add 47.5);
#   host peak 97.0 GB (two round buffers + panels) plus the 38.5 GB SM_prime the
#   driver holds -> --mem=180G; 8 accumulation threads, row blocks of 1622 rows.
# Disk: needs ~242 GB free (0.95 headroom rule); check before submitting.
# Expected: -s 2.5 min + -c ~39 min + accumulation 6.2e15 flops (~3 h at
# ~600 GFLOPS on the host, plus the unmeasured host SpGEMM that forms the dense
# rows) + 9 rounds of Cholesky (~2.2 h) -> 6-10 h, no GPU at all.
#
# Run from the case directory (CASE defaults to SLURM_SUBMIT_DIR):
#   sbatch submit_4p50_stream_cpu.sh
#SBATCH --partition=cpu192
#SBATCH --job-name=Mg2C60_4p50_stream_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --output=stream.out
#SBATCH --error=stream.err
set -u
CASE=${CASE:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$CASE" || exit 1
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate mace-gpu
unset PYTHONPATH

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
# fit settings as in the resident runs ...
export PHEASY_LASSO_TWOLEVEL=1
export PHEASY_LASSO_SPARSE=1
export PHEASY_HARM_DENSE=1
export PHEASY_CV_GROUP_SIZE=1488
export PHEASY_SM_DTYPE=float32
export PHEASY_SM_INDEX_DTYPE=int32
export PHEASY_EXACT_GRAM_GB=65
export PHEASY_EXACT_GRAM_ALL=1
export PHEASY_TWOLEVEL_CACHE_T=0
export PHEASY_TWOLEVEL_ADJOINT=lazy
# ... but no resident operator and no matvec: the streamed Grams answer every round
export PHEASY_GPU_SM=0
export PHEASY_GPU_RFE_RESIDENT=0
export PHEASY_GPU_LASSO_RESIDENT=0
export PHEASY_GPU_OLS_RESIDENT=0
export PHEASY_GPU_MODE=cpu
export PHEASY_GPU_FALLBACK=0
unset PHEASY_USE_GPU
export PHEASY_RFE_GRAM_CV=1
export PHEASY_RFE_GRAM_DTYPE=float64
export PHEASY_RFE_GRAM_STORE=disk
export PHEASY_RFE_GRAM_TOTAL=0
export PHEASY_RFE_GRAM_GB=120
export PHEASY_RFE_GRAM_DIR=${PHEASY_RFE_GRAM_DIR:-$CASE/rfe_gram_store}
export PHEASY_RFE_GRAM_GPU=0
unset PHEASY_COL_NORMS_FILE          # the norms come from the Gram diagonal
export FCFIT_REUSE_SM=1

echo "[4p50sc] start $(date) host=$(hostname) c3=4.5 streaming CPU-only"

echo "=== preflight ==="
# df -BG reports GiB: the store is 237.5 GB = 221 GiB, and the engine's own
# plan wants need_disk <= 0.95 * disk_free -> ~233 GiB.
DFREE=$(df -BG --output=avail "$CASE" | tail -1 | tr -dc '0-9')
echo "[disk] ${DFREE} GiB free in $CASE (the Gram store needs ~221 GiB)"
if [ "${DFREE:-0}" -lt 235 ]; then
    echo "[ABORT] less than 235 GiB free; free space or use the resident GPU path"
    exit 5
fi
python - <<PYCHK
import hashlib, sys
import pheasy_gpu as _pg
import pheasy_gpu.core.optimizer as o
def md5(mod):
    return hashlib.md5(open(mod.__file__, "rb").read()).hexdigest()
print("pheasy pkg:", _pg.__file__)
print("optimizer md5:", md5(o))
if not hasattr(o, "_RFEStreamGramCV"):
    sys.exit("[ABORT] this build has no streaming Gram engine (PHEASY_RFE_GRAM_CV)")
if not hasattr(o, "harm_dense_setting"):
    sys.exit("[ABORT] no HARM_DENSE")
print("HARM_DENSE OK:", o.harm_dense_setting("RFE-OLS"))
PYCHK
[ $? -ne 0 ] && exit 1

set -e
MD5_BEFORE=$(md5sum disp_matrix.pkl)
MD5_BEFORE=${MD5_BEFORE%% *}
python -u fc_fit_driver.py prep fit_config.json > prep_stream.log 2>&1
MD5_AFTER=$(md5sum disp_matrix.pkl)
MD5_AFTER=${MD5_AFTER%% *}
echo "[4p50sc] disp_matrix md5 before=$MD5_BEFORE after=$MD5_AFTER"
if [ -f sm_prime.npz ] && [ "$MD5_BEFORE" = "$MD5_AFTER" ]; then
    touch sm_prime.npz
    echo "[4p50sc] disp_matrix unchanged -> sm_prime.npz (57.8 GB) may be reused"
else
    echo "[4p50sc] WARNING: disp_matrix changed -> -d rebuild would be needed"
fi

if [ -f fit.log ]; then mv -f fit.log "fit_prev_resident.log"; fi
python -u fc_fit_driver.py fit fit_config.json > fit.log 2>&1 &
FITPID=$!
set +e
wait "$FITPID"; FITRC=$?
set -e
echo "[4p50sc] fit rc=$FITRC $(date)"

python -u fc_fit_driver.py post fit_config.json > post.log 2>&1 || echo "[4p50sc] post rc=$?"

echo "=== RESULT ==="
grep -E "RFE-GRAM-STREAM|reusing sm_prime|the -d pass is skipped|per-fold Grams" fit.log 2>/dev/null | head -8
echo -n "RFE rounds: "; grep -c "\[RFE\] Round" fit.log || true
grep -E "\[RFE\] (Round|final exact refit: optimality)" fit.log | tail -4 || true
ls -la fc2.hdf5 fc3.hdf5 fit_manifest.json fit_metrics.json 2>/dev/null || true
echo "[4p50sc] done $(date)"
