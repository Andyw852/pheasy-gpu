#!/bin/bash
# Mg2C60 joint RFE-OLS fit at cut3 = 4.0 A in STREAMING mode -- phase-4 validation.
#
# Purpose: exercise PHEASY_RFE_GRAM_CV=1 on a cutoff whose RESIDENT result is
# already known (fc2/fc3 + kappa k1=35.094 k2=20.802 k3=6.583, backed up in
# _ref_resident_20261114/ before this run), at a size that fits the disk:
#   p = 55,643 -> one float64 Gram = 24.77 GB; store = 5 x 24.77 = 123.9 GB on
#   disk (PHEASY_RFE_GRAM_TOTAL=0 skips the extra sum_k G_k slot);
#   host peak = 2 round buffers + panels ~ 57.5 GB, plus the 26.1 GB SM_prime the
#   driver holds -> --mem=130G.
# Accumulation is n*p^2 = 3.2e15 flops: ~1.5 h at ~600 GFLOPS on 32 host cores,
# plus the unmeasured host SpGEMM that forms the dense row blocks of SM@NS.
# Rounds: 9 x (K+1) Cholesky factorizations, s = 55,643 -> ~1.2 h.
# Expected total 4-7 h (plus a -d rebuild of 39.07 GB if prep invalidates it).
#
# CPU only: no GPU is requested, so the job can start while other fits hold the
# cards.  STREAM_GPU=1 (with a GPU request) would move the accumulation to one
# card; not used for this validation.
#
# Run from the case directory (CASE defaults to SLURM_SUBMIT_DIR):
#   sbatch submit_4p00_stream.sh
#SBATCH --partition=cpu192
#SBATCH --job-name=Mg2C60_4p00_stream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=130G
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
# fit settings as in the resident 4.0 run ...
export PHEASY_LASSO_TWOLEVEL=1
export PHEASY_LASSO_SPARSE=1
export PHEASY_HARM_DENSE=1
export PHEASY_CV_GROUP_SIZE=1488
export PHEASY_SM_DTYPE=float32
export PHEASY_EXACT_GRAM_GB=65
export PHEASY_EXACT_GRAM_ALL=1
export PHEASY_TWOLEVEL_CACHE_T=0
export PHEASY_TWOLEVEL_ADJOINT=lazy
# ... but no resident operator and no LSMR: the streamed Grams answer every round
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
export PHEASY_RFE_GRAM_DIR=${PHEASY_RFE_GRAM_DIR:-$CASE/rfe_gram_store}
export PHEASY_RFE_GRAM_GB=100
export PHEASY_RFE_GRAM_GPU=0
unset PHEASY_COL_NORMS_FILE          # the norms come from the Gram diagonal
# Without FCFIT_REUSE_SM=1 the driver deletes sm_prime.npz before the md5 guard
# below can keep it (README rule 4 / review note).
export FCFIT_REUSE_SM=1

echo "[4p00s] start $(date) host=$(hostname) c3=4.0 stream store=disk dir=$PHEASY_RFE_GRAM_DIR"

echo "=== preflight ==="
python - <<PYCHK
import hashlib, sys
import pheasy_gpu as _pg
import pheasy_gpu.core.optimizer as o
import pheasy_gpu.core.gpu_backend as gb
def md5(mod):
    return hashlib.md5(open(mod.__file__, "rb").read()).hexdigest()
print("pheasy pkg:", _pg.__file__)
print("optimizer md5:", md5(o), " gpu_backend md5:", md5(gb))
if not hasattr(o, "_RFEStreamGramCV"):
    sys.exit("[ABORT] this build has no streaming Gram engine (PHEASY_RFE_GRAM_CV)")
if not hasattr(gb, "GpuGramAccumulator"):
    sys.exit("[ABORT] this build has no GPU Gram accumulator")
if not hasattr(o, "harm_dense_setting"):
    sys.exit("[ABORT] no HARM_DENSE")
print("HARM_DENSE OK:", o.harm_dense_setting("RFE-OLS"))
PYCHK
[ $? -ne 0 ] && exit 1

# The resident reference must be preserved: this run overwrites fc2/fc3.
if [ ! -f _ref_resident_20261114/fc2.hdf5 ] || [ ! -f _ref_resident_20261114/fc3.hdf5 ]; then
    echo "[ABORT] _ref_resident_20261114/ is missing fc2/fc3 -- back up the resident result first"
    exit 4
fi
DFREE=$(df -BG --output=avail "$CASE" | tail -1 | tr -dc '0-9')
echo "[disk] ${DFREE} GB free in $CASE; this run needs ~124 GB of Gram store"
if [ "${DFREE:-0}" -lt 150 ]; then
    echo "[ABORT] less than 150 GB free; the Gram store would not fit"
    exit 5
fi

set -e
MD5_BEFORE=$(md5sum disp_matrix.pkl)
MD5_BEFORE=${MD5_BEFORE%% *}
python -u fc_fit_driver.py prep fit_config.json > prep_stream.log 2>&1
MD5_AFTER=$(md5sum disp_matrix.pkl)
MD5_AFTER=${MD5_AFTER%% *}
echo "[4p00s] disp_matrix md5 before=$MD5_BEFORE after=$MD5_AFTER"
if [ -f sm_prime.npz ] && [ "$MD5_BEFORE" = "$MD5_AFTER" ]; then
    touch sm_prime.npz
    echo "[4p00s] disp_matrix unchanged -> keeping the 39.07 GB sm_prime.npz (no -d rebuild)"
else
    echo "[4p00s] disp_matrix changed -> the driver will rebuild sm_prime.npz (-d, ~3.5 h)"
fi

if [ -f fit.log ]; then mv -f fit.log "fit_prev_stream.log"; fi
python -u fc_fit_driver.py fit fit_config.json > fit.log 2>&1 &
FITPID=$!
set +e
wait "$FITPID"; FITRC=$?
set -e
echo "[4p00s] fit rc=$FITRC $(date)"

python -u fc_fit_driver.py post fit_config.json > post.log 2>&1 || echo "[4p00s] post rc=$?"

echo "=== RESULT ==="
grep -E "RFE-GRAM-STREAM|per-fold Grams" fit.log 2>/dev/null | head -6
echo -n "RFE rounds: "; grep -c "\[RFE\] Round" fit.log || true
grep -E "\[RFE\] (Round|final exact refit: optimality)" fit.log | tail -4 || true
python - <<PYCMP
import json
try:
    ref = json.load(open("_ref_resident_20261114/fit_metrics.json"))
    new = json.load(open("fit_metrics.json"))
    print("[compare] ref:", {k: ref[k] for k in list(ref)[:4]})
    print("[compare] new:", {k: new[k] for k in list(new)[:4]})
except Exception as e:
    print("[compare] skipped:", e)
PYCMP
ls -la fc2.hdf5 fc3.hdf5 fit_manifest.json fit_metrics.json 2>/dev/null || true
echo "[4p00s] done $(date)"
