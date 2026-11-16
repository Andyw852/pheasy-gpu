#!/bin/bash
# Mg2C60 joint RFE-OLS fit at cut3 = 4.5 A in STREAMING mode: no resident SpMV,
# 0 or 1 GPU (SPEC_gpu_sm_streaming.md, phase 4; GPU.md "Streaming exact RFE-OLS").
#
# PHEASY_RFE_GRAM_CV=1 replaces the ~450,000 LSMR matvec pairs of the resident run
# by one pass that builds the 5 per-fold Grams (float64, 47.5 GB each at
# p = 77,063) fold by fold and then solves every RFE round exactly from them.
# Nothing in this path falls back: a plan that does not fit, a dropped row block,
# a failed GPU accumulation or a float32 store that is too coarse raises.
#
# Sizing at c3=4.5: Gram store 237.5 GB on disk (PHEASY_RFE_GRAM_DIR needs ~242 GB
# free); host peak ~97 GB (two 47.5 GB round buffers) on top of the ~38.5 GB
# SM_prime the driver holds -> --mem=200G.  The engine budget is 0.8 of the RAM
# available when it starts (PHEASY_RFE_GRAM_GB overrides); it prints its plan.
# Expected: accumulation ~6.2e15 flops (n p^2, ~3 h at ~600 GFLOPS on the host
# plus the unmeasured host SpGEMM that forms the dense row blocks), rounds
# ~2.2 h of Cholesky.  STREAM_GPU=1 moves the accumulation to the freest card
# (float64 GEMM; 2-3 waves = passes per fold on a 24 GB card).
#
# Run from the case directory (CASE defaults to SLURM_SUBMIT_DIR):
#   sbatch submit_4p50_stream.sh                              # CPU only
#   STREAM_GPU=1 sbatch --gres=gpu:5 submit_4p50_stream.sh    # 1 card by free VRAM
#SBATCH --partition=cpu192
#SBATCH --job-name=Mg2C60_4p50_stream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=stream.out
#SBATCH --error=stream.err
set -u
CASE=${CASE:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$CASE" || exit 1
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate mace-gpu
unset PYTHONPATH
STREAM_GPU=${STREAM_GPU:-0}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
# fit settings as in submit_4p50_resume5.sh ...
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
export PHEASY_GPU_FALLBACK=0
export PHEASY_RFE_GRAM_CV=1
export PHEASY_RFE_GRAM_DTYPE=float64
export PHEASY_RFE_GRAM_STORE=${PHEASY_RFE_GRAM_STORE:-disk}
export PHEASY_RFE_GRAM_DIR=${PHEASY_RFE_GRAM_DIR:-$CASE/rfe_gram_store}
unset PHEASY_COL_NORMS_FILE          # the norms come from the Gram diagonal
# Without FCFIT_REUSE_SM=1 the driver's prep deletes sm_prime.npz before the md5
# guard below can keep it (see submit_4p50_resume5.sh / README rule 4).
export FCFIT_REUSE_SM=1

if [ "$STREAM_GPU" = 1 ]; then
    export PHEASY_USE_GPU=1
    export PHEASY_GPU_MODE=required
    # one card, the freest that can hold a few panels (SLURM GRES is not trusted
    # on the shared box, see README rule 3)
    SEL=$(python - <<PYGPU
import sys, torch
best = None
for i in range(torch.cuda.device_count()):
    try:
        f, _t = torch.cuda.mem_get_info(i)
    except Exception as e:
        print("# gpu%d unqueryable: %s" % (i, e), file=sys.stderr)
        continue
    print("# gpu%d free %.1f GiB" % (i, f / 2**30), file=sys.stderr)
    if f >= 12 * 2**30 and (best is None or f > best[0]):
        best = (f, i)
print("NO_GPU" if best is None else best[1])
PYGPU
)
    echo "[gpu] picked: $SEL"
    if [ "$SEL" = "NO_GPU" ] || [ -z "$SEL" ]; then
        echo "[ABORT] STREAM_GPU=1 but no card with >=12 GiB free; rerun with STREAM_GPU=0"
        exit 3
    fi
    export PHEASY_RFE_GRAM_GPU=1
    export PHEASY_RFE_GRAM_GPU_DEVICES="$SEL"
    export PHEASY_GPU_DEVICES="$SEL"
else
    # an explicit CPU run, not a downgrade: nothing in this fit asks for a GPU
    unset PHEASY_USE_GPU
    export PHEASY_GPU_MODE=cpu
    export PHEASY_RFE_GRAM_GPU=0
fi
echo "[4p50s] start $(date) host=$(hostname) c3=4.5 stream gpu=$STREAM_GPU store=$PHEASY_RFE_GRAM_STORE dir=$PHEASY_RFE_GRAM_DIR"

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

set -e
# sm_prime.npz reuse, exactly as submit_4p50_resume5.sh (rule 4 of the README)
MD5_BEFORE=$(md5sum disp_matrix.pkl)
MD5_BEFORE=${MD5_BEFORE%% *}
python -u fc_fit_driver.py prep fit_config.json > prep_stream.log 2>&1
MD5_AFTER=$(md5sum disp_matrix.pkl)
MD5_AFTER=${MD5_AFTER%% *}
echo "[4p50s] disp_matrix md5 before=$MD5_BEFORE after=$MD5_AFTER"
if [ -f sm_prime.npz ] && [ "$MD5_BEFORE" = "$MD5_AFTER" ]; then
    touch sm_prime.npz
    echo "[4p50s] disp_matrix unchanged -> keeping the 57.8 GB sm_prime.npz (no -d rebuild)"
else
    echo "[4p50s] WARNING: disp_matrix changed -> -d rebuild would be needed"
fi

if [ -f fit.log ]; then mv -f fit.log "fit_prev_stream.log"; fi
python -u fc_fit_driver.py fit fit_config.json > fit.log 2>&1 &
FITPID=$!
set +e
wait "$FITPID"; FITRC=$?
set -e
echo "[4p50s] fit rc=$FITRC $(date)"

python -u fc_fit_driver.py post fit_config.json > post.log 2>&1 || echo "[4p50s] post rc=$?"

echo "=== RESULT ==="
grep -E "RFE-GRAM-STREAM|exact Gram CV|SpMV on|SpMV unavailable" fit.log 2>/dev/null | head -12
echo -n "RFE rounds: "; grep -c "\[RFE\] Round" fit.log || true
grep -E "\[RFE\] (Round|argmin|selected)" fit.log | tail -4 || true
ls -la fc2.hdf5 fc3.hdf5 fit_manifest.json fit_metrics.json 2>/dev/null || true
ls "$PHEASY_RFE_GRAM_DIR" 2>/dev/null | head -3 | sed 's/^/[store left] /' || true
echo "[4p50s] done $(date)"
