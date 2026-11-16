#!/bin/bash
# Mg2C60 joint RFE-OLS fit at cut3 = 5.0 A -- resume with a RUNTIME card choice.
#
# The 5.0 A sensing matrix is already built and banked:
#   sm_prime.npz 90.5 GB (2026-11-15 15:09), SM_prime (1040112 x 136251),
#   nnz = 7,543,798,128, float32 values with int64 indices (narrowed to int32 on
#   load -> ~60 GB resident).  ns_harm/ns_anharm3 and the disp/force pickles are
#   all present, so FCFIT_REUSE_SM=1 + the md5/touch pair below skip the -d pass
#   (4-6 h) entirely; the job pays -c (~40 min) + the RFE only.
#
# Sizing (low-memory layout: one row block per card, 8 B/nonzero on the device):
#   4 cards -> 1.886e9 nnz/block, 14.1 GiB/card
#   5 cards -> 1.509e9 nnz/block, 11.2 GiB/card
#   3 cards would need 2 shards/card (row_shard_count(7.54e9,3)=6) -> 20.6 GiB,
#   too tight on a shared card, so the selector refuses it.
# Host side: SM_prime ~60 GB in RAM, the final exact refit needs a p_selected^2
# float64 Gram (PHEASY_EXACT_GRAM_GB=65, blocks of 20000) -> --mem=180G.
#
# Selection: 5 cards with >=13 GiB free, else 4 cards with >=16 GiB free, else
# abort (PHEASY_GPU_FALLBACK=0: never run the fit on one CPU core).
#SBATCH --partition=cpu192
#SBATCH --gres=gpu:5
#SBATCH --job-name=Mg2C60_c5_5p00_r
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --output=queue5p.out
#SBATCH --error=queue5p.err
set -u
CASE=${CASE:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$CASE" || exit 1
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate mace-gpu
unset PYTHONPATH

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export PHEASY_USE_GPU=1
export PHEASY_GPU_SM=1
export PHEASY_GPU_FALLBACK=0
export PHEASY_GPU_SM_LOWMEM=1
export PHEASY_TWOLEVEL_CACHE_T=0
export PHEASY_LASSO_TWOLEVEL=1
export PHEASY_LASSO_SPARSE=1
export PHEASY_GPU_MODE=required
export PHEASY_GPU_LASSO_RESIDENT=0
export PHEASY_GPU_OLS_RESIDENT=0
export PHEASY_EXACT_GRAM_GB=65
export PHEASY_EXACT_GRAM_ALL=1
export PHEASY_GPU_MEM_FRACTION=1.0
export PHEASY_TWOLEVEL_ADJOINT=lazy
export PHEASY_COLNORM_CPU_FALLBACK=1
export PHEASY_COLNORM_CPU_MAX_WORK=400000000000
export PHEASY_HARM_DENSE=1
export PHEASY_CV_GROUP_SIZE=1488
export PHEASY_SM_DTYPE=float32
# keep int32 column indices while assembling/loading: the npz stores int64
# (90.5 GB) and narrowing at load halves the resident matrix.
export PHEASY_SM_INDEX_DTYPE=int32
export FCFIT_REUSE_SM=1
COLNORMS=$CASE/colnorms_5p00.npy
unset PHEASY_COL_NORMS_FILE

echo "[5p00r] start $(date) host=$(hostname) c3=5.0 fallback=0 lowmem=1"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-?}"

SEL=$(python - <<PYGPU
import sys, torch
lim5, lim4 = 13 * 2**30, 16 * 2**30
free = []
for i in range(torch.cuda.device_count()):
    try:
        f, t = torch.cuda.mem_get_info(i)
    except Exception as e:
        print("# gpu%d unqueryable: %s" % (i, e), file=sys.stderr)
        continue
    print("# gpu%d used %.1f GiB free %.1f GiB" % (i, (t - f) / 2**30, f / 2**30),
          file=sys.stderr)
    free.append((f, i))
free.sort(reverse=True)
pick = [i for f, i in free if f >= lim5][:5]
if len(pick) >= 5:
    print("5:" + ",".join(str(i) for i in pick[:5]))
else:
    pick = [i for f, i in free if f >= lim4][:4]
    print("4:" + ",".join(str(i) for i in pick) if len(pick) == 4 else "NO_GPUS")
PYGPU
)
echo "[gpu] selection: $SEL"
NGPU=${SEL%%:*}
DEVS=${SEL#*:}
if [ "$SEL" = "NO_GPUS" ] || [ -z "${DEVS:-}" ]; then
    echo "[ABORT] not enough free VRAM: need 5 cards >=13 GiB or 4 cards >=16 GiB"
    exit 3
fi
echo "[gpu] NGPU=$NGPU devs=$DEVS"
export PHEASY_GPU_SM_NGPU=$NGPU
export PHEASY_GPU_SM_DEVICES="$DEVS"
export PHEASY_GPU_DEVICES="$DEVS"

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
print("row_shard_count(7.5438e9, cards):")
for n in (3, 4, 5):
    print("   %d cards -> %d shards" % (n, gb.row_shard_count(7543798128, n)))
if not hasattr(gb, "row_shard_count") or not hasattr(o, "_gpu_required"):
    sys.exit("[ABORT] this build lacks the required-mode fix (main >= 02ee2eb)")
print("HARM_DENSE OK:", o.harm_dense_setting("RFE-OLS"))
PYCHK
[ $? -ne 0 ] && exit 1

set -e
MD5_BEFORE=$(md5sum disp_matrix.pkl)
MD5_BEFORE=${MD5_BEFORE%% *}
python -u fc_fit_driver.py prep fit_config.json > prep5p.log 2>&1
MD5_AFTER=$(md5sum disp_matrix.pkl)
MD5_AFTER=${MD5_AFTER%% *}
echo "[5p00r] disp_matrix md5 before=$MD5_BEFORE after=$MD5_AFTER"
if [ -f sm_prime.npz ] && [ "$MD5_BEFORE" = "$MD5_AFTER" ]; then
    touch sm_prime.npz
    echo "[5p00r] disp_matrix unchanged -> keeping the 90.5 GB sm_prime.npz (no -d rebuild)"
    if [ -f "$COLNORMS" ]; then
        export PHEASY_COL_NORMS_FILE=$COLNORMS
        echo "[5p00r] column norms from $COLNORMS"
    else
        echo "[5p00r] no $COLNORMS -> the fit computes the column norms (streamed GPU pass)"
    fi
else
    echo "[5p00r] WARNING: disp_matrix changed -> -d rebuild would be needed (~5 h)"
fi

if [ -f fit.log ]; then mv -f fit.log "fit_prev_r.log"; fi
python -u fc_fit_driver.py fit fit_config.json > fit.log 2>&1 &
FITPID=$!
set +e
wait "$FITPID"; FITRC=$?
set -e
echo "[5p00r] fit rc=$FITRC $(date)"

python -u fc_fit_driver.py post fit_config.json > post.log 2>&1 || echo "[5p00r] post rc=$?"

echo "=== RESULT ==="
grep -E "SpMV on|SpMV unavailable|column norms|no -d rebuild" fit.log | head -5
echo -n "RFE rounds: "; grep -c "\[RFE\] Round" fit.log || true
grep -E "\[RFE\] (Round|final exact refit: optimality)" fit.log | tail -4 || true
ls -la fc2.hdf5 fc3.hdf5 fit_manifest.json fit_metrics.json 2>/dev/null || true
echo "[5p00r] done $(date)"
