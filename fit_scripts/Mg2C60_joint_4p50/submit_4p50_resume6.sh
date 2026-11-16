#!/bin/bash
# Mg2C60 joint RFE-OLS fit at cut3 = 4.5 A -- resume with a RUNTIME card choice.
#
# Why r6: job 7619 picked its three cards at 14:41 and by 00:30 another user's
# non-SLURM campaign (zls) held 16.4 GiB on one of them, so the 3-shard layout
# (13.2 GiB/card) cannot be built there.  A new job picks again.  Two layouts are
# supported by the sharding fix (main 9a7064d):
#   3 cards -> 3 shards, 1.60e9 nnz/block, 13.2 GiB/card
#   2 cards -> 4 shards, 1.20e9 nnz/block, 19.8 GiB/card (needs LOWMEM=1 and an
#              explicit device list, because the auto filter sizes a stored
#              transpose at 35.9 GiB/card)
# Selection: 3 cards with >=14 GiB free, else 2 cards with >=20.5 GiB free, else
# abort (PHEASY_GPU_FALLBACK=0: never run the fit on one CPU core).
#
# sm_prime.npz (57.8 GB) and ns_*.npz are reused: FCFIT_REUSE_SM=1 keeps the
# matrix, and the md5 pair below touches it only when prep leaves disp_matrix.pkl
# byte-identical.
#SBATCH --partition=cpu192
#SBATCH --gres=gpu:5
#SBATCH --job-name=Mg2C60_c4_4p50_r6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --output=queue6.out
#SBATCH --error=queue6.err
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
export PHEASY_COLNORM_CPU_MAX_WORK=200000000000
export PHEASY_HARM_DENSE=1
export PHEASY_CV_GROUP_SIZE=1488
export PHEASY_SM_DTYPE=float32
# keep the sensing matrix's column indices int32 while it is assembled too: the
# -d vstack otherwise builds an int64 copy (57.8 GB) beside the 699 block list
# and the node swaps when other users hold memory (seen 2026-11-17 00:35).
export PHEASY_SM_INDEX_DTYPE=int32
export FCFIT_REUSE_SM=1
COLNORMS=$CASE/colnorms_4p50.npy
unset PHEASY_COL_NORMS_FILE

echo "[4p50r6] start $(date) host=$(hostname) c3=4.5 fallback=0 lowmem=1"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-?}"

SEL=$(python - <<PYGPU
import sys, torch
limit3, limit2 = 14 * 2**30, 20.5 * 2**30
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
pick = [i for f, i in free if f >= limit3][:3]
if len(pick) == 3:
    print("3:" + ",".join(str(i) for i in pick))
else:
    pick = [i for f, i in free if f >= limit2][:2]
    print("2:" + ",".join(str(i) for i in pick) if len(pick) == 2 else "NO_GPUS")
PYGPU
)
echo "[gpu] selection: $SEL"
NGPU=${SEL%%:*}
DEVS=${SEL#*:}
if [ "$SEL" = "NO_GPUS" ] || [ -z "${DEVS:-}" ]; then
    echo "[ABORT] not enough free VRAM: need 3 cards >=14 GiB or 2 cards >=20.5 GiB"
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
print("row_shard_count(4.816e9, 2):", gb.row_shard_count(4816356048, 2))
if not hasattr(gb, "row_shard_count") or not hasattr(o, "_gpu_required"):
    sys.exit("[ABORT] this build lacks the required-mode fix (main >= 02ee2eb)")
if not hasattr(o, "harm_dense_setting"):
    sys.exit("[ABORT] no HARM_DENSE")
print("HARM_DENSE OK:", o.harm_dense_setting("RFE-OLS"))
PYCHK
[ $? -ne 0 ] && exit 1

set -e
MD5_BEFORE=$(md5sum disp_matrix.pkl)
MD5_BEFORE=${MD5_BEFORE%% *}
python -u fc_fit_driver.py prep fit_config.json > prep6.log 2>&1
MD5_AFTER=$(md5sum disp_matrix.pkl)
MD5_AFTER=${MD5_AFTER%% *}
echo "[4p50r6] disp_matrix md5 before=$MD5_BEFORE after=$MD5_AFTER"
if [ -f sm_prime.npz ] && [ "$MD5_BEFORE" = "$MD5_AFTER" ]; then
    touch sm_prime.npz
    echo "[4p50r6] disp_matrix unchanged -> keeping sm_prime.npz (no -d rebuild)"
    if [ -f "$COLNORMS" ]; then
        export PHEASY_COL_NORMS_FILE=$COLNORMS
        echo "[4p50r6] column norms from $COLNORMS"
    fi
else
    echo "[4p50r6] WARNING: disp_matrix changed -> -d rebuild would be needed"
fi

if [ -f fit.log ]; then mv -f fit.log "fit_prev_r5.log"; fi
python -u fc_fit_driver.py fit fit_config.json > fit.log 2>&1 &
FITPID=$!
set +e
wait "$FITPID"; FITRC=$?
set -e
echo "[4p50r6] fit rc=$FITRC $(date)"

python -u fc_fit_driver.py post fit_config.json > post.log 2>&1 || echo "[4p50r6] post rc=$?"

echo "=== RESULT ==="
grep -E "SpMV on|SpMV unavailable|column norms loaded|no -d rebuild" fit.log | head -5
echo -n "RFE rounds: "; grep -c "\[RFE\] Round" fit.log || true
grep -E "\[RFE\] (Round|final exact refit: optimality)" fit.log | tail -4 || true
ls -la fc2.hdf5 fc3.hdf5 fit_manifest.json fit_metrics.json 2>/dev/null || true
echo "[4p50r6] done $(date)"
