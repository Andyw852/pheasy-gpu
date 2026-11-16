#!/bin/bash
# Mg2C60 joint RFE-OLS fit at cut3 = 4.5 A -- 3 free GPUs, FAIL-FAST on GPU problems.
#
# History: 7594 (2 GPUs) died on the int32 nnz limit, 7601 (4 GPUs) died because
# two of its cards were actually occupied by another user's non-SLURM campaign
# (zls/campaign.py, 16-19 GiB each) -> GpuSparseMV raised -> the code printed
# "[GPU-SM] SpMV unavailable; using CPU" and ran the fit on ONE CPU core (weeks).
# PHEASY_GPU_FALLBACK=1 is what allowed that silent downgrade; this script sets it
# to 0 so a GPU problem aborts the job instead of burning days of CPU.  Cards are
# selected by ACTUAL free VRAM (the shared box lies to SLURM GRES) and the run
# needs 3: nnz/3 = 1.60e9 < 2**31 (int32 CUDA CSR valid), low-mem layout ~12.9 GiB
# per card.
#
# Reuses sm_prime.npz (57.8 GB, 19:33) and ns_*.npz (15:22): prep's disp_matrix.pkl
# is md5-checked and sm_prime.npz is touched only when the content is identical.
#SBATCH --partition=cpu192
#SBATCH --gres=gpu:5
#SBATCH --job-name=Mg2C60_c4_4p50_r5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --output=queue5.out
#SBATCH --error=queue5.err
set -u
CASE=/home/wangchaoyue852/taskflow/work/Mg2C60_joint_4p50
cd "$CASE" || exit 1
source /home/wangchaoyue852/miniconda3/etc/profile.d/conda.sh
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
export PHEASY_COL_NORMS_FILE=$CASE/colnorms_4p50.npy

echo "[4p50r5] start $(date) host=$(hostname) c3=4.5 fallback=0 lowmem=1"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-?}"

SEL=$(python - <<PYGPU
import sys, torch
need = 15 * 2**30
free = []
for i in range(torch.cuda.device_count()):
    try:
        f, t = torch.cuda.mem_get_info(i)
    except Exception as e:
        print("# gpu%d unqueryable: %s" % (i, e), file=sys.stderr)
        continue
    print("# gpu%d free %.1f GiB" % (i, f / 2**30), file=sys.stderr)
    if f >= need:
        free.append((f, i))
free.sort(reverse=True)
picked = [i for _f, i in free[:3]]
print("NO_GPUS" if len(picked) < 3 else ",".join(str(i) for i in picked))
PYGPU
)
echo "[gpu] picked: $SEL"
if [ "$SEL" = "NO_GPUS" ] || [ -z "$SEL" ]; then
    echo "[ABORT] fewer than 3 cards with >=15 GiB free VRAM; refusing to run the fit on the CPU"
    exit 3
fi
NGPU=$(echo "$SEL" | awk -F, '{print NF}')
echo "[gpu] NGPU=$NGPU devs=$SEL"
export PHEASY_GPU_SM_NGPU=$NGPU
export PHEASY_GPU_SM_DEVICES="$SEL"
export PHEASY_GPU_DEVICES="$SEL"

echo "=== preflight ==="
python - <<PYCHK
import hashlib, sys
import pheasy_gpu as _pg
import pheasy_gpu.core.optimizer as o
import pheasy_gpu.core.gpu_backend as gb
h = hashlib.md5(open(o.__file__, "rb").read()).hexdigest()
print("pheasy pkg:", _pg.__file__)
print("optimizer md5:", h)
if h != "c950eca85a84d2eb978d312e2ed8fb81":
    sys.exit("[ABORT] expected the 73cc175 build; got a different one")
if not hasattr(gb, "_csr_column_block_t"):
    sys.exit("[ABORT] lazy-adjoint fix missing")
if not hasattr(o, "harm_dense_setting"):
    sys.exit("[ABORT] no HARM_DENSE")
print("HARM_DENSE OK:", o.harm_dense_setting("RFE-OLS"))
PYCHK
[ $? -ne 0 ] && exit 1

set -e
MD5_BEFORE=$(md5sum disp_matrix.pkl)
MD5_BEFORE=${MD5_BEFORE%% *}
python -u fc_fit_driver.py prep fit_config.json > prep5.log 2>&1
MD5_AFTER=$(md5sum disp_matrix.pkl)
MD5_AFTER=${MD5_AFTER%% *}
echo "[4p50r5] disp_matrix md5 before=$MD5_BEFORE after=$MD5_AFTER"
if [ -f sm_prime.npz ] && [ "$MD5_BEFORE" = "$MD5_AFTER" ]; then
    touch sm_prime.npz
    echo "[4p50r5] disp_matrix unchanged -> keeping the 57.8 GB sm_prime.npz (no -d rebuild)"
else
    echo "[4p50r5] WARNING: disp_matrix changed -> -d rebuild would be needed"
fi

if [ -f fit.log ]; then mv -f fit.log "fit_prev_r4.log"; fi
python -u fc_fit_driver.py fit fit_config.json > fit.log 2>&1 &
FITPID=$!
set +e
wait "$FITPID"; FITRC=$?
set -e
echo "[4p50r5] fit rc=$FITRC $(date)"

python -u fc_fit_driver.py post fit_config.json > post.log 2>&1 || echo "[4p50r5] post rc=$?"

echo "=== RESULT ==="
grep -E "SpMV on|SpMV unavailable|column norms loaded|no -d rebuild|will be rebuilt" fit.log 2>/dev/null | head -6
echo -n "RFE rounds: "; grep -c "\[RFE\] Round" fit.log || true
grep -E "\[RFE\] Round" fit.log | tail -3 || true
ls -la fc2.hdf5 fc3.hdf5 fit_manifest.json fit_metrics.json 2>/dev/null || true
echo "[4p50r5] done $(date)"
