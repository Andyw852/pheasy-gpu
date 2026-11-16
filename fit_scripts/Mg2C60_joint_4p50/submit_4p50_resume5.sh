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
# needs 3: nnz/3 = 1.60e9 < 2**31 (int32 CUDA CSR valid), low-mem layout 12.8 GB
# (12.0 GiB) resident per card, 13.2 GiB with the 10% workspace the pre-flight
# reserves.  3 cards = 3 shards, i.e. the one-block-per-card path: this script
# does not exercise the n_shards > n_cards code (dev/check_gpu_sm_shards.py does).
#
# Reuses sm_prime.npz (57.8 GB, 19:33) and ns_*.npz (15:22): prep's disp_matrix.pkl
# is md5-checked and sm_prime.npz is touched only when the content is identical.
# The precomputed column norms are used only together with a reused sm_prime.npz.
#
# Amended after the production run (which used build 73cc175): CASE / conda are
# parameters instead of a hard-coded home directory, the build pin is a feature
# check, and a stale colnorms file is moved aside when sm_prime.npz is rebuilt.
# Run from the case directory (sbatch sets SLURM_SUBMIT_DIR), or export CASE.
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
# Without this the driver calls _clear_pheasy_sm_cache() BEFORE the -s probe and
# deletes the 57.8 GB sm_prime.npz, so the md5/touch guard below never gets a
# chance and the -d pass runs again (4-5 h).  The 2026-11-16 production run
# (job 7619) lost the reuse exactly this way: env unset -> rebuild at 20:34.
export FCFIT_REUSE_SM=1
# PHEASY_COL_NORMS_FILE is exported below, only when sm_prime.npz is reused.
COLNORMS=$CASE/colnorms_4p50.npy
unset PHEASY_COL_NORMS_FILE

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
def md5(mod):
    return hashlib.md5(open(mod.__file__, "rb").read()).hexdigest()
print("pheasy pkg:", _pg.__file__)
print("optimizer md5:", md5(o), " gpu_backend md5:", md5(gb))
# Feature checks, not a build pin: the production run pinned the 73cc175
# optimizer.py md5, which later builds share and which says nothing about
# gpu_backend.py.  The md5s above go to the log for the record.
print("row_shard_count (more shards than cards):", hasattr(gb, "row_shard_count"))
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
    if [ -f "$COLNORMS" ]; then
        export PHEASY_COL_NORMS_FILE=$COLNORMS
        echo "[4p50r5] column norms from $COLNORMS"
    else
        echo "[4p50r5] no $COLNORMS -> the fit computes the column norms itself"
    fi
else
    echo "[4p50r5] WARNING: disp_matrix changed -> -d rebuild would be needed"
    # The precomputed norms belong to the OLD sm_prime.npz and the loader checks
    # only their shape, so a stale file would be accepted silently (rule 4).
    if [ -f "$COLNORMS" ]; then
        mv -f "$COLNORMS" "$COLNORMS.stale"
        echo "[4p50r5] moved $COLNORMS aside (.stale); rerun submit_colnorm.sh after this rebuild"
    fi
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
