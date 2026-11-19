#!/bin/bash
# Mg2C60 joint RFE-OLS fit, cut3 = 5.0 A, on the RTX 3090 box -- the TIGHT fit.
#
# nnz = 7,543,798,128 -> the stored-transpose layout needs
#   2 x nnz x 8 B / 5 cards = 24.14 GB = 22.48 GiB per card,
# against 23.18-23.43 GiB free on an otherwise idle 24 GB card.  That is a
# 0.7-0.95 GiB margin, and the 10% workspace reserve in the pre-flight
# (_cuda_spmv_block_budget_counts) turns it into a refusal, so this run sets
# PHEASY_GPU_SM_WS_FRAC=0.01 (cuSPARSE's SpMV workspace really is a few MB; the
# reserve is a guard, not a measurement).  Validate the layout first with
#   PHEASY_GPU_SM_WS_FRAC=0.01 python dev/check_gpu_sm_shards.py \
#     --npz $CASE/sm_prime.npz --devices 0,1,2,3,4 --shards 5
# The low-memory layout cannot help: it is retired (torch copies every block in
# "R.t() @ u"; jobs 7620/7664 OOMed there).
#
# Nothing expensive is recomputed: sm_prime.npz (90.5 GB), cs.pkl, ns_*.npz and
# colnorms_5p00.npy are all on disk, so FCFIT_REUSE_SM=1 + FCFIT_KEEP_CS=1 skip
# -d (5 h) and -c (2:30 h); only the ~1:41 order probe runs.
#SBATCH --partition=cpu192
#SBATCH --gres=gpu:5
#SBATCH --job-name=Mg2C60_c5_5p00_5c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --output=queue5c.out
#SBATCH --error=queue5c.err
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
export PHEASY_GPU_SM_NGPU=5
export PHEASY_GPU_SM_DEVICES=0,1,2,3,4
export PHEASY_GPU_DEVICES=0,1,2,3,4
export PHEASY_GPU_SM_LOWMEM=0
export PHEASY_GPU_SM_WS_FRAC=0.01
export PHEASY_GPU_MODE=required
export PHEASY_GPU_FALLBACK=0
export PHEASY_TWOLEVEL_CACHE_T=0
export PHEASY_LASSO_TWOLEVEL=1
export PHEASY_LASSO_SPARSE=1
export PHEASY_GPU_LASSO_RESIDENT=0
export PHEASY_GPU_OLS_RESIDENT=0
export PHEASY_EXACT_GRAM_GB=65
export PHEASY_EXACT_GRAM_ALL=1
export PHEASY_GPU_MEM_FRACTION=1.0
export PHEASY_TWOLEVEL_ADJOINT=lazy
export PHEASY_HARM_DENSE=1
export PHEASY_CV_GROUP_SIZE=1488
export PHEASY_SM_DTYPE=float32
export PHEASY_SM_INDEX_DTYPE=int32
export PHEASY_COLNORM_CPU_FALLBACK=1
export PHEASY_COLNORM_CPU_MAX_WORK=400000000000
export FCFIT_REUSE_SM=1
export FCFIT_KEEP_CS=1
COLNORMS=$CASE/colnorms_5p00.npy

echo "[5p00c] start $(date) host=$(hostname) c3=5.0 5 cards stored-transpose ws_frac=0.01"

echo "=== preflight ==="
python - <<PYCHK
import hashlib, os, sys
import torch
import pheasy_gpu as _pg
import pheasy_gpu.core.optimizer as o
import pheasy_gpu.core.gpu_backend as g
def md5(mod):
    return hashlib.md5(open(mod.__file__, "rb").read()).hexdigest()
print("pheasy pkg:", _pg.__file__)
print("optimizer md5:", md5(o), " gpu_backend md5:", md5(g))
if md5(o) != "e348c1f73eda8fb57b1ea4db9dc3b7f1":
    sys.exit("[ABORT] optimizer is not the c3=5.0-tested build")
for name in ("sm_prime.npz", "disp_matrix.pkl", "cs.pkl", "ns_harm.npz",
             "ns_anharm3.npz", "colnorms_5p00.npy", "fc_dataset.json"):
    print("  %-22s %.1f MB" % (name, os.path.getsize(name) / 1e6))
print("row_shard_count(7.5438e9, 5) =", o.row_shard_count(7543798128, 5))
print("per-card resident = 2 x 7.5438e9 x 8 / 5 = %.2f GiB"
      % (2 * 7543798128 * 8 / 5 / 2**30))
print("WS_FRAC =", os.environ.get("PHEASY_GPU_SM_WS_FRAC"))
free = []
for i in range(torch.cuda.device_count()):
    f, _t = torch.cuda.mem_get_info(i)
    free.append(f / 2**30)
    print("  gpu%d free %.2f GiB" % (i, f / 2**30))
need = 2 * 7543798128 * 8 / 5 / 2**30 + 0.25
if min(free) < need:
    sys.exit("[ABORT] every card needs >= %.2f GiB free, min is %.2f" % (need, min(free)))
print("HARM_DENSE OK:", o.harm_dense_setting("RFE-OLS"))
PYCHK
[ $? -ne 0 ] && exit 1

echo "=== prep ==="
MD5_BEFORE=$(md5sum disp_matrix.pkl); MD5_BEFORE=${MD5_BEFORE%% *}
python -u fc_fit_driver.py prep fit_config.json > prep_5c.log 2>&1 || echo "[5p00c] prep rc=$?"
MD5_AFTER=$(md5sum disp_matrix.pkl); MD5_AFTER=${MD5_AFTER%% *}
echo "[5p00c] disp_matrix md5 before=$MD5_BEFORE after=$MD5_AFTER"
if [ -f sm_prime.npz ] && [ "$MD5_BEFORE" = "$MD5_AFTER" ]; then
    touch sm_prime.npz
    echo "[5p00c] matrix unchanged -> sm_prime.npz (90.5 GB) reusable"
fi
if [ -f "$COLNORMS" ]; then
    export PHEASY_COL_NORMS_FILE=$COLNORMS
    echo "[5p00c] column norms from $COLNORMS"
fi

if [ -f fit.log ]; then mv -f fit.log fit_prev.log; fi
python -u fc_fit_driver.py fit fit_config.json > fit.log 2>&1 &
FITPID=$!
set +e
wait "$FITPID"; FITRC=$?
set -e
echo "[5p00c] fit rc=$FITRC $(date)"
python -u fc_fit_driver.py post fit_config.json > post.log 2>&1 || echo "[5p00c] post rc=$?"

echo "=== RESULT ==="
grep -E "FCFIT_KEEP_CS|reusing sm_prime|SpMV on|SpMV unavailable|RFE. START" fit.log 2>/dev/null | head -6
echo -n "RFE rounds: "; grep -c "\[RFE\] Round" fit.log || true
grep -E "\[RFE\] (Round|final exact refit|selected)" fit.log | tail -4 || true
ls -la fc2.hdf5 fc3.hdf5 fit_manifest.json fit_metrics.json 2>/dev/null || true
echo "[5p00c] done $(date)"
