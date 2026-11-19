#!/bin/bash
# Mg2C60 joint RFE-OLS fit, cut3 = 5.0 A, on the A800 cluster.
#
# Why A800: the c3=5.0 sensing matrix has nnz = 7.5438e9, i.e. 60.35 GB in the
# float32/int32 layout.  The production layout stores BOTH the row blocks and
# their transposes (the adjoint is then copy-free), so it needs 120.7 GB of VRAM;
# five 24 GB RTX3090 cards have ~117 GB and cannot hold it, and the low-memory
# layout OOMed in its first rmatvec on 2026-11-19 (torch materialises a CSR copy
# of every block inside "R.t() @ u" and the per-block copies do not share a
# buffer).  An A800 has 80 GB per card: three cards give 40.2 GB/card = 37.5 GiB.
#
# The fit does NOT recompute anything expensive here: disp_matrix.pkl, cs.pkl,
# ns_*.npz, colnorms_5p00.npy and sm_prime.npz (90.5 GB) are the artefacts of
# the c3=5.0 series on the 3090 (identical POSCAR/SPOSCAR md5s, identical
# dataset md5s), so FCFIT_REUSE_SM=1 + FCFIT_KEEP_CS=1 skip -d (5 h) and -c
# (2.5 h).  Only the ~1:41 order probe runs.
#SBATCH --partition=a800
#SBATCH --gres=gpu:3
#SBATCH --job-name=Mg2C60_c5_5p00_a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --output=queue_a800.out
#SBATCH --error=queue_a800.err
set -u
CASE=${CASE:-/fs0/home/wangcch/taskflow/work/Mg2C60_joint_5p00}
cd "$CASE" || exit 1
source /fs0/home/wangcch/miniconda3/etc/profile.d/conda.sh
conda activate mace
unset PYTHONPATH

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
# three A800 cards, stored-transpose layout (no per-call copy)
export PHEASY_USE_GPU=1
export PHEASY_GPU_SM=1
export PHEASY_GPU_SM_NGPU=3
export PHEASY_GPU_SM_DEVICES=0,1,2
export PHEASY_GPU_DEVICES=0,1,2
export PHEASY_GPU_SM_LOWMEM=0
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

echo "[5p00a] start $(date) host=$(hostname) c3=5.0 A800 3 cards"

echo "=== preflight ==="
python - <<PYCHK
import hashlib, sys
import pheasy_gpu as _pg
import pheasy_gpu.core.optimizer as o
import pheasy_gpu.core.gpu_backend as g
def md5(mod):
    return hashlib.md5(open(mod.__file__, "rb").read()).hexdigest()
print("pheasy pkg:", _pg.__file__)
print("optimizer md5:", md5(o), " gpu_backend md5:", md5(g))
if md5(o) != "e348c1f73eda8fb57b1ea4db9dc3b7f1":
    sys.exit("[ABORT] core/optimizer.py is not main a5ea2c0 (e348c1f7)")
if md5(g) != "5dc85969c789e9fca599a4bfdcfef421":
    sys.exit("[ABORT] core/gpu_backend.py is not the sharding build")
for name in ("sm_prime.npz", "disp_matrix.pkl", "cs.pkl", "ns_harm.npz",
             "ns_anharm3.npz", "colnorms_5p00.npy", "fc_dataset.json"):
    import os
    print("  %-22s %.1f MB" % (name, os.path.getsize(name) / 1e6))
print("row_shard_count(%.4e, cards):" % 7.543798128e9)
for n in (3, 4, 5):
    print("   %d cards -> %d shards" % (n, o.row_shard_count(7543798128, n)))
print("HARM_DENSE OK:", o.harm_dense_setting("RFE-OLS"))
PYCHK
[ $? -ne 0 ] && exit 1

set -e
echo "=== prep (dataset is byte-identical to the 3090 series) ==="
MD5_BEFORE=$(md5sum disp_matrix.pkl); MD5_BEFORE=${MD5_BEFORE%% *}
python -u fc_fit_driver.py prep fit_config.json > prep_a800.log 2>&1 || echo "[5p00a] prep rc=$?"
MD5_AFTER=$(md5sum disp_matrix.pkl); MD5_AFTER=${MD5_AFTER%% *}
echo "[5p00a] disp_matrix md5 before=$MD5_BEFORE after=$MD5_AFTER"
if [ -f sm_prime.npz ] && [ "$MD5_BEFORE" = "$MD5_AFTER" ]; then
    touch sm_prime.npz
    echo "[5p00a] matrix unchanged -> sm_prime.npz (90.5 GB) reusable"
fi

if [ -f fit.log ]; then mv -f fit.log fit_prev.log; fi
python -u fc_fit_driver.py fit fit_config.json > fit.log 2>&1 &
FITPID=$!
set +e
wait "$FITPID"; FITRC=$?
set -e
echo "[5p00a] fit rc=$FITRC $(date)"
python -u fc_fit_driver.py post fit_config.json > post.log 2>&1 || echo "[5p00a] post rc=$?"

echo "=== RESULT ==="
grep -E "FCFIT_KEEP_CS|reusing sm_prime|SpMV on|RFE. START" fit.log 2>/dev/null | head -6
echo -n "RFE rounds: "; grep -c "\[RFE\] Round" fit.log || true
grep -E "\[RFE\] (Round|final exact refit)" fit.log | tail -4 || true
ls -la fc2.hdf5 fc3.hdf5 fit_manifest.json fit_metrics.json 2>/dev/null || true
echo "[5p00a] done $(date)"
