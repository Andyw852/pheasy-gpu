#!/bin/bash
# Column norms of SM_prime @ NS for c3=5.0, on 32 CPU workers.
#
# Why this exists separately: the streamed GPU column-norm path in the fit
# densifies NS (136,251 x 116,108 float64 = 126 GB) next to the 60 GB SM_prime
# and the job was OOM-killed at --mem=180G on 2026-11-19 (rc=137).  This script
# loads the sparse factors only (SM_prime 60.5 GB after narrowing the column
# indices to int32, NS 12.6M nnz) and writes colnorms_5p00.npy, which the fit
# picks up through PHEASY_COL_NORMS_FILE.
#SBATCH --partition=cpu192
#SBATCH --job-name=colnorm_5p00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --output=colnorm5p.out
#SBATCH --error=colnorm5p.err
set -u
CASE=${CASE:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$CASE" || exit 1
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate mace-gpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "[colnorm5p] start $(date) host=$(hostname)"
df -BG --output=avail "$CASE" | tail -1
python -u "$CASE/colnorm_parallel.py" "$CASE" "$CASE/colnorms_5p00.npy" 32
RC=$?
echo "[colnorm5p] rc=$RC $(date)"
ls -la colnorms_5p00.npy 2>/dev/null || echo "  NO colnorms file"
