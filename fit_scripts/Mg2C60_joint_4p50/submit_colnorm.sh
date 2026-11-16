#!/bin/bash
# Precompute the exact column norms of SM_prime @ NS on CPU, in parallel row blocks.
# ~2.7 min with 32 workers instead of ~9 h serial (c3=4.5, nnz=4.82e9).
# The fit then reads them via PHEASY_COL_NORMS_FILE instead of running the
# (streamed, single-device) GPU column-norm pass inside the RFE-OLS setup.
# The norms belong to THIS sm_prime.npz: rerun after every -d rebuild
# (submit_4p50_resume5.sh moves the old file aside when it sees one coming).
# Run from the case directory (sbatch sets SLURM_SUBMIT_DIR), or export CASE.
#SBATCH --partition=cpu192
#SBATCH --job-name=colnorm_4p50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=colnorm.out
#SBATCH --error=colnorm.err
CASE=${CASE:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$CASE" || exit 1
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate mace-gpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u "$CASE/colnorm_parallel.py" "$CASE" "$CASE/colnorms_4p50.npy" 32
