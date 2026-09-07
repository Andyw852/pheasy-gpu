#!/bin/bash
#SBATCH --job-name=pheasy-validation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:6
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=pheasy-validation-%j.out
# Required: PHEASY_CHECKOUT (checkout path), PHEASY_DATA (prebuilt data path).
# Use an activated Python environment that imports this checkout as pheasy_gpu.
set -euo pipefail
: "${PHEASY_CHECKOUT:?Set PHEASY_CHECKOUT to the source checkout}"
: "${PHEASY_DATA:?Set PHEASY_DATA to the prepared 699-configuration directory}"
PHEASY_RESULTS=${PHEASY_RESULTS:-$SLURM_SUBMIT_DIR/validation_results}
PHEASY_PYTHON=${PHEASY_PYTHON:-python}
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$PHEASY_PYTHON" "$PHEASY_CHECKOUT/dev/validate_large_fit.py" full "$PHEASY_DATA" \
    --output "$PHEASY_RESULTS/full"
"$PHEASY_PYTHON" "$PHEASY_CHECKOUT/dev/validate_large_fit.py" holdout "$PHEASY_DATA" \
    --output "$PHEASY_RESULTS/holdout"
