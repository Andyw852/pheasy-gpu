#!/bin/bash
set -u
T=<gpu-host>:/data/taskflow/work/Mg4C60_c3scan_ols/Mg4C60_fixfc_3p50_confirm
cd "$T" || exit 1
source <gpu-host>:/home/miniconda3/etc/profile.d/conda.sh
conda activate mace-gpu
export OMP_NUM_THREADS=48
export PHEASY_RFE_GRAM_GB=21
echo "[b3] start $(date) driver md5 $(md5sum fc_fit_driver.py | cut -d' ' -f1)"
python -u fc_fit_driver.py prep fit_config.json > b3_prep.log 2>&1; echo "[b3] prep rc=$? $(date)"
python -u fc_fit_driver.py fit fit_config.json > b3_fit.log 2>&1; echo "[b3] fit rc=$? $(date)"
python -u fc_fit_driver.py post fit_config.json > b3_post.log 2>&1; echo "[b3] post rc=$? $(date)"
echo "[b3] done $(date)"
