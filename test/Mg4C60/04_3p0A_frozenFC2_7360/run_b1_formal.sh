#!/bin/bash
set -u
C=<gpu-host>:/data/taskflow/work/Mg4C60_c3scan_ols/Mg4C60_fixfc_3p00_confirm
cd $C || exit 1
source <gpu-host>:/home/miniconda3/etc/profile.d/conda.sh
conda activate mace-gpu
export OMP_NUM_THREADS=48
echo "[b1] start $(date) driver md5 $(md5sum fc_fit_driver.py | cut -d' ' -f1)"
python -u fc_fit_driver.py prep fit_config.json > b1_prep.log 2>&1; echo "[b1] prep rc=$? $(date)"
python -u fc_fit_driver.py fit fit_config.json > b1_fit.log 2>&1; echo "[b1] fit rc=$? $(date)"
python -u fc_fit_driver.py post fit_config.json > b1_post.log 2>&1; echo "[b1] post rc=$? $(date)"
grep -E "RFE\\] START|exact Gram|RFE\\] Round|least-squares|FIX_FC2|self-check|ARTIFACT|FAIL|scan standard" b1_fit.log b1_post.log | tail -25
echo "[b1] done $(date)"
