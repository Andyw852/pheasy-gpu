#!/bin/bash
set -u
T=<gpu-host>:/data/taskflow/work/Mg4C60_c3scan_ols/Mg4C60_joint_3p50_exactgram
cd "$T" || exit 1
source <gpu-host>:/home/miniconda3/etc/profile.d/conda.sh
conda activate mace-gpu

# ---------------------------------------------------------------------------
# Faithful reproduction of the c3_3p50 joint-RFE environment (the run that
# completed 11 RFE rounds in ~2h40m on 2026-10-23), with one difference:
# this box now has ~244 GB available instead of the 17.1 GB that forced the
# final refit onto the iterative LSMR fallback.
#
# Why the earlier attempts were slow: they exported PHEASY_RFE_GRAM_GB=21
# (a FIX_FC2-path knob) but NOT PHEASY_EXACT_GRAM_GB / PHEASY_EXACT_GRAM_ALL,
# so Gram could not be built and every RFE round fell back to iterative LSMR
# at the float32 precision floor -> >7 h per round, zero rounds in 8.7 h.
# The c3_3p50 recipe below is what actually works for the joint case.
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=8
export PHEASY_USE_GPU=1
export PHEASY_GPU_MODE=required
export PHEASY_GPU_SM=1
export PHEASY_GPU_SM_NGPU=3
export PHEASY_GPU_SM_DEVICES=0,1,2
export PHEASY_GPU_DEVICES=0,1,2
export PHEASY_TWOLEVEL_CACHE_T=0
export PHEASY_LASSO_TWOLEVEL=1
export PHEASY_LASSO_SPARSE=1
export PHEASY_GPU_LASSO_RESIDENT=0
export PHEASY_GPU_OLS_RESIDENT=0
export PHEASY_GPU_FALLBACK=1
# exact Gram for every CV fold (the key to the joint case)
export PHEASY_EXACT_GRAM_GB=75
export PHEASY_EXACT_GRAM_ALL=1
export PHEASY_RFE_1SE=1
export FCFIT_SCAN_ALLOW_NONSTANDARD=1
export PYTHONUNBUFFERED=1

DRV=fc_fit_driver.py
echo "[joint-eg] start $(date)"
echo "[joint-eg] driver md5=$(md5sum $DRV | cut -d' ' -f1) (frozen production f069d1aa)"
echo "[joint-eg] EXACT_GRAM_GB=$PHEASY_EXACT_GRAM_GB EXACT_GRAM_ALL=$PHEASY_EXACT_GRAM_ALL"
echo "[joint-eg] mem available: $(free -g | awk '/^Mem:/{print $7" GiB"}')"
echo "[joint-eg] config: fix_fc2=$(python -c 'import json;c=json.load(open("fit_config.json"));print(c.get("fix_fc2","<absent>"))') c3=$(python -c 'import json;print(json.load(open("fit_config.json"))["pheasy_c3_cutoff"])') cv=$(python -c 'import json;print(json.load(open("fit_config.json"))["pheasy_cv"])')"
python -u $DRV prep fit_config.json > joint_eg_prep.log 2>&1; echo "[joint-eg] prep rc=$? $(date)"
python -u $DRV fit  fit_config.json > joint_eg_fit.log  2>&1; echo "[joint-eg] fit  rc=$? $(date)"
echo "--- artifacts ---"
ls -la rfe_support.npy fit_manifest.json fit_metrics.json fc2.hdf5 fc3.hdf5 2>/dev/null
python -c "import numpy as np; a=np.load('rfe_support.npy'); print('support_n =', a.shape)" 2>/dev/null
echo "[joint-eg] done $(date)"
