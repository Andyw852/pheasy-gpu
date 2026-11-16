#!/bin/bash
# fit-fc-thermal S1_fit 模板 —— 3090 版（2026-10-16 按用户指示从 A800 切过来）
# 改动：--partition=gpu / --gres=gpu:1（单卡更易排上，3090 上多卡实测反而慢）/
#       --mem=120G（沿用 3090 凹槽：gpu 分区 DefMemPerCPU=4000，只写 cpus=8 会按
#       32GB 记账，而 fc3 实测峰值 99GB）；保留 HARM_DENSE/GPU 驻留那套开关。
#SBATCH --partition=cpu192
# [2026-10-17] cpu192 分区在 3090 上同样带 gpu:5（见 sinfo），且优先级更高；gpu 分区会被
# cpu192 的作业挤住（"reserved for jobs in higher priority partitions"）。
#SBATCH --gres=gpu:3
#SBATCH --job-name=Mg4C60_c3_HD_3p50
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# [2026-10-17] 3090 节点只剩 4 空闲核：拟合主要用 GPU，4 核足够（原来 8 核排不上）
#SBATCH --mem=180G
#SBATCH --output=queue.out
#SBATCH --error=queue.err
cd $SLURM_SUBMIT_DIR

if [ -n "mace-gpu" ] && [ -x "mace-gpu/bin/python" ]; then
    source "mace-gpu/bin/activate"
else
    source <gpu-host>:/home/miniconda3/etc/profile.d/conda.sh
    conda activate mace-gpu
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PHEASY_USE_GPU=1
export PHEASY_GPU_SM=1
export PHEASY_GPU_SM_NGPU=3
export PHEASY_GPU_SM_DEVICES=0,1,2
export PHEASY_GPU_DEVICES=0,1,2
export PHEASY_TWOLEVEL_CACHE_T=0
export PHEASY_LASSO_TWOLEVEL=1
export PHEASY_LASSO_SPARSE=1
export PHEASY_GPU_MODE=required
# [2026-10-16 根因] 光设 GPU_MODE=required 不够：pheasy 的 _resident_default() 是
# "required 且【未显式设 GPU_SM】"，而站点模板带了 PHEASY_GPU_SM=1 ⇒ 默认分支失效，
# 于是两级算子被送进 CPU 的 _LassoCVIterative（14h 无进度、GPU 闲置）。显式设下面
# 这个开关会走 _resident_lasso_requested() 分支，与 GPU_SM 无关，必然驻留 GPU。
# [2026-10-17] c3=4.00 时驻留因子需 22.15 GB > 单卡预算 20.13 GB；按 pheasy 报错
# 自带的补救办法改走两级 GPU-SM matvec（SM_prime 按卡分片，作者称本机已验证）。
export PHEASY_GPU_LASSO_RESIDENT=0
export PHEASY_GPU_FALLBACK=1
# [2026-10-20 用户指示] 精确 float64 Gram 解（规避 CGLS 精度地板）
export PHEASY_GPU_OLS_RESIDENT=0
export PHEASY_EXACT_GRAM_GB=75
export PHEASY_EXACT_GRAM_ALL=1
# OLS 不剪枝，无需 HARM_DENSE

echo "[env] host=$(hostname) env=${CONDA_DEFAULT_ENV:-${VIRTUAL_ENV:-}} gpus=${PHEASY_GPU_SM_NGPU}"
for _m in numpy scipy phonopy spglib h5py; do
    python -c "import ${_m}" 2>/dev/null && echo "  ok  ${_m}" \
        || echo "  MISSING ${_m} (required by fit-fc-thermal)"
done
python -c "import pheasy_gpu" 2>/dev/null && echo "  ok  pheasy_gpu" \
    || echo "  MISSING pheasy_gpu (PHEASY_BIN=pheasy-gpu needs the GPU build)"
nvidia-smi -L 2>/dev/null || echo "  note: nvidia-smi unavailable"

set -e
python fc_fit_driver.py prep fit_config.json
python fc_fit_driver.py fit  fit_config.json
python fc_fit_driver.py post fit_config.json
