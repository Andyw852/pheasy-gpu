# Mg2C60 六卡大内存验证复现

参数固定为二阶 7.0 Å、三阶 4.5 Å、2×2×2 超胞、float64。
运行目录使用独立副本；原始数据和服务器原安装不被覆盖。

## 环境与数据

测试服务器 Python 环境：`wc`，PyTorch 2.6.0+cu124，六张 RTX 3090。
隔离目录：`/home/wangchaoyue852/software/pheasy-gpu-validation/20260906_mg2c60`。
其 `src/pheasy_gpu` 是源码快照，`data` 是准备好的数据与缓存。
运行时把 `src` 加入 `PYTHONPATH`，确认 `import pheasy_gpu; print(pheasy_gpu.__file__)`
指向此副本。

原始数组为 700×496×3；坐标是分数坐标，最后一帧为未位移参考。
对原子做精确周期几何匹配，并同时重排坐标和力。有效构型为 699，
所有力减去参考帧残余力。原始数组不修改。

```bash
python "$PHEASY_CHECKOUT/tools/prepare_dataset.py" SPOSCAR dataset_disps.npy dataset_forces.npy \
    --frac --ref -1 --align-reference
```

在含有原始 `.npy` 文件的独立数据目录执行此命令。
本次使用的置换和位移统计见 `data/dataset_alignment.json`。

## 已有缓存直接复用

`data/sm_prime.npz` 约 57 GiB；禁止为了复现拟合重复建表或复制它。
与其配套使用 `ns_harm.npz`、`ns_anharm3.npz` 和同一结构/构型顺序。
全量矩阵为 1,040,112×66,375，约束后为 52,283 列。

在已激活正确 Python 环境的服务器上：

```bash
export PHEASY_CHECKOUT=/absolute/path/to/isolated/src/pheasy_gpu
export PYTHONPATH=/absolute/path/to/isolated/src
export PHEASY_DATA=/absolute/path/to/prebuilt/data
export PHEASY_RESULTS=/absolute/path/to/new/results
sbatch "$PHEASY_CHECKOUT/dev/large_validation.sbatch"
```

该脚本申请六 GPU、16 CPU、200 GB 内存，在同一分配内先全量 OLS，后 80/20
构型对照。默认保持 LSMR `atol=btol=1e-8`，最多 50,000 次；全量及对照 OLS
启用精确列范数 Jacobi 缩放，列范数临时分块预算 256 MiB。
RFE 子问题使用 `PHEASY_LSQR_*` 参数。当前源代码中的确切默认值以
`dev/validate_large_fit.py` 为准，每次运行均另存 `configuration.json`。

RFE 数据划分：NumPy `default_rng(20260907).permutation(699)[:100]`，前 80
构型用于训练，后 20 构型独立验证，各自排序后切取真实矩阵行；构型内 1,488
个力分量一起进出交叉验证折。三折分组 CV，删除比例 0.5，最少保留 13,071
列，按默认一标准误差规则选模型。输出 `split.json` 是实际使用的构型索引。

RFE-OLS-TSQR 在 TwoLevelSM 上实际使用 LSMR。它不执行全量 TSQR 分解。
真正的分块 TSQR 与 SVD 对照单独见本地回归测试。

## 小规模测试

以下测试不需要材料数据：

```bash
python dev/test_optimizer_large_regressions.py
python dev/test_gpu_memory_regressions.py
python dev/test_dataset_preparation.py
python dev/test_compact_fc3.py
python dev/test_ifc2_read.py
python dev/test_cli_harmonic_chain.py
```

在独立六卡分配内运行：

```bash
python dev/validate_gpu_backends.py --devices 0,1,2,3,4,5 --json gpu_validation.json
python dev/test_cli_harmonic_chain.py --gpu --workdir /absolute/new/si_gpu_run
```

独立 Si 弹簧模型用实空间 `F=-Phi@u` 生成力，不由拟合矩阵生成力。
周期边界必须开启。旧结果目录不要重复传给全链路测试。

## 判读输出

以 `result.json` 中的 PASS/FAIL、逐次求解 `*_solvers.json` 和日志一起验收。
达到迭代上限、条件数上限或发生 CPU 回退均不算 GPU 拟合通过。
`resources.jsonl` 每约 15 秒记录进程 RSS、高水位及各卡 PyTorch 分配/保留显存，
显存统计不含 CUDA 上下文和其他进程。

`*_coef.npy` 是约束后的拟合参数；`data/phi.npz` 中的 `Phi` 为轨道 IFC 向量。
`fc2.hdf5`/`fc3.hdf5` 为展开的紧凑张量。验收会重新打开文件、逐块检查有限值、
形状和声学求和规则。训练误差和 20 构型的独立验证误差分别报告。

历史失败作业 1512 使用未缩放 LSMR，在 10,000 次迭代达到上限，未计为通过；
其日志和诊断保存在 `full_validation_1512_unconverged` 及
`large_fit.1512.unconverged.out`。
