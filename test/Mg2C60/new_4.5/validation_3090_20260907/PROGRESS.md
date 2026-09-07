# Mg2C60 / 3090 验证进度

本文件记录进行中的验证，最终结论以 REPORT_zh.md 为准。

- 数据：699 构型 × 496 原子；二阶 7.0 Å、三阶 4.5 Å。
- 原始坐标与 SPOSCAR 原子块顺序不同，位移和力已按同一精确几何映射重排。
- 全量 SM_prime：1,040,112 × 66,375，3,780,068,976 nnz，float64，60.49 GB。
- 零空间后参数：52,283（二阶 10,252，三阶 42,031）。
- 建表作业 1508 成功，耗时 46:53；SLURM MaxRSS 151410516 KiB。
- 20 项本地回归通过；真实 CPU/单卡/六卡 14 项对照通过。
- CPU two-level、CPU explicit、GPU explicit 的独立 Si 弹簧模型全链路通过。
- 作业 1512：六卡内存稳定，LSMR 在 10,000 次迭代达到上限，判为失败；未产生可验收拟合结果。主机峰值 83.5 GiB，各卡分配峰值 12.1–14.9 GiB。
- 作业 1513：六卡、16 CPU、200 GB，启用精确列范数 Jacobi 缩放，保持 atol/btol=1e-8，上限 50,000；执行完整 699 构型 OLS，再执行固定 80/20 holdout 对照；进行中。

远端隔离目录：
`/home/wangchaoyue852/software/pheasy-gpu-validation/20260906_mg2c60`

进度/结果文件：`large_fit.out`、`full_validation/resources.jsonl`、
`full_validation/result.json`、`holdout_validation/result.json`。
严禁把未收敛、CPU 回退或仅有部分产物判为通过。

本轮修复位于工作树，未 commit/push。用户原有
`interface/shengbte.py` 和 `structure/force_constants.py` 哈希与开始时相同。

复现入口：`dev/validate_large_fit.py`、`dev/large_validation.sbatch`。
小规模对照：`dev/validate_gpu_backends.py`。
最终报告需回收 phi.npz、fc2/fc3.hdf5、审计 JSON 和资源记录；不要下载 60 GB SM 缓存。
