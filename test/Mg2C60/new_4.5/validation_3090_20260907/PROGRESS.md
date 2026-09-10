# Mg2C60 / 3090 验证进度

本次计划范围验证已完成，最终结论以 REPORT_zh.md 为准。

最终作业 1515 COMPLETED / 0:0，耗时 01:18:34。RFE 的 13 次 LSMR 均收敛，
选取 26141 个特征，独立测试 RMSE 0.01133981079 eV/Å，较 OLS 降低 2.622%。
全部原始记录与系数已取回 tmp/resume_3090_evidence/job_1515/；没有剩余测试作业。
下列为各阶段历史记录。

- 数据：699 构型 × 496 原子；二阶 7.0 Å、三阶 4.5 Å。
- 原始坐标与 SPOSCAR 原子块顺序不同，位移和力已按同一精确几何映射重排。
- 全量 SM_prime：1,040,112 × 66,375，3,780,068,976 nnz，float64，60.49 GB。
- 零空间后参数：52,283（二阶 10,252，三阶 42,031）。
- 建表作业 1508 成功，耗时 46:53；SLURM MaxRSS 151410516 KiB。
- 20 项本地回归通过；真实 CPU/单卡/六卡 14 项对照通过。
- CPU two-level、CPU explicit、GPU explicit 的独立 Si 弹簧模型全链路通过。
- 作业 1512：六卡内存稳定，LSMR 在 10,000 次迭代达到上限，判为失败；未产生可验收拟合结果。主机峰值 83.5 GiB，各卡分配峰值 12.1–14.9 GiB。
- 作业 1513：全量 699 构型 OLS 已收敛（9952 次，RMSE 0.00812052 eV/Å），phi/fc2/fc3 检查通过。后续 80/20 OLS 测试 RMSE 0.01164517 eV/Å。RFE 初始全特征求解达到 50000 次上限，致整体作业 FAILED。
- 作业 1514：补充 GPU 对照 16/16 通过。
- 续接修复：可选 PHEASY_RFE_JACOBI=1；14 项求解器回归 + 8 项内存/IFC 测试通过后提交 1515。其新六卡数值检查 16/16 通过；holdout OLS 基线已收敛（13457 次，istop=2），测试 RMSE 0.01164517185 eV/Å，与旧结果相对变化 2.54e-9。修复后真实 RFE 初始全特征求解已在 13433 次迭代收敛（istop=2），通过旧失败点；首轮三折 CV 也全部收敛（17428/17289/17249 次），52283 特征 CV RMSE=0.01479785 eV/Å，标准误约 6.17e-5；后续筛选及独立测试仍待完成。
- 1515 已经用户明确授权不经 tf 独立提交；6 卡、16 CPU、200 GB，新输出目录 holdout_jacobi_resume_01。核心求解器及验证驱动 SHA256 与提交版本核对一致，不修改运行中副本。
- 本地随后补充了分组 CV 与显式矩阵的选择/预测对照，15 项求解器测试通过；新增测试尚未同步到运行中作业。
- 压缩力常数已取回 tmp/resume_3090_evidence/exports/，本地逐块回读也通过；无需下载 60 GB 缓存。

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
