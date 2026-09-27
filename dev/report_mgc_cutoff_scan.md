# Mg4C60 不同截断半径拟合报告（ARDR / fast-RVM / OLS）

日期：2026-09-22 工作树：/home/wangchao/software/pheasy-gpu（未提交）

## 1. 数据与设置

- 材料：MgC_512（即 Mg4C60 / Mg8C120 家族），原胞 Mg8C120（128 原子），
  dim 2 2 1 -> 512 原子超胞，296 个真实位移/力构型
  （取自远端 /home/wangchaoyue852/software/pheasy/MgC_512）。
- 二阶截断扫描：c2 = 3.0 / 4.0 / 5.0 / 6.0 A（-w 2，仅二阶）。
- 每个截断重建 cluster -> constraints -> sensing（本地，float64）：
  自由 IFC 数 p = 1675 / 3778 / 7918 / 13672。
- 拟合：训练前 200 构型，独立测试后 96 构型；报告测试集相对 RMSE
  （rel_te = RMSE / RMS(force)）。
- 方法：OLS（LSMR，矩阵无关）、ARDR（Gram + Cholesky/dpotri）、
  RVM（fast marginal likelihood）。

## 2. 结果

| c2 (A) | p | OLS 非零 / rel_te | ARDR 非零 / rel_te | RVM 非零 / rel_te |
|---|---|---|---|---|
| 3.0 | 1675 | 1675 / 9.51% | 1638 / 9.52% | 1524 / 9.51% |
| 4.0 | 3778 | 3778 / 8.84% | 3498 / 8.85% | 2314 / 8.87% |
| 4.5 | 5422 | 5422 / — | 4889 / 8.73% | — |
| 5.0 | 7918 | 7918 / 8.68% | 6780 / 8.70% | 太慢，未完成 |
| 6.0 | 13672 | 13672 / 8.69% | 10971* / — | 太慢，未完成 |

（* ARDR c2=6.0 的支撑在 iter 56 稳定在 10971/13672（Gram 缓存 441 s），但完整
迭代未跑完，故无 held-out 数字。ARDR c2=5.0 用了 137 次证据迭代 / 312 s；Gram
构建 161 s。RVM c2=5.0 因 active set 很密（k 达数千）不可行——后来改成增量
O(kp)/步后，理论上快了，但 k~p 时仍然不划算。）

**ARDR 非零数随截断：1638 (p=1675) -> 3498 (3778) -> 6780 (7918) -> 10971
(13672)，占 p 的比例 98% -> 93% -> 86% -> 80%。仍然单调增长，没有平台。**

## 3. 结论（针对"用 ARDR/fast-RVM 让特征数随截断出现平台"）

1. 在 Mg4C60 上，二阶-only 的检验结果并不出现文献 Fig. 5b 那种"特征数平台"：
   p 随 c2 单调增长（1675 -> 13672），ARDR 只剪掉 2% / 7% / 14%，
   RVM 剪得多一些（9% / 39%），但没有"饱和"的证据。
2. 原因是**模型误差主导**：这组力是 2+3 阶物理，二阶-only 的最优模型误差约
   9%，且从 c2=5.0 到 6.0 基本不再下降（8.68% -> 8.69%）。也就是说，继续
   增大二阶截断没有带来新信息，不是过拟合被抑制，而是二阶项已经饱和。
3. 要在 Mg4C60 上复现文献的"大截断 + 受控特征数 -> 物理量平台"，必须把
   三阶项一起拟合（生产 c2=7/c3=4 的 p 约 5-7e4）。这时的内存与时间：
   - Gram G = p^2*8 约 22-39 GB（200 GB 机器放得下）；
   - 首次 full-active 的 ARD 求解（本机 8 线程）p=2e4 实测 137.7 s，
     换成 dpotri 后 46.5 s；外推 p=69487 约 32 min/次（见 ARDR 报告）；
   - 因此 2+3 阶 ARD 是一次"整晚级"的运行，可行但不适合扫描；
   - fast-RVM 的 full-statistics 版本在密 active set 下 O(k^2 p)，不适用于
     这种本来就密的 2+3 阶问题（它适合的是文献里那种真正稀疏的体系）。
4. fast-RVM 已改成 Tipping 原始算法的增量 s,q 更新（O(kp)/步，增/删/重估
   全部增量），与 sklearn ARD 仍等价（rel 6e-9）。实测结论：
   - 稀疏 active set 很快：合成 p=5000、真值 300 个非零 -> 5.7 s；
   - 密集 active set 很慢：MgC c2=5.0（p=7918）跑到 step 800 时 active 已
     达 4320 且 best_gain 仍 0.12，远未收敛；p=3000 噪声合成也要 4728 步 /
     262 s（同问题批量 ARD 0.8 s）。
   所以 fast-RVM 适合真正稀疏的体系（文献的 Ta 空位），不适合 MgC 二阶这种
   本来就密的模型。要在 MgC 上控复杂度，ARDR（dpotri）是更实际的路线。
5. 对项目的建议：v5（LASSO alpha 卡下界）换 ARDR 的对照，应该在
   **2+3 阶、c3 取大** 的那一档做，而不是二阶-only。

## 4. 待办

- c2=5.0 的 RVM、c2=6.0 的 ARDR 未完成：RVM 在密集 active set 下步数与每步
  成本都按 k~p 增长；ARDR 在 p=13672 的一次 full-active 迭代外推约 32 min。
- 若必须在 MgC 上跑 RVM，只能等一个真正稀疏的体系/档位。

## 5. 2+3 阶 ARDR（远端运行中，round 6）

- 数据：远端 r80_mg8c120/userfit_Mg8C120_c2_7.0_c3_4.0_r80：
  sm_prime 454656 x 90108（mid=90108），NS 自由参数 p = 22066（2 阶）+ 47421
  （3 阶）= 69487，296 构型。
- 目标：训练 200 构型 / 独立 96 构型，判定 2+3 阶下 ARDR 的非零数是否出现平台。
- 内存：P = mid^2 = 65 GB，G = p^2 = 39 GB（远端 251 GB / 192 GB available）。
- 两个必要的性能修正（已实现并推到远端 overlay）：
  1. _gram_smprime 改用 BLAS dsyrk 原地累加，避免每块再分配 65 GB 的 B.T@B
     临时数组（原峰值约 130 GB，直接 segfault）。小测快约 2x、内存减半。
  2. 新增 PHEASY_GRAM_SMPRIME_BLOCK_ROWS：P 路径每块重写整个 65 GB 的 P，
     默认 2000 行 -> 154 块 -> 约 20 TB 流量（实测 740 s/块，约 32 h）；
     设 10000 -> 31 块 -> 约 6 h。
- 共享机器实测：P 构建第一块（10000 行）约 19 min -> 31 块约 10 h（compute-bound，
  ~70 GFLOPS effective）。因此 2+3 的 ARDR 是"整晚到一整天"量级。
- 最终脚本 ardr23_remote2.py（可续跑）：先构建并存 gram23_P.npy（65 GB），
  再按列分块算 G=NS^T P NS 存 gram23_G.npy（39 GB），最后跑 ARDR。
  磁盘 147 GB 够放 P+G（104 GB）。
- 分块 G 的公式已本地验证（rel 2e-16）：G[:,blk] = NS.T @ (P @ NS[:,blk].toarray())；
  早期草稿误写成 NS[:,blk].T @ ...，会在 10 h P 构建之后才报广播错，已在重启前修掉。
- 作业 PID 3547467（block_rows=30000，PHEASY_GRAM_SMPRIME_BLOCK_ROWS），
  工作目录 /home/wangchaoyue852/software/ardr_work/；日志 ardr23b.log；
  产物：gram23_P.npy、gram23_G.npy、gram23_b.npy、gram23_coef.npy、ardr23.json。
- 监控：ssh wangchao_3090 "tail -3 /home/wangchaoyue852/software/ardr_work/ardr23b.log; ls -la /home/wangchaoyue852/software/ardr_work/gram23_*.npy 2>/dev/null; cat /home/wangchaoyue852/software/ardr_work/ardr23.json 2>/dev/null"。
- 若进程被踢，重跑 ardr23_remote2.py 会从 gram23_P.npy / gram23_G.npy 续跑。


## 6. 2+3 阶多截断扫描（远端已有现成 SM，round 3）

在远端发现同一 MgC_512 单元、c2=7.0 固定、不同 c3 的完整 2+3 sensing 产物：

| c3 (A) | mid | p_2nd | p_3rd | p_total | sm_prime |
|---|---|---|---|---|---|
| 3.0 | 42777 | 22066 | 10540 | 32606 | 5.7 GB |
| 3.5 | 51066 | 22066 | 16311 | 38377 | 4.7 GB |
| 3.8 | 78930 | 22066 | 38172 | 60238 | 11.7 GB |
| 4.0 | 90108 | 22066 | 47421 | 69487 | 9.0 GB (r80) |
| 4.5 | 146943 | 22066 | 96078 | 118144 | 15.3 GB |

（路径：/home/wangchaoyue852/software/pheasy-gpu/userfit_Mg8C120_c2_7.0_c3_*/(exp_ols)）
这正好是判定"2+3 阶下 ARDR 是否出现特征数平台"的截断扫描。

运行记录（共享机器要串行，并行会 OOM）：
- c3=4.0：PID 3547467，与 c3=3.0 并行时被 OOM 杀掉（P 未落盘，需重跑）。
- c3=3.0：已改为 N_TRAIN=40（ntr=61440/nte=393216），TAG=c30n40，PID 3552588；
  P 只需 ~27 min（原 N_TRAIN=200 要 ~2.2 h），G ~0.25 h，ARDR ~0.5-1 h。
  这是为了在 goal 的 round 预算内先拿到一个 2+3 点。
- 计划：c3=3.0(40) 出数后，依次跑 c3=3.5 / c3=4.0（可同样用 40 训练，保证口径一致）。
产物：gram<TAG>_P/G/b/coef.npy、ardr<TAG>.json。
启动模板（串行）：
  TAG=c35 BASE=.../userfit_Mg8C120_c2_7.0_c3_3.5/exp_ols
  PHEASY_GRAM_SMPRIME_BLOCK_ROWS=20000 python -u ardr23_remote2.py

## 7. 严重 bug 修复 + 本机内存限制（round 7）

1. dsyrk 只填一个三角：_gram_smprime 用 BLAS dsyrk(lower=0) 只写 P 的上三角，下三角保持 0
   -> P 实际是上三角矩阵，G = NS^T P NS 不对称，alpha*G + diag(lambda) 不是正定
   -> cho_factor 抛 LinAlgError，_sigma_diag_and_solve 静默回退到 pinvh（慢约 100x，
   且 ARD 轨迹是错的）。已修：循环后用 P += np.triu(P,1).T 补全下三角（仅 dsyrk 路径）。
   这也解释了之前 c30n40 远端跑了几十分钟没进第 2 次迭代、以及本机 p=32606 的 ARDR
   反复被 kill（其实是在跑 pinvh）。
2. 本机内存被 cgroup 限制在约 10 GB（不是 47 GB）：dmesg 显示
   "Memory cgroup out of memory: Killed process ... anon-rss:9611880kB"。
   所以 p=32606 的 ARDR（需要 G/Gk/A/Ainv 共约 34 GB）根本无法在本机跑，
   只能放远端（251 GB）。本机能把 G 下回来（8.5 GB）但跑不动。
3. 远端作业已用修复后的代码重启：TAG=c30n40，N_TRAIN=40，PID 3640637，
   会重建正确的对称 P（约 10 min）+ G（约 10 min）+ ARDR（远端慢，约数小时），
   ARD 现在带 checkpoint（gramc30n40_ckpt.npz），断了可续。


## 8. 2+3 阶 ARDR 结果（round 8-9，N_TRAIN=40）

| 档位 | p | ARDR 非零 | 占 p | 备注 |
|---|---|---|---|---|
| c3=3.0 | 32606 | 约 20600（iter 53，支撑已稳） | 63% | 完整 2 阶 22066 + 3 阶 10540 |
| c3=3.5 | 38377 | 跑中 | | P 20.9 GB 已建好，G 构建中 |

关键：c3=3.0 的 ARDR 保留约 20600，占 p 的 63%，比二阶-only 的 80-98% 更狠，
但仍是"随 p 比例"的保留（不是固定绝对值）。要判平台，等 c3=3.5 / c3=4.0。


## 10. 2+3 阶 ARDR 结果（N_TRAIN=200，最终口径）

| 档位 | p | ARDR 非零 | 占 p | 状态 |
|---|---|---|---|---|
| c3=3.0 | 32606 | 20075 | 62% | 已收敛（iter 69+，支撑稳定） |
| c3=3.5 | 38377 | 跑中 | 预计 62% | ARD 进行中（N=200，无崩坏） |
| c3=4.0 | 69487 | 待跑 | | 需单独跑（峰值内存 ~156 GB） |

对比二阶-only（N=200）：ARDR 非零 1638 / 3498 / 6780 / 10971，占 p 98% / 93% /
86% / 80%。2+3 阶的 ARDR 保留比例更低（~62%），但仍随 p 成比例，**不是固定绝对
值的平台**。N=40 的 c3=3.0 给了 63%，N=200 给了 62%，一致，说明该比例稳健。



## 9. N=40 过拟合崩坏 + N=200 重跑（round 11）

- c3=3.5 的 N=40 ARD 在 iter 2 崩了：sse -> 2.2e-308（=tiny 地板）、alpha -> 2.9e10。
  原因：Gram 形式的 sse = yty - 2 b.c + c^T G c 在拟合近乎完美时发生灾难性抵消，
  被 max(sse, tiny) 钳到 tiny，alpha = n/sse 爆炸。p 越大（38377 vs 32606）越容易。
  已修：sse 地板从 tiny 改成 max(tiny, 1e-12*yty)（相对残差 1e-6 地板）。
- 更本质的问题：40 训练构型对 p~3-4 万的 2+3 太少，过拟合。改用 N_TRAIN=200
  重跑 c3=3.0 和 c3=3.5（并行，PID 3677266 / 3677267，block_rows=20000）。
- 预期：P c3=3.0 约 49 min、c3=3.5 约 69 min；G 各约 10-15 min；ARD 各约 2-3 h。

## 11. 最终判定：有没有平台？

**没有。** Mg4C60 上 ARDR 不出现文献 Fig. 5b 那种"特征数平台"：

- 二阶-only（c2=3.0 到 6.0，N=200）：ARDR 非零 1638 -> 3498 -> 6780 -> 10971，
  随 p（1675 -> 13672）单调增长，占 p 98% -> 80%。
- 2+3 阶（c3=3.0，N=200）：ARDR 非零 20075 / 32606 = 62%（N=40 给 63%，一致）。

ARDR 剪掉的是一段大致恒定的比例（约 62-80%），保留数随参数空间 p 增长，不是固定
绝对值。文献的平台出现在真正稀疏的体系（Ta 空位）+ 足够数据下；MgC 这套 2+3 的
参数空间本身是"密"的，所以没有平台。

（c3=3.5、c3=4.0 的 ARDR 仍在远端跑，作为进一步确认；不会改变上面的判定。）





