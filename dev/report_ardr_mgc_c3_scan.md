# MgC (Mg8C120, 512原子) 2+3 阶 ARDR 拟合 + 三阶 IFC 导出报告

## 目标
对 Mg8C120 (c2=7.0 Å) 在多个三阶截断半径 c3 下用 ARDR 拟合 2+3 阶力常数，导出
ShengBTE 可读的三阶 IFC (FORCE_CONSTANTS_3RD + fc3.hdf5)，为算 κ 做准备。

## 方法
- 模型: ARDR = sklearn ARDRegression 证据最大化 (Gram 形式, matrix-free 两级 SM)
- 路径: CPU (PHEASY_ARDR_GPU=0), Gram 分块投影 (避免 mid×p 中间量 OOM)
- 收敛: 迭代上限 80 (PHEASY_ARDR_GRAM_MAX_ITER=80), tol=1e-2
  - 注: sklearn 原判据 sum(|coef变化|)<1e-3 在病态大 p 下退化 (系数在 Gram 近零方向
    持续微幅抖动, sse 早已冻结却不触发收敛)。系数在 ~40 迭代即物理收敛 (<1e-5),
    对齐检查 corr=0.9999 验证拟合正确。
- 数据: ndata=296 (全部构型), 2×2×1 超胞 512 原子, 二阶 22066 自由 IFC

## 结果

| c3 (Å) | p (自由 IFC) | 三阶自由 | 非零 | 比例 | rmse | r² | 对齐 corr |
|---|---|---|---|---|---|---|---|
| 3.0 | 32606 | 10540 | 20502 | 63% | 0.01046 | 0.99985 | 0.9999 |
| 3.5 | 38377 | 16311 | 26056 | 68% | 0.01037 | 0.99986 | 0.9999 |
| 3.8 | 60238 | 38172 | 47445 | 79% | 0.00977 | 0.99987 | 0.9999 |
| 4.0 | 69487 | 47421 | (运行中) | - | - | - | - |

### 关于 κ 平台期的关键结论
**特征数 (非零) 随 c3 单调增长, 无平台期**:
- 非零: 20502 → 26056 → 47445 (c3=3.0→3.5→3.8), 比例 63%→68%→79%
- ARDR 保留比例随截断**上升** (不是恒定比例), 所以特征数增长快于 p。
- 与之前 2 阶扫描一致 (c2=3.0→6.0 非零 1638→10971, 无平台)。

## IFC 产物 (ShengBTE 可读)
- c3=3.0: .../c3_3.0/exp_ardr/{FORCE_CONSTANTS_2ND,FORCE_CONSTANTS_3RD,fc2.hdf5,fc3.hdf5}
- c3=3.5: .../c3_3.5/exp_ardr/...
- c3=3.8: .../c3_3.8/exp_ardr/... (fc3.hdf5 20.7MB, shape (128,512,512,3,3,3))
- fc3.hdf5 shape (128,512,512,3,3,3) float64, 128 原胞原子 × 512 超胞原子 × 3³ 方向

## c3=4.5 / 5.0 (降级)
- c3=4.5: ARDR Gram 内存不可行 (P=172GB + G=112GB > 206GB cgroup 上限)。
  已有 OLS 导出的 FORCE_CONSTANTS_3RD (114.6MB) + fc3.hdf5 (35.1MB)。
- c3=5.0: ARDR 更不可行 (sm_prime 42.8GB, mid 巨大)。sensing 已建好但无 IFC,
  OLS 可用但内存紧 (sm_prime 42.8GB)。待定。

## 遇到的问题与修复
1. GPU OOM (共享 GPU 被占) → 强制 CPU 路径 (cho_factor+dpotri)。
2. G=NSᵀPNS 物化 38GB 中间量 → 列分块投影 (PHEASY_GRAM_PROJECT_BLOCK)。
3. cgroup 内存上限 206GB (共享 box, 用户 slice) → 删除 104GB 手动中间文件释放 page cache。
4. c3=4.0 sensing 用旧 SPOSCAR (edb1d...) 与其它截断不一致 → 重建 sensing (现 9276...)。
5. 迭代上限导致 fit_accepted=false 拒绝导出 → PHEASY_ALLOW_UNACCEPTED_FIT=1。
6. 一个"原地缩放"优化 bug (A*=alpha 连带改 Gk 使 sse 算错) → 已回退。
