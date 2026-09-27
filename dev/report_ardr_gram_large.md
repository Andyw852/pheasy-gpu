# ARDR 大体系路径报告：Gram（矩阵无关）+ 可选 GPU

日期：2026-09-22  工作树：/home/wangchao/software/pheasy-gpu（未提交）
目标：回答"ARDR 是不是都在 GPU 上求解"，并为 ARDR 增加大体系能力，评估 Mg4C60。

---

## 0. 先回答你的问题

不是。在这次改动之前，ARDR 完全在 CPU 上跑：它用 scikit-learn 的 ARDRegression，
必须先把 SM 变成 dense 的 n_samples x p 矩阵，backend 标的是 cpu_dense_ardr。

现在加了两条路径：

| 路径 | 触发条件 | 设计矩阵 | 线性代数 | backend |
|---|---|---|---|---|
| dense（原语义不变） | 默认，dense/sparse 输入 | 物化 n x p | sklearn ARDRegression (CPU) | cpu_dense_ardr |
| Gram（矩阵无关） | TwoLevelSM/LinearOperator 自动开启，或 PHEASY_ARDR_GRAM=1 | 不物化 SM | CPU: scipy.linalg.pinvh；GPU: torch Cholesky | cpu_gram_ardr / gpu_gram_ardr |

也就是说：GPU 只用在 Gram 路径每轮那个 p x p 的证据求解上（Cholesky/逆），
Gram 本身的构造仍在 CPU（SM_prime 的稀疏乘法）。这样做是因为 ARD 的更新
只用 G = X^T X、b = X^T y、yty = y^T y，所以根本不需要 SM 的 n x p 形态。

## 1. 为什么能去掉 n_samples 这一维

sklearn ARDRegression 的迭代（逐行核对了 _update_sigma / update_coeff / sse）：

    sigma = inv(lambda_active * I + alpha * X_active^T X_active)
    coef_active = alpha * sigma @ (X_active^T y)
    sse = ||y - X coef||^2 = yty - 2 * c.b + c^T G c
    gamma = 1 - lambda_active * diag(sigma)
    lambda_active = (gamma + 2*lambda1) / (coef_active^2 + 2*lambda2)
    alpha = (n - sum(gamma) + 2*alpha1) / (sse + 2*alpha2)
    active = lambda < threshold_lambda

全部只用 G、b、yty。新函数 _ardr_evidence_gram 就是这段循环（CPU 用
scipy.linalg.pinvh —— sklearn 自己调用的同一个函数；GPU 用 torch Cholesky）。

因此 Gram 路径与 sklearn 数学等价（验证见下），去掉的只是 n_samples x p 的
物化；剩下 p 的硬墙：p x p 的 Gram（O(p^2) 内存）和每轮 O(p^3) 求解。这是
ARD 算法本身的性质，不是实现问题，也是 Fransson 2020 说它标度差的原因。

## 2. 验证（全部通过）

| 检查 | 规模 | 结果 |
|---|---|---|
| Gram vs sklearn（CPU pinvh） | n=800, p=200 | rel coef 2.5e-15，n_iter 完全相同，支持集相同 |
| TwoLevelSM（矩阵无关）vs 显式 dense SM | n=20000, mid=600, p=400 | rel coef 4.6e-15，backend=cpu_gram_ardr |
| Gram GPU vs sklearn | n=500, p=150 | rel coef 2.7e-15，n_iter 相同 |
| Gram GPU vs Gram CPU | n=20000, mid=1500, p=1500 | rel coef 2.7e-10（Cholesky vs pinvh），GPU 1.2s / CPU 7.3s |
| 大体系 CPU | n=120000, mid=5000, p=5000 | 显式 SM 需 4.8 GB，Gram 仅 0.20 GB；Gram 26.9s；11 轮收敛；nz=307/5000（真值 300，全部找回）；rmse 9.99e-3；195s |
| 大体系 GPU (RTX 4060) | n=60000, mid=8000, p=8000 | 显式 SM 需 3.8 GB，Gram 0.51 GB；Gram 30.3s；587 轮收敛；nz=480/8000，找回 379/400；rmse 9.93e-3；总计 87s |

复现：
- CPU 等价/一致性/大体系：python dev/test_ardr_gram.py
  （默认 --n 150000 --mid 5000 --p 5000；本报告用 --n 120000）
- GPU 大体系：用 tmp/resident_gpu_venv 的 torch，PHEASY_ARDR_GPU=1

一个重要发现：p≈8000 时 sklearn 默认的 300 次迭代上限不够——证据路径的
support 已稳定（881 个），但逐次系数变化之和仍 > tol，第一版跑到 iteration_limit。
已把 Gram 模式的默认上限提到 1000（PHEASY_ARDR_GRAM_MAX_ITER，可覆盖），
dense 路径保持 300 不变；提高后 587 轮收敛。这个默认值改动只影响 Gram 模式。

## 3. Mg4C60 可行边界（关键结论）

MgC 生产体系的实测尺寸（GPU.md / 本地报告）：

| 体系 | 构型 | 截断 | SM / SM_prime | 自由 IFC p |
|---|---|---|---|---|
| Mg8C120（512 原子） | 296 | c2=7.0 / c3=4.0 | 454,656 x 69,487 | 69,487 |
| Mg8C120 切片 | 24 | 同上 | 36,864 x 69,487 | 69,487 |
| Mg2C60（496 原子） | 699 | c2=7.0 / c3=4.5 | SM_prime 1,040,112 x 66,375 | 52,283（2 阶 10,252 + 3 阶 42,031） |

内存对比（float64；1 GB = 1e9 字节）：

| 量 | p=10,252（2 阶 only） | p=52,283（2+3 阶 Mg2C60） | p=69,487（2+3 阶 Mg8C120） |
|---|---|---|---|
| 旧 dense 路径 n x p | — | 20.5 GB（24 构型切片）到 552 GB（699 构型） | 20.5 GB（24 构型）到 252.7 GB（296 构型） |
| Gram G = p^2 * 8 | 0.84 GB | 21.9 GB | 38.6 GB |
| 两级中间 P = mid^2 * 8 | mid 约 1e4 -> 0.8 GB | mid=66,375 -> 35.2 GB | mid 约 66,375 -> 35.2 GB |
| P-free 块 Gram 峰值 | G + n*blk | 同上 | 同上（不再需要 P） |

更正：上一版把 GB 误写成 TB（差了 1000 倍），真实是 GB 量级。

结论（修正后）：

1. 内存不是不可逾越的墙。2+3 阶 p 约 7e4 时 G 约 38.6 GB、P 约 35.2 GB，
   远程那台约 200 GB 的机器放得下（旧 dense 路径要 252.7 GB，反而更险）；
   多卡也可以把 G 分片。
2. P 可以完全不要：新增 P-free 逐列块 Gram（_compute_gram_blockwise；
   PHEASY_ARDR_GRAM_NO_P=1 强制，或 P=mid^2 超过预算时自动回退），峰值只有
   G + n*blk。已验证它与 P 路径结果一致（rel 3.6e-15）。而且 mid 很大时
   P 路径本来也极慢（_gram_smprime 是 n*mid^2 计算量），块路径是唯一现实选择。
3. 真正的墙是每轮 O(p^3) 求解：p=69,487 单轮约 3.4e14 flops。sklearn 的 ARD
   初始把所有特征都设为 active，所以头几轮就是满 p 的分解。按本机实测 pinvh
   的有效速率（p=5000 约 8 GFLOP/s）外推，单轮约 11 小时；即便换优化 Cholesky
   和更多核，也是数小时/轮。active set 会很快剪到几百到几千，只有前 2-3 轮贵，
   但每轮仍是小时级。
4. 多卡只解决内存（把 G 分片），解决不了 O(p^3)：torch 稳定版没有分布式
   Cholesky；单卡内还要 2-3 份 p^2 工作副本（G、A、因子），p=69k 约 116 GB，
   一张 24 GB 放不下。
5. 所以修正结论：
   - 2 阶-only（p 约 1e4）：内存和算量都可行（本机 CPU/GPU 已实测 p=8000）。
   - 生产 2+3 阶（p 约 5-7e4）：内存可行（200 GB 机器 + P-free 块 Gram），但
     O(p^3) 决定它是"每轮小时级、整晚级"的一次性尝试，不适合做扫描。
   - 要在 2+3 阶上做可扫描的大体系稀疏剪枝，正确路线是 active-set /
     fast marginal likelihood 的 RVM（从稀疏 active set 起步、逐个增删基函数、
     O(k^2) 增量更新、按需取 G 的列），那是另一个实现。

## 4. 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| PHEASY_ARDR_GRAM | auto（算子输入=on） | 强制开/关 Gram 矩阵无关路径 |
| PHEASY_ARDR_TWOLEVEL | 1 | CLI 是否用 TwoLevelSM 喂 ARDR（关掉会退回 dense SM） |
| PHEASY_ARDR_GPU | auto（有 CUDA 就 on） | Gram 证据求解用 torch Cholesky |
| PHEASY_ARDR_GRAM_MAX_ITER | max(300,1000)=1000 | Gram 模式迭代上限 |
| PHEASY_ARDR_GRAM_MAX_GB | PHEASY_GRAM_MAX_GB，默认 4 | 内存预算：P 路径看 mid^2，块路径看 G=p^2 |
| PHEASY_ARDR_GRAM_NO_P | auto（P 超预算自动回退） | 强制用 P-free 逐列块 Gram |
| PHEASY_ARDR_VERBOSE | 1 | 每轮打印 active/sse/alpha |
| PHEASY_ARDR_THRESHOLD | 1e4 | 剪枝阈值（文献值） |
| PHEASY_ARDR_MAX_GB / MAX_FEATURES | 16 / 0 | dense 路径的内存/特征数护栏 |

## 5. 用法

    # 大体系：算子输入自动走 Gram；有 CUDA 时求解自动上 GPU
    python -m pheasy_gpu.run_pheasy --dim ... -w 3 --c2 ... --c3 ... \
        -d --ndata N --disp_file  -f --ndata N --disp_file --full_ifc -l ARDR

日志会打印：
    [SM-twolevel] ARDR 两级 matvec (不生成 SM): SM_prime(...) @ NS(...)
    [ARDR] building the design Gram (matrix-free; ...)
    [ARDR] design Gram ready: 8000x8000 (0.51 GB) in 30.3s
    [ARDR] matrix-free Gram fit on GPU: G=8000x8000 (0.51 GB)
    [ARDR]   iter k: active=... sse=... alpha=...
    ... - ARDR converged after N evidence iterations.
    ... - in-sample (Gram mode, no CV) RMSE

## 6. 未做 / 待办

- GPU 只在本机 RTX 4060（8.6 GB）上实测；远程 3090 还没跑（需要部署代码）。
- Mg4C60 的真实数据还没跑：需要先确认它的二阶-only 具体 c2 与 p
  （可在该 material 目录跑 -c 看 Free IFC number，或用 p 的实测值）。
- P-free 逐列块 Gram 已实现并验证（dev/test_ardr_gram.py 的 check_no_p）。
- Mg4C60 2+3 阶的真实瓶颈是每轮 O(p^3) 时间，不是内存；要真正可扫描需另做
  active-set / fast-RVM。

## 7. 更新：CPU ARD 求解器换成 Cholesky + dpotri

原来 _ardr_evidence_gram 的 CPU 路径用 scipy.linalg.pinvh（特征分解）求
alpha*G + diag(lambda) 的逆。实测 p=5000 单次 41.6 s，而 Cholesky 只需 0.4 s
（约 100x）。该矩阵是 PD（lambda>0），Cholesky 精确。已改为 cho_factor 分解、
cho_solve 求 sigma @ b、LAPACK dpotri 求 diag(A^-1)。dpotri 比
solve_triangular(L, I) 快 4x（p=20000：30.4 s vs 120.1 s，两者互差 1.15e-14）。
PHEASY_ARDR_PINVH=1 可回到 sklearn 的 pinvh 路径做对照。

本机 8 线程实测（p=20000 合成 TwoLevelSM，n=50000，mid=2000）：
- Gram（P 路径）6.8 s；cho_factor 16.1 s；cho_solve(b) 0.4 s；dpotri 30.4 s；
- 合计约 46.5 s / 次 full-active 迭代（旧 solve_triangular 版 137.7 s）；
- 外推 p=69487：约 32 min / 次 full-active（旧版约 96 min）；
- 与 sklearn ARDRegression 的等价性仍为 rel coef 1.9e-15。

同时：TwoLevelSM 增加了 _matmat / _rmatmat，避免 2-D 乘积退化为逐列 _matvec。
_build_gram_matrix 的 P-vs-block 代价启发式默认关闭（实测 block 反而慢：
p=7918/mid=8361 的 block Gram 489 s，P 路径 161 s），P 在内存放得下时优先。

## 8. fast-RVM（Tipping & Faul 2003）已实现

core/fast_rvm.py：从空模型开始，按最大边缘似然增益增/删/重估单个基函数，
不需要 p x p 分解；alpha 超过 lambda_t=1e4 后剪枝并在支撑上重解。
- 与 sklearn ARD 等价：小问题上 rel coef 6e-9、支持集相同；
- 已作为 Optimizer 方法 "RVM"（别名 FAST-RVM），CLI -l RVM 可用；
- Mg4C60 扫描 c2=3.0 (1524/1675)、c2=4.0 (2314/3778) 完成；
- 已进一步改成 Tipping 原始算法的增量 s,q 更新：增/删/重估都只 O(kp)/步，
  仍然与 sklearn ARD 等价（rel 6e-9）。稀疏 active set 下很快（p=5000、真值
  300 个非零 -> 5.7 s）；但 active set 本身变密时（MgC 二阶，k 达数千）步数与
  每步成本都按 k~p 增长，仍然不可行（c2=5.0 跑到 step 800 时 active=4320、
  best_gain 仍 0.12）。fast-RVM 的适用区是真正稀疏的体系（文献的 Ta 空位），
  MgC 二阶这种本来就密的模型应走 ARDR（dpotri）。





