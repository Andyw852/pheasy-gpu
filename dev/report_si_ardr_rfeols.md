# Si 测试报告：RFE-OLS / ARDR 的加入与验证

日期：2026-09-22   工作树：/home/wangchao/software/pheasy-gpu（未提交）
文献：E. Fransson, F. Eriksson, P. Erhart, *Efficient construction of linear
models in materials modeling and applications to force constant expansions*,
npj Comput. Mater. 6, 135 (2020), arXiv:1902.01271。

---

## 1. 加入前：本仓库支持哪些方法？

core/optimizer.py 的 Optimizer 在本次修改前支持：

| 方法名 | 实现 | 说明 |
|---|---|---|
| OLS | _fit_ols | 最小二乘 |
| LASSO | _LassoCVModel / FISTA / resident | L1 + CV alpha |
| ALASSO | _AdaptiveLassoCV | 自适应 LASSO (Zou 2006) |
| RFE | PheasyRFECV | RFE，**基估计器就是 OLS**（ridge_alpha 默认 0） |
| RFE-OLS-TSQR | PheasyRFE_OLS_TSQR | RFE + Q-less TSQR 的严格 OLS |
| RIDGE | _ridge_solve | 遗留 L2 |

结论：

- **RFE-OLS 其实已经支持**，只是名字叫 RFE（文档字符串原话是
  "RFE with an OLS (optionally ridge-regularized) base estimator"），
  另有 RFE-OLS-TSQR 这一严格 OLS 变体。命令行不接受 RFE-OLS 这个名字。
- **ARDR 完全不支持**：全仓库无 ARDRegression / ARDR 代码。

## 2. 本次新增

### 2.1 ARDR（automatic relevance determination regression）

在 core/optimizer.py 新增 _ARDRModel，包住 scikit-learn 的 ARDRegression：

- 每个系数一个精度超参，证据最大化把无关系数的精度推到阈值以上并置零，
  是稀疏贝叶斯 / RVM 回归。
- 文献用的就是 scikit-learn 实现，剪枝阈值 lambda_t = 1e4（也是 sklearn
  threshold_lambda 的默认值），这里默认一致；可用 PHEASY_ARDR_THRESHOLD
  或 PHEASY_ARDR_THRESHOLDS（逗号分隔，多值走分组 CV 选阈值）覆盖。
- 默认对每个阈值做分组 CV 以报告 CV RMSE；单阈值时不改变全数据拟合结果。
- 需要完整设计矩阵（构造并求逆 A^T A）：矩阵无关算子 TwoLevelSM 会被明确
  拒绝；稀疏输入在内存预算内致密化。O(n_features^3) 时间 / O(n_features^2)
  内存，与文献报告的不利标度一致。PHEASY_ARDR_MAX_GB（默认 16）和
  PHEASY_ARDR_MAX_FEATURES（默认 0 = 不限）是显式护栏。

支持 kwargs：threshold_lambda、thresholds、cv、max_iter（兼容老版
sklearn 的 n_iter）、tol、fit_intercept、alpha_1/2、lambda_1/2。

### 2.2 RFE-OLS 显式别名

Optimizer._fit_impl 现在把 RFE-OLS 归一化为 RFE（同一个 PheasyRFECV +
OLS 基估计器，语义完全等价），并接受 ARD / ARD-REGRESSION 作为 ARDR 别名。

### 2.3 CLI

- basic_io.py：-l / --model 的 choices 增加 RFE-OLS 与 ARDR，帮助文本更新。
- run_pheasy.py：RFE-OLS 走 RFE 的 sparse/two-level 判定与汇总分支；
  新增 ARDR 的拟合日志与结果汇总分支；
  未知方法报错信息与支持列表更新。

改动文件：core/optimizer.py、basic_io.py、run_pheasy.py；
新增测试：dev/test_si_ardr_rfeols.py。

## 3. Si 基准测试

脚本：dev/test_si_ardr_rfeols.py（CPU 即可，无需 GPU）。
运行：python dev/test_si_ardr_rfeols.py --jobs 4

数据集（对应文献低对称 Ta 空位那组图）：

- 3x3x3 Si 金刚石超胞（54 原子）去掉 1 个原子 = 53 原子空位胞；
  空位破坏对称性，使二阶自由 IFC 数随截断增长（41 -> 180 -> 368 -> 431），
  与文献 Fig. 5 的大参数空间情形同构。
- 真值：最近邻弹簧模型，直接在实空间生成（绝不用 F = SM @ coef），
  满足对称性与 ASR。
- 力：真值力 + 2% 高斯噪声；验证集用未见过的构型的**无噪声**力评分。
- 感应矩阵由生产命令行构建（cluster -> constraints -> sensing），
  拟合直接调 Optimizer API。截断 c2 = 3.0 / 4.5 / 5.5 / 6.5 A，
  24 训练构型 / 36 验证构型。

### 结果（rel_test = 验证 RMSE / 验证力 RMS）

| c2 (A) | 自由参数 | 方法 | 非零特征 | rel_test |
|---|---|---|---|---|
| 3.0 | 41  | OLS          | 41  | 2.39e-3 |
| 3.0 | 41  | LASSO        | 35  | 2.37e-3 |
| 3.0 | 41  | RFE-OLS      | 18  | 1.51e-3 |
| 3.0 | 41  | RFE-OLS-TSQR | 41  | 2.39e-3 |
| 3.0 | 41  | ARDR         | 18  | 1.45e-3 |
| 4.5 | 180 | OLS          | 180 | 4.81e-3 |
| 4.5 | 180 | LASSO        | 119 | 4.60e-3 |
| 4.5 | 180 | RFE-OLS      | 37  | 2.66e-3 |
| 4.5 | 180 | RFE-OLS-TSQR | 100 | 4.58e-3 |
| 4.5 | 180 | ARDR         | 31  | 2.06e-3 |
| 5.5 | 368 | OLS          | 368 | 6.70e-3 |
| 5.5 | 368 | LASSO        | 175 | 5.94e-3 |
| 5.5 | 368 | RFE-OLS      | 49  | 3.77e-3 |
| 5.5 | 368 | RFE-OLS-TSQR | 100 | 5.32e-3 |
| 5.5 | 368 | ARDR         | 29  | 1.64e-3 |
| 6.5 | 431 | OLS          | 431 | 7.46e-3 |
| 6.5 | 431 | LASSO        | 133 | 5.75e-3 |
| 6.5 | 431 | RFE-OLS      | 64  | 4.51e-3 |
| 6.5 | 431 | RFE-OLS-TSQR | 100 | 5.54e-3 |
| 6.5 | 431 | ARDR         | 29  | 1.64e-3 |

### 从 3.0 -> 6.5 A 的增长

| 方法 | 非零特征 | 特征增长 | 验证误差 | 误差增长 |
|---|---|---|---|---|
| OLS          | 41 -> 431   | 10.5x | 2.39e-3 -> 7.46e-3 | 3.1x |
| LASSO        | 35 -> 133   | 3.8x  | 2.37e-3 -> 5.75e-3 | 2.4x |
| RFE-OLS      | 18 -> 64    | 3.6x  | 1.51e-3 -> 4.51e-3 | 3.0x |
| RFE-OLS-TSQR | 41 -> 100   | 2.4x（min_features=100 地板） | 2.39e-3 -> 5.54e-3 | 2.3x |
| ARDR         | 18 -> 29    | 1.6x  | 1.45e-3 -> 1.64e-3 | 1.1x |

断言检查全部通过（report.json）：ARDR / RFE-OLS 在最大截断下特征数远少于
OLS；验证误差不劣于 OLS 的 2 倍；特征数增长不超过 OLS 的一半。
数值行为与文献 Fig. 5a/5b 一致：ARDR 最平，RFE-OLS 次之且明显优于
OLS/LASSO，裸 OLS 特征数爆炸、验证误差最大。

## 4. 回答：RFE-OLS 是不是最好

不是单一维度的"最好"，文献本身也没有把 RFE-OLS 单独捧为最优：

1. 验证误差：大截断下 ARDR 最平（本测试 1.45e-3 -> 1.64e-3），
   RFE-OLS 轻微上翘但仍明显优于 LASSO / OLS。排序 ARDR >= RFE-OLS >
   LASSO > OLS。
2. 收敛/数据量：RFE-OLS 与 ARDR 约 5 个构型进入参考值 10% 以内；
   ARDR 约 12 个构型到 2%，作者评价"甚至比截断选择还好"。LASSO 与 OLS
   在该数据量下未收敛。
3. 成本：特征选择方法可比 OLS 贵两个数量级以上；文献在 Ta 空位最终选择
   "OLS + 审慎选定的截断"。RFE 通常要 100-1000 次 OLS；ARDR 随体系规模
   标度更差，作者明确说它无法处理最大的感应矩阵。
4. 小参数空间：方法几乎无关，收敛由截断决定，不是算法决定。

因此更准确的表述是：**RFE-OLS 与 ARDR 是第一梯队、最稳的控复杂度手段；
其中 ARDR 在大截断下略优，RFE-OLS 明显优于 LASSO；裸 OLS 最便宜，但必须
人工把参数空间关小**。这也是本次 Si 测试复现出来的排序。

对应到项目：把 v5（alpha 卡在网格下界的 LASSO 变体）换成 RFE-OLS 或
ARDR、同时把 c3 保持在上限，是最有信息量的一次对照。现在 -l RFE-OLS 与
-l ARDR 都已可用。

## 5. 命令行用法

    # ARDR（文献默认阈值 1e4）
    python -m pheasy_gpu.run_pheasy ... -f --ndata N --disp_file --full_ifc -l ARDR

    # RFE-OLS（= 旧 RFE，OLS 基估计器）
    python -m pheasy_gpu.run_pheasy ... -f --ndata N --disp_file --full_ifc -l RFE-OLS

可用环境变量：PHEASY_ARDR_THRESHOLD（默认 1e4）、
PHEASY_ARDR_THRESHOLDS（多值，分组 CV 选阈值）、PHEASY_ARDR_MAX_ITER、
PHEASY_ARDR_MAX_GB、PHEASY_ARDR_MAX_FEATURES。

## 6. 验证与待办

已验证：

- python -m py_compile core/optimizer.py basic_io.py run_pheasy.py 通过。
- 合成稀疏回归：OLS / LASSO / ALASSO / RFE / RFE-OLS / RFE-OLS-TSQR /
  RIDGE / ARDR 均可跑；ARDR 在真值 15 个非零系数上恰好选中 15 个。
- CLI 端到端：-l ARDR（空位胞 368 参数 -> 29 非零，4 次迭代，1.1 s）、
  -l RFE-OLS（-> 84 非零，26 s）均成功写出 FORCE_CONSTANTS_2ND。
- 既有回归 dev/test_cli_harmonic_chain.py 通过（fc2 rel 3.3e-16）。

待办（需用户确认后再做）：

- 尚未提交 git；未部署到 GPU 机（$REMOTE/pheasy）。按 AGENTS.md，提交和
  远程操作是受控操作，需明确同意后执行。
- 真实 DFT / Tersoff Si 数据（4x4x4, c3=7.0）尚未用新方法跑；本报告用的是
  可完全复现的合成 Si 空位基准。
