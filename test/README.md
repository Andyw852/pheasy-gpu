# pheasy 测试材料

本目录存放力常数拟合的测试材料，按材料分门别类。

## 目录结构

    test/
    ├── MnIn2Se4/            # 测试材料：MnIn2Se4（原胞 + 3×3×3 超胞）
    │   ├── POSCAR           # 原胞结构
    │   ├── SPOSCAR          # 超胞结构 (3×3×3, 189 原子)
    │   ├── pcell.symops     # 原胞对称操作
    │   ├── scell.symops     # 超胞对称操作
    │   ├── disp_matrix.pkl  # 位移矩阵（45 个构型）
    │   ├── force_matrix.pkl # 力矩阵
    │   └── dataset_*.npy    # 原始位移/力数据（用 tools/prepare_dataset.py 可从 npy 重新生成 pkl）
    ├── Mg4C60/              # 归档材料：Mg4C60（C120Mg8）3.0 Å / 3.5 Å 力常数拟合 + Wigner κ 对比
    │   ├── README.md        # 英文报告：两套拟合口径、五次拟合、κ 对比与结论
    │   ├── 00_shared_inputs/# 五次拟合共用的结构 + 位移/力数据
    │   ├── 01..03_*joint*   # FC2+FC3 联合重拟合（3.0 Å / 3.5 Å ×2）
    │   └── 04..05_*frozen*  # 冻结 FC2、只解 FC3（3.0 Å / 3.5 Å）
    └── results/             # c3=7.0 扫描的收敛结果（convergence.csv/png）

## 怎么跑

脚本 pheasy_fit.sh 在项目根目录，直接 cd 到材料目录运行：

    cd test/MnIn2Se4
    bash ../../pheasy_fit.sh FIT_METHOD=OLS    C3_CUTOFF=5.2
    bash ../../pheasy_fit.sh FIT_METHOD=LASSO  C3_CUTOFF=5.2 NDATA=45 NCPU=8
    bash ../../pheasy_fit.sh FIT_METHOD=ALASSO C3_CUTOFF=5.2
    bash ../../pheasy_fit.sh FIT_METHOD=RFE    C3_CUTOFF=5.2
    bash ../../pheasy_fit.sh FIT_METHOD=RIDGE  C3_CUTOFF=5.2

## 材料说明

- Mg4C60：C120Mg8，2×2×1 超胞（512 原子），297 帧位移/力数据（296 帧参与拟合），
  五个拟合共用同一份数据：
  - `01`/`02`/`03`：FC2+FC3 **联合重拟合**（RFE-OLS，HARM_DENSE=1，74017 谐波列全保留），
    3.0 Å 选中 7683/10540 非谐项、3.5 Å 选中 8669/16311 非谐项（总支撑 82686/90328）；
  - `04`/`05`：**冻结 FC2**（读入 `01` 的 fc2，md5 313d0f61…）、只解 FC3，支撑 7360/10540、9765/16311。
  - κ（phono3py 3.24.0 Wigner，8×8×8，300 K，含同位素）：同一份数据因拟合口径不同给出
    **1.73–2.06 W/(m·K)**；κ_C 在全部五个拟合中都落在 0.542–0.548（≤1.1%），差异全在 κ_P。
    详见 `test/Mg4C60/README.md`。
- MnIn2Se4：原胞，超胞 3×3×3（189 原子），45 个随机位移构型。
  - 二阶截断 C3_CUTOFF=5.2 Å（原来的默认，对应 MnIn2Se4_lasso）。
  - 三阶截断 C3_CUTOFF=7.0 Å 时自由 IFC 数从 1303 涨到 3678（对应 MnIn2Se4_c7 扫描）。
  - 同一份位移/力数据，只改 cutoff 即可复现两种扫描。
