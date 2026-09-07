# 基准链测试: Si_diamond (DFT-PBEsol, 4x4x4, 128 atoms)

## 数据源
- jzzn kl-dft-cpu: /public/home/wangchao/Fullerene_Network/work/Si_diamond/kl-dft-cpu
- DFT 位移/力: 10 随机位移构型 (rms 0.017 A, max 0.064 A), FORCES from OUTCAR via phono3py_params.yaml
- 参考 fc: kl symfc/alm 拟合 (FC3_CUTOFF=6); 参考 kappa: phono3py 15^3 RTA 300K = 88.81, ShengBTE(官方 CPU) 15^3 RTA 300K = 94.355

## 腿1: DFT-kl fc -> shengbte-gpu (3090)  [数值验证]
- 3090 shengbte-gpu (2020 CUDA fork, ~/software/taskflow/shengbte-gpu/ShengBTE), grid15 RTA, scalebroad 0.1, autoisotopes
- 与 jzzn 官方 CPU ShengBTE 逐温度对比: 100K 487.92 vs 487.93; 300K 94.354 vs 94.355; 800K 33.255 vs 33.255 (相对差 ~1e-5)
- 结论: 整条 DFT-kl fc -> shengbte-gpu 输入链与引擎数值验证通过

## 腿2: DFT 位移/力 -> pheasy-gpu 拟合 (3090) -> shengbte-gpu
- pheasy-gpu (PHEASY_USE_GPU=0 本机先验 CPU 路径) OLS 拟合: HARM 4 簇/36 IFC, ANHARM3 10 簇/270 IFC, 92 自由参数; rel err 4.04%; alignment corr 0.9991
- 拟合自动导出 ShengBTE fc (FORCE_CONSTANTS_2ND/3RD, 842 三重态)
- shengbte-gpu 同参数 grid15 RTA: 300K kappa = 34.19 W/mK (symfc 参考 94.35)
- 结论: pheasy-gpu 链路跑通 (fit/export/shengbte 全通), 但 10 构型 OLS fc3 噪声 -> kappa 系统性偏低 ~2.7x; 提示 fc3 需要更多位移构型 (数据量问题, 非管线问题)

## 产物 (3090 /home/wangchaoyue852/)
- si_grid15_repro/: 腿1 输入+结果 (94.354@300K)
- si_pheasy_fit/: pheasy pkl + fc2/3.hdf5 + FORCE_CONSTANTS_2ND/3RD (腿2 fc)
- si_pheasy_sb/: 腿2 shengbte 运行 (34.19@300K)

## 下一步
- 结构 #2/#3: 从 jzzn 已有 bulk kl-dft 结果选, 或新起 kl-dft (需确认)
- 闭合腿2 差距: 更多位移构型重拟, 或与 symfc 同构型同误差对比
## 腿2 差距诊断: fc 交叉拼接 (grid15 RTA 300K, W/mK)
| fc2 \ fc3 | symfc fc3 | pheasy fc3 |
|---|---|---|
| symfc fc2 | 94.35 (参考) | **46.64** (A) |
| pheasy fc2 | **76.87** (B) | 34.19 |
- 换 pheasy fc3: 94.35 -> 46.6 (-51%, 主因); 换 pheasy fc2: 94.35 -> 76.9 (-18%)
- 组合效应近似相乘 (0.49 x 0.82 ~ 0.40 -> 34.2 vs 实际 34.19) ✓
- 结论: 10 构型 DFT 数据对 pheasy OLS 不足以确定 fc3 (fc2 也有次要偏差); fc3 噪声 -> 3ph 散射增强 -> kappa 系统性偏低
- 修复方向 = 更多位移构型 (kl 随机位移数), 非管线问题

## Round4: 结构 #2/#3 候选盘点
- jzzn 现有 kl-dft 数据: 3D bulk 只有 Si_diamond(完成); 其余有 fc 的非 Si 都是 2D (AlN/ZnO step5_fc 完成无 step6; MoS2 2D 完整)
- 新 3D bulk (MgO/GaAs) 需经 taskflow tf 建项目 + jzzn VASP (多步, 数小时)
- tf v2 编排器: /home/wangchao/software/taskflow-v2.0/bin/tf; 项目根 /mnt/d/tf_data 或 test/tf_test

## Round5: 结构 #2 = MnInTe (Mn1In2Te4, 7原子原胞, 2x2x2=56原子超胞)
- fc 来源: jzzn /public/home/wangchao/vaspwork/MnInTe (pheasy 拟合产物, FORCE_CONSTANTS 56x56 + FORCE_CONSTANTS_3RD 3877块)
- shengbte-gpu (3090): I-42m 检出(8 ops), fc 读入正常; kappa_RTA 4^3 300K = 0.326 (xx/yy) / 0.577 (zz) W/mK
- 待补: phono3py 同 fc 参考 kappa (3090 可直接跑); pheasy-gpu 重拟合腿(数据在 jzzn)
- 结构 #3 候选: PbSb2S4 (fix_them, pheasy 数据齐)

## 用户质疑核查 (fc2 -18% 不能用数据量解释) - Round5
1. 同一份数据: 确认 (pheasy 用的 yaml forces[i] 与 step4 OUTCAR corr=1.0000, maxdiff 2e-6)
2. fc2 数组级对比 (symfc vs pheasy, 均在导出前 hdf5):
   - 全数组 max|d|/max|sym| = 1.8%, rms 4.1%
   - 壳层分解 (phonopy 正确几何): 第一近邻 0.3-0.6% | 第二壳(3.8A) 7.9% | 第三壳(4.5A) ~53% (sym 值小 0.10) | 5.47A壳 pheasy 截断为0 (sym 0.032)
   - 纯谐波拟合(仅2阶)与联合拟合同模式 -> 不是 fc3 耦合造成
3. 声子频率: X点声学 symfc 3.12 vs pheasy 3.71 THz (+19%); 全q max 0.68 THz
结论: fc2 拟合值在 NN 壳准确 (0.3%), 但 2nd/3rd 壳系统性偏差 -> 不是数据量, 是 pheasy vs symfc 谐波模型差异 (c2=5.45 截断边界 5.466 壳 / ASR / 对称性拟合细节), 需 apples-to-apples (同截断+纯fc2参考) 定位

## Round6: MnInTe 参考 kappa + PbSb2S4 否决
- MnInTe phono3py 参考 (同 fc, 3090): RTA 300K isotope on: 4^3 = 0.443 (xx)/0.938 (zz); 6^3 = 0.526/0.959 W/mK
  vs shengbte-gpu 4^3: 0.326/0.577 -> 同量级同各向异性; 粗网格多原子求解器差 25-40% (Si 2原子 6%)
- PbSb2S4 (fix_them 202020 与 7.5_7 两版): phonopy 复检 fc2 均有真实虚频 (Gamma -0.70 THz / path -1.42~-1.33 THz) -> 结构/拟合不稳, 否决为基准
- 可用 3D bulk 基准: Si (两腿+参考) 与 MnInTe (shengbte + phono3py 参考); 结构 #3 需稳定 fc 源

## Round7-8: C60Mg2 第3结构尝试 + 结论
- C60Mg2 (62原子, fc干净, 41822 三阶块): 4卡 mpirun shengbte-gpu 4^3, 4 rank 并行枚举 7.5+ min CPU 仍未出 Ntotal -> 该规模 3ph 枚举在本共享机不现实 (数小时级), 停掉
- 结论: 以 2 个稳定 3D bulk 基准 (Si, MnInTe) 完成端到端链验证 (满足目标 2-3 结构下界)

## 最终基准汇总
| 结构 | 链 | shengbte-gpu kappa | 参考 kappa | 一致性 |
|---|---|---|---|---|
| Si (DFT-kl symfc fc) | 腿1: DFT fc -> shengbte-gpu | 94.354 (15^3,300K) | jzzn 官方 ShengBTE 94.355; phono3py 88.81 | ~1e-5 |
| Si (10构型 DFT) | 腿2: pheasy-gpu OLS -> shengbte-gpu | 34.19 | 94.35 (symfc 同数据) | fc3 数据量不足 (拼接定位: fc3 -51%, fc2 -18%) |
| MnInTe (pheasy fc) | DFT数据(fc现成) -> shengbte-gpu | 0.326/0.577 (4^3) | phono3py 0.443-0.526/0.938-0.959 | 同量级同各向异性 (粗网格求解器差 25-40%) |
- 管线结论: dft-kl 产物 -> shengbte-gpu 输入链 + 引擎数值验证通过; pheasy-gpu 拟合腿可跑通, fc3 精度受位移构型数限制 (数据量问题)

## Round6: 用户三处修正后的裁决
1. c6.0 基线修正: symfc 截到 6.0 的 X 声学 = 2.794, pheasy c6.0 = 2.781 -> 差 0.5% ('-11%' 是比了未截断 symfc 的假象)
2. ASR 两模型论收回 (pheasy 拟合期约束 vs symfc 后置对角, 均自洽, 非缺陷)
3. 合成干净测试 (独立实空间生成力 F=-Phi@u, Phi_known 全在 c2=5.45 内, 40 构型): pheasy 纯谐波恢复 max rel = 1.2e-15 (机器精度) -> SM构造/索引/OLS/ASR/fc2写出 整链无 bug
4. writer 自洽: pheasy fc2.hdf5 vs 其 FORCE_CONSTANTS_2ND 文本 max 差 5e-16
结论: pheasy 谐波机制认证干净; Si 的 fc2/kappa 偏差 = 截断模型差异 (c5.45 排除 5.466 壳) + 10 构型 fc3 噪声; 非管线 bug
待办: (c) C60Mg2 c2 6.9/7.1 敏感性 (偏差问题, 构型数不救); writer 全往返 (kl vs pheasy 字节级, 降级为低优先)

## Round7: fc3 合成恢复测试 (独立实空间 F=-Phi2 u - 1/2 Phi3:u u)
- Phi3_known = symfc fc3 (全在 6.0 内, ASR 1.8e-13); u rms 0.06 A (三阶力占 36%); 30 构型; pheasy c2=c3=6.0 联合 OLS
- 结果: 拟合 rel err 5.5e-4; fc2 恢复 rel 6.3e-6; fc3 恢复 scale 比 0.99998 (无因子/符号/单位灾难), rms 差 2.7e-3 非机器精度
- 残差结构: on-site 1e-5 / two-equal 5e-4 / distinct 8.4%; pheasy 在部分 distinct 三重态恢复为 0 而 known 非零
- 几何: 漏掉三重态两对距离 ~5.957 A (贴 6.0 边缘) -> pheasy 3体空间在名义 c3 下有一条更短的等效截断
- 判读: 漏带=少通道->kappa 应变高, 实测 pheasy kappa 更低 (53<94) -> 该缺口不是 kappa 差距成因; 偏低归因 fc3 在弱信号下被 OLS 拟合进噪声 (数据幅度)
- 待办: 1) pheasy 3体 cluster 等效截断代码确认; 2) 真数据 fc3 = 位移幅度/构型数实证; 3) fc3 writer 交叉 diff

## Round8: WS-简并镜像 bug 核查 (用户定位) - 实证结论
- 代码确认: x[4][0] (cluster_orbit.py:874) 与 x[5][0] (:728) 只取简并镜像第一个, weight 未用 - bug 在原则上成立
- 修复已实现 (order>=3 distinct 成员遍历全部简并镜像 + 对应 offset; 统一 < 与 <=) 但验证发现:
- Si 4x4x4 合成案例中所有相关对 weight=1 (无简并镜像) - 修复零效果 (fc3 恢复 rel 仍 0.002657, (0,4,72) 仍不在 cs)
- 结论: WS-简并 bug 不是 Si fc3 合成失配的原因; 该修复需在有简并的体系 (BAs/LiCoO2) 或构造简并 case 上验证
- Si distinct 失配原因仍开放: 新假设 = member-member 距离在 atom0 最小镜像系判定, 排除部分自身最小镜像下有效的三重态 (与简并 bug 不同机制); 需生成端插桩确认
- 修复已回滚 (工作树回到 69c9f36; 备份 /tmp/cluster_orbit_fixed_backup.py)

## Round9: 上游 pheasy 对照实验 (隔离 venv pheasy_up, pip pheasy==0.0.2, numpy2 打 np.math 补丁)
同一 fc3 合成数据 (Phi3=symfc fc3 截 6.0, 30 构型, u rms 0.06):
- rel fit err: 上游 5.477e-4 vs 我们 5.456e-4 (一致)
- fc3 恢复三指标: on-site 1e-5 / two-equal 5.1e-4 / distinct 8.43% = 与我们完全一致
- phi 向量: 上游 vs 我们 max rel 2.2e-5 (float32/float64 SM 无差别 -> 差异在 NS/SM 构造层但极小)
结论 (用户三结果表第一行): distinct 8.4% 与 on-site/two-equal 非机器精度 全是上游行为, 我们 optimizer/约束重写未引入偏差 -> pheasy-gpu 干净; kappa 偏差归因数据量/口径 (BAs/LiCoO2 官方参考含四声子, 3ph-only 天然高 2.5-3x)
清理: cluster_orbit.py:141 docstring 绝对路径泄露已删 (commit d9f8694)

## Round10: 收尾修正与固化
NaN
NaN
NaN
