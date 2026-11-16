# 审查说明 —— GPU-SM 分块修复 + c3=4.5 工具包（修订版）

`Andyw852/pheasy-gpu` 上共三个提交，现已全部合入 `main`。本文供不打开 diff 也能审查，
英文原本在 `~/.dsh/.dsh-drops/REVIEW_NOTE_gpu_sm_sharding.md`。

| 提交 | 标题 | 涉及文件 | 规模 |
|---|---|---|---|
| `8b1e331` | feat(gpu-sm): allow more row shards than cards so every int32 CUDA CSR block fits | `core/gpu_backend.py`、`dev/test_gpu_memory_regressions.py`、`REPORT_gpu_sm_sharding_20261116.md` | +340 −13 |
| `744931a` | docs(fit_scripts): archive the c3=4.5 GPU fit toolkit (Mg2C60 joint) | `fit_scripts/Mg2C60_joint_4p50/*`、`fit_scripts/README.md` | +417 |
| `6c3b013` | fix(gpu-sm): required mode at construction; correct the sharding report and c3=4.5 toolkit | `core/optimizer.py`、`core/gpu_backend.py`（docstring）、2 个测试文件、新增 `dev/check_gpu_sm_shards.py`、REPORT、工具包 README + 2 个脚本、`GPU.md` | +488 −68 |

本地审查（纯 CPU，不需要 CUDA，在仓库根目录执行）：

```bash
git show 8b1e331          # 分块改动
git show 744931a          # 实操工具包
git show 6c3b013          # 审查修正
python3 dev/test_gpu_memory_regressions.py                   # 11 个测试
PYTHONPATH=. python3 dev/test_gpu_failure_semantics.py       # 6 个测试
```

---

## 1. 提交 `8b1e331` —— 代码改动

### 问题

`GpuSparseMV` 把 `SM_prime` 按行切成**每张卡正好一块**。这只在每块非零数 < `2**31 − 1` 时成立，
因为 CUDA CSR 路径的行指针和列索引都用 int32 存 —— 正是这一点让 38.5 GB 的 float32 矩阵
能装进 24 GB 消费卡。

对 Mg2C60 c3=4.5 拟合（`SM_prime (1040112, 92079)`，`nnz = 4,816,356,048`），2 张卡使每块
2.41e9 个非零，`_check_cuda_csr_indices()` 直接拒绝。

这个拟合经历过两次生产失败，结局都一样 —— 静默退到串行 CPU SpMV（单核、慢约 1000 倍、
`pcpu = 106%`、4.6 小时没跑完一轮 RFE），但**只有第一次是 int32 上限**：

- **(a) 2 张卡：** int32 拒绝（`nnz=2408182532 exceeds the int32 index range`）。
- **(b) 4 张卡：** 每卡显存预检拒绝，因为另外一名用户的**非 SLURM** 进程占着其中两张卡。
  pheasy 拒绝是正确的。

两次都"降级"而不是"中止"，是因为作业设了 `PHEASY_GPU_FALLBACK=1`。`PHEASY_GPU_MODE=required`
没能拦住，因为 `TwoLevelSM.__init__` 的构造分支只读了 fallback 开关；`matvec`/`rmatvec`
本来就遵守 required 模式。**已在 `6c3b013` 修复。**

### 根因

两个不同的量被混为一谈：

- 分块数的下界由索引格式上限决定；
- 卡数由硬件决定。

### 改动

```python
_CUDA_INT32_LIMIT = 2 ** 31 - 1
_CUDA_SHARD_SAFE_NNZ = int(_CUDA_INT32_LIMIT * 0.90)   # 10% 余量

def row_shard_count(nnz, n_dev):
    forced = os.environ.get("PHEASY_GPU_SM_SHARDS", "").strip()
    if forced:                       # 测试/调试用覆盖
        return max(1, int(forced))
    n_dev = max(1, int(n_dev))
    need = -(-int(nnz) // _CUDA_SHARD_SAFE_NNZ)          # 向上取整
    need = max(need, n_dev)
    return int(np.ceil(need / float(n_dev))) * n_dev     # 取 n_dev 的整数倍
```

`GpuSparseMV.__init__` 建 `n_shards = row_shard_count(nnz, len(devs))` 个块，按轮转映射到卡上
（`self._block_devs = [devs[i % G] ...]`）。所有依赖设备的循环都改用 `self._block_devs[i]`：
上载循环、`_mv_blocks()`（`matvec` 与默认布局的 `rmatvec` 都走它）、`_rmatvec_from_row_blocks()`。
`close()` 不依赖设备。

### 结果分块数（`6c3b013` 起测试逐格断言）

| 矩阵 | nnz | 1 卡 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| c3=4.5 Mg2C60 联合集（1040112 × 92079） | 4.816e9 | 3 | 4 | 3 | 4 | 5 |
| c3=5.0（454656 × 265662，更小的数据集，见 GPU.md） | 3.570e9 | 2 | 2 | 3 | 4 | 5 |
| 假想 | 6.0e9 | 4 | 4 | **6** | 4 | 5 |
| 小矩阵 | — | 1 | 2 | 3 | 4 | 5（不变） |

本说明第一版把 c3=5.0 那行写成"~6.0e9 → 4 4 3 4 5"：这个 nnz 在仓库里没有实测来源，
而且 3 卡那格是错的（需要 4 块，向上取整到 3 的整数倍是 6）。

10% 余量对"按 nnz 均衡"的边界来说很宽裕：一块最多超出 `nnz/n_shards` 一行（≤ 92,079 个非零；
转置块按一列算，≤ 1,040,112），而松弛量约 2.1e8。它**不覆盖** `PHEASY_SHARD_BALANCE=0`
（linspace 边界，会把 c3=5.0 的 65% 非零堆在第 0 块）；那种情况下 int32 守卫仍然拒绝，
不会悄悄溢出。

### 为什么安全

- `matvec` 按行序拼接各块结果，块放在哪张卡无关。
- `rmatvec` 要么按列序拼接已存转置块（默认布局），要么在低显存模式下把
  `R_i.t() @ u[r0:r1]` 求和。两种都与放置无关。
- 每块的内存预检在上载前执行并重新读取该卡的空闲显存，所以同一张卡上的第二块会看到
  已经减少的余量。（fake-torch 测试用常量空闲显存，因此这部分只在真 GPU 上被验证。）
- 同样的行、同样的值、同样的 float32 CSR，只是多几次更小的 kernel 调用。

### 每卡显存

同一张卡上的各块加起来非零数不变，所以每卡占用不变。在**低显存布局**下
（`PHEASY_GPU_SM_LOWMEM=1`，每个非零 4 B 值 + 4 B int32 索引），一张卡约需：

```
1.1 × (nnz / n_cards) × 8 B        （= 每卡 k 块时 k × 1.1 × (nnz / n_shards) × 8 B）
```

| c3=4.5 | 常驻 | 含 10% 工作区 | 对 23.18 GiB 空闲（RTX 3090） |
|---|---|---|---|
| 3 卡 3 块 | 12.8 GB = 12.0 GiB | 13.2 GiB | 宽裕 |
| 2 卡 4 块 | 19.27 GB = 17.9 GiB | 19.7 GiB | 能装下，偏紧 |

默认的"存转置"布局需要两倍（2 卡时每卡约 39 GiB，装不下）。

**2 卡还必须显式给 `PHEASY_GPU_SM_DEVICES`。** 不给的话 `_pick_devices()` 会按
`2 × SM / n_gpu` 过滤卡片（它假设存转置布局）：2 卡时每卡 35.9 GiB，24 GB 卡一张都不合格，
构造会在分块代码执行前就以 "no usable CUDA device" 失败。

### 向后兼容

只要 `nnz / n_dev ≤ 0.9 × (2**31 − 1)`，`row_shard_count()` 就返回 `n_dev`，于是
`_block_devs == devs`，每个循环退化为改动前的代码。已验证过的生产配置
（c3=4.0 用 5 卡、c3=4.5 用 3 卡）走的正是旧路线。`self._devs` 保留；
`self._n_shards` 与 `self._block_devs` 是新增。

### 刻意不动

- `GpuTwoLevelOperator`（常驻 LASSO/OLS 算子）仍是每卡一块。在
  `PHEASY_GPU_OLS_RESIDENT=0` / `PHEASY_GPU_LASSO_RESIDENT=0` 下它不启用。
- `sensing_matrix_on_gpu()` 保留整矩阵 `nnz ≥ 2**31` 守卫：它在单卡上物化 `SM @ NS`，
  无法用同样方式分块。

### 请审查者权衡的风险

`n_shards > n_cards` 这条路径**还没在真 GPU 上跑过**，只在 fake-torch 测试框架下通过。
在跑的作业用的是 `73cc175` 构建（本改动之前），卡数也使它走未改动的路径。第一次真 GPU
验证现在有脚本可用（`6c3b013` 新增）：

```bash
python dev/check_gpu_sm_shards.py --synthetic 400000x20000 --density 2e-3 \
    --devices 0,1 --shards 4 --lowmem
python dev/check_gpu_sm_shards.py --npz $CASE/sm_prime.npz \
    --devices 0,1 --shards 4 --lowmem
```

它打印分块布局与每块非零数，并把 `matvec`/`rmatvec` 与分块 float64 scipy 参考做比较。
退出码 0 表示两个相对误差都在 1e-4（float32）以下；块错位或丢块会给出 O(1) 误差。
比较逻辑本身也在 fake torch 下被测（包括必须被检出的"丢块"用例）。

---

## 2. 提交 `744931a` —— 实操工具包（在 `6c3b013` 中修订）

`fit_scripts/Mg2C60_joint_4p50/` 是 c3=4.5 拟合背后真正跑起来的那套东西。

| 文件 | 作用 |
|---|---|
| `submit_4p50_resume5.sh` | prep → `pheasy -f` → post。挑实测空闲显存最多的 3 张卡（每张需 ≥ 15 GiB 空闲，不足 3 张就中止），设 `PHEASY_GPU_FALLBACK=0`，并且只在 `disp_matrix.pkl` 逐字节相同时复用 `sm_prime.npz`（prep 前后各算一次 md5） |
| `pick_gpus.py` | 独立小工具：按实测空闲显存排序的可见 GPU 索引。提交脚本**并不调用**它（脚本内有自己的内联选择器） |
| `submit_colnorm.sh` + `colnorm_parallel.py` | 32 个 CPU worker 算 `SM_prime @ NS` 的列范数：2.7 分钟，串行约 9 小时。与串行定义吻合到 float64 舍入（各块按完成顺序累加），不是逐位相同 |
| `README.md` | 规模数字、环境变量表、四条铁律、实测各阶段耗时、运维坑位 |

`6c3b013` 中的修订：

- `CASE` / `CONDA_SH` 参数化。此前两个脚本把 GPU 机器上那个用户的家目录写死，
  而项目要求这类路径不进公开仓库。
- `optimizer.py` 的 md5 固定值（`73cc175`）改为特性检查；两个模块的 md5 仍会打印。
  原来的固定值对之后的构建也"匹配"，而且完全没约束 `gpu_backend.py`。
- `PHEASY_COL_NORMS_FILE` 只在复用 `sm_prime.npz` 时导出；要重建时把范数文件移到
  `.stale`（见铁律 4）。

**这个脚本走的是"一卡一块"路径。** c3=4.5 用 3 卡就是 3 块。它成功只能验证旧路线，
不能验证 `8b1e331`。

### 四条铁律（每条都是一次事后复盘）

1. **卡数 ≠ 分块数。** c3=4.5 用 2 卡 → 每块 2.41e9 非零 → 被拒。`row_shard_count()`
   现在把它切成 4 块、每块 1.20e9。3 卡时每块 1.605e9，本来就合法。两张 24 GB 卡只有在
   `PHEASY_GPU_SM_LOWMEM=1` **并且**显式给 `PHEASY_GPU_SM_DEVICES` 时才可行。
2. **生产环境用 `PHEASY_GPU_FALLBACK=0`。** 设成 `1` 时，GPU-SM 失败会退到单核 CPU，
   日志只有一行。`6c3b013` 之后 required 模式本身也会中止构造，但仍建议保持 `0`。
3. **共享机器上不要用 SLURM GRES 选卡。** 两次分到的卡上正跑着别人的非 SLURM 进程
   （各占 16–19 GiB）。做法：申请 `--gres=gpu:5`，实测 `torch.cuda.mem_get_info()`，
   只留 ≥ 15 GiB 空闲的卡，不足 3 张就中止。所需余量是**每卡**的
   `1.1 × (nnz / n_cards) × 8 B`（低显存布局）。第一版写成 `nnz / n_shards`，
   把 2 卡 4 块的需量低估了一半。
4. **prep 可能让 57.8 GB 的 `sm_prime.npz` 失效**：它会重写 `disp_matrix.pkl`
   （原子顺序记录）。之后驱动会正确拒绝缓存并重建（约 4–5 小时）。做法是 prep 前后各
   取一次 md5，只有相同才 `touch` 该 npz。无条件 `touch` 会得到一个原子顺序不同的
   感知矩阵，即静默错误的结果（早先一个作业因此从 6.9% 漂到 11%）。
   **`colnorms_4p50.npy` 同理**：它的加载器只检查形状，所以每次重建后都要重跑
   `submit_colnorm.sh`。

### 实测各阶段耗时（RTX 3090，24 GB，56 核主机）

| 阶段 | 墙钟 | 备注 |
|---|---|---|
| prep | 3–20 秒 | 复用 pickle |
| `-s` 团簇空间 ×2 | 各 1:11 | |
| `-c` 对称约束 + 零空间 | 37 分钟 | 每次都重算；只有 `-f` 会读 `ns_*.npz` 缓存 |
| `-d` 感知矩阵 | 3:20–4:40 | 铁律 4 允许复用时跳过 |
| `-f` RFE-OLS，9 轮 | 10–14 小时 | GPU SpMV 只占约 30 分钟；瓶颈是主机侧 Gram/Cholesky |
| GPU 上载（3 卡，低显存） | 2–4 分钟 | 38.5 GB 过 PCIe |
| 列范数（32 worker） | 2.7 分钟 | |

---

## 3. 提交 `6c3b013` —— 审查修正

| 位置 | 改动 |
|---|---|
| `core/optimizer.py` | `TwoLevelSM.__init__` 在 required 模式（`PHEASY_GPU_MODE=required` 或 `PHEASY_USE_GPU=1`）下，即使 `PHEASY_GPU_FALLBACK=1`，`GpuSparseMV` 失败也直接抛错 |
| `core/gpu_backend.py` | `row_shard_count` docstring：删掉错误的"at c3=5.0 even four are not enough"；写清 10% 余量覆盖什么、不覆盖什么 |
| `dev/check_gpu_sm_shards.py` | 新增真 GPU 验证脚本（见上文） |
| 测试 | `ConstructionFailurePolicyTest`（3 个测试，在旧代码上会失败）；分块数测试断言整张表；`test_real_gpu_check_script_logic` |
| REPORT | 修正 §3.2 表、§3.3 / §6.2 的每卡公式与 2 卡条件、§5 测试计数、§7 措辞与可运行配方；新增 §9 逐项列出所有更正 |
| 工具包 | 脚本与 README 即第 2 节所述 |
| `GPU.md` | 把"构造"也写进 required 模式的约定 |

**测试。** `test_gpu_memory_regressions.py` 11 个、`test_gpu_failure_semantics.py` 6 个、
`test_resident_footprint_estimate.py` 11 个、`test_colnorm_failfast.py` 7 个，全过。
（第一版把后两个文件写成"28"个；实际是 18 个。）全部 43 个 `dev/test_*.py` 里有 25 个通过；
其余 18 个在改动前后失败方式完全相同，多数是因为审查容器缺少 `torch`、`ase`、`h5py`、
`spglib` 或已安装的 `pheasy_gpu`。`test_gpu_operator_ridge.py` 另有 2 个断言失败；
另有 1 个陈旧测试见第 5 节（已在本轮一并修好，见下）。

---

## 4. 部署状态

| 主机 | 构建 | 状态 |
|---|---|---|
| GitHub `main` | `6c3b013` | 已推送（原 `claude/quirky-dirac-wl1p62` 已合入） |
| 本地 `~/software/pheasy-gpu` | `6c3b013` | 已同步，测试通过 |
| 3090 `~/software/pheasy-gpu-gh` | `73cc175` → 待同步 | c3=4.5 拟合正在跑，本轮已安排同步 |
| A800、jzzn | `73cc175` → 待同步 | 同上 |

`8b1e331` 推送前已 rebase 到 `340844a`（另一会话的 `test/Mg4C60/` κ 图提交），无冲突。

---

## 5. 留给审查者的问题

- 分块数是否应直接由实际边界算出（对均衡与 linspace 两种边界都精确），而不是用 0.9 的目标值？
- `_pick_devices()` 的自动过滤是否应按低显存布局来估算（`1.1 × SM / n_gpu` 而不是
  `2 × SM / n_gpu`）？那样 2 卡 c3=4.5 就不必显式给 `PHEASY_GPU_SM_DEVICES`。
  之所以没改，是因为它会改变自动路径挑哪几张卡。
- `GpuTwoLevelOperator` 现在就同样处理，还是继续一卡一块？
- `PHEASY_GPU_SM_SHARDS` 是不是合适的逃生口？强制值不是卡数整数倍时会被接受，
  并导致各卡负载不均。
- 是否应由驱动自己对 `sm_prime.npz` 和列范数做内容哈希比较，而不是靠 shell 脚本里的 md5 守卫？
- `dev/test_optimizer_large_regressions.py::test_gpu_construction_failure_does_not_cache_host_transpose`
  是陈旧测试：它没有设 `PHEASY_GPU_FALLBACK=1` 却期待 CPU 回退，而且它装的假模块并不是
  优化器真正 import 的那个。改动前后都失败。（本轮已修好：改为 patch
  `core.gpu_backend.GpuSparseMV` 并显式设 `PHEASY_GPU_FALLBACK=1`，该文件 29 个测试全过。）
- 约十个旧文件（报告、`test/Mg2C60/` 下的验证日志）仍带 GPU 机器的用户家目录。两个新脚本
  已不再带，但该路径仍留在 `744931a` 的已发布历史里。
- `8b1e331` 的提交信息把 CSC 写成了 "CSP"；修正它需要重写已发布历史。

仓库里的长文档：`REPORT_gpu_sm_sharding_20261116.md` 与
`fit_scripts/Mg2C60_joint_4p50/README.md`。
