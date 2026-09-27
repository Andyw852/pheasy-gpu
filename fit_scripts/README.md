# fit_scripts/ —— 拟合包装脚本

这个目录放**拟合的入口包装脚本**；真正的共享驱动 `pheasy_fit.sh` 仍留在仓库根目录，
原因见下。

## 内容

| 文件 | 作用 |
| --- | --- |
| `fit_3090.sh` | 生产包装（原仓库根的那个）：把 KEY=VAL 收敛成一组 GPU/两级默认值再交给 `pheasy_fit.sh`。典型用法 `bash fit_3090.sh FIT_METHOD=RIDGE DEVICES=0,1,2 NDATA=296`。 |
| `fit_3090_parallel.sh` | **一卡一 fit** 的并行编排：按空闲显存挑卡、每个方法独立运行目录、结束后从各自 `fit_manifest.json` 打印汇总表。用法 `bash fit_3090_parallel.sh FIT_METHODS="RIDGE OLS LASSO ALASSO" NDATA=296 NCPU=8 DEVICES="0 1 2 3"`。 |

仓库根目录保留了一个 `fit_3090.sh` **转发入口**（6 行，`exec` 到本目录），所以既有命令
`bash /path/to/pheasy-gpu/fit_3090.sh ...` 照旧可用。

两个脚本的**头部注释就是方法手册**：`fit_3090.sh` 有 6 个方法（OLS/RIDGE/LASSO/ALASSO/
RFE/RFE-OLS-TSQR）的求解路径、特征与坑位对照表，外加选方法经验（本项目材料分组 holdout：
LASSO 1.11e-01 / ALASSO 1.24e-01 / OLS 5.66e-01 / RIDGE 5.66e-01），`fit_3090_parallel.sh`
有并行场景下各方法的注意点。`bash <脚本> --help` 会把对应头部打出来（非 KEY=VAL 的参数
也走这条路径，未知键则立刻报错退出，不会先去占卡）。证书字段的含义（`stop_reason`、
`atol_effective`、`stall_floor`、`environment`…）写在仓库根 `pheasy_fit.sh` 的头部。

## 为什么 `pheasy_fit.sh` 没搬进来

它是**已纳入 git** 的共享驱动，而且被按**名字**引用：`scan_methods.sh` 会 `cp`
`${ROOT}/pheasy_fit.sh` 进每个扫描子目录后在那里执行；`fit_3090.sh` 通过 `ROOT_DIR`
找到它。搬走会同时打断这两处以及所有材料目录里的既有习惯。两个包装脚本现在会先找
上一级、再回退到本目录（也方便把整个 `fit_scripts/` 拷到材料目录里用）。
如果确实要连它一起搬，需要同时改 `scan_methods.sh`（第 48/78 行的按名 cp）——告诉我即可。

## 两条实测出来的硬规则（别改回去）

1. **输入小文件必须复制，不能符号链接。** 拟合阶段会在工作目录里**重写** `SPOSCAR` 和
   `phi.npz`（见 `run_pheasy.py` 里 `_content_sig` 的注释："-s rewrites SPOSCAR"），符号
   链接会让写入穿透到源材料目录——本模板早期版本就是这样把用户材料里的 `SPOSCAR` 覆盖了
   （后来按 `cs.pkl.meta.json` 记录的 sha256 前缀从 `SPOSCAR.orig` 逐字节复原）。现在
   规则是**按体积**：< 8 MiB 复制（`POSCAR`/`SPOSCAR`/`phi.npz`/`cs.pkl`），只有 GB 级
   只读缓存（`sm_prime.npz`、`ns_*.npz`、位移/力矩阵）符号链接；并且套规则前先删掉上一次
   运行残留的链接。
2. **一次 fit 用一张卡。** 实测（Mg8C120, 296 构型, 454656x69487, float32）：RIDGE cv=5
   单卡 606.8 s、三卡 713.5 s——本机 GPU 两两之间没有 P2P，跨卡拷贝经主机内存且落在关键
   路径上；而每个驻留 fit 自己要 ~17-19 GiB 显存，24 GiB 卡一张只放得下一个 fit。所以把
   卡分给**不同的 fit**，不要拆同一个 fit。

## 与生产脚本一致的默认值

`fit_3090_parallel.sh` 给每个 fit 注入的 GPU/两级默认值与 `fit_3090.sh` 导出的那几项一致
（`PHEASY_GPU_MODE=required`、`PHEASY_GPU_FALLBACK=0`、`PHEASY_GPU_{LASSO,RIDGE,OLS,RFE}_RESIDENT=1`、
`PHEASY_LASSO_TWOLEVEL=1`、`LASSO_TWOLEVEL=1`、`LASSO_SPARSE=1`、`PHEASY_CV_TOL/PHEASY_CV_MAX_ITER` 等）。
不设这些时流水线会把 SM 稠密化（实测 NDATA=24 就写了 10.2 GB 的 `sm_dense.npy`），随后稠密
GPU 求解因显存预算不足被 fail-closed 拒绝——那是配置问题，不是求解器问题。

刻意**不**设 `PHEASY_GPU_TSQR`：流式 TSQR 要求矩阵是高的，小 NDATA 下 SM 是宽的
（实测 NDATA=24 为 36864x69487），开了它 OLS 会被正确拒绝（"gpu_tsqr requires a tall matrix"）。

## 用别的源码树跑（验证改动）

`PHEASY_SRC=<含 run_pheasy.py 的树>` 会让模板用那棵树而不是已安装的 `pheasy-gpu`：它写一个
一行 shim 作为 `PHEASY_EXECUTABLE`，并把一个名为 `pheasy_gpu` 的符号链接放到 `PYTHONPATH`
最前面。注意 `run_pheasy.py` 会把自己所在目录加入 `sys.path`——如果那棵树里还留着指向**别的**
checkout 的 `pheasy_gpu` 链接，导入会悄悄落到旧代码上（本模板调试时踩过：manifest 里没有新证书
字段、告警行号对不上）。
