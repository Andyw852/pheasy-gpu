#!/usr/bin/env bash
# =============================================================================
#  fit_3090.sh —— RTX 3090 单机拟合包装（生产入口）
#
#  在准备好的数据目录里运行（需要 POSCAR / SPOSCAR / disp_matrix.pkl /
#  force_matrix.pkl；cs.pkl、ns_*.npz、sm_prime.npz、phi.npz 等缓存存在时会自动复用）：
#
#      bash /path/to/pheasy-gpu/fit_3090.sh [KEY=VALUE ...]
#      bash /path/to/pheasy-gpu/fit_3090.sh FIT_METHOD=LASSO DEVICES=0,1,2 NDATA=296
#
#  本脚本只做两件事：(1) 把 KEY=VALUE 收敛成一组 GPU/两级默认值（如下），
#  (2) 交给仓库根的 pheasy_fit.sh 执行。参数总表见 pheasy_fit.sh 头部注释。
#
#  导出的默认值（都可用同名 KEY=VALUE 覆盖）:
#      GPU_MODE=required  GPU_FALLBACK=0      DEVICES=0,1,2   GPU_RESIDENT=1
#      TWOLEVEL=1         DEBIAS=0            CV=5            NMU=20
#      STANDARDIZE=true   SM_DTYPE=float32    NCPU=8          NDATA=(全部)
#      LASSO_TOL=1e-4     LASSO_MAX_ITER=20000
#      CV_TOL=1e-3        CV_MAX_ITER=400
#  ⚠ DEVICES 是**可见卡数**而不是卡号列表：CUDA_VISIBLE_DEVICES 只给一张卡时，默认
#  DEVICES=0,1,2 会直接报 "PHEASY_GPU_DEVICES must contain unique visible CUDA device
#  IDs"（实测踩过）。单卡跑就把 DEVICES 设成 0；多卡跑要保证 CUDA_VISIBLE_DEVICES 里
#  的可见数量不少于 DEVICES 的项数。
#
#  GPU_MODE=required + GPU_FALLBACK=0 表示"要 GPU 就必须是 GPU"：任何回退都会报错，
#  而不是悄悄用 CPU 跑完再告诉你。
#
# -----------------------------------------------------------------------------
#  各种拟合方法（FIT_METHOD=...）
# -----------------------------------------------------------------------------
#  | 方法           | GPU 求解路径                          | 特征 / 备注
#  | OLS            | gpu_resident_iterative（两级 CGLS，   | 无正则化基准解；Jacobi 列缩放默认开。
#  |                | Jacobi 默认开）                       | 宽矩阵（小 NDATA，如 36864x69487）走不了
#  |                |                                       | 流式 TSQR —— 那要求高矩阵。float32 下
#  |                |                                       | 无惩罚判据在 ~1e-3 见底，会以实测地板
#  |                |                                       | (precision_floor) 认证；不要用放大的 tol
#  |                |                                       | 硬凑：实测 atol=3e-3 在 45 步就"通过"，
#  |                |                                       | 但残差比按地板停的那次差 20 倍。
#  | RIDGE          | gpu_resident_iterative（增广 CGLS）   | L2 岭回归，CV 选 alpha（MU_MIN/MU_MAX/NMU）。
#  |                |                                       | 不产生稀疏性（nnz 满）。CV 折现在是父算子
#  |                |                                       | 的"行视图"：整次拟合只上传一次因子。
#  | LASSO          | 驻留 FISTA + CV（GPU 上标准化）       | L1 稀疏解 + CV 选 alpha；--std 对该方法生效。
#  |                |                                       | PHEASY_LASSO_DEBIAS=1 追加 relaxed 去偏
#  |                |                                       | （残差显著下降，推荐）。alpha 网格过窄会把
#  |                |                                       | alpha* 顶到网格边界 —— 用 ALPHA_DECADES /
#  |                |                                       | PHEASY_LASSO_GRID_FLOOR 放宽。
#  | ALASSO         | 同上 + 自适应权重                     | 权重来自 pilot 解；pilot 有独立容差
#  |                |                                       | PHEASY_ALASSO_PILOT_TOL（默认 1e-5），
#  |                |                                       | 太紧的 pilot 会让权重退化成"截断 OLS"。
#  |                |                                       | 通常稀疏性最强、泛化最好。
#  | RFE            | 驻留子集求解（GPU subset CGLS）       | 递归特征消除 + 分组 CV：每轮都要做完整子集
#  |                |                                       | 求解，轮数多、耗时最长。外层必须串行
#  |                |                                       | （PHEASY_N_JOBS!=1 时会自动串行并告警）。
#  |                |                                       | 先在小构型子集上试，再放大。
#  | RFE-OLS-TSQR   | RFE + 高瘦 QR（PHEASY_GPU_TSQR）      | 超大规模专用；TSQR 需要保留 O(p^2) 的 R 因子，
#  |                |                                       | 内存要先算够；同样要求高矩阵。
#
#  选方法的经验（本项目材料 Mg8C120 实测，24 构型/2 折分组 holdout 的相对 L2 误差）:
#      LASSO 1.11e-01   ALASSO 1.24e-01   OLS 5.66e-01   RIDGE 5.66e-01
#  即在该问题上 L1 稀疏化比 OLS/RIDGE 好约 5 倍；所以常规流程是：先跑 LASSO / ALASSO，
#  用 OLS / RIDGE 做无正则化基准（RIDGE 还能给"列尺度是否失控"的旁证）。
#
#  每个方法的求解都各自出证书：结果与 fit_manifest.json 里能读到 solver_info /
#  regularized_solver_info（backend、stop_reason、converged、normr/normar、以及 CGLS 的
#  probe_start/probe_count/stall_floor 等）。fit_accepted=false 时先看 stop_reason：
#  precision_floor / stall_above_floor 表示"这个精度到不了请求的容差"（换容差或接受拒绝），
#  iteration_limit 才是"预算不够"。
#
#
#  ⚠ RFE / RFE-OLS-TSQR 的两种跑法（实测 2026-09-17，Mg8C120，required 模式）：
#  * 默认（推荐先跑这个）：子集求解是**排序**用途，允许在**实测地板**处停下并记录
#    （info.floor_accepted / floor_note，随 iterative_diagnostics 进 manifest），
#    拟合能跑完（实测 5 轮 / 4 轮、选中 4343 / 8685 特征）。但**验收仍是
#    accepted=False**：全量那一步的实测地板高于可认证上限 1e-3，按诚实语义只能拒绝——
#    系数照常写出，理由写在证书里。
#    `PHEASY_RFE_RANKING_FLOOR=0` 可退回旧的 fail-closed（会中途抛错）。
#  * 想让 accepted=True：给子集求解一个**够得着**的容差 + 更大预算（实测通过）：
#        PHEASY_LSQR_MAXITER=20000 PHEASY_LSQR_ATOL=1e-3 PHEASY_LSQR_BTOL=1e-3
#  例：PHEASY_LSQR_MAXITER=20000 PHEASY_LSQR_ATOL=1e-3 bash fit_3090.sh \
#          FIT_METHOD=RFE C3_CUTOFF=4.0 NDATA=296
#  注意代价：放宽容差换到的迭代并不是更准的解，而且会**改变排序**（实测同一份数据：
#  放宽路径 round-0 CV_RMSE 2.505e-01、默认地板路径 3.401e-01，最终选中的特征数也不同），
#  （相对判据在 ||r|| 大时也满足）。若你要的是"排序可用"而不是"系数可信"，这没问题；
#  若要保留 fail-closed 语义，就别放宽容差，而是把这一步改成"允许在实测地板处停下并
#  记录"——这属于设计取舍，需要你拍板。
#
#  多方法并行（一卡一 fit）：用同目录的 fit_3090_parallel.sh，不要指望把一次拟合拆到多卡
#  更快 —— 实测 RIDGE cv=5 单卡 606.8 s、三卡 713.5 s（本机 GPU 之间无 P2P）。
# =============================================================================

set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # 仓库根（本脚本现位于 fit_scripts/）
PHEASY_EXECUTABLE="${PHEASY_EXECUTABLE:-pheasy-gpu}"
PYTHON="${PYTHON:-python3}"
DEVICES="${DEVICES:-0,1,2}"
GPU_MODE="${GPU_MODE:-required}"
GPU_FALLBACK="${GPU_FALLBACK:-0}"
GPU_RESIDENT="${GPU_RESIDENT:-1}"
TWOLEVEL="${TWOLEVEL:-1}"
DEBIAS="${DEBIAS:-0}"
FIT_METHOD="${FIT_METHOD:-LASSO}"
FIT_ORDER="${FIT_ORDER:-3}"
C2_CUTOFF="${C2_CUTOFF:-None}"
C3_CUTOFF="${C3_CUTOFF:-5.2}"
C4_CUTOFF="${C4_CUTOFF:-None}"
NDATA="${NDATA:-}"
CV="${CV:-5}"
NMU="${NMU:-20}"
ALPHA_DECADES="${ALPHA_DECADES:-4.0}"
STANDARDIZE="${STANDARDIZE:-true}"
LASSO_TOL="${LASSO_TOL:-1e-4}"
LASSO_MAX_ITER="${LASSO_MAX_ITER:-20000}"
CV_TOL="${CV_TOL:-1e-3}"
CV_MAX_ITER="${CV_MAX_ITER:-400}"
NCPU="${NCPU:-8}"
SM_DTYPE="${SM_DTYPE:-float32}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
_ALLOWED="FIT_METHOD FIT_ORDER C2_CUTOFF C3_CUTOFF C4_CUTOFF NDATA CV NMU ALPHA_DECADES STANDARDIZE LASSO_TOL LASSO_MAX_ITER CV_TOL CV_MAX_ITER NCPU SM_DTYPE FORCE_REBUILD DEVICES GPU_MODE GPU_FALLBACK GPU_RESIDENT TWOLEVEL DEBIAS PHEASY_EXECUTABLE PYTHON"
for kv in "$@"; do
  case "$kv" in
    *=*) key="${kv%%=*}"; value="${kv#*=}"; case " $_ALLOWED " in *" $key "*) printf -v "$key" '%s' "$value" ;; *) echo "Unknown KEY=$key" >&2; exit 2 ;; esac ;;
    *) echo "Arguments must be KEY=VALUE: $kv" >&2; exit 2 ;;
  esac
done
case "$FIT_METHOD" in OLS|LASSO|ALASSO|RFE|RFE-OLS-TSQR|RIDGE) ;; *) echo "Unsupported FIT_METHOD=$FIT_METHOD" >&2; exit 2 ;; esac
command -v "$PYTHON" >/dev/null || { echo "Python not found: $PYTHON" >&2; exit 2; }
command -v "$PHEASY_EXECUTABLE" >/dev/null || { echo "pheasy-gpu not found; install with pip install -e '.[gpu]'" >&2; exit 2; }
for f in POSCAR SPOSCAR disp_matrix.pkl force_matrix.pkl; do [[ -f "$f" ]] || { echo "Missing $f in $(pwd)" >&2; exit 2; }; done
export PHEASY_GPU_MODE="$GPU_MODE" PHEASY_GPU_FALLBACK="$GPU_FALLBACK" PHEASY_GPU_DEVICES="$DEVICES"
export PHEASY_GPU_NGPU="$(awk -F, '{print NF}' <<< "$DEVICES")"
export PHEASY_GPU_LASSO_RESIDENT="$GPU_RESIDENT" PHEASY_LASSO_TWOLEVEL="$TWOLEVEL" PHEASY_LASSO_DEBIAS="$DEBIAS"
export LASSO_SPARSE="$GPU_RESIDENT" LASSO_TWOLEVEL="$TWOLEVEL"
export PHEASY_CV_TOL="$CV_TOL" PHEASY_CV_MAX_ITER="$CV_MAX_ITER"
export OPENBLAS_NUM_THREADS="$NCPU" OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU" PHEASY_N_JOBS="$NCPU"
printf '%s\n' "[pheasy-gpu/3090] cwd=$(pwd)" "[pheasy-gpu/3090] devices=$DEVICES ngpu=$PHEASY_GPU_NGPU" "[pheasy-gpu/3090] method=$FIT_METHOD order=$FIT_ORDER c3=$C3_CUTOFF" "[pheasy-gpu/3090] cv=$CV nalpha=$NMU cv_tol=$CV_TOL cv_max_iter=$CV_MAX_ITER" "[pheasy-gpu/3090] final_tol=$LASSO_TOL final_max_iter=$LASSO_MAX_ITER debias=$DEBIAS"
exec bash "$ROOT_DIR/pheasy_fit.sh" FIT_METHOD="$FIT_METHOD" FIT_ORDER="$FIT_ORDER" C2_CUTOFF="$C2_CUTOFF" C3_CUTOFF="$C3_CUTOFF" C4_CUTOFF="$C4_CUTOFF" NDATA="$NDATA" CV="$CV" NMU="$NMU" ALPHA_DECADES="$ALPHA_DECADES" STANDARDIZE="$STANDARDIZE" LASSO_TOL="$LASSO_TOL" LASSO_MAX_ITER="$LASSO_MAX_ITER" NCPU="$NCPU" SM_DTYPE="$SM_DTYPE" FORCE_REBUILD="$FORCE_REBUILD" LASSO_SPARSE="$GPU_RESIDENT" LASSO_TWOLEVEL="$TWOLEVEL"
