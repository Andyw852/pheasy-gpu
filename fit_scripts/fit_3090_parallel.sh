#!/bin/bash
# =============================================================================
#  fit_3090_parallel.sh —— 一卡一 fit 的并行编排模板
#
#  依据（实测，Mg8C120 296 构型 454656x69487 float32，RTX3090）：把一次 fit 拆到
#  多张卡并不加速 —— RIDGE cv=5 三卡 713.5 s、单卡 606.8 s。本机 GPU 两两之间没有
#  P2P（torch.cuda.can_device_access_peer 全 False），跨卡拷贝经主机内存且落在关键
#  路径上，分片只该用于"因子装不进一张卡"的场合；而每个驻留 fit 自己要 ~17-19 GiB
#  显存，24 GiB 的卡一张只放得下一个 fit。所以：卡分给不同的 fit。
#
#  用法（在含 POSCAR / SPOSCAR / disp_matrix.pkl / force_matrix.pkl 的材料目录里）：
#     bash fit_3090_parallel.sh FIT_METHODS="RIDGE OLS LASSO ALASSO" [其它 KEY=VAL]
#
#  本脚本自己的键（不转发给 pheasy_fit.sh）：
#     FIT_METHODS      空格分隔的方法列表              (默认 "RIDGE OLS")
#     DEVICES          可用卡号列表                    (默认 "0 1 2 3")
#     MIN_FREE_MIB     派活前要求的最小空闲显存 MiB     (默认 19000)
#     PYTHON           解释器                          (默认 python3)
#     PHEASY_SRC       源码树根(含 run_pheasy.py)；设了就用它，而不是已安装的
#                      pheasy-gpu
#     OUTROOT          输出根目录                      (默认 ./parallel_fits)
#     WAIT_MAX_MIN     等空闲卡的最长分钟数             (默认 180)
#     FIT_SCRIPT       pheasy_fit.sh 路径（默认与本脚本同目录）
#  其余 KEY=VAL 原样转发；FIT_METHOD 由本脚本逐方法注入。
#
#  每个方法在自己的 $OUTROOT/<方法>/ 目录里跑，因此 fc2/fc3/fit_manifest.json
#  互不覆盖。输入按体积分流：< COPY_MAX_MIB(8 MiB) 的**复制**（POSCAR/SPOSCAR/
#  phi.npz/cs.pkl —— 拟合阶段会重写 SPOSCAR 与 phi.npz，符号链接会让写入穿透到
#  源材料目录），只有 GB 级只读缓存（sm_prime.npz / ns_*.npz / 位移力矩阵）才符号
#  链接，避免重建 9 GB 的 sensing matrix。套规则前会先删掉上一次运行残留的链接。
#
# -----------------------------------------------------------------------------
#  各种方法并行时的注意点（FIT_METHODS 里怎么写）
# -----------------------------------------------------------------------------
#  * 每个方法独占一张卡（每个驻留 fit 约 17-19 GiB 显存），所以 FIT_METHODS 的个数
#    不该超过可用卡数；多出来的方法会排队等卡（WAIT_MAX_MIN）。
#  * `OLS` / `RIDGE`：单卡满尺寸实测约 470 s / 607 s（NDATA=296, cv=5），适合做基准。
#  * `LASSO` / `ALASSO`：分钟级，且泛化最好（本项目材料分组 holdout 比 OLS/RIDGE 好
#    约 5 倍）；两者都写进去是最常见的组合，各自独立目录便于对比 fit_manifest.json。
#  * `RFE-OLS`（别名 `RFE`，会被改写成全称）/ `RFE-OLS-TSQR`：耗时最长（每轮完整
#    子集求解），**外层强制串行**。
#    [FIX RFE-GRAM] 特征数进入 PHEASY_RFE_GRAM_GB 预算（默认 min(16 GB, 空闲/4)，
#    5 折约 1.5 万特征）后每一轮都是 float64 精确求解；稠密 SM 输入本来就在 GPU 上做
#    float64 QR。只有超出预算的大问题（如 Mg8C120）还走迭代子集求解：默认允许停在
#    实测地板（floor_accepted/floor_note），全量那步可能 accepted=False；要
#    accepted=True 就加 PHEASY_LSQR_MAXITER=20000 PHEASY_LSQR_ATOL=1e-3
#    PHEASY_LSQR_BTOL=1e-3。
#    两者的消除过程相同（每个子集都是精确 OLS、同样的排序），区别只在**怎么选特征数**：
#    RFE-OLS 用分组 CV + 1-SE（CV 最小值一个标准误以内最稀疏的那个，PHEASY_RFE_1SE=1；
#    =0 取 CV 最小）；RFE-OLS-TSQR 默认用 AIC（PHEASY_TSQR_CRITERION=aic|bic|cv，n 取
#    力分量行数）。设成 cv 时两者结果逐位相同，一起跑只是重复。
#  * 小 NDATA 时 SM 是**宽**的（实测 NDATA=24 为 36864x69487），所以不要开
#    PHEASY_GPU_TSQR（要求高矩阵，会正确拒绝）；本脚本刻意不设它。
#  * 每个方法的产物（fc2/fc3/fit_manifest.json/fit.log）在各自目录里，脚本结束后
#    打印的汇总表直接读各自的 fit_manifest.json（accepted / backend / stop_reason /
#    rmse / CGLS 探针字段）。
#
#  与生产脚本一致的默认值（fit_3090.sh 导出的那几项）已注入每个 fit：
#    PHEASY_GPU_MODE=required  PHEASY_GPU_FALLBACK=0  PHEASY_GPU_LASSO_RESIDENT=1
#    PHEASY_LASSO_TWOLEVEL=1   LASSO_TWOLEVEL=1        LASSO_SPARSE=1
#    PHEASY_CV_TOL=1e-3        PHEASY_CV_MAX_ITER=400  PHEASY_N_JOBS/线程数=NCPU
#  [FIX EXACT-LS] RIDGE / OLS / RFE 的常驻开关只有显式给了环境变量 GPU_RESIDENT=0|1
#  才导出（和 fit_3090.sh 一样）。以前这里无条件导出 PHEASY_GPU_{RIDGE,OLS,RFE}_RESIDENT=1，
#  而显式 =1 表示"就要常驻迭代求解器"，于是 float32 两级 SM 上的 RIDGE/OLS 永远停在
#  float32 精度地板（日志里"LSMR 容差被抬到 1.19e-6"），第五轮的 CPU float64 精确解
#  （cpu_gram_exact / cpu_gram_ridge_exact，p 在 PHEASY_EXACT_GRAM_GB 内时）从未生效。
#  不设时：预算内走精确解，超出预算仍是常驻 GPU（required 模式默认）。
#  不设这些时流水线会把 SM 稠密化（实测 NDATA=24 就写了 10.2 GB sm_dense.npy），
#  随后稠密 GPU 求解因显存预算不足被 fail-closed 拒绝 —— 那是配置问题，不是求解器
#  问题，而报错信息看起来很像求解器故障。
# =============================================================================
set -uo pipefail

FIT_METHODS="RIDGE OLS"
DEVICES="0 1 2 3"
MIN_FREE_MIB=19000
PYTHON="${PYTHON:-python3}"    # set -u：不设时直接取默认，别写 PYTHON="$PYTHON"
PHEASY_SRC=""
OUTROOT="$PWD/parallel_fits"
WAIT_MAX_MIN=180
_SELF_DIR="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
_ROOT_DIR="$(cd "$_SELF_DIR/.." && pwd)"
# pheasy_fit.sh 是仓库根的共享驱动（scan_methods.sh 与两个包装脚本都按名字引用它），
# 默认先找上一级，再回退到本目录（也便于把整个 fit_scripts/ 拷到材料目录里用）。
if [ -f "$_ROOT_DIR/pheasy_fit.sh" ]; then
  FIT_SCRIPT="$_ROOT_DIR/pheasy_fit.sh"
else
  FIT_SCRIPT="$_SELF_DIR/pheasy_fit.sh"
fi

PASSTHRU=()
for kv in "$@"; do
  key="$kv"; val="$kv"
  case "$kv" in
    *=*) key="${kv%%=*}"; val="${kv#*=}" ;;
    *)
      # 不是 KEY=VAL：当作 --help 处理，把脚本头部的用法/方法说明打出来。
      # 以前这里静默 continue，敲错一个参数会被无声忽略。
      sed -n "2,/^# =====/p" "$BASH_SOURCE" | sed "s/^# \{0,1\}//"
      case "$kv" in -h|--help) exit 0 ;; *) exit 2 ;; esac ;;
  esac
  case "$key" in
    FIT_METHODS)  FIT_METHODS="$val" ;;
    DEVICES)      DEVICES="$val" ;;
    MIN_FREE_MIB) MIN_FREE_MIB="$val" ;;
    PYTHON)       PYTHON="$val" ;;
    PHEASY_SRC)   PHEASY_SRC="$val" ;;
    OUTROOT)      OUTROOT="$val" ;;
    WAIT_MAX_MIN) WAIT_MAX_MIN="$val" ;;
    FIT_SCRIPT)   FIT_SCRIPT="$val" ;;
    *)            PASSTHRU+=("$kv") ;;
  esac
done
# [RFE-OLS] RFE 是 RFE-OLS 的别名：先改写成全称，输出目录和汇总都用 RFE-OLS
_fm=""
for _m in $FIT_METHODS; do
  [ "$_m" = "RFE" ] && _m="RFE-OLS"
  _fm="${_fm:+$_fm }$_m"
done
FIT_METHODS="$_fm"
if [ ! -f "$FIT_SCRIPT" ]; then
  echo "找不到 pheasy_fit.sh: $FIT_SCRIPT（用 FIT_SCRIPT=... 指定）" >&2; exit 2
fi
# 转发的键先校验：以前未知键会被静默转发，脚本照样去排卡、起拟合，等 pheasy_fit.sh
# 在某个 fit 内部报 Unknown KEY 时卡已经被占了（实测踩过）。这里直接对着 pheasy_fit.sh
# 的文本核对（它的头部注释表与 _ALLOWED 列全了所有键），不认识就立刻退出。
for _kv in "${PASSTHRU[@]:-}"; do
  [ -z "$_kv" ] && continue
  _key="${_kv%%=*}"
  if ! grep -q -- "$_key" "$FIT_SCRIPT"; then
    echo "未知键 $_key：$FIT_SCRIPT 里没有它。用 --help 看本脚本自己的键；" >&2
    echo "pheasy_fit.sh 的键见它头部注释表。" >&2
    exit 2
  fi
done
command -v "$PYTHON" >/dev/null 2>&1 || { echo "找不到解释器 $PYTHON" >&2; exit 2; }

# pheasy_fit.sh 以 "$PHEASY_EXECUTABLE" 调用（带引号，所以必须是一个可执行文件，
# 不能写成 "python run_pheasy.py"），而 run_pheasy.py 里 import 的是 pheasy_gpu.*
# 包名。shim 因此同时做两件事：执行指定树里的脚本，并把名为 pheasy_gpu 的符号
# 链接放到 PYTHONPATH 最前面，让包解析到那棵树。
mkdir -p "$OUTROOT/_bin"
if [ -n "$PHEASY_SRC" ]; then
  ln -sfn "$PHEASY_SRC" "$OUTROOT/_bin/pheasy_gpu"
  printf '#!/bin/bash\nexec env PYTHONPATH="%s" %s %s/run_pheasy.py "$@"\n' \
      "$OUTROOT/_bin" "$PYTHON" "$PHEASY_SRC" > "$OUTROOT/_bin/pheasy-gpu"
  chmod +x "$OUTROOT/_bin/pheasy-gpu"
  export PHEASY_EXECUTABLE="$OUTROOT/_bin/pheasy-gpu"
  echo "[setup] 源码树 $PHEASY_SRC (shim: $PHEASY_EXECUTABLE)"
else
  export PHEASY_EXECUTABLE="${PHEASY_EXECUTABLE:-pheasy-gpu}"
  echo "[setup] 已安装的 $PHEASY_EXECUTABLE"
fi
# pheasy_fit.sh 结尾用 python3 读 hdf5：把所选解释器目录放到 PATH 前面，免得一个
# 没有 h5py 的 python3 把已经成功的拟合报成失败。
export PATH="$(dirname "$(command -v "$PYTHON")"):$PATH"

# 输入清单 + 复制/链接规则。规则是**按大小**：小于 COPY_MAX_MIB 的一律复制，只有
# GB 级的数据缓存才符号链接。理由是实测事故：拟合阶段会重写 SPOSCAR 与 phi.npz，
# 符号链接会让写入穿透到源材料目录（两次把用户材料里的 SPOSCAR 覆盖掉，已按
# cs.pkl.meta.json 记录的 sha256 前缀从 SPOSCAR.orig 复原）。
INPUTS="POSCAR SPOSCAR disp_matrix.pkl force_matrix.pkl cs.pkl cs.pkl.meta.json
neighbor_list.pkl phi.npz dataset_disps.npy dataset_forces.npy dataset_alignment.json
sm_prime.npz ns_harm.npz ns_harm.npz.meta.json ns_anharm3.npz .pheasy_stamp_struct .pheasy_stamp_data"
COPY_MAX_MIB=8
free_mib() { nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d " "; }
pick_card() {
  used=" $1 "
  for c in $DEVICES; do
    case "$used" in *" $c "*) continue ;; esac
    f=$(free_mib "$c")
    [ -z "$f" ] && continue
    if [ "$f" -ge "$MIN_FREE_MIB" ]; then echo "$c"; return 0; fi
  done
  return 1
}

mkdir -p "$OUTROOT"
BUSY="" ; PIDS=() ; TAGS=() ; STARTS=()
for m in $FIT_METHODS; do
  card="" ; waited=0
  while [ -z "$card" ]; do
    card=$(pick_card "$BUSY") || card=""
    if [ -z "$card" ]; then
      if [ "$waited" -ge "$WAIT_MAX_MIN" ]; then
        echo "[sched] 等空闲卡超过 $WAIT_MAX_MIN 分钟，放弃 $m" >&2; break 2
      fi
      sleep 30 ; waited=$((waited + 1))
      [ $((waited % 4)) -eq 0 ] && echo "[sched] $m 等卡中 ($waited 次探测)"
    fi
  done
  BUSY="$BUSY $card"
  RUN="$OUTROOT/$m" ; mkdir -p "$RUN"
  # 小文件一律**复制**：流水线自己在拟合阶段会重写 SPOSCAR（见 run_pheasy.py 的
  # _content_sig 注释），而符号链接会让这次写入穿透到源材料目录。实测事故：用另一棵
  # 代码树跑本模板时，生成器产出的 SPOSCAR 与材料目录原有内容不同，符号链接把用户
  # 的 SPOSCAR 覆盖掉了（已按 cs.pkl.meta.json 记录的 sha256 前缀从 SPOSCAR.orig
  # 复原）。只有 GB 级、且拟合阶段只读的数据缓存才用符号链接。
  for f in $INPUTS; do
    [ -e "$f" ] || continue
    if [ -L "$RUN/$f" ]; then rm -f "$RUN/$f"; fi      # 旧运行留下的链接不能留：
                                                        # 它会继续把写入穿透出去
    [ -e "$RUN/$f" ] && continue
    sz=$(stat -Lc %s "$f")
    if [ "$sz" -lt $((COPY_MAX_MIB * 1024 * 1024)) ]; then
      cp -p "$f" "$RUN/$f"                    # 小文件：复制（可能被流水线重写）
    else
      ln -s "$PWD/$f" "$RUN/$f"               # GB 级缓存：只读链接
    fi
  done
  echo "[sched] $m -> 卡 $card ($(free_mib "$card") MiB 空闲)  $RUN"
  # 每个 fit 的 GPU/两级默认值：与生产脚本 fit_3090.sh 导出的那几项一致。不设时
  # 流水线会把 SM 稠密化（实测 NDATA=24 就写了 10.2 GB 的 sm_dense.npy），随后稠密
  # GPU 求解因显存预算不足被 fail-closed 拒绝 —— 配置问题，不是求解器问题。
  # 刻意**不**开 PHEASY_GPU_TSQR：流式 TSQR 要求矩阵是高的，小 NDATA 下 SM 是宽的
  # （实测 NDATA=24 为 36864x69487），开了它 OLS 会被正确拒绝（"gpu_tsqr requires
  # a tall matrix"）。生产脚本 fit_3090.sh 也没开。
  # [FIX EXACT-LS] RIDGE/OLS/RFE 的常驻开关只在显式 GPU_RESIDENT 时导出（见头部注释）
  RES_ENV=()
  if [ -n "${GPU_RESIDENT:-}" ]; then
    RES_ENV=(PHEASY_GPU_RIDGE_RESIDENT="$GPU_RESIDENT" PHEASY_GPU_OLS_RESIDENT="$GPU_RESIDENT"
             PHEASY_GPU_RFE_RESIDENT="$GPU_RESIDENT")
  fi
  ( cd "$RUN" && env \
      CUDA_VISIBLE_DEVICES="$card" PHEASY_GPU_DEVICES=0 PHEASY_GPU_NGPU=1 \
      PHEASY_GPU_MODE=required PHEASY_GPU_FALLBACK=0 \
      PHEASY_GPU_LASSO_RESIDENT="${GPU_RESIDENT:-1}" \
      ${RES_ENV[@]+"${RES_ENV[@]}"} \
      PHEASY_GPU_DEBIAS="${GPU_DEBIAS:-1}" \
      PHEASY_LASSO_TWOLEVEL="${TWOLEVEL:-1}" LASSO_TWOLEVEL="${TWOLEVEL:-1}" \
      LASSO_SPARSE="${GPU_RESIDENT:-1}" PHEASY_LASSO_DEBIAS="${DEBIAS:-0}" \
      PHEASY_CV_TOL="${CV_TOL:-1e-3}" PHEASY_CV_MAX_ITER="${CV_MAX_ITER:-400}" \
      PHEASY_N_JOBS="${NCPU:-8}" OPENBLAS_NUM_THREADS="${NCPU:-8}" \
      OMP_NUM_THREADS="${NCPU:-8}" MKL_NUM_THREADS="${NCPU:-8}" \
      bash "$FIT_SCRIPT" FIT_METHOD="$m" LASSO_TWOLEVEL="${TWOLEVEL:-1}" \
          LASSO_SPARSE="${GPU_RESIDENT:-1}" "${PASSTHRU[@]}" > fit.log 2>&1 ) &
  PIDS+=("$!") ; TAGS+=("$m") ; STARTS+=("$(date +%s)")
done

FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then st=OK; else st=FAIL; FAILED=1; fi
  dt=$(( $(date +%s) - ${STARTS[$i]} ))
  printf "[done] %-10s %s  %ss  log=%s\n" "${TAGS[$i]}" "$st" "$dt" "$OUTROOT/${TAGS[$i]}/fit.log"
done

cat > "$OUTROOT/_bin/summary.py" <<'PYEOF'
"""Print the per-method verdict from a fit_3090_parallel.sh run."""
import json, os, sys

root, methods = sys.argv[1], sys.argv[2:]
print("%-10s %-9s %-26s %-16s %-12s %s" % ("method", "accepted", "backend",
      "stop_reason", "rmse", "probe / floor"))
for m in methods:
    path = os.path.join(root, m, "fit_manifest.json")
    if not os.path.exists(path):
        print("%-10s %s" % (m, "no fit_manifest.json (见 fit.log)"))
        continue
    with open(path) as fh:
        d = json.load(fh)
    r = d.get("results", {})
    mt = d.get("metrics", {})
    info = r.get("solver_info") or r.get("regularized_solver_info") or {}
    extra = ""
    if info.get("probe_count"):
        extra = "probe=%s stall_floor=%s @%s" % (info.get("probe_count"),
                  info.get("stall_floor"), info.get("stall_iteration"))
    rmse = mt.get("rmse")
    print("%-10s %-9s %-26s %-16s %-12s %s" % (
        m, r.get("fit_accepted"), r.get("execution_backend"),
        info.get("stop_reason"),
        ("%.6e" % rmse) if isinstance(rmse, (int, float)) else "-", extra))
    env = d.get("environment", {})
    keys = sorted(k for k in env if k.startswith("PHEASY_"))
    notable = [k for k in ("PHEASY_CGLS_PROBE_START", "PHEASY_HOST_FACTOR_CACHE",
                           "PHEASY_GPU_RIDGE_RESIDENT", "PHEASY_GPU_OLS_RESIDENT",
                           "PHEASY_SM_DTYPE") if k in keys]
    print("           manifest 记录 %d 个 PHEASY_* 变量%s" % (
          len(keys), ("（含 " + ", ".join(notable) + "）") if notable else ""))
PYEOF
"$PYTHON" "$OUTROOT/_bin/summary.py" "$OUTROOT" $FIT_METHODS
exit $FAILED
