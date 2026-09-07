#!/bin/bash
#SBATCH --partition=cpu192
#SBATCH --job-name=shengbte-aocl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8        # 每节点 8 MPI(=8 NUMA 域)
#SBATCH --cpus-per-task=24         # 每 MPI 24 OMP(=每个 NUMA 域 24 核)
#SBATCH --exclusive
#SBATCH --output=shengbte.out
#SBATCH --error=shengbte.err
#SBATCH --qos=premium

module purge
module load gcc/14.1
module load openmpi/4.0.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate atomate2_p_a

cd $SLURM_SUBMIT_DIR

# ---- 输入文件检查 ----
if [ ! -f FORCE_CONSTANTS_2ND ]; then
    if [ -f FORCE_CONSTANTS ]; then
        ln -sf FORCE_CONSTANTS FORCE_CONSTANTS_2ND
    else
        echo "❌ 找不到 FORCE_CONSTANTS / FORCE_CONSTANTS_2ND" >&2
        exit 1
    fi
fi
if [ ! -f FORCE_CONSTANTS_3RD ]; then
    echo "❌ 找不到 FORCE_CONSTANTS_3RD" >&2
    exit 1
fi
if [ ! -f CONTROL ]; then
    echo "❌ 找不到 CONTROL" >&2
    exit 1
fi

# ---- OpenMP / BLIS ----
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK    # = 24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NESTED=FALSE
export BLIS_NUM_THREADS=$OMP_NUM_THREADS
export AOCL_ENABLE_INSTRUCTIONS=AVX512

# ---- 库路径 ----
export LD_LIBRARY_PATH=/public/home/wangchao/software/aocl-gcc/5.0.0/gcc/lib_LP64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
unset MKL_NUM_THREADS MKL_DEBUG_CPU_TYPE I_MPI_PMI_LIBRARY
ulimit -s unlimited

echo "开始: $(date)"
echo "节点: $SLURM_JOB_NODELIST"
echo "MPI 进程: $SLURM_NTASKS  每进程 OMP 线程: $OMP_NUM_THREADS"

mpirun -n $SLURM_NTASKS \
    --map-by numa:PE=$OMP_NUM_THREADS \
    --bind-to core \
    --report-bindings \
    --mca pml ucx --mca osc ucx \
    --mca btl ^openib,tcp \
    -x UCX_TLS=rc,sm,self \
    -x UCX_NET_DEVICES=mlx5_0:1 \
    -x UCX_LOG_LEVEL=error \
    -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
    -x BLIS_NUM_THREADS -x AOCL_ENABLE_INSTRUCTIONS \
    -x LD_LIBRARY_PATH \
    /public/home/wangchao/software/sousaw-shengbte-aocl/ShengBTE \
    > shengbte.log 2>&1

echo "结束: $(date)"

if [ -s BTE.KappaTensorVsT_RTA ]; then
    echo "✅ RTA 完成"
    tail -1 BTE.KappaTensorVsT_RTA
else
    echo "❌ 失败"
    tail -30 shengbte.log
fi