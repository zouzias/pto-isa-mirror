#!/bin/bash
# --------------------------------------------------------------------------------
# Run allgather_gemm_multi with msprof profiling
# 使用华为昇腾性能分析工具 msprof 进行性能测试
# --------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Source CANN environment (user may override via ASCEND_HOME_PATH)
if [ -z "${ASCEND_HOME_PATH}" ]; then
    if [ -f /usr/local/Ascend/latest/set_env.sh ]; then
        source /usr/local/Ascend/latest/set_env.sh
    else
        echo "[ERROR] ASCEND_HOME_PATH not set. Please source your CANN set_env.sh first."
        exit 1
    fi
fi

# Default configuration
SIZE="large"
N_RANKS=2
SOC_VERSION="Ascend910B1"
RUN_MODE="npu"
PROFILE_MODE="application"  # application, op, or timeline
OUTPUT_DIR="./profiling_output"

SHORT=r:,s:,n:,v:,p:,o:,h
LONG=run-mode:,size:,n-ranks:,soc-version:,profile-mode:,output:,help
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

print_usage() {
    echo "Usage: bash run_with_msprof.sh [options]"
    echo ""
    echo "Options:"
    echo "  -r, --run-mode MODE       Run mode: npu or sim (default: npu)"
    echo "  -s, --size SIZE          Matrix size (default: large)"
    echo "                           Available: small, medium, large, xlarge, xxlarge"
    echo "                           KV-cache 7B: kv_7b_2k, kv_7b_4k, kv_7b_8k, kv_7b_16k, kv_7b_32k, kv_7b_64k, kv_7b_128k, kv_7b_256k"
    echo "                           KV-cache 13B: kv_13b_4k, kv_13b_8k, kv_13b_16k, kv_13b_32k, kv_13b_64k, kv_13b_128k"
    echo "                           KV-cache 70B: kv_70b_4k, kv_70b_8k, kv_70b_16k, kv_70b_32k, kv_70b_64k, kv_70b_128k"
    echo "  -n, --n-ranks N          Number of ranks (default: 2)"
    echo "  -v, --soc-version VER    SoC version (default: Ascend910B1)"
    echo "  -p, --profile-mode MODE  Profiling mode: application, op, timeline (default: application)"
    echo "                           - application: 应用级性能分析，捕获完整时间线"
    echo "                           - op: 算子级性能分析，包含流水线可视化"
    echo "                           - timeline: 详细时间线，包含指令级信息"
    echo "  -o, --output DIR         Output directory for profiling data (default: ./profiling_output)"
    echo "  -h, --help               Show this help"
    echo ""
    echo "Examples:"
    echo "  # 基础应用级性能分析"
    echo "  bash run_with_msprof.sh -r npu -v Ascend910B1 -s large -n 2"
    echo ""
    echo "  # 算子级性能分析（带流水线可视化）"
    echo "  bash run_with_msprof.sh -r npu -v Ascend910B1 -s large -n 2 -p op"
    echo ""
    echo "  # 详细时间线分析（指令级）"
    echo "  bash run_with_msprof.sh -r npu -v Ascend910B1 -s kv_7b_16k -n 4 -p timeline"
    echo ""
    echo "  # 指定输出目录"
    echo "  bash run_with_msprof.sh -r npu -v Ascend910B1 -s large -n 2 -o ./my_profiling"
}

while :
do
    case "$1" in
        (-r | --run-mode )
            RUN_MODE="$2"
            shift 2;;
        (-s | --size )
            SIZE="$2"
            shift 2;;
        (-n | --n-ranks )
            N_RANKS="$2"
            shift 2;;
        (-v | --soc-version )
            SOC_VERSION="$2"
            shift 2;;
        (-p | --profile-mode )
            PROFILE_MODE="$2"
            shift 2;;
        (-o | --output )
            OUTPUT_DIR="$2"
            shift 2;;
        (-h | --help )
            print_usage
            exit 0;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            print_usage
            exit 1;;
    esac
done

# Generate input and golden data
echo "[Step 1] Generating input data..."
python ./scripts/gen_data.py ${SIZE} --n-ranks ${N_RANKS}

# Size configurations (same as run.sh)
case "${SIZE}" in
    small)
        G_M=512; G_K=512; G_N=256
        ;;
    medium)
        G_M=1024; G_K=1024; G_N=512
        ;;
    large)
        G_M=2048; G_K=2048; G_N=1024
        ;;
    xlarge)
        G_M=4096; G_K=4096; G_N=2048
        ;;
    gemm_6k)
        G_M=6144; G_K=6144; G_N=6144
        ;;
    kv_7b_2k)
        G_M=2048; G_K=4096; G_N=4096
        ;;
    kv_7b_4k)
        G_M=4096; G_K=4096; G_N=4096
        ;;
    kv_7b_8k)
        G_M=8192; G_K=4096; G_N=4096
        ;;
    kv_7b_16k)
        G_M=16384; G_K=4096; G_N=4096
        ;;
    kv_7b_32k)
        G_M=32768; G_K=4096; G_N=4096
        ;;
    kv_7b_64k)
        G_M=65536; G_K=4096; G_N=4096
        ;;
    kv_7b_128k)
        G_M=131072; G_K=4096; G_N=4096
        ;;
    kv_7b_256k)
        G_M=262144; G_K=4096; G_N=4096
        ;;
    kv_13b_4k)
        G_M=4096; G_K=5120; G_N=5120
        ;;
    kv_13b_8k)
        G_M=8192; G_K=5120; G_N=5120
        ;;
    kv_13b_16k)
        G_M=16384; G_K=5120; G_N=5120
        ;;
    kv_13b_32k)
        G_M=32768; G_K=5120; G_N=5120
        ;;
    kv_13b_64k)
        G_M=65536; G_K=5120; G_N=5120
        ;;
    kv_13b_128k)
        G_M=131072; G_K=5120; G_N=5120
        ;;
    kv_70b_4k)
        G_M=4096; G_K=8192; G_N=8192
        ;;
    kv_70b_8k)
        G_M=8192; G_K=8192; G_N=8192
        ;;
    kv_70b_16k)
        G_M=16384; G_K=8192; G_N=8192
        ;;
    kv_70b_32k)
        G_M=32768; G_K=8192; G_N=8192
        ;;
    kv_70b_64k)
        G_M=65536; G_K=8192; G_N=8192
        ;;
    kv_70b_128k)
        G_M=131072; G_K=8192; G_N=8192
        ;;
    xxlarge)
        G_M=8192; G_K=8192; G_N=4096
        ;;
    *)
        echo "[ERROR] Unsupported size: ${SIZE}"
        echo "Available sizes:"
        echo "  Basic:    small, medium, large, xlarge, xxlarge, gemm_6k"
        echo "  KV-cache 7B:  kv_7b_2k, kv_7b_4k, kv_7b_8k, kv_7b_16k, kv_7b_32k, kv_7b_64k, kv_7b_128k, kv_7b_256k"
        echo "  KV-cache 13B: kv_13b_4k, kv_13b_8k, kv_13b_16k, kv_13b_32k, kv_13b_64k, kv_13b_128k"
        echo "  KV-cache 70B: kv_70b_4k, kv_70b_8k, kv_70b_16k, kv_70b_32k, kv_70b_64k, kv_70b_128k"
        exit 1
        ;;
esac

echo "============================================================"
echo "  AllGather + GEMM 性能分析 (msprof)"
echo "============================================================"
echo "  规模: ${SIZE} (M=${G_M}, K=${G_K}, N=${G_N})"
echo "  Rank 数: ${N_RANKS}"
echo "  SoC 版本: ${SOC_VERSION}"
echo "  运行模式: ${RUN_MODE}"
echo "  性能分析模式: ${PROFILE_MODE}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "============================================================"

# Validate SOC_VERSION
if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi

if [[ "${SOC_VERSION}" =~ ^Ascend910B4-1 ]] && [ "${RUN_MODE}" == "sim" ]; then
    echo "[ERROR] SocVersion: ${SOC_VERSION} can not support sim mode, please use Ascend910B4."
    exit 1
fi

# Build
echo "[Step 2] Building project..."
rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
set -euo pipefail

# Map "run" to "npu" for CMakeLists.txt compatibility
if [ "${RUN_MODE}" == "run" ]; then
    CMAKE_RUN_MODE="npu"
else
    CMAKE_RUN_MODE="${RUN_MODE}"
fi

# Block configuration (auto-detect from SoC variant)
if [[ "${SOC_VERSION}" =~ ^Ascend910B[34] ]] || [[ "${SOC_VERSION}" =~ ^Ascend910_93[67] ]]; then
    COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM:-20}
    COMM_BLOCK_NUM=${COMM_BLOCK_NUM:-40}
else
    COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM:-24}
    COMM_BLOCK_NUM=${COMM_BLOCK_NUM:-48}
fi

# Peak TFLOPS (FP16 Cube)
if [ -z "${PEAK_TFLOPS_FP16:-}" ]; then
    case "${SOC_VERSION}" in
        Ascend910B1|Ascend910_939*)
            PEAK_TFLOPS_FP16=320.0 ;;
        Ascend910B2*|Ascend910_938*)
            PEAK_TFLOPS_FP16=311.0 ;;
        Ascend910B3|Ascend910_937*)
            PEAK_TFLOPS_FP16=259.0 ;;
        Ascend910B4*|Ascend910_936*)
            PEAK_TFLOPS_FP16=216.0 ;;
        *)
            PEAK_TFLOPS_FP16=320.0 ;;
    esac
fi

cmake -DRUN_MODE=${CMAKE_RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
      -DG_M=${G_M} -DG_K=${G_K} -DG_N=${G_N} \
      -DSIZE_NAME=${SIZE} \
      -DCOMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM} \
      -DCOMM_BLOCK_NUM=${COMM_BLOCK_NUM} \
      -DPEAK_TFLOPS_FP16=${PEAK_TFLOPS_FP16} ..
make -j16

# Create output directory with proper permissions
# msopprof requires that output directory and parent directories don't have write permissions for other users
if [[ "${OUTPUT_DIR}" = /* ]]; then
    parent_dir=$(dirname "${OUTPUT_DIR}")
    if [ -d "${parent_dir}" ] && [ "${parent_dir}" != "/" ]; then
        parent_perm=$(stat -c "%a" "${parent_dir}" 2>/dev/null || echo "")
        if [ -n "${parent_perm}" ] && [[ "${parent_perm}" =~ [0-9][37][37] ]]; then
            echo "[WARN] 输出目录的父目录 ${parent_dir} 的权限可能不符合 msopprof 要求"
            echo "[WARN] msopprof 要求输出目录及其父目录不能有组或其他用户的写权限"
            echo "[WARN] 当前父目录权限: ${parent_perm}"
            echo "[WARN] 建议使用相对路径（如 ./ms_output）或修改父目录权限为 755"
            echo ""
            if chmod 755 "${parent_dir}" 2>/dev/null; then
                echo "[INFO] 已尝试修复父目录权限"
            else
                echo "[WARN] 无法修改父目录权限（可能需要 root 权限或目录所有者）"
            fi
        fi
    fi
fi

mkdir -p ${OUTPUT_DIR}
chmod 755 ${OUTPUT_DIR}

# Check if profiling tools exist
echo "[Step 3] Checking profiling tools..."
MSPROF_TOOL=""
MSOPPROF_TOOL=""

if command -v msprof &> /dev/null; then
    MSPROF_TOOL="msprof"
elif [ -n "${ASCEND_HOME_PATH}" ] && [ -f "${ASCEND_HOME_PATH}/tools/profiler/bin/msprof" ]; then
    MSPROF_TOOL="${ASCEND_HOME_PATH}/tools/profiler/bin/msprof"
fi

if command -v msopprof &> /dev/null; then
    MSOPPROF_TOOL="msopprof"
elif [ -n "${ASCEND_TOOLKIT_HOME}" ] && [ -f "${ASCEND_TOOLKIT_HOME}/tools/msopt/bin/msopprof" ]; then
    MSOPPROF_TOOL="${ASCEND_TOOLKIT_HOME}/tools/msopt/bin/msopprof"
fi

if [ "${PROFILE_MODE}" == "application" ]; then
    if [ -z "$MSPROF_TOOL" ]; then
        echo "[ERROR] 未找到 msprof 性能分析工具！"
        echo ""
        echo "请检查："
        echo "  1. CANN 环境是否正确安装"
        echo "  2. 是否已 source 环境脚本（env.sh）"
        echo "  3. ASCEND_HOME_PATH 环境变量是否设置"
        echo ""
        echo "可以尝试手动查找工具："
        echo "  find \${ASCEND_HOME_PATH} -name msprof 2>/dev/null"
        exit 1
    fi
    echo "[INFO] 找到性能分析工具: ${MSPROF_TOOL}"
else
    if [ -z "$MSOPPROF_TOOL" ]; then
        echo "[ERROR] 未找到 msopprof 算子性能分析工具！"
        echo ""
        echo "请检查："
        echo "  1. ascend-toolkit 是否已正确安装"
        echo "  2. 是否已 source 环境脚本（set_env.sh）"
        echo ""
        echo "可以尝试手动查找工具："
        echo "  find \${ASCEND_HOME_PATH} -name msopprof 2>/dev/null"
        exit 1
    fi
    echo "[INFO] 找到算子性能分析工具: ${MSOPPROF_TOOL}"
fi

# Run with msprof based on profile mode
echo "[Step 4] Running with msprof (mode: ${PROFILE_MODE})..."
export N_RANKS=${N_RANKS}
TIMEOUT=${TIMEOUT:-300}

case "${PROFILE_MODE}" in
    application)
        echo "[INFO] 应用级性能分析模式：捕获完整时间线和各组件性能指标"
        timeout ${TIMEOUT}s ${MSPROF_TOOL} --output=${OUTPUT_DIR} \
               --task-time=on \
               --runtime-api=on \
               --ai-core=on \
               --aic-metrics=PipeUtilization \
               --aicpu=on \
               --hccl=on \
               --msproftx=on \
               ./allgather_gemm
        ;;
    op)
        echo "[INFO] 算子级性能分析模式：包含流水线可视化，分析算子执行细节"
        timeout ${TIMEOUT}s ${MSOPPROF_TOOL} --application=./allgather_gemm \
                  --output=${OUTPUT_DIR} \
                  --aic-metrics=Default \
                  --launch-count=50
        ;;
    timeline)
        echo "[INFO] 详细时间线分析模式：包含指令级信息，深入分析性能瓶颈"
        timeout ${TIMEOUT}s ${MSOPPROF_TOOL} --application=./allgather_gemm \
                  --output=${OUTPUT_DIR} \
                  --aic-metrics=Default \
                  --launch-count=10
        ;;
    *)
        echo "[ERROR] Unknown profile mode: ${PROFILE_MODE}"
        echo "Available modes: application, op, timeline"
        exit 1;;
esac

echo ""
echo "============================================================"
echo "  性能分析完成！"
echo "============================================================"
echo "  输出目录: ${OUTPUT_DIR}"
echo ""
echo "  查看结果的方法："
echo "  1. 使用 MindStudio Insight 打开性能分析数据"
echo "  2. 或者解析生成的数据文件"
echo ""
echo "  生成的文件："
ls -la ${OUTPUT_DIR}/
echo ""
echo "  性能分析模式说明："
echo "  - application: 应用级分析，适合查看整体性能和时间线"
echo "  - op: 算子级分析，适合查看算子执行细节和流水线"
echo "  - timeline: 时间线分析，适合深入分析性能瓶颈"
echo "============================================================"
