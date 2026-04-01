#!/bin/bash

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

# Default size configuration
SIZE="large"
N_RANKS=2  # Default to 2 ranks

SHORT=r:,v:,s:,n:,
LONG=run-mode:,soc-version:,size:,n-ranks:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :
do
    case "$1" in
        (-r | --run-mode )
            RUN_MODE="$2"
            shift 2;;
        (-v | --soc-version )
            SOC_VERSION="$2"
            shift 2;;
        (-s | --size )
            SIZE="$2"
            shift 2;;
        (-n | --n-ranks )
            N_RANKS="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

# Size configurations (same as gemm_allgather_multi)
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
    5416_6144_1408)
        G_M=5416; G_K=6144; G_N=1408
        ;;
    *)
        echo "[ERROR] Unsupported size: ${SIZE}"
        echo "Available sizes:"
        echo "  Basic:    small, medium, large, xlarge, xxlarge"
        echo "  KV-cache 7B:  kv_7b_2k, kv_7b_4k, kv_7b_8k, kv_7b_16k, kv_7b_32k, kv_7b_64k, kv_7b_128k, kv_7b_256k"
        echo "  KV-cache 13B: kv_13b_4k, kv_13b_8k, kv_13b_16k, kv_13b_32k, kv_13b_64k, kv_13b_128k"
        echo "  KV-cache 70B: kv_70b_4k, kv_70b_8k, kv_70b_16k, kv_70b_32k, kv_70b_64k, kv_70b_128k"
        echo "  Custom:       5416_6144_1408"
        exit 1
        ;;
esac

# Save original (unpadded) dimensions
ORIG_M=${G_M}; ORIG_K=${G_K}; ORIG_N=${G_N}

# Pad dimensions to tile-aligned sizes for kernel execution:
#   M must satisfy: M % BASE_M == 0, M % N_RANKS == 0, (M/N_RANKS) % BASE_M == 0
#   This requires M to be a multiple of BASE_M * N_RANKS.
#   K must be divisible by BASE_N (256) — used as k_chunk size
#   N must be divisible by BASE_N (256) — used as n_tile size
BASE_M=128; BASE_N=256

M_ALIGN=$(( BASE_M * N_RANKS ))

G_M=$(( ((ORIG_M + M_ALIGN - 1) / M_ALIGN) * M_ALIGN ))
G_K=$(( ((ORIG_K + BASE_N - 1) / BASE_N) * BASE_N ))
G_N=$(( ((ORIG_N + BASE_N - 1) / BASE_N) * BASE_N ))

if [ ${G_M} -ne ${ORIG_M} ] || [ ${G_K} -ne ${ORIG_K} ] || [ ${G_N} -ne ${ORIG_N} ]; then
    echo "[INFO] Padded dimensions for tile alignment: M=${ORIG_M}->${G_M}, K=${ORIG_K}->${G_K}, N=${ORIG_N}->${G_N}"
fi

# Generate input and golden data with padded dimensions
# Remove stale data if dimensions changed (padded sizes affect binary layout)
rm -rf input/${SIZE} output/${SIZE}
python ./scripts/gen_data.py ${SIZE} --n-ranks ${N_RANKS} --padded-m ${G_M} --padded-k ${G_K} --padded-n ${G_N}

# Auto-compute HCCL_BUFFSIZE based on shared-window memory requirement.
# AllGather puts the full M×K matrix (fp16) into the HCCL window, plus a small
# TileFlagMatrix.  Default HCCL window is 200 MB which is too small for large
# sizes.  We round up to the nearest 256 MB with some headroom.
SHMEM_INPUT_MB=$(( (G_M * G_K * 2 + 1048575) / 1048576 ))
REQUIRED_MB=$(( SHMEM_INPUT_MB + 64 ))          # headroom for tile flags + alignment
REQUIRED_MB=$(( ((REQUIRED_MB + 255) / 256) * 256 ))  # round up to 256 MB
if [ -z "${HCCL_BUFFSIZE:-}" ] || [ "${HCCL_BUFFSIZE:-0}" -lt "${REQUIRED_MB}" ]; then
    export HCCL_BUFFSIZE=${REQUIRED_MB}
fi
echo "[INFO] HCCL_BUFFSIZE=${HCCL_BUFFSIZE} MB (shmem_input=${SHMEM_INPUT_MB} MB)"

echo "[INFO] Using size: ${SIZE} (orig M=${ORIG_M}, K=${ORIG_K}, N=${ORIG_N}) (padded M=${G_M}, K=${G_K}, N=${G_N}), n_ranks=${N_RANKS}"

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi

if [[ "${SOC_VERSION}" =~ ^Ascend910B4-1 ]] && [ "${RUN_MODE}" == "sim" ]; then
    echo "[ERROR] SocVersion: ${SOC_VERSION} can not support sim mode, please use Ascend910B4."
    exit 1
fi

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

# Block configuration (AIC/AIV are INDEPENDENT resources):
#   - Comm kernel (vector/AIV) → AllGather producer, uses COMM_BLOCK_NUM blocks
#   - Compute kernel (cube/AIC) → GEMM consumer, uses COMPUTE_BLOCK_NUM blocks
#   - In pipelined mode, AIV blocks run TPUT while AIC blocks run TMATMUL
#     achieving true overlap.
#
# AICORE Architecture (Ascend 910B):
#   - Each AICORE contains: 1 AIC + 2 AIV (independently schedulable)
#   - 24 AICOREs → 24 AIC + 48 AIV available
#   - AIC and AIV do NOT compete for resources
#
# Maximum parallel configuration:
#   - COMPUTE_BLOCK_NUM = 24 (all AIC units)
#   - COMM_BLOCK_NUM = 48 (all AIV units)
#
# TPUT/TGET support atomic operations for lock-free signaling.
COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM:-24}
COMM_BLOCK_NUM=${COMM_BLOCK_NUM:-48}

# Clear conda-injected flags that conflict with bisheng compiler
unset CXXFLAGS CFLAGS LDFLAGS

CC=bisheng CXX=bisheng cmake -DRUN_MODE=${CMAKE_RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
      -DG_M=${G_M} -DG_K=${G_K} -DG_N=${G_N} \
      -DORIG_M=${ORIG_M} -DORIG_K=${ORIG_K} -DORIG_N=${ORIG_N} \
      -DSIZE_NAME=${SIZE} \
      -DCOMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM} \
      -DCOMM_BLOCK_NUM=${COMM_BLOCK_NUM} ..
make -j16

# Export N_RANKS environment variable for the executable
export N_RANKS=${N_RANKS}
# Set default timeout to 120 seconds (HCCL init may take longer)
TIMEOUT=${TIMEOUT:-120}

# Find MPI
MPI_BIN=""
if command -v mpirun &>/dev/null; then
    MPI_BIN="mpirun"
elif [ -f /usr/local/mpich/bin/mpirun ]; then
    MPI_BIN="/usr/local/mpich/bin/mpirun"
else
    echo "[ERROR] mpirun not found. Please install MPI or set PATH."
    exit 1
fi

echo "[INFO] Launching with: ${MPI_BIN} -n ${N_RANKS} ./allgather_gemm"
timeout ${TIMEOUT}s ${MPI_BIN} -n ${N_RANKS} ./allgather_gemm
