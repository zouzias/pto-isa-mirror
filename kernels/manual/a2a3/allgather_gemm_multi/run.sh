#!/bin/bash

# Source environment scripts
source /home/ntlab/qifeng/pypto/env.sh
source /home/ntlab/qifeng/pypto/third_party_path/shmem/install/set_env.sh

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

# Generate input and golden data (same format as gemm_allgather)
python ./scripts/gen_data.py ${SIZE} --n-ranks ${N_RANKS}

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
    *)
        echo "[ERROR] Unsupported size: ${SIZE}"
        echo "Available sizes:"
        echo "  Basic:    small, medium, large, xlarge, xxlarge"
        echo "  KV-cache 7B:  kv_7b_2k, kv_7b_4k, kv_7b_8k, kv_7b_16k, kv_7b_32k, kv_7b_64k, kv_7b_128k, kv_7b_256k"
        echo "  KV-cache 13B: kv_13b_4k, kv_13b_8k, kv_13b_16k, kv_13b_32k, kv_13b_64k, kv_13b_128k"
        echo "  KV-cache 70B: kv_70b_4k, kv_70b_8k, kv_70b_16k, kv_70b_32k, kv_70b_64k, kv_70b_128k"
        exit 1
        ;;
esac

echo "[INFO] Using size: ${SIZE} (M=${G_M}, K=${G_K}, N=${G_N}), n_ranks=${N_RANKS}"

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

cmake -DRUN_MODE=${CMAKE_RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
      -DG_M=${G_M} -DG_K=${G_K} -DG_N=${G_N} \
      -DSIZE_NAME=${SIZE} \
      -DCOMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM} \
      -DCOMM_BLOCK_NUM=${COMM_BLOCK_NUM} ..
make -j16

# Export N_RANKS environment variable for the executable
export N_RANKS=${N_RANKS}
# Set default timeout to 60 seconds
TIMEOUT=${TIMEOUT:-60}
timeout ${TIMEOUT}s ./allgather_gemm
