#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Usage:
#   bash run.sh <device_list>                                      # functional test (all shapes in CSV)
#   bash run.sh <device_list> --perf                               # perf test (all shapes in CSV)
#   bash run.sh <device_list> --perf -M 2048 -K 2048 -N 1024      # perf test (single shape)
#   bash run.sh <device_list> -v Ascend910B1                       # specify SoC version
#
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
CSV_FILE="${SCRIPT_DIR}/scripts/test_shapes.csv"

DEVICE_LIST_ARG="$1"
shift

PERF_FLAG=""
SINGLE_M=""
SINGLE_K=""
SINGLE_N=""
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
RUN_MODE="${RUN_MODE:-npu}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --perf)
            PERF_FLAG="--perf"
            shift ;;
        -M)
            SINGLE_M="$2"; shift 2 ;;
        -K)
            SINGLE_K="$2"; shift 2 ;;
        -N)
            SINGLE_N="$2"; shift 2 ;;
        -v|--soc-version)
            SOC_VERSION="$2"; shift 2 ;;
        -r|--run-mode)
            RUN_MODE="$2"; shift 2 ;;
        *)
            echo "[WARN] Unknown option: $1"; shift ;;
    esac
done

IFS=',' read -ra DEVICE_ID_LIST <<< "$DEVICE_LIST_ARG"
PE_SIZE=${#DEVICE_ID_LIST[@]}
if [ $PE_SIZE -gt 8 ]; then
    echo "[ERROR] PE size must be <= 8"
    exit 1
fi

# Source CANN environment — search common install paths
if [ -z "${ASCEND_HOME_PATH}" ]; then
    CANN_ENV_PATHS=(
        /usr/local/Ascend/latest/set_env.sh
        /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
    )
    # Also search sibling directories of PROJECT_ROOT (e.g. /mnt/data/.../Ascend-*/cann-*/set_env.sh)
    for d in $(dirname "${PROJECT_ROOT}")/Ascend-*/cann-*/set_env.sh \
             $(dirname "${PROJECT_ROOT}")/Ascend-*/ascend-toolkit/set_env.sh; do
        [ -f "$d" ] && CANN_ENV_PATHS+=("$d")
    done
    SOURCED=0
    for p in "${CANN_ENV_PATHS[@]}"; do
        if [ -f "$p" ]; then
            echo "[INFO] Sourcing CANN env: $p"
            source "$p"
            SOURCED=1
            break
        fi
    done
    if [ ${SOURCED} -eq 0 ]; then
        echo "[ERROR] ASCEND_HOME_PATH not set and no set_env.sh found."
        echo "        Please source your CANN set_env.sh first."
        exit 1
    fi
fi
echo "[INFO] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

cd ${SCRIPT_DIR}

DATA_DIR=$(realpath ./out)
mkdir -p ${DATA_DIR}

# Tile alignment constants
BASE_M=128; BASE_N=256

# Block configuration based on SoC
if [[ "${SOC_VERSION}" =~ ^Ascend910B[34] ]] || [[ "${SOC_VERSION}" =~ ^Ascend910_93[67] ]]; then
    COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM:-20}
    COMM_BLOCK_NUM=${COMM_BLOCK_NUM:-40}
else
    COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM:-24}
    COMM_BLOCK_NUM=${COMM_BLOCK_NUM:-48}
fi

# Peak TFLOPS based on SoC variant (FP16 Cube)
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

# Find MPI — prefer MPICH, search common install paths
MPI_BIN=""
MPI_SEARCH_PATHS=(
    /usr/local/mpich/bin/mpirun
    /home/mpich/bin/mpirun
    /usr/local/bin/mpirun
    /usr/bin/mpirun
)
for p in "${MPI_SEARCH_PATHS[@]}"; do
    if [ -f "$p" ]; then
        MPI_BIN="$p"
        MPI_LIB_DIR="$(dirname "$(dirname "$p")")/lib"
        export MPI_LIB_PATH="${MPI_LIB_PATH:-${MPI_LIB_DIR}/libmpi.so}"
        break
    fi
done
if [ -z "$MPI_BIN" ] && command -v mpirun &>/dev/null; then
    MPI_BIN="mpirun"
fi
if [ -z "$MPI_BIN" ]; then
    echo "[ERROR] mpirun not found. Searched: ${MPI_SEARCH_PATHS[*]}"
    echo "        Please install MPICH or set PATH."
    exit 1
fi
echo "[INFO] Using MPI: ${MPI_BIN}"

run_one_shape() {
    local M=$1 K=$2 N=$3
    echo ""
    echo "================================================================"
    echo "Processing test case: M=${M}, K=${K}, N=${N}, PE_SIZE=${PE_SIZE}"
    echo "================================================================"

    # Save original dimensions
    local ORIG_M=${M}; local ORIG_K=${K}; local ORIG_N=${N}

    # Pad dimensions for tile alignment
    local M_ALIGN=$(( BASE_M * PE_SIZE ))
    M=$(( ((ORIG_M + M_ALIGN - 1) / M_ALIGN) * M_ALIGN ))
    K=$(( ((ORIG_K + BASE_N - 1) / BASE_N) * BASE_N ))
    N=$(( ((ORIG_N + BASE_N - 1) / BASE_N) * BASE_N ))

    if [ ${M} -ne ${ORIG_M} ] || [ ${K} -ne ${ORIG_K} ] || [ ${N} -ne ${ORIG_N} ]; then
        echo "[INFO] Padded dimensions: M=${ORIG_M}->${M}, K=${ORIG_K}->${K}, N=${ORIG_N}->${N}"
    fi

    # Generate input data
    rm -rf ${DATA_DIR}/*.bin
    python3 ${SCRIPT_DIR}/scripts/gen_data.py \
        --n-ranks ${PE_SIZE} \
        --m ${ORIG_M} --k ${ORIG_K} --n ${ORIG_N} \
        --padded-m ${M} --padded-k ${K} --padded-n ${N} \
        --output-dir ${DATA_DIR}

    # Auto-compute HCCL_BUFFSIZE
    local SHMEM_INPUT_MB=$(( (M * K * 2 + 1048575) / 1048576 ))
    local REQUIRED_MB=$(( SHMEM_INPUT_MB + 64 ))
    REQUIRED_MB=$(( ((REQUIRED_MB + 255) / 256) * 256 ))
    if [ -z "${HCCL_BUFFSIZE:-}" ] || [ "${HCCL_BUFFSIZE:-0}" -lt "${REQUIRED_MB}" ]; then
        export HCCL_BUFFSIZE=${REQUIRED_MB}
    fi
    echo "[INFO] HCCL_BUFFSIZE=${HCCL_BUFFSIZE} MB"

    # Build
    rm -rf build
    mkdir build
    cd build

    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${ASCEND_HOME_PATH}/lib64:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}

    local CMAKE_RUN_MODE="${RUN_MODE}"
    if [ "${RUN_MODE}" == "run" ]; then
        CMAKE_RUN_MODE="npu"
    fi

    unset CXXFLAGS CFLAGS LDFLAGS

    CC=bisheng CXX=bisheng cmake -DRUN_MODE=${CMAKE_RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
          -DG_M=${M} -DG_K=${K} -DG_N=${N} \
          -DORIG_M=${ORIG_M} -DORIG_K=${ORIG_K} -DORIG_N=${ORIG_N} \
          -DCOMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM} \
          -DCOMM_BLOCK_NUM=${COMM_BLOCK_NUM} \
          -DPEAK_TFLOPS_FP16=${PEAK_TFLOPS_FP16} ..
    if ! make -j16; then
        echo "[ERROR] Build failed!"
        exit 1
    fi

    cd ${SCRIPT_DIR}

    local EXEC_BIN=$(realpath ./build/allgather_gemm)

    # Launch via MPI
    export N_RANKS=${PE_SIZE}
    local TIMEOUT=${TIMEOUT:-120}

    if [ -n "${PERF_FLAG}" ]; then
        export ALLGATHER_GEMM_PERF_MODE=1
    else
        unset ALLGATHER_GEMM_PERF_MODE
    fi
    export ALLGATHER_GEMM_DATA_DIR="${DATA_DIR}"

    echo "[INFO] Launching: ${MPI_BIN} -n ${PE_SIZE} ${EXEC_BIN}"
    timeout ${TIMEOUT}s ${MPI_BIN} -n ${PE_SIZE} ${EXEC_BIN}
    local RET=$?

    if [ ${RET} -ne 0 ]; then
        echo "[ERROR] Test failed with exit code ${RET}"
        exit 1
    fi

    if [ -z "${PERF_FLAG}" ]; then
        echo "[INFO] Verifying results..."
        python3 ${SCRIPT_DIR}/scripts/verify_result.py \
            ${DATA_DIR}/output_rank0.bin ${DATA_DIR}/golden.bin \
            ${M} ${N} ${ORIG_M} ${ORIG_N}
        if [ $? -ne 0 ]; then
            echo "[ERROR] Verification failed!"
            exit 1
        fi
    fi
}

echo "[INFO] SoC: ${SOC_VERSION}, RunMode: ${RUN_MODE}, PE_SIZE: ${PE_SIZE}"
echo "[INFO] PEAK_TFLOPS_FP16=${PEAK_TFLOPS_FP16}, COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM}, COMM_BLOCK_NUM=${COMM_BLOCK_NUM}"

if [ -n "${SINGLE_M}" ] && [ -n "${SINGLE_K}" ] && [ -n "${SINGLE_N}" ]; then
    run_one_shape "${SINGLE_M}" "${SINGLE_K}" "${SINGLE_N}"
else
    tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N; do
        run_one_shape "$M" "$K" "$N"
    done
fi

cd ${CURRENT_DIR}
