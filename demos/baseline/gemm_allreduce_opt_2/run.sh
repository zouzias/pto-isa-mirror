#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# ============================================================================
# GEMM AllReduce Demo - Build and Run Script
#
# This script builds and runs the dual-stream GEMM + AllReduce demo.
# Architecture:
#   - Compute kernel: GEMM computation on Cube cores, signals via queue
#   - Comm kernel: Polls compute queues, TPUT to remote, TREDUCE when ready
#   - Both kernels run in parallel on separate streams
# ============================================================================

# Source CANN environment
source /usr/local/Ascend/cann-8.5.0/set_env.sh

# Source shmem environment (use qifeng's shmem for compatibility)
source /home/ntlab/qifeng/pypto/third_party_path/shmem/install/set_env.sh

# Set CMAKE_PREFIX_PATH for gtest (bisheng compiled)
export CMAKE_PREFIX_PATH="$HOME/.local/lib64/cmake:$CMAKE_PREFIX_PATH"

SHORT=r:,v:,n:,d:
LONG=run-mode:,soc-version:,nranks:,ndevices:,
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
        (-n | --nranks )
            NRANKS="$2"
            shift 2;;
        (-d | --ndevices )
            NDEVICES="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

: "${NRANKS:=2}"
: "${NDEVICES:=2}"
: "${RUN_MODE:=npu}"
: "${SOC_VERSION:=Ascend910B1}"

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi

if [[ "${SOC_VERSION}" =~ ^Ascend910B4-1 ]] && [ "${RUN_MODE}" == "sim" ]; then
    echo "[ERROR] SocVersion: ${SOC_VERSION} can not support sim mode, please use Ascend910B4."
    exit 1
fi

echo "=== GEMM AllReduce Demo Configuration ==="
echo "  RUN_MODE: ${RUN_MODE}"
echo "  SOC_VERSION: ${SOC_VERSION}"
echo "  NRANKS: ${NRANKS}"
echo "  NDEVICES: ${NDEVICES}"
echo "=========================================="

export NUM_RANKS="${NRANKS}"

# Generate input data
if [ -f "./scripts/gen_data.py" ]; then
    python ./scripts/gen_data.py --nranks ${NRANKS}
else
    echo "[WARNING] gen_data.py not found, skipping data generation"
fi

rm -rf build
mkdir build
cd build

# Set up library paths
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}

# Add conda environment libstdc++ to LD_LIBRARY_PATH if available
# This fixes GLIBCXX version issues when using conda's newer libstdc++
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${CONDA_PREFIX}/aarch64-conda-linux-gnu/lib:${LD_LIBRARY_PATH}
elif [ -d "${HOME}/miniconda3/envs/pypto_haoran/lib" ]; then
    export LD_LIBRARY_PATH=${HOME}/miniconda3/envs/pypto_haoran/lib:${HOME}/miniconda3/envs/pypto_haoran/aarch64-conda-linux-gnu/lib:${LD_LIBRARY_PATH}
fi

set -euo pipefail

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} ..
make -j16

echo ""
echo "=== Running GEMM AllReduce Demo ==="
echo "  Ranks: ${NRANKS}"
echo "  Devices: ${NDEVICES}"
echo "==================================="

# Pass nranks to executable via command line or environment variable
# The executable supports --nranks and --first-device arguments
# Default first-device is 0, but can be overridden via environment
FIRST_DEVICE="${FIRST_DEVICE:-0}"

./gemm_allreduce --nranks ${NRANKS} --first-device ${FIRST_DEVICE}
