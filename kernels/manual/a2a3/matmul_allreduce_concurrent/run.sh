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

# MATMUL + ALLREDUCE Concurrent Demo Runner
# This demo demonstrates concurrentping GEMM computation with AllReduce communication

SHORT=r:,v:,d,h
LONG=run-mode:,soc-version:,debug,help
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

RUN_MODE="npu"
SOC_VERSION="Ascend910B4"
DEBUG_MODE="OFF"

while :
do
    case "$1" in
        (-r | --run-mode )
            RUN_MODE="$2"
            shift 2;;
        (-v | --soc-version )
            SOC_VERSION="$2"
            shift 2;;
        (-d | --debug )
            DEBUG_MODE="ON"
            shift;;
        (-h | --help )
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -r, --run-mode MODE      Run mode: npu or sim (default: npu)"
            echo "  -v, --soc-version VER    SoC version (default: Ascend910B4)"
            echo "  -d, --debug              Enable debug mode (enables cce::printf)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -r npu -v Ascend910B4"
            echo "  $0 -r npu -d              # Run with debug enabled"
            exit 0;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi

# Check required environment variables
if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set. Please run set_env.sh first."
    exit 1
fi

if [ -z "${SHMEM_HOME_PATH}" ]; then
    echo "[ERROR] SHMEM_HOME_PATH is not set. Please set SHMEM_HOME_PATH environment variable."
    exit 1
fi

echo "============================================================"
echo "  MATMUL + ALLREDUCE Concurrent Demo"
echo "============================================================"
echo "  Run Mode:     ${RUN_MODE}"
echo "  SoC Version:  ${SOC_VERSION}"
echo "  Debug Mode:   ${DEBUG_MODE}"
echo "  ASCEND_HOME:  ${ASCEND_HOME_PATH}"
echo "  SHMEM_HOME:   ${SHMEM_HOME_PATH}"
echo "============================================================"

# Clean and create build directory
rm -rf build
mkdir -p build
cd build

# Set up library paths
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${SHMEM_HOME_PATH}/shmem/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${SHMEM_HOME_PATH}/memfabric_hybrid/lib:$LD_LIBRARY_PATH

if [ "${RUN_MODE}" == "sim" ]; then
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

set -euo pipefail

# Build
echo ""
echo "[INFO] Building..."
cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} -DDEBUG_MODE=${DEBUG_MODE} ..
make -j16

# Run
echo ""
echo "[INFO] Running matmul_allreduce_concurrent..."
./matmul_allreduce_concurrent "$@"
