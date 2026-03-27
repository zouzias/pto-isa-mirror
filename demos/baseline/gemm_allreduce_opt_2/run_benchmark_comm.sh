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
# Benchmark Single Comm Core - Run Script
# ============================================================================

# Source CANN environment
source /usr/local/Ascend/cann-8.5.0/set_env.sh

# Source shmem environment
source /home/ntlab/qifeng/pypto/third_party_path/shmem/install/set_env.sh

# Add conda environment libstdc++ to LD_LIBRARY_PATH if available
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${CONDA_PREFIX}/aarch64-conda-linux-gnu/lib:${LD_LIBRARY_PATH}
elif [ -d "${HOME}/miniconda3/envs/pypto_haoran/lib" ]; then
    export LD_LIBRARY_PATH=${HOME}/miniconda3/envs/pypto_haoran/lib:${HOME}/miniconda3/envs/pypto_haoran/aarch64-conda-linux-gnu/lib:${LD_LIBRARY_PATH}
fi

# Default values
NRANKS=8
FIRST_DEVICE=0
NUM_TILES=100

# Parse arguments
SHORT=n:,d:,t:
LONG=nranks:,first-device:,num-tiles:
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :
do
    case "$1" in
        (-n | --nranks )
            NRANKS="$2"
            shift 2;;
        (-d | --first-device )
            FIRST_DEVICE="$2"
            shift 2;;
        (-t | --num-tiles )
            NUM_TILES="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

echo "=== Benchmark Single Comm Core ==="
echo "  Ranks: ${NRANKS}"
echo "  First Device: ${FIRST_DEVICE}"
echo "  Num Tiles: ${NUM_TILES}"
echo "==================================="

# Run from build directory
cd build
./benchmark_single_comm_core --nranks ${NRANKS} --first-device ${FIRST_DEVICE} --num-tiles ${NUM_TILES}
