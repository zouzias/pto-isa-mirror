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

# P2P Communication Demo — Build and Run Script
#
# Prerequisites:
#   1. CANN toolkit installed and set_env.sh sourced (ASCEND_HOME_PATH set)
#   2. MPI (mpich) available in PATH
#   3. At least 2 NPU devices available
#
# Usage:
#   ./run.sh                         # default SoC (ascend910b1)
#   ./run.sh Ascend910_9599          # for A5 devices
#   SOC_VERSION=Ascend910_9599 ./run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOC_VERSION="${1:-${SOC_VERSION:-ascend910b1}}"

echo "=== P2P Comm Demo: Building (SOC_VERSION=${SOC_VERSION}) ==="

cd "${SCRIPT_DIR}"
rm -rf build
mkdir -p build && cd build
cmake .. -DSOC_VERSION="${SOC_VERSION}" 2>&1
make -j"$(nproc)" 2>&1
cd "${SCRIPT_DIR}"

echo ""
echo "=== P2P Comm Demo: Running with 2 ranks ==="
echo ""

mpirun -n 2 ./build/bin/p2p_demo
