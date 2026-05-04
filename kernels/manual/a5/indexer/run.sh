#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------


SHORT=r:,v:,
LONG=run-mode:,soc-version:,
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

if [[ "${SOC_VERSION}" =~ ^Ascend910B4-1 ]] && [ "${RUN_MODE}" == "sim" ]; then
    echo "[ERROR] SocVersion: ${SOC_VERSION} can not support sim mode, please use Ascend910B4."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
INDEXER_TEST_N="${INDEXER_TEST_N:-131072}"
INDEXER_TOPK="${INDEXER_TOPK:-2048}"
PYTHON_BIN="python3"
if [[ -x "${SCRIPT_DIR}/../../../../.venv-indexer-sim/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/../../../../.venv-indexer-sim/bin/python"
fi
if ! "${PYTHON_BIN}" -c "import ml_dtypes" 2>/dev/null; then
    echo "[ERROR] Install Python dependency in selected python (${PYTHON_BIN}): pip install ml_dtypes numpy"
    exit 1
fi
INDEXER_TEST_N="${INDEXER_TEST_N}" INDEXER_TOPK="${INDEXER_TOPK}" "${PYTHON_BIN}" scripts/gen_data.py

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
set -euo pipefail

cmake  -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} -DINDEXER_TEST_N=${INDEXER_TEST_N} -DINDEXER_TOPK=${INDEXER_TOPK} ..
make -j16

./indexer_mxfp8