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

KERNEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SHORT=r:,v:,C:,a:,p
LONG=run-mode:,soc-version:,compiler:,cases:,profile
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
PROFILE_MODE=0
while :
do
    case "$1" in
        (-r | --run-mode )
            RUN_MODE="$2"
            shift 2;;
        (-v | --soc-version )
            SOC_VERSION="$2"
            shift 2;;
        (-C | --compiler )
            CMAKE_COMPILER="$2"
            shift 2;;
        (-p | --profile )
            PROFILE_MODE=1
            shift 1;;
        (-a | --cases )
            CASES_RAW="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done


: "${CMAKE_COMPILER:=bisheng}"

source "${KERNEL_DIR}/../common.sh"

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

GEN_CASE_ARGS=()
if [[ -n "${CASES_RAW:-}" ]]; then
    GEN_CASE_ARGS+=(--cases "${CASES_RAW}")
fi

python3 ./scripts/generate_cases.py "${GEN_CASE_ARGS[@]}"
python3 ./scripts/gen_data.py

cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
set -euo pipefail

cmake  -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} -DCMAKE_COMPILER=${CMAKE_COMPILER} ..
make -j16

run_bin ./topk
