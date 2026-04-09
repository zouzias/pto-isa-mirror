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


SHORT=r:,v:,n:,m:,k:,
LONG=run-mode:,soc-version:,n-ranks:,gm:,gk:,gn:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

N_RANKS=2
G_M=2048
G_K=2048
G_N=1024

while :
do
    case "$1" in
        (-r | --run-mode )
            RUN_MODE="$2"
            shift 2;;
        (-v | --soc-version )
            SOC_VERSION="$2"
            shift 2;;
        (-n | --n-ranks )
            N_RANKS="$2"
            shift 2;;
        (--gm )
            G_M="$2"
            shift 2;;
        (--gk )
            G_K="$2"
            shift 2;;
        (--gn )
            G_N="$2"
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

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

python3 scripts/gen_data.py --n-ranks "${N_RANKS}" --m "${G_M}" --k "${G_K}" --n "${G_N}" --output-dir ./out

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH

set -euo pipefail

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
      -DG_M=${G_M} -DG_K=${G_K} -DG_N=${G_N} ..
make -j16

export N_RANKS=${N_RANKS}
export ALLGATHER_GEMM_DATA_DIR="${SCRIPT_DIR}/out"
mpirun -n "${N_RANKS}" ./allgather_gemm
