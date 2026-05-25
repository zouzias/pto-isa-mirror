#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${ASCEND_CANN_PATH:=/usr/local/Ascend/cann-8.5.0/set_env.sh}"
if [ ! -f "${ASCEND_CANN_PATH}" ]; then
    ASCEND_CANN_PATH=$(ls -1d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort -V | tail -1)
fi
if [ -z "${ASCEND_CANN_PATH}" ] || [ ! -f "${ASCEND_CANN_PATH}" ]; then
    echo "[ERROR] Cannot find CANN set_env.sh. Set ASCEND_CANN_PATH to <cann-install>/set_env.sh"
    exit 1
fi

ORIG_ARGS=("$@")
set +e +u
set --
source "${ASCEND_CANN_PATH}"
SET_ENV_STATUS=$?
set -- "${ORIG_ARGS[@]}"
set -euo pipefail
if [ ${SET_ENV_STATUS} -ne 0 ]; then
    echo "[ERROR] source ${ASCEND_CANN_PATH} failed"
    exit ${SET_ENV_STATUS}
fi

export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:${PATH}
export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:${LD_LIBRARY_PATH:-}
export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so

RUN_MODE=npu
SOC_VERSION=Ascend910B1
PES=2
M=64
K=7168
TOPK=8
EXPERT_PER_PE=2
SKIP_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--run-mode) RUN_MODE="$2"; shift 2 ;;
        -v|--soc-version) SOC_VERSION="$2"; shift 2 ;;
        -pes|--pes|--nranks) PES="$2"; shift 2 ;;
        -M|--tokens) M="$2"; shift 2 ;;
        -K|--hidden) K="$2"; shift 2 ;;
        -topK|--topk) TOPK="$2"; shift 2 ;;
        -expertPerPe|--experts-per-rank) EXPERT_PER_PE="$2"; shift 2 ;;
        --skip-run) SKIP_RUN="$2"; shift 2 ;;
        --case|--case-all)
            echo "[ERROR] case presets are unsupported; pass explicit shape parameters"
            exit 1
            ;;
        *)
            echo "[ERROR] Unknown option for Task1 scaffold: $1"
            exit 1
            ;;
    esac
done

echo "=== dispatch_combine_tile scaffold build ==="
echo "RUN_MODE=${RUN_MODE} SOC_VERSION=${SOC_VERSION} PES=${PES} M=${M} K=${K} TOPK=${TOPK} EXPERT_PER_PE=${EXPERT_PER_PE}"
echo "skip_run=${SKIP_RUN}"

rm -rf "${SCRIPT_DIR}/build"
mkdir -p "${SCRIPT_DIR}/build"
cd "${SCRIPT_DIR}/build"

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}

cmake -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}" ..
make -j16

if [ "${SKIP_RUN}" = "1" ]; then
    exit 0
fi

./dispatch_combine_tile
