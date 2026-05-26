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

: "${ASCEND_CANN_PATH:=/home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh}"
if [ ! -f "${ASCEND_CANN_PATH}" ]; then
    ASCEND_CANN_PATH=$(ls -1d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort -V | tail -1 || true)
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
SOC_VERSION=Ascend950PR_958b
PES=2
M=64
K=7168
TOPK=8
EXPERT_PER_PE=2
AIV_BLOCKS=0
TILE_COLS=1024
METADATA_PAD=16
MAX_OUTPUT_SIZE=0
HCCL_BUFFSIZE_MB=0
SKIP_BUILD=0
CLEAN_BUILD=1
VERIFY=1

print_help() {
    cat <<'EOF'
Usage: bash run.sh [options]

Shape:
  -pes, --pes, --nranks N
  -M, --tokens N
  -K, --hidden N
  -topK, --topk N
  -expertPerPe, --experts-per-rank N
  --max-output-size N

Runtime/build:
  -v, --soc-version NAME
  -aivBlocks, --aiv-blocks N
  -tileCols, --tile-cols N
  --metadata-pad N
  --hccl-buffsize-mb N
  --verify 0|1
  --skip-build 0|1
  --clean-build 0|1

This A5 scaffold defaults to verification.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) print_help; exit 0 ;;
        -r|--run-mode) RUN_MODE="$2"; shift 2 ;;
        -v|--soc-version) SOC_VERSION="$2"; shift 2 ;;
        -pes|--pes|--nranks) PES="$2"; shift 2 ;;
        -M|--tokens) M="$2"; shift 2 ;;
        -K|--hidden) K="$2"; shift 2 ;;
        -topK|--topk) TOPK="$2"; shift 2 ;;
        -expertPerPe|--experts-per-rank) EXPERT_PER_PE="$2"; shift 2 ;;
        --max-output-size) MAX_OUTPUT_SIZE="$2"; shift 2 ;;
        -aivBlocks|--aiv-blocks) AIV_BLOCKS="$2"; shift 2 ;;
        -tileCols|--tile-cols) TILE_COLS="$2"; shift 2 ;;
        --metadata-pad) METADATA_PAD="$2"; shift 2 ;;
        --hccl-buffsize-mb) HCCL_BUFFSIZE_MB="$2"; shift 2 ;;
        --verify) VERIFY="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD="$2"; shift 2 ;;
        --clean-build) CLEAN_BUILD="$2"; shift 2 ;;
        *)
            echo "[ERROR] Unknown option: $1"
            exit 1
            ;;
    esac
done

BUILD_DIR="${SCRIPT_DIR}/out"
if [ "${SKIP_BUILD}" -eq 0 ]; then
    if [ "${CLEAN_BUILD}" -ne 0 ]; then
        rm -rf "${BUILD_DIR}"
    fi
    cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}"
    cmake --build "${BUILD_DIR}" --target moe_dispatch -j8
fi

BIN="${BUILD_DIR}/moe_dispatch"
if [ ! -x "${BIN}" ]; then
    echo "[ERROR] ${BIN} not found; build first or pass --skip-build 0"
    exit 1
fi

export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}"
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/aarch64-conda-linux-gnu/lib:${LD_LIBRARY_PATH}"
fi

"${BIN}" \
    -r "${RUN_MODE}" \
    -v "${SOC_VERSION}" \
    -pes "${PES}" \
    -M "${M}" \
    -K "${K}" \
    -topK "${TOPK}" \
    -expertPerPe "${EXPERT_PER_PE}" \
    --max-output-size "${MAX_OUTPUT_SIZE}" \
    --aiv-blocks "${AIV_BLOCKS}" \
    --tile-cols "${TILE_COLS}" \
    --metadata-pad "${METADATA_PAD}" \
    --hccl-buffsize-mb "${HCCL_BUFFSIZE_MB}" \
    --verify "${VERIFY}"
