#!/bin/bash
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
CASE=smoke
DEVICE=0
RANKS=2
EXPERTS_PER_RANK=2
TOKENS=4
HIDDEN=8
TOPK=2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-mode) RUN_MODE="$2"; shift 2 ;;
        --soc-version) SOC_VERSION="$2"; shift 2 ;;
        --case) CASE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --ranks) RANKS="$2"; shift 2 ;;
        --experts-per-rank) EXPERTS_PER_RANK="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --hidden) HIDDEN="$2"; shift 2 ;;
        --topk) TOPK="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

rm -rf "${SCRIPT_DIR}/build"
mkdir "${SCRIPT_DIR}/build"
cd "${SCRIPT_DIR}/build"

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} ..
make -j16

./dispatch_combine --case "${CASE}" --device "${DEVICE}" --ranks "${RANKS}" \
    --experts-per-rank "${EXPERTS_PER_RANK}" --tokens "${TOKENS}" --hidden "${HIDDEN}" --topk "${TOPK}"
