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
AIV_BLOCKS=0
TILE_COLS=1024
TILE_COLS_SET=0
ROW_CHUNK=0
METADATA_PAD=16
MAX_OUTPUT_SIZE=0
DEVICE_BASE=0
NDEVICES=""
MPI_BIN=""
HCCL_BUFFSIZE_MB=0
KEEP_HCCL_SHM=0
RANK_FROM_MPI=1
RANK=""
DEBUG=0
ITERS=1
WARMUP=1
SEED=1234
DATA_DIR="${SCRIPT_DIR}/out"
GEN_DATA=1
VERIFY=1
RTOL=1e-2
ATOL=1e-2
SKIP_RUN=0
SKIP_BUILD=0
CLEAN_BUILD=1
SKIP_KERNELS=0
HOST_GOLDEN_ONLY=0
COMBINE_RETURN_ONLY=0

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

Kernel/runtime:
  -r, --run-mode npu
  -v, --soc-version NAME
  -aivBlocks, --aiv-blocks N
  -tileCols, --tile-cols N
  --row-chunk N
  --metadata-pad N
  -device-base, --device-base, --first-device N
  --ndevices N
  --mpi-bin DIR
  --hccl-buffsize-mb N
  --keep-hccl-shm 0|1
  --rank-from-mpi 0|1
  --rank N

Data/debug:
  -debug, --debug 0|1|2
  -iters, --iters N
  -warmup, --warmup N
  --seed N
  --data-dir DIR
  --gen-data 0|1
  --verify 0|1
  --rtol FLOAT
  --atol FLOAT

Build/stage gates:
  --skip-run 0|1
  --skip-build 0|1
  --clean-build 0|1
  --skip-kernels 0|1
  --host-golden-only 0|1
  --combine-return-only 0|1

This project does not support --case presets; pass explicit shape parameters.
EOF
}

align_up() {
    local value=$1
    local alignment=$2
    echo $(( ((value + alignment - 1) / alignment) * alignment ))
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
        -tileCols|--tile-cols) TILE_COLS="$2"; TILE_COLS_SET=1; shift 2 ;;
        --row-chunk) ROW_CHUNK="$2"; shift 2 ;;
        --metadata-pad) METADATA_PAD="$2"; shift 2 ;;
        -device-base|--device-base|--first-device) DEVICE_BASE="$2"; shift 2 ;;
        --ndevices) NDEVICES="$2"; shift 2 ;;
        --mpi-bin) MPI_BIN="$2"; shift 2 ;;
        --hccl-buffsize-mb) HCCL_BUFFSIZE_MB="$2"; shift 2 ;;
        --keep-hccl-shm) KEEP_HCCL_SHM="$2"; shift 2 ;;
        --rank-from-mpi) RANK_FROM_MPI="$2"; shift 2 ;;
        --rank) RANK="$2"; shift 2 ;;
        -debug|--debug) DEBUG="$2"; shift 2 ;;
        -iters|--iters) ITERS="$2"; shift 2 ;;
        -warmup|--warmup) WARMUP="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --gen-data) GEN_DATA="$2"; shift 2 ;;
        --verify) VERIFY="$2"; shift 2 ;;
        --rtol) RTOL="$2"; shift 2 ;;
        --atol) ATOL="$2"; shift 2 ;;
        --skip-run) SKIP_RUN="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD="$2"; shift 2 ;;
        --clean-build) CLEAN_BUILD="$2"; shift 2 ;;
        --skip-kernels) SKIP_KERNELS="$2"; shift 2 ;;
        --host-golden-only) HOST_GOLDEN_ONLY="$2"; shift 2 ;;
        --combine-return-only) COMBINE_RETURN_ONLY="$2"; shift 2 ;;
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

if [ "${RUN_MODE}" != "npu" ]; then
    echo "[ERROR] run-mode must be npu for the first version"
    exit 1
fi
if [ "${TILE_COLS_SET}" -eq 0 ] && [ "${K}" -lt "${TILE_COLS}" ]; then
    TILE_COLS="${K}"
fi
if [ "${PES}" -le 0 ] || [ "${M}" -le 0 ] || [ "${K}" -le 0 ] || [ "${TOPK}" -le 0 ] || \
   [ "${EXPERT_PER_PE}" -le 0 ] || [ "${TILE_COLS}" -le 0 ] || [ "${METADATA_PAD}" -le 0 ]; then
    echo "[ERROR] shape fields, TILE_COLS, and METADATA_PAD must be nonzero"
    exit 1
fi
if [ $(( K % TILE_COLS )) -ne 0 ]; then
    echo "[ERROR] K % tileCols must be 0"
    exit 1
fi
if [ -z "${NDEVICES}" ]; then
    NDEVICES="${PES}"
fi
if [ $(( DEVICE_BASE + PES )) -gt "${NDEVICES}" ]; then
    echo "[ERROR] deviceBase + pes > ndevices"
    exit 1
fi

EXPERT_NUM=$(( PES * EXPERT_PER_PE ))
if [ "${MAX_OUTPUT_SIZE}" -eq 0 ]; then
    MAX_OUTPUT_SIZE=$(( PES * M * TOPK ))
fi
REQUIRED_ROWS=$(( PES * M * TOPK ))
if [ "${MAX_OUTPUT_SIZE}" -lt "${REQUIRED_ROWS}" ]; then
    echo "[ERROR] maxOutputSize is smaller than EP * M * topK; capacity/drop is unsupported"
    exit 1
fi
EFFECTIVE_AIV_BLOCKS="${AIV_BLOCKS}"
if [ "${EFFECTIVE_AIV_BLOCKS}" -eq 0 ]; then
    EFFECTIVE_AIV_BLOCKS=8
fi
AIV_BLOCKS="${EFFECTIVE_AIV_BLOCKS}"
EXPERT_NUM_PADDED=$(align_up "${EXPERT_NUM}" "${METADATA_PAD}")
EXPANDED_ROWS=$(( M * TOPK ))

workspace_offset=0
append_workspace_field() {
    local bytes=$1
    workspace_offset=$(align_up "${workspace_offset}" 64)
    workspace_offset=$(( workspace_offset + bytes ))
}
append_workspace_field $(( EXPERT_NUM_PADDED * 4 ))
append_workspace_field $(( EFFECTIVE_AIV_BLOCKS * EXPERT_NUM_PADDED * 4 ))
append_workspace_field $(( EFFECTIVE_AIV_BLOCKS * EXPERT_NUM_PADDED * 4 ))
append_workspace_field $(( PES * EXPERT_NUM_PADDED * 4 ))
append_workspace_field $(( EXPERT_PER_PE * 4 ))
append_workspace_field $(( PES * EXPERT_PER_PE * 4 ))
SYNC_SLOTS=$(( EFFECTIVE_AIV_BLOCKS * (8 + EXPERT_NUM_PADDED) ))
if [ "${SYNC_SLOTS}" -lt 64 ]; then
    SYNC_SLOTS=64
fi
append_workspace_field $(( SYNC_SLOTS * 4 ))
append_workspace_field $(( EFFECTIVE_AIV_BLOCKS * TILE_COLS * 4 ))
append_workspace_field $(( MAX_OUTPUT_SIZE * K * 2 ))
append_workspace_field $(( EXPANDED_ROWS * K * 2 ))
WORKSPACE_BYTES=$(align_up "${workspace_offset}" 64)

peer_offset=0
append_peer_field() {
    local bytes=$1
    peer_offset=$(align_up "${peer_offset}" 64)
    peer_offset=$(( peer_offset + bytes ))
}
append_peer_field $(( PES * EXPERT_NUM_PADDED * 4 ))
append_peer_field $(( EXPANDED_ROWS * 4 ))
append_peer_field $(( EXPANDED_ROWS * K * 2 ))
append_peer_field $(( EXPANDED_ROWS * K * 2 ))
append_peer_field $(( PES * 4 ))
append_peer_field $(( PES * 4 ))
PEER_WINDOW_BYTES=$(align_up "${peer_offset}" 64)
AUTO_HCCL_BUFFSIZE=$(align_up $(( PEER_WINDOW_BYTES + 64 * 1024 * 1024 )) $(( 1024 * 1024 )))
AUTO_HCCL_BUFFSIZE=$(( AUTO_HCCL_BUFFSIZE / 1024 / 1024 ))
if [ "${HCCL_BUFFSIZE_MB}" -eq 0 ]; then
    export HCCL_BUFFSIZE="${AUTO_HCCL_BUFFSIZE}"
else
    export HCCL_BUFFSIZE="${HCCL_BUFFSIZE_MB}"
fi

if [ -n "${MPI_BIN}" ]; then
    if [ ! -x "${MPI_BIN}/mpirun" ]; then
        echo "[ERROR] --mpi-bin does not contain executable mpirun: ${MPI_BIN}"
        exit 1
    fi
    export PATH="${MPI_BIN}:${PATH}"
    MPI_LIB_DIR="$(cd "${MPI_BIN}/.." && pwd)/lib"
    export LD_LIBRARY_PATH="${MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    if [ -f "${MPI_LIB_DIR}/libmpi.so" ]; then
        export MPI_LIB_PATH="${MPI_LIB_DIR}/libmpi.so"
    fi
else
    if ! command -v mpirun >/dev/null 2>&1; then
        MPI_SEARCH_DIRS="/usr/local/mpich/bin /home/mpich/bin"
        for candidate in /home/*/mpich/bin /home/*/*/mpich/bin; do
            [ -d "$candidate" ] && MPI_SEARCH_DIRS="${MPI_SEARCH_DIRS} ${candidate}"
        done
        for d in ${MPI_SEARCH_DIRS}; do
            if [ -x "$d/mpirun" ]; then
                export PATH="$d:${PATH}"
                MPI_LIB_DIR="$(cd "$d/.." && pwd)/lib"
                export LD_LIBRARY_PATH="${MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
                if [ -f "${MPI_LIB_DIR}/libmpi.so" ]; then
                    export MPI_LIB_PATH="${MPI_LIB_DIR}/libmpi.so"
                fi
                break
            fi
        done
    fi
fi
if ! command -v mpirun >/dev/null 2>&1; then
    echo "[ERROR] Cannot find mpirun. Pass --mpi-bin <dir> or set PATH"
    exit 1
fi

if [ "${KEEP_HCCL_SHM}" != "1" ]; then
    rm -rf /dev/shm/sem.hccl* 2>/dev/null || true
    ipcrm -a 2>/dev/null || true
fi

echo "=== combine_tile scaffold build ==="
echo "RUN_MODE=${RUN_MODE}"
echo "SOC_VERSION=${SOC_VERSION}"
echo "PES=${PES} DEVICE_BASE=${DEVICE_BASE} NDEVICES=${NDEVICES}"
echo "M=${M} K=${K} TOPK=${TOPK} EXPERT_PER_PE=${EXPERT_PER_PE} MAX_OUTPUT_SIZE=${MAX_OUTPUT_SIZE}"
echo "AIV_BLOCKS=${AIV_BLOCKS} TILE_COLS=${TILE_COLS} ROW_CHUNK=${ROW_CHUNK} METADATA_PAD=${METADATA_PAD}"
echo "workspace_bytes=${WORKSPACE_BYTES}"
echo "peer_window_bytes=${PEER_WINDOW_BYTES}"
echo "HCCL_BUFFSIZE=${HCCL_BUFFSIZE}"
echo "DATA_DIR=${DATA_DIR}"
echo "WARMUP=${WARMUP} ITERS=${ITERS} VERIFY=${VERIFY} DEBUG=${DEBUG}"
echo "skip_run=${SKIP_RUN}"
echo "MPI_LIB_PATH=${MPI_LIB_PATH}"

if [ "${CLEAN_BUILD}" = "1" ] && [ "${SKIP_BUILD}" != "1" ]; then
    rm -rf "${SCRIPT_DIR}/build"
fi
mkdir -p "${SCRIPT_DIR}/build"
cd "${SCRIPT_DIR}/build"

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}

if [ "${SKIP_BUILD}" != "1" ]; then
    cmake -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}" ..
    make -j16
fi

if [ "${SKIP_RUN}" = "1" ]; then
    exit 0
fi

HOST_ARGS=(
    --run-mode "${RUN_MODE}"
    --soc-version "${SOC_VERSION}"
    --pes "${PES}"
    --tokens "${M}"
    --hidden "${K}"
    --topk "${TOPK}"
    --experts-per-rank "${EXPERT_PER_PE}"
    --max-output-size "${MAX_OUTPUT_SIZE}"
    --aiv-blocks "${AIV_BLOCKS}"
    --tile-cols "${TILE_COLS}"
    --row-chunk "${ROW_CHUNK}"
    --metadata-pad "${METADATA_PAD}"
    --device-base "${DEVICE_BASE}"
    --ndevices "${NDEVICES}"
    --hccl-buffsize-mb "${HCCL_BUFFSIZE}"
    --keep-hccl-shm "${KEEP_HCCL_SHM}"
    --rank-from-mpi "${RANK_FROM_MPI}"
    --debug "${DEBUG}"
    --iters "${ITERS}"
    --warmup "${WARMUP}"
    --seed "${SEED}"
    --data-dir "${DATA_DIR}"
    --gen-data "${GEN_DATA}"
    --verify "${VERIFY}"
    --rtol "${RTOL}"
    --atol "${ATOL}"
    --skip-kernels "${SKIP_KERNELS}"
    --host-golden-only "${HOST_GOLDEN_ONLY}"
    --combine-return-only "${COMBINE_RETURN_ONLY}"
)
if [ -n "${RANK}" ]; then
    HOST_ARGS+=(--rank "${RANK}")
fi

echo "=== Running combine_tile (HCCL, mpirun) ==="
mpirun -n "${PES}" ./combine_tile "${HOST_ARGS[@]}"
