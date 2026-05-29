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
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/run_a3.sh"

print_help() {
    cat <<'EOF'
Usage: bash run_a3.sh [options]

Shape:
  -pes, --pes, --nranks N
  -M, --tokens N
  -K, --hidden-size N
  -N, --intermediate-size N
  -topK, --topk N
  -expertPerPe, --experts-per-rank N
  --max-tokens-per-expert N
  --payload-tile-cols N
  --gmm-block-m N
  --gmm-block-n N
  --gmm-block-k N

Runtime:
  -r, --run-mode npu|sim
  -v, --soc-version NAME
  --first-device N          default 4 on this A3 host
  --ndevices N              default 8 on this A3 host
  --rank-from-mpi 0|1    default 1; MPI rank is the default rankId source
  --rank N               debug override; use with --rank-from-mpi 0
  --case-name NAME       label only: small, balanced, skewed, zero-token
  --backend NAME         m1-mock (default) or int8
  --seed N
  --data-dir DIR
  --timeline 0|1
  --debug 0|1
  --dry-run 0|1
  --skip-kernel-launch 0|1
  --dispatch-metadata-only 0|1
  --dispatch-only 0|1
  --gmm1-only 0|1
  --gmm1-epilogue-only 0|1
  --activation-only 0|1
  --gmm2-only 0|1
  --combine-return-only 0|1
  --hccl-buffsize-mb N
  --m1-suite 0|1       build once, then run explicit M1 real dispatch/combine cases with mock GMM payload
  --m2-suite 0|1       build once, then run explicit M2 int8 full-chain cases

Build:
  --skip-run 0|1
  --skip-build 0|1
  --clean-build 0|1
  --mpi-bin DIR

Fixed measurement policy:
  warmup_iters=3 and measure_iters=5 are fixed and are not command-line options.
  Correctness, perf, and timeline reports print to stdout by default.

Explicit templates:
  small:
    --case-name small -pes 1 -M 8 -K 64 -N 32 -topK 1 -expertPerPe 1 --max-tokens-per-expert 8
  balanced:
    --case-name balanced -pes 2 -M 16 -K 64 -N 32 -topK 2 -expertPerPe 2 --max-tokens-per-expert 32
  skewed:
    --case-name skewed -pes 2 -M 16 -K 64 -N 32 -topK 2 -expertPerPe 2 --max-tokens-per-expert 32
  zero-token:
    --case-name zero-token -pes 2 -M 16 -K 64 -N 32 -topK 2 -expertPerPe 2 --max-tokens-per-expert 32

This project does not support hidden --case presets; pass explicit shape parameters.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_help
    exit 0
fi

RUN_MODE=npu
SOC_VERSION=Ascend910B1
PES=2
M=16
HIDDEN_SIZE=64
INTERMEDIATE_SIZE=32
TOPK=2
EXPERT_PER_PE=2
MAX_TOKENS_PER_EXPERT=64
PAYLOAD_TILE_COLS=64
GMM_BLOCK_M=16
GMM_BLOCK_N=16
GMM_BLOCK_K=32
DEVICE_BASE=4
NDEVICES=8
RANK_FROM_MPI=1
RANK=""
CASE_NAME=small
BACKEND=m1-mock
SEED=1234
DATA_DIR="${PROJECT_DIR}/out"
TIMELINE=0
DEBUG=0
DRY_RUN=1
SKIP_KERNEL_LAUNCH=1
HCCL_BUFFSIZE_MB=0
SKIP_RUN=0
SKIP_BUILD=0
CLEAN_BUILD=1
MPI_BIN=""
M1_SUITE=0
M2_SUITE=0
DISPATCH_METADATA_ONLY=0
DISPATCH_ONLY=0
GMM1_ONLY=0
GMM1_EPILOGUE_ONLY=0
ACTIVATION_ONLY=0
GMM2_ONLY=0
COMBINE_RETURN_ONLY=0

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
        -K|--hidden|--hidden-size) HIDDEN_SIZE="$2"; shift 2 ;;
        -N|--intermediate|--intermediate-size) INTERMEDIATE_SIZE="$2"; shift 2 ;;
        -topK|--topk) TOPK="$2"; shift 2 ;;
        -expertPerPe|--experts-per-rank) EXPERT_PER_PE="$2"; shift 2 ;;
        --max-tokens-per-expert) MAX_TOKENS_PER_EXPERT="$2"; shift 2 ;;
        --payload-tile-cols|-tileCols) PAYLOAD_TILE_COLS="$2"; shift 2 ;;
        --gmm-block-m) GMM_BLOCK_M="$2"; shift 2 ;;
        --gmm-block-n) GMM_BLOCK_N="$2"; shift 2 ;;
        --gmm-block-k) GMM_BLOCK_K="$2"; shift 2 ;;
        --first-device|--device-base|-device-base) DEVICE_BASE="$2"; shift 2 ;;
        --ndevices) NDEVICES="$2"; shift 2 ;;
        --rank-from-mpi) RANK_FROM_MPI="$2"; shift 2 ;;
        --rank) RANK="$2"; shift 2 ;;
        --case-name) CASE_NAME="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --timeline) TIMELINE="$2"; shift 2 ;;
        --debug) DEBUG="$2"; shift 2 ;;
        --dry-run) DRY_RUN="$2"; shift 2 ;;
        --skip-kernel-launch) SKIP_KERNEL_LAUNCH="$2"; shift 2 ;;
        --dispatch-metadata-only) DISPATCH_METADATA_ONLY="$2"; shift 2 ;;
        --dispatch-only) DISPATCH_ONLY="$2"; shift 2 ;;
        --gmm1-only) GMM1_ONLY="$2"; shift 2 ;;
        --gmm1-epilogue-only) GMM1_EPILOGUE_ONLY="$2"; shift 2 ;;
        --activation-only) ACTIVATION_ONLY="$2"; shift 2 ;;
        --gmm2-only) GMM2_ONLY="$2"; shift 2 ;;
        --combine-return-only) COMBINE_RETURN_ONLY="$2"; shift 2 ;;
        --hccl-buffsize-mb) HCCL_BUFFSIZE_MB="$2"; shift 2 ;;
        --m1-suite) M1_SUITE="$2"; shift 2 ;;
        --m2-suite) M2_SUITE="$2"; shift 2 ;;
        --skip-run) SKIP_RUN="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD="$2"; shift 2 ;;
        --clean-build) CLEAN_BUILD="$2"; shift 2 ;;
        --mpi-bin) MPI_BIN="$2"; shift 2 ;;
        --warmup|--measure-iters|--iters)
            echo "[ERROR] warmup_iters=3 and measure_iters=5 are fixed"
            exit 1
            ;;
        --case|--case-all)
            echo "[ERROR] hidden case presets are unsupported; pass explicit shape parameters"
            exit 1
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "${NDEVICES}" ]; then
    NDEVICES="${PES}"
fi
if [ "${RANK_FROM_MPI}" = "0" ] && [ "${PES}" -gt 1 ] && [ -z "${RANK}" ]; then
    echo "[ERROR] --rank is required when --rank-from-mpi 0 and --pes > 1"
    exit 1
fi
if [ $(( DEVICE_BASE + PES )) -gt "${NDEVICES}" ]; then
    echo "[ERROR] deviceBase + pes > ndevices"
    exit 1
fi
if [ "${HIDDEN_SIZE}" -le 0 ] || [ "${INTERMEDIATE_SIZE}" -le 0 ] || [ "${PAYLOAD_TILE_COLS}" -le 0 ]; then
    echo "[ERROR] hidden/intermediate/payload tile sizes must be nonzero"
    exit 1
fi
if [ "${BACKEND}" != "m1-mock" ] && [ "${BACKEND}" != "int8" ]; then
    echo "[ERROR] --backend must be m1-mock or int8"
    exit 1
fi
if [ $(( HIDDEN_SIZE % PAYLOAD_TILE_COLS )) -ne 0 ]; then
    echo "[ERROR] hiddenSize must be divisible by payloadTileCols"
    exit 1
fi

DISPATCH_ROW_BYTES=$(align_up "${HIDDEN_SIZE}" 64)
RETURN_ROW_BYTES=$(align_up $(( HIDDEN_SIZE * 2 )) 64)
EXPANDED_ROWS=$(( M * TOPK ))
TOKEN_MATRIX_BYTES=$(( PES * PES * EXPERT_PER_PE * 4 ))
WORKSPACE_BYTES=$(align_up $(( TOKEN_MATRIX_BYTES + EXPANDED_ROWS * (RETURN_ROW_BYTES + 12) + 64 * 1024 )) 64)
PEER_WINDOW_BYTES=$(align_up $(( 256 + TOKEN_MATRIX_BYTES + EXPANDED_ROWS * (DISPATCH_ROW_BYTES + RETURN_ROW_BYTES + 4) + 64 * 1024 )) 64)
AUTO_HCCL_BUFFSIZE_MB=$(align_up $(( PEER_WINDOW_BYTES + 64 * 1024 * 1024 )) $(( 1024 * 1024 )))
AUTO_HCCL_BUFFSIZE_MB=$(( AUTO_HCCL_BUFFSIZE_MB / 1024 / 1024 ))
if [ "${HCCL_BUFFSIZE_MB}" -eq 0 ]; then
    HCCL_BUFFSIZE_MB="${AUTO_HCCL_BUFFSIZE_MB}"
fi
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE_MB}"

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

USE_DIRECT_HOST=0
if { [ "${PES}" -eq 1 ] || { [ "${RANK_FROM_MPI}" = "0" ] && [ -n "${RANK}" ]; }; } &&
    { [ "${DRY_RUN}" = "1" ] || [ "${SKIP_KERNEL_LAUNCH}" = "1" ]; }; then
    USE_DIRECT_HOST=1
fi

if [ -n "${MPI_BIN}" ]; then
    if [ ! -x "${MPI_BIN}/mpirun" ]; then
        echo "[ERROR] --mpi-bin does not contain executable mpirun: ${MPI_BIN}"
        exit 1
    fi
    export PATH="${MPI_BIN}:${PATH}"
fi
if [ "${USE_DIRECT_HOST}" != "1" ] && ! command -v mpirun >/dev/null 2>&1; then
    echo "[ERROR] Cannot find mpirun. Pass --mpi-bin <dir> or set PATH"
    exit 1
fi

rm -rf /dev/shm/sem.hccl* 2>/dev/null || true
ipcrm -a 2>/dev/null || true

echo "=== moe_dispatch_combine_a8w8 fused dispatch/combine runtime ==="
echo "RUN_MODE=${RUN_MODE} SOC_VERSION=${SOC_VERSION}"
echo "PES=${PES} DEVICE_BASE=${DEVICE_BASE} NDEVICES=${NDEVICES}"
echo "M=${M} HIDDEN_SIZE=${HIDDEN_SIZE} INTERMEDIATE_SIZE=${INTERMEDIATE_SIZE} TOPK=${TOPK} EXPERT_PER_PE=${EXPERT_PER_PE}"
echo "BACKEND=${BACKEND}"
echo "MAX_TOKENS_PER_EXPERT=${MAX_TOKENS_PER_EXPERT} PAYLOAD_TILE_COLS=${PAYLOAD_TILE_COLS}"
echo "GMM_BLOCK_M=${GMM_BLOCK_M} GMM_BLOCK_N=${GMM_BLOCK_N} GMM_BLOCK_K=${GMM_BLOCK_K}"
echo "rank_source=$([ "${RANK_FROM_MPI}" = "1" ] && echo mpi || echo manual)"
echo "workspace_bytes_estimate=${WORKSPACE_BYTES}"
echo "peer_window_bytes_estimate=${PEER_WINDOW_BYTES}"
echo "HCCL_BUFFSIZE=${HCCL_BUFFSIZE}"
echo "warmup_iters=3 measure_iters=5"
echo "skip_run=${SKIP_RUN} dry_run=${DRY_RUN} timeline=${TIMELINE}"
echo "launcher=$([ "${USE_DIRECT_HOST}" = "1" ] && echo direct-host || echo mpirun)"

if [ "${CLEAN_BUILD}" = "1" ] && [ "${SKIP_BUILD}" != "1" ]; then
    rm -rf "${PROJECT_DIR}/build"
fi
mkdir -p "${PROJECT_DIR}/build"
cd "${PROJECT_DIR}/build"

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}

if [ "${SKIP_BUILD}" != "1" ]; then
    cmake -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}" -DFUSED_KERNEL_ARCH=dav-c220-vec ..
    make -j16
fi

if [ "${SKIP_RUN}" = "1" ]; then
    exit 0
fi

if [ "${M1_SUITE}" = "1" ]; then
    echo "=== Running M1 real dispatch/combine suite ==="
    bash "${SCRIPT_PATH}" --m1-suite 0 --skip-build 1 --clean-build 0 --dry-run 0 --skip-kernel-launch 0 \
        --first-device "${DEVICE_BASE}" --ndevices "${NDEVICES}" \
        --case-name small -pes 1 -M 8 -K 64 -N 32 -topK 1 -expertPerPe 1 --max-tokens-per-expert 8
    bash "${SCRIPT_PATH}" --m1-suite 0 --skip-build 1 --clean-build 0 --dry-run 0 --skip-kernel-launch 0 \
        --first-device "${DEVICE_BASE}" --ndevices "${NDEVICES}" \
        --case-name balanced -pes 2 -M 16 -K 64 -N 32 -topK 2 -expertPerPe 2 --max-tokens-per-expert 32
    bash "${SCRIPT_PATH}" --m1-suite 0 --skip-build 1 --clean-build 0 --dry-run 0 --skip-kernel-launch 0 \
        --first-device "${DEVICE_BASE}" --ndevices "${NDEVICES}" \
        --case-name skewed -pes 2 -M 16 -K 64 -N 32 -topK 2 -expertPerPe 2 --max-tokens-per-expert 32
    bash "${SCRIPT_PATH}" --m1-suite 0 --skip-build 1 --clean-build 0 --dry-run 0 --skip-kernel-launch 0 \
        --first-device "${DEVICE_BASE}" --ndevices "${NDEVICES}" \
        --case-name zero-token -pes 2 -M 16 -K 64 -N 32 -topK 2 -expertPerPe 2 --max-tokens-per-expert 32
    exit 0
fi

if [ "${M2_SUITE}" = "1" ]; then
    echo "=== Running M2 int8 full-chain suite ==="
    bash "${SCRIPT_PATH}" --m2-suite 0 --backend int8 --skip-build 1 --clean-build 0 --dry-run 0 \
        --skip-kernel-launch 0 --first-device "${DEVICE_BASE}" --ndevices "${NDEVICES}" \
        --case-name small -pes 1 -M 8 -K 64 -N 32 -topK 1 -expertPerPe 1 --max-tokens-per-expert 8
    bash "${SCRIPT_PATH}" --m2-suite 0 --backend int8 --skip-build 1 --clean-build 0 --dry-run 0 \
        --skip-kernel-launch 0 --first-device "${DEVICE_BASE}" --ndevices "${NDEVICES}" \
        --case-name balanced -pes 2 -M 16 -K 64 -N 32 -topK 2 -expertPerPe 2 --max-tokens-per-expert 32
    bash "${SCRIPT_PATH}" --m2-suite 0 --backend int8 --skip-build 1 --clean-build 0 --dry-run 0 \
        --skip-kernel-launch 0 --first-device "${DEVICE_BASE}" --ndevices "${NDEVICES}" \
        --case-name skewed -pes 2 -M 16 -K 64 -N 32 -topK 2 -expertPerPe 2 --max-tokens-per-expert 32
    bash "${SCRIPT_PATH}" --m2-suite 0 --backend int8 --skip-build 1 --clean-build 0 --dry-run 0 \
        --skip-kernel-launch 0 --first-device "${DEVICE_BASE}" --ndevices "${NDEVICES}" \
        --case-name zero-token -pes 2 -M 16 -K 64 -N 32 -topK 2 -expertPerPe 2 --max-tokens-per-expert 32
    exit 0
fi

HOST_ARGS=(
    --run-mode "${RUN_MODE}"
    --soc-version "${SOC_VERSION}"
    --rank-num "${PES}"
    --tokens "${M}"
    --hidden-size "${HIDDEN_SIZE}"
    --intermediate-size "${INTERMEDIATE_SIZE}"
    --topk "${TOPK}"
    --experts-per-rank "${EXPERT_PER_PE}"
    --max-tokens-per-expert "${MAX_TOKENS_PER_EXPERT}"
    --payload-tile-cols "${PAYLOAD_TILE_COLS}"
    --gmm-block-m "${GMM_BLOCK_M}"
    --gmm-block-n "${GMM_BLOCK_N}"
    --gmm-block-k "${GMM_BLOCK_K}"
    --first-device "${DEVICE_BASE}"
    --ndevices "${NDEVICES}"
    --rank-from-mpi "${RANK_FROM_MPI}"
    --case-name "${CASE_NAME}"
    --backend "${BACKEND}"
    --seed "${SEED}"
    --data-dir "${DATA_DIR}"
    --timeline "${TIMELINE}"
    --debug "${DEBUG}"
    --dry-run "${DRY_RUN}"
    --skip-kernel-launch "${SKIP_KERNEL_LAUNCH}"
    --dispatch-metadata-only "${DISPATCH_METADATA_ONLY}"
    --dispatch-only "${DISPATCH_ONLY}"
    --gmm1-only "${GMM1_ONLY}"
    --gmm1-epilogue-only "${GMM1_EPILOGUE_ONLY}"
    --activation-only "${ACTIVATION_ONLY}"
    --gmm2-only "${GMM2_ONLY}"
    --combine-return-only "${COMBINE_RETURN_ONLY}"
    --hccl-buffsize-mb "${HCCL_BUFFSIZE_MB}"
)
if [ -n "${RANK}" ]; then
    HOST_ARGS+=(--rank "${RANK}")
fi

if [ "${USE_DIRECT_HOST}" = "1" ]; then
    echo "=== Running moe_dispatch_combine_a8w8 (direct host) ==="
    ./moe_dispatch_combine_a8w8 "${HOST_ARGS[@]}"
else
    echo "=== Running moe_dispatch_combine_a8w8 (mpirun) ==="
    mpirun -n "${PES}" ./moe_dispatch_combine_a8w8 "${HOST_ARGS[@]}"
fi
