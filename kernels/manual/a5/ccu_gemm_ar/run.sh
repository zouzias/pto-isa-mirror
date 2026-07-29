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

# CCU GEMM AllReduce — Pull Reduce + Push Broadcast (Persistent CCU)

# Ascend CANN environment: honor ASCEND_CANN_PATH or auto-detect
export CMAKE_PREFIX_PATH="$HOME/.local/lib64/cmake:${CMAKE_PREFIX_PATH:-}"

# CCU custom kernel mode — required for HcommCcuKernelRegister to work
export HCCL_CCU_CUSTOM_OP_MODE=1

# MPI setup: search common mpich install locations.
# Override with MPI_SEARCH_DIRS (space-separated list of bin/ directories).
if [ -z "${MPI_SEARCH_DIRS:-}" ]; then
    MPI_SEARCH_DIRS="/usr/local/mpich/bin /home/mpich/bin"
    for candidate in /home/*/mpich/bin /home/*/*/mpich/bin; do
        [ -d "$candidate" ] && MPI_SEARCH_DIRS="$MPI_SEARCH_DIRS $candidate"
    done
fi
for d in ${MPI_SEARCH_DIRS}; do
    if [ -x "$d/mpirun" ]; then
        export PATH="$d:$PATH"
        MPI_LIB_DIR="$(dirname "$d")/lib"
        export LD_LIBRARY_PATH="$MPI_LIB_DIR:${LD_LIBRARY_PATH:-}"
        export MPI_LIB_PATH="$MPI_LIB_DIR/libmpi.so"
        break
    fi
done

SHORT=r:,v:,n:,d:
LONG=run-mode:,soc-version:,nranks:,ndevices:,compute-blocks:,comm-blocks:,comm-group-tiles:,ccu-pipe-depth:,mission:,
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
        (-n | --nranks )
            NRANKS="$2"
            shift 2;;
        (-d | --ndevices )
            NDEVICES="$2"
            shift 2;;
        (--compute-blocks )
            COMPUTE_BLOCKS="$2"
            shift 2;;
        (--comm-blocks )
            COMM_BLOCKS="$2"
            shift 2;;
        (--comm-group-tiles )
            COMM_GROUP_TILES="$2"
            shift 2;;
        (--ccu-pipe-depth )
            CCU_PIPE_DEPTH="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

: "${NRANKS:=2}"
: "${NDEVICES:=2}"
# Stable path: single CCU mission per rank (gate registry is one gate/rank).
: "${CCU_MISSION_PARALLEL:=1}"
if [ "${CCU_MISSION_PARALLEL}" != "1" ]; then
    echo "[ERROR] CCU_MISSION_PARALLEL=${CCU_MISSION_PARALLEL} unsupported; stable path requires 1."
    exit 1
fi
# AIV->CCU progress handshake depth. 1 = stable path. 2 = cross-group pipelining
# (Broadcast of group g overlaps Reduce of group g+1); UNVERIFIED ON HARDWARE.
: "${CCU_PIPE_DEPTH:=1}"
if [ "${CCU_PIPE_DEPTH}" != "1" ] && [ "${CCU_PIPE_DEPTH}" != "2" ]; then
    echo "[ERROR] CCU_PIPE_DEPTH=${CCU_PIPE_DEPTH} unsupported; use 1 or 2."
    exit 1
fi
if [ "${CCU_PIPE_DEPTH}" = "2" ]; then
    echo "[WARN] CCU_PIPE_DEPTH=2 is experimental and unverified on hardware."
fi
# COMM_GROUP_TILES: unset → config.h default (16). Residual last groups OK (owner shard padded).

: "${RUN_MODE:=npu}"
# A5 (dav-c310 + CCU): default matches allgather_gemm / gemm_ar README.
: "${SOC_VERSION:=Ascend950PR_958b}"

# Clean stale HCCL shared-memory state from any previous crashed run
rm -rf /dev/shm/sem.hccl* 2>/dev/null

A5_SOC_PATTERN="^Ascend950PR_|^Ascend910_9599"
if [[ ! "${SOC_VERSION}" =~ ${A5_SOC_PATTERN} ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    echo "        ccu_gemm_ar requires A5 SoC (e.g. Ascend950PR_958b, Ascend950PR_9599, Ascend910_9599)."
    echo "        Ascend910B* is A3/B series and does not support CCU."
    exit 1
fi

if [[ "${SOC_VERSION}" =~ ^Ascend950PR ]] && [ "${RUN_MODE}" == "sim" ]; then
    echo "[WARN] Simulator support depends on the installed ${SOC_VERSION} package. Proceeding with sim mode."
fi

: "${G_M:=5416}"
: "${G_K:=6144}"
: "${G_N:=1408}"
: "${G_BASE_M:=}"
: "${G_BASE_N:=}"

# Pad M/N to G_BASE_M / G_BASE_N (defaults 128 / 256 match gemm_ar_config.h). G_BASE_K is fixed at 64 in kernel.
BM=${G_BASE_M:-128}
BN=${G_BASE_N:-256}
PAD_M=$(( ((G_M + BM - 1) / BM) * BM ))
PAD_N=$(( ((G_N + BN - 1) / BN) * BN ))

# HCCL window = packed gemm + packed reduced + row-major output + signal/metadata margin
# Extra margin covers owner-shard padding up to (group_tiles-1) tiles per rank.
NEEDED_MB=$(( PAD_M * PAD_N * 2 * 3 / 1024 / 1024 + 128 ))
CURRENT_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
if [ "${CURRENT_BUFFSIZE}" -lt "${NEEDED_MB}" ]; then
    echo "[INFO] Raising HCCL_BUFFSIZE from ${CURRENT_BUFFSIZE} to ${NEEDED_MB} MB for M=${G_M}(pad=${PAD_M}) N=${G_N}(pad=${PAD_N}) nranks=${NRANKS}"
    export HCCL_BUFFSIZE="${NEEDED_MB}"
fi

# cann-9.2: public AscendC::ccu headers under include/hcomm/ccu (ccu_launch.h /
# ccu_primitives.hpp). Optional internal pkg_inc still used by CMake for CKE
# CompletedEvent meta (ccu_datatype_v1.h); override with HCOMM_PKG_INC.
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "[ERROR] ASCEND_HOME_PATH is empty. source set_env.sh first (cann-9.2)."
    exit 1
fi

HCOMM_OPTS=""
_INTERNAL_PKG=""
for _cand in \
    "${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/hccl/internal/hcomm/pkg_inc" \
    "${ASCEND_HOME_PATH}/x86_64-linux/asc/include/adv_api/hccl/internal/hcomm/pkg_inc" \
    "${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc" \
    "${ASCEND_HOME_PATH}/x86_64-linux/pkg_inc" \
    "${ASCEND_HOME_PATH}/pkg_inc"
do
    if [ -f "${_cand}/hcomm/ccu/ccu_datatype_v1.h" ] || [ -f "${_cand}/hcomm/ccu/ccu_kernel.h" ]; then
        _INTERNAL_PKG="${_cand}"
        break
    fi
done

if [ -n "${HCOMM_PKG_INC:-}" ]; then
    HCOMM_OPTS="-DHCOMM_PKG_INC=${HCOMM_PKG_INC}"
elif [ -n "${_INTERNAL_PKG}" ]; then
    # Pass internal tree so CMake finds CompletedEvent / assist headers on 9.2.
    HCOMM_OPTS="-DHCOMM_PKG_INC=${_INTERNAL_PKG}"
    HCOMM_PKG_INC="${_INTERNAL_PKG}"
fi

if [ ! -f "${ASCEND_HOME_PATH}/include/hcomm/ccu/ccu_launch.h" ] && \
   [ ! -f "${ASCEND_HOME_PATH}/include/hcomm/ccu/ccu_primitives.hpp" ] && \
   [ -z "${HCOMM_PKG_INC:-}" ]; then
    echo "[ERROR] Cannot find cann-9.2 hcomm CCU headers under ${ASCEND_HOME_PATH}."
    echo "  expected: include/hcomm/ccu/ccu_launch.h (public AscendC::ccu)"
    echo "  and/or:   .../internal/hcomm/pkg_inc/hcomm/ccu/ccu_datatype_v1.h"
    echo "  source set_env.sh so ASCEND_HOME_PATH points at cann-9.2, or:"
    echo "  export HCOMM_PKG_INC=/path/to/pkg_inc"
    exit 1
fi

echo "=== CCU GEMM AllReduce (Persistent CCU) ==="
echo "  RUN_MODE: ${RUN_MODE}  SOC_VERSION: ${SOC_VERSION}"
echo "  NRANKS: ${NRANKS}  NDEVICES: ${NDEVICES}"
echo "  HCCL_BUFFSIZE: ${HCCL_BUFFSIZE:-200} MB"
echo "  COMPUTE_BLOCKS: ${COMPUTE_BLOCKS:-default}  COMM_BLOCKS: ${COMM_BLOCKS:-default}  (per-tile CCU_WHILE progress CKE)"
echo "  COMM_GROUP_TILES: ${COMM_GROUP_TILES:-default}"
echo "  CCU_MISSION_PARALLEL (K): ${CCU_MISSION_PARALLEL}"
echo "  CCU_PIPE_DEPTH: ${CCU_PIPE_DEPTH}  (1=stable, 2=cross-group pipelining, experimental)"
echo "  HCOMM_PKG_INC: ${HCOMM_PKG_INC:-auto}"
echo "==========================="

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}

if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${CONDA_PREFIX}/aarch64-conda-linux-gnu/lib:${LD_LIBRARY_PATH}
fi

set -euo pipefail

BLOCK_OPTS=""
if [ -n "${COMPUTE_BLOCKS:-}" ]; then
    BLOCK_OPTS="$BLOCK_OPTS -DCOMPUTE_BLOCKS=${COMPUTE_BLOCKS}"
fi
if [ -n "${COMM_BLOCKS:-}" ]; then
    BLOCK_OPTS="$BLOCK_OPTS -DCOMM_BLOCKS=${COMM_BLOCKS}"
fi
if [ -n "${COMM_GROUP_TILES:-}" ]; then
    BLOCK_OPTS="$BLOCK_OPTS -DCOMM_GROUP_TILES=${COMM_GROUP_TILES}"
fi
TILE_OPTS=""
[ -n "${G_BASE_M}" ] && TILE_OPTS="$TILE_OPTS -DCONFIG_G_BASE_M=${G_BASE_M}"
[ -n "${G_BASE_N}" ] && TILE_OPTS="$TILE_OPTS -DCONFIG_G_BASE_N=${G_BASE_N}"

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
      -DCONFIG_G_M=${G_M} -DCONFIG_G_K=${G_K} -DCONFIG_G_N=${G_N} \
      -DCONFIG_CCU_MISSION_PARALLEL=${CCU_MISSION_PARALLEL} \
      -DCONFIG_CCU_PIPE_DEPTH=${CCU_PIPE_DEPTH} \
      ${HCOMM_OPTS} ${BLOCK_OPTS} ${TILE_OPTS} ..
make -j16

echo ""
echo "=== Running CCU GEMM AllReduce (mpirun) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRST_DEVICE="${FIRST_DEVICE:-0}"
# ccu_gemm_ar owns its input cache directory (decoupled from gemm_ar).
export GEMM_AR_DIR="${CCU_INPUT_CACHE_DIR:-${SCRIPT_DIR}/.ccu_gemm_ar_input_cache}"
mkdir -p "${GEMM_AR_DIR}"
mpirun -n ${NRANKS} ./ccu_gemm_allreduce --first-device ${FIRST_DEVICE}
