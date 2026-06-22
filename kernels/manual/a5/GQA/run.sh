#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

SHORT=r:,v:,n:,c:,a:,p:,m:,i,d,k,q:,l:
LONG=run-mode:,soc-version:,npu:,case:,cases:,qk-preload:,mode:,intermediate,debug,mask,mode_dn,num-q-heads:,num-kv-heads:
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
        (-n | --npu )
            NPU_ID="$2"
            shift 2;;
        (-c | --case )
            CASE_FILTER="$2"
            shift 2;;
        (-a | --cases )
            CASES_RAW="$2"
            shift 2;;
        (-p | --qk-preload )
            QK_PRELOAD="$2"
            shift 2;;
        (-m | --mode )
            FIFO_MODE="$2"
            shift 2;;
        (-i | --intermediate )
            INTERMEDIATE=1
            shift 1;;
        (-d | --debug )
            DEBUG_BUILD=1
            shift 1;;
        (--mode_dn )
            MODE_DN=1
            shift 1;;
        (-k | --mask )
            CAUSAL_MASK=1
            shift 1;;
        (-q | --num-q-heads )
            NUM_Q_HEADS="$2"
            shift 2;;
        (-l | --num-kv-heads )
            NUM_KV_HEADS="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

pattern="^Ascend950PR_9599"
if [[ ! "$SOC_VERSION" =~ $pattern ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}, this folder only support A5."
    exit 1
fi

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
set -euo pipefail

: "${NPU_ID:=0}"
: "${QK_PRELOAD:=2}"
: "${FIFO_MODE:=1}"

GEN_CASE_ARGS=()
if [[ -n "${CASE_FILTER:-}" && "${CASE_FILTER}" == --* ]]; then
    CASE_FILTER=""
fi

if [[ -n "${CASES_RAW:-}" ]]; then
    IFS=';' read -ra CASE_ENTRIES <<< "${CASES_RAW}"
    for entry in "${CASE_ENTRIES[@]}"; do
        GEN_CASE_ARGS+=(--cases "$entry")
    done
elif [[ -n "${CASE_FILTER:-}" ]]; then
    if [[ "${CASE_FILTER}" == *","* ]]; then
        GEN_CASE_ARGS+=(--cases "${CASE_FILTER}")
    fi
fi

echo "[RUN.SH] CASE_FILTER=${CASE_FILTER:-}<none>"
echo "[RUN.SH] CASES_RAW=${CASES_RAW:-}<none>"
echo "[RUN.SH] NPU_ID=${NPU_ID}"
echo "[RUN.SH] QK_PRELOAD=${QK_PRELOAD}"
echo "[RUN.SH] FIFO_MODE=${FIFO_MODE} (0=ALL_GM, 1=ALL_UB, 2=QK_PV_UB_ONLY)"
echo "[RUN.SH] NUM_Q_HEADS=${NUM_Q_HEADS:-<from cases>}"
echo "[RUN.SH] NUM_KV_HEADS=${NUM_KV_HEADS:-<from cases>}"
echo "[RUN.SH] GEN_CASE_ARGS=${GEN_CASE_ARGS[*]:-<none>}"
echo "[RUN.SH] INTERMEDIATE=${INTERMEDIATE:-0}"
echo "[RUN.SH] CAUSAL_MASK=${CAUSAL_MASK:-0}"
echo "[RUN.SH] DEBUG=${DEBUG_BUILD:-0}"
echo "[RUN.SH] DN_MODE=${MODE_DN:-1}"

EXTRA_GEN_ARGS=()
if [[ -n "${NUM_Q_HEADS:-}" ]]; then
    EXTRA_GEN_ARGS+=(--num-q-heads "${NUM_Q_HEADS}")
fi
if [[ -n "${NUM_KV_HEADS:-}" ]]; then
    EXTRA_GEN_ARGS+=(--num-kv-heads "${NUM_KV_HEADS}")
fi

python3 ../scripts/generate_cases.py --qk-preload "${QK_PRELOAD}" "${GEN_CASE_ARGS[@]}" --causal-mask "${CAUSAL_MASK:-0}" "${EXTRA_GEN_ARGS[@]}"
python3 ../scripts/validate_buffer_usage.py --mode dn --cases generated_cases.json

CMAKE_EXTRA=()
if [[ -n "${DEBUG_BUILD:-}" ]]; then
    CMAKE_EXTRA+=(-DDEBUG_MODE=ON)
fi
CMAKE_EXTRA+=(-DMODE_DN=ON)
CMAKE_EXTRA+=(-DFIFO_MODE=${FIFO_MODE})

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} "${CMAKE_EXTRA[@]}" ..
make -j16

EXTRA_BIN_ARGS=()
if [[ -n "${INTERMEDIATE:-}" ]]; then
    EXTRA_BIN_ARGS+=(--intermediate)
fi
EXTRA_BIN_ARGS+=(--sys_cnt_multiple=1.0)

EXTRA_DATA_ARGS=()
if [[ -n "${NUM_Q_HEADS:-}" ]]; then
    EXTRA_DATA_ARGS+=(--num-q-heads "${NUM_Q_HEADS}")
fi
if [[ -n "${NUM_KV_HEADS:-}" ]]; then
    EXTRA_DATA_ARGS+=(--num-kv-heads "${NUM_KV_HEADS}")
fi

if [[ -n "${CASE_FILTER:-}" ]]; then
    python3 ../scripts/gen_data.py --case="${CASE_FILTER}" "${GEN_CASE_ARGS[@]}" "${EXTRA_DATA_ARGS[@]}" --causal-mask "${CAUSAL_MASK:-0}"
    time ./gqa_performance_dn --npu="${NPU_ID}" --case="${CASE_FILTER}" "${EXTRA_BIN_ARGS[@]}"
elif [[ -n "${CASES_RAW:-}" ]]; then
    python3 ../scripts/gen_data.py "${GEN_CASE_ARGS[@]}" "${EXTRA_DATA_ARGS[@]}" --causal-mask "${CAUSAL_MASK:-0}"
    time ./gqa_performance_dn --npu="${NPU_ID}" --cases="${CASES_RAW}" "${EXTRA_BIN_ARGS[@]}"
else
    python3 ../scripts/gen_data.py "${GEN_CASE_ARGS[@]}" "${EXTRA_DATA_ARGS[@]}" --causal-mask "${CAUSAL_MASK:-0}"
    time ./gqa_performance_dn --npu="${NPU_ID}" "${EXTRA_BIN_ARGS[@]}"
fi
