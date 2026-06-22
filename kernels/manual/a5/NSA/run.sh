#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

SHORT=r:,v:,n:,c:,a:,p:,m:,i,d,k
LONG=run-mode:,soc-version:,npu:,case:,cases:,qk-preload:,mode:,intermediate,debug,mask,mode_dn
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :
do
    case "$1" in
        (-r | --run-mode ) RUN_MODE="$2"; shift 2;;
        (-v | --soc-version ) SOC_VERSION="$2"; shift 2;;
        (-n | --npu ) NPU_ID="$2"; shift 2;;
        (-c | --case ) CASE_FILTER="$2"; shift 2;;
        (-a | --cases ) CASES_RAW="$2"; shift 2;;
        (-p | --qk-preload ) QK_PRELOAD="$2"; shift 2;;
        (-m | --mode ) FIFO_MODE="$2"; shift 2;;
        (-i | --intermediate ) INTERMEDIATE=1; shift 1;;
        (-d | --debug ) DEBUG_BUILD=1; shift 1;;
        (--mode_dn ) MODE_DN=1; shift 1;;
        (-k | --mask ) CAUSAL_MASK=1; shift 1;;
        (--) shift; break;;
        (*) echo "[ERROR] Unexpected option: $1"; break;;
    esac
done

pattern="^Ascend950PR_9599"
if [[ ! "$SOC_VERSION" =~ $pattern ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}, this folder only support A5."
    exit 1
fi

rm -rf build && mkdir build && cd build
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
set -euo pipefail

: "${NPU_ID:=0}"
: "${QK_PRELOAD:=2}"
: "${FIFO_MODE:=1}"

GEN_CASE_ARGS=()
if [[ -n "${CASES_RAW:-}" ]]; then
    IFS=';' read -ra CASE_ENTRIES <<< "${CASES_RAW}"
    for entry in "${CASE_ENTRIES[@]}"; do
        GEN_CASE_ARGS+=(--cases "$entry")
    done
elif [[ -n "${CASE_FILTER:-}" && "${CASE_FILTER}" == *","* ]]; then
    GEN_CASE_ARGS+=(--cases "${CASE_FILTER}")
fi

python3 ../scripts/generate_cases.py --qk-preload "${QK_PRELOAD}" "${GEN_CASE_ARGS[@]}" --causal-mask "${CAUSAL_MASK:-0}"
python3 ../scripts/validate_buffer_usage.py --mode dn --cases generated_cases.json

CMAKE_EXTRA=(-DMODE_DN=ON -DFIFO_MODE=${FIFO_MODE})
[[ -n "${DEBUG_BUILD:-}" ]] && CMAKE_EXTRA+=(-DDEBUG_MODE=ON)

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} "${CMAKE_EXTRA[@]}" ..
make -j16

EXTRA_BIN_ARGS=(--sys_cnt_multiple=1.0)
python3 ../scripts/gen_data.py "${GEN_CASE_ARGS[@]}" --causal-mask "${CAUSAL_MASK:-0}"
time ./nsa_performance_dn --npu="${NPU_ID}" "${EXTRA_BIN_ARGS[@]}"
