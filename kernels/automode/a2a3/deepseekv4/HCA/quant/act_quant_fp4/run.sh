#!/bin/bash
# act_quant_fp4 — block-wise FP4 quant inplace (auto-mode A3, vector).
# Single-case-per-binary; shape from ../scripts/generate_cases.py.
# Usage:
#   bash run.sh -r npu -v Ascend910B1
#   bash run.sh -r npu -v Ascend910B1 -a "32,128,32,1"
# Case format: M,N,BLOCK_SIZE,INPLACE   (FP4 default BLOCK_SIZE=32)

KERNEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SHORT=r:,v:,C:,n:,a:,p
LONG=run-mode:,soc-version:,compiler:,npu:,cases:,profile
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
PROFILE_MODE=0
while :
do
    case "$1" in
        (-r | --run-mode )    RUN_MODE="$2"        ; shift 2;;
        (-v | --soc-version ) SOC_VERSION="$2"     ; shift 2;;
        (-C | --compiler )    CMAKE_COMPILER="$2"  ; shift 2;;
        (-n | --npu )         NPU_ID="$2"          ; shift 2;;
        (-a | --cases )       CASES_RAW="$2"       ; shift 2;;
        (-p | --profile )     PROFILE_MODE=1       ; shift 1;;
        (--)                  shift; break;;
        (*)                   echo "[ERROR] Unexpected option: $1"; break;;
    esac
done

: "${CMAKE_COMPILER:=bisheng}"

source "${KERNEL_DIR}/../../../../common.sh"

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi

if [[ "${SOC_VERSION}" =~ ^Ascend910B4-1 ]] && [ "${RUN_MODE}" == "sim" ]; then
    echo "[ERROR] SocVersion: ${SOC_VERSION} can not support sim mode, please use Ascend910B4."
    exit 1
fi

: "${NPU_ID:=0}"

# FP4 default block size is 32 — if user did not pass --cases, force the
# generator into the FP4-appropriate default by overriding via --cases.
GEN_CASE_ARGS=()
if [[ -n "${CASES_RAW:-}" ]]; then
    GEN_CASE_ARGS+=(--cases "${CASES_RAW}")
else
    GEN_CASE_ARGS+=(--cases "32,128,32,1")
fi
python3 ../scripts/generate_cases.py "${GEN_CASE_ARGS[@]}" || exit 1

python ./scripts/gen_data.py || exit 1

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
set -euo pipefail

cmake  -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} -DCMAKE_COMPILER=${CMAKE_COMPILER} ..
make -j16

run_bin ./act_quant_fp4
