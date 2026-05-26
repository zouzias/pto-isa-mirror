#!/bin/bash
# --------------------------------------------------------------------------------
# add_tile_array - run.sh
# Mirrors kernels/manual/a2a3/topk/run.sh exactly so the same invocation works:
#   bash run.sh -r npu -v Ascend910B1
# (or -r sim for the simulator).
# --------------------------------------------------------------------------------

SHORT=r:,v:,C:,a:,
LONG=run-mode:,soc-version:,compiler:,cases:,msopprof,
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
        (-C | --compiler )
            CMAKE_COMPILER="$2"
            shift 2;;
        (-a | --cases )
            CASES_RAW="$2"
            shift 2;;
        (--msopprof )
            MSOPPROF=1
            shift 1;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

run_msopprof() {
    if [[ "${MSOPPROF:-0}" == "1" ]]; then
        mkdir -p msopprof_data
        msopprof --output=msopprof_data "$@"
    else
        "$@"
    fi
}

: "${CMAKE_COMPILER:=bisheng}"

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi

if [[ "${SOC_VERSION}" =~ ^Ascend910B4-1 ]] && [ "${RUN_MODE}" == "sim" ]; then
    echo "[ERROR] SocVersion: ${SOC_VERSION} can not support sim mode, please use Ascend910B4."
    exit 1
fi

rm -rf build
mkdir build

GEN_CASE_ARGS=()
if [[ -n "${CASES_RAW:-}" ]]; then
    GEN_CASE_ARGS+=(--cases "${CASES_RAW}")
fi

python3 ./scripts/generate_cases.py "${GEN_CASE_ARGS[@]}"
python3 ./scripts/gen_data.py

cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
set -euo pipefail

cmake  -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} -DCMAKE_COMPILER=${CMAKE_COMPILER} ..
make -j16

run_msopprof ./add_tile_array
