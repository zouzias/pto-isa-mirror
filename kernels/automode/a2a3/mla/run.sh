#!/bin/bash
# --------------------------------------------------------------------------------
# mla_basic - run.sh.
# Mirrors kernels/automode/a2a3/moe_segmented_ffn_top1/run.sh.
#   bash run.sh -r npu -v Ascend910B1
# (or -r sim for the simulator).
# --------------------------------------------------------------------------------

KERNEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python ./scripts/gen_data.py

SHORT=r:,v:,C:,p
LONG=run-mode:,soc-version:,compiler:,profile
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
PROFILE_MODE=0
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
        (-p | --profile )
            PROFILE_MODE=1
            shift 1;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

: "${CMAKE_COMPILER:=bisheng}"

source "${KERNEL_DIR}/../common.sh"

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
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
set -euo pipefail

cmake  -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} -DCMAKE_COMPILER=${CMAKE_COMPILER} ..
make -j16

run_bin ./mla_basic
echo ""
echo "[run.sh] --- Python per-stage comparison ---"
python ../scripts/compare_outputs.py
