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


DEBUG_MODE="OFF"
WARMUP_ITERS=5
PERF_ITERS=20
BENCH_MODE="--both"   # --both | --overlap-only | --no-overlap-only

SHORT=r:,v:,d,w:,i:,
LONG=run-mode:,soc-version:,debug,warmup:,iters:,no-verify,both,overlap-only,no-overlap-only,quiet,
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
        (-d | --debug )
            DEBUG_MODE="ON"
            shift;;
        (-w | --warmup )
            WARMUP_ITERS="$2"
            shift 2;;
        (-i | --iters )
            PERF_ITERS="$2"
            shift 2;;
        (--no-verify )
            NO_VERIFY="--no-verify"
            shift;;
        (--both )
            BENCH_MODE="--both"
            shift;;
        (--overlap-only )
            BENCH_MODE="--overlap-only"
            shift;;
        (--no-overlap-only )
            BENCH_MODE="--no-overlap-only"
            shift;;
        (--quiet )
            QUIET="-q"
            shift;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

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

cmake  -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} -DDEBUG_MODE=${DEBUG_MODE} ..
make -j16

echo "============================================================"
echo "  Running GEMM + AllReduce Performance Benchmark"
echo "  Warmup: ${WARMUP_ITERS}, Measure: ${PERF_ITERS}"
echo "  Mode:   ${BENCH_MODE}"
echo "============================================================"

./gemm_ar_performance -w ${WARMUP_ITERS} -i ${PERF_ITERS} ${BENCH_MODE} ${NO_VERIFY:-} ${QUIET:-}
