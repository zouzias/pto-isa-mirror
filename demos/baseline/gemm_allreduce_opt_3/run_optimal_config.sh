#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# --------------------------------------------------------------------------------
# 运行 benchmark_optimal_config：测单核通信/计算效率并写回最优核数配置。
# 需先执行 run.sh 或 run_performance_test.sh 完成编译。
#

set -euo pipefail
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /home/ntlab/qifeng/pypto/third_party_path/shmem/install/set_env.sh
[ -n "${CONDA_PREFIX:-}" ] && export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${CONDA_PREFIX}/aarch64-conda-linux-gnu/lib:${LD_LIBRARY_PATH:-}
[ -d "${HOME}/miniconda3/envs/pypto_haoran/lib" ] && export LD_LIBRARY_PATH=${HOME}/miniconda3/envs/pypto_haoran/lib:${LD_LIBRARY_PATH:-}

NRANKS=8
FIRST_DEVICE=0
TOTAL_CORES=24
SKIP_BENCHMARK=false

SHORT=n:,d:,c:,s
LONG=nranks:,first-device:,total-cores:,skip-benchmark
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@") || true
eval set -- "${OPTS:-}"
while true; do
    case "$1" in
        -n|--nranks) NRANKS="$2"; shift 2 ;;
        -d|--first-device) FIRST_DEVICE="$2"; shift 2 ;;
        -c|--total-cores) TOTAL_CORES="$2"; shift 2 ;;
        -s|--skip-benchmark) SKIP_BENCHMARK=true; shift ;;
        --) shift; break ;;
        *) shift ;;
    esac
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
[ ! -d build ] && { echo "[ERROR] No build dir. Run ./run.sh first."; exit 1; }

CMD="./benchmark_optimal_config --nranks $NRANKS --first-device $FIRST_DEVICE --total-cores $TOTAL_CORES"
[ "$SKIP_BENCHMARK" = true ] && CMD="$CMD --skip-benchmark"

echo "=== 最优核配置测试 ==="
echo "  $CMD"
echo "==============================="
cd build
$CMD
