#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
SOC_VERSION="${SOC_VERSION:-ascend910_93}"
BUILD_JOBS="${BUILD_JOBS:-16}"
RUN_MATRIX=0
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --soc)
            SOC_VERSION="$2"
            shift 2
            ;;
        --matrix)
            RUN_MATRIX=1
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DSOC_VERSION="${SOC_VERSION}"
cmake --build "${BUILD_DIR}" --target dispatch_combine_moe_v2_dispatch_combine_kernel dispatch_combine_moe_v2_compute_vec_kernel dispatch_combine_moe_v2_comm_vec_queue_kernel dispatch_combine_moe_v2_compute_cube_kernel dispatch_combine_moe_v2_compute_cube_queue_kernel dispatch_combine_moe_v2 -j"${BUILD_JOBS}"
export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"

if [[ "${RUN_MATRIX}" -eq 1 ]]; then
    export DISPATCH_COMBINE_MOE_V2_WARMUP_ITERS="${DISPATCH_COMBINE_MOE_V2_WARMUP_ITERS:-0}"
    export DISPATCH_COMBINE_MOE_V2_MEASURE_ITERS="${DISPATCH_COMBINE_MOE_V2_MEASURE_ITERS:-1}"
    mpirun -n 2 "${BUILD_DIR}/dispatch_combine_moe_v2" "${ARGS[@]}"
    mpirun -n 2 "${BUILD_DIR}/dispatch_combine_moe_v2" --m 16 --max-output-size 32 "${ARGS[@]}"
    mpirun -n 2 "${BUILD_DIR}/dispatch_combine_moe_v2" --m 4097 --max-output-size 32 "${ARGS[@]}"
    exit 0
fi

exec "${BUILD_DIR}/dispatch_combine_moe_v2" "${ARGS[@]}"
