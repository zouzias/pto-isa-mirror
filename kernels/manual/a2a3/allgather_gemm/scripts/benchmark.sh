#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Performance benchmark: PTO AllGather-GEMM vs SHMEM AllGather-MatMul
#
# Usage:
#   bash scripts/benchmark.sh <device_list> [options]
#
# Examples:
#   bash scripts/benchmark.sh 6,7                          # Run all shapes
#   bash scripts/benchmark.sh 6,7 --shapes small           # Only small shapes
#   bash scripts/benchmark.sh 6,7 --shapes medium+large    # Medium + large
#   bash scripts/benchmark.sh 6,7 --skip-shmem             # Skip SHMEM build/run
#   bash scripts/benchmark.sh 6,7 --skip-pto               # Skip PTO build/run
#   bash scripts/benchmark.sh 6,7 -v Ascend910B3           # Specify SoC
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

cleanup_on_exit() {
    pkill -9 -f allgather_gemm 2>/dev/null || true
    pkill -9 -f allgather_matmu 2>/dev/null || true
}
trap cleanup_on_exit EXIT

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PTO_PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SHMEM_PROJECT_ROOT="/mnt/data/ntlab/qifeng/shmem_test/shmem"
SHMEM_EXAMPLE_DIR="${SHMEM_PROJECT_ROOT}/examples/allgather_matmul"

RESULT_DIR="${PTO_PROJECT_DIR}/benchmark_results"
mkdir -p "${RESULT_DIR}"

DEVICE_LIST_ARG="${1:-}"
if [ -z "${DEVICE_LIST_ARG}" ]; then
    echo "[ERROR] Usage: bash scripts/benchmark.sh <device_list> [options]"
    echo "        Example: bash scripts/benchmark.sh 6,7"
    exit 1
fi
shift

IFS=',' read -ra DEVICE_ID_LIST <<< "${DEVICE_LIST_ARG}"
PE_SIZE=${#DEVICE_ID_LIST[@]}

SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
RUN_MODE="${RUN_MODE:-npu}"
SKIP_SHMEM=0
SKIP_PTO=0
SHAPE_FILTER="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--soc-version)   SOC_VERSION="$2"; shift 2 ;;
        -r|--run-mode)      RUN_MODE="$2"; shift 2 ;;
        --skip-shmem)       SKIP_SHMEM=1; shift ;;
        --skip-pto)         SKIP_PTO=1; shift ;;
        --shapes)           SHAPE_FILTER="$2"; shift 2 ;;
        *)                  echo "[WARN] Unknown option: $1"; shift ;;
    esac
done

# Peak TFLOPS based on SoC variant (FP16 Cube)
if [ -z "${PEAK_TFLOPS_FP16:-}" ]; then
    case "${SOC_VERSION}" in
        Ascend910B1|Ascend910_939*)    PEAK_TFLOPS_FP16=320.0 ;;
        Ascend910B2*|Ascend910_938*)   PEAK_TFLOPS_FP16=311.0 ;;
        Ascend910B3|Ascend910_937*)    PEAK_TFLOPS_FP16=259.0 ;;
        Ascend910B4*|Ascend910_936*)   PEAK_TFLOPS_FP16=216.0 ;;
        *)                             PEAK_TFLOPS_FP16=320.0 ;;
    esac
fi

# Block config
if [[ "${SOC_VERSION}" =~ ^Ascend910B[34] ]] || [[ "${SOC_VERSION}" =~ ^Ascend910_93[67] ]]; then
    COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM:-20}
    COMM_BLOCK_NUM=${COMM_BLOCK_NUM:-40}
else
    COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM:-24}
    COMM_BLOCK_NUM=${COMM_BLOCK_NUM:-48}
fi

# CANN environment
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    CANN_ENV_PATHS=(
        /usr/local/Ascend/latest/set_env.sh
        /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
    )
    for p in "${CANN_ENV_PATHS[@]}"; do
        if [ -f "$p" ]; then
            echo "[INFO] Sourcing CANN env: $p"
            source "$p"
            break
        fi
    done
fi
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "[ERROR] ASCEND_HOME_PATH not set."
    exit 1
fi
echo "[INFO] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

# MPI
MPI_BIN=""
MPI_SEARCH_PATHS=( /usr/local/mpich/bin/mpirun /home/mpich/bin/mpirun /usr/local/bin/mpirun /usr/bin/mpirun )
for p in "${MPI_SEARCH_PATHS[@]}"; do
    if [ -f "$p" ]; then
        MPI_BIN="$p"
        MPI_LIB_DIR="$(dirname "$(dirname "$p")")/lib"
        export MPI_LIB_PATH="${MPI_LIB_PATH:-${MPI_LIB_DIR}/libmpi.so}"
        break
    fi
done
[ -z "$MPI_BIN" ] && command -v mpirun &>/dev/null && MPI_BIN="mpirun"
if [ -z "$MPI_BIN" ]; then
    echo "[ERROR] mpirun not found."
    exit 1
fi

# ============================================================================
# Matrix shape definitions
# ============================================================================
SMALL_SHAPES=("1024,1024,512" "1024,2048,1024")
MEDIUM_SHAPES=("2048,2048,1024" "4096,4096,2048")
LARGE_SHAPES=("8192,4096,4096" "16384,4096,4096")

SHAPES_TO_TEST=()
case "${SHAPE_FILTER}" in
    all)
        SHAPES_TO_TEST=("${SMALL_SHAPES[@]}" "${MEDIUM_SHAPES[@]}" "${LARGE_SHAPES[@]}") ;;
    small)
        SHAPES_TO_TEST=("${SMALL_SHAPES[@]}") ;;
    medium)
        SHAPES_TO_TEST=("${MEDIUM_SHAPES[@]}") ;;
    large)
        SHAPES_TO_TEST=("${LARGE_SHAPES[@]}") ;;
    *)
        IFS='+' read -ra GROUPS <<< "${SHAPE_FILTER}"
        for g in "${GROUPS[@]}"; do
            case "$g" in
                small)  SHAPES_TO_TEST+=("${SMALL_SHAPES[@]}") ;;
                medium) SHAPES_TO_TEST+=("${MEDIUM_SHAPES[@]}") ;;
                large)  SHAPES_TO_TEST+=("${LARGE_SHAPES[@]}") ;;
                *)      echo "[WARN] Unknown shape group: $g" ;;
            esac
        done
        ;;
esac

BASE_M=128; BASE_N=256

CSV_FILE="${RESULT_DIR}/benchmark_data.csv"
echo "shape_label,M,K,N,pe_size,operator,mode,time_us,tflops,comm_bw_gbps,comm_time_us,compute_time_us,overlap_pct" > "${CSV_FILE}"

timestamp=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${RESULT_DIR}/logs_${timestamp}"
mkdir -p "${LOG_DIR}"

echo ""
echo "================================================================"
echo "  AllGather-GEMM Performance Benchmark"
echo "  PTO vs SHMEM/Catcoc"
echo "================================================================"
echo "  SoC:        ${SOC_VERSION}"
echo "  PE_SIZE:    ${PE_SIZE}"
echo "  Devices:    ${DEVICE_LIST_ARG}"
echo "  Shapes:     ${#SHAPES_TO_TEST[@]} configurations"
echo "  Results:    ${RESULT_DIR}"
echo "================================================================"
echo ""

# ============================================================================
# Build SHMEM (once)
# ============================================================================
build_shmem() {
    if [ ${SKIP_SHMEM} -eq 1 ]; then
        echo "[INFO] Skipping SHMEM build (--skip-shmem)"
        return 0
    fi

    echo "[INFO] Building SHMEM allgather_matmul (with perf instrumentation)..."

    # Backup original main.cpp if main_perf.cpp exists
    if [ -f "${SHMEM_EXAMPLE_DIR}/main_perf.cpp" ]; then
        cp "${SHMEM_EXAMPLE_DIR}/main.cpp" "${SHMEM_EXAMPLE_DIR}/main_orig.cpp.bak"
        cp "${SHMEM_EXAMPLE_DIR}/main_perf.cpp" "${SHMEM_EXAMPLE_DIR}/main.cpp"
    fi

    cd "${SHMEM_PROJECT_ROOT}"
    bash scripts/build.sh -examples 2>&1 | tail -10
    if [ ! -f "${SHMEM_PROJECT_ROOT}/build/bin/allgather_matmul" ]; then
        echo "[ERROR] SHMEM build failed - binary not found"
        # Restore original
        if [ -f "${SHMEM_EXAMPLE_DIR}/main_orig.cpp.bak" ]; then
            mv "${SHMEM_EXAMPLE_DIR}/main_orig.cpp.bak" "${SHMEM_EXAMPLE_DIR}/main.cpp"
        fi
        return 1
    fi

    # Restore original main.cpp
    if [ -f "${SHMEM_EXAMPLE_DIR}/main_orig.cpp.bak" ]; then
        mv "${SHMEM_EXAMPLE_DIR}/main_orig.cpp.bak" "${SHMEM_EXAMPLE_DIR}/main.cpp"
    fi

    echo "[INFO] SHMEM build OK"
    cd "${PTO_PROJECT_DIR}"
}

# ============================================================================
# Run PTO for one shape
# ============================================================================
run_pto_shape() {
    local M=$1 K=$2 N=$3 LABEL=$4

    local ORIG_M=${M} ORIG_K=${K} ORIG_N=${N}
    local M_ALIGN=$(( BASE_M * PE_SIZE ))
    M=$(( ((ORIG_M + M_ALIGN - 1) / M_ALIGN) * M_ALIGN ))
    K=$(( ((ORIG_K + BASE_N - 1) / BASE_N) * BASE_N ))
    N=$(( ((ORIG_N + BASE_N - 1) / BASE_N) * BASE_N ))

    echo "[PTO] Building M=${ORIG_M}->${M}, K=${ORIG_K}->${K}, N=${ORIG_N}->${N}"

    local DATA_DIR="${PTO_PROJECT_DIR}/out"
    mkdir -p "${DATA_DIR}"
    rm -rf "${DATA_DIR}"/*.bin

    python3 "${PTO_PROJECT_DIR}/scripts/gen_data.py" \
        --n-ranks ${PE_SIZE} \
        --m ${ORIG_M} --k ${ORIG_K} --n ${ORIG_N} \
        --padded-m ${M} --padded-k ${K} --padded-n ${N} \
        --output-dir "${DATA_DIR}" 2>&1 | tail -3

    local SHMEM_INPUT_MB=$(( (M * K * 2 + 1048575) / 1048576 ))
    local REQUIRED_MB=$(( SHMEM_INPUT_MB + 64 ))
    REQUIRED_MB=$(( ((REQUIRED_MB + 255) / 256) * 256 ))
    export HCCL_BUFFSIZE=${REQUIRED_MB}

    rm -rf "${PTO_PROJECT_DIR}/build"
    mkdir -p "${PTO_PROJECT_DIR}/build"
    cd "${PTO_PROJECT_DIR}/build"

    export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${ASCEND_HOME_PATH}/lib64:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"

    local CMAKE_RUN_MODE="${RUN_MODE}"
    [ "${RUN_MODE}" == "run" ] && CMAKE_RUN_MODE="npu"

    unset CXXFLAGS CFLAGS LDFLAGS 2>/dev/null || true

    CC=bisheng CXX=bisheng cmake -DRUN_MODE=${CMAKE_RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
          -DG_M=${M} -DG_K=${K} -DG_N=${N} \
          -DORIG_M=${ORIG_M} -DORIG_K=${ORIG_K} -DORIG_N=${ORIG_N} \
          -DCOMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM} \
          -DCOMM_BLOCK_NUM=${COMM_BLOCK_NUM} \
          -DPEAK_TFLOPS_FP16=${PEAK_TFLOPS_FP16} .. > /dev/null 2>&1

    if ! make -j16 > /dev/null 2>&1; then
        echo "[ERROR] PTO build failed for ${LABEL}"
        cd "${PTO_PROJECT_DIR}"
        return 1
    fi
    cd "${PTO_PROJECT_DIR}"

    local EXEC_BIN="${PTO_PROJECT_DIR}/build/allgather_gemm"
    export N_RANKS=${PE_SIZE}
    export ALLGATHER_GEMM_PERF_MODE=1
    export ALLGATHER_GEMM_DATA_DIR="${DATA_DIR}"

    local LOG_FILE="${LOG_DIR}/pto_${LABEL}.log"

    local PTO_TIMEOUT=${PTO_TIMEOUT:-600}
    echo "[PTO] Running performance test (timeout ${PTO_TIMEOUT}s)..."
    timeout --signal=KILL ${PTO_TIMEOUT}s ${MPI_BIN} -n ${PE_SIZE} ${EXEC_BIN} 2>&1 | tee "${LOG_FILE}"
    local PTO_RET=$?

    if [ ${PTO_RET} -ne 0 ]; then
        pkill -9 -f allgather_gemm 2>/dev/null || true
        sleep 1
    fi

    parse_pto_results "${LOG_FILE}" "${LABEL}" "${ORIG_M}" "${ORIG_K}" "${ORIG_N}"
}

# ============================================================================
# Parse PTO results from log
# ============================================================================
parse_pto_results() {
    local LOG_FILE=$1 LABEL=$2 ORIG_M=$3 ORIG_K=$4 ORIG_N=$5

    local AVG_TIME_MS=$(grep '\[PERF\] op=hccl_ag_gemm' "${LOG_FILE}" 2>/dev/null | head -1 | sed 's/.*avg_time=\([0-9.]*\)ms.*/\1/')
    if [ -z "${AVG_TIME_MS}" ]; then
        echo "[WARN] Could not parse PTO results for ${LABEL} (missing op=hccl_ag_gemm avg_time= in ${LOG_FILE}; check timeout or MPI log prefix)."
        return
    fi

    local AVG_TIME_US=$(python3 -c "print(${AVG_TIME_MS} * 1000)")
    local TFLOPS=$(grep 'TFLOPS:' "${LOG_FILE}" | head -1 | awk '{print $2}')
    local COMM_BW=$(grep 'Comm BW:' "${LOG_FILE}" | head -1 | awk '{print $3}')

    local GEMM_FLOPS COMM_BYTES PURE_COMPUTE_US PURE_COMM_US SEQ_TOTAL_US OVERLAP_PCT
    read -r GEMM_FLOPS COMM_BYTES PURE_COMPUTE_US PURE_COMM_US SEQ_TOTAL_US OVERLAP_PCT <<< $(python3 -c "
m, k, n, pe = ${ORIG_M}, ${ORIG_K}, ${ORIG_N}, ${PE_SIZE}
peak = ${PEAK_TFLOPS_FP16}
fused = ${AVG_TIME_US}
gemm_flops = 2.0 * m * k * n
comm_bytes = (m / pe) * k * 2 * (pe - 1)
pure_comp = gemm_flops / (peak * 1e12) * 1e6
pure_comm = comm_bytes / (50e9) * 1e6
seq = pure_comp + pure_comm
overlap = max(0, (1 - fused / seq) * 100) if seq > 0 else 0
print(f'{gemm_flops} {comm_bytes} {pure_comp} {pure_comm} {seq} {overlap:.1f}')
")

    echo "${LABEL},${ORIG_M},${ORIG_K},${ORIG_N},${PE_SIZE},PTO,fused,${AVG_TIME_US},${TFLOPS:-0},${COMM_BW:-0},0,0,${OVERLAP_PCT}" >> "${CSV_FILE}"
    echo "${LABEL},${ORIG_M},${ORIG_K},${ORIG_N},${PE_SIZE},PTO,sequential,${SEQ_TOTAL_US},0,0,${PURE_COMM_US},${PURE_COMPUTE_US},0" >> "${CSV_FILE}"

    echo "[PTO] ${LABEL}: fused=${AVG_TIME_US} us, TFLOPS=${TFLOPS}, CommBW=${COMM_BW} GB/s, overlap=${OVERLAP_PCT}%"
}

# ============================================================================
# Run SHMEM for one shape
# ============================================================================
run_shmem_shape() {
    local M=$1 K=$2 N=$3 LABEL=$4

    local SHMEM_BIN="${SHMEM_PROJECT_ROOT}/build/bin/allgather_matmul"
    if [ ! -f "${SHMEM_BIN}" ]; then
        echo "[WARN] SHMEM binary not found, skipping"
        return 0
    fi

    local DATA_DIR="${SHMEM_EXAMPLE_DIR}/out"
    mkdir -p "${DATA_DIR}"
    rm -rf "${DATA_DIR}"/*.bin

    local SHMEM_UTILS="${SHMEM_PROJECT_ROOT}/examples/utils"

    echo "[SHMEM] Generating data M=${M}, K=${K}, N=${N}..."
    cd "${SHMEM_EXAMPLE_DIR}"

    source "${SHMEM_PROJECT_ROOT}/install/set_env.sh" 2>/dev/null || true

    python3 "${SHMEM_UTILS}/gen_data.py" 1 1 ${PE_SIZE} ${M} ${N} ${K} 0 0 "${DATA_DIR}" 2>&1 | tail -3

    local IPPORT="tcp://127.0.0.1:8899"
    export SHMEM_UID_SESSION_ID=127.0.0.1:8899

    local LOG_FILE="${LOG_DIR}/shmem_${LABEL}.log"

    echo "[SHMEM] Running performance test with aclrtEvent timing..."

    # Launch all PE processes; only pe_id=0 prints perf results
    local PIDS=()
    for (( idx = 0; idx < PE_SIZE; idx++ )); do
        if [ ${idx} -eq 0 ]; then
            ${SHMEM_BIN} "${PE_SIZE}" "${idx}" "${IPPORT}" "${M}" "${N}" "${K}" "${DATA_DIR}" "${DEVICE_LIST_ARG}" --perf 2>&1 | tee "${LOG_FILE}" &
        else
            ${SHMEM_BIN} "${PE_SIZE}" "${idx}" "${IPPORT}" "${M}" "${N}" "${K}" "${DATA_DIR}" "${DEVICE_LIST_ARG}" --perf > /dev/null 2>&1 &
        fi
        PIDS+=($!)
    done

    local SHMEM_TIMEOUT=${SHMEM_TIMEOUT:-300}
    local ALL_OK=1
    local WAIT_START=$(date +%s)
    for pid in "${PIDS[@]}"; do
        local ELAPSED=$(( $(date +%s) - WAIT_START ))
        local REMAINING=$(( SHMEM_TIMEOUT - ELAPSED ))
        if [ ${REMAINING} -le 0 ]; then
            ALL_OK=0
            break
        fi
        if ! timeout ${REMAINING}s tail --pid=$pid -f /dev/null 2>/dev/null; then
            ALL_OK=0
        fi
        if ! wait $pid 2>/dev/null; then ALL_OK=0; fi
    done

    if [ ${ALL_OK} -eq 0 ]; then
        pkill -9 -f allgather_matmu 2>/dev/null || true
        sleep 1
    fi

    cd "${PTO_PROJECT_DIR}"

    if [ ${ALL_OK} -eq 0 ]; then
        echo "[WARN] SHMEM run failed for ${LABEL}"
        return 0
    fi

    parse_shmem_results "${LOG_FILE}" "${LABEL}" "${M}" "${K}" "${N}"
}

# ============================================================================
# Parse SHMEM results from log
# ============================================================================
parse_shmem_results() {
    local LOG_FILE=$1 LABEL=$2 ORIG_M=$3 ORIG_K=$4 ORIG_N=$5

    local AVG_TIME_MS=$(grep '\[PERF\] op=shmem_ag_matmul' "${LOG_FILE}" 2>/dev/null | head -1 | sed 's/.*avg_time=\([0-9.]*\)ms.*/\1/')
    if [ -z "${AVG_TIME_MS}" ]; then
        echo "[WARN] Could not parse SHMEM results for ${LABEL}"
        return
    fi

    local AVG_TIME_US=$(python3 -c "print(${AVG_TIME_MS} * 1000)")
    local TFLOPS=$(grep 'TFLOPS:' "${LOG_FILE}" | head -1 | awk '{print $2}')
    local COMM_BW=$(grep 'Comm BW:' "${LOG_FILE}" | head -1 | awk '{print $3}')

    local GEMM_FLOPS COMM_BYTES PURE_COMPUTE_US PURE_COMM_US SEQ_TOTAL_US OVERLAP_PCT
    read -r GEMM_FLOPS COMM_BYTES PURE_COMPUTE_US PURE_COMM_US SEQ_TOTAL_US OVERLAP_PCT <<< $(python3 -c "
m, k, n, pe = ${ORIG_M}, ${ORIG_K}, ${ORIG_N}, ${PE_SIZE}
peak = ${PEAK_TFLOPS_FP16}
fused = ${AVG_TIME_US}
g_m = m * pe
gemm_flops = 2.0 * g_m * k * n
comm_bytes = m * k * 2 * (pe - 1)
pure_comp = gemm_flops / (peak * 1e12) * 1e6
pure_comm = comm_bytes / (50e9) * 1e6
seq = pure_comp + pure_comm
overlap = max(0, (1 - fused / seq) * 100) if seq > 0 else 0
print(f'{gemm_flops} {comm_bytes} {pure_comp} {pure_comm} {seq} {overlap:.1f}')
")

    echo "${LABEL},${ORIG_M},${ORIG_K},${ORIG_N},${PE_SIZE},SHMEM,fused,${AVG_TIME_US},${TFLOPS:-0},${COMM_BW:-0},0,0,${OVERLAP_PCT}" >> "${CSV_FILE}"

    echo "[SHMEM] ${LABEL}: fused=${AVG_TIME_US} us, TFLOPS=${TFLOPS}, CommBW=${COMM_BW} GB/s, overlap=${OVERLAP_PCT}%"
}

# ============================================================================
# Main benchmark loop
# ============================================================================

if [ ${SKIP_SHMEM} -eq 0 ]; then
    build_shmem || SKIP_SHMEM=1
fi

for shape in "${SHAPES_TO_TEST[@]}"; do
    IFS=',' read -r M K N <<< "${shape}"
    LABEL="${M}x${K}x${N}"

    echo ""
    echo "================================================================"
    echo "  Shape: M=${M}, K=${K}, N=${N}, PE_SIZE=${PE_SIZE}"
    echo "================================================================"

    if [ ${SKIP_PTO} -eq 0 ]; then
        run_pto_shape "${M}" "${K}" "${N}" "${LABEL}" || true
    fi

    if [ ${SKIP_SHMEM} -eq 0 ]; then
        run_shmem_shape "${M}" "${K}" "${N}" "${LABEL}" || true
    fi
done

echo ""
echo "================================================================"
echo "  Benchmark Complete!"
echo "  Results saved to: ${CSV_FILE}"
echo "  Logs saved to: ${LOG_DIR}"
echo "================================================================"
echo ""

# Auto-generate charts
if command -v python3 &>/dev/null; then
    echo "[INFO] Generating charts..."
    python3 "${SCRIPT_DIR}/plot_benchmark.py" "${CSV_FILE}" -o "${RESULT_DIR}/charts_${timestamp}"
fi
