#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || realpath "${SCRIPT_DIR}/../../../../../../..")"
cd "${REPO_ROOT}"

if [ "$#" -eq 0 ]; then
    RUN_ST_ARGS=(-r sim -v a5)
else
    RUN_ST_ARGS=("$@")
fi

LOG_DIR="${TQUANT_EXP2D_LOG_DIR:-${REPO_ROOT}/tests/npu/a5/src/st/testcase/tquant/logs/exp2d_fuzz_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_FILE="${LOG_DIR}/summary.txt"
mkdir -p "${LOG_DIR}"

CASES=(
    TQUANTTEST.case_mxfp8_bf16_17x64_static17x64_exp2d_fuzz01_nd
    TQUANTTEST.case_mxfp8_bf16_17x64_static19x128_exp2d_fuzz02_nd
    TQUANTTEST.case_mxfp8_bf16_9x128_static11x192_exp2d_fuzz03_nd
    TQUANTTEST.case_mxfp8_bf16_17x64_static19x256_exp2d_fuzz04_nd
    TQUANTTEST.case_mxfp8_bf16_13x256_static23x320_exp2d_fuzz05_nd
    TQUANTTEST.case_mxfp8_bf16_29x448_static31x512_exp2d_fuzz06_nd
    TQUANTTEST.case_mxfp8_bf16_31x640_static37x768_exp2d_fuzz07_nd
    TQUANTTEST.case_mxfp8_bf16_39x960_static41x1024_exp2d_fuzz08_nd
    TQUANTTEST.case_mxfp8_bf16_5x1984_static7x2048_exp2d_fuzz09_nd
    TQUANTTEST.case_mxfp8_bf16_3x4032_static3x4096_exp2d_fuzz10_nd
    TQUANTTEST.case_mxfp8_bf16_1x8192_static2x8192_exp2d_fuzz11_nd
    TQUANTTEST.case_mxfp8_bf16_113x64_static127x64_exp2d_fuzz12_nd
    TQUANTTEST.case_mxfp8_bf16_503x64_static509x64_exp2d_fuzz13_nd
    TQUANTTEST.case_mxfp8_bf16_251x64_static257x128_exp2d_fuzz14_nd
    TQUANTTEST.case_mxfp8_bf16_127x192_static129x256_exp2d_fuzz15_nd
    TQUANTTEST.case_mxfp8_bf16_93x64_static95x512_exp2d_fuzz16_nd
    TQUANTTEST.case_mxfp8_bf16_67x704_static71x768_exp2d_fuzz17_nd
    TQUANTTEST.case_mxfp8_bf16_61x128_static63x1024_exp2d_fuzz18_nd
    TQUANTTEST.case_mxfp8_bf16_15x1472_static17x1536_exp2d_fuzz19_nd
    TQUANTTEST.case_mxfp8_bf16_31x1856_static33x2048_exp2d_fuzz20_nd
    TQUANTTEST.case_mxfp8_fp16_17x64_static17x64_exp2d_fuzz21_nd
    TQUANTTEST.case_mxfp8_fp16_17x128_static19x192_exp2d_fuzz22_nd
    TQUANTTEST.case_mxfp8_fp16_13x192_static15x256_exp2d_fuzz23_nd
    TQUANTTEST.case_mxfp8_fp16_19x320_static21x384_exp2d_fuzz24_nd
    TQUANTTEST.case_mxfp8_fp16_25x448_static27x512_exp2d_fuzz25_nd
    TQUANTTEST.case_mxfp8_fp16_33x576_static35x640_exp2d_fuzz26_nd
    TQUANTTEST.case_mxfp8_fp16_41x832_static43x896_exp2d_fuzz27_nd
    TQUANTTEST.case_mxfp8_fp16_53x960_static55x1024_exp2d_fuzz28_nd
    TQUANTTEST.case_mxfp8_fp16_7x1024_static8x2048_exp2d_fuzz29_nd
    TQUANTTEST.case_mxfp8_fp16_1x4096_static4x4096_exp2d_fuzz30_nd
    TQUANTTEST.case_mxfp8_fp16_1x8128_static1x8192_exp2d_fuzz31_nd
    TQUANTTEST.case_mxfp8_fp16_181x64_static191x64_exp2d_fuzz32_nd
    TQUANTTEST.case_mxfp8_fp16_379x64_static383x64_exp2d_fuzz33_nd
    TQUANTTEST.case_mxfp8_fp16_189x128_static191x128_exp2d_fuzz34_nd
    TQUANTTEST.case_mxfp8_fp16_125x64_static127x256_exp2d_fuzz35_nd
    TQUANTTEST.case_mxfp8_fp16_77x384_static79x512_exp2d_fuzz36_nd
    TQUANTTEST.case_mxfp8_fp16_43x512_static47x1024_exp2d_fuzz37_nd
    TQUANTTEST.case_mxfp8_fp16_23x64_static25x1536_exp2d_fuzz38_nd
    TQUANTTEST.case_mxfp8_fp16_11x3008_static13x3072_exp2d_fuzz39_nd
    TQUANTTEST.case_mxfp8_fp16_2x4096_static2x8192_exp2d_fuzz40_nd
)

: >"${SUMMARY_FILE}"
printf 'repo: %s\n' "${REPO_ROOT}" | tee -a "${SUMMARY_FILE}"
printf 'args: %s -t tquant -g <case>\n' "${RUN_ST_ARGS[*]}" | tee -a "${SUMMARY_FILE}"
printf 'logs: %s\n\n' "${LOG_DIR}" | tee -a "${SUMMARY_FILE}"

failed=0
for case_name in "${CASES[@]}"; do
    log_file="${LOG_DIR}/${case_name}.log"
    printf '\n[RUN] %s\n' "${case_name}" | tee -a "${SUMMARY_FILE}"
    python3 tests/script/run_st.py "${RUN_ST_ARGS[@]}" -t tquant -g "${case_name}" 2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    if [ "${status}" -eq 0 ]; then
        printf '[PASS] %s\n' "${case_name}" | tee -a "${SUMMARY_FILE}"
    else
        failed=$((failed + 1))
        printf '[FAIL] %s status=%s log=%s\n' "${case_name}" "${status}" "${log_file}" | tee -a "${SUMMARY_FILE}"
    fi
done

printf '\nfailed: %s / %s\n' "${failed}" "${#CASES[@]}" | tee -a "${SUMMARY_FILE}"
printf 'summary: %s\n' "${SUMMARY_FILE}"

if [ "${failed}" -ne 0 ]; then
    exit 1
fi
