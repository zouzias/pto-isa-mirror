#!/usr/bin/env bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${INITQUANT_BASELINE_LOG_DIR:-$(mktemp -d /tmp/initquant_baseline.XXXXXX)}"
STOP_STAGE="${INITQUANT_STOP_STAGE:-17}"
CLEAN_BUILD="${INITQUANT_CLEAN_BUILD:-0}"

COMMON_ARGS=(
    --backend int8
    --m2-fused-full 1
    --m2-fused-debug-stop-stage "${STOP_STAGE}"
    --dry-run 0
    --skip-kernel-launch 0
)

extract_values() {
    local key="$1"
    local log_file="$2"
    grep -aoE "${key}=[[:space:]]*[^[:space:]]+" "${log_file}" |
        sed -E "s/^${key}=[[:space:]]*//" |
        sort -u |
        paste -sd, -
}

extract_numbers() {
    local key="$1"
    local log_file="$2"
    grep -aoE "${key}=[[:space:]]*[-+]?[0-9]+([.][0-9]+)?" "${log_file}" |
        sed -E "s/^${key}=[[:space:]]*//" |
        sort -u |
        paste -sd, -
}

run_pto_case() {
    local name="$1"
    local world_size="$2"
    local m="$3"
    local k="$4"
    local n="$5"
    local topk="$6"
    local experts="$7"
    local max_output_size="$8"
    local log_file="${LOG_DIR}/pto_${name}.log"

    if bash "${SCRIPT_DIR}/run_a3.sh" "${COMMON_ARGS[@]}" \
        --case-name "baseline-${name}-stop${STOP_STAGE}" \
        --clean-build "${CLEAN_BUILD}" \
        -pes "${world_size}" \
        -M "${m}" \
        -K "${k}" \
        -N "${n}" \
        -topK "${topk}" \
        -expertPerPe "${experts}" \
        --max-output-size "${max_output_size}" >"${log_file}" 2>&1; then
        echo -n "pass"
    else
        echo -n "failed"
    fi
}

run_original_case() {
    local name="$1"
    local world_size="$2"
    local m="$3"
    local k="$4"
    local n="$5"
    local topk="$6"
    local experts="$7"
    local max_output_size="$8"
    local log_file="${LOG_DIR}/original_${name}.log"

    if [[ -z "${ORIGINAL_FFN_CMD:-}" ]]; then
        echo -n "missing_command"
        return
    fi

    CASE_NAME="${name}" WORLD_SIZE="${world_size}" M="${m}" K="${k}" N="${n}" TOPK="${topk}" EXPERTS="${experts}" \
        MAX_OUTPUT_SIZE="${max_output_size}" LOG_FILE="${log_file}" bash -lc "${ORIGINAL_FFN_CMD}" \
        >"${log_file}" 2>&1 && echo -n "pass" || echo -n "failed"
}

print_case_summary() {
    local name="$1"
    local world_size="$2"
    local m="$3"
    local k="$4"
    local n="$5"
    local topk="$6"
    local experts="$7"
    local max_output_size="$8"

    local pto_status
    pto_status="$(run_pto_case "${name}" "${world_size}" "${m}" "${k}" "${n}" "${topk}" "${experts}" \
        "${max_output_size}")"
    local pto_log="${LOG_DIR}/pto_${name}.log"
    local original_status
    original_status="$(run_original_case "${name}" "${world_size}" "${m}" "${k}" "${n}" "${topk}" "${experts}" \
        "${max_output_size}")"
    local original_log="${LOG_DIR}/original_${name}.log"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${name}" \
        "${pto_status}" \
        "$(extract_numbers init_quant_e2e_us "${pto_log}")" \
        "$(extract_numbers init_quant_worker_count "${pto_log}")" \
        "$(extract_numbers init_quant_active_workers "${pto_log}")" \
        "$(extract_values route_quant_path "${pto_log}")" \
        "$(extract_values route_quant_scalar_fallback_reason "${pto_log}")" \
        "${original_status}" \
        "$(test -f "${original_log}" && extract_numbers init_quant_e2e_us "${original_log}" || true)" \
        "${LOG_DIR}"
}

echo -e "case\tpto_status\tpto_e2e_us\tpto_worker_count\tpto_active_workers\tpto_route_quant_path\tpto_fallback_reason\toriginal_status\toriginal_e2e_us\tlog_dir"
print_case_summary small 2 16 128 128 2 2 32
print_case_summary large 2 4097 128 128 2 2 8194
