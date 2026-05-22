#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
#
# run_all.sh - unified runner for every auto-mode kernel under
# kernels/automode/a2a3/. Selects which kernels to run, then cd-s into each
# kernel directory and invokes its run.sh with -r/-v/-n forwarded.
#
# Usage:
#   bash run_all.sh -r npu -v Ascend910B1
#   bash run_all.sh -r npu -v Ascend910B1 --kernels add_tile_array,topk
#   bash run_all.sh -r npu -v Ascend910B1 --kernels MoE --moe-subkernels expert_ffn,gather
#   bash run_all.sh -r npu -v Ascend910B1 --cases-moe "512,128,128,32,1"
#   bash run_all.sh -r npu -v Ascend910B1 --cases-flash-atten "128,128,1024,128,256"
#
# Available top-level kernels (default = all):
#   add_tile_array, topk, router_topk_small, mla, flash_atten, MoE, MoEv2
#
# For MoE / MoEv2 families, all 8 sub-kernels are run unless filtered via
# --moe-subkernels (applied to BOTH families when both are selected).
#
# Notes:
#   - --cases-moe is passed to each MoE sub-kernel via -a (only the MoE family
#     supports per-shape overrides today; the other kernels run their default
#     hardcoded shape).
#   - --cases-flash-atten is passed only to flash_atten's -a.
#   - Each kernel writes its own report; this script aggregates pass/fail at
#     the end (a kernel "passes" if its run.sh exits with status 0).
# --------------------------------------------------------------------------------

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----- Defaults -------------------------------------------------------------

ALL_KERNELS=(add_tile_array topk router_topk_small mla flash_atten MoE MoEv2)
MOE_SUBKERNELS=(router_matmul moe_topk moe_topk_padded scatter expert_ffn gather full_moe_separate full_moe_combined)

RUN_MODE=""
SOC_VERSION=""
NPU_ID=""
KERNEL_FILTER=""
MOE_SUBKERNEL_FILTER=""
CASES_MOE=""
CASES_FLASH_ATTEN=""
RUN_LOG_DIR="${HERE}/run_log"

# ----- Arg parsing ----------------------------------------------------------

print_usage() {
    grep '^# ' "$0" | sed 's/^# //'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--run-mode)        RUN_MODE="$2"; shift 2;;
        -v|--soc-version)     SOC_VERSION="$2"; shift 2;;
        -n|--npu)             NPU_ID="$2"; shift 2;;
        --kernels)            KERNEL_FILTER="$2"; shift 2;;
        --kernels=*)          KERNEL_FILTER="${1#*=}"; shift;;
        --moe-subkernels)     MOE_SUBKERNEL_FILTER="$2"; shift 2;;
        --moe-subkernels=*)   MOE_SUBKERNEL_FILTER="${1#*=}"; shift;;
        --cases-moe)          CASES_MOE="$2"; shift 2;;
        --cases-moe=*)        CASES_MOE="${1#*=}"; shift;;
        --cases-flash-atten)  CASES_FLASH_ATTEN="$2"; shift 2;;
        --cases-flash-atten=*) CASES_FLASH_ATTEN="${1#*=}"; shift;;
        -h|--help)            print_usage; exit 0;;
        *) echo "[ERROR] Unknown argument: $1"; print_usage; exit 1;;
    esac
done

if [[ -z "${RUN_MODE}" || -z "${SOC_VERSION}" ]]; then
    echo "[ERROR] -r/--run-mode and -v/--soc-version are required."
    print_usage
    exit 1
fi

# ----- Build the selected kernel list ---------------------------------------

split_csv() {
    local IFS=','
    read -ra OUT <<< "$1"
    echo "${OUT[@]}"
}

# Determine which top-level kernels to run.
declare -a KERNELS_TO_RUN
if [[ -n "${KERNEL_FILTER}" ]]; then
    requested=($(split_csv "${KERNEL_FILTER}"))
    for k in "${requested[@]}"; do
        found=0
        for known in "${ALL_KERNELS[@]}"; do
            if [[ "$k" == "$known" ]]; then
                KERNELS_TO_RUN+=("$k"); found=1; break
            fi
        done
        if [[ $found -eq 0 ]]; then
            echo "[ERROR] Unknown kernel: '$k'. Known: ${ALL_KERNELS[*]}"
            exit 1
        fi
    done
else
    KERNELS_TO_RUN=("${ALL_KERNELS[@]}")
fi

# Determine which MoE sub-kernels to run.
declare -a MOE_SUBS_TO_RUN
if [[ -n "${MOE_SUBKERNEL_FILTER}" ]]; then
    requested=($(split_csv "${MOE_SUBKERNEL_FILTER}"))
    for s in "${requested[@]}"; do
        found=0
        for known in "${MOE_SUBKERNELS[@]}"; do
            if [[ "$s" == "$known" ]]; then
                MOE_SUBS_TO_RUN+=("$s"); found=1; break
            fi
        done
        if [[ $found -eq 0 ]]; then
            echo "[ERROR] Unknown MoE sub-kernel: '$s'. Known: ${MOE_SUBKERNELS[*]}"
            exit 1
        fi
    done
else
    MOE_SUBS_TO_RUN=("${MOE_SUBKERNELS[@]}")
fi

# ----- Helper: run one kernel directory's run.sh, record result -------------

declare -a RESULT_NAMES
declare -a RESULT_STATUS
declare -a RESULT_TIME

run_one() {
    local label="$1"   # display name (e.g. "MoE/expert_ffn")
    local dir="$2"     # absolute path to the kernel directory
    shift 2
    local extra_args=("$@")  # any extra args (e.g. -a "...")

    local cmd=(bash run.sh -r "${RUN_MODE}" -v "${SOC_VERSION}")
    if [[ -n "${NPU_ID}" ]]; then
        cmd+=(-n "${NPU_ID}")
    fi
    if [[ ${#extra_args[@]} -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
    fi

    echo
    echo "============================================================================"
    echo "[run_all] ${label}"
    echo "[run_all]   dir : ${dir}"
    echo "[run_all]   cmd : ${cmd[*]}"
    echo "[run_all]   logs: ${RUN_LOG_DIR}/${label//\//_}_<start_ts>.out  ${RUN_LOG_DIR}/${label//\//_}_<start_ts>.err"
    echo "============================================================================"

    local start_ts end_ts
    start_ts=$(date +%s)
    # ensure run_log dir exists
    mkdir -p "${RUN_LOG_DIR}"
    # safe label for filenames: replace non-alnum with underscore
    local safe_label
    safe_label=$(echo "${label}" | sed 's#[^A-Za-z0-9._-]#_##g')
    local out_log="${RUN_LOG_DIR}/${safe_label}_${start_ts}.out"
    local err_log="${RUN_LOG_DIR}/${safe_label}_${start_ts}.err"

    # Run the kernel, capturing stdout/stderr to separate files
    ( cd "${dir}" && "${cmd[@]}" ) > "${out_log}" 2> "${err_log}"
    local rc=$?
    end_ts=$(date +%s)
    local elapsed=$((end_ts - start_ts))

    echo "[run_all] logs saved: stdout=${out_log} stderr=${err_log}"

    RESULT_NAMES+=("${label}")
    RESULT_TIME+=("${elapsed}s")
    if [[ $rc -eq 0 ]]; then
        RESULT_STATUS+=("PASS")
    else
        RESULT_STATUS+=("FAIL (rc=$rc)")
    fi
}

# ----- Helper: run one MoE/MoEv2 family --------------------------------------

run_moe_family() {
    local family="$1"   # "MoE" or "MoEv2"
    local family_dir="${HERE}/${family}"
    if [[ ! -d "${family_dir}" ]]; then
        echo "[WARN] Family directory missing: ${family_dir}"
        return
    fi
    for sub in "${MOE_SUBS_TO_RUN[@]}"; do
        local sub_dir="${family_dir}/${sub}"
        if [[ ! -d "${sub_dir}" ]]; then
            echo "[WARN] Sub-kernel directory missing: ${sub_dir}"
            continue
        fi
        if [[ ! -f "${sub_dir}/run.sh" ]]; then
            echo "[WARN] No run.sh in ${sub_dir}"
            continue
        fi
        local label="${family}/${sub}"
        # Only the MoE family supports -a today; MoEv2 sub-kernels' run.sh
        # ignores unknown args, so we forward -a unconditionally if --cases-moe
        # was supplied (it's a no-op for MoEv2).
        if [[ -n "${CASES_MOE}" && "${family}" == "MoE" ]]; then
            run_one "${label}" "${sub_dir}" -a "${CASES_MOE}"
        else
            run_one "${label}" "${sub_dir}"
        fi
    done
}

# ----- Dispatch -------------------------------------------------------------

OVERALL_START=$(date +%s)

for kernel in "${KERNELS_TO_RUN[@]}"; do
    kernel_dir="${HERE}/${kernel}"
    case "${kernel}" in
        MoE|MoEv2)
            run_moe_family "${kernel}"
            ;;
        flash_atten)
            if [[ -n "${CASES_FLASH_ATTEN}" ]]; then
                run_one "${kernel}" "${kernel_dir}" -a "${CASES_FLASH_ATTEN}"
            else
                run_one "${kernel}" "${kernel_dir}"
            fi
            ;;
        *)
            if [[ ! -f "${kernel_dir}/run.sh" ]]; then
                echo "[WARN] No run.sh in ${kernel_dir}; skipping."
                continue
            fi
            run_one "${kernel}" "${kernel_dir}"
            ;;
    esac
done

OVERALL_END=$(date +%s)

# ----- Summary --------------------------------------------------------------

echo
echo "============================================================================"
echo "[run_all] SUMMARY"
echo "============================================================================"
PASS=0
FAIL=0
TOTAL=${#RESULT_NAMES[@]}
for i in "${!RESULT_NAMES[@]}"; do
    printf "  %-40s  %-16s  %s\n" "${RESULT_NAMES[$i]}" "${RESULT_STATUS[$i]}" "${RESULT_TIME[$i]}"
    if [[ "${RESULT_STATUS[$i]}" == "PASS" ]]; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
done
echo "  ----------------------------------------------------------------------------"
printf "  %-40s  %d/%d passed  (elapsed: %ds)\n" "TOTAL" "${PASS}" "${TOTAL}" "$((OVERALL_END - OVERALL_START))"
echo "============================================================================"

[[ ${FAIL} -eq 0 ]] && exit 0 || exit 1
