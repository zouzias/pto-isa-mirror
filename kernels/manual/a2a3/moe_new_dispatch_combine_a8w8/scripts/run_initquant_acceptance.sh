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

COMMON_ARGS=(
    --backend int8
    --m2-fused-full 1
    --dry-run 0
    --skip-kernel-launch 0
)

run_case() {
    local name="$1"
    local clean_build="$2"
    shift 2
    local stop_stage=17
    echo "[InitQuantAcceptance] case=${name} stop=${stop_stage}"
    bash "${SCRIPT_DIR}/run_a3.sh" "${COMMON_ARGS[@]}" \
        --m2-fused-debug-stop-stage "${stop_stage}" \
        --clean-build "${clean_build}" \
        --case-name "${name}-stop${stop_stage}" "$@"
}

run_case small 1 \
    -pes 2 \
    -M 16 \
    -K 128 \
    -N 128 \
    -topK 2 \
    -expertPerPe 2 \
    --max-output-size 32

run_case large 0 \
    -pes 2 \
    -M 4097 \
    -K 128 \
    -N 128 \
    -topK 2 \
    -expertPerPe 2 \
    --max-output-size 8194
