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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
OUT_SVG="swimlane.svg"
EXTRA_ARGS=()

for arg in "$@"; do
  if [[ "${arg}" == --* ]]; then
    EXTRA_ARGS+=("${arg}")
  else
    OUT_SVG="${arg}"
  fi
done

python3 "${SCRIPT_DIR}/pipeline_swimlane_svg.py" \
  --cube-start "${BUILD_DIR}/core0.cubecore0.instr_popped_log.dump" \
  --cube-end "${BUILD_DIR}/core0.cubecore0.instr_log.dump" \
  --vec-start "${BUILD_DIR}/core0.veccore0.instr_popped_log.dump" \
  --vec-end "${BUILD_DIR}/core0.veccore0.instr_log.dump" \
  --vec-mte2-issque "${BUILD_DIR}/core0.veccore0.ccu.mte2_issque.dump" \
  --out-svg "${OUT_SVG}" \
  --svg-divisor 100 \
  --slot-width 8 \
  "${EXTRA_ARGS[@]}"

