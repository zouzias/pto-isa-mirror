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
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PATTERN='AscendC::|Catlass::|catlass/|using namespace AscendC|TQue|TPipe|TBuf|LocalTensor|DataCopyPad'

if rg -n --glob '*.cpp' --glob '*.hpp' "${PATTERN}" "${PROJECT_DIR}/kernel" "${PROJECT_DIR}/include"; then
    echo "[FAIL] non-PTO device API boundary violation found" >&2
    exit 1
fi

echo "[PASS] PTO-native boundary check passed"
