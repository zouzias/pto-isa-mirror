#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail
SESSION_NAME="${1:-pto-isa-sync}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${REPO_DIR}/logs/sync_cann_master_to_github_main.log"
mkdir -p "${REPO_DIR}/logs"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists"
  exit 0
fi
tmux new-session -d -s "${SESSION_NAME}" "cd '${REPO_DIR}' && echo '[tmux] starting sync at '$(date -Is) | tee -a '${LOG_FILE}' && ./scripts/sync_cann_master_to_github_main.sh | tee -a '${LOG_FILE}'"
echo "started tmux session: ${SESSION_NAME}"
echo "attach with: tmux attach -t ${SESSION_NAME}"
