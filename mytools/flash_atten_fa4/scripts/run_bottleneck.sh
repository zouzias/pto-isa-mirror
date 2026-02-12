#!/bin/bash
# --------------------------------------------------------------------------------
# Author: ywangmu from HKUST
# Runner script for bottleneck analysis.
#
# Usage:
#   bash run_bottleneck.sh [FIFO_MODE]
#
# This script:
#   1. Runs run_timeline.sh to generate timeline.csv from build logs
#   2. Runs bottleneck_analysis.py on the generated CSV
#
# Examples:
#   bash run_bottleneck.sh          # auto-detect FIFO_MODE
#   bash run_bottleneck.sh 1        # force FIFO_MODE=1
#   bash run_bottleneck.sh 2        # force FIFO_MODE=2
# --------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIFO_MODE="${1:-${FIFO_MODE:-auto}}"

echo "=== Step 1: Generate timeline CSV ==="
bash "${SCRIPT_DIR}/run_timeline.sh" "${FIFO_MODE}"

echo ""
echo "=== Step 2: Bottleneck analysis ==="
python3 "${SCRIPT_DIR}/bottleneck_analysis.py" --csv "${SCRIPT_DIR}/timeline.csv"
