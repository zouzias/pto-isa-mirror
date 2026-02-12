#!/bin/bash
# Generate timeline CSV and run phase analysis.
# Author: ywangmu from HKUST

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <build_dir> <out_dir> [fifo_mode]"
    exit 1
fi

BUILD_DIR="$1"
OUT_DIR="$2"
FIFO_MODE="${3:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${OUT_DIR}"

python3 "${SCRIPT_DIR}/pipeline_log_analysis.py" \
    --device-addrs "${BUILD_DIR}/device_addrs.toml" \
    --cube-start "${BUILD_DIR}/core0.cubecore0.instr_popped_log.dump" \
    --cube-end "${BUILD_DIR}/core0.cubecore0.instr_log.dump" \
    --vec-start "${BUILD_DIR}/core0.veccore0.instr_popped_log.dump" \
    --vec-end "${BUILD_DIR}/core0.veccore0.instr_log.dump" \
    --vec-mte3-issque "${BUILD_DIR}/core0.veccore0.ccu.mte3_issque.dump" \
    --fifo-mode "${FIFO_MODE}" \
    --out-csv "${OUT_DIR}/timeline.csv" \
    --out-json "${OUT_DIR}/timeline.json" \
    --out-agg "${OUT_DIR}/timeline_agg.csv" \
    --out-svg "${OUT_DIR}/timeline.svg"

python3 "${SCRIPT_DIR}/phase_analysis.py" \
    --csv "${OUT_DIR}/timeline.csv" \
    > "${OUT_DIR}/phase_analysis.txt"
