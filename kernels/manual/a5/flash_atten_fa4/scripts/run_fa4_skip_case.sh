#!/bin/bash
# Run FA4 simulation with optional skip-rescale knobs.
# Author: ywangmu from HKUST

set -euo pipefail

SKIP_ENABLE="${SKIP_ENABLE:-}"
SKIP_EPS="${SKIP_EPS:-}"
AUTO_STOP_ON_CQ="${AUTO_STOP_ON_CQ:-0}"
AUTO_STOP_LOG="${AUTO_STOP_LOG:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-enable)
            SKIP_ENABLE="$2"
            shift 2
            ;;
        --skip-eps)
            SKIP_EPS="$2"
            shift 2
            ;;
        --auto-stop-on-cq)
            AUTO_STOP_ON_CQ=1
            shift 1
            ;;
        --no-auto-stop-on-cq)
            AUTO_STOP_ON_CQ=0
            shift 1
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ -n "${SKIP_ENABLE}" ]]; then
    export FA4_SKIP_RESCALE_ENABLE="${SKIP_ENABLE}"
fi
if [[ -n "${SKIP_EPS}" ]]; then
    export FA4_SKIP_RESCALE_EPS="${SKIP_EPS}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FA4_DIR="${SCRIPT_DIR}/.."

RUN_CMD=(bash "${FA4_DIR}/run.sh" "$@")
if [[ "${AUTO_STOP_ON_CQ}" != "1" ]]; then
    exec "${RUN_CMD[@]}"
fi

if [[ -z "${AUTO_STOP_LOG}" ]]; then
    AUTO_STOP_LOG="${FA4_DIR}/build/auto_stop.log"
fi
mkdir -p "$(dirname "${AUTO_STOP_LOG}")"
touch "${AUTO_STOP_LOG}"

echo "[RUN_FA4_SKIP] AUTO_STOP_ON_CQ=1, log=${AUTO_STOP_LOG}"
"${RUN_CMD[@]}" > >(tee "${AUTO_STOP_LOG}") 2>&1 &
RUN_PID=$!

CQ_MARKER="send_stars_interrupt:get cq_0 base_addr: 10020000"
while kill -0 "${RUN_PID}" 2>/dev/null; do
    if grep -qF "${CQ_MARKER}" "${AUTO_STOP_LOG}"; then
        echo "[RUN_FA4_SKIP] CQ marker detected, stopping simulation process tree."
        pkill -P "${RUN_PID}" 2>/dev/null || true
        kill "${RUN_PID}" 2>/dev/null || true
        break
    fi
    sleep 1
done

wait "${RUN_PID}" 2>/dev/null || true
