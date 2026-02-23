#!/bin/bash
# Run FA4 simulation with optional skip-rescale knobs.
# Author: ywangmu from HKUST

set -euo pipefail

SKIP_ENABLE="${SKIP_ENABLE:-}"
SKIP_EPS="${SKIP_EPS:-}"
AUTO_STOP="${AUTO_STOP:-1}"
AUTO_STOP_LOG="${AUTO_STOP_LOG:-}"
AUTO_STOP_TIMEOUT_SEC="${AUTO_STOP_TIMEOUT_SEC:-0}"

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
        --auto-stop|--auto-stop-on-cq)
            AUTO_STOP=1
            shift 1
            ;;
        --no-auto-stop|--no-auto-stop-on-cq)
            AUTO_STOP=0
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
if [[ "${AUTO_STOP}" != "1" ]]; then
    exec "${RUN_CMD[@]}"
fi

if [[ -z "${AUTO_STOP_LOG}" ]]; then
    AUTO_STOP_LOG="${FA4_DIR}/run_monitor.log"
fi
mkdir -p "$(dirname "${AUTO_STOP_LOG}")"
touch "${AUTO_STOP_LOG}"

SUMMARY_LOG="${FA4_DIR}/build/core0_summary_log"
if [[ "${AUTO_STOP_TIMEOUT_SEC}" != "0" ]]; then
    DEADLINE_SEC=$((SECONDS + AUTO_STOP_TIMEOUT_SEC))
else
    DEADLINE_SEC=0
fi

echo "[RUN_FA4_SKIP] AUTO_STOP=1, summary=${SUMMARY_LOG}, log=${AUTO_STOP_LOG}, timeout_sec=${AUTO_STOP_TIMEOUT_SEC}"
"${RUN_CMD[@]}" > >(tee "${AUTO_STOP_LOG}") 2>&1 &
RUN_PID=$!

while kill -0 "${RUN_PID}" 2>/dev/null; do
    if [[ -s "${SUMMARY_LOG}" ]]; then
        echo "[RUN_FA4_SKIP] Non-empty core0_summary_log detected, stopping simulation process tree."
        pkill -P "${RUN_PID}" 2>/dev/null || true
        kill "${RUN_PID}" 2>/dev/null || true
        break
    fi
    if [[ "${DEADLINE_SEC}" != "0" && "${SECONDS}" -ge "${DEADLINE_SEC}" ]]; then
        echo "[RUN_FA4_SKIP] Timeout reached without non-empty core0_summary_log."
        break
    fi
    sleep 2
done

wait "${RUN_PID}" 2>/dev/null || true
