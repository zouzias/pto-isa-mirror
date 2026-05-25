#!/bin/bash
# --------------------------------------------------------------------------------
# common.sh - shared helpers for run.sh scripts under kernels/automode/a2a3/.
#
# Usage from a kernel's run.sh:
#   KERNEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   # ... getopt loop that sets PROFILE_MODE=1 when -p/--profile is given ...
#   source "${KERNEL_DIR}/../common.sh"           # top-level kernels
#   source "${KERNEL_DIR}/../../common.sh"        # MoE/MoEv2 sub-kernels
#
# Inputs (must be set by caller before sourcing):
#   KERNEL_DIR     absolute path to the kernel directory containing run.sh
#   PROFILE_MODE   "1" to enable profiling, anything else to disable
#
# Exposes:
#   PROF_DIR   absolute path to <KERNEL_DIR>/prof (only created if profiling)
#   run_bin    wraps a command in `msopprof --output=<PROF_DIR>` when
#              PROFILE_MODE=1, runs it bare otherwise
# --------------------------------------------------------------------------------

: "${PROFILE_MODE:=0}"
: "${KERNEL_DIR:?KERNEL_DIR must be set before sourcing common.sh}"

PROF_DIR=""
if [[ "${PROFILE_MODE}" == "1" ]]; then
    PROF_DIR="${KERNEL_DIR}/prof"
    mkdir -p "${PROF_DIR}"
    echo "[RUN.SH] profile mode: writing to ${PROF_DIR}"
fi

run_bin() {
    if [[ "${PROFILE_MODE}" == "1" ]]; then
        msopprof --output="${PROF_DIR}" "$@"
    else
        "$@"
    fi
}
