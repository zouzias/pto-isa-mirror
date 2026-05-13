#!/bin/bash
# Run a CCU GTest binary with process-level test isolation.
#
# hcomm CCU kernel handles are one-shot-per-HcclComm, so running multiple
# GTest cases in a single process crashes at HcclCcuKernelRegister.
# This script discovers test cases and invokes the binary once per test.
#
# Usage:
#   ./run_ccu_isolated.sh <binary> [nranks] [extra_gtest_args...]
#
# Examples:
#   ./run_ccu_isolated.sh ./treduce_ccu 2
#   ./run_ccu_isolated.sh ./tscatter_ccu 4 --gtest_also_run_disabled_tests

set -euo pipefail

BINARY="${1:?Usage: $0 <binary> [nranks] [extra_args...]}"
NRANKS="${2:-2}"
shift 2 || shift $#

MPIRUN="${MPIRUN:-$(command -v mpirun 2>/dev/null || echo /usr/local/mpich/bin/mpirun)}"
if [[ ! -x "$MPIRUN" ]]; then
    echo "[ERROR] mpirun not found. Set MPIRUN env or install MPICH." >&2
    exit 1
fi

TESTS=$("$BINARY" --gtest_list_tests "$@" 2>/dev/null | awk '
    /^[A-Za-z]/ { suite=$1 }
    /^  /        { gsub(/^  /,""); print suite $0 }
')

if [[ -z "$TESTS" ]]; then
    echo "[ERROR] No tests discovered in $BINARY" >&2
    exit 1
fi

TOTAL=$(echo "$TESTS" | wc -l | tr -d ' ')
PASSED=0
FAILED=0
IDX=0

echo "[==========] $TOTAL test(s) to run (process isolation mode)"

while IFS= read -r TEST; do
    IDX=$((IDX + 1))
    echo "[----------] [$IDX/$TOTAL] $TEST"

    if "$MPIRUN" -n "$NRANKS" "$BINARY" --gtest_filter="$TEST" "$@"; then
        echo "[  PASSED  ] $TEST"
        PASSED=$((PASSED + 1))
    else
        echo "[  FAILED  ] $TEST"
        FAILED=$((FAILED + 1))
    fi
done <<< "$TESTS"

echo "[==========] $TOTAL test(s) ran. $PASSED passed, $FAILED failed."
exit $FAILED
