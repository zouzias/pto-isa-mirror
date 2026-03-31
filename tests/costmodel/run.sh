#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ARCH="${PTO_COSTMODEL_TEST_ARCH:-a2a3}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build/${TEST_ARCH}}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

case "${TEST_ARCH}" in
    a2a3)
        ALL_CASES=(runtime_stub tload tadd tadds tsub tsubs tmul tmuls tmax tmaxs tmin tmins tabs tneg texp tlog tsqrt trsqrt tnot tand tor tmatmul tmatmul_acc tmatmul_bias tgemv tgemv_acc tgemv_bias tlrelu taxpy tdivs trelu tstore)
        ;;
    a5)
        ALL_CASES=(runtime_stub tload tadd tadds tsub tsubs tmul tmuls tdiv tdivs tmax tmaxs tmin tmins tabs taxpy tneg texp tlog tsqrt trsqrt tnot tand tands tor tors txor txors tshl tshls tshr tshrs tlrelu trelu tstore)
        ;;
    *)
        echo "unsupported costmodel test arch: ${TEST_ARCH}" >&2
        exit 1
        ;;
esac

is_valid_case() {
    local candidate="$1"
    local testcase
    for testcase in "${ALL_CASES[@]}"; do
        if [[ "${testcase}" == "${candidate}" ]]; then
            return 0
        fi
    done
    return 1
}

if [[ $# -eq 0 || "$1" == "all" ]]; then
    SELECTED_CASES=("${ALL_CASES[@]}")
    CONFIG_CASES="all"
    BUILD_ARGS=()
else
    SELECTED_CASES=()
    for testcase in "$@"; do
        if ! is_valid_case "${testcase}"; then
            echo "unknown testcase: ${testcase}" >&2
            echo "available: ${ALL_CASES[*]}" >&2
            exit 1
        fi
        SELECTED_CASES+=("${testcase}")
    done

    CONFIG_CASES="$(IFS=';'; echo "${SELECTED_CASES[*]}")"
    BUILD_ARGS=(--target "${SELECTED_CASES[@]}")
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    CACHE_SOURCE_DIR="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)"
    if [[ -n "${CACHE_SOURCE_DIR}" && "${CACHE_SOURCE_DIR}" != "${SCRIPT_DIR}" ]]; then
        rm -rf "${BUILD_DIR}"
    fi
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DPTO_COSTMODEL_TEST_ARCH="${TEST_ARCH}" -DPTO_COSTMODEL_CASES="${CONFIG_CASES}"
cmake --build "${BUILD_DIR}" "${BUILD_ARGS[@]}"

for testcase in "${SELECTED_CASES[@]}"; do
    "${BUILD_DIR}/bin/${testcase}"
done
