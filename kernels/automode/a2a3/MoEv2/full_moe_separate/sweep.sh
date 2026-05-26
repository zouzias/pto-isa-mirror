#!/bin/bash
# full_moe_separate/sweep.sh - run one generated end-to-end shape.
#
# Usage:
#   bash sweep.sh <kT> <kH> <kF> <kE> <kTopK> <RUN_MODE> <SOC_VERSION>

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 7 ]; then
    grep '^# ' "$0" | sed 's/^# //'
    exit 1
fi

CASE="$1,$2,$3,$4,$5"
RUN_MODE=$6
SOC_VERSION=$7

echo "==========================================================================="
echo "[sweep] full_moe_separate case=${CASE} (${RUN_MODE} / ${SOC_VERSION})"
echo "==========================================================================="

( cd "${HERE}" && bash run.sh -r "${RUN_MODE}" -v "${SOC_VERSION}" -a "${CASE}" )
exit $?
