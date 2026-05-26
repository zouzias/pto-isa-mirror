#!/bin/bash
# MoEv2/sweep.sh - run one generated shape across all MoEv2 leaf kernels.
#
# Usage:
#   bash sweep.sh <kT> <kH> <kF> <kE> <kTopK> <RUN_MODE> <SOC_VERSION>
#
# Example:
#   bash sweep.sh 256 64 64 32 1 npu Ascend910B1

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 7 ]; then
    grep '^# ' "$0" | sed 's/^# //'
    exit 1
fi

KT=$1; KH=$2; KF=$3; KE=$4; KTOPK=$5; RUN_MODE=$6; SOC_VERSION=$7
CASE="${KT},${KH},${KF},${KE},${KTOPK}"
FOLDERS=(router_matmul moe_topk moe_topk_padded scatter expert_ffn gather full_moe_separate full_moe_combined)

echo "==========================================================================="
echo "[sweep] MoEv2 case=${CASE} (${RUN_MODE} / ${SOC_VERSION})"
echo "==========================================================================="

declare -a RESULTS
PASS=0
FAIL=0

for folder in "${FOLDERS[@]}"; do
    echo
    echo "--- ${folder} ---"
    fdir="${HERE}/${folder}"
    if [ ! -f "${fdir}/run.sh" ]; then
        RESULTS+=("  ${folder}: FAIL  (missing run.sh)")
        FAIL=$((FAIL + 1))
        continue
    fi

    ( cd "${fdir}" && bash run.sh -r "${RUN_MODE}" -v "${SOC_VERSION}" -a "${CASE}" )
    rc=$?

    if [ ${rc} -eq 0 ]; then
        RESULTS+=("  ${folder}: PASS")
        PASS=$((PASS + 1))
    else
        RESULTS+=("  ${folder}: FAIL  (run.sh exit ${rc})")
        FAIL=$((FAIL + 1))
    fi
done

echo
echo "==========================================================================="
echo "[sweep] Summary - MoEv2 case=${CASE}"
echo "==========================================================================="
for r in "${RESULTS[@]}"; do
    echo "$r"
done
echo "  -----"
echo "  ${PASS}/$((PASS + FAIL)) folders passed"

exit ${FAIL}
