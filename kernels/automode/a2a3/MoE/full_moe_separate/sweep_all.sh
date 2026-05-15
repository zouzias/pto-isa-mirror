#!/bin/bash
# full_moe_separate/sweep_all.sh - run a curated 20-config matrix of the
# end-to-end pipeline. Walks the same shapes as MoE/sweep_all.sh but each
# row drives the full chained pipeline (router -> topk -> scatter -> ffn ->
# gather), validating the FINAL C against the numpy golden per shape.
#
# Designed to be fired before walking away. Writes:
#   sweep_all.log     — full streaming output of every config
#   sweep_all.summary — one-line-per-config PASS/FAIL matrix
#
# Usage:
#   bash sweep_all.sh [-r npu|sim] [-v Ascend910B1]   (defaults: npu Ascend910B1)
#
# Skipped: kH=kF=256 — Split-K territory, postponed.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_MODE="npu"
SOC_VERSION="Ascend910B1"

while [ $# -gt 0 ]; do
    case "$1" in
        -r|--run-mode)    RUN_MODE="$2";    shift 2;;
        -v|--soc-version) SOC_VERSION="$2"; shift 2;;
        *) echo "[sweep_all] unknown arg: $1"; exit 1;;
    esac
done

LOG="$HERE/sweep_all.log"
SUMMARY="$HERE/sweep_all.summary"
: > "$LOG"
: > "$SUMMARY"

# Same curated matrix as MoE/sweep_all.sh.
CONFIGS=(
    "256  64  64  32   1   v1_baseline"
    "256  64  64  32   2   kTopK=2"
    "256  64  64  32   4   kTopK=4"
    "256  64  64  32   8   kTopK=8"
    "256  64  64  32  16   kTopK=16"
    "256  64  64  16   1   kE=16"
    "128  64  64  32   1   kT=128"
    "512  64  64  32   1   kT=512"
    "256 128 128  32   1   kH=kF=128"
    "256  64  64  16   2   kE=16_kTopK=2"
    "256  64  64  16   8   kE=16_kTopK=8"
    "256  64  64  16  16   kE=16_kTopK=16"
    "128  64  64  16   1   kT=128_kE=16"
    "512  64  64  16   1   kT=512_kE=16"
    "512  64  64  32   2   kT=512_kTopK=2"
    "512  64  64  32   4   kT=512_kTopK=4"
    "128  64  64  32  16   kT=128_kTopK=16"
    "256 128 128  32   2   kH=128_kTopK=2"
    "512 128 128  32   1   kT=512_kH=128"
    "128 128 128  32   1   kT=128_kH=128"
)

NUM=${#CONFIGS[@]}
echo "[sweep_all] full_moe_separate · $NUM configs · $RUN_MODE / $SOC_VERSION" | tee -a "$LOG"
echo "[sweep_all] log:     $LOG"      | tee -a "$LOG"
echo "[sweep_all] summary: $SUMMARY"  | tee -a "$LOG"
echo                                  | tee -a "$LOG"

START_TS=$(date +%s)
declare -a TABLE
PASS_CFG=0
FAIL_CFG=0

i=0
for entry in "${CONFIGS[@]}"; do
    i=$((i + 1))
    read -r KT KH KF KE KTOPK LABEL <<< "$entry"

    echo "===========================================================================" | tee -a "$LOG"
    echo "[sweep_all] [$i/$NUM] $LABEL   (kT=$KT kH=$KH kF=$KF kE=$KE kTopK=$KTOPK)" | tee -a "$LOG"
    echo "===========================================================================" | tee -a "$LOG"

    bash "$HERE/sweep.sh" "$KT" "$KH" "$KF" "$KE" "$KTOPK" "$RUN_MODE" "$SOC_VERSION" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}

    if [ "$rc" -eq 0 ]; then
        TABLE+=("[$i/$NUM] PASS  kT=$KT kH=$KH kF=$KF kE=$KE kTopK=$KTOPK   $LABEL")
        PASS_CFG=$((PASS_CFG + 1))
    else
        TABLE+=("[$i/$NUM] FAIL  kT=$KT kH=$KH kF=$KF kE=$KE kTopK=$KTOPK   $LABEL   (exit $rc)")
        FAIL_CFG=$((FAIL_CFG + 1))
    fi
done

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

{
    echo
    echo "==========================================================================="
    echo "[sweep_all] full_moe_separate FINAL MATRIX   ($RUN_MODE / $SOC_VERSION)"
    echo "[sweep_all] elapsed: ${ELAPSED}s"
    echo "==========================================================================="
    for row in "${TABLE[@]}"; do
        echo "$row"
    done
    echo "---------------------------------------------------------------------------"
    echo "[sweep_all] $PASS_CFG/$NUM configs passed end-to-end; $FAIL_CFG failed"
} | tee -a "$LOG" > "$SUMMARY"

cat "$SUMMARY"

exit $FAIL_CFG
