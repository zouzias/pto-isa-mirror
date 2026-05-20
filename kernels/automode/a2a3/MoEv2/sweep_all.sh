#!/bin/bash
# MoE/sweep_all.sh - run a curated 20-config matrix across all 5 MoE kernels.
#
# Designed to be fired before walking away from the machine. Each config calls
# sweep.sh, which patches constants in 5 folders, builds, runs, and reports.
# Results are written to:
#   sweep_all.log     — full streaming output of every config
#   sweep_all.summary — one-line-per-config PASS/FAIL matrix at the end
#
# Usage:
#   bash sweep_all.sh [-r npu|sim] [-v Ascend910B1]   (defaults: npu Ascend910B1)
#
# Skipped: kH=kF=256 — that's Split-K territory, postponed (see
# pto_auto_mode_hw_optimization_guide.md §1.3 / assumptions_to_verify.md §11.9).

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

# Curated matrix. Each row: "kT kH kF kE kTopK   # label"
#
# Grouping logic:
#   1-5    : kTopK axis at v1 base shape (kT=256, kH=kF=64, kE=32)
#   6      : kE = 16 single-axis
#   7-8    : kT axis (128, 512) — v1 base otherwise
#   9      : kH = kF = 128 single-axis
#   10-20  : combined / cross-axis stress cases
CONFIGS=(
    "256  64  64  32   1   v1_baseline"
    "256 256 256  64   1   kT=256_kH=256-1"
    "256 256 256  64   2   kT=256_kH=256-2"
    "256 256 256  64   4   kT=256_kH=256-4"
    "256 256 256  64   8   kT=256_kH=256-8"
    "256 256 256  64   16   kT=256_kH=256-16"
    "2048 7480 2048  64   1   kT=2048_kH=7480-1"
    "2048 7480 2048  64   2   kT=2048_kH=7480-2"
    "2048 7480 2048  64   4   kT=2048_kH=7480-4"
    "2048 7480 2048  64   8   kT=2048_kH=7480-8"
    "2048 7480 2048  64   16   kT=2048_kH=7480-16"
    "2048 1024 1024  64   1   kT=1024_kH=1024-1"
    "1024 1024 1024  64   2   kT=1024_kH=1024-2"
    "1024 1024 1024  64   4   kT=1024_kH=1024-4"
    "1024 1024 1024  64   8   kT=1024_kH=1024-8"
    "1024 1024 1024  64   16   kT=1024_kH=1024-16"
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
echo "[sweep_all] $NUM configs against $RUN_MODE / $SOC_VERSION" | tee -a "$LOG"
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

    # sweep.sh prints per-folder PASS/FAIL and returns the count of failing folders.
    bash "$HERE/sweep.sh" "$KT" "$KH" "$KF" "$KE" "$KTOPK" "$RUN_MODE" "$SOC_VERSION" 2>&1 | tee -a "$LOG"
    sweep_rc=${PIPESTATUS[0]}

    if [ "$sweep_rc" -eq 0 ]; then
        TABLE+=("[$i/$NUM] PASS  kT=$KT kH=$KH kF=$KF kE=$KE kTopK=$KTOPK   $LABEL")
        PASS_CFG=$((PASS_CFG + 1))
    else
        TABLE+=("[$i/$NUM] FAIL  kT=$KT kH=$KH kF=$KF kE=$KE kTopK=$KTOPK   $LABEL   ($sweep_rc folder(s) failed)")
        FAIL_CFG=$((FAIL_CFG + 1))
    fi
done

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

# Final summary — both to the log and to the standalone summary file.
{
    echo
    echo "==========================================================================="
    echo "[sweep_all] FINAL MATRIX   ($RUN_MODE / $SOC_VERSION)"
    echo "[sweep_all] elapsed: ${ELAPSED}s"
    echo "==========================================================================="
    for row in "${TABLE[@]}"; do
        echo "$row"
    done
    echo "---------------------------------------------------------------------------"
    echo "[sweep_all] $PASS_CFG/$NUM configs fully passed; $FAIL_CFG had at least one folder fail"
} | tee -a "$LOG" > "$SUMMARY"

# Also echo the summary so the human walking back to the terminal sees it.
cat "$SUMMARY"

exit $FAIL_CFG
