#!/bin/bash
# MoE/sweep.sh - patch shape constants in all four MoE kernel folders, build,
# run, and summarize. Each invocation tests ONE shape across all four kernels;
# loop in bash for a matrix sweep.
#
# Usage:
#   bash sweep.sh <kT> <kH> <kF> <kE> <kTopK> <RUN_MODE> <SOC_VERSION>
#
# Examples:
#   # Just v1:
#   bash sweep.sh 256 64 64 32 1  npu Ascend910B1
#
#   # kTopK axis (kE=32, kT=256, kH=kF=64):
#   for k in 1 2 4 8 16; do bash sweep.sh 256 64 64 32 $k npu Ascend910B1; done
#
#   # kT axis (kTopK=1):
#   for t in 128 256 512; do bash sweep.sh $t 64 64 32 1 npu Ascend910B1; done
#
#   # kE axis (kTopK=1):
#   for e in 16 32; do bash sweep.sh 256 64 64 $e 1 npu Ascend910B1; done
#
#   # kH=kF axis (kTopK=1; skip 256 — needs Split-K, postponed):
#   for h in 64 128; do bash sweep.sh 256 $h $h 32 1 npu Ascend910B1; done
#
# Notes:
#   - Constants are patched IN-PLACE in scripts/gen_data.py, main.cpp, and
#     *_kernel.cpp of each folder. The patches survive after the run; use
#     `git diff` and `git checkout -- <path>` to revert if needed.
#   - kF is only used by expert_ffn; the other folders are unaffected by it.
#   - kE in gather/gen_data.py is used only to synthesize a valid A_id and
#     is harmless to vary.
#   - kTileM in expert_ffn is held fixed at 16 (not patched by this script).
#   - Per-folder run.sh has set -euo pipefail; on build error, that folder
#     fails and the script continues with the next folder.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 7 ]; then
    grep '^# ' "$0" | sed 's/^# //'
    exit 1
fi

KT=$1; KH=$2; KF=$3; KE=$4; KTOPK=$5; RUN_MODE=$6; SOC_VERSION=$7

echo "==========================================================================="
echo "[sweep] kT=$KT kH=$KH kF=$KF kE=$KE kTopK=$KTOPK   ($RUN_MODE / $SOC_VERSION)"
echo "==========================================================================="

# Patch shape constants in a single file. Patterns are anchored to start-of-line
# and capture leading whitespace so derived names (kPackedRows, kAlloc, kTileM,
# kOverspillPad) are never matched: each pattern requires `kT` / `kH` / `kF` /
# `kE` / `kTopK` to be the *complete* identifier (next char must be whitespace
# or `=`).
patch_file() {
    local f=$1
    sed -i -E \
        -e "s/^([[:space:]]*)kT[[:space:]]*=.*$/\1kT = $KT/"                                       \
        -e "s/^([[:space:]]*)kH[[:space:]]*=.*$/\1kH = $KH/"                                       \
        -e "s/^([[:space:]]*)kF[[:space:]]*=.*$/\1kF = $KF/"                                       \
        -e "s/^([[:space:]]*)kE[[:space:]]*=.*$/\1kE = $KE/"                                       \
        -e "s/^([[:space:]]*)kTopK[[:space:]]*=.*$/\1kTopK = $KTOPK/"                              \
        -e "s/^([[:space:]]*)constexpr int kT[[:space:]]*=.*$/\1constexpr int kT = $KT;/"          \
        -e "s/^([[:space:]]*)constexpr int kH[[:space:]]*=.*$/\1constexpr int kH = $KH;/"          \
        -e "s/^([[:space:]]*)constexpr int kF[[:space:]]*=.*$/\1constexpr int kF = $KF;/"          \
        -e "s/^([[:space:]]*)constexpr int kE[[:space:]]*=.*$/\1constexpr int kE = $KE;/"          \
        -e "s/^([[:space:]]*)constexpr int kTopK[[:space:]]*=.*$/\1constexpr int kTopK = $KTOPK;/" \
        -e "s/^([[:space:]]*)constexpr unsigned kT[[:space:]]*=.*$/\1constexpr unsigned kT = $KT;/"          \
        -e "s/^([[:space:]]*)constexpr unsigned kH[[:space:]]*=.*$/\1constexpr unsigned kH = $KH;/"          \
        -e "s/^([[:space:]]*)constexpr unsigned kF[[:space:]]*=.*$/\1constexpr unsigned kF = $KF;/"          \
        -e "s/^([[:space:]]*)constexpr unsigned kE[[:space:]]*=.*$/\1constexpr unsigned kE = $KE;/"          \
        -e "s/^([[:space:]]*)constexpr unsigned kTopK[[:space:]]*=.*$/\1constexpr unsigned kTopK = $KTOPK;/" \
        "$f"
}

declare -a RESULTS
PASS=0
FAIL=0

for folder in moe_topk_padded scatter expert_ffn gather; do
    echo
    echo "--- $folder ---"
    fdir="$HERE/$folder"

    for f in "$fdir/scripts/gen_data.py" "$fdir/main.cpp" "$fdir"/*_kernel.cpp; do
        if [ -f "$f" ]; then
            patch_file "$f"
        fi
    done

    # Subshell so cd doesn't leak between iterations. Capture exit code of run.sh.
    ( cd "$fdir" && bash run.sh -r "$RUN_MODE" -v "$SOC_VERSION" )
    rc=$?

    if [ $rc -eq 0 ]; then
        RESULTS+=("  $folder: PASS")
        PASS=$((PASS + 1))
    else
        RESULTS+=("  $folder: FAIL  (run.sh exit $rc)")
        FAIL=$((FAIL + 1))
    fi
done

echo
echo "==========================================================================="
echo "[sweep] Summary — kT=$KT kH=$KH kF=$KF kE=$KE kTopK=$KTOPK"
echo "==========================================================================="
for r in "${RESULTS[@]}"; do
    echo "$r"
done
echo "  -----"
echo "  $PASS/$((PASS + FAIL)) folders passed"

exit $FAIL
