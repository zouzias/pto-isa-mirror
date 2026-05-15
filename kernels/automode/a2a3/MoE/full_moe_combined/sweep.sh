#!/bin/bash
# full_moe_combined/sweep.sh - patch shape constants and run end-to-end.
#
# Patches local copies in this folder ONLY (kernels/*.cpp, main.cpp,
# scripts/gen_data.py). Does NOT touch the original 5 kernel folders.
#
# Usage:
#   bash sweep.sh <kT> <kH> <kF> <kE> <kTopK> <RUN_MODE> <SOC_VERSION>
#
# Examples:
#   bash sweep.sh 256 64 64 32 1 npu Ascend910B1
#
#   # kTopK sweep:
#   for k in 1 2 4 8 16; do bash sweep.sh 256 64 64 32 $k npu Ascend910B1; done

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 7 ]; then
    grep '^# ' "$0" | sed 's/^# //'
    exit 1
fi

KT=$1; KH=$2; KF=$3; KE=$4; KTOPK=$5; RUN_MODE=$6; SOC_VERSION=$7

echo "==========================================================================="
echo "[sweep] full_moe_combined  kT=$KT kH=$KH kF=$KF kE=$KE kTopK=$KTOPK  ($RUN_MODE / $SOC_VERSION)"
echo "==========================================================================="

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

for f in "$HERE/scripts/gen_data.py" "$HERE/main.cpp" "$HERE"/kernels/*_kernel.cpp; do
    if [ -f "$f" ]; then
        patch_file "$f"
    fi
done

( cd "$HERE" && bash run.sh -r "$RUN_MODE" -v "$SOC_VERSION" )
exit $?
