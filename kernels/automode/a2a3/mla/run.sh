#!/bin/bash
# --------------------------------------------------------------------------------
# mla - DeepSeek-V2 Multi-Head Latent Attention test runner.
#
# Each case tuple format: S,H,Nh,Hd,L,qL,Rd
#   S=seq_len, H=hidden, Nh=num_heads, Hd=head_dim,
#   L=kv_latent, qL=q_latent, Rd=rope_dim
#
# Default cases (run when neither -c nor -a is provided):
#   128,4096,32,128,64,64,64
#   256,4096,32,128,64,64,64
#
# This kernel compiles against ONE case at a time. For multi-case runs we
# loop here: per case, regenerate the header, rebuild, run.
#
# Usage:
#   bash run.sh -r npu -v Ascend910B1
#   bash run.sh -r npu -v Ascend910B1 -c "128,4096,32,128,64,64,64"
#   bash run.sh -r npu -v Ascend910B1 -a "128,4096,32,128,64,64,64;256,4096,32,128,64,64,64"
#   bash run.sh -r sim -v Ascend910B4 -n 1 -d -i
#   bash run.sh -r npu -v Ascend910B1 -p            # wrap binary in msopprof
# --------------------------------------------------------------------------------

KERNEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SHORT=r:,v:,C:,n:,c:,a:,i,d,p
LONG=run-mode:,soc-version:,compiler:,npu:,case:,cases:,intermediate,debug,profile
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
PROFILE_MODE=0
while :
do
    case "$1" in
        (-r | --run-mode )    RUN_MODE="$2"; shift 2;;
        (-v | --soc-version ) SOC_VERSION="$2"; shift 2;;
        (-C | --compiler )    CMAKE_COMPILER="$2"; shift 2;;
        (-n | --npu )         NPU_ID="$2"; shift 2;;
        (-c | --case )        CASE_FILTER="$2"; shift 2;;
        (-a | --cases )       CASES_RAW="$2"; shift 2;;
        (-i | --intermediate) INTERMEDIATE=1; shift 1;;
        (-d | --debug )       DEBUG_BUILD=1; shift 1;;
        (-p | --profile )     PROFILE_MODE=1; shift 1;;
        (--) shift; break;;
        (*) echo "[ERROR] Unexpected option: $1"; break;;
    esac
done

: "${CMAKE_COMPILER:=bisheng}"
: "${NPU_ID:=0}"

# common.sh reads PROFILE_MODE and KERNEL_DIR; exposes run_bin() which wraps
# the binary with `msopprof --output=<KERNEL_DIR>/prof` when PROFILE_MODE=1.
source "${KERNEL_DIR}/../common.sh"

pattern="^Ascend910B|^Ascend910_9599"
if [[ ! "${SOC_VERSION:-}" =~ $pattern ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION:-<unset>}. This folder supports A2/A3/A5."
    exit 1
fi
pattern="^Ascend910B4-1"
if [[ "${SOC_VERSION}" =~ $pattern ]] && [ "${RUN_MODE:-}" == "sim" ]; then
    echo "[ERROR] SocVersion: ${SOC_VERSION} cannot run in sim mode; use Ascend910B4 or Ascend910_9599."
    exit 1
fi

# ----- Default cases (§8 of kernel_test_guidance.md) ---------------------------
DEFAULT_CASES=(
    "128,4096,32,128,64,64,64"
    "256,4096,32,128,64,64,64"
)

# Build a flat array of case tuples to iterate over.
declare -a CASE_LIST
if [[ -n "${CASE_FILTER:-}" ]]; then
    CASE_LIST=("${CASE_FILTER}")
elif [[ -n "${CASES_RAW:-}" ]]; then
    IFS=';' read -ra CASE_LIST <<< "${CASES_RAW}"
else
    CASE_LIST=("${DEFAULT_CASES[@]}")
fi

echo "[RUN.SH] RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}  NPU=${NPU_ID}"
echo "[RUN.SH] DEBUG=${DEBUG_BUILD:-0}  INTERMEDIATE=${INTERMEDIATE:-0}  PROFILE=${PROFILE_MODE}"
echo "[RUN.SH] cases to run (${#CASE_LIST[@]}):"
for c in "${CASE_LIST[@]}"; do echo "          - $c"; done

# ----- Per-case loop: regenerate header, rebuild, run --------------------------

set -uo pipefail

OVERALL_RC=0
declare -a SUMMARY

for CASE in "${CASE_LIST[@]}"; do
    echo
    echo "================================================================================"
    echo "[RUN.SH] case: ${CASE}"
    echo "================================================================================"

    rm -rf build
    mkdir build
    cd build

    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}

    # Step 1: emit build/generated_cases.{h,json} for this case (active idx 0).
    if ! python3 ../scripts/generate_cases.py --cases "${CASE}"; then
        echo "[RUN.SH] generate_cases.py failed for ${CASE}"
        OVERALL_RC=1
        SUMMARY+=("${CASE}: GENERATE_FAIL")
        cd ..
        continue
    fi

    # Step 2: cmake + make.
    CMAKE_EXTRA=()
    if [[ -n "${DEBUG_BUILD:-}" ]]; then
        CMAKE_EXTRA+=(-DDEBUG_MODE=ON)
    fi
    if ! cmake -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}" \
               -DCMAKE_COMPILER="${CMAKE_COMPILER}" "${CMAKE_EXTRA[@]}" .. ; then
        echo "[RUN.SH] cmake failed for ${CASE}"
        OVERALL_RC=1
        SUMMARY+=("${CASE}: CMAKE_FAIL")
        cd ..
        continue
    fi
    if ! make -j16 ; then
        echo "[RUN.SH] make failed for ${CASE}"
        OVERALL_RC=1
        SUMMARY+=("${CASE}: BUILD_FAIL")
        cd ..
        continue
    fi

    # Step 3: gen_data.py writes inputs/goldens to ./input and ./output
    # relative to its CWD. main.cpp reads from ../input/ and ../output/
    # (one level up from build/), so we must run gen_data.py from mla/
    # (parent of build/) — i.e. clear any stale files and write fresh ones
    # into the directories main.cpp will read.
    rm -rf ../input ../output
    if ! ( cd .. && python3 scripts/gen_data.py --case "${CASE}" ) ; then
        echo "[RUN.SH] gen_data.py failed for ${CASE}"
        OVERALL_RC=1
        SUMMARY+=("${CASE}: GENDATA_FAIL")
        cd ..
        continue
    fi

    # Step 4: run binary with case forwarded for sanity-check.
    EXTRA_BIN_ARGS=(--npu="${NPU_ID}" --case="${CASE}")
    if [[ -n "${INTERMEDIATE:-}" ]]; then
        EXTRA_BIN_ARGS+=(--intermediate)
    fi
    if [[ "${SOC_VERSION}" == "Ascend910_9599" ]]; then
        EXTRA_BIN_ARGS+=(--sys_cnt_multiple=1.0)
    fi
    set +e
    time run_bin ./mla_basic "${EXTRA_BIN_ARGS[@]}"
    RC=$?
    set -e
    if [[ $RC -eq 0 ]]; then
        SUMMARY+=("${CASE}: PASS")
    else
        SUMMARY+=("${CASE}: FAIL (rc=$RC)")
        OVERALL_RC=1
    fi

    # Optional: post-mortem comparison (writes a richer report than main.cpp).
    # compare_outputs.py reads ../output/, so it must be run from build/.
    python3 ../scripts/compare_outputs.py || true

    cd ..
done

echo
echo "================================================================================"
echo "[RUN.SH] SUMMARY"
echo "================================================================================"
for line in "${SUMMARY[@]}"; do echo "  ${line}"; done

exit ${OVERALL_RC}
