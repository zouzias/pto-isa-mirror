#!/bin/bash
set -euo pipefail

SHORT=r:,v:,C:,n:,b:,m:,s:,h:,d:,k:,a:
LONG=run-mode:,soc-version:,compiler:,npu:,batch:,m-seq:,n-seq:,heads:,head-dim:,topk:,cases:,msopprof
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

RUN_MODE=npu
SOC_VERSION=Ascend910B1
NPU_ID=0
B=1
M=6
N=6
H=16
D=256
K=6

while :
do
    case "$1" in
        (-r | --run-mode )
            RUN_MODE="$2"
            shift 2;;
        (-v | --soc-version )
            SOC_VERSION="$2"
            shift 2;;
        (-C | --compiler )
            CMAKE_COMPILER="$2"
            shift 2;;
        (-n | --npu )
            NPU_ID="$2"
            shift 2;;
        (-b | --batch )
            B="$2"
            shift 2;;
        (-m | --m-seq )
            M="$2"
            shift 2;;
        (-s | --n-seq )
            N="$2"
            shift 2;;
        (-h | --heads )
            H="$2"
            shift 2;;
        (-d | --head-dim )
            D="$2"
            shift 2;;
        (-k | --topk )
            K="$2"
            shift 2;;
        (-a | --cases )
            CASES_RAW="$2"
            shift 2;;
        (--msopprof )
            MSOPPROF=1
            shift 1;;
        (--)
            shift
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1"
            exit 1;;
    esac
done

run_msopprof() {
    if [[ "${MSOPPROF:-0}" == "1" ]]; then
        mkdir -p msopprof_data
        msopprof --output=msopprof_data "$@"
    else
        "$@"
    fi
}

: "${CMAKE_COMPILER:=bisheng}"

rm -rf build
mkdir -p build

GEN_CASE_ARGS=()
if [[ -n "${CASES_RAW:-}" ]]; then
    GEN_CASE_ARGS+=(--cases "${CASES_RAW}")
    IFS=',' read -r B M N H D K <<< "${CASES_RAW}"
else
    GEN_CASE_ARGS+=(--cases "${B},${M},${N},${H},${D},${K}")
fi

python3 generate_cases.py "${GEN_CASE_ARGS[@]}"

cd build

cmake -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}" -DCMAKE_COMPILER="${CMAKE_COMPILER}" ..
make -j

cd ..

python3 gen.py --out_dir data --b "${B}" --m "${M}" --n "${N}" --h "${H}" --d "${D}" --topk "${K}"
run_msopprof ./build/main data "${NPU_ID}"
python3 verify.py --data_dir data

echo "[DONE] sparse flash attention run_mode=${RUN_MODE}"
rm -f *.{dump,toml}
