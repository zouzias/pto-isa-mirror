#!/bin/bash
set -euo pipefail

SHORT=r:,v:,C:,n:,b:,m:,s:,h:,d:,k:
LONG=run-mode:,soc-version:,compiler:,npu:,batch:,m-seq:,n-seq:,heads:,head-dim:,topk:
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
            TOPK="$2"
            shift 2;;
        (--)
            shift
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1"
            exit 1;;
    esac
done

: "${CMAKE_COMPILER:=bisheng}"

rm -rf build
mkdir -p build
cd build

cmake -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}" -DCMAKE_COMPILER="${CMAKE_COMPILER}" ..
make -j

cd ..

python3 gen.py --out_dir data --b "${B}" --m "${M}" --n "${N}" --h "${H}" --d "${D}" --topk "${K}"
./build/main data "${NPU_ID}"
python3 verify.py --data_dir data

echo "[DONE] sparse flash attention run_mode=${RUN_MODE}"
rm -f *.{dump,toml}