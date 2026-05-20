#!/usr/bin/env bash
set -euo pipefail

WORLD_SIZE=2
SOC_VERSION=""
M=16
K=128
N=128
TOPK=2
EXPERTS=2
MAX_OUTPUT_SIZE=32
SEED=20260515
ATOL=1e-3
RTOL=1e-3
WARMUP_ITERS=3
MEASURE_ITERS=5

: "${ASCEND_CANN_PATH:=$(ls -1d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort -V | tail -1)}"
if [ -d "${ASCEND_CANN_PATH}" ]; then
  ASCEND_CANN_PATH="${ASCEND_CANN_PATH%/}/set_env.sh"
fi
if [ -z "${ASCEND_CANN_PATH}" ] || [ ! -f "${ASCEND_CANN_PATH}" ]; then
  echo "[ERROR] Cannot find CANN set_env.sh. Set ASCEND_CANN_PATH to <cann-install> or <cann-install>/set_env.sh"
  exit 1
fi
set +eu
source "${ASCEND_CANN_PATH}"
set -euo pipefail

if [ -n "${MPI_ENV_BIN:-}" ] && [ -x "${MPI_ENV_BIN}/mpirun" ]; then
  export PATH="${MPI_ENV_BIN}:$PATH"
  if [ -n "${MPI_ENV_LIB:-}" ]; then
    export LD_LIBRARY_PATH="${MPI_ENV_LIB}:${LD_LIBRARY_PATH:-}"
    export MPI_LIB_PATH="${MPI_LIB_PATH:-${MPI_ENV_LIB}/libmpi.so}"
  fi
elif [ -z "${MPI_RUNNER:-}" ]; then
  if [ -z "${MPI_SEARCH_DIRS:-}" ]; then
    MPI_SEARCH_DIRS="/home/ntlab/miniconda3/envs/ltr_pto/bin /usr/local/mpich/bin /home/mpich/bin"
    for candidate in /home/*/mpich/bin /home/*/*/mpich/bin; do
      [ -d "$candidate" ] && MPI_SEARCH_DIRS="$MPI_SEARCH_DIRS $candidate"
    done
  fi
  for d in ${MPI_SEARCH_DIRS}; do
    if [ -x "$d/mpirun" ]; then
      export PATH="$d:$PATH"
      MPI_LIB_DIR="$(dirname "$d")/lib"
      export LD_LIBRARY_PATH="$MPI_LIB_DIR:${LD_LIBRARY_PATH:-}"
      export MPI_LIB_PATH="${MPI_LIB_PATH:-${MPI_LIB_DIR}/libmpi.so}"
      break
    fi
  done
fi
MPI_RUNNER=${MPI_RUNNER:-mpirun}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --soc|--soc-version) SOC_VERSION="$2"; shift 2 ;;
    --world-size) WORLD_SIZE="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --experts) EXPERTS="$2"; shift 2 ;;
    --max-output-size) MAX_OUTPUT_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --atol) ATOL="$2"; shift 2 ;;
    --rtol) RTOL="$2"; shift 2 ;;
    --warmup-iters) WARMUP_ITERS="$2"; shift 2 ;;
    --measure-iters) MEASURE_ITERS="$2"; shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

if [ -n "${SOC_VERSION}" ] && [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
  echo "[ERROR] Unsupported A5 SOC_VERSION: ${SOC_VERSION}"
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="${SCRIPT_DIR}/out"
BUILD_DIR="${DISPATCH_FFN_COMBINE_V3_BUILD_DIR:-/tmp/dispatch_ffn_combine_v3_a5_run_build}"

rm -rf /dev/shm/sem.hccl* 2>/dev/null || true
ipcrm -a 2>/dev/null || true

python3 "${SCRIPT_DIR}/scripts/gen_data.py" \
  --output-dir "${OUT_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --m "${M}" --k "${K}" --n "${N}" \
  --topk "${TOPK}" --experts "${EXPERTS}" \
  --max-output-size "${MAX_OUTPUT_SIZE}" \
  --seed "${SEED}" \
  --atol "${ATOL}" \
  --rtol "${RTOL}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --target dispatch_ffn_combine_v3 -j16

export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"
export DISPATCH_FFN_COMBINE_V3_CASE_DIR="${OUT_DIR}"
if [ -n "${SOC_VERSION}" ]; then
  export DISPATCH_FFN_COMBINE_V3_SOC_VERSION="${SOC_VERSION}"
else
  unset DISPATCH_FFN_COMBINE_V3_SOC_VERSION
fi
export DISPATCH_FFN_COMBINE_V3_WARMUP_ITERS="${WARMUP_ITERS}"
export DISPATCH_FFN_COMBINE_V3_MEASURE_ITERS="${MEASURE_ITERS}"

"${MPI_RUNNER}" -n "${WORLD_SIZE}" "${BUILD_DIR}/dispatch_ffn_combine_v3"
