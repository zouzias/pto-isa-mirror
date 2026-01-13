#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

set -euo pipefail

usage() {
  cat <<EOF
Usage: ./tests/run_st.sh <soc> <sim|npu> <simple|all> [build_only]

  soc: a2 | a3 | a5 | a2_a3 | a3_a5 | a2_a3_a5
  mode: sim | npu
  suite: simple | all

Environment:
  PTO_ST_SERIAL=1            Force serial execution (no parallel runner)
  PTO_ST_PARALLEL_ARGS="..." Extra args passed to run_st_parallel.py
  PTO_ST_SIMPLE_TESTCASES="tadd,tmatmul,..." Override the default simple set
EOF
}

SOC_SPEC="${1:-}"
RUN_TYPE="${2:-}"
SUITE="${3:-all}"
ACTION="${4:-run}"

if [[ -z "${SOC_SPEC}" || -z "${RUN_TYPE}" ]]; then
  usage
  exit 2
fi

# Best-effort: source Ascend toolkit env when user hasn't done so.
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  if [[ -f "${HOME}/Ascend/ascend-toolkit/set_env.sh" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/Ascend/ascend-toolkit/set_env.sh" >/dev/null 2>&1 || true
  elif [[ -f "${HOME}/Ascend/ascend-toolkit/bin/setenv.bash" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/Ascend/ascend-toolkit/bin/setenv.bash" >/dev/null 2>&1 || true
  elif [[ -f "${HOME}/Ascend/ascend-toolkit/latest/bin/setenv.bash" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/Ascend/ascend-toolkit/latest/bin/setenv.bash" >/dev/null 2>&1 || true
  fi
fi

case "${RUN_TYPE}" in
  sim|npu) ;;
  *) usage; exit 2 ;;
esac

case "${SUITE}" in
  simple|all) ;;
  *) usage; exit 2 ;;
esac

SOC_LIST=()
case "${SOC_SPEC}" in
  a2) SOC_LIST=(a2) ;;
  a3) SOC_LIST=(a3) ;;
  a5) SOC_LIST=(a5) ;;
  a2_a3) SOC_LIST=(a2 a3) ;;
  a3_a5) SOC_LIST=(a3 a5) ;;
  a2_a3_a5) SOC_LIST=(a2 a3 a5) ;;
  *) usage; exit 2 ;;
esac

default_simple="tmatmul,tadd,tsub,tmul,tdiv,tload,tstore,ttrans,textract,tmov"
simple_cases="${PTO_ST_SIMPLE_TESTCASES:-${default_simple}}"

run_one_soc() {
  local soc="$1"
  if [[ "${ACTION}" == "build_only" ]]; then
    python3 tests/script/build_st.py -r "${RUN_TYPE}" -v "${soc}" -t all
    return
  fi

  if [[ "${RUN_TYPE}" == "npu" && "${SUITE}" == "all" && "${PTO_ST_SERIAL:-0}" != "1" ]]; then
    python3 tests/script/run_st_parallel.py -v "${soc}" ${PTO_ST_PARALLEL_ARGS:-}
    return
  fi

  if [[ "${SUITE}" == "all" ]]; then
    python3 tests/script/run_st.py -r "${RUN_TYPE}" -v "${soc}" -t all
    return
  fi

  IFS=',' read -r -a cases <<< "${simple_cases}"
  for tc in "${cases[@]}"; do
    tc="$(echo "${tc}" | xargs)"
    [[ -z "${tc}" ]] && continue
    python3 tests/script/run_st.py -r "${RUN_TYPE}" -v "${soc}" -t "${tc}"
  done
}

for soc in "${SOC_LIST[@]}"; do
  run_one_soc "${soc}"
done

