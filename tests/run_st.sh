#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

set -e

ENABLE_A3=false
ENABLE_A5=false
ENABLE_KIRIN9030=false
ENABLE_KIRINX90=false
ENABLE_SIMPLE=false
ENABLE_ALL=false
ENABLE_COMM=false
ARGS=" "
IS_AUTO_MODE=false

checkopts() {
  while true; do
    case "$1" in
      --a3)
        ENABLE_A3=true
        shift
        ;;
      --a5)
        ENABLE_A5=true
        shift
        ;;
      --a3_a5)
        ENABLE_A3=true
        ENABLE_A5=true
        shift
        ;;
      --kirin9030)
        ENABLE_KIRIN9030=true
        shift
        ;;
      --kirinX90)
        ENABLE_KIRINX90=true
        shift
        ;;
      --sim)
        ARGS+=" -r sim "
        shift
        ;;
      --npu)
        ARGS+="-r npu "
        shift
        ;;
      --comm)
        ENABLE_COMM=true
        shift
        ;;
      --simple)
        ENABLE_SIMPLE=true
        shift
        ;;
      --all)
        ENABLE_ALL=true
        shift
        ;;
      --auto_mode)
        ARGS+="-a "
        IS_AUTO_MODE=true
        shift
        ;;
      --)
        shift
        break
        ;;
      *)
        break
        ;;
    esac
  done
}

checkopts "$@"

if [ "$ENABLE_KIRIN9030" = "true" ]; then
  python3 tests/script/build_st.py $ARGS -v kirin9030 -t all
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t textract
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmov
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tadd
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolsum
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tpartadd
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowsum
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tsort32
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcvt
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmrgsort
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tgather
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tsub
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmatmul
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tload
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t ttrans
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tstore
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowexpand
fi

if [ "$ENABLE_COMM" == "true" ]; then
  if [ "$ENABLE_A3" = "true" ]; then
    python3 tests/script/run_st.py $ARGS -v a3 -t comm/tput -g TPut.Vec_Int32Large -n -2
    python3 tests/script/run_st.py $ARGS -v a3 -t comm/tput -g TPut.Shape2D_Float16x16 -n -2
    python3 tests/script/run_st.py $ARGS -v a3 -t comm/tget -g TGet.Vec_Int32Large -n -2
    python3 tests/script/run_st.py $ARGS -v a3 -t comm/tget
    python3 tests/script/run_st.py $ARGS -v a3 -t comm/tput
  fi
fi
