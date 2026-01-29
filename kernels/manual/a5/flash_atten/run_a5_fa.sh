#!/bin/bash
RUN_MODE=${1}
S0=${2}
S1=${S0}
msprof bash run.sh -r ${RUN_MODE} -v Ascend910_9599 --cases "128,${S0},${S1},128,128" --qk-preload 2
