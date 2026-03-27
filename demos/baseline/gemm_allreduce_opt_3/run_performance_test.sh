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
#
# GEMM AllReduce 性能测试入口脚本
# 用法见下方 USAGE 或执行: ./run_performance_test.sh --help
#

set -eo pipefail
trap 'echo "[ERROR] 脚本在 line $LINENO 失败 (exit $?)"' ERR

echo "正在加载环境..."
# Source CANN environment (不使用 set -u，避免 CANN set_env.sh 中未定义变量如 PYTHONPATH 报错)
CANN_SETENV=/usr/local/Ascend/cann-8.5.0/set_env.sh
SHMEM_SETENV=/home/ntlab/qifeng/pypto/third_party_path/shmem/install/set_env.sh
if [ ! -f "$CANN_SETENV" ]; then
    echo "[ERROR] CANN 环境不存在: $CANN_SETENV"
    exit 1
fi
if [ ! -f "$SHMEM_SETENV" ]; then
    echo "[ERROR] Shmem 环境不存在: $SHMEM_SETENV"
    exit 1
fi
echo "正在加载 CANN..."
set +e
source "$CANN_SETENV"
_cann_ret=$?
set -e
if [ $_cann_ret -ne 0 ]; then
    echo "[ERROR] source CANN 失败 (exit $_cann_ret)"
    exit 1
fi
echo "正在加载 Shmem..."
set +e
source "$SHMEM_SETENV"
_shmem_ret=$?
set -e
if [ $_shmem_ret -ne 0 ]; then
    echo "[ERROR] source Shmem 失败 (exit $_shmem_ret)"
    exit 1
fi
echo "环境加载完成。"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-$HOME/.local/lib64/cmake}"

# Conda libstdc++
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${CONDA_PREFIX}/aarch64-conda-linux-gnu/lib:${LD_LIBRARY_PATH:-}
elif [ -d "${HOME}/miniconda3/envs/pypto_haoran/lib" ]; then
    export LD_LIBRARY_PATH=${HOME}/miniconda3/envs/pypto_haoran/lib:${HOME}/miniconda3/envs/pypto_haoran/aarch64-conda-linux-gnu/lib:${LD_LIBRARY_PATH:-}
fi

NRANKS=8
FIRST_DEVICE=0
SOC_VERSION=Ascend910B1
RUN_MODE=npu
SKIP_BUILD=false
HELP=false

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -n, --nranks N           rank 数量 (默认: 8)"
    echo "  -d, --first-device ID   首个设备 ID (默认: 0)"
    echo "  -v, --soc-version VER    SoC 版本 (默认: Ascend910B1)"
    echo "  -s, --skip-build        不重新构建，仅运行 (需已存在 build/)"
    echo "  -h, --help               显示此帮助"
    echo ""
    echo "示例:"
    echo "  $0 --nranks 8              # 构建并运行 8 rank 性能测试"
    echo "  $0 -n 4 -d 0               # 4 rank，从 device 0 开始"
    echo "  $0 -n 8 --skip-build       # 仅运行，不重新编译"
    exit 0
}

SHORT=n:,d:,v:,s,h
LONG=nranks:,first-device:,soc-version:,skip-build,help
if [ $# -gt 0 ]; then
    OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@" 2>/dev/null) || true
    if [ -n "$OPTS" ]; then
        eval set -- "$OPTS"
    fi
fi
while [ $# -gt 0 ]; do
    case "$1" in
        -n|--nranks)        NRANKS="$2"; shift 2 ;;
        -d|--first-device)  FIRST_DEVICE="$2"; shift 2 ;;
        -v|--soc-version)   SOC_VERSION="$2"; shift 2 ;;
        -s|--skip-build)    SKIP_BUILD=true; shift ;;
        -h|--help)          HELP=true; shift ;;
        --) shift; break ;;
        *) shift ;;
    esac
done

if [ "$HELP" = true ]; then
    usage
fi

export NUM_RANKS="${NRANKS}"

echo "=== GEMM AllReduce 性能测试 ==="
echo "  NRANKS: ${NRANKS}"
echo "  FIRST_DEVICE: ${FIRST_DEVICE}"
echo "  SOC_VERSION: ${SOC_VERSION}"
echo "  SKIP_BUILD: ${SKIP_BUILD}"
echo "==============================="

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# 生成输入数据（与 run.sh 一致）
if [ -f "./scripts/gen_data.py" ]; then
    GENDATA_PYTHON="${GENDATA_PYTHON:-python}"
    if ! "$GENDATA_PYTHON" -c "import numpy" 2>/dev/null; then
        for _try_py in /home/ntlab/miniconda3/envs/pypto_haoran/bin/python /home/ntlab/miniconda3/envs/pto-comm-isa/bin/python; do
            if [ -x "$_try_py" ] && "$_try_py" -c "import numpy" 2>/dev/null; then
                GENDATA_PYTHON="$_try_py"
                break
            fi
        done
    fi
    echo "  gen_data python: $GENDATA_PYTHON"
    "$GENDATA_PYTHON" ./scripts/gen_data.py --nranks "${NRANKS}"
fi

# 构建
if [ "$SKIP_BUILD" != true ]; then
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}
    rm -rf build
    mkdir -p build
    cd build
    cmake -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}" ..
    make -j16
    cd ..
fi

# 运行性能测试（主程序内置 Warmup / Compute-Only / Sequential / Pipelined 统计）
cd build
./gemm_allreduce --nranks "${NRANKS}" --first-device "${FIRST_DEVICE}"

echo ""
echo "性能测试完成。如需核数调优，请运行: ./run_optimal_config.sh --nranks ${NRANKS}"
