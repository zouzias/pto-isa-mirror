#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set +e
set -o pipefail

echo "${ut_type:-}"
echo "${TARGET_BRANCH:-}"
echo "${obs_path:-}"

export USE_CCACHE=1
export PATH=/usr/local/ccache/bin:$PATH
export CCACHE_SECONDARY_STORAGE=redis://10.0.0.135:6379
export CCACHE_COMPILERCHECK=content
export CCACHE_SLOPPINESS=include_file_mtime,time_macros,include_file_ctime
export CCACHE_UMASK=002
export CMAKE_CXX_COMPILER_LAUNCHER=/usr/local/ccache/bin/ccache

sudo apt update
sudo apt install -y redis-tools libhiredis-dev libhiredis0.14
if redis-cli -h 10.0.0.135 -p 6379 ping >/dev/null 2>&1; then
    echo "Redis connection OK"
else
    echo "Redis connection FAILED"
fi
/usr/local/ccache/bin/ccache -V
/usr/local/ccache/bin/ccache -z
grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2
sudo update-alternatives --set gcc /usr/bin/gcc-14
gcc --version

# Color definitions for log output
Purple="\033[0;35m"
BPurple="\033[1;35m"
Color_Off="\033[0m"

# Print a section header
function LOG_HEAD() {
    echo -e "${BPurple}[Header]${Color_Off} ${Purple}$*${Color_Off}"
}

# Print and execute a command, capturing its exit code in the global variable ${ret}
function LOG_DO() {
    local date_time
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BPurple}[Command]${Color_Off} ${date_time} ${Purple}$*${Color_Off}"
    "$@" && ret=0 || ret=$?
    return "${ret}"
}

main() {
    cd "${WORKSPACE}" || exit

    LOG_HEAD "Start run c++ testcase"
    echo "Y" | apt install libgtest-dev libgmock-dev
    gcc --version
    rm -rf /opt/rh/devtoolset-7
    bisheng -v
    sudo apt-get update && sudo apt-get install clang -y
    clang --version
    clang++ --version

    # Only run UT tests on the master branch
    if [[ "${TARGET_BRANCH}" != "master" ]]; then
        echo "Skip UT test on non-master branch"
        exit 0
    fi

    if [[ "${task_name}" == "A3" ]]; then
        LOG_DO python3 tests/script/build_st.py -a -r npu -v a3 -t all
    elif [[ "${task_name}" == "A5" ]]; then
        LOG_DO python3 tests/script/build_st.py -a -r npu -v a5 -t all
    else
        LOG_DO bash build.sh --cpu
    fi

    if [[ ${ret} -ne 0 ]]; then
        echo "ERROR: UT testcase build failed"
        exit 1
    fi
    echo "Run UT TESTCASE"
}

main "$@"
