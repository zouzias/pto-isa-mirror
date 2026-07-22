#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set +e

REPOSITORY_NAME="pto-isa"

# Color definitions for log output
Purple="\033[0;35m"
BPurple="\033[1;35m"
Color_Off="\033[0m"

# Print and execute a command, capturing its exit code in the global variable ${ret}
function LOG_DO() {
    local date_time
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BPurple}[Command]${Color_Off} ${date_time} ${Purple}$*${Color_Off}"
    "$@" && ret=0 || ret=$?
    return "${ret}"
}

#########
# Install
#########
cd "${WORKSPACE}" || exit
source /home/jenkins/Ascend/cann/bin/setenv.bash

#########
# Build
#########
echo "Y" | apt install libgtest-dev libgmock-dev
gcc --version
rm -rf /opt/rh/devtoolset-7
bisheng -v

if [[ "${task_name}" =~ API_Check ]]; then
    sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
fi

if [[ "${task_name}" == *A5* ]]; then
    LOG_DO bash build.sh --a5 --build
    if [[ ${ret} -eq 0 ]]; then
        echo "build success"
    fi
    exit 0
else
    LOG_DO bash build.sh --pkg
    if [[ ${ret} -eq 0 ]]; then
        echo "build success"
    fi
fi

# Locate the generated .run package
compile_package_name=$(ls "${WORKSPACE}/build_out/" 2>/dev/null | grep -E '\.run$' | head -n1)
if [[ -z "${compile_package_name}" ]]; then
    echo "ERROR: no .run package found in ${WORKSPACE}/build_out/"
    exit 1
fi
# 防止A5包上传OBS覆盖普通包，统一改名
mv "./build_out/${compile_package_name}" "./build_out/${package_name}"

echo "compile original package name is: ${compile_package_name}"
echo "compile package name is: ${package_name}"
chmod +x "./build_out/${package_name}"

# Install and verify the package
echo "Start to verify the package: ${package_name}"
echo "y" | "./build_out/${package_name}" --full --install-path="${WORKSPACE}/tmp" 2>&1 | tee "${WORKSPACE}/compile_log.txt"
if grep -q "ERROR" "${WORKSPACE}/compile_log.txt"; then
    echo "find key word 'ERROR' in install logs"
    exit 1
else
    echo "verify the package: ${package_name} success"
fi

# Uninstall and verify
echo "y" | "./build_out/${package_name}" --uninstall --install-path="${WORKSPACE}/tmp" 2>&1 | tee "${WORKSPACE}/compile_uninstall_log.txt"
if grep -q "ERROR" "${WORKSPACE}/compile_uninstall_log.txt"; then
    echo "find key word 'ERROR' in uninstall logs"
    exit 1
else
    echo "uninstall the package: ${compile_package_name} success"
fi

# Verify the install directory has been cleaned up
if [[ ! -d "${WORKSPACE}/tmp" ]]; then
    echo "uninstall success: directory removed"
elif [[ -z "$(ls -A "${WORKSPACE}/tmp" 2>/dev/null)" ]]; then
    echo "uninstall success: directory empty"
    rmdir "${WORKSPACE}/tmp" 2>/dev/null  # try to remove the empty directory
else
    echo "uninstall failed: ${WORKSPACE}/tmp not empty"
    ls -la "${WORKSPACE}/tmp"
    exit 1
fi
echo "package_name=${package_name}" >> "${ATOMGIT_OUTPUT}"
exit 0
