#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# 批量执行 costmodel 测试的脚本（支持 st 与 st_fit）
# ===================== 配置区（可根据需要修改）=====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET_DIR="${SCRIPT_DIR}/.."

# 测试用例目录
ST_TESTCASE_DIR="${SCRIPT_DIR}/costmodel/st/testcase"
FIT_TESTCASE_DIR="${SCRIPT_DIR}/costmodel/st_fit/testcase"

# 需要执行的测试用例列表（留空则自动发现 st 和 st_fit 下所有子目录）
# 示例：TESTCASES=("tsub" "time_predict")
TESTCASES=()

# 测试命令的固定参数
TEST_ARGS="--clean --verbose"
# ==================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    PYTHON_BIN=python
fi

# 函数：打印错误信息并退出
error_exit() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# 记录结果的数组
PASSED_CASES=()
FAILED_CASES=()
RUN_ITEMS=()
INVALID_CASES=()

# 函数：把数组里的元素做清洗，兼容 TESTCASES=("tsub", "time_predict") 这种写法
sanitize_name() {
    local name="$1"
    name="${name//,/}"
    name="${name//\"/}"
    name="${name//\'/}"
    echo "${name}"
}

# 函数：将单个用例名映射到 st/st_fit（可能两个都命中）
append_run_items_for_case() {
    local testcase="$1"
    local matched=0

    if [ -d "${ST_TESTCASE_DIR}/${testcase}" ]; then
        RUN_ITEMS+=("stub:${testcase}")
        matched=1
    fi
    if [ -d "${FIT_TESTCASE_DIR}/${testcase}" ]; then
        RUN_ITEMS+=("fit:${testcase}")
        matched=1
    fi

    if [ ${matched} -eq 0 ]; then
        echo -e "${YELLOW}[WARN] Unknown testcase skipped: ${testcase}${NC}"
        INVALID_CASES+=("${testcase}")
    fi
}

# 1. 检查目标目录是否存在
if [ ! -d "${TARGET_DIR}" ]; then
    error_exit "Dir not exists：${TARGET_DIR}"
fi

# 2. 解析执行列表
if [ ! -d "${ST_TESTCASE_DIR}" ] || [ ! -d "${FIT_TESTCASE_DIR}" ]; then
    error_exit "Testcase dir not exists：${ST_TESTCASE_DIR} or ${FIT_TESTCASE_DIR}"
fi

if [ ${#TESTCASES[@]} -eq 0 ]; then
    # Delegate "run all" to run_costmodel.py itself when --testcase is omitted.
    RUN_ITEMS+=("stub:__ALL__")
    RUN_ITEMS+=("fit:__ALL__")
    echo -e "${YELLOW}[INFO] TESTCASES is empty, run all testcases in stub and fit suites${NC}"
else
    for raw_name in "${TESTCASES[@]}"; do
        testcase="$(sanitize_name "${raw_name}")"
        [ -z "${testcase}" ] && continue
        append_run_items_for_case "${testcase}"
    done
fi

if [ ${#RUN_ITEMS[@]} -eq 0 ]; then
    error_exit "No runnable testcase found"
fi

# 3. 进入目标目录
echo -e "${YELLOW}[INFO] Enter Dir：${TARGET_DIR}${NC}"
cd "${TARGET_DIR}" || error_exit "Enter Dir Failed：${TARGET_DIR}"

# 4. 遍历测试用例并执行
for item in "${RUN_ITEMS[@]}"; do
    suite="${item%%:*}"
    testcase="${item#*:}"
    if [ "${testcase}" = "__ALL__" ]; then
        label="${suite}/all"
    else
        label="${suite}/${testcase}"
    fi

    echo -e "\n========================================"
    echo -e "${YELLOW}[INFO] Start Test Case:${label}${NC}"
    echo -e "========================================"

    # 构建测试命令
    test_cmd="${PYTHON_BIN} tests/run_costmodel.py --suite ${suite} ${TEST_ARGS}"
    if [ "${testcase}" != "__ALL__" ]; then
        test_cmd="${test_cmd} --testcase ${testcase}"
    fi
    echo -e "${YELLOW}[INFO] Execute cmd:${test_cmd}${NC}"

    # 执行命令并捕获退出码
    ${test_cmd}
    exit_code=$?

    # 根据退出码判断执行结果
    if [ ${exit_code} -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] Test Case ${label} Finished${NC}"
        PASSED_CASES+=("${label}")
    else
        echo -e "${RED}[FAIL] Test Case ${label} Failed (Exit Code: ${exit_code})${NC}"
        FAILED_CASES+=("${label}")
    fi
done

# 5. 输出总览
total=${#RUN_ITEMS[@]}
total=$((total + ${#INVALID_CASES[@]}))
passed=${#PASSED_CASES[@]}
failed=${#FAILED_CASES[@]}
failed=$((failed + ${#INVALID_CASES[@]}))

echo -e "\n========================================"
echo -e "           ST Results Summary"
echo -e "========================================"
echo -e "Total:  ${total}"
echo -e "${GREEN}Passed: ${passed}${NC}"
echo -e "${RED}Failed: ${failed}${NC}"

if [ ${passed} -gt 0 ]; then
    echo -e "\n${GREEN}[PASSED]${NC}"
    for tc in "${PASSED_CASES[@]}"; do
        echo -e "  ${GREEN}✔ ${tc}${NC}"
    done
fi

if [ ${failed} -gt 0 ]; then
    echo -e "\n${RED}[FAILED]${NC}"
    for tc in "${FAILED_CASES[@]}"; do
        echo -e "  ${RED}✘ ${tc}${NC}"
    done
    for tc in "${INVALID_CASES[@]}"; do
        echo -e "  ${RED}✘ unknown/${tc}${NC}"
    done
fi

echo -e "========================================"

if [ ${failed} -gt 0 ]; then
    exit 1
fi
exit 0
