#!/bin/bash

# 批量执行 costmodel 测试的脚本
# 功能：进入指定目录，批量执行多个 testcase 的测试命令

# ===================== 配置区（可根据需要修改）=====================
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 目标工作目录（pto-isa-costmodel 根目录）
TARGET_DIR="${SCRIPT_DIR}/.."

# 需要执行的测试用例列表（可按需添加/删除）
TESTCASES=("tadd" "tmul" "tsub")

# 测试命令的固定参数
TEST_ARGS="--clean --verbose"
# ==================================================================

# 颜色定义（用于终端输出提示）
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # 重置颜色

# 函数：打印错误信息并退出
error_exit() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# 1. 检查目标目录是否存在
if [ ! -d "${TARGET_DIR}" ]; then
    error_exit "目标目录不存在：${TARGET_DIR}"
fi

# 2. 进入目标目录
echo -e "${YELLOW}[INFO] 进入目录：${TARGET_DIR}${NC}"
cd "${TARGET_DIR}" || error_exit "无法进入目录：${TARGET_DIR}"

# 3. 遍历测试用例并执行
for testcase in "${TESTCASES[@]}"; do
    echo -e "\n========================================"
    echo -e "${YELLOW}[INFO] Start Test Case:${testcase}${NC}"
    echo -e "========================================"

    # 构建测试命令
    test_cmd="python tests/run_costmodel.py --testcase ${testcase} ${TEST_ARGS}"
    echo -e "${YELLOW}[INFO] Execute cmd:${test_cmd}${NC}"

    # 执行命令并捕获退出码
    ${test_cmd}
    exit_code=$?

    # 根据退出码判断执行结果
    if [ ${exit_code} -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] Test Case ${testcase} Finished${NC}"
    else
        echo -e "${RED}[FAIL] Test Case ${testcase} Failed (Exit Code: ${exit_code})${NC}"
        # 可选：如果某个用例失败是否继续执行后续用例
        # error_exit "测试用例 ${testcase} 执行失败，终止脚本"
    fi
done

# 4. 脚本执行完成
echo -e "\n${GREEN}[INFO] All Test Case Finished${NC}"
exit 0