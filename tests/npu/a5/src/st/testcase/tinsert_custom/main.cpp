/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <algorithm>

#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"

#define TINSERT_CUSTOM_DEBUG 0
#define DEBUG_PRINT_COUNT 20

using namespace std;
using namespace PtoTestCommon;

template <int32_t testKey>
void launchTInsertCustom(uint64_t *out, uint64_t *src, void *stream);

template <typename T>
void printDebugValues(const std::string &label, const T *data, size_t count, size_t maxPrint = 100)
{
#if TINSERT_CUSTOM_DEBUG
    size_t printCount = std::min(count, maxPrint);
    std::cout << BLUE << "\n[DEBUG] " << label
               << " (first " << printCount << " of " << count << " values):" << RESET << std::endl;

    std::cout << std::fixed << std::setprecision(4);

    for (size_t i = 0; i < printCount; ++i) {
        std::cout << "  " << YELLOW << "[" << std::setw(3) << i << "]" << RESET
                   << " = " << GREEN << static_cast<double>(data[i]) << RESET;

        if ((i + 1) % 5 == 0) {
            std::cout << std::endl;
        }
    }

    if (printCount % 5 != 0) {
        std::cout << std::endl;
    }
#endif
}

template <typename T>
void printDiffAnalysis(const std::vector<T> &golden, const std::vector<T> &result)
{
#if TINSERT_CUSTOM_DEBUG
    double maxDiff = 0.0;
    size_t maxDiffIdx = 0;
    size_t errCount = 0;
    double tolerance = 0.001;

    for (size_t i = 0; i < golden.size(); ++i) {
        double diff = std::abs(static_cast<double>(golden[i]) - static_cast<double>(result[i]));
        if (diff > tolerance) {
            ++errCount;
        }
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIdx = i;
        }
    }

    std::cout << "\n[DEBUG] Diff Analysis:" << std::endl;
    std::cout << "  Total elements: " << golden.size() << std::endl;
    std::cout << "  Error count (>" << tolerance << "): " << errCount << std::endl;
    std::cout << "  Max diff: " << maxDiff << " at index " << maxDiffIdx << std::endl;
    if (maxDiff > 0) {
        std::cout << "  Golden[" << maxDiffIdx << "] = " << static_cast<double>(golden[maxDiffIdx]) << std::endl;
        std::cout << "  Result[" << maxDiffIdx << "] = " << static_cast<double>(result[maxDiffIdx]) << std::endl;
    }
#endif
}

class TInsertCustomTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir() {
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <int32_t testKey, typename dType>
void testTInsertCustom(int32_t rows, int32_t cols)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t srcByteSize = rows * cols * sizeof(dType);
    size_t dstByteSize = rows * cols * sizeof(dType);
    uint64_t *dstHost, *srcHost, *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstByteSize);
    aclrtMallocHost((void**)(&srcHost), srcByteSize);
    aclrtMalloc((void**)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", srcByteSize, srcHost, srcByteSize);

#if TINSERT_CUSTOM_DEBUG
    std::cout << "\n========== Test Case " << testKey << " (" << rows << "x" << cols << ") =========" << std::endl;
    printDebugValues<dType>("Input (ND format)", reinterpret_cast<dType*>(srcHost), rows * cols);
#endif

    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTInsertCustom<testKey>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dType> golden(dstByteSize / sizeof(dType));
    std::vector<dType> devFinal(dstByteSize / sizeof(dType));
    ReadFile(GetGoldenDir() + "/golden_output.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", dstByteSize, devFinal.data(), dstByteSize);

#if TINSERT_CUSTOM_DEBUG
    printDebugValues<dType>("Golden (NZ format)", golden.data(), golden.size());
    printDebugValues<dType>("Device Output (NZ format)", devFinal.data(), devFinal.size());
    printDiffAnalysis<dType>(golden, devFinal);
#endif

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TInsertCustomTest, case1)
{
    testTInsertCustom<1, float>(16, 32);
}

TEST_F(TInsertCustomTest, case2)
{
    testTInsertCustom<2, float>(16, 32);
}

TEST_F(TInsertCustomTest, case3)
{
    testTInsertCustom<3, float>(32, 64);
}

TEST_F(TInsertCustomTest, case4)
{
    testTInsertCustom<4, int32_t>(32, 32);
}

TEST_F(TInsertCustomTest, case5)
{
    testTInsertCustom<5, float>(32, 32);
}

TEST_F(TInsertCustomTest, case6)
{
    testTInsertCustom<6, float>(32, 32);
}

TEST_F(TInsertCustomTest, case7)
{
    testTInsertCustom<7, float>(64, 64);
}