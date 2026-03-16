/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include <gtest/gtest.h>
#include <acl/acl.h>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTPHILOXRANDOMTestCase(void *out, aclrtStream stream);

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

class TPHILOXRANDOMTest : public testing::Test {
public:
    aclrtStream stream;
    void *dstHost;
    void *dstDevice;

protected:
    void SetUp() override
    {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclrtCreateStream(&stream);
    }

    void TearDown() override
    {
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }

    template <typename T>
    bool CompareGolden(size_t byteSize, bool printAllEn = false)
    {
        std::vector<T> golden(byteSize);
        std::vector<T> result(byteSize);
        ReadFile(GetGoldenDir() + "/golden.bin", byteSize, golden.data(), byteSize);
        ReadFile(GetGoldenDir() + "/output.bin", byteSize, result.data(), byteSize);

        float eps = 0.001f;
        if (printAllEn) {
            return ResultCmp(golden, result, eps, 0, 1000, true);
        }
        return ResultCmp(golden, result, eps, 0, 1000, false, true);
    }

    template <uint32_t caseId, typename T, int rows, int cols, int validRows, int validCols>
    bool TPHILOXRANDOMTestFramework()
    {
        size_t byteSize = rows * cols * sizeof(T);
        aclrtMallocHost(&dstHost, byteSize);
        aclrtMalloc(&dstDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);

        launchTPHILOXRANDOMTestCase<caseId>(dstDevice, stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(dstHost, byteSize, dstDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        WriteFile(GetGoldenDir() + "/output.bin", dstHost, byteSize);

        aclrtFree(dstDevice);
        aclrtFreeHost(dstHost);

        return CompareGolden<T>(byteSize);
    }
};

TEST_F(TPHILOXRANDOMTest, case01)
{
    bool ret = TPHILOXRANDOMTestFramework<1, uint32_t, 4, 64, 4, 64>();
    EXPECT_TRUE(ret);
}
