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
#include <gtest/gtest.h>
#include <acl/acl.h>
#include <pto/common/type.hpp>

using namespace std;
using namespace PtoTestCommon;

template <bool isBf16, int kTRows_, int kTCols_, int num_blocks, int vRows, int vCols>
void LaunchEngramGradWReduce(
    void *grad_weight_hidden,
    void *grad_weight_embed,
    void *grad_w_partial,
    void *weight_hidden,
    void *weight_embed,
    void *stream);

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

class EngramGradWReduceTest : public testing::Test {
public:
    aclrtStream stream;
    void *gradWhHost;
    void *gradWeHost;
    void *partialHost;
    void *whHost;
    void *weHost;
    void *gradWhDevice;
    void *gradWeDevice;
    void *partialDevice;
    void *whDevice;
    void *weDevice;

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

    template <bool isBf16, int kTRows_, int kTCols_, int num_blocks, int vRows, int vCols>
    bool TestFramework()
    {
        using T = float;
        using W = std::conditional_t<isBf16, uint16_t, float>;
        size_t gradByteSize = 1 * vCols * sizeof(T);
        size_t partialByteSize = num_blocks * vCols * sizeof(T);
        size_t weightByteSize = 1 * vCols * sizeof(W);

        aclrtMallocHost(&gradWhHost, gradByteSize);
        aclrtMallocHost(&gradWeHost, gradByteSize);
        aclrtMallocHost(&partialHost, partialByteSize);
        aclrtMallocHost(&whHost, weightByteSize);
        aclrtMallocHost(&weHost, weightByteSize);

        aclrtMalloc(&gradWhDevice, gradByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&gradWeDevice, gradByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&partialDevice, partialByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&whDevice, weightByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&weDevice, weightByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/grad_wh_init.bin", gradByteSize, gradWhHost, gradByteSize);
        ReadFile(GetGoldenDir() + "/grad_we_init.bin", gradByteSize, gradWeHost, gradByteSize);
        ReadFile(GetGoldenDir() + "/grad_w_partial.bin", partialByteSize, partialHost, partialByteSize);
        ReadFile(GetGoldenDir() + "/weight_hidden.bin", weightByteSize, whHost, weightByteSize);
        ReadFile(GetGoldenDir() + "/weight_embed.bin", weightByteSize, weHost, weightByteSize);

        aclrtMemcpy(gradWhDevice, gradByteSize, gradWhHost, gradByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(gradWeDevice, gradByteSize, gradWeHost, gradByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(partialDevice, partialByteSize, partialHost, partialByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(whDevice, weightByteSize, whHost, weightByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(weDevice, weightByteSize, weHost, weightByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        LaunchEngramGradWReduce<isBf16, kTRows_, kTCols_, num_blocks, vRows, vCols>(
            gradWhDevice, gradWeDevice, partialDevice, whDevice, weDevice, stream);

        aclrtSynchronizeStream(stream);

        aclrtMemcpy(gradWhHost, gradByteSize, gradWhDevice, gradByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(gradWeHost, gradByteSize, gradWeDevice, gradByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

        WriteFile(GetGoldenDir() + "/grad_wh_output.bin", gradWhHost, gradByteSize);
        WriteFile(GetGoldenDir() + "/grad_we_output.bin", gradWeHost, gradByteSize);

        aclrtFree(gradWhDevice);
        aclrtFree(gradWeDevice);
        aclrtFree(partialDevice);
        aclrtFree(whDevice);
        aclrtFree(weDevice);

        aclrtFreeHost(gradWhHost);
        aclrtFreeHost(gradWeHost);
        aclrtFreeHost(partialHost);
        aclrtFreeHost(whHost);
        aclrtFreeHost(weHost);

        return CompareGolden<T>(gradByteSize);
    }

    template <typename T>
    bool CompareGolden(size_t byteSize, bool printAllEn = false)
    {
        std::vector<T> goldenWh(byteSize);
        std::vector<T> goldenWe(byteSize);
        std::vector<T> resultWh(byteSize);
        std::vector<T> resultWe(byteSize);
        float eps = 0.001f;

        ReadFile(GetGoldenDir() + "/golden_grad_wh.bin", byteSize, goldenWh.data(), byteSize);
        ReadFile(GetGoldenDir() + "/golden_grad_we.bin", byteSize, goldenWe.data(), byteSize);
        ReadFile(GetGoldenDir() + "/grad_wh_output.bin", byteSize, resultWh.data(), byteSize);
        ReadFile(GetGoldenDir() + "/grad_we_output.bin", byteSize, resultWe.data(), byteSize);

        bool retWh = ResultCmp(goldenWh, resultWh, eps, 0, 1000, printAllEn, true);
        bool retWe = ResultCmp(goldenWe, resultWe, eps, 0, 1000, printAllEn, true);

        return retWh && retWe;
    }
};

TEST_F(EngramGradWReduceTest, case_4blocks_128cols)
{
    bool ret = TestFramework<true, 4, 128, 4, 4, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(EngramGradWReduceTest, case_8blocks_256cols)
{
    bool ret = TestFramework<true, 8, 256, 8, 8, 256>();
    EXPECT_TRUE(ret);
}

TEST_F(EngramGradWReduceTest, case_16blocks_512cols)
{
    bool ret = TestFramework<true, 16, 512, 16, 16, 512>();
    EXPECT_TRUE(ret);
}