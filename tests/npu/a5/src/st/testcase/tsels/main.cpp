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

using namespace std;
using namespace PtoTestCommon;

class TSELSTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir() {
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}


template <typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW,
    int srcTileH, int srcTileW, int vRows, int vCols>
void LaunchTSels(T *out, TMask *mask, T *src, T scalar, void *stream);

template<typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW,
    int srcTileH, int srcTileW, int vRows, int vCols>
void test_tsels()
{
    size_t dstFileSize = sizeof(T) * dstTileH * dstTileW;
    size_t maskFileSize = sizeof(TMask) * maskTileH * maskTileW;
    size_t srcFileSize = sizeof(T) * srcTileH * srcTileW;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    TMask *maskHost, *maskDevice;
    T *dstDevice, *srcDevice;
    T scalar;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&maskHost), maskFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&maskDevice, maskFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/mask.bin", maskFileSize, maskHost, maskFileSize);
    ReadFile(GetGoldenDir() + "/input1.bin", srcFileSize, srcHost, srcFileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", sizeof(scalar), &scalar, sizeof(scalar));

    aclrtMemcpy(dstDevice, dstFileSize, dstHost, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(maskDevice, maskFileSize, maskHost, maskFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTSels<T, TMask, dstTileH, dstTileW, maskTileH, maskTileW, srcTileH, srcTileW, vRows, vCols>
        (dstDevice, maskDevice, srcDevice, scalar, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(maskDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(maskHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.0001f);

    EXPECT_TRUE(ret);
}

TEST_F(TSELSTest, case_float_uint8_64x64_64x64_64x64_64x64) {
    test_tsels<float, uint8_t, 64, 64, 64, 64, 64, 64, 64, 64>();
}
