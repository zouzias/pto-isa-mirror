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

class TANDSTest : public testing::Test {
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

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int vRows, int vCols, bool sameTile>
void LaunchTAndS(T *out, T *src, T scalar, void *stream);

template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols,
    bool sameTile = (dstTileH == srcTileH && dstTileW == srcTileW)>
void test_tands() {
    size_t fileSizeDst = dstTileH * dstTileW * sizeof(T);
    size_t fileSizeSrc0 = srcTileH * srcTileW * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host;
    T *dstDevice, *src0Device;
    T scalar;

    aclrtMallocHost((void **)(&dstHost), fileSizeDst);
    aclrtMallocHost((void **)(&src0Host), fileSizeSrc0);

    aclrtMalloc((void **)&dstDevice, fileSizeDst, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSizeSrc0, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSizeSrc0, src0Host, fileSizeSrc0);
    ReadFile(GetGoldenDir() + "/input2.bin", sizeof(T), &scalar, sizeof(T));

    aclrtMemcpy(src0Device, fileSizeSrc0, src0Host, fileSizeSrc0, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTAndS<T, dstTileH, dstTileW, srcTileH, srcTileW, vRows, vCols, sameTile>
        (dstDevice, src0Device, scalar, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSizeDst, dstDevice, fileSizeDst, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSizeDst);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSizeDst);
    std::vector<T> devFinal(fileSizeDst);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSizeDst, golden.data(), fileSizeDst);
    ReadFile(GetGoldenDir() + "/output.bin", fileSizeDst, devFinal.data(), fileSizeDst);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TANDSTest, case_int32_64x64_64x64_64x64) {
    test_tands<int32_t, 64, 64, 64, 64, 64, 64>();
}
TEST_F(TANDSTest, case_int16_64x64_64x64_64x64_64x64) {
    test_tands<int16_t, 64, 64, 64, 64, 64, 64>();
}
TEST_F(TANDSTest, case_int16_32x128_32x128_32x256_32x128) {
    test_tands<int16_t, 32, 128, 32, 128, 32, 256>();
}
TEST_F(TANDSTest, case_int32_16x32_16x64_16x32_16x32) {
    test_tands<int32_t, 16, 32, 16, 64, 16, 32, 16, 32>();
}
TEST_F(TANDSTest, case_int16_32x128_32x128_32x256_32x127) {
    test_tands<int16_t, 32, 128, 32, 128, 32, 256, 32, 127>();
}
TEST_F(TANDSTest, case_int32_16x32_16x64_16x32_16x31) {
    test_tands<int32_t, 16, 32, 16, 64, 16, 32, 16, 31>();
}
