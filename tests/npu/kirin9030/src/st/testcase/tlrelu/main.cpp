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
void launchTLRELUTestCase(void *out, void *src, float scalar, aclrtStream stream);

class TLRELUTest : public testing::Test {
public:
protected:
    void SetUp() override
    {}

    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <uint32_t caseId, typename T, int dstTileRow, int dstTileCol, int row, int vaildRow, int col, int srcVaildCol>
bool TLReluTestFramework()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = dstTileRow * dstTileCol * sizeof(T);
    size_t srcByteSize = row * col * sizeof(T);
    T *dstHost;
    T *srcHost;
    T *dstDevice;
    T *srcDevice;
    float scalar;

    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);

    aclrtMalloc((void **)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
    std::string scalar_file = GetGoldenDir() + "/divider.bin";
    std::ifstream file(scalar_file, std::ios::binary);

    file.read(reinterpret_cast<char *>(&scalar), 4);
    file.close();

    aclrtMemset(dstHost, dstByteSize, 0, dstByteSize);

    aclrtMemcpy(dstDevice, dstByteSize, dstHost, dstByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTLRELUTestCase<caseId>(dstDevice, srcDevice, scalar, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstByteSize);
    std::vector<T> devFinal(dstByteSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    return ResultCmp<T>(golden, devFinal, 0.001f);
}

TEST_F(TLRELUTest, case1)
{
    bool ret = TLReluTestFramework<1, float, 32, 128, 32, 32, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TLRELUTest, case2)
{
    bool ret = TLReluTestFramework<2, aclFloat16, 63, 128, 63, 63, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TLRELUTest, case3)
{
    bool ret = TLReluTestFramework<3, float, 7, 512, 7, 7, 448, 448>();
    EXPECT_TRUE(ret);
}

TEST_F(TLRELUTest, case4)
{
    bool ret = TLReluTestFramework<4, float, 256, 32, 256, 256, 16, 16>();
    EXPECT_TRUE(ret);
}

// Group A: FP32 (block_size=64, align_unit=8) - 8 systematic cases, col 32B aligned
TEST_F(TLRELUTest, case5)
{
    bool ret = TLReluTestFramework<5, float, 8, 64, 8, 8, 64, 64>();
    EXPECT_TRUE(ret);
}
TEST_F(TLRELUTest, case6)
{
    bool ret = TLReluTestFramework<6, float, 8, 64, 8, 8, 48, 48>();
    EXPECT_TRUE(ret);
}
TEST_F(TLRELUTest, case7)
{
    bool ret = TLReluTestFramework<7, float, 8, 64, 8, 8, 56, 56>();
    EXPECT_TRUE(ret);
}
TEST_F(TLRELUTest, case8)
{
    bool ret = TLReluTestFramework<8, float, 12, 64, 12, 8, 48, 48>();
    EXPECT_TRUE(ret);
}
TEST_F(TLRELUTest, case9)
{
    bool ret = TLReluTestFramework<9, float, 4, 96, 4, 4, 96, 96>();
    EXPECT_TRUE(ret);
}
TEST_F(TLRELUTest, case10)
{
    bool ret = TLReluTestFramework<10, float, 4, 96, 4, 4, 72, 72>();
    EXPECT_TRUE(ret);
}
TEST_F(TLRELUTest, case11)
{
    bool ret = TLReluTestFramework<11, float, 4, 96, 4, 4, 80, 80>();
    EXPECT_TRUE(ret);
}
TEST_F(TLRELUTest, case12)
{
    bool ret = TLReluTestFramework<12, float, 8, 96, 8, 4, 80, 80>();
    EXPECT_TRUE(ret);
}

// Group B: FP16 (block_size=128, align_unit=16) - 8 systematic cases, col 32B aligned
TEST_F(TLRELUTest, case13)
{
    bool ret = TLReluTestFramework<13, aclFloat16, 8, 64, 8, 8, 64, 64>();
    EXPECT_TRUE(ret);
}
TEST_F(TLRELUTest, case14)
{
    bool ret = TLReluTestFramework<14, aclFloat16, 8, 64, 8, 8, 48, 48>();
    EXPECT_TRUE(ret);
}
TEST_F(TLRELUTest, case15)
{
    bool ret = TLReluTestFramework<15, aclFloat16, 8, 64, 8, 8, 32, 32>();
    EXPECT_TRUE(ret);
}
TEST_F(TLRELUTest, case16)
{
    bool ret = TLReluTestFramework<16, aclFloat16, 12, 64, 12, 8, 48, 48>();
    EXPECT_TRUE(ret);
}
TEST_F(TLRELUTest, case17)
{
    bool ret = TLReluTestFramework<17, aclFloat16, 2, 144, 2, 2, 144, 144>();
    EXPECT_TRUE(ret);
}
TEST_F(TLRELUTest, case18)
{
    bool ret = TLReluTestFramework<18, aclFloat16, 2, 144, 2, 2, 128, 128>();
    EXPECT_TRUE(ret);
}
TEST_F(TLRELUTest, case19)
{
    bool ret = TLReluTestFramework<19, aclFloat16, 2, 144, 2, 2, 112, 112>();
    EXPECT_TRUE(ret);
}
TEST_F(TLRELUTest, case20)
{
    bool ret = TLReluTestFramework<20, aclFloat16, 4, 144, 4, 2, 112, 112>();
    EXPECT_TRUE(ret);
}