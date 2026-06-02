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

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTCOLCMINTestCase(void *out, void *src, aclrtStream stream);

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

class TCOLCMINTest : public testing::Test {
public:
    aclrtStream stream;
    void *dstHost;
    void *srcHost;
    void *dstDevice;
    void *srcDevice;

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

    // template <typename T>
    bool CompareGolden(size_t dstByteSize, bool printAllEn = false)
    {
        std::vector<uint32_t> golden(dstByteSize);
        std::vector<uint32_t> result(dstByteSize);
        float eps = 0.001f;
        ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
        ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, result.data(), dstByteSize);
        if (printAllEn) {
            return ResultCmp(golden, result, eps, 0, 1000, true);
        }
        return ResultCmp(golden, result, eps, 0, 1000, false, true);
    }

    template <uint32_t caseId, typename T, int srcRow, int srcValidRow, int dstRow, int col, int validCol>
    bool TCOLCMINTestFramework()
    {
        size_t dstByteSize = dstRow * col * sizeof(uint32_t);
        size_t srcByteSize = srcRow * col * sizeof(T);
        aclrtMallocHost(&dstHost, dstByteSize);
        aclrtMallocHost(&srcHost, srcByteSize);
        aclrtMalloc(&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
        aclrtMemset(dstHost, dstByteSize, 0, dstByteSize);

        aclrtMemcpy(dstDevice, dstByteSize, dstHost, dstByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTCOLCMINTestCase<caseId>(dstDevice, srcDevice, stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

        aclrtFree(dstDevice);
        aclrtFree(srcDevice);
        aclrtFreeHost(dstHost);
        aclrtFreeHost(srcHost);

        return CompareGolden(dstByteSize);
    }
};

TEST_F(TCOLCMINTest, case01)
{
    bool ret = TCOLCMINTestFramework<1, float, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case02)
{
    bool ret = TCOLCMINTestFramework<2, float, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case03)
{
    bool ret = TCOLCMINTestFramework<3, float, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case11)
{
    bool ret = TCOLCMINTestFramework<11, aclFloat16, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case12)
{
    bool ret = TCOLCMINTestFramework<12, aclFloat16, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case13)
{
    bool ret = TCOLCMINTestFramework<13, aclFloat16, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case21)
{
    bool ret = TCOLCMINTestFramework<21, int8_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case22)
{
    bool ret = TCOLCMINTestFramework<22, int8_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case23)
{
    bool ret = TCOLCMINTestFramework<23, int8_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case31)
{
    bool ret = TCOLCMINTestFramework<31, uint8_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case32)
{
    bool ret = TCOLCMINTestFramework<32, uint8_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case33)
{
    bool ret = TCOLCMINTestFramework<33, uint8_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLCMINTest, case41)
{
    bool ret = TCOLCMINTestFramework<41, int16_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case42)
{
    bool ret = TCOLCMINTestFramework<42, int16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case43)
{
    bool ret = TCOLCMINTestFramework<43, int16_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case51)
{
    bool ret = TCOLCMINTestFramework<51, uint16_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case52)
{
    bool ret = TCOLCMINTestFramework<52, uint16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case53)
{
    bool ret = TCOLCMINTestFramework<53, uint16_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case61)
{
    bool ret = TCOLCMINTestFramework<61, int32_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case62)
{
    bool ret = TCOLCMINTestFramework<62, int32_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case63)
{
    bool ret = TCOLCMINTestFramework<63, int32_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case71)
{
    bool ret = TCOLCMINTestFramework<71, uint32_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case72)
{
    bool ret = TCOLCMINTestFramework<72, uint32_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case73)
{
    bool ret = TCOLCMINTestFramework<73, uint32_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

// Group A: FP32 (block_size=64, align_unit=8) - 8 systematic cases
TEST_F(TCOLCMINTest, case81)
{
    bool ret = TCOLCMINTestFramework<81, float, 8, 8, 1, 64, 64>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case82)
{
    bool ret = TCOLCMINTestFramework<82, float, 8, 8, 1, 64, 48>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case83)
{
    bool ret = TCOLCMINTestFramework<83, float, 8, 8, 1, 64, 63>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case84)
{
    bool ret = TCOLCMINTestFramework<84, float, 12, 8, 1, 64, 48>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case85)
{
    bool ret = TCOLCMINTestFramework<85, float, 4, 4, 1, 96, 96>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case86)
{
    bool ret = TCOLCMINTestFramework<86, float, 4, 4, 1, 96, 72>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case87)
{
    bool ret = TCOLCMINTestFramework<87, float, 4, 4, 1, 96, 65>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case88)
{
    bool ret = TCOLCMINTestFramework<88, float, 8, 4, 1, 96, 65>();
    EXPECT_TRUE(ret);
}

// Group B: FP16 (block_size=128, align_unit=16) - 8 systematic cases
TEST_F(TCOLCMINTest, case91)
{
    bool ret = TCOLCMINTestFramework<91, aclFloat16, 8, 8, 1, 64, 64>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case92)
{
    bool ret = TCOLCMINTestFramework<92, aclFloat16, 8, 8, 1, 64, 48>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case93)
{
    bool ret = TCOLCMINTestFramework<93, aclFloat16, 8, 8, 1, 64, 33>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case94)
{
    bool ret = TCOLCMINTestFramework<94, aclFloat16, 12, 8, 1, 64, 48>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case95)
{
    bool ret = TCOLCMINTestFramework<95, aclFloat16, 2, 2, 1, 144, 144>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case96)
{
    bool ret = TCOLCMINTestFramework<96, aclFloat16, 2, 2, 1, 144, 128>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case97)
{
    bool ret = TCOLCMINTestFramework<97, aclFloat16, 2, 2, 1, 144, 129>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMINTest, case98)
{
    bool ret = TCOLCMINTestFramework<98, aclFloat16, 4, 2, 1, 144, 129>();
    EXPECT_TRUE(ret);
}
