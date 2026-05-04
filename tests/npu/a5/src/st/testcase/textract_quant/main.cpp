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
#include "acl/acl.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

template <int testKey>
void launchTEXTRACTQuant(uint8_t* out, uint8_t* srcIn, uint8_t* quantIn, void* stream);

class TEXTRACTQuantTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <int testKey, typename SrcT, typename DstT>
void runTEXTRACTQuantTest(
    uint32_t srcRows, uint32_t srcCols, 
    uint32_t dstRows, uint32_t dstCols,
    bool isVQuant)
{
    size_t srcFileSize = srcRows * srcCols * sizeof(SrcT);
    size_t dstFileSize = dstRows * dstCols * sizeof(DstT);
    size_t quantFileSize = isVQuant ? dstCols * sizeof(uint64_t) : sizeof(uint64_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcHost, *quantHost;
    uint8_t *dstDevice, *srcDevice, *quantDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&quantHost), quantFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&quantDevice, quantFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", srcFileSize, srcHost, srcFileSize);
    ReadFile(GetGoldenDir() + "/quant.bin", quantFileSize, quantHost, quantFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(quantDevice, quantFileSize, quantHost, quantFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTEXTRACTQuant<testKey>(dstDevice, srcDevice, quantDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(quantDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(quantHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<DstT> golden(dstFileSize / sizeof(DstT));
    std::vector<DstT> devFinal(dstFileSize / sizeof(DstT));
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);
    
    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TEXTRACTQuantTest, case_1) {
    runTEXTRACTQuantTest<1, int32_t, int8_t>(128, 64, 128, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_2) {
    runTEXTRACTQuantTest<2, int32_t, int8_t>(128, 64, 96, 32, false);
}
TEST_F(TEXTRACTQuantTest, case_3) {
    runTEXTRACTQuantTest<3, int32_t, int8_t>(128, 128, 64, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_4) {
    runTEXTRACTQuantTest<4, int32_t, int8_t>(256, 128, 128, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_5) {
    runTEXTRACTQuantTest<5, int32_t, int8_t>(128, 64, 64, 32, true);
}
TEST_F(TEXTRACTQuantTest, case_6) {
    runTEXTRACTQuantTest<6, int32_t, int8_t>(96, 96, 64, 64, true);
}
TEST_F(TEXTRACTQuantTest, case_7) {
    runTEXTRACTQuantTest<7, int32_t, int8_t>(128, 128, 96, 96, true);
}
TEST_F(TEXTRACTQuantTest, case_8) {
    runTEXTRACTQuantTest<8, int32_t, int8_t>(256, 64, 128, 32, true);
}

TEST_F(TEXTRACTQuantTest, case_9) {
    runTEXTRACTQuantTest<9, int32_t, uint8_t>(128, 64, 128, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_10) {
    runTEXTRACTQuantTest<10, int32_t, uint8_t>(128, 64, 96, 32, false);
}
TEST_F(TEXTRACTQuantTest, case_11) {
    runTEXTRACTQuantTest<11, int32_t, uint8_t>(128, 128, 64, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_12) {
    runTEXTRACTQuantTest<12, int32_t, uint8_t>(256, 128, 128, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_13) {
    runTEXTRACTQuantTest<13, int32_t, uint8_t>(128, 64, 64, 32, true);
}
TEST_F(TEXTRACTQuantTest, case_14) {
    runTEXTRACTQuantTest<14, int32_t, uint8_t>(96, 96, 64, 64, true);
}
TEST_F(TEXTRACTQuantTest, case_15) {
    runTEXTRACTQuantTest<15, int32_t, uint8_t>(128, 128, 96, 96, true);
}
TEST_F(TEXTRACTQuantTest, case_16) {
    runTEXTRACTQuantTest<16, int32_t, uint8_t>(256, 64, 128, 32, true);
}

TEST_F(TEXTRACTQuantTest, case_17) {
    runTEXTRACTQuantTest<17, int32_t, uint16_t>(128, 64, 128, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_18) {
    runTEXTRACTQuantTest<18, int32_t, uint16_t>(128, 64, 96, 32, false);
}
TEST_F(TEXTRACTQuantTest, case_19) {
    runTEXTRACTQuantTest<19, int32_t, uint16_t>(128, 128, 64, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_20) {
    runTEXTRACTQuantTest<20, int32_t, uint16_t>(256, 128, 128, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_21) {
    runTEXTRACTQuantTest<21, int32_t, uint16_t>(128, 64, 64, 32, true);
}
TEST_F(TEXTRACTQuantTest, case_22) {
    runTEXTRACTQuantTest<22, int32_t, uint16_t>(96, 96, 64, 64, true);
}
TEST_F(TEXTRACTQuantTest, case_23) {
    runTEXTRACTQuantTest<23, int32_t, uint16_t>(128, 128, 96, 96, true);
}
TEST_F(TEXTRACTQuantTest, case_24) {
    runTEXTRACTQuantTest<24, int32_t, uint16_t>(256, 64, 128, 32, true);
}

TEST_F(TEXTRACTQuantTest, case_25) {
    runTEXTRACTQuantTest<25, float, int8_t>(128, 64, 128, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_26) {
    runTEXTRACTQuantTest<26, float, int8_t>(128, 64, 96, 32, false);
}
TEST_F(TEXTRACTQuantTest, case_27) {
    runTEXTRACTQuantTest<27, float, int8_t>(128, 128, 64, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_28) {
    runTEXTRACTQuantTest<28, float, int8_t>(256, 128, 128, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_29) {
    runTEXTRACTQuantTest<29, float, int8_t>(128, 64, 64, 32, true);
}
TEST_F(TEXTRACTQuantTest, case_30) {
    runTEXTRACTQuantTest<30, float, int8_t>(96, 96, 64, 64, true);
}
TEST_F(TEXTRACTQuantTest, case_31) {
    runTEXTRACTQuantTest<31, float, int8_t>(128, 128, 96, 96, true);
}
TEST_F(TEXTRACTQuantTest, case_32) {
    runTEXTRACTQuantTest<32, float, int8_t>(256, 64, 128, 32, true);
}

TEST_F(TEXTRACTQuantTest, case_33) {
    runTEXTRACTQuantTest<33, float, uint8_t>(128, 64, 128, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_34) {
    runTEXTRACTQuantTest<34, float, uint8_t>(128, 64, 96, 32, false);
}
TEST_F(TEXTRACTQuantTest, case_35) {
    runTEXTRACTQuantTest<35, float, uint8_t>(128, 128, 64, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_36) {
    runTEXTRACTQuantTest<36, float, uint8_t>(256, 128, 128, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_37) {
    runTEXTRACTQuantTest<37, float, uint8_t>(128, 64, 64, 32, true);
}
TEST_F(TEXTRACTQuantTest, case_38) {
    runTEXTRACTQuantTest<38, float, uint8_t>(96, 96, 64, 64, true);
}
TEST_F(TEXTRACTQuantTest, case_39) {
    runTEXTRACTQuantTest<39, float, uint8_t>(128, 128, 96, 96, true);
}
TEST_F(TEXTRACTQuantTest, case_40) {
    runTEXTRACTQuantTest<40, float, uint8_t>(256, 64, 128, 32, true);
}

TEST_F(TEXTRACTQuantTest, case_41) {
    runTEXTRACTQuantTest<41, float, uint16_t>(128, 64, 128, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_42) {
    runTEXTRACTQuantTest<42, float, uint16_t>(128, 64, 96, 32, false);
}
TEST_F(TEXTRACTQuantTest, case_43) {
    runTEXTRACTQuantTest<43, float, uint16_t>(128, 128, 64, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_44) {
    runTEXTRACTQuantTest<44, float, uint16_t>(256, 128, 128, 64, false);
}

TEST_F(TEXTRACTQuantTest, case_45) {
    runTEXTRACTQuantTest<45, float, uint16_t>(128, 64, 128, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_46) {
    runTEXTRACTQuantTest<46, float, uint16_t>(128, 64, 96, 32, false);
}
TEST_F(TEXTRACTQuantTest, case_47) {
    runTEXTRACTQuantTest<47, float, uint16_t>(128, 128, 64, 64, false);
}
TEST_F(TEXTRACTQuantTest, case_48) {
    runTEXTRACTQuantTest<48, float, uint16_t>(256, 128, 128, 64, false);
}
