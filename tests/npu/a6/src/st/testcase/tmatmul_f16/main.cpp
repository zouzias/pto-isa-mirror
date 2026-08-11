/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
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

template <int32_t tilingKey>
void LaunchTMATMULF16(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

class TMATMUL_F16_TEST : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

// MMAD.f16f32: A[half] x B[half] -> C[float].
// Inputs are stored as uint16 (fp16); the device kernel casts internally.
template <int32_t key>
void tmatmul_f16_test(uint32_t m, uint32_t k, uint32_t n)
{
    constexpr uint32_t aElemSize = sizeof(uint16_t); // fp16
    constexpr uint32_t bElemSize = sizeof(uint16_t); // fp16
    constexpr uint32_t cElemSize = sizeof(float);    // fp32
    size_t aFileSize = m * k * aElemSize;
    size_t bFileSize = k * n * bElemSize;
    size_t cFileSize = m * n * cElemSize;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTMATMULF16<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(cFileSize / sizeof(float));
    std::vector<float> devFinal(cFileSize / sizeof(float));
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

#define DEFINE_F16_CASE(CASE_NAME, KEY, M, K, N) \
    TEST_F(TMATMUL_F16_TEST, CASE_NAME) { tmatmul_f16_test<KEY>(M, K, N); }

// --- ND (row-major) cases ---
DEFINE_F16_CASE(case_mmad_f16f32_nd_16x16x16, 1, 16, 16, 16)
DEFINE_F16_CASE(case_mmad_f16f32_nd_64x64x64, 2, 64, 64, 64)
DEFINE_F16_CASE(case_mmad_f16f32_nd_128x128x128, 3, 128, 128, 128)
DEFINE_F16_CASE(case_mmad_f16f32_nd_17x33x31, 4, 17, 33, 31)
DEFINE_F16_CASE(case_mmad_f16f32_nd_95x33x79, 5, 95, 33, 79)
DEFINE_F16_CASE(case_mmad_f16f32_nd_127x96x95, 6, 127, 96, 95)
DEFINE_F16_CASE(case_mmad_f16f32_nd_1x256x64, 7, 1, 256, 64)
DEFINE_F16_CASE(case_mmad_f16f32_nd_2x80x48, 8, 2, 80, 48)
DEFINE_F16_CASE(case_mmad_f16f32_nd_128x128x256, 9, 128, 128, 256)
DEFINE_F16_CASE(case_mmad_f16f32_nd_129x95x33, 10, 129, 95, 33)

// --- DN (transposed) cases ---
DEFINE_F16_CASE(case_mmad_f16f32_dn_31x96x47, 11, 31, 96, 47)
DEFINE_F16_CASE(case_mmad_f16f32_dn_127x33x95, 12, 127, 33, 95)
DEFINE_F16_CASE(case_mmad_f16f32_dn_64x64x64, 13, 64, 64, 64)
DEFINE_F16_CASE(case_mmad_f16f32_dn_1x256x64, 14, 1, 256, 64)
DEFINE_F16_CASE(case_mmad_f16f32_dn_65x90x89, 15, 65, 90, 89)

#undef DEFINE_F16_CASE
