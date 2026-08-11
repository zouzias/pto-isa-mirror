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
#include "runtime/rt.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void LaunchMixedTallMatmul(uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);

class TMATMULTallOutputMixTest : public testing::Test {
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

// rings = 1 for a DIR_V2C pipe, 2 for DIR_BOTH (which lays the V2C ring after the C2V one,
// so the GM buffer must cover both even when only one direction is driven).
template <typename T, int32_t key, uint32_t rings = 1>
void MixedTallTest(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(T);
    size_t bFileSize = K * N * sizeof(T);
    size_t cFileSize = M * N * sizeof(T);
    // SLOT_NUM (8) slots, each big enough for the larger operand — must match the kernel.
    size_t fifoSize = rings * 8 * std::max(aFileSize, bFileSize);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *fifoDevice;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&fifoDevice, fifoSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    LaunchMixedTallMatmul<key>((uint8_t*)ffts, dstDevice, src0Device, src1Device, fifoDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(fifoDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cFileSize);
    std::vector<T> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.0001f);

    EXPECT_TRUE(ret) << "M=" << M << " K=" << K << " N=" << N << (N < M ? "  (result tile is taller than wide)" : "");
}

// Operands cross a DIR_V2C pipe, result tile is taller than wide.
TEST_F(TMATMULTallOutputMixTest, case1_tall_64x64x32) { MixedTallTest<float, 1>(64, 64, 32); }

// Same path, square result. Control.
TEST_F(TMATMULTallOutputMixTest, case2_square_64x64x64) { MixedTallTest<float, 2>(64, 64, 64); }

// A second tall shape.
TEST_F(TMATMULTallOutputMixTest, case3_tall_32x32x16) { MixedTallTest<float, 3>(32, 32, 16); }

// The same three cases over a DIR_BOTH pipe instead of DIR_V2C. Everything else — data flow,
// slot geometry, tile types, flag ledger — is identical, so a difference in outcome isolates
// the defect to the V2C branch of TPush.hpp rather than to cross-core transport in general.
TEST_F(TMATMULTallOutputMixTest, case4_tall_64x64x32_dir_both) { MixedTallTest<float, 4, 2>(64, 64, 32); }

TEST_F(TMATMULTallOutputMixTest, case5_square_64x64x64_dir_both) { MixedTallTest<float, 5, 2>(64, 64, 64); }

TEST_F(TMATMULTallOutputMixTest, case6_tall_32x32x16_dir_both) { MixedTallTest<float, 6, 2>(32, 32, 16); }
