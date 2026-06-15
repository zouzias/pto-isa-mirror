/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include "acl/acl.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

namespace TQuantDNTest {

template <int M, int N, int N_pad>
void LaunchTQuantDN(uint16_t *src, int8_t *fp8_nz, uint8_t *e8_zz, void *stream);

} // namespace TQuantDNTest

class TQUANTDNTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

template <int M, int N, int N_pad>
void test_tquant_dn_bf16()
{
    constexpr int paddedCols = N_pad;
    constexpr int paddedRows16 = ((M + 15) / 16) * 16;
    constexpr int virtualRow = paddedRows16 + 1;
    constexpr int groupedCols = paddedCols / 32;
    constexpr int numGroupsFlat = paddedRows16 * groupedCols;

    size_t srcFileSize = M * paddedCols * sizeof(uint16_t);
    size_t fp8NZFileSize = virtualRow * paddedCols * sizeof(int8_t);
    size_t e8ZZFileSize = numGroupsFlat * sizeof(uint8_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *srcHost, *fp8NZHost, *e8ZZHost;
    uint16_t *srcDevice;
    int8_t *fp8NZDevice;
    uint8_t *e8ZZDevice;

    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&fp8NZHost), fp8NZFileSize);
    aclrtMallocHost((void **)(&e8ZZHost), e8ZZFileSize);

    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fp8NZDevice, fp8NZFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&e8ZZDevice, e8ZZFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    TQuantDNTest::LaunchTQuantDN<M, N, N_pad>(srcDevice, fp8NZDevice, e8ZZDevice, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();

    aclrtMemcpy(fp8NZHost, fp8NZFileSize, fp8NZDevice, fp8NZFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(e8ZZHost, e8ZZFileSize, e8ZZDevice, e8ZZFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_fp8_nz.bin", fp8NZHost, fp8NZFileSize);
    WriteFile(GetGoldenDir() + "/output_e8_zz.bin", e8ZZHost, e8ZZFileSize);

    aclrtFree(srcDevice);
    aclrtFree(fp8NZDevice);
    aclrtFree(e8ZZDevice);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(fp8NZHost);
    aclrtFreeHost(e8ZZHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint8_t> goldenFp8(fp8NZFileSize);
    std::vector<uint8_t> goldenE8(e8ZZFileSize);
    std::vector<uint8_t> outFp8(fp8NZFileSize);
    std::vector<uint8_t> outE8(e8ZZFileSize);

    ReadFile(GetGoldenDir() + "/golden_fp8_nz.bin", fp8NZFileSize, goldenFp8.data(), fp8NZFileSize);
    ReadFile(GetGoldenDir() + "/golden_e8_zz.bin", e8ZZFileSize, goldenE8.data(), e8ZZFileSize);
    ReadFile(GetGoldenDir() + "/output_fp8_nz.bin", fp8NZFileSize, outFp8.data(), fp8NZFileSize);
    ReadFile(GetGoldenDir() + "/output_e8_zz.bin", e8ZZFileSize, outE8.data(), e8ZZFileSize);

    EXPECT_TRUE(ResultCmp<uint8_t>(goldenFp8, outFp8, 0.0f));
    EXPECT_TRUE(ResultCmp<uint8_t>(goldenE8, outE8, 0.0f));
}

TEST_F(TQUANTDNTest, case_bf16_64x64) { test_tquant_dn_bf16<64, 64, 64>(); }
TEST_F(TQUANTDNTest, case_bf16_128x64) { test_tquant_dn_bf16<128, 64, 64>(); }
TEST_F(TQUANTDNTest, case_bf16_64x128) { test_tquant_dn_bf16<64, 128, 128>(); }
TEST_F(TQUANTDNTest, case_bf16_128x128) { test_tquant_dn_bf16<128, 128, 128>(); }
TEST_F(TQUANTDNTest, case_bf16_64x256) { test_tquant_dn_bf16<64, 256, 256>(); }
TEST_F(TQUANTDNTest, case_bf16_128x256) { test_tquant_dn_bf16<128, 256, 256>(); }
