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
#include <cstring>

using namespace std;
using namespace PtoTestCommon;

namespace TmatmulMxExp {
template <int caseId>
void Launch(uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
} // namespace TmatmulMxExp

class TMATMUL_MX_EXP_TEST : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

// All cases are 128x128x128. Byte sizes depend on A/B dtype (fp8=1B/elem,
// fp4=0.5B/elem packed). Scale bytes: M*K/32 (A) + K*N/32 (B). Output: f32.
struct CaseGeometry
{
    size_t aDataBytes;
    size_t bDataBytes;
    size_t aScaleBytes;
    size_t bScaleBytes;
};

// fp8 = 1 byte/elem; fp4_e2m1x2 = 2 nibbles/byte = 0.5 byte/elem.
static constexpr size_t fp8Bytes(int elems) { return static_cast<size_t>(elems); }
static constexpr size_t fp4Bytes(int elems) { return static_cast<size_t>(elems) / 2; }

template <int caseId>
constexpr CaseGeometry GetGeometry()
{
    constexpr int M = 128, K = 128, N = 128;
    constexpr int totalA = M * K, totalB = K * N;
    // E1-E3: fp8_e4m3 (1B) A x fp4_e2m1 (0.5B) B.
    // E4:    fp4_e1m2x2 (0.5B) A x fp4_e1m2x2 (0.5B) B.
    if constexpr (caseId == 1 || caseId == 2 || caseId == 3) {
        return {fp8Bytes(totalA), fp4Bytes(totalB), totalA / 32, totalB / 32};
    } else { // caseId == 4
        return {fp4Bytes(totalA), fp4Bytes(totalB), totalA / 32, totalB / 32};
    }
}

template <int caseId>
void RunCase(const std::string& goldenDir)
{
    constexpr auto geo = GetGeometry<caseId>();
    constexpr int totalOut = 128 * 128;
    constexpr size_t outBytes = totalOut * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *aDataHost, *aScaleHost, *bDataHost, *bScaleHost, *outHost;
    uint8_t *aDataDev, *aScaleDev, *bDataDev, *bScaleDev, *outDev;

    aclrtMallocHost((void**)(&aDataHost), geo.aDataBytes);
    aclrtMallocHost((void**)(&aScaleHost), geo.aScaleBytes);
    aclrtMallocHost((void**)(&bDataHost), geo.bDataBytes);
    aclrtMallocHost((void**)(&bScaleHost), geo.bScaleBytes);
    aclrtMallocHost((void**)(&outHost), outBytes);

    aclrtMalloc((void**)&aDataDev, geo.aDataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&aScaleDev, geo.aScaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&bDataDev, geo.bDataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&bScaleDev, geo.bScaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t rd = geo.aDataBytes;
    ReadFile(goldenDir + "/a_data.bin", rd, aDataHost, geo.aDataBytes);
    rd = geo.aScaleBytes;
    ReadFile(goldenDir + "/a_scale.bin", rd, aScaleHost, geo.aScaleBytes);
    rd = geo.bDataBytes;
    ReadFile(goldenDir + "/b_data.bin", rd, bDataHost, geo.bDataBytes);
    rd = geo.bScaleBytes;
    ReadFile(goldenDir + "/b_scale.bin", rd, bScaleHost, geo.bScaleBytes);

    aclrtMemcpy(aDataDev, geo.aDataBytes, aDataHost, geo.aDataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(aScaleDev, geo.aScaleBytes, aScaleHost, geo.aScaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(bDataDev, geo.bDataBytes, bDataHost, geo.bDataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(bScaleDev, geo.bScaleBytes, bScaleHost, geo.bScaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    TmatmulMxExp::Launch<caseId>(outDev, aDataDev, aScaleDev, bDataDev, bScaleDev, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();

    aclrtMemcpy(outHost, outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(goldenDir + "/output.bin", outHost, outBytes);

    std::vector<float> golden(totalOut);
    std::vector<float> devFinal(totalOut);
    rd = outBytes;
    ReadFile(goldenDir + "/golden.bin", rd, golden.data(), outBytes);
    memcpy(devFinal.data(), outHost, outBytes);

    bool ret = ResultCmp(golden, devFinal, 0.2f);
    EXPECT_TRUE(ret) << "MMAD_MX experiment case mismatch";

    aclrtFree(outDev);
    aclrtFree(bScaleDev);
    aclrtFree(bDataDev);
    aclrtFree(aScaleDev);
    aclrtFree(aDataDev);
    aclrtFreeHost(outHost);
    aclrtFreeHost(bScaleHost);
    aclrtFreeHost(bDataHost);
    aclrtFreeHost(aScaleHost);
    aclrtFreeHost(aDataHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TMATMUL_MX_EXP_TEST, case_e1_fp8xfp4_varied) { RunCase<1>(GetGoldenDir()); }
TEST_F(TMATMUL_MX_EXP_TEST, case_e2_fp8xfp4_neutral_a) { RunCase<2>(GetGoldenDir()); }
TEST_F(TMATMUL_MX_EXP_TEST, case_e3_fp8xfp4_neutral_b) { RunCase<3>(GetGoldenDir()); }
TEST_F(TMATMUL_MX_EXP_TEST, case_e4_fp4e1m2xfp4e1m2_baseline) { RunCase<4>(GetGoldenDir()); }
