/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <pto/common/type.hpp>
#include "acl/acl.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

namespace TmatmulMxHif4A6 {
template <typename OutT, int validM, int validK, int validN>
void Launch(uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
} // namespace TmatmulMxHif4A6

class TMATMUL_MX_HIF4_A6_TEST : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

template <int validM, int validK, int validN>
void RunHif4MatmulCase(const std::string& goldenDir)
{
    constexpr int totalA = validM * validK;
    constexpr int totalB = validK * validN;
    constexpr int totalOut = validM * validN;

    constexpr size_t aDataBytes = totalA / 2;
    constexpr size_t bDataBytes = totalB / 2;
    constexpr size_t aScaleBytes = (totalA / 64) * 4;
    constexpr size_t bScaleBytes = (totalB / 64) * 4;
    constexpr size_t outBytes = totalOut * sizeof(uint16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    std::vector<uint8_t> aDataHost(aDataBytes);
    std::vector<uint8_t> aScaleHost(aScaleBytes);
    std::vector<uint8_t> bDataHost(bDataBytes);
    std::vector<uint8_t> bScaleHost(bScaleBytes);
    size_t aDataRead = aDataBytes;
    size_t aScaleRead = aScaleBytes;
    size_t bDataRead = bDataBytes;
    size_t bScaleRead = bScaleBytes;
    ReadFile(goldenDir + "/a_data.bin", aDataRead, aDataHost.data(), aDataBytes);
    ReadFile(goldenDir + "/a_scale.bin", aScaleRead, aScaleHost.data(), aScaleBytes);
    ReadFile(goldenDir + "/b_data.bin", bDataRead, bDataHost.data(), bDataBytes);
    ReadFile(goldenDir + "/b_scale.bin", bScaleRead, bScaleHost.data(), bScaleBytes);

    uint8_t *aDataDev = nullptr, *aScaleDev = nullptr;
    uint8_t *bDataDev = nullptr, *bScaleDev = nullptr;
    uint8_t* outDev = nullptr;
    aclrtMalloc((void**)&aDataDev, aDataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&aScaleDev, aScaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&bDataDev, bDataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&bScaleDev, bScaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(aDataDev, aDataBytes, aDataHost.data(), aDataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(aScaleDev, aScaleBytes, aScaleHost.data(), aScaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(bDataDev, bDataBytes, bDataHost.data(), bDataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(bScaleDev, bScaleBytes, bScaleHost.data(), bScaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    TmatmulMxHif4A6::Launch<uint16_t, validM, validK, validN>(outDev, aDataDev, aScaleDev, bDataDev, bScaleDev, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();

    std::vector<uint8_t> outHost(outBytes);
    std::vector<uint8_t> goldenHost(outBytes);
    aclrtMemcpy(outHost.data(), outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(goldenDir + "/output.bin", outHost.data(), outBytes);
    size_t goldenRead = outBytes;
    ReadFile(goldenDir + "/golden_out.bin", goldenRead, goldenHost.data(), outBytes);

    // Output is BF16 (fixpipe FP32->BF16 cast). ResultCmp needs float for
    // correct relative-error semantics. BF16 -> FP32 = bitcast uint16 << 16.
    auto bf16Vec = [](const uint8_t* raw, int n) {
        std::vector<float> v(n);
        auto* u16 = reinterpret_cast<const uint16_t*>(raw);
        for (int i = 0; i < n; i++) {
            uint32_t bits = static_cast<uint32_t>(u16[i]) << 16;
            std::memcpy(&v[i], &bits, 4);
        }
        return v;
    };
    auto outVals = bf16Vec(outHost.data(), totalOut);
    auto goldenVals = bf16Vec(goldenHost.data(), totalOut);
    EXPECT_TRUE(ResultCmp<float>(goldenVals, outVals, 0.2f)) << "HiF4 matmul output mismatch";

    aclrtFree(outDev);
    aclrtFree(bScaleDev);
    aclrtFree(bDataDev);
    aclrtFree(aScaleDev);
    aclrtFree(aDataDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TMATMUL_MX_HIF4_A6_TEST, case_hif4_128x128x128_nd) { RunHif4MatmulCase<128, 128, 128>(GetGoldenDir()); }

TEST_F(TMATMUL_MX_HIF4_A6_TEST, case_hif4_128x256x128_nd) { RunHif4MatmulCase<128, 256, 128>(GetGoldenDir()); }

TEST_F(TMATMUL_MX_HIF4_A6_TEST, case_hif4_256x128x128_nd) { RunHif4MatmulCase<256, 128, 128>(GetGoldenDir()); }

TEST_F(TMATMUL_MX_HIF4_A6_TEST, case_hif4_64x64x64_nd) { RunHif4MatmulCase<64, 64, 64>(GetGoldenDir()); }

TEST_F(TMATMUL_MX_HIF4_A6_TEST, case_hif4_256x256x256_nd) { RunHif4MatmulCase<256, 256, 256>(GetGoldenDir()); }

TEST_F(TMATMUL_MX_HIF4_A6_TEST, case_hif4_128x512x128_nd) { RunHif4MatmulCase<128, 512, 128>(GetGoldenDir()); }

TEST_F(TMATMUL_MX_HIF4_A6_TEST, case_hif4_512x128x512_nd) { RunHif4MatmulCase<512, 128, 512>(GetGoldenDir()); }

TEST_F(TMATMUL_MX_HIF4_A6_TEST, case_hif4_128x128x256_nd) { RunHif4MatmulCase<128, 128, 256>(GetGoldenDir()); }

TEST_F(TMATMUL_MX_HIF4_A6_TEST, case_hif4_256x128x512_nd) { RunHif4MatmulCase<256, 128, 512>(GetGoldenDir()); }
