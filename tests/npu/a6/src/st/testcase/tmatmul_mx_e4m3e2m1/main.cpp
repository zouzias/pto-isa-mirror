/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include <securec.h>
#include <string>
#include <vector>
#include <pto/common/type.hpp>
#include "acl/acl.h"
#include "test_common.h"

using namespace PtoTestCommon;

namespace TmatmulMxE4m3E2m1 {
template <typename OutT, int validM, int validK, int validN>
void Launch(uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
} // namespace TmatmulMxE4m3E2m1

class TMATMUL_MX_E4M3E2M1_TEST : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

namespace {

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

std::vector<float> Bf16BytesToFloat(const uint8_t* raw, int n)
{
    std::vector<float> v(n);
    const auto* u16 = reinterpret_cast<const uint16_t*>(raw);
    for (int i = 0; i < n; i++) {
        uint32_t bits = static_cast<uint32_t>(u16[i]) << 16;
        if (memcpy_s(&v[i], sizeof(float), &bits, sizeof(bits)) != EOK) {
            return {};
        }
    }
    return v;
}

uint8_t* UploadToDevice(const std::vector<uint8_t>& host)
{
    uint8_t* dev = nullptr;
    aclrtMalloc((void**)&dev, host.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dev, host.size(), host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    return dev;
}

std::vector<uint8_t> ReadInput(const std::string& path, size_t bytes)
{
    std::vector<uint8_t> host(bytes);
    size_t read = bytes;
    ReadFile(path, read, host.data(), bytes);
    return host;
}

} // namespace

template <int validM, int validK, int validN>
void RunE4m3E2m1MxMatmulCase(const std::string& goldenDir)
{
    constexpr int totalA = validM * validK;
    constexpr int totalB = validK * validN;
    constexpr int totalOut = validM * validN;
    constexpr size_t aDataBytes = totalA;
    constexpr size_t bDataBytes = totalB / 2;
    constexpr size_t aScaleBytes = totalA / 32;
    constexpr size_t bScaleBytes = totalB / 32;
    constexpr size_t outBytes = totalOut * sizeof(uint16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    auto aDataHost = ReadInput(goldenDir + "/a_data.bin", aDataBytes);
    auto aScaleHost = ReadInput(goldenDir + "/a_scale.bin", aScaleBytes);
    auto bDataHost = ReadInput(goldenDir + "/b_data.bin", bDataBytes);
    auto bScaleHost = ReadInput(goldenDir + "/b_scale.bin", bScaleBytes);

    uint8_t* aDataDev = UploadToDevice(aDataHost);
    uint8_t* aScaleDev = UploadToDevice(aScaleHost);
    uint8_t* bDataDev = UploadToDevice(bDataHost);
    uint8_t* bScaleDev = UploadToDevice(bScaleHost);
    uint8_t* outDev = nullptr;
    aclrtMalloc((void**)&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    TmatmulMxE4m3E2m1::Launch<uint16_t, validM, validK, validN>(
        outDev, aDataDev, aScaleDev, bDataDev, bScaleDev, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();

    std::vector<uint8_t> outHost(outBytes);
    std::vector<uint8_t> goldenHost(outBytes);
    aclrtMemcpy(outHost.data(), outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(goldenDir + "/output.bin", outHost.data(), outBytes);
    size_t goldenRead = outBytes;
    ReadFile(goldenDir + "/golden_out.bin", goldenRead, goldenHost.data(), outBytes);

    auto outVals = Bf16BytesToFloat(outHost.data(), totalOut);
    auto goldenVals = Bf16BytesToFloat(goldenHost.data(), totalOut);
    EXPECT_TRUE(ResultCmp<float>(goldenVals, outVals, 0.03f)) << "e4m3-e2m1 MX matmul output mismatch";

    aclrtFree(outDev);
    aclrtFree(bScaleDev);
    aclrtFree(bDataDev);
    aclrtFree(aScaleDev);
    aclrtFree(aDataDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TMATMUL_MX_E4M3E2M1_TEST, case_e4m3e2m1_128x128x128_nd)
{
    RunE4m3E2m1MxMatmulCase<128, 128, 128>(GetGoldenDir());
}

// Experiment A: neutral A-scale (all e8m0 = 127, scale = 1.0). Same kernel/
// dtype as the baseline case; only the A-scale bytes differ. If this case
// STILL produces NaN at MMAD_MX output, the failure is the MX_A_ZZ A-scale
// tile being incompatible with a non-fp4 A dtype (presence/encoding), NOT
// the specific A-scale values. See gen_data.py --neutral_a_scale.
TEST_F(TMATMUL_MX_E4M3E2M1_TEST, case_e4m3e2m1_128x128x128_neutral_a_scale)
{
    RunE4m3E2m1MxMatmulCase<128, 128, 128>(GetGoldenDir());
}
