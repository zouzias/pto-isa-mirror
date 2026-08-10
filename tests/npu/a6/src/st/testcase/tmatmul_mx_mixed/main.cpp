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

namespace TmatmulMxMixedA6 {
template <int caseId>
void Launch(uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
} // namespace TmatmulMxMixedA6

class TMATMUL_MX_MIXED_A6_TEST : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

enum class RightKind { E2M1, HIF4 };
enum class LeftKind { FP8E4M3, FP16, BF16 };

template <int caseId>
struct CaseConfig;

#define DEFINE_CASE_CONFIG(ID, M, K, N, LEFT, RIGHT) \
    template <> \
    struct CaseConfig<ID> { \
        static constexpr int kM = M; \
        static constexpr int kK = K; \
        static constexpr int kN = N; \
        static constexpr LeftKind kLeft = LEFT; \
        static constexpr RightKind kRight = RIGHT; \
    }

DEFINE_CASE_CONFIG(1, 128, 128, 128, LeftKind::FP8E4M3, RightKind::HIF4);
DEFINE_CASE_CONFIG(2, 64, 128, 64, LeftKind::FP8E4M3, RightKind::HIF4);
DEFINE_CASE_CONFIG(3, 128, 128, 128, LeftKind::FP16, RightKind::E2M1);
DEFINE_CASE_CONFIG(4, 64, 128, 64, LeftKind::FP16, RightKind::E2M1);
DEFINE_CASE_CONFIG(5, 128, 128, 128, LeftKind::BF16, RightKind::E2M1);
DEFINE_CASE_CONFIG(6, 64, 128, 64, LeftKind::BF16, RightKind::E2M1);
DEFINE_CASE_CONFIG(7, 128, 128, 128, LeftKind::FP16, RightKind::HIF4);
DEFINE_CASE_CONFIG(8, 64, 128, 64, LeftKind::FP16, RightKind::HIF4);
DEFINE_CASE_CONFIG(9, 128, 128, 128, LeftKind::BF16, RightKind::HIF4);
DEFINE_CASE_CONFIG(10, 64, 128, 64, LeftKind::BF16, RightKind::HIF4);

#undef DEFINE_CASE_CONFIG

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

template <int caseId>
void RunCase(const std::string& goldenDir)
{
    using Cfg = CaseConfig<caseId>;
    constexpr int totalA = Cfg::kM * Cfg::kK;
    constexpr int totalB = Cfg::kK * Cfg::kN;
    constexpr int totalOut = Cfg::kM * Cfg::kN;
    constexpr size_t aDataBytes = (Cfg::kLeft == LeftKind::FP8E4M3) ? totalA : totalA * sizeof(uint16_t);
    constexpr size_t bDataBytes = totalB / 2;
    constexpr size_t aScaleBytes = totalA / 32;
    constexpr size_t bScaleBytes = (Cfg::kRight == RightKind::E2M1) ? (totalB / 32) : ((totalB / 64) * 4);
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

    TmatmulMxMixedA6::Launch<caseId>(outDev, aDataDev, aScaleDev, bDataDev, bScaleDev, stream);

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
    EXPECT_TRUE(ResultCmp<float>(goldenVals, outVals, 0.2f)) << "TMATMUL_MX mixed case mismatch";

    aclrtFree(outDev);
    aclrtFree(bScaleDev);
    aclrtFree(bDataDev);
    aclrtFree(aScaleDev);
    aclrtFree(aDataDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

#define DEFINE_MIXED_CASE(TEST_NAME, ID) TEST_F(TMATMUL_MX_MIXED_A6_TEST, TEST_NAME) { RunCase<ID>(GetGoldenDir()); }

DEFINE_MIXED_CASE(case_mmad_mx_e4m3hi4_128x128x128_nd, 1)
DEFINE_MIXED_CASE(case_mmad_mx_e4m3hi4_64x128x64_nd, 2)
DEFINE_MIXED_CASE(case_mmad_mx_fp16e2m1_128x128x128_nd, 3)
DEFINE_MIXED_CASE(case_mmad_mx_fp16e2m1_64x128x64_nd, 4)
DEFINE_MIXED_CASE(case_mmad_mx_bf16e2m1_128x128x128_nd, 5)
DEFINE_MIXED_CASE(case_mmad_mx_bf16e2m1_64x128x64_nd, 6)
DEFINE_MIXED_CASE(case_mmad_mx_fp16hi4_128x128x128_nd, 7)
DEFINE_MIXED_CASE(case_mmad_mx_fp16hi4_64x128x64_nd, 8)
DEFINE_MIXED_CASE(case_mmad_mx_bf16hif4_128x128x128_nd, 9)
DEFINE_MIXED_CASE(case_mmad_mx_bf16hif4_64x128x64_nd, 10)

#undef DEFINE_MIXED_CASE
