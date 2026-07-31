/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
e1m2 MX oracle test harness — mirrors tmatmul_mx_hif4/main.cpp.
*/

#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <pto/common/type.hpp>
#include "acl/acl.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

namespace TmatmulMxE1m2 {
template <typename OutT, int validM, int validK, int validN>
void Launch(uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
} // namespace TmatmulMxE1m2

class TMATMUL_MX_E1M2_TEST : public testing::Test {
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
void RunE1m2MxMatmulCase(const std::string& goldenDir)
{
    constexpr int totalA = validM * validK;
    constexpr int totalB = validK * validN;
    constexpr int totalOut = validM * validN;

    constexpr size_t aDataBytes = totalA / 2;
    constexpr size_t bDataBytes = totalB / 2;
    // MX: per-32-element scale. For A[M,K]: M * (K/32) bytes total (e8m0 per group).
    // For B[K,N]: (K/32) * N bytes total.
    constexpr size_t aScaleBytes = (totalA / 32);
    constexpr size_t bScaleBytes = (totalB / 32);
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

    TmatmulMxE1m2::Launch<uint16_t, validM, validK, validN>(outDev, aDataDev, aScaleDev, bDataDev, bScaleDev, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();

    std::vector<uint8_t> outHost(outBytes);
    std::vector<uint8_t> goldenHost(outBytes);
    aclrtMemcpy(outHost.data(), outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(goldenDir + "/output.bin", outHost.data(), outBytes);
    size_t goldenRead = outBytes;
    ReadFile(goldenDir + "/golden_out.bin", goldenRead, goldenHost.data(), outBytes);

    // BF16 -> FP32 bitcast for relative-error comparison.
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
    // e1m2 MX is bit-exact (no quantization hierarchy noise). Use tight tolerance.
    EXPECT_TRUE(ResultCmp<float>(goldenVals, outVals, 0.01f)) << "e1m2 MX matmul output mismatch";

    aclrtFree(outDev);
    aclrtFree(bScaleDev);
    aclrtFree(bDataDev);
    aclrtFree(aScaleDev);
    aclrtFree(aDataDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TMATMUL_MX_E1M2_TEST, case_e1m2_128x128x128_nd) { RunE1m2MxMatmulCase<128, 128, 128>(GetGoldenDir()); }
