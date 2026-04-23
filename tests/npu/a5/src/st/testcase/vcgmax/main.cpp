/**
 * Test harness for vcgmax testcase
 */
#include "test_common.h"
#include <acl/acl.h>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchVCGMAXTestCase(void *out, void *src, aclrtStream stream);

namespace {

float Fp16BitsToFloat(uint16_t bits)
{
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exp = (bits >> 10) & 0x1Fu;
    const uint32_t mant = bits & 0x03FFu;

    uint32_t raw = 0;
    if (exp == 0) {
        if (mant == 0) {
            raw = sign;
        } else {
            uint32_t mantNorm = mant;
            int shift = -1;
            do {
                ++shift;
                mantNorm <<= 1;
            } while ((mantNorm & 0x0400u) == 0);
            mantNorm &= 0x03FFu;
            raw = sign | static_cast<uint32_t>((127 - 15 - shift) << 23) | (mantNorm << 13);
        }
    } else if (exp == 0x1Fu) {
        raw = sign | 0x7F800000u | (mant << 13);
    } else {
        raw = sign | static_cast<uint32_t>((exp + 127 - 15) << 23) | (mant << 13);
    }

    float value;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

} // namespace

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

class VCGMAXTest : public testing::Test {
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
};

TEST_F(VCGMAXTest, case1_nan_fp16)
{
    size_t srcSize = 16 * sizeof(uint16_t);
    size_t dstAllocSize = 8 * sizeof(uint16_t);
    size_t resultSize = sizeof(uint16_t);

    aclrtMallocHost(&dstHost, dstAllocSize);
    aclrtMallocHost(&srcHost, srcSize);
    aclrtMalloc(&dstDevice, dstAllocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&srcDevice, srcSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcSize, srcHost, srcSize);
    aclrtMemset(dstHost, dstAllocSize, 0, dstAllocSize);

    aclrtMemcpy(dstDevice, dstAllocSize, dstHost, dstAllocSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice, srcSize, srcHost, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchVCGMAXTestCase<1>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(dstHost, dstAllocSize, dstDevice, dstAllocSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, resultSize);

    uint16_t goldenBits = 0;
    uint16_t resultBits = 0;
    ReadFile(GetGoldenDir() + "/golden.bin", resultSize, &goldenBits, resultSize);
    ReadFile(GetGoldenDir() + "/output.bin", resultSize, &resultBits, resultSize);

    ASSERT_TRUE(std::isnan(Fp16BitsToFloat(goldenBits)));
    EXPECT_TRUE(std::isnan(Fp16BitsToFloat(resultBits)));

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
}
