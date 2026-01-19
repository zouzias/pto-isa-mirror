#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

#include "acl/acl.h"

using namespace std;
using namespace PtoTestCommon;

struct ShapeInfo{
    uint32_t shape0;
    uint32_t shape1;
    uint32_t shape2;
    uint32_t shape3;
};

class DsHcPostTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir() {
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    std::cout<<"golden:"<<fullPath<<std::endl;
    return fullPath;
}
template <typename T, int S0, int S1, int S2, int S3>
void LaunchDeepseek_hc_post(T *out, T *src0, T *src1, ShapeInfo* param, void *stream);


template <typename T, int S0, int S1, int S2, int S3>
void test_deepseek_hc_post() {

    size_t fileSize = S0 * S1 * S2 * S3 * sizeof(T);
    size_t paraSize = sizeof(ShapeInfo); // TODO

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host, *src2Host;
    T *dstDevice, *src0Device, *src1Device;
    ShapeInfo *src2Device;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&src0Host), fileSize);
    aclrtMallocHost((void **)(&src1Host), fileSize);
    aclrtMallocHost((void **)(&src2Host), paraSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Device, paraSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, fileSize);
    ReadFile(GetGoldenDir() + "/input3.bin", paraSize, src2Host, paraSize);

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, paraSize, src2Host, paraSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchDeepseek_hc_post<T, S0, S1, S2, S3>(dstDevice, src0Device, src1Device, src2Device, stream);
    // LaunchDeepseek_hc_post<T, S0, S1, S2, S3>(dstDevice, src0Device, src1Device, src2Device, stream);
    // LaunchDeepseek_hc_post<T, S0, S1, S2, S3>(dstDevice, src0Device, src1Device, src2Device, stream);
    // LaunchDeepseek_hc_post<T, S0, S1, S2, S3>(dstDevice, src0Device, src1Device, src2Device, stream);
    // LaunchDeepseek_hc_post<T, S0, S1, S2, S3>(dstDevice, src0Device, src1Device, src2Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

// TEST_F(TADDTest, case_float_1_1_107_152) {
//     test_tadd<float, 1, 1, 107, 152>();
// }
// TEST_F(TADDTest, case_float_1_8_1_2048) {
//     test_tadd<float, 1, 8, 1, 2048>();
// }
// TEST_F(TADDTest, case_float_1_8_1_32) {
//     test_tadd<float, 1, 8, 1, 32>();
// }
// TEST_F(TADDTest, case_float_1_1_256_16) {
//     test_tadd<float, 1, 1, 256, 16>();
// }

// TEST_F(TADDTest, case_float_1_8_8_128) {
//     test_tadd<float, 1, 8, 8, 128>();
// }

// TEST_F(TADDTest, case_float_16_16_1_64) {
//     test_tadd<float, 16, 16, 1, 64>();
// }
// TEST_F(TADDTest, case_float_1_1_1_16384) {
//     test_tadd<float, 1, 1, 1, 16384>();
// }
// TEST_F(TADDTest, case_float_16_8_2_64) {
//     test_tadd<float, 16, 8, 2, 64>();
// }

TEST_F(DsHcPostTest, case_float_3_33_1_8) {
    test_deepseek_hc_post<float, 3, 33, 1, 8>();
}
// TEST_F(TADDTest, case_float_2_2_2_1024) {
//     test_tadd<float, 2, 2, 2, 1024>();
// }
