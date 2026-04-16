/**
 * TADD Benchmark Test Suite - Auto-generated from input.csv
 * Run: python3 generate_code.py
 */
#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TADDBenchTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchTAddBench(T *out, T *src0, T *src1, void *stream);

template <typename T, int tileH, int tileW, int vRows, int vCols>
void test_tadd_bench()
{
    size_t fileSize = tileH * tileW * sizeof(T);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;
    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&src0Host), fileSize);
    aclrtMallocHost((void **)(&src1Host), fileSize);
    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, fileSize);
    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTAddBench<T, tileH, tileW, vRows, vCols>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    
    aclrtFree(dstDevice); aclrtFree(src0Device); aclrtFree(src1Device);
    aclrtFreeHost(dstHost); aclrtFreeHost(src0Host); aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

// Test cases (auto-generated from input.csv)
TEST_F(TADDBenchTest, case_float_64x64) { test_tadd_bench<float, 64, 64, 64, 64>(); }
TEST_F(TADDBenchTest, case_float_8x512) { test_tadd_bench<float, 8, 512, 8, 512>(); }
TEST_F(TADDBenchTest, case_float_1x4096) { test_tadd_bench<float, 1, 4096, 1, 4096>(); }
TEST_F(TADDBenchTest, case_float_64x128) { test_tadd_bench<float, 64, 128, 64, 128>(); }
TEST_F(TADDBenchTest, case_float_16x512) { test_tadd_bench<float, 16, 512, 16, 512>(); }
TEST_F(TADDBenchTest, case_float_1x8192) { test_tadd_bench<float, 1, 8192, 1, 8192>(); }
TEST_F(TADDBenchTest, case_float_128x128) { test_tadd_bench<float, 128, 128, 128, 128>(); }
TEST_F(TADDBenchTest, case_float_32x512) { test_tadd_bench<float, 32, 512, 32, 512>(); }
TEST_F(TADDBenchTest, case_float_1x16384) { test_tadd_bench<float, 1, 16384, 1, 16384>(); }
