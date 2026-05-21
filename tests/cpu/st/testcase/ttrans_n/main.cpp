#include "test_common.h"
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5,
          int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTTRANSConv(T *out, T *src, void *stream);

template <typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5,
          int gShape6, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4,
          int gWholeShape5>
void LaunchTTRANSGroupConv(T *out, T *src, void *stream);

class TTRANSConvTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5,
          int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void test_ttrans()
{
    size_t srcFileSize = gShape0 * gShape1 * gShape2 * gShape3 * gShape4 * gShape5 * sizeof(T);
    size_t dstFileSize = srcFileSize;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTTRANSConv<T, format, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gWholeShape0, gWholeShape1,
                     gWholeShape2, gWholeShape3, gWholeShape4>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> result(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5,
          int gShape6, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4,
          int gWholeShape5>
void test_ttrans_group()
{
    size_t srcFileSize = gShape0 * gShape1 * gShape2 * gShape3 * gShape4 * gShape5 * gShape6 * sizeof(T);
    size_t dstFileSize = srcFileSize;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTTRANSGroupConv<T, format, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gShape6, gWholeShape0,
                          gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4, gWholeShape5>(dstDevice, srcDevice,
                                                                                               stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> result(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

// =========================================================================
// MODE 1: NCHW2NC1HWC0 (Format ID: 1)
// =========================================================================

// Case 1.1: Mode1_FP16_Aligned
TEST_F(TTRANSConvTest, Mode1_FP16_Aligned)
{
    // T=uint16_t (for FP16 raw storage matching Python), format=1
    test_ttrans<uint16_t, 1, 1, 2, 14, 14, 16, 1, 1, 1, 32, 14, 14>();
}

// Case 1.2: Mode1_UInt16_Unaligned_Fixed
TEST_F(TTRANSConvTest, Mode1_UInt16_Unaligned_Fixed)
{
    test_ttrans<uint16_t, 1, 1, 2, 2, 16, 16, 1, 1, 1, 26, 2, 16>();
}

// Case 1.3: Mode1_Int8_RGB_Input
TEST_F(TTRANSConvTest, Mode1_Int8_RGB_Input)
{
    test_ttrans<int8_t, 1, 4, 1, 224, 224, 32, 1, 4, 1, 3, 224, 224>();
}

// Case 1.4: Mode1_FP32_MultiBatch
TEST_F(TTRANSConvTest, Mode1_FP32_MultiBatch)
{
    test_ttrans<float, 1, 16, 4, 7, 7, 8, 1, 16, 1, 32, 7, 7>();
}

// Case 1.5: Mode1_Micro_Boundary
TEST_F(TTRANSConvTest, Mode1_Micro_Boundary)
{
    test_ttrans<uint16_t, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1>();
}

// Case 1.6: Mode1_Large_Enterprise_Layer
TEST_F(TTRANSConvTest, Mode1_Large_Enterprise_Layer)
{
    test_ttrans<uint16_t, 1, 2, 64, 56, 56, 16, 1, 2, 1, 1000, 56, 56>();
}

// =========================================================================
// MODE 2: NC1HWC02C1HWN1N0C0 (Format ID: 2)
// =========================================================================

// Case 2.1: Mode2_Aligned_Split
TEST_F(TTRANSConvTest, Mode2_Aligned_Split)
{
    test_ttrans<uint16_t, 2, 16, 4, 14, 14, 16, 1, 16, 4, 14, 14, 16>();
}

// Case 2.2: Mode2_Unaligned_Batch_Padding
TEST_F(TTRANSConvTest, Mode2_Unaligned_Batch_Padding)
{
    // Note: Python memory buffer is allocated to target_n (16) * C1 * H * W * C0
    // Therefore structural loop dimension gShape0 passes 16 down to the host/device allocations
    test_ttrans<uint16_t, 2, 16, 2, 7, 7, 16, 1, 13, 2, 7, 7, 16>();
}

// Case 2.3: Mode2_Single_Batch_Static
TEST_F(TTRANSConvTest, Mode2_Single_Batch_Static)
{
    test_ttrans<float, 2, 1, 8, 28, 28, 8, 1, 1, 8, 28, 28, 8>();
}

// Case 2.4: Mode2_Quantized_Int8_Blocks
TEST_F(TTRANSConvTest, Mode2_Quantized_Int8_Blocks)
{
    test_ttrans<int8_t, 2, 64, 16, 10, 10, 32, 1, 64, 16, 10, 10, 32>();
}

// Case 2.5: Mode2_Micro_Boundary
TEST_F(TTRANSConvTest, Mode2_Micro_Boundary)
{
    // target_n = 1 * 2 = 2
    test_ttrans<uint16_t, 2, 2, 1, 1, 1, 16, 1, 1, 1, 1, 1, 16>();
}

// Case 2.6: Mode2_Heavy_Industrial_Scale
TEST_F(TTRANSConvTest, Mode2_Heavy_Industrial_Scale)
{
    test_ttrans<uint16_t, 2, 32, 32, 32, 32, 16, 1, 32, 32, 32, 32, 16>();
}

// =========================================================================
// MODE 3: GNCHW2GNC1HWC0 (Format ID: 3) -> Uses test_ttrans_group
// =========================================================================

// Case 3.1: Mode3_Grouped_Standard
TEST_F(TTRANSConvTest, Mode3_Grouped_Standard)
{
    // gShape6 = Group Count (2)
    test_ttrans_group<uint16_t, 3, 1, 2, 14, 14, 16, 1, 2, 1, 1, 32, 14, 14, 1>();
}

// Case 3.2: Mode3_HighGroup_Unaligned_Channels
TEST_F(TTRANSConvTest, Mode3_HighGroup_Unaligned_Channels)
{
    // gShape6 = Group Count (8)
    test_ttrans_group<uint16_t, 3, 2, 1, 28, 28, 16, 1, 8, 2, 2, 11, 28, 28, 1>();
}

// Case 3.3: Mode3_Quantized_Int8_Groups
TEST_F(TTRANSConvTest, Mode3_Quantized_Int8_Groups)
{
    // gShape6 = Group Count (4)
    test_ttrans_group<int8_t, 3, 1, 2, 40, 40, 32, 1, 4, 1, 1, 45, 40, 40, 1>();
}

// Case 3.4: Mode3_Depthwise_Simulation
TEST_F(TTRANSConvTest, Mode3_Depthwise_Simulation)
{
    // gShape6 = Group Count (32)
    test_ttrans_group<uint16_t, 3, 1, 1, 112, 112, 16, 1, 32, 1, 1, 1, 112, 112, 1>();
}

// Case 3.5: Mode3_Micro_Boundary
TEST_F(TTRANSConvTest, Mode3_Micro_Boundary)
{
    // gShape6 = Group Count (1)
    test_ttrans_group<uint16_t, 3, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 1>();
}

// Case 3.6: Mode3_Massive_Spatial_Footprint
TEST_F(TTRANSConvTest, Mode3_Massive_Spatial_Footprint)
{
    // gShape6 = Group Count (4)
    test_ttrans_group<float, 3, 2, 4, 128, 128, 8, 1, 4, 2, 2, 24, 128, 128, 1>();
}

// =========================================================================
// MODE 4: GNC1HWC02C1HWN1N0C0 (Format ID: 4) -> Uses test_ttrans_group
// =========================================================================

// Case 4.1: Mode4_Grouped_Blocked_Aligned
TEST_F(TTRANSConvTest, Mode4_Grouped_Blocked_Aligned)
{
    // gShape6 = Group Count (2), gWholeShape5 = Group Count (2)
    test_ttrans_group<uint16_t, 4, 16, 4, 14, 14, 16, 1, 2, 16, 4, 14, 14, 16, 2>();
}

// Case 4.2: Mode4_Grouped_Unaligned_Batch
TEST_F(TTRANSConvTest, Mode4_Grouped_Unaligned_Batch)
{
    // target_n = 2 * 4 = 8. Host/Device allocations scaled up to padded buffer boundaries
    test_ttrans_group<uint16_t, 4, 8, 2, 7, 7, 16, 1, 4, 7, 2, 7, 7, 16, 4>();
}

// Case 4.3: Mode4_HighGroup_Weight_Transform
TEST_F(TTRANSConvTest, Mode4_HighGroup_Weight_Transform)
{
    // target_n = 1 * 4 = 4
    test_ttrans_group<uint16_t, 4, 4, 2, 5, 5, 16, 1, 16, 4, 2, 5, 5, 16, 16>();
}

// Case 4.4: Mode4_Quantized_Int8_GroupBlocks
TEST_F(TTRANSConvTest, Mode4_Quantized_Int8_GroupBlocks)
{
    // target_n = 2 * 16 = 32
    test_ttrans_group<int8_t, 4, 32, 8, 10, 10, 32, 1, 2, 32, 8, 10, 10, 32, 2>();
}

// Case 4.5: Mode4_Micro_Boundary
TEST_F(TTRANSConvTest, Mode4_Micro_Boundary)
{
    // target_n = 1 * 2 = 2
    test_ttrans_group<uint16_t, 4, 2, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 16, 1>();
}

// Case 4.6: Mode4_Max_Scale_Industrial_Stress
TEST_F(TTRANSConvTest, Mode4_Max_Scale_Industrial_Stress)
{
    // target_n = 4 * 4 = 16
    test_ttrans_group<float, 4, 16, 16, 20, 20, 8, 1, 8, 16, 16, 20, 20, 8, 8>();
}