/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

/*
 * format 0: NCHW -> NC1HWC0,   p = (N, C, H, W, -),     v = (VN, VC, VH, VW, -)
 * format 1: NC1HWC0 -> NCHW,   p = (N, C1, H, W, C0),   v = (VN, VC1, VH, VW, VC0)
 * format 2: GNCHW -> GNC1HWC0, p = (G, N, C, H, W),     v = (VG, VN, VC, VH, VW)
 */
template <
    typename T, int format, int p0, int p1, int p2, int p3, int p4, int v0, int v1, int v2, int v3, int v4>
void LaunchTTRANSConvValidShape(T* out, T* src, void* stream);

void LaunchTTRANSConvValidShapeApi(int32_t* out, void* stream);

class TTRANSConvValidShapeTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

// NPU reads the src channel dim as the padded C1 * C0, so src and dst hold the same element count.
template <typename T, int format, int p0, int p1, int p2, int p3, int p4>
constexpr size_t ConvElemNum()
{
    constexpr size_t C0 = 32 / sizeof(T);
    if constexpr (format == 0) {
        constexpr size_t C1 = (p1 + C0 - 1) / C0;
        return static_cast<size_t>(p0) * C1 * p2 * p3 * C0;
    } else if constexpr (format == 1) {
        static_assert(static_cast<size_t>(p4) == C0, "C0 must match the dtype block size");
        return static_cast<size_t>(p0) * p1 * p2 * p3 * p4;
    } else {
        static_assert(format == 2, "unsupported TTRANS validshape format");
        constexpr size_t C1 = (p2 + C0 - 1) / C0;
        return static_cast<size_t>(p0) * p1 * C1 * p3 * p4 * C0;
    }
}

template <
    typename T, int format, int p0, int p1, int p2, int p3, int p4, int v0, int v1, int v2, int v3, int v4>
void test_ttrans_axis_valid()
{
    constexpr size_t elemNum = ConvElemNum<T, format, p0, p1, p2, p3, p4>();
    size_t bufSize = elemNum * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), bufSize);
    aclrtMallocHost((void**)(&srcHost), bufSize);
    aclrtMalloc((void**)&dstDevice, bufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, bufSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t inputSize = bufSize;
    ReadFile(GetGoldenDir() + "/input.bin", inputSize, srcHost, bufSize);
    EXPECT_EQ(inputSize, bufSize);

    aclrtMemcpy(srcDevice, bufSize, srcHost, bufSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTTRANSConvValidShape<T, format, p0, p1, p2, p3, p4, v0, v1, v2, v3, v4>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, bufSize, dstDevice, bufSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, bufSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(elemNum);
    std::vector<T> result(elemNum);
    size_t goldenSize = bufSize;
    size_t outputSize = bufSize;
    ReadFile(GetGoldenDir() + "/golden.bin", goldenSize, golden.data(), bufSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputSize, result.data(), bufSize);
    EXPECT_EQ(goldenSize, bufSize);
    EXPECT_EQ(outputSize, bufSize);

    bool ret = ResultCmp(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

// =============================================================================
// ConvTile ValidShape API probe, executed on device and dumped back to host:
// static valid per layout, default valid == physical, and the dynamic setters.
// =============================================================================
TEST_F(TTRANSConvValidShapeTest, case_api_validshape_probe)
{
    constexpr size_t kDumpNum = 69;
    const std::vector<int32_t> expected = {
        // A: NCHW phys(8, 20, 7, 9) valid(3, 7, 5, 4) -> (shape, valid) per axis
        8, 3, 20, 7, 7, 5, 9, 4,
        // B: NC1HWC0 phys(8, 2, 7, 9, 16) valid(3, 1, 5, 4, 7)
        8, 3, 2, 1, 7, 5, 9, 4, 16, 7,
        // C: FRACTAL_Z phys(126, 4, 2, 16) valid(63, 3, 2, 16)
        126, 63, 4, 3, 2, 2, 16, 16,
        // D: GNCHW phys(2, 4, 20, 7, 9) valid(1, 3, 7, 5, 4)
        2, 1, 4, 3, 20, 7, 7, 5, 9, 4,
        // E: GNC1HWC0 phys(2, 4, 2, 7, 9, 16) valid(1, 3, 1, 5, 4, 7)
        2, 1, 4, 3, 2, 1, 7, 5, 9, 4, 16, 7,
        // F: NCHW without explicit valid -> valid defaults to physical
        8, 8, 20, 20, 7, 7, 9, 9,
        // G: dynamic axes, ctor SetDynamicShape(16, 8) resets valid to physical
        16, 8, 16, 8,
        //    SetAllValidShape(10, 5), static axes keep their valid value
        10, 4, 5, 8,
        //    SetValidShape(0, 8)
        8,
        //    SetDynamicShape(12, 6) resets valid to physical again
        12, 6, 12, 6,
    };
    ASSERT_EQ(expected.size(), kDumpNum);

    size_t bufSize = kDumpNum * sizeof(int32_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int32_t* host = nullptr;
    int32_t* device = nullptr;
    aclrtMallocHost((void**)(&host), bufSize);
    aclrtMalloc((void**)&device, bufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    std::fill(host, host + kDumpNum, -1);
    aclrtMemcpy(device, bufSize, host, bufSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTTRANSConvValidShapeApi(device, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(host, bufSize, device, bufSize, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<int32_t> result(host, host + kDumpNum);

    aclrtFree(device);
    aclrtFreeHost(host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    for (size_t i = 0; i < kDumpNum; i++) {
        EXPECT_EQ(expected[i], result[i]) << "dump index " << i;
    }
}

/*-------------------- NCHW -> NC1HWC0 --------------------*/

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_n_half)
{
    test_ttrans_axis_valid<aclFloat16, 0, 4, 20, 4, 16, 1, 2, 20, 4, 16, 1>();
}

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_c_half)
{
    test_ttrans_axis_valid<aclFloat16, 0, 4, 20, 4, 16, 1, 4, 7, 4, 16, 1>();
}

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_h_half)
{
    test_ttrans_axis_valid<aclFloat16, 0, 4, 16, 8, 16, 1, 4, 16, 5, 16, 1>();
}

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_w_half)
{
    test_ttrans_axis_valid<aclFloat16, 0, 4, 16, 4, 16, 1, 4, 16, 4, 4, 1>();
}

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_all_half)
{
    test_ttrans_axis_valid<aclFloat16, 0, 4, 20, 8, 16, 1, 2, 7, 5, 4, 1>();
}

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_all_float)
{
    test_ttrans_axis_valid<float, 0, 4, 20, 4, 16, 1, 2, 7, 3, 4, 1>();
}

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_eq_phys_half)
{
    test_ttrans_axis_valid<aclFloat16, 0, 4, 20, 4, 16, 1, 4, 20, 4, 16, 1>();
}

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_min_half)
{
    test_ttrans_axis_valid<aclFloat16, 0, 4, 20, 4, 16, 1, 1, 1, 1, 1, 1>();
}

/*-------------------- NC1HWC0 -> NCHW --------------------*/

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_n_half)
{
    test_ttrans_axis_valid<aclFloat16, 1, 4, 2, 4, 16, 16, 3, 2, 4, 16, 16>();
}

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_c1_half)
{
    test_ttrans_axis_valid<aclFloat16, 1, 4, 2, 4, 16, 16, 4, 1, 4, 16, 16>();
}

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_c0_half)
{
    test_ttrans_axis_valid<aclFloat16, 1, 4, 2, 4, 16, 16, 4, 2, 4, 16, 7>();
}

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_h_half)
{
    test_ttrans_axis_valid<aclFloat16, 1, 4, 2, 8, 16, 16, 4, 2, 5, 16, 16>();
}

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_w_half)
{
    test_ttrans_axis_valid<aclFloat16, 1, 4, 2, 4, 16, 16, 4, 2, 4, 4, 16>();
}

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_all_half)
{
    test_ttrans_axis_valid<aclFloat16, 1, 4, 2, 8, 16, 16, 2, 1, 5, 4, 7>();
}

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_all_int32)
{
    test_ttrans_axis_valid<int32_t, 1, 4, 2, 4, 8, 8, 2, 1, 3, 4, 5>();
}

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_eq_phys_half)
{
    test_ttrans_axis_valid<aclFloat16, 1, 4, 2, 4, 16, 16, 4, 2, 4, 16, 16>();
}

/*-------------------- GNCHW -> GNC1HWC0 --------------------*/

TEST_F(TTRANSConvValidShapeTest, GNCHW2GNC1HWC0_valid_g_half)
{
    test_ttrans_axis_valid<aclFloat16, 2, 2, 2, 20, 4, 16, 1, 2, 20, 4, 16>();
}

TEST_F(TTRANSConvValidShapeTest, GNCHW2GNC1HWC0_valid_all_half)
{
    test_ttrans_axis_valid<aclFloat16, 2, 2, 2, 20, 4, 16, 1, 1, 7, 3, 4>();
}
