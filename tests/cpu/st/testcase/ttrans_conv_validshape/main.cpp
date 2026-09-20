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
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <gtest/gtest.h>
#include <fstream>

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

template <typename T, int N, int C, int H, int W, int VN>
void LaunchTTRANSConvValid(T* out, T* src, void* stream);

template <typename T, int N, int C, int H, int W>
void LaunchTTRANSConvDefault(T* out, T* src, void* stream);

class TTRANSConvValidShapeTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TTRANSConvValidShapeTest, case_api_validshape_nchw)
{
    constexpr int N = 4, C = 4, H = 4, W = 4;
    constexpr int VN = 2;

    using SrcPhysShape = ConvTileShape<N, C, H, W>;
    using SrcValidShape = ConvTileValidShape<VN, C, H, W>;
    using SrcCT = ConvTile<TileType::Vec, half, 256 * sizeof(half), Layout::NCHW, SrcPhysShape, SrcValidShape>;
    SrcCT srcTile;

    EXPECT_EQ(srcTile.GetShape(0), N);
    EXPECT_EQ(srcTile.GetShape(1), C);
    EXPECT_EQ(srcTile.GetShape(2), H);
    EXPECT_EQ(srcTile.GetShape(3), W);

    EXPECT_EQ(srcTile.GetValidShape(0), VN);
    EXPECT_EQ(srcTile.GetValidShape(1), C);
    EXPECT_EQ(srcTile.GetValidShape(2), H);
    EXPECT_EQ(srcTile.GetValidShape(3), W);

    constexpr int C0 = 16;
    constexpr int C1 = 1;
    using DstPhysShape = ConvTileShape<N, C1, H, W, C0>;
    using DstCT = ConvTile<TileType::Vec, half, 1024 * sizeof(half), Layout::NC1HWC0, DstPhysShape>;
    DstCT dstTile;

    EXPECT_EQ(dstTile.GetShape(0), N);
    EXPECT_EQ(dstTile.GetShape(1), C1);
    EXPECT_EQ(dstTile.GetShape(2), H);
    EXPECT_EQ(dstTile.GetShape(3), W);
    EXPECT_EQ(dstTile.GetShape(4), C0);

    EXPECT_EQ(dstTile.GetValidShape(0), N);
    EXPECT_EQ(dstTile.GetValidShape(1), C1);
    EXPECT_EQ(dstTile.GetValidShape(2), H);
    EXPECT_EQ(dstTile.GetValidShape(3), W);
    EXPECT_EQ(dstTile.GetValidShape(4), C0);
}

TEST_F(TTRANSConvValidShapeTest, case_api_default_valid_eq_physical)
{
    constexpr int N = 4, C = 4, H = 4, W = 4;

    using SrcCT = ConvTile<TileType::Vec, half, 256 * sizeof(half), Layout::NCHW, ConvTileShape<N, C, H, W>>;
    SrcCT srcTile;

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(srcTile.GetValidShape(i), srcTile.GetShape(i));
    }

    constexpr int C0 = 16;
    constexpr int C1 = 1;
    using DstCT = ConvTile<TileType::Vec, half, 1024 * sizeof(half), Layout::NC1HWC0, ConvTileShape<N, C1, H, W, C0>>;
    DstCT dstTile;

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(dstTile.GetValidShape(i), dstTile.GetShape(i));
    }
}

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

inline size_t GetFileSize(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    return in.is_open() ? static_cast<size_t>(in.tellg()) : 0;
}

template <typename T, int N, int C, int H, int W, int VN>
void test_ttrans_valid()
{
    constexpr size_t DTypeSize = sizeof(T);
    constexpr size_t C0 = 32 / DTypeSize;
    constexpr size_t C1 = (C + C0 - 1) / C0;
    size_t srcFileSize = N * C * H * W * sizeof(T);
    size_t dstFileSize = N * C1 * H * W * C0 * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTTRANSConvValid<T, N, C, H, W, VN>(dstDevice, srcDevice, stream);

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

    std::vector<T> golden(dstFileSize / sizeof(T), 0);
    std::vector<T> result(dstFileSize / sizeof(T), 0);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, result, 0.001f);
    EXPECT_TRUE(ret);
}

template <typename T, int N, int C, int H, int W>
void test_ttrans_default()
{
    constexpr size_t DTypeSize = sizeof(T);
    constexpr size_t C0 = 32 / DTypeSize;
    constexpr size_t C1 = (C + C0 - 1) / C0;
    size_t srcFileSize = N * C * H * W * sizeof(T);
    size_t dstFileSize = N * C1 * H * W * C0 * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTTRANSConvDefault<T, N, C, H, W>(dstDevice, srcDevice, stream);

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

    std::vector<T> golden(dstFileSize / sizeof(T), 0);
    std::vector<T> result(dstFileSize / sizeof(T), 0);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, result, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TTRANSConvValidShapeTest, case_default_half) { test_ttrans_default<half, 4, 4, 4, 4>(); }

TEST_F(TTRANSConvValidShapeTest, case_valid_half) { test_ttrans_valid<half, 4, 4, 4, 4, 2>(); }

TEST_F(TTRANSConvValidShapeTest, case_default_float) { test_ttrans_default<float, 4, 4, 4, 4>(); }

TEST_F(TTRANSConvValidShapeTest, case_valid_float) { test_ttrans_valid<float, 4, 4, 4, 4, 2>(); }
