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

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

/*
 * format 0: NCHW -> NC1HWC0,     p = (N, C, H, W, -),  v = (VN, VC, VH, VW, -)
 * format 1: NC1HWC0 -> NCHW,     p = (N, C1, H, W, C0), v = (VN, VC1, VH, VW, VC0)
 * format 2: GNCHW -> GNC1HWC0,   p = (G, N, C, H, W),  v = (VG, VN, VC, VH, VW)
 */
template <
    typename T, int format, int p0, int p1, int p2, int p3, int p4, int v0, int v1, int v2, int v3, int v4>
void LaunchTTRANSConvAxisValid(T* out, T* src, void* stream);

class TTRANSConvValidShapeTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// Part 1. static ValidShape API: every axis unequal / unaligned
// =============================================================================

// NCHW 4D: physical (8, 20, 7, 9), valid (3, 7, 5, 4) — every axis unequal, C/H/W unaligned
TEST_F(TTRANSConvValidShapeTest, case_api_nchw_axis_valid)
{
    using CT = ConvTile<
        TileType::Vec, half, 8 * 20 * 7 * 9 * sizeof(half), Layout::NCHW, ConvTileShape<8, 20, 7, 9>,
        ConvTileValidShape<3, 7, 5, 4>>;
    CT tile;

    EXPECT_EQ(tile.GetShape(0), 8);
    EXPECT_EQ(tile.GetShape(1), 20);
    EXPECT_EQ(tile.GetShape(2), 7);
    EXPECT_EQ(tile.GetShape(3), 9);

    EXPECT_EQ(tile.GetValidShape(0), 3);
    EXPECT_EQ(tile.GetValidShape(1), 7);
    EXPECT_EQ(tile.GetValidShape(2), 5);
    EXPECT_EQ(tile.GetValidShape(3), 4);

    EXPECT_EQ(tile.GetValidShape(4), -1);
    EXPECT_EQ(tile.GetShape(4), -1);
}

// NC1HWC0 5D: physical (8, 2, 7, 9, 16), valid (3, 1, 5, 4, 7) — C0 axis unequal + unaligned
TEST_F(TTRANSConvValidShapeTest, case_api_nc1hwc0_axis_valid)
{
    using CT = ConvTile<
        TileType::Vec, half, 8 * 2 * 7 * 9 * 16 * sizeof(half), Layout::NC1HWC0, ConvTileShape<8, 2, 7, 9, 16>,
        ConvTileValidShape<3, 1, 5, 4, 7>>;
    CT tile;

    const int64_t phys[] = {8, 2, 7, 9, 16};
    const int64_t valid[] = {3, 1, 5, 4, 7};
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(tile.GetShape(i), phys[i]);
        EXPECT_EQ(tile.GetValidShape(i), valid[i]);
    }
}

// FRACTAL_Z 4D: physical (126, 2, 4, 16), valid (63, 1, 3, 7)
TEST_F(TTRANSConvValidShapeTest, case_api_fractal_z_axis_valid)
{
    using CT = ConvTile<
        TileType::Vec, half, 126 * 2 * 4 * 16 * sizeof(half), Layout::FRACTAL_Z, ConvTileShape<126, 2, 4, 16>,
        ConvTileValidShape<63, 1, 3, 7>>;
    CT tile;

    const int64_t phys[] = {126, 2, 4, 16};
    const int64_t valid[] = {63, 1, 3, 7};
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(tile.GetShape(i), phys[i]);
        EXPECT_EQ(tile.GetValidShape(i), valid[i]);
    }
}

// GNCHW 5D: G axis keeps physical, N/C/H/W all unequal
TEST_F(TTRANSConvValidShapeTest, case_api_gnchw_axis_valid)
{
    using CT = ConvTile<
        TileType::Vec, half, 2 * 8 * 20 * 7 * 9 * sizeof(half), Layout::GNCHW, ConvTileShape<2, 8, 20, 7, 9>,
        ConvTileValidShape<2, 3, 7, 5, 4>>;
    CT tile;

    const int64_t phys[] = {2, 8, 20, 7, 9};
    const int64_t valid[] = {2, 3, 7, 5, 4};
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(tile.GetShape(i), phys[i]);
        EXPECT_EQ(tile.GetValidShape(i), valid[i]);
    }
}

// GNC1HWC0 6D: covers the 6th axis (C0) as well
TEST_F(TTRANSConvValidShapeTest, case_api_gnc1hwc0_axis_valid)
{
    using CT = ConvTile<
        TileType::Vec, half, 2 * 8 * 2 * 7 * 9 * 16 * sizeof(half), Layout::GNC1HWC0,
        ConvTileShape<2, 8, 2, 7, 9, 16>, ConvTileValidShape<2, 3, 1, 5, 4, 7>>;
    CT tile;

    const int64_t phys[] = {2, 8, 2, 7, 9, 16};
    const int64_t valid[] = {2, 3, 1, 5, 4, 7};
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(tile.GetShape(i), phys[i]);
        EXPECT_EQ(tile.GetValidShape(i), valid[i]);
    }
}

// =============================================================================
// Part 2. default ValidShape == physical (backward compatibility)
// =============================================================================
TEST_F(TTRANSConvValidShapeTest, case_api_default_valid_eq_physical)
{
    using NchwCT =
        ConvTile<TileType::Vec, half, 8 * 20 * 7 * 9 * sizeof(half), Layout::NCHW, ConvTileShape<8, 20, 7, 9>>;
    NchwCT nchwTile;
    for (int i = 0; i < nchwTile.totalDimCount; ++i) {
        EXPECT_EQ(nchwTile.GetValidShape(i), nchwTile.GetShape(i));
    }

    using Nc1hwc0CT = ConvTile<
        TileType::Vec, half, 8 * 2 * 7 * 9 * 16 * sizeof(half), Layout::NC1HWC0, ConvTileShape<8, 2, 7, 9, 16>>;
    Nc1hwc0CT nc1hwc0Tile;
    for (int i = 0; i < nc1hwc0Tile.totalDimCount; ++i) {
        EXPECT_EQ(nc1hwc0Tile.GetValidShape(i), nc1hwc0Tile.GetShape(i));
    }

    using GnchwCT =
        ConvTile<TileType::Vec, half, 2 * 8 * 20 * 7 * 9 * sizeof(half), Layout::GNCHW, ConvTileShape<2, 8, 20, 7, 9>>;
    GnchwCT gnchwTile;
    for (int i = 0; i < gnchwTile.totalDimCount; ++i) {
        EXPECT_EQ(gnchwTile.GetValidShape(i), gnchwTile.GetShape(i));
    }

    using Gnc1hwc0CT = ConvTile<
        TileType::Vec, half, 2 * 8 * 2 * 7 * 9 * 16 * sizeof(half), Layout::GNC1HWC0,
        ConvTileShape<2, 8, 2, 7, 9, 16>>;
    Gnc1hwc0CT gnc1hwc0Tile;
    for (int i = 0; i < gnc1hwc0Tile.totalDimCount; ++i) {
        EXPECT_EQ(gnc1hwc0Tile.GetValidShape(i), gnc1hwc0Tile.GetShape(i));
    }
}

// =============================================================================
// Part 3. dynamic ValidShape API: SetAllValidShape / SetValidShape per axis
// =============================================================================
TEST_F(TTRANSConvValidShapeTest, case_api_dynamic_valid_all_axis)
{
    using CT = ConvTile<
        TileType::Vec, half, 16 * 4 * 13 * 11 * sizeof(half), Layout::NCHW, ConvTileShape<DYNAMIC, 4, DYNAMIC, DYNAMIC>,
        ConvTileValidShape<DYNAMIC, 4, DYNAMIC, DYNAMIC>>;

    int64_t physN = 8;
    int64_t physH = 7;
    int64_t physW = 9;
    CT tile(physN, physH, physW);

    // dynamic valid dims default to the physical value
    EXPECT_EQ(tile.GetShape(0), 8);
    EXPECT_EQ(tile.GetShape(2), 7);
    EXPECT_EQ(tile.GetShape(3), 9);
    EXPECT_EQ(tile.GetValidShape(0), 8);
    EXPECT_EQ(tile.GetValidShape(1), 4);
    EXPECT_EQ(tile.GetValidShape(2), 7);
    EXPECT_EQ(tile.GetValidShape(3), 9);

    tile.SetAllValidShape(int64_t(3), int64_t(5), int64_t(4));
    EXPECT_EQ(tile.GetValidShape(0), 3);
    EXPECT_EQ(tile.GetValidShape(1), 4);
    EXPECT_EQ(tile.GetValidShape(2), 5);
    EXPECT_EQ(tile.GetValidShape(3), 4);

    tile.SetValidShape(0, int64_t(6));
    EXPECT_EQ(tile.GetValidShape(0), 6);

    // re-assigning the physical shape resets valid to physical
    tile.SetDynamicShape(int64_t(16), int64_t(13), int64_t(11));
    EXPECT_EQ(tile.GetShape(0), 16);
    EXPECT_EQ(tile.GetShape(2), 13);
    EXPECT_EQ(tile.GetShape(3), 11);
    EXPECT_EQ(tile.GetValidShape(0), 16);
    EXPECT_EQ(tile.GetValidShape(1), 4);
    EXPECT_EQ(tile.GetValidShape(2), 13);
    EXPECT_EQ(tile.GetValidShape(3), 11);
}

// =============================================================================
// Part 4. data driven: TTRANS with per-axis valid shape
//
// The CPU simulator implementation (include/pto/cpu/TTrans.hpp) derives all axes from
// GetShape(), i.e. ValidShape does not change the TTRANS result there. These cases
// therefore pin down that a per-axis valid configuration (unequal / unaligned) keeps
// TTRANS correct, and they exercise the same ConvTile types the NPU path consumes.
// =============================================================================

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <
    typename T, int format, int p0, int p1, int p2, int p3, int p4, int v0, int v1, int v2, int v3, int v4>
void test_ttrans_axis_valid()
{
    constexpr size_t DTypeSize = sizeof(T);
    constexpr size_t C0 = 32 / DTypeSize;

    size_t srcElemNum = 0;
    size_t dstElemNum = 0;
    if constexpr (format == 0) {
        constexpr size_t C1 = (p1 + C0 - 1) / C0;
        srcElemNum = static_cast<size_t>(p0) * p1 * p2 * p3;
        dstElemNum = static_cast<size_t>(p0) * C1 * p2 * p3 * C0;
    } else if constexpr (format == 1) {
        static_assert(p4 == C0, "C0 must match the dtype block size");
        srcElemNum = static_cast<size_t>(p0) * p1 * p2 * p3 * p4;
        dstElemNum = static_cast<size_t>(p0) * (p1 * p4) * p2 * p3;
    } else {
        static_assert(format == 2, "unsupported TTRANS validshape format");
        constexpr size_t C1 = (p2 + C0 - 1) / C0;
        srcElemNum = static_cast<size_t>(p0) * p1 * p2 * p3 * p4;
        dstElemNum = static_cast<size_t>(p0) * p1 * C1 * p3 * p4 * C0;
    }

    size_t srcFileSize = srcElemNum * sizeof(T);
    size_t dstFileSize = dstElemNum * sizeof(T);

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

    LaunchTTRANSConvAxisValid<T, format, p0, p1, p2, p3, p4, v0, v1, v2, v3, v4>(dstDevice, srcDevice, stream);

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
    size_t goldenSize = dstFileSize;
    size_t outputSize = dstFileSize;
    ReadFile(GetGoldenDir() + "/golden.bin", goldenSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputSize, result.data(), dstFileSize);
    EXPECT_EQ(goldenSize, dstFileSize);
    EXPECT_EQ(outputSize, dstFileSize);

    bool ret = ResultCmp<T>(golden, result, 0.001f);
    EXPECT_TRUE(ret);
}

/*-------------------- NCHW -> NC1HWC0 --------------------*/

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_n_half)
{
    test_ttrans_axis_valid<half, 0, 8, 8, 4, 4, 1, 3, 8, 4, 4, 1>();
}

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_c_half)
{
    test_ttrans_axis_valid<half, 0, 4, 20, 4, 4, 1, 4, 7, 4, 4, 1>();
}

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_h_half)
{
    test_ttrans_axis_valid<half, 0, 4, 16, 7, 4, 1, 4, 16, 5, 4, 1>();
}

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_w_half)
{
    test_ttrans_axis_valid<half, 0, 4, 16, 4, 9, 1, 4, 16, 4, 4, 1>();
}

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_all_half)
{
    test_ttrans_axis_valid<half, 0, 8, 20, 7, 9, 1, 3, 7, 5, 4, 1>();
}

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_all_float)
{
    test_ttrans_axis_valid<float, 0, 8, 20, 7, 9, 1, 3, 7, 5, 4, 1>();
}

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_eq_phys_half)
{
    test_ttrans_axis_valid<half, 0, 8, 20, 7, 9, 1, 8, 20, 7, 9, 1>();
}

TEST_F(TTRANSConvValidShapeTest, NCHW2NC1HWC0_valid_min_half)
{
    test_ttrans_axis_valid<half, 0, 8, 20, 7, 9, 1, 1, 1, 1, 1, 1>();
}

/*-------------------- NC1HWC0 -> NCHW --------------------*/

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_n_half)
{
    test_ttrans_axis_valid<half, 1, 8, 2, 4, 4, 16, 5, 2, 4, 4, 16>();
}

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_c1_half)
{
    test_ttrans_axis_valid<half, 1, 8, 2, 4, 4, 16, 8, 1, 4, 4, 16>();
}

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_c0_half)
{
    test_ttrans_axis_valid<half, 1, 8, 2, 4, 4, 16, 8, 2, 4, 4, 7>();
}

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_h_half)
{
    test_ttrans_axis_valid<half, 1, 4, 2, 7, 4, 16, 4, 2, 5, 4, 16>();
}

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_w_half)
{
    test_ttrans_axis_valid<half, 1, 4, 2, 4, 9, 16, 4, 2, 4, 4, 16>();
}

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_all_half)
{
    test_ttrans_axis_valid<half, 1, 8, 2, 7, 9, 16, 3, 1, 5, 4, 7>();
}

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_all_int32)
{
    test_ttrans_axis_valid<int32_t, 1, 8, 2, 7, 9, 8, 3, 1, 5, 4, 5>();
}

TEST_F(TTRANSConvValidShapeTest, NC1HWC02NCHW_valid_eq_phys_half)
{
    test_ttrans_axis_valid<half, 1, 8, 2, 7, 9, 16, 8, 2, 7, 9, 16>();
}

/*-------------------- GNCHW -> GNC1HWC0 --------------------*/

TEST_F(TTRANSConvValidShapeTest, GNCHW2GNC1HWC0_valid_g_half)
{
    test_ttrans_axis_valid<half, 2, 2, 4, 20, 5, 7, 1, 4, 20, 5, 7>();
}

TEST_F(TTRANSConvValidShapeTest, GNCHW2GNC1HWC0_valid_all_half)
{
    test_ttrans_axis_valid<half, 2, 2, 4, 20, 5, 7, 1, 3, 7, 4, 5>();
}
