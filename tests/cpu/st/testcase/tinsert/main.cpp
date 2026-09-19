/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#include "test_common.h"

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

constexpr struct {
    BLayout bl;
    SLayout sl;
} IDS2LAYOUTS[] = {
    {BLayout::RowMajor, SLayout::NoneBox},  // ND
    {BLayout::ColMajor, SLayout::NoneBox},  // DN
    {BLayout::ColMajor, SLayout::RowMajor}, // NZ
    {BLayout::RowMajor, SLayout::ColMajor}, // ZN
    {BLayout::RowMajor, SLayout::RowMajor}, // ZZ
    {BLayout::ColMajor, SLayout::ColMajor}  // NN
};

template <
    typename ST, typename DT, TileType locSrc, TileType locDst, size_t rows, size_t cols, size_t srcValidRows,
    size_t srcValidCols, size_t dstValidRows, size_t dstValidCols, uint16_t srcLayout, uint16_t dstLayout>
AICORE inline void runTINSERT(__gm__ DT* out, __gm__ ST* src)
{
    constexpr int idxRow = dstValidRows - srcValidRows;
    constexpr int idxCol = dstValidCols - srcValidCols;

    using GlobalDataSrc = GlobalTensor<
        ST, pto::Shape<1, 1, 1, srcValidRows, srcValidCols>,
        pto::Stride<
            1 * srcValidRows * srcValidCols, 1 * srcValidRows * srcValidCols, srcValidRows * srcValidCols, srcValidCols,
            1>>;
    using GlobalDataDst = GlobalTensor<
        DT, pto::Shape<1, 1, 1, dstValidRows, dstValidCols>,
        pto::Stride<
            1 * dstValidRows * dstValidCols, 1 * dstValidRows * dstValidCols, dstValidRows * dstValidCols, dstValidCols,
            1>>;

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(out);

    // 0-ND, 1-DN, 2 - NZ, 3 - ZN, 4 - ZZ, 5 - NN
    constexpr BLayout srcBL = IDS2LAYOUTS[srcLayout].bl;
    constexpr SLayout srcSL = IDS2LAYOUTS[srcLayout].sl;
    constexpr BLayout dstBL = IDS2LAYOUTS[dstLayout].bl;
    constexpr SLayout dstSL = IDS2LAYOUTS[dstLayout].sl;

    Tile<locSrc, ST, rows, cols, srcBL, srcValidRows, srcValidCols, srcSL, 512> srcTile;
    Tile<locDst, DT, rows, cols, dstBL, dstValidRows, dstValidCols, dstSL, 512> dstTile;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    std::fill(dstTile.data(), dstTile.data() + rows * cols, 0);

    /*************************************TLOAD****************************************/
    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /********************************** TINSERT**********************************/
    TINSERT(dstTile, srcTile, idxRow, idxCol);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

class TINSERTTest : public testing::Test {
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

template <
    typename ST, typename DT, TileType locSrc, TileType locDst, size_t rows, size_t cols, size_t srcValidRows,
    size_t srcValidCols, size_t dstValidRows, size_t dstValidCols, uint16_t srcLayout, uint16_t dstLayout>
void tinsert_test()
{
    size_t srcFileSize = srcValidRows * srcValidCols * sizeof(ST);
    size_t dstFileSize = dstValidRows * dstValidCols * sizeof(DT);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcHost;
    uint8_t *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t inputSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", inputSize, srcHost, srcFileSize));
    aclrtMemset(dstDevice, dstFileSize, 0, dstFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    runTINSERT<
        ST, DT, locSrc, locDst, rows, cols, srcValidRows, srcValidCols, dstValidRows, dstValidCols, srcLayout,
        dstLayout>((DT*)dstDevice, (ST*)srcDevice);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    std::vector<DT> golden(dstFileSize / sizeof(DT));
    size_t goldenSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", goldenSize, golden.data(), dstFileSize));

    bool ret = ResultCmp(golden, (DT*)dstHost, 0, 100, 1000, false, true);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    EXPECT_TRUE(ret);
}

TEST_F(TINSERTTest, case_half_half_Mat_Mat_32_32_32_32_DST_32_32_L_0_0)
{
    tinsert_test<half, half, TileType::Mat, TileType::Mat, 32, 32, 32, 32, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_half_float_Mat_Mat_32_32_32_32_DST_32_32_L_0_0)
{
    tinsert_test<half, float, TileType::Mat, TileType::Mat, 32, 32, 32, 32, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_128_96_DST_128_96_L_0_0)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 128, 96, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int32_t_float_Mat_Mat_128_96_128_96_DST_128_96_L_0_0)
{
    tinsert_test<int32_t, float, TileType::Mat, TileType::Mat, 128, 96, 128, 96, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int8_t_int32_t_Mat_Mat_128_64_128_64_DST_128_64_L_0_0)
{
    tinsert_test<int8_t, int32_t, TileType::Mat, TileType::Mat, 128, 64, 128, 64, 128, 64, 0, 0>();
}

TEST_F(TINSERTTest, case_half_half_Mat_Mat_32_32_24_16_DST_24_16_L_0_0)
{
    tinsert_test<half, half, TileType::Mat, TileType::Mat, 32, 32, 24, 16, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_half_float_Mat_Mat_32_32_24_16_DST_24_16_L_0_0)
{
    tinsert_test<half, float, TileType::Mat, TileType::Mat, 32, 32, 24, 16, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_24_16_DST_24_16_L_0_0)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 24, 16, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int32_t_float_Mat_Mat_128_96_24_16_DST_24_16_L_0_0)
{
    tinsert_test<int32_t, float, TileType::Mat, TileType::Mat, 128, 96, 24, 16, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int8_t_int32_t_Mat_Mat_128_64_24_16_DST_24_16_L_0_0)
{
    tinsert_test<int8_t, int32_t, TileType::Mat, TileType::Mat, 128, 64, 24, 16, 128, 64, 0, 0>();
}

TEST_F(TINSERTTest, case_half_half_Mat_Mat_32_32_23_16_DST_23_16_L_0_0)
{
    tinsert_test<half, half, TileType::Mat, TileType::Mat, 32, 32, 23, 16, 31, 31, 0, 0>();
}

TEST_F(TINSERTTest, case_half_float_Mat_Mat_32_32_23_16_DST_23_16_L_0_0)
{
    tinsert_test<half, float, TileType::Mat, TileType::Mat, 32, 32, 23, 16, 31, 31, 0, 0>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_23_16_DST_23_16_L_0_0)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 23, 16, 125, 93, 0, 0>();
}

TEST_F(TINSERTTest, case_int32_t_float_Mat_Mat_128_96_23_16_DST_23_16_L_0_0)
{
    tinsert_test<int32_t, float, TileType::Mat, TileType::Mat, 128, 96, 23, 16, 125, 93, 0, 0>();
}

TEST_F(TINSERTTest, case_int8_t_int32_t_Mat_Mat_128_64_23_16_DST_23_16_L_0_0)
{
    tinsert_test<int8_t, int32_t, TileType::Mat, TileType::Mat, 128, 64, 23, 16, 125, 61, 0, 0>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_0_1)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 0, 1>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_0_2)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 0, 2>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_1_0)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 1, 0>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_1_1)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 1, 1>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_1_2)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 1, 2>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_2_0)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 2, 0>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_2_1)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 2, 1>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_2_2)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 2, 2>();
}

TEST_F(TINSERTTest, case_half_half_Mat_Vec_32_32_8_16_DST_8_16_L_0_0)
{
    tinsert_test<half, half, TileType::Mat, TileType::Vec, 32, 32, 8, 16, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_half_float_Mat_Vec_32_32_8_16_DST_8_16_L_0_0)
{
    tinsert_test<half, float, TileType::Mat, TileType::Vec, 32, 32, 8, 16, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Vec_128_96_8_16_DST_8_16_L_0_0)
{
    tinsert_test<float, float, TileType::Mat, TileType::Vec, 128, 96, 8, 16, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int32_t_float_Mat_Vec_128_96_8_16_DST_8_16_L_0_0)
{
    tinsert_test<int32_t, float, TileType::Mat, TileType::Vec, 128, 96, 8, 16, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int8_t_int32_t_Mat_Vec_128_64_8_16_DST_8_16_L_0_0)
{
    tinsert_test<int8_t, int32_t, TileType::Mat, TileType::Vec, 128, 64, 8, 16, 128, 64, 0, 0>();
}

TEST_F(TINSERTTest, case_half_half_Vec_Vec_32_32_8_16_DST_8_16_L_0_0)
{
    tinsert_test<half, half, TileType::Vec, TileType::Vec, 32, 32, 8, 16, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_half_float_Vec_Vec_32_32_8_16_DST_8_16_L_0_0)
{
    tinsert_test<half, float, TileType::Vec, TileType::Vec, 32, 32, 8, 16, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_float_float_Vec_Vec_128_96_8_16_DST_8_16_L_0_0)
{
    tinsert_test<float, float, TileType::Vec, TileType::Vec, 128, 96, 8, 16, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int32_t_float_Vec_Vec_128_96_8_16_DST_8_16_L_0_0)
{
    tinsert_test<int32_t, float, TileType::Vec, TileType::Vec, 128, 96, 8, 16, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int8_t_int32_t_Vec_Vec_128_64_8_16_DST_8_16_L_0_0)
{
    tinsert_test<int8_t, int32_t, TileType::Vec, TileType::Vec, 128, 64, 8, 16, 128, 64, 0, 0>();
}

TEST_F(TINSERTTest, case_half_float_Vec_Vec_32_32_8_16_DST_8_16_L_0_3)
{
    tinsert_test<half, float, TileType::Vec, TileType::Vec, 32, 32, 8, 16, 32, 32, 0, 3>();
}

TEST_F(TINSERTTest, case_float_float_Vec_Vec_128_96_8_16_DST_8_16_L_0_4)
{
    tinsert_test<float, float, TileType::Vec, TileType::Vec, 128, 96, 8, 16, 128, 96, 0, 4>();
}

TEST_F(TINSERTTest, case_int32_t_float_Vec_Vec_128_96_8_16_DST_8_16_L_0_5)
{
    tinsert_test<int32_t, float, TileType::Vec, TileType::Vec, 128, 96, 8, 16, 128, 96, 0, 5>();
}

TEST_F(TINSERTTest, case_half_float_Vec_Vec_32_32_8_16_DST_8_16_L_3_1)
{
    tinsert_test<half, float, TileType::Vec, TileType::Vec, 32, 32, 8, 16, 32, 32, 3, 1>();
}

TEST_F(TINSERTTest, case_float_float_Vec_Vec_128_96_8_16_DST_8_16_L_4_1)
{
    tinsert_test<float, float, TileType::Vec, TileType::Vec, 128, 96, 8, 16, 128, 96, 4, 1>();
}

TEST_F(TINSERTTest, case_int32_t_float_Vec_Vec_128_96_8_16_DST_8_16_L_5_1)
{
    tinsert_test<int32_t, float, TileType::Vec, TileType::Vec, 128, 96, 8, 16, 128, 96, 5, 1>();
}

TEST_F(TINSERTTest, case_half_float_Vec_Vec_32_32_8_16_DST_8_16_L_3_2)
{
    tinsert_test<half, float, TileType::Vec, TileType::Vec, 32, 32, 8, 16, 32, 32, 3, 2>();
}

TEST_F(TINSERTTest, case_float_float_Vec_Vec_128_96_8_16_DST_8_16_L_3_2)
{
    tinsert_test<float, float, TileType::Vec, TileType::Vec, 128, 96, 8, 16, 128, 96, 3, 2>();
}

TEST_F(TINSERTTest, case_int32_t_float_Vec_Vec_128_96_8_16_DST_8_16_L_3_2)
{
    tinsert_test<int32_t, float, TileType::Vec, TileType::Vec, 128, 96, 8, 16, 128, 96, 3, 2>();
}

TEST_F(TINSERTTest, case_half_half_Mat_Mat_32_32_24_16_DST_24_16_L_3_2)
{
    tinsert_test<half, half, TileType::Mat, TileType::Mat, 32, 32, 24, 16, 32, 32, 3, 2>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_3_2)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 3, 2>();
}

namespace {
template <typename T, int Rows, int Cols, int ValidRows, int ValidCols, int IndexCol, bool FullLoad>
void checkNzWindowInsert()
{
    NPU_MEMORY_INIT(NPUArch::A5);
    constexpr int C0_ELEMENTS = BLOCK_BYTE_SIZE / sizeof(T);
    using Global = GlobalTensor<
        T, Shape<1, Cols / C0_ELEMENTS, Rows / 16, 16, C0_ELEMENTS>,
        Stride<Rows * Cols, Rows * C0_ELEMENTS, 16 * C0_ELEMENTS, C0_ELEMENTS, 1>, Layout::NZ>;
    using Src = Tile<
        TileType::Vec, T, Rows, Cols, BLayout::ColMajor, ValidRows, ValidCols, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>;
    using Dst = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    std::vector<T> input(Rows * Cols);
    for (int row = 0; row < Rows; ++row) {
        for (int col = 0; col < Cols; ++col) {
            input[(col / C0_ELEMENTS) * Rows * C0_ELEMENTS + row * C0_ELEMENTS + col % C0_ELEMENTS] =
                T((row * 3 + col) % 127);
        }
    }
    Global gm(input.data());
    Src src;
    Dst dst;
    TASSIGN(src, 0);
    TASSIGN(dst, 0);
    std::fill_n(src.data(), Rows * Cols, T(-1));
    std::fill_n(dst.data(), Rows * Cols, T(-1));
    if constexpr (FullLoad) {
        // Same physical NZ storage, with a full valid shape to isolate TINSERT from TLOAD.
        Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor> full;
        TASSIGN(full, 0);
        TLOAD(full, gm);
    } else {
        TLOAD(src, gm);
    }
    for (int row = 0; row < Rows; ++row) {
        for (int col = 0; col < Cols; ++col) {
            const float expected = row < (FullLoad ? Rows : ValidRows) ? (row * 3 + col) % 127 : -1;
            ASSERT_EQ(expected, static_cast<float>(src.GetElement(row, col))) << "row=" << row << " col=" << col;
        }
    }
    TINSERT(dst, src, 0, IndexCol);
    for (int row = 0; row < Rows; ++row) {
        for (int col = 0; col < Cols; ++col) {
            const float expected = row < ValidRows && col >= IndexCol && col < IndexCol + ValidCols ?
                                       (row * 3 + col - IndexCol) % 127 :
                                       -1;
            ASSERT_EQ(expected, static_cast<float>(dst.GetElement(row, col))) << "row=" << row << " col=" << col;
        }
    }
}

template <TInsertMode Mode>
void checkSplitInsert()
{
    using Src = Tile<
        TileType::Vec, float, 17, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
    using Dst = Tile<TileType::Mat, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor>;
    NPU_MEMORY_INIT(NPUArch::A5);
    Src src;
    Dst dst;
    TASSIGN(src, 0);
    TASSIGN(dst, 0);
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 64; ++col) {
            src.SetElement(row, col, row * 64 + col);
        }
    }
    TINSERT<Mode>(dst, src);
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 64; ++col) {
            ASSERT_EQ(row * 64 + col, dst.GetElement(row, col)) << "row=" << row << " col=" << col;
        }
    }
}
} // namespace

TEST(TInsertLogRepro, NzHalfIndex0) { checkNzWindowInsert<half, 128, 64, 64, 32, 0, false>(); }
TEST(TInsertLogRepro, NzHalfIndex32) { checkNzWindowInsert<half, 128, 64, 64, 32, 32, false>(); }
TEST(TInsertLogRepro, NzFloatIndex8) { checkNzWindowInsert<float, 64, 32, 32, 16, 8, false>(); }
TEST(TInsertLogRepro, NzHalfFullLoadControl) { checkNzWindowInsert<half, 128, 64, 64, 32, 32, true>(); }
TEST(TInsertLogRepro, NzFloatFullLoadControl) { checkNzWindowInsert<float, 64, 32, 32, 16, 8, true>(); }
#if defined(CPU_SIM_BFLOAT_ENABLED)
TEST(TInsertLogRepro, NzBfloat16Index16) { checkNzWindowInsert<bfloat16_t, 128, 128, 80, 48, 16, false>(); }
TEST(TInsertLogRepro, NzBfloat16FullLoadControl) { checkNzWindowInsert<bfloat16_t, 128, 128, 80, 48, 16, true>(); }
#endif

TEST(TInsertLogRepro, Split2PublicApi) { checkSplitInsert<TInsertMode::SPLIT2>(); }

TEST(TInsertLogRepro, Split4PublicApi) { checkSplitInsert<TInsertMode::SPLIT4>(); }

namespace {
template <TInsertMode Mode, CompactMode Compact>
void checkSplitAlignedTail()
{
    NPU_MEMORY_INIT(NPUArch::A5);
    constexpr int SRC_ROWS = Compact == CompactMode::Null ? 48 : 33;
    using Src = Tile<
        TileType::Vec, float, SRC_ROWS, 48, BLayout::ColMajor, 19, 35, SLayout::RowMajor, 512, PadValue::Null, Compact>;
    using Dst = Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor>;
    Src src;
    Dst dst;
    TASSIGN(src, 0);
    TASSIGN(dst, 0);
    std::fill_n(src.data(), SRC_ROWS * 48, -99.0f);
    std::fill_n(dst.data(), 64 * 64, -1.0f);
    int srcStrideRows = 32;
    if constexpr (Compact == CompactMode::Null) {
        srcStrideRows = 48;
    } else if constexpr (Compact == CompactMode::RowPlusOne) {
        srcStrideRows = 33;
    }
    for (int row = 0; row < 32; ++row) {
        for (int col = 0; col < 40; ++col) {
            src.data()[(col / 8) * srcStrideRows * 8 + row * 8 + col % 8] = row * 40 + col;
        }
    }
    TINSERT<Mode>(dst, src, 16, 8);
    // Five blocks exercise a split remainder; both row and column tails are transferred.
    for (int row = 0; row < 64; ++row) {
        for (int col = 0; col < 64; ++col) {
            const float expected = row >= 16 && row < 48 && col >= 8 && col < 48 ? (row - 16) * 40 + col - 8 : -1;
            ASSERT_EQ(expected, dst.GetElement(row, col)) << "row=" << row << " col=" << col;
        }
    }
}

template <typename A, typename B, int M = 16, bool RawHalfInput = false>
void checkAccToMatFp8()
{
    NPU_MEMORY_INIT(NPUArch::A5);
    constexpr int K = 32, N = 32;
    auto valueA = [](int row, int k) {
        if constexpr (RawHalfInput) {
            return 0.5f;
        } else {
            return ((row + 2 * k) % 7 - 3) * 0.5f;
        }
    };
    auto valueB = [](int k, int col) {
        if constexpr (RawHalfInput) {
            return 0.5f;
        } else {
            return ((k + 3 * col) % 5 - 2) * 0.5f;
        }
    };
    std::vector<A> src0(M * K);
    std::vector<B> src1(K * N);
    for (int row = 0; row < M; ++row) {
        for (int k = 0; k < K; ++k) {
            if constexpr (RawHalfInput) {
                static_assert(std::is_same_v<A, hifloat8_t>);
                src0[row * K + k] = A::FromRaw(0x18);
            } else {
                src0[row * K + k] = A(valueA(row, k));
            }
            ASSERT_EQ(valueA(row, k), static_cast<float>(src0[row * K + k]));
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int col = 0; col < N; ++col) {
            if constexpr (RawHalfInput) {
                static_assert(std::is_same_v<B, hifloat8_t>);
                src1[k * N + col] = B::FromRaw(0x18);
            } else {
                src1[k * N + col] = B(valueB(k, col));
            }
            ASSERT_EQ(valueB(k, col), static_cast<float>(src1[k * N + col]));
        }
    }
    GlobalTensor<A, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>> src0Global(src0.data());
    GlobalTensor<B, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, N, 1>> src1Global(src1.data());
    Tile<TileType::Mat, A, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor> src0Mat;
    Tile<TileType::Mat, B, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor> src1Mat;
    TileLeft<A, M, K, M, K> left;
    TileRight<B, K, N, K, N> right;
    TileAcc<float, M, N, M, N> acc;
    Tile<TileType::Mat, float, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor> dst;
    TASSIGN(src0Mat, 0);
    TASSIGN(src1Mat, 0x10000);
    TASSIGN(left, 0);
    TASSIGN(right, 0);
    TASSIGN(acc, 0);
    TASSIGN(dst, 0);
    TLOAD(src0Mat, src0Global);
    TLOAD(src1Mat, src1Global);
    TMOV(left, src0Mat);
    TMOV(right, src1Mat);
    TMATMUL(acc, left, right);
    TINSERT(dst, acc, 0, 0);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float expected = 0;
            for (int k = 0; k < K; ++k) {
                expected += valueA(row, k) * valueB(k, col);
            }
            ASSERT_EQ(expected, dst.GetElement(row, col)) << "row=" << row << " col=" << col;
        }
    }
}
} // namespace

TEST(TInsertLogRepro, Split2AlignedTail) { checkSplitAlignedTail<TInsertMode::SPLIT2, CompactMode::Null>(); }
TEST(TInsertLogRepro, Split4AlignedTail) { checkSplitAlignedTail<TInsertMode::SPLIT4, CompactMode::Null>(); }
TEST(TInsertLogRepro, Split2RowPlusOneTail) { checkSplitAlignedTail<TInsertMode::SPLIT2, CompactMode::RowPlusOne>(); }
TEST(TInsertLogRepro, Split4RowPlusOneTail) { checkSplitAlignedTail<TInsertMode::SPLIT4, CompactMode::RowPlusOne>(); }
TEST(TInsertLogRepro, Split2CompactTail) { checkSplitAlignedTail<TInsertMode::SPLIT2, CompactMode::Normal>(); }
TEST(TInsertLogRepro, Split4CompactTail) { checkSplitAlignedTail<TInsertMode::SPLIT4, CompactMode::Normal>(); }
TEST(TInsertLogRepro, AccE4M3) { checkAccToMatFp8<float8_e4m3_t, float8_e4m3_t>(); }
TEST(TInsertLogRepro, AccE5M2) { checkAccToMatFp8<float8_e5m2_t, float8_e5m2_t>(); }
TEST(TInsertLogRepro, AccMixed) { checkAccToMatFp8<float8_e4m3_t, float8_e5m2_t>(); }
TEST(TInsertLogRepro, AccMixedReverse) { checkAccToMatFp8<float8_e5m2_t, float8_e4m3_t>(); }
TEST(TInsertLogRepro, AccHif8) { checkAccToMatFp8<hifloat8_t, hifloat8_t>(); }
TEST(TInsertLogRepro, AccFp8MaxM) { checkAccToMatFp8<float8_e4m3_t, float8_e4m3_t, 2032>(); }

TEST(TInsertLogRepro, NzGappedOuterBlocks)
{
    NPU_MEMORY_INIT(NPUArch::A5);
    using Src = GlobalTensor<float, Shape<2, 2, 4, 16, 8>, Stride<1408, 640, 128, 8, 1>, Layout::NZ>;
    using Dst = Tile<TileType::Vec, float, 64, 48, BLayout::ColMajor, 19, 8, SLayout::RowMajor, 512, PadValue::Zero>;
    std::vector<float> input(2816, -99.0f);
    for (int row = 0; row < 64; ++row) {
        for (int col = 0; col < 32; ++col) {
            input[(col / 16) * 1408 + ((col / 8) % 2) * 640 + row * 8 + col % 8] = row * 32 + col;
        }
    }
    Src src(input.data());
    Dst dst;
    TASSIGN(dst, 0);
    std::fill_n(dst.data(), 64 * 48, -1.0f);
    TLOAD(dst, src);
    for (int row = 0; row < 64; ++row) {
        for (int col = 0; col < 48; ++col) {
            const float expected = row < 19 && col < 32 ? row * 32 + col : -1;
            ASSERT_EQ(expected, dst.GetElement(row, col)) << "row=" << row << " col=" << col;
        }
    }
}

#ifndef NDEBUG
TEST(TInsertLogRepro, A2A3NzStillRequiresMatchingShape)
{
    NPU_MEMORY_INIT(NPUArch::A2A3);
    using Global = GlobalTensor<float, Shape<1, 4, 4, 16, 8>, Stride<2048, 512, 128, 8, 1>, Layout::NZ>;
    std::vector<float> input(2048, 1.0f);
    Global src(input.data());
    Tile<TileType::Vec, float, 64, 32, BLayout::ColMajor, 32, 16, SLayout::RowMajor> dst;
    TASSIGN(dst, 0);
    EXPECT_DEATH(TLOAD(dst, src), "Assertion");
}

TEST(TInsertLogRepro, A2A3RejectsFp8Matmul)
{
    NPU_MEMORY_INIT(NPUArch::A2A3);
    TileLeft<float8_e4m3_t, 16, 32> left;
    TileRight<float8_e4m3_t, 32, 32> right;
    TileAcc<float, 16, 32> acc;
    EXPECT_DEATH(TMATMUL(acc, left, right), "FP8/HIF8 TMATMUL requires A5");
}
#endif

namespace {
template <typename T, TInsertMode Mode>
void checkSplitPackedFp4()
{
    NPU_MEMORY_INIT(NPUArch::A5);
    using Src = Tile<
        TileType::Vec, T, 33, 192, BLayout::ColMajor, 19, 160, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
    using Dst = Tile<TileType::Mat, T, 64, 256, BLayout::ColMajor, 64, 256, SLayout::RowMajor>;
    Src src;
    Dst dst;
    TASSIGN(src, 0);
    TASSIGN(dst, 0);
    auto* input = reinterpret_cast<uint8_t*>(src.data());
    auto* output = reinterpret_cast<uint8_t*>(dst.data());
    for (size_t i = 0; i < 33 * 192 / 2; ++i) {
        input[i] = static_cast<uint8_t>(i * 13 + 7);
    }
    std::fill_n(output, 64 * 256 / 2, 0xab);
    TINSERT<Mode>(dst, src, 16, 64);
    for (int block = 0; block < 4; ++block) {
        for (int row = 0; row < 64; ++row) {
            for (int byte = 0; byte < 32; ++byte) {
                const uint8_t expected =
                    block >= 1 && row >= 16 && row < 48 ? input[(block - 1) * 33 * 32 + (row - 16) * 32 + byte] : 0xab;
                ASSERT_EQ(expected, output[block * 64 * 32 + row * 32 + byte])
                    << "block=" << block << " row=" << row << " byte=" << byte;
            }
        }
    }
}
} // namespace

TEST(TInsertLogRepro, Split2Fp4E2M1) { checkSplitPackedFp4<float4_e2m1x2_t, TInsertMode::SPLIT2>(); }
TEST(TInsertLogRepro, Split4Fp4E2M1) { checkSplitPackedFp4<float4_e2m1x2_t, TInsertMode::SPLIT4>(); }
TEST(TInsertLogRepro, Split2Fp4E1M2) { checkSplitPackedFp4<float4_e1m2x2_t, TInsertMode::SPLIT2>(); }
TEST(TInsertLogRepro, Split4Fp4E1M2) { checkSplitPackedFp4<float4_e1m2x2_t, TInsertMode::SPLIT4>(); }

TEST(TInsertLogRepro, AccHif8ExternalHalf) { checkAccToMatFp8<hifloat8_t, hifloat8_t, 16, true>(); }

TEST(TInsertLogRepro, Hif8AllEncodings)
{
    // Golden values for raw 0x00..0x7f from en_dtypes 0.0.4, independent of the CPU decoder.
    constexpr double POSITIVE_VALUES[] = {
        0x0p+0,  0x1p-22,   0x1p-21,  0x1p-20,   0x1p-19,  0x1p-18,   0x1p-17,  0x1p-16,
        0x1p+0,  0x1.2p+0,  0x1.4p+0, 0x1.6p+0,  0x1.8p+0, 0x1.ap+0,  0x1.cp+0, 0x1.ep+0,
        0x1p+1,  0x1.2p+1,  0x1.4p+1, 0x1.6p+1,  0x1.8p+1, 0x1.ap+1,  0x1.cp+1, 0x1.ep+1,
        0x1p-1,  0x1.2p-1,  0x1.4p-1, 0x1.6p-1,  0x1.8p-1, 0x1.ap-1,  0x1.cp-1, 0x1.ep-1,
        0x1p+2,  0x1.2p+2,  0x1.4p+2, 0x1.6p+2,  0x1.8p+2, 0x1.ap+2,  0x1.cp+2, 0x1.ep+2,
        0x1p+3,  0x1.2p+3,  0x1.4p+3, 0x1.6p+3,  0x1.8p+3, 0x1.ap+3,  0x1.cp+3, 0x1.ep+3,
        0x1p-2,  0x1.2p-2,  0x1.4p-2, 0x1.6p-2,  0x1.8p-2, 0x1.ap-2,  0x1.cp-2, 0x1.ep-2,
        0x1p-3,  0x1.2p-3,  0x1.4p-3, 0x1.6p-3,  0x1.8p-3, 0x1.ap-3,  0x1.cp-3, 0x1.ep-3,
        0x1p+4,  0x1.4p+4,  0x1.8p+4, 0x1.cp+4,  0x1p+5,   0x1.4p+5,  0x1.8p+5, 0x1.cp+5,
        0x1p+6,  0x1.4p+6,  0x1.8p+6, 0x1.cp+6,  0x1p+7,   0x1.4p+7,  0x1.8p+7, 0x1.cp+7,
        0x1p-4,  0x1.4p-4,  0x1.8p-4, 0x1.cp-4,  0x1p-5,   0x1.4p-5,  0x1.8p-5, 0x1.cp-5,
        0x1p-6,  0x1.4p-6,  0x1.8p-6, 0x1.cp-6,  0x1p-7,   0x1.4p-7,  0x1.8p-7, 0x1.cp-7,
        0x1p+8,  0x1.8p+8,  0x1p+9,   0x1.8p+9,  0x1p+10,  0x1.8p+10, 0x1p+11,  0x1.8p+11,
        0x1p+12, 0x1.8p+12, 0x1p+13,  0x1.8p+13, 0x1p+14,  0x1.8p+14, 0x1p+15,  std::numeric_limits<double>::infinity(),
        0x1p-8,  0x1.8p-8,  0x1p-9,   0x1.8p-9,  0x1p-10,  0x1.8p-10, 0x1p-11,  0x1.8p-11,
        0x1p-12, 0x1.8p-12, 0x1p-13,  0x1.8p-13, 0x1p-14,  0x1.8p-14, 0x1p-15,  0x1.8p-15,
    };
    for (unsigned raw = 0; raw < 256; ++raw) {
        const double actual = static_cast<double>(hifloat8_t::FromRaw(static_cast<uint8_t>(raw)));
        if (raw == 0x80) {
            EXPECT_TRUE(std::isnan(actual));
            continue;
        }
        const double expected = (raw & 0x80) ? -POSITIVE_VALUES[raw & 0x7f] : POSITIVE_VALUES[raw];
        EXPECT_EQ(expected, actual) << "raw=" << raw;
        if (std::isfinite(expected)) {
            EXPECT_EQ(raw, hifloat8_t(expected).RawData()) << "value=" << expected;
        }
    }
}
