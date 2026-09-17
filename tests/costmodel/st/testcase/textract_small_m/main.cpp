/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
template <typename Func>
void CapturePtoAssertion(Func invoke)
{
    testing::internal::CaptureStdout();
    std::set_terminate([] {
        std::cerr << testing::internal::GetCapturedStdout();
        std::abort();
    });
    invoke();
}

template <int Rows, int Cols, int ValidRows = Rows, int ValidCols = Cols>
using Source = Tile<TileType::Mat, half, Rows, Cols, BLayout::ColMajor, ValidRows, ValidCols, SLayout::RowMajor, 512>;

template <CompactMode Compact, int Cols = 128, int ValidCols = 63>
using Left = Tile<
    TileType::Left, half, 16, Cols, BLayout::RowMajor, 4, ValidCols, SLayout::RowMajor, 512, PadValue::Null, Compact>;

template <CompactMode Compact>
void CheckCopyExtent(uint64_t expectedRepeats)
{
    Source<32, 160> src;
    Left<Compact> dst;
    TASSIGN(src, 0x10000);
    TASSIGN(dst, 0);
    mocker::ResetTrace();
    TEXTRACT(dst, src, 15, 16);
    const auto& instructions = mocker::GetTrace().executed_pto;
    ASSERT_EQ(instructions.size(), 1);
    const auto& calls = instructions[0].cce_calls;
    ASSERT_EQ(calls.size(), 1);
    EXPECT_EQ(calls[0].name, "load_cbuf_to_ca");
    ASSERT_EQ(calls[0].args.size(), 9);
    EXPECT_EQ(calls[0].args[0], reinterpret_cast<uintptr_t>(dst.data()));
    EXPECT_EQ(calls[0].args[1], 0x10000 + 32 * 16 * 2 + 15 * 32);
    EXPECT_EQ(calls[0].args[3], expectedRepeats);
    EXPECT_EQ(calls[0].args[4], 2);
    EXPECT_EQ(calls[0].args[5], 0);
}

template <int Rows, int Cols, int ValidRows, int ValidCols>
void ExtractAt(uint16_t row, uint16_t col)
{
    Source<Rows, Cols, ValidRows, ValidCols> src;
    Left<CompactMode::Null, 64, 64> dst;
    TASSIGN(src, 0x10000);
    TASSIGN(dst, 0);
    TEXTRACT(dst, src, row, col);
}

template <CompactMode Compact>
void CheckDynamicFullM(uint64_t expectedColumns)
{
    Source<32, 160> src;
    Tile<
        TileType::Left, half, 16, 128, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Null,
        Compact>
        dst;
    dst.SetValidShape(16, 63);
    TASSIGN(src, 0x10000);
    TASSIGN(dst, 0);
    mocker::ResetTrace();
    TEXTRACT(dst, src, 16, 16);
    const auto& instructions = mocker::GetTrace().executed_pto;
    ASSERT_EQ(instructions.size(), 1);
    int copies = 0;
    for (const auto& call : instructions[0].cce_calls) {
        EXPECT_NE(call.name, "load_cbuf_to_ca");
        if (call.name == "img2colv2_cbuf_to_ca") {
            ++copies;
            ASSERT_GE(call.args.size(), 4);
            EXPECT_EQ(call.args[2], expectedColumns);
            EXPECT_EQ(call.args[3], 16);
        }
    }
    EXPECT_EQ(copies, 1);
}
} // namespace

TEST(TExtractSmallM, NormalCopiesPhysicalColumns) { CheckCopyExtent<CompactMode::Null>(8); }
TEST(TExtractSmallM, CompactCopiesRoundedValidColumns) { CheckCopyExtent<CompactMode::Normal>(4); }
TEST(TExtractSmallM, DynamicFullMNormalKeepsLoad3d) { CheckDynamicFullM<CompactMode::Null>(128); }
TEST(TExtractSmallM, DynamicFullMCompactKeepsLoad3d) { CheckDynamicFullM<CompactMode::Normal>(64); }

TEST(TExtractSmallM, FourHeadsUseOneLoad2dEach)
{
    Source<16, 80, 16, 64> src;
    Left<CompactMode::Null, 64, 64> dst;
    TASSIGN(src, 0x10000);
    TASSIGN(dst, 0);
    mocker::ResetTrace();
    for (uint16_t row : {0, 4, 8, 12}) {
        TEXTRACT(dst, src, row, 0);
    }
    const auto& instructions = mocker::GetTrace().executed_pto;
    ASSERT_EQ(instructions.size(), 4);
    for (int head = 0; head < 4; ++head) {
        const auto& calls = instructions[head].cce_calls;
        ASSERT_EQ(calls.size(), 1);
        EXPECT_EQ(calls[0].name, "load_cbuf_to_ca");
        ASSERT_EQ(calls[0].args.size(), 9);
        EXPECT_EQ(calls[0].args[1], 0x10000 + head * 4 * 32);
        EXPECT_EQ(calls[0].args[3], 4);
    }
}

TEST(TExtractSmallM, ExactPhysicalReadBoundary) { ExtractAt<16, 64, 16, 64>(0, 0); }

TEST(TExtractSmallM, CompactFitsSourceSmallerThanDestinationPhysicalWidth)
{
    Source<32, 64> src;
    Left<CompactMode::Normal> dst;
    TASSIGN(src, 0x10000);
    TASSIGN(dst, 0);
    mocker::ResetTrace();
    TEXTRACT(dst, src, 0, 0);
    const auto& instructions = mocker::GetTrace().executed_pto;
    ASSERT_EQ(instructions.size(), 1);
    ASSERT_EQ(instructions[0].cce_calls.size(), 1);
    const auto& call = instructions[0].cce_calls[0];
    ASSERT_EQ(call.args.size(), 9);
    EXPECT_EQ(call.name, "load_cbuf_to_ca");
    EXPECT_EQ(call.args[3], 4);
}

TEST(TExtractSmallM, DynamicSourceValidWindow)
{
    Source<32, 96, DYNAMIC, DYNAMIC> src(16, 64);
    Left<CompactMode::Null, 64, 64> dst;
    TASSIGN(src, 0x10000);
    TASSIGN(dst, 0);
    TEXTRACT(dst, src, 12, 0);
}

// This oversized L0A tile is only an address model for the 8-bit repeat limit.
TEST(TExtractSmallM, RepeatChunkingAddressModel)
{
    Source<16, 4112, 16, 4096> src;
    Left<CompactMode::Null, 4096, 4096> dst;
    TASSIGN(src, 0x10000);
    TASSIGN(dst, 0);
    mocker::ResetTrace();
    TEXTRACT(dst, src, 12, 0);
    const auto& instructions = mocker::GetTrace().executed_pto;
    ASSERT_EQ(instructions.size(), 1);
    const auto& calls = instructions[0].cce_calls;
    ASSERT_EQ(calls.size(), 2);
    for (const auto& call : calls) {
        EXPECT_EQ(call.name, "load_cbuf_to_ca");
        ASSERT_EQ(call.args.size(), 9);
        EXPECT_EQ(call.args[4], 1);
        EXPECT_EQ(call.args[5], 0);
    }
    EXPECT_EQ(calls[0].args[3], 255);
    EXPECT_EQ(calls[1].args[3], 1);
    EXPECT_EQ(calls[0].args[1], 0x10000 + 12 * 32);
    EXPECT_EQ(calls[1].args[0] - calls[0].args[0], 255 * 512);
    EXPECT_EQ(calls[1].args[1] - calls[0].args[1], 255 * 512);
}

TEST(TExtractSmallMDeathTest, RejectsMissingPhysicalGuard)
{
    for (uint16_t row : {4, 8, 12}) {
        EXPECT_DEATH(
            CapturePtoAssertion([&] { ExtractAt<16, 64, 16, 64>(row, 0); }),
            "full-fractal read exceeds source storage");
    }
}
TEST(TExtractSmallMDeathTest, RejectsUnalignedColumn)
{
    EXPECT_DEATH(CapturePtoAssertion([] { ExtractAt<32, 96, 32, 96>(0, 1); }), "indexCol must be C0-aligned");
}
TEST(TExtractSmallMDeathTest, RejectsInvalidRowWindow)
{
    EXPECT_DEATH(CapturePtoAssertion([] { ExtractAt<32, 96, 8, 96>(8, 0); }), "window exceeds source valid rows");
}
TEST(TExtractSmallMDeathTest, RejectsInvalidColumnWindow)
{
    EXPECT_DEATH(CapturePtoAssertion([] { ExtractAt<32, 96, 32, 64>(0, 16); }), "window exceeds source valid columns");
}
TEST(TExtractSmallMDeathTest, RejectsNormalCopyWiderThanSource)
{
    Source<32, 64> src;
    Left<CompactMode::Null> dst;
    EXPECT_DEATH(CapturePtoAssertion([&] { TEXTRACT(dst, src, 0, 0); }), "copy exceeds source physical columns");
}
TEST(TExtractSmallMDeathTest, RejectsInvalidDynamicSourceWindow)
{
    Source<32, 96, DYNAMIC, DYNAMIC> src(15, 64);
    Left<CompactMode::Null, 64, 64> dst;
    EXPECT_DEATH(CapturePtoAssertion([&] { TEXTRACT(dst, src, 12, 0); }), "window exceeds source valid rows");
    src.SetValidShape(16, 63);
    EXPECT_DEATH(CapturePtoAssertion([&] { TEXTRACT(dst, src, 12, 0); }), "window exceeds source valid columns");
}
TEST(TExtractSmallMDeathTest, RejectsInvalidDynamicDestinationColumns)
{
    Source<32, 160> src;
    for (uint32_t columns : {0u, 129u}) {
        Tile<TileType::Left, half, 16, 128, BLayout::RowMajor, 4, DYNAMIC, SLayout::RowMajor, 512> dst(columns);
        EXPECT_DEATH(CapturePtoAssertion([&] { TEXTRACT(dst, src, 0, 0); }), "invalid valid columns");
    }
}
