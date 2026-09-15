/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <gtest/gtest.h>

struct A5Assertion {};
[[noreturn]] inline void trap() { throw A5Assertion{}; }

namespace cce {
template <typename... Args>
inline int printf(const char* fmt, Args... args);
}

#include "textract_a5_stubs.hpp"

namespace {
std::vector<std::array<uint64_t, 9>> calls;
}

// Model only the A5 non-transposed load2d primitive; checks and address selection
// below exercise the production A5 dispatcher and helper.
template <typename T>
void load_cbuf_to_ca(
    T* dst, T* src, uint16_t mStart, uint16_t kStart, uint8_t mStep, uint8_t kStep, uint16_t srcStride,
    uint16_t dstStride, int transpose)
{
    calls.push_back(
        {reinterpret_cast<uintptr_t>(dst), reinterpret_cast<uintptr_t>(src), mStart, kStart, mStep, kStep, srcStride,
         dstStride, static_cast<uint64_t>(transpose)});
    ASSERT_EQ(transpose, 0);
    for (uint32_t k = 0; k < kStep; ++k) {
        for (uint32_t m = 0; m < mStep; ++m) {
            std::memcpy(
                reinterpret_cast<uint8_t*>(dst) + (k * dstStride + m) * 512,
                reinterpret_cast<uint8_t*>(src) + ((kStart + k) * srcStride + mStart + m) * 512, 512);
        }
    }
}

#include <pto/npu/a5/TExtract.hpp>

using namespace pto;

namespace {
// Own host storage independently of simulated L1/L0A capacity.
template <typename TileData>
struct HostTile : TileData {
    std::vector<typename TileData::DType> storage;
    template <typename... Args>
    explicit HostTile(Args... args) : TileData(args...), storage(TileData::Numel)
    {
        this->data() = storage.data();
    }
};

template <typename T, int R, int C, int M = R, int K = C>
using Source = HostTile<Tile<TileType::Mat, T, R, C, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;

template <typename T, CompactMode Compact, int C = 128, int M = 4, int K = 63>
using Left =
    HostTile<Tile<TileType::Left, T, 16, C, BLayout::ColMajor, M, K, SLayout::RowMajor, 512, PadValue::Null, Compact>>;

template <typename Dst, typename Src>
void Extract(Dst& dst, Src& src, uint16_t row, uint16_t col)
{
    static_assert(IsTExtractMatToLeftSmallM<Dst, Src>());
    calls.clear();
    TEXTRACT_TILE_IMPL(dst, src, row, col);
}

template <typename T, CompactMode Compact>
void CheckValues()
{
    Source<T, 32, 160> src;
    Left<T, Compact> dst;
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 160; ++c) {
            src.SetElement(r, c, T((r * 7 + c) % 31 - 15));
        }
    }
    Extract(dst, src, 15, 16);
    ASSERT_EQ(calls.size(), 1);
    EXPECT_EQ(calls[0][1] - reinterpret_cast<uintptr_t>(src.data()), 32 * 16 * 2 + 15 * 32);
    EXPECT_EQ(calls[0][2], 0);
    EXPECT_EQ(calls[0][3], 0);
    EXPECT_EQ(calls[0][4], 1);
    EXPECT_EQ(calls[0][5], Compact == CompactMode::Normal ? 4 : 8);
    EXPECT_EQ(calls[0][6], 2);
    EXPECT_EQ(calls[0][7], 1);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 63; ++c) {
            EXPECT_EQ(float(dst.GetElement(r, c)), float(src.GetElement(r + 15, c + 16)));
        }
    }
}

template <CompactMode Compact>
void CheckFullM()
{
    Source<half, 32, 160> src;
    Left<half, Compact, 128, DYNAMIC, DYNAMIC> dst(16, 63);
    Extract(dst, src, 16, 16);
    ASSERT_EQ(calls.size(), 1);
    EXPECT_EQ(calls[0][1], reinterpret_cast<uintptr_t>(src.data()));
    EXPECT_EQ(calls[0][2], 1);
    EXPECT_EQ(calls[0][3], 1);
    EXPECT_EQ(calls[0][4], 1);
    EXPECT_EQ(calls[0][5], Compact == CompactMode::Normal ? 4 : 8);
    EXPECT_EQ(calls[0][6], 2);
    EXPECT_EQ(calls[0][7], 1);
}

template <typename Func>
void Rejects(Func invoke, const char* message)
{
    testing::internal::CaptureStdout();
    EXPECT_THROW(invoke(), A5Assertion);
    const auto output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find(message), std::string::npos) << output;
}
} // namespace

TEST(TExtractSmallMA5, HalfOrdinary) { CheckValues<half, CompactMode::Null>(); }
TEST(TExtractSmallMA5, HalfCompact) { CheckValues<half, CompactMode::Normal>(); }
#ifdef CPU_SIM_BFLOAT_ENABLED
static_assert(!std::is_same_v<half, bfloat16_t>);
TEST(TExtractSmallMA5, Bfloat16Ordinary) { CheckValues<bfloat16_t, CompactMode::Null>(); }
TEST(TExtractSmallMA5, Bfloat16Compact) { CheckValues<bfloat16_t, CompactMode::Normal>(); }
#endif
TEST(TExtractSmallMA5, DynamicFullMOrdinary) { CheckFullM<CompactMode::Null>(); }
TEST(TExtractSmallMA5, DynamicFullMCompact) { CheckFullM<CompactMode::Normal>(); }

TEST(TExtractSmallMA5, FourHeads)
{
    Source<half, 16, 80, 16, 64> src;
    Left<half, CompactMode::Null, 64, 4, 64> dst;
    for (int row : {0, 4, 8, 12}) {
        Extract(dst, src, row, 0);
        ASSERT_EQ(calls.size(), 1);
        EXPECT_EQ(calls[0][1] - reinterpret_cast<uintptr_t>(src.data()), row * 32);
        EXPECT_EQ(calls[0][5], 4);
    }
}

TEST(TExtractSmallMA5, DynamicSmallMAndTailK)
{
    Source<half, 32, 96, DYNAMIC, DYNAMIC> src(32, 80);
    Left<half, CompactMode::Normal, 128, DYNAMIC, DYNAMIC> dst;
    for (int m : {1, 2, 4, 8, 15}) {
        for (int k : {1, 17, 63, 64}) {
            dst.SetValidShape(m, k);
            Extract(dst, src, 32 - m, 16);
            ASSERT_EQ(calls.size(), 1);
            EXPECT_EQ(calls[0][5], (k + 15) / 16);
        }
    }
}

TEST(TExtractSmallMA5, ExactReadEnd)
{
    Source<half, 16, 64> src;
    Left<half, CompactMode::Null, 64, 4, 64> dst;
    Extract(dst, src, 0, 0);
    ASSERT_EQ(calls.size(), 1);
    EXPECT_EQ(calls[0][5] * 512, sizeof(half) * src.Numel);
}

TEST(TExtractSmallMA5, FullMCompactKeepsSmallerSourceSupport)
{
    Source<half, 32, 64> src;
    Left<half, CompactMode::Normal, 128, DYNAMIC, DYNAMIC> dst(16, 63);
    Extract(dst, src, 16, 0);
    ASSERT_EQ(calls.size(), 1);
    EXPECT_EQ(calls[0][2], 1);
    EXPECT_EQ(calls[0][5], 4);
}

// This tests 255+1 chunk arithmetic in host storage, not hardware L0A capacity.
TEST(TExtractSmallMA5, KStepChunkingAddressModel)
{
    Source<half, 32, 4112, 32, 4096> src;
    Left<half, CompactMode::Null, 4096, 4, 4096> dst;
    Extract(dst, src, 28, 0);
    ASSERT_EQ(calls.size(), 2);
    EXPECT_EQ(calls[0][5], 255);
    EXPECT_EQ(calls[1][5], 1);
    EXPECT_EQ(calls[1][0] - calls[0][0], 255 * 512);
    EXPECT_EQ(calls[1][1] - calls[0][1], 255 * 2 * 512);
}

TEST(TExtractSmallMA5, RejectsMissingGuard)
{
    Source<half, 16, 64> src;
    Left<half, CompactMode::Null, 64, 4, 64> dst;
    for (int row : {4, 8, 12}) {
        Rejects([&] { Extract(dst, src, row, 0); }, "full-fractal read exceeds source storage");
    }
}

TEST(TExtractSmallMA5, RejectsUnalignedColumn)
{
    Source<half, 32, 160> src;
    Left<half, CompactMode::Null> dst;
    Rejects([&] { Extract(dst, src, 0, 1); }, "indexCol must be C0-aligned");
}

TEST(TExtractSmallMA5, RejectsSourceValidWindow)
{
    Source<half, 32, 96, DYNAMIC, DYNAMIC> src(15, 64);
    Left<half, CompactMode::Null, 64, 4, 64> dst;
    Rejects([&] { Extract(dst, src, 12, 0); }, "window exceeds source valid rows");
    src.SetValidShape(16, 63);
    Rejects([&] { Extract(dst, src, 12, 0); }, "window exceeds source valid columns");
}

TEST(TExtractSmallMA5, RejectsOrdinaryWidthBeyondSource)
{
    Source<half, 32, 64> src;
    Left<half, CompactMode::Null> dst;
    Rejects([&] { Extract(dst, src, 0, 0); }, "copy exceeds source physical columns");
}

TEST(TExtractSmallMA5, RejectsInvalidDestinationColumns)
{
    Source<half, 32, 160> src;
    for (int k : {0, 129}) {
        Left<half, CompactMode::Null, 128, 4, DYNAMIC> dst(k);
        Rejects([&] { Extract(dst, src, 0, 0); }, "invalid valid columns");
    }
}
