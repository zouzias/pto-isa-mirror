/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// CPU-sim TASSIGN tests:
//   - TAssignAliasTest:  an integer argument is a byte offset, a host pointer into
//     a simulated region is an alias to that storage.
//   - TAssignWindowTest: a windowed sub-view (full block shape, smaller valid
//     shape, non-zero byte offset) resolves against its valid-region footprint.

#include <cstdint>

#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
// A2/A3 L0C is 128 KB. A [128,256] int32 accumulator is exactly 128 KB, so it
// fills the whole region and leaves zero head-room for a non-zero strip offset.
constexpr std::size_t kL0CBytes = 128 * 1024;
constexpr int kRows = 128;
constexpr int kCols = 256;
constexpr int kStripRows = 16;

// Full accumulator: 128 rows valid (dynamic valid shape, exercises the Numel
// fallback path) placed at offset 0 -- the exact-fit boundary case.
using AccFull = Tile<TileType::Acc, int32_t, kRows, kCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
// 16-row strip view: static valid shape 16x256, the windowed sub-view PTOAS pushes.
using AccStrip =
    Tile<TileType::Acc, int32_t, kRows, kCols, BLayout::ColMajor, kStripRows, kCols, SLayout::RowMajor, 1024>;
} // namespace

class TAssignAliasTest : public testing::Test {
protected:
    void SetUp() override
    {
        NPU_MEMORY_INIT(NPUArch::A2A3);
        NPU_MEMORY_CLEAR();
    }
};

TEST_F(TAssignAliasTest, exact_alias_reuses_same_backing_buffer)
{
    using BaseTile = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, 8, 64>;
    using AliasTile = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, 8, 64>;

    BaseTile base;
    AliasTile alias;

    TASSIGN(base, 0);
    TASSIGN(alias, reinterpret_cast<std::uintptr_t>(base.data()));

    ASSERT_EQ(base.data(), alias.data());

    base.SetValue(5, 3.5f);
    EXPECT_FLOAT_EQ(alias.GetValue(5), 3.5f);

    alias.SetValue(17, -2.25f);
    EXPECT_FLOAT_EQ(base.GetValue(17), -2.25f);
}

TEST_F(TAssignAliasTest, interior_alias_treats_pointer_as_pointer_not_offset)
{
    using BaseTile = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, 8, 64>;
    using TailTile = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64>;

    constexpr std::size_t kRowOffset = 4;
    constexpr std::size_t kColCount = 64;
    constexpr std::size_t kElementOffset = kRowOffset * kColCount;

    BaseTile base;
    TailTile tail;

    TASSIGN(base, 0);
    TASSIGN(tail, reinterpret_cast<std::uintptr_t>(base.data() + kElementOffset));

    ASSERT_EQ(tail.data(), base.data() + kElementOffset);

    tail.SetValue(3, 9.0f);
    EXPECT_FLOAT_EQ(base.GetValue(kElementOffset + 3), 9.0f);

    base.SetValue(kElementOffset + 10, -7.0f);
    EXPECT_FLOAT_EQ(tail.GetValue(10), -7.0f);
}

class TAssignWindowTest : public testing::Test {
protected:
    void SetUp() override
    {
        NPU_MEMORY_INIT(NPUArch::A2A3);
        NPU_MEMORY_CLEAR();
    }
};

// The full 128x256 accumulator exactly fills L0C; assigning it at offset 0 must
// still be accepted (valid-region footprint equals the full block Numel here).
TEST_F(TAssignWindowTest, full_accumulator_exactly_fills_l0c)
{
    auto *base = NPUMemoryModel::Instance().GetL0CBase();
    ASSERT_NE(base, nullptr);

    AccFull full(kRows, kCols);
    TASSIGN(full, static_cast<uint64_t>(0));
    EXPECT_EQ(reinterpret_cast<char *>(full.data()), base);
}

// Each 16-row strip is TASSIGN'd at byte offset rowStart * kStripRows * sizeof,
// matching the PTOAS-generated offsets (0, 1024, 2048, ... for int32). A strip
// addresses only its 16 valid rows, so every offset stays in-bounds and resolves
// to a pointer inside L0C.
TEST_F(TAssignWindowTest, windowed_strip_offsets_stay_in_bounds)
{
    auto *base = NPUMemoryModel::Instance().GetL0CBase();
    ASSERT_NE(base, nullptr);

    for (int rowStart = 0; rowStart < kRows; rowStart += kStripRows) {
        const std::size_t byteOffset =
            static_cast<std::size_t>(rowStart) * static_cast<std::size_t>(kStripRows) * sizeof(int32_t);

        AccStrip strip;
        TASSIGN(strip, static_cast<uint64_t>(byteOffset));

        auto *ptr = reinterpret_cast<char *>(strip.data());
        EXPECT_EQ(ptr, base + byteOffset) << "strip offset " << byteOffset;
        // The strip's whole valid-region footprint must lie within the L0C buffer.
        const std::size_t footprintBytes =
            (static_cast<std::size_t>(GetTileOffset<AccStrip>(kStripRows - 1, kCols - 1)) + 1) * sizeof(int32_t);
        EXPECT_LE(byteOffset + footprintBytes, kL0CBytes) << "strip offset " << byteOffset;
    }
}

// Writing through a windowed strip and reading back via the raw L0C base confirms
// the resolved pointer addresses real, in-bounds backing storage (no overflow).
TEST_F(TAssignWindowTest, windowed_strip_roundtrips_through_backing_storage)
{
    auto *base = reinterpret_cast<int32_t *>(NPUMemoryModel::Instance().GetL0CBase());
    ASSERT_NE(base, nullptr);

    constexpr std::size_t byteOffset =
        static_cast<std::size_t>(kStripRows) * static_cast<std::size_t>(kStripRows) * sizeof(int32_t); // 1024
    AccStrip strip;
    TASSIGN(strip, static_cast<uint64_t>(byteOffset));

    const std::size_t elemOffset = byteOffset / sizeof(int32_t);
    strip.data()[0] = 0x5a5a5a5a;
    EXPECT_EQ(base[elemOffset], 0x5a5a5a5a);
}
