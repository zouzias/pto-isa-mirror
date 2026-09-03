/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// GM <-> UB transfers pick their traversal from the tile's own layout, the way the
// hardware selects TLoadGm2ubNd2nd / TLoadGm2ubDn2dn and the matching TStore paths.
// The two agree for ordinary kernels; they part company in the one-row / one-column
// case TSTORE explicitly allows, where a ColMajor [N, 1] tile meets an ND GlobalTensor
// whose row stride is wider than one element. The hardware moves the column as one
// contiguous burst there and never looks at that row stride.

#include <pto/pto-inst.hpp>

#include <gtest/gtest.h>
#include <vector>

using namespace pto;

namespace {

constexpr int kRows = 16;
constexpr int kPitch = 8; // ND row stride of the [N, kPitch] global buffer

class TStoreLayoutTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// TSTORE of a ColMajor column tile through an ND GlobalTensor writes the column
// contiguously, so a later contiguous read of the same base sees every row.
TEST_F(TStoreLayoutTest, store_colmajor_tile_through_nd_tensor_is_contiguous)
{
    using ColTile = Tile<TileType::Vec, float, kRows, 1, BLayout::ColMajor>;
    using GlobalND = GlobalTensor<float, pto::Shape<1, 1, 1, kRows, 1>, pto::Stride<128, 128, 128, kPitch, 1>>;

    std::vector<float> global(static_cast<std::size_t>(kRows) * kPitch, -1.0f);

    ColTile src;
    TASSIGN(src, 0);
    for (int r = 0; r < kRows; ++r) {
        src.data()[GetTileElementOffset<ColTile>(r, 0)] = static_cast<float>(r + 1);
    }

    GlobalND dst(global.data(), pto::Shape<1, 1, 1, kRows, 1>(), pto::Stride<128, 128, 128, kPitch, 1>());
    TSTORE(dst, src);

    for (int r = 0; r < kRows; ++r) {
        EXPECT_FLOAT_EQ(global[r], static_cast<float>(r + 1)) << "contiguous element " << r;
    }
    for (std::size_t i = kRows; i < global.size(); ++i) {
        EXPECT_FLOAT_EQ(global[i], -1.0f) << "untouched element " << i;
    }
}

// The load side of the same round trip: a wide RowMajor tile reads the block back
// densely, so its first kRows elements are the column that was stored.
TEST_F(TStoreLayoutTest, store_then_dense_load_round_trips)
{
    using ColTile = Tile<TileType::Vec, float, kRows, 1, BLayout::ColMajor>;
    using BlockTile = Tile<TileType::Vec, float, kRows, kPitch>;
    using GlobalCol = GlobalTensor<float, pto::Shape<1, 1, 1, kRows, 1>, pto::Stride<128, 128, 128, kPitch, 1>>;
    using GlobalBlock = GlobalTensor<float, pto::Shape<1, 1, 1, kRows, kPitch>, pto::Stride<128, 128, 128, kPitch, 1>>;

    std::vector<float> global(static_cast<std::size_t>(kRows) * kPitch, 0.0f);

    ColTile src;
    TASSIGN(src, 0);
    for (int r = 0; r < kRows; ++r) {
        src.data()[GetTileElementOffset<ColTile>(r, 0)] = static_cast<float>(100 + r);
    }
    GlobalCol colView(global.data(), pto::Shape<1, 1, 1, kRows, 1>(), pto::Stride<128, 128, 128, kPitch, 1>());
    TSTORE(colView, src);

    BlockTile block;
    TASSIGN(block, static_cast<int>(sizeof(float)) * kRows);
    GlobalBlock blockView(global.data(), pto::Shape<1, 1, 1, kRows, kPitch>(), pto::Stride<128, 128, 128, kPitch, 1>());
    TLOAD(block, blockView);

    for (int r = 0; r < kRows; ++r) {
        EXPECT_FLOAT_EQ(block.data()[r], static_cast<float>(100 + r)) << "round trip element " << r;
    }
}

// The mirror carve-out: a single-row RowMajor tile stored through a DN GlobalTensor.
// TStore takes the ND path from TileData::isRowMajor and writes the row as one burst,
// so the DN column stride does not spread it out.
TEST_F(TStoreLayoutTest, store_single_row_tile_through_dn_tensor_is_contiguous)
{
    constexpr int kCols = 16;
    constexpr int kColPitch = 8; // DN column stride of the global buffer
    using RowTile = Tile<TileType::Vec, float, 1, kCols>;
    using GlobalDN =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, kCols>, pto::Stride<128, 128, 128, 1, kColPitch>, pto::Layout::DN>;

    std::vector<float> global(static_cast<std::size_t>(kCols) * kColPitch, -1.0f);

    RowTile src;
    TASSIGN(src, 0);
    for (int c = 0; c < kCols; ++c) {
        src.data()[GetTileElementOffset<RowTile>(0, c)] = static_cast<float>(c + 1);
    }

    GlobalDN dst(global.data(), pto::Shape<1, 1, 1, 1, kCols>(), pto::Stride<128, 128, 128, 1, kColPitch>());
    TSTORE(dst, src);

    for (int c = 0; c < kCols; ++c) {
        EXPECT_FLOAT_EQ(global[c], static_cast<float>(c + 1)) << "contiguous element " << c;
    }
    for (std::size_t i = kCols; i < global.size(); ++i) {
        EXPECT_FLOAT_EQ(global[i], -1.0f) << "untouched element " << i;
    }
}

// A ColMajor tile paired with a DN GlobalTensor keeps the ordinary mapping.
TEST_F(TStoreLayoutTest, matching_dn_layout_is_unchanged)
{
    using ColTile = Tile<TileType::Vec, float, kRows, 1, BLayout::ColMajor>;
    using GlobalDN =
        GlobalTensor<float, pto::Shape<1, 1, 1, kRows, 1>, pto::Stride<kRows, kRows, kRows, 1, kRows>, pto::Layout::DN>;

    std::vector<float> global(kRows, 0.0f);

    ColTile src;
    TASSIGN(src, 0);
    for (int r = 0; r < kRows; ++r) {
        src.data()[GetTileElementOffset<ColTile>(r, 0)] = static_cast<float>(r * 3 + 1);
    }

    GlobalDN dst(global.data(), pto::Shape<1, 1, 1, kRows, 1>(), pto::Stride<kRows, kRows, kRows, 1, kRows>());
    TSTORE(dst, src);

    for (int r = 0; r < kRows; ++r) {
        EXPECT_FLOAT_EQ(global[r], static_cast<float>(r * 3 + 1)) << "dn element " << r;
    }
}

} // namespace
