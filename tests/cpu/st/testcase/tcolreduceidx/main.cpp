#include <pto/pto-inst.hpp>
#include "cpu_tile_test_utils.h"

#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TColReduceIdxTest, ArgMaxAndArgMinReturnFirstMatchingIndex)
{
    using SrcTile = Tile<TileType::Vec, int32_t, 4, 8, BLayout::RowMajor, 4, 4>;
    using DstTile = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 4>;
    using TmpTile = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 1>;

    SrcTile src;
    DstTile argmax;
    DstTile argmin;
    TmpTile tmp;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, argmax, argmin, tmp);

    const int values[4][4] = {{1, 8, -2, 3}, {7, 8, 9, 1}, {-5, -7, 9, 5}, {4, -7, 0, 5}};
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            SetValue(src, r, c, values[r][c]);
        }
    }

    TCOLARGMAX(argmax, src, tmp);
    TCOLARGMIN(argmin, src, tmp);

    const int expectedMax[4] = {1, 0, 1, 2};
    const int expectedMin[4] = {2, 2, 0, 1};
    for (int c = 0; c < argmax.GetValidCol(); ++c) {
        ExpectValueEquals(GetValue(argmax, 0, c), expectedMax[c]);
        ExpectValueEquals(GetValue(argmin, 0, c), expectedMin[c]);
    }
}

} // namespace
