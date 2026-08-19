#include <pto/pto-inst.hpp>
#include "cpu_tile_test_utils.h"

#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TRowReduceIdxTest, ArgMaxAndArgMinReturnFirstMatchingIndex)
{
    using SrcTile = Tile<TileType::Vec, int32_t, 3, 8, BLayout::RowMajor, 3, 5>;
    using DstTile = Tile<TileType::Vec, int32_t, 3, 8, BLayout::RowMajor, 3, 1>;
    using TmpTile = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 1>;

    SrcTile src;
    DstTile argmax;
    DstTile argmin;
    TmpTile tmp;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, argmax, argmin, tmp);

    const int values[3][5] = {{1, 8, 8, -3, 4}, {7, 2, 9, 9, 1}, {-5, -6, -6, 0, 0}};
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            SetValue(src, r, c, values[r][c]);
        }
    }

    TROWARGMAX(argmax, src, tmp);
    TROWARGMIN(argmin, src, tmp);

    const int expectedMax[3] = {1, 2, 3};
    const int expectedMin[3] = {3, 4, 1};
    for (int r = 0; r < argmax.GetValidRow(); ++r) {
        ExpectValueEquals(GetValue(argmax, r, 0), expectedMax[r]);
        ExpectValueEquals(GetValue(argmin, r, 0), expectedMin[r]);
    }
}

} // namespace
