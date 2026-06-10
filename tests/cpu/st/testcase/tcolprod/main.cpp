#include <pto/pto-inst.hpp>
#include "cpu_tile_test_utils.h"

#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TColProdTest, ProducesProductPerColumn)
{
    using SrcTile = Tile<TileType::Vec, int32_t, 3, 8, BLayout::RowMajor, 3, 4>;
    using DstTile = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 4>;

    SrcTile src;
    DstTile dst;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dst);

    const int values[3][4] = {{2, 3, 4, 5}, {1, -2, 3, -4}, {7, 1, 2, 0}};
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            SetValue(src, r, c, values[r][c]);
        }
    }

    TCOLPROD(dst, src);

    const int expected[4] = {14, -6, 24, 0};
    for (int c = 0; c < dst.GetValidCol(); ++c) {
        ExpectValueEquals(GetValue(dst, 0, c), expected[c]);
    }
}

} // namespace
