#include <pto/pto-inst.hpp>
#include "cpu_tile_test_utils.h"

#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TPartMulTest, MultipliesOverlapAndCopiesSrc0OutsideOverlap)
{
    using DstTile = Tile<TileType::Vec, int32_t, 2, 8>;
    using Src0Tile = Tile<TileType::Vec, int32_t, 2, 8>;
    using Src1Tile = Tile<TileType::Vec, int32_t, 2, 8, BLayout::RowMajor, 1, 4>;

    DstTile dst;
    Src0Tile src0;
    Src1Tile src1;
    std::size_t addr = 0;
    AssignTileStorage(addr, dst, src0, src1);

    FillLinear(src0, 1);
    FillLinear(src1, 10);
    TPARTMUL(dst, src0, src1);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            if (r < src1.GetValidRow() && c < src1.GetValidCol()) {
                ExpectValueEquals(GetValue(dst, r, c), GetValue(src0, r, c) * GetValue(src1, r, c));
            } else {
                ExpectValueEquals(GetValue(dst, r, c), GetValue(src0, r, c));
            }
        }
    }
}

TEST(TPartMulTest, UsesSrc1WhenSrc0IsOutOfRange)
{
    using DstTile = Tile<TileType::Vec, int32_t, 2, 8, BLayout::RowMajor, 2, 6>;
    using Src0Tile = Tile<TileType::Vec, int32_t, 2, 8, BLayout::RowMajor, 1, 4>;
    using Src1Tile = Tile<TileType::Vec, int32_t, 2, 8, BLayout::RowMajor, 2, 6>;

    DstTile dst;
    Src0Tile src0;
    Src1Tile src1;
    std::size_t addr = 0;
    AssignTileStorage(addr, dst, src0, src1);

    FillLinear(src0, 1);
    FillLinear(src1, 20);
    TPARTMUL(dst, src0, src1);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            if (r < src0.GetValidRow() && c < src0.GetValidCol()) {
                ExpectValueEquals(GetValue(dst, r, c),
                                  static_cast<int32_t>(GetValue(src0, r, c) * GetValue(src1, r, c)));
            } else {
                ExpectValueEquals(GetValue(dst, r, c), GetValue(src1, r, c));
            }
        }
    }
}

} // namespace
