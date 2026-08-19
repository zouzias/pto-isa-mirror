#include <pto/pto-inst.hpp>
#include "cpu_tile_test_utils.h"

#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TGetScaleAddrTest, AliasesSourceStorage)
{
    using TileData = Tile<TileType::Vec, float, 2, 8>;

    TileData src;
    TileData dst;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dst);

    FillLinear(src, 1.0f);
    TGET_SCALE_ADDR(dst, src);

    ASSERT_EQ(dst.data(), src.data());
    src.data()[3] = 42.0f;
    ExpectValueEquals(dst.data()[3], 42.0f);
}

} // namespace
