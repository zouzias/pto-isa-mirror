/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>

// Prediction may reuse recorder TileTraits, but must not depend on the provider.
#include <pto/costmodel/a5/vf_costmodel.hpp>
#if defined(PTO_PERF_SIM_COSTMODEL_PROVIDER_HPP)
#error "A5 prediction must not include costmodel_provider.hpp"
#endif

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#include "a5_host_tileop_check.hpp"

using namespace pto;

namespace {

template <typename T>
concept HasLegacyRem = requires(T& dst, T& src0, T& src1) { pto::TREM(dst, src0, src1); };

template <typename T>
concept HasLegacyRems = requires(T& dst, T& src, typename T::DType scalar) { pto::TREMS(dst, src, scalar); };

using RemInterfaceTile = Tile<TileType::Vec, float, 1, 64>;
static_assert(!HasLegacyRem<RemInterfaceTile>);
static_assert(!HasLegacyRems<RemInterfaceTile>);
static_assert(requires(RemInterfaceTile& dst, RemInterfaceTile& src, RemInterfaceTile& tmp) {
    pto::TREM(dst, src, src, tmp);
    pto::TREMS(dst, src, 1.0f, tmp);
});

template <typename T, int rows, int cols>
void runTAdd()
{
    using TileData = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(rows, cols);
    TileData src1Tile(rows, cols);
    TileData dstTile(rows, cols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);
    TASSIGN(dstTile, 0x8000);

    ::pto::mocker::ResetTrace();
    TADD(dstTile, src0Tile, src1Tile);

    constexpr uint64_t repeat = (static_cast<uint64_t>(rows) * cols + 63) / 64;
    constexpr uint64_t expectedCycles = [] {
        if constexpr (rows == 1) {
            return static_cast<uint64_t>(1.2469671 * repeat + 55.073657 + 0.5);
        }
        constexpr uint64_t innerRepeat = (static_cast<uint64_t>(cols) + 63) / 64;
        return static_cast<uint64_t>((1.8749443 * innerRepeat + 2.8762808) * rows + 44.344619 + 0.5);
    }();
    pto::test::a5::ExpectSupportedVfTileOp("TADD", expectedCycles);
}

} // namespace

TEST(TAdd, float_1x64) { runTAdd<float, 1, 64>(); }

TEST(TAdd, float_1x512) { runTAdd<float, 1, 512>(); }

TEST(TAdd, float_1x1024) { runTAdd<float, 1, 1024>(); }

TEST(TAdd, float_1x2048) { runTAdd<float, 1, 2048>(); }

TEST(TAdd, float_1x4096) { runTAdd<float, 1, 4096>(); }

TEST(TAdd, float_1x6144) { runTAdd<float, 1, 6144>(); }

// Dynamic ValidCol is not a compile-time contiguous tile, even when its
// runtime valid width happens to equal Cols. The NPU BinaryInstr therefore
// selects its 2D implementation path.
TEST(TAdd, float_dynamic_10x672_uses_2d_path) { runTAdd<float, 10, 672>(); }

TEST(A5Integration, assign_preserves_address)
{
    RemInterfaceTile tile;
    TASSIGN(tile, 0x4000);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(tile.data()), 0x4000U);
}

TEST(A5Integration, synchronization_wrappers_are_empty_stubs)
{
    using Workspace = GlobalTensor<int32_t, Shape<1, 1, 1, 1, 1>, Stride<1, 1, 1, 1, 1>>;
    Workspace workspace;
    pto::mocker::ResetTrace();
    pto::perf_sim::PtoRecorder::Clear();
    pto::perf_sim::SyncRecorder::Clear();
    TSYNC<Op::TADD>();
    TSYNC();
    SYNCALL<>();
    SYNCALL<SyncCoreType::AICOnly>();
    SYNCALL<SyncCoreType::Mix>();
    SYNCALL<SyncAllMode::Hard>(workspace, 2);
    SYNCALL<SyncAllMode::Soft>(workspace, 2);
    SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(workspace, 2);
    SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(workspace, 2);
    EXPECT_TRUE(pto::mocker::GetTrace().executed_pto.empty());
    EXPECT_TRUE(pto::perf_sim::PtoRecorder::Get().empty());
    EXPECT_TRUE(pto::perf_sim::SyncRecorder::Get().empty());
}

TEST(A5Integration, template_values_survive_type_prefixes)
{
    using namespace pto::mocker;
    EXPECT_EQ(A5OptionsFromTemplate<RecipAlgorithm::HIGH_PRECISION>().op_params, "high_precision");
    const auto oneType = A5OptionsFromTemplate<RemInterfaceTile, SqrtAlgorithm::HIGH_PRECISION>();
    const auto twoTypes = A5OptionsFromTemplate<RemInterfaceTile, RemInterfaceTile, DivAlgorithm::HIGH_PRECISION>();
    const auto threeTypes =
        A5OptionsFromTemplate<RemInterfaceTile, RemInterfaceTile, RemInterfaceTile, ExpAlgorithm::HIGH_PRECISION>();
    const auto fourTypes = A5OptionsFromTemplate<
        RemInterfaceTile, RemInterfaceTile, RemInterfaceTile, RemInterfaceTile, RsqrtAlgorithm::HIGH_PRECISION>();
    EXPECT_EQ(oneType.op_params, "high_precision");
    EXPECT_EQ(twoTypes.op_params, "high_precision");
    EXPECT_EQ(threeTypes.op_params, "high_precision");
    EXPECT_EQ(fourTypes.op_params, "high_precision");
}

TEST(A5Integration, binary_selects_once_from_tile_metadata)
{
    using namespace pto::mocker;
    using StaticTile = Tile<TileType::Vec, float, 4, 64>;
    StaticTile dst, src0, src1;
    auto metadata = BuildA5TileOpMetadata(dst, src0, src1);
    A5VfExecutionPath path;
    ASSERT_TRUE(SelectA5BinaryCostmodelPath("TADD", {}, metadata, path));
    EXPECT_EQ(path.implementation, A5VfTemplate::OneDPostUpdate);
    metadata.vf_impl_kind = VFImplKind::VFIMPL_2D_NO_POST_UPDATE;
    ASSERT_TRUE(SelectA5BinaryCostmodelPath("TADD", {}, metadata, path));
    EXPECT_EQ(path.implementation, A5VfTemplate::TwoDNoPostUpdate);
    metadata.static_full_cols_mask = 0;
    metadata.vf_impl_kind = VFImplKind::VFIMPL_1D_POST_UPDATE;
    ASSERT_TRUE(SelectA5BinaryCostmodelPath("TADD", {}, metadata, path));
    EXPECT_EQ(path.implementation, A5VfTemplate::TwoDPostUpdate);
}
