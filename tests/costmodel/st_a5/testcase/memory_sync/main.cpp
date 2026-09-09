/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>

#include "pto/costmodel/perf_sim/recorder.hpp"
#include "pto/costmodel/trace.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace pto;

namespace {

namespace perf = ::pto::perf_sim;
namespace vf = ::pto::mocker::vf;

void ResetCostmodelRecords()
{
    ::pto::mocker::ResetTrace();
    perf::PtoRecorder::Clear();
    perf::SyncRecorder::Clear();
}

TEST(A5CostmodelMemory, GmTransfersOnlyEnterPerfSim)
{
    using TileData = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1>;
    using GlobalData = GlobalTensor<float, Shape<1, 1, 1, 1, 64>, Stride<64, 64, 64, 64, 1>>;

    TileData tile(1, 64);
    TASSIGN(tile, 0x0);
    GlobalData global(reinterpret_cast<float*>(0x10000));

    ResetCostmodelRecords();
    TLOAD(tile, global);
    ASSERT_EQ(perf::PtoRecorder::Get().size(), 1U);
    EXPECT_EQ(perf::PtoRecorder::Get().back().opcode, "TLOAD");
    EXPECT_EQ(perf::PtoRecorder::Get().back().stage, perf::PipeStage::MTE2_AIV);
    EXPECT_GT(perf::PtoRecorder::Get().back().estimated_cycles, 0U);
    ASSERT_EQ(::pto::mocker::GetTrace().executed_pto.size(), 1U);
    EXPECT_TRUE(::pto::mocker::GetTrace().executed_pto.back().vf_infos.empty());
    EXPECT_EQ(
        ::pto::mocker::GetTrace().executed_pto.back().total_cycles, perf::PtoRecorder::Get().back().estimated_cycles);

    ResetCostmodelRecords();
    TSTORE(global, tile);
    ASSERT_EQ(perf::PtoRecorder::Get().size(), 1U);
    EXPECT_EQ(perf::PtoRecorder::Get().back().opcode, "TSTORE");
    EXPECT_EQ(perf::PtoRecorder::Get().back().stage, perf::PipeStage::MTE3);
    EXPECT_GT(perf::PtoRecorder::Get().back().estimated_cycles, 0U);
    ASSERT_EQ(::pto::mocker::GetTrace().executed_pto.size(), 1U);
    EXPECT_TRUE(::pto::mocker::GetTrace().executed_pto.back().vf_infos.empty());
    EXPECT_EQ(
        ::pto::mocker::GetTrace().executed_pto.back().total_cycles, perf::PtoRecorder::Get().back().estimated_cycles);
}

TEST(A5CostmodelMemory, OnChipTransfersAndPaddingAreRecordedOutsideVfSim)
{
    uint8_t ub[64]{};

    ResetCostmodelRecords();
    ::pto::mocker::BeginPtoInstr("UB_COPY");
    copy_ubuf_to_ubuf(ub, ub, 1, 1, 0, 0);
    set_mov_pad_val(0);
    ::pto::mocker::EndPtoInstr();

    const auto& trace = ::pto::mocker::GetTrace();
    ASSERT_EQ(trace.executed_pto.size(), 1U);
    EXPECT_TRUE(trace.executed_pto.back().vf_infos.empty());
    EXPECT_GT(trace.executed_pto.back().total_cycles, 0U);
}

TEST(A5CostmodelCube, MatmulEntersCubeTraceInsteadOfVfSim)
{
    using LeftTile = TileLeft<half, 16, 16, 16, 16>;
    using RightTile = TileRight<half, 16, 16, 16, 16>;
    using AccTile = TileAcc<float, 16, 16, 16, 16>;

    LeftTile left;
    RightTile right;
    AccTile acc;
    TASSIGN(left, 0x0);
    TASSIGN(right, 0x10000);
    TASSIGN(acc, 0x20000);

    ResetCostmodelRecords();
    TMATMUL(acc, left, right);

    ASSERT_EQ(perf::PtoRecorder::Get().size(), 1U);
    EXPECT_EQ(perf::PtoRecorder::Get().back().opcode, "TMATMUL");
    EXPECT_EQ(perf::PtoRecorder::Get().back().stage, perf::PipeStage::Matrix);
    EXPECT_GT(perf::PtoRecorder::Get().back().estimated_cycles, 0U);
    ASSERT_EQ(::pto::mocker::GetTrace().executed_pto.size(), 1U);
    EXPECT_TRUE(::pto::mocker::GetTrace().executed_pto.back().vf_infos.empty());
}

TEST(A5CostmodelSync, VfMemBarOnlyEntersVfInfo)
{
    ResetCostmodelRecords();
    ::pto::mocker::BeginPtoInstr("VF_MEMBAR");
    {
        __VEC_SCOPE__ { mem_bar(VST_VLD); }
    }
    ::pto::mocker::EndPtoInstr();

    EXPECT_TRUE(perf::SyncRecorder::Get().empty());
    const auto& trace = ::pto::mocker::GetTrace();
    ASSERT_EQ(trace.executed_pto.size(), 1U);
    ASSERT_EQ(trace.executed_pto.back().vf_infos.size(), 1U);
    const auto& tree = trace.executed_pto.back().vf_infos.front().tree;
    ASSERT_EQ(tree.size(), 1U);
    ASSERT_TRUE(vf::IsMemBar(tree.front()));
    EXPECT_EQ(vf::AsMemBar(tree.front()).name, "VST_VLD");
    EXPECT_GT(trace.executed_pto.back().total_cycles, 0U);
}

TEST(A5CostmodelSync, PipeAndEventSyncOnlyEnterPerfSim)
{
    ResetCostmodelRecords();
    pipe_barrier(PIPE_MTE3);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);

    EXPECT_TRUE(::pto::mocker::GetTrace().executed_pto.empty());
    const auto& sync = perf::SyncRecorder::Get();
    ASSERT_EQ(sync.size(), 3U);
    EXPECT_EQ(sync[0].kind, perf::SyncKind::Barrier);
    EXPECT_EQ(sync[0].src_pipe, PIPE_MTE3);
    EXPECT_EQ(sync[1].kind, perf::SyncKind::Signal);
    EXPECT_EQ(sync[1].event_id, EVENT_ID2);
    EXPECT_EQ(sync[2].kind, perf::SyncKind::Wait);
    EXPECT_EQ(sync[2].event_id, EVENT_ID2);
}

TEST(A5CostmodelSync, SyncAllAvoidsDeviceAtomicPath)
{
    ResetCostmodelRecords();
    SYNCALL<SyncCoreType::AIVOnly>();

    EXPECT_TRUE(::pto::mocker::GetTrace().executed_pto.empty());
    const auto& sync = perf::SyncRecorder::Get();
    ASSERT_EQ(sync.size(), 3U);
    EXPECT_EQ(sync[0].kind, perf::SyncKind::Barrier);
    EXPECT_EQ(sync[0].src_pipe, PIPE_ALL);
    EXPECT_EQ(sync[1].kind, perf::SyncKind::Signal);
    EXPECT_TRUE(sync[1].cross_core);
    EXPECT_EQ(sync[2].kind, perf::SyncKind::Wait);
    EXPECT_TRUE(sync[2].cross_core);
}

} // namespace
