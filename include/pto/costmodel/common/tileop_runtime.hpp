/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_COSTMODEL_TILEOP_RUNTIME_HPP
#define PTO_COSTMODEL_TILEOP_RUNTIME_HPP

// Internal support for costmodel wrappers. Does not include pto_instr.hpp.
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "pto/common/pto_instr_impl.hpp"
#include "pto/costmodel/trace.hpp"
#include "pto/costmodel/perf_sim/recorder.hpp"
#include "pto/costmodel/perf_sim/tile_dep_tracker.hpp"
#include "pto/costmodel/perf_sim/latency.hpp"
#include "pto/costmodel/perf_sim/costmodel_provider.hpp"
#include "pto/costmodel/a5/tileop_costmodel_selector.hpp"
namespace perf_sim = ::pto::perf_sim;

namespace pto::mocker {

inline uint64_t GetCurrentPtoInstrCycles()
{
    auto& trace = GetMutableTrace();
    if (trace.executed_pto.empty()) {
        return 0;
    }

    if (trace.active_pto_stack.size() == 1) {
        FlushAllPendingTailsExceptVector();
    }

    if (trace.active_pto_stack.empty()) {
        return trace.executed_pto.back().total_cycles;
    }
    return trace.executed_pto[trace.active_pto_stack.back()].total_cycles;
}

template <typename T>
inline void InjectTileCycles(T& obj)
{
    if constexpr (requires { obj.SetLastCycle(0.0f); }) {
        obj.SetLastCycle(static_cast<float>(GetCurrentPtoInstrCycles()));
    }
}

// Record one PTO instruction to the pipeline simulator. A2/A3 calls this after
// running _IMPL; A5 calls it directly at the TileOp boundary.

// Helper: try to extract dimensions + dtype from a single tile; returns true if successful.
template <typename T>
inline bool TryTileInfo(int& rows, int& cols, std::string& dtype, T&& tile)
{
    if constexpr (requires {
                      tile.GetValidRow();
                      tile.GetValidCol();
                  }) {
        rows = static_cast<int>(tile.GetValidRow());
        cols = static_cast<int>(tile.GetValidCol());
        dtype = perf_sim::TileTraits<std::remove_reference_t<T>>::dtype_str();
        return true;
    } else if constexpr (requires {
                             tile.Rows;
                             tile.Cols;
                         }) {
        rows = tile.Rows;
        cols = tile.Cols;
        dtype = perf_sim::TileTraits<std::remove_reference_t<T>>::dtype_str();
        return true;
    }
    return false;
}

// Recursive: try tiles in order, extract dims+dtype from first one that has them.
template <typename T>
inline void ExtractFirstTileInfo(int& rows, int& cols, std::string& dtype, T&& tile)
{
    TryTileInfo(rows, cols, dtype, std::forward<T>(tile));
}

template <typename T, typename T2, typename... Rest>
inline void ExtractFirstTileInfo(int& rows, int& cols, std::string& dtype, T&& first, T2&& second, Rest&&... rest)
{
    if (!TryTileInfo(rows, cols, dtype, std::forward<T>(first))) {
        ExtractFirstTileInfo(rows, cols, dtype, std::forward<T2>(second), std::forward<Rest>(rest)...);
    }
}

inline void RecordInstrWithOptions(
    const char* opcode, const A5TileOpOptions& a5_options, auto&& first_tile, auto&&... rest_tiles)
{
    // Look up pipeline stage from opcode name and tile types (TLOAD/TSTORE need tile routing)
    perf_sim::PipeStage stage = perf_sim::ResolvePipeStageArgs(opcode, first_tile, rest_tiles...);

    // 1. Track data dependencies (cross-pipe waits + signal event for this instr)
    auto dep = perf_sim::TileDepTracker::TrackByAddr(opcode, stage, first_tile, rest_tiles...);

    // 2. Build InstrRecord
    perf_sim::InstrRecord r;
    r.opcode = opcode;
    r.stage = stage;
    r.signal_event = dep.signal_event;
    for (int i = 0; i < dep.wait_count && i < perf_sim::InstrRecord::MAX_WAIT_EVENTS; ++i) {
        r.wait_events[i] = dep.wait_events[i];
    }
    r.wait_count = dep.wait_count;

    // Tile dimensions + dtype: try each tile argument in order, use first one with info.
    // For TSTORE(dst=GlobalData, src=TileData), this skips GlobalData and uses TileData.
    ExtractFirstTileInfo(r.rows, r.cols, r.dtype, first_tile, rest_tiles...);

    const uint64_t measured_cycles = GetLastPtoInstrCycles();
    uint64_t cycles = measured_cycles;
    uint64_t estimated_cycles = 0;
    bool useEstimatedCycles = (cycles == 0);

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 3510))
    if (stage == perf_sim::PipeStage::Vector) {
        cycles = EstimateA5TileOpCycles(opcode, a5_options, first_tile, rest_tiles...);
        estimated_cycles = cycles;
        useEstimatedCycles = false;
        auto& trace = ::pto::mocker::GetMutableTrace();
        if (!trace.executed_pto.empty()) {
            trace.executed_pto.back().total_cycles = cycles;
        }
    }
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    const bool compareWithEstimate =
        std::string_view(opcode) == "TROWEXPAND" ||
        (std::string_view(opcode) == "TDIVS" && (r.dtype == "int16" || r.dtype == "int32"));
#else
    const bool compareWithEstimate = false;
#endif

    if (useEstimatedCycles || compareWithEstimate) {
        estimated_cycles =
            perf_sim::EstimateInstrCycles(opcode, r.rows, r.cols, r.dtype.empty() ? "unknown" : r.dtype.c_str());
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        useEstimatedCycles = useEstimatedCycles ||
                             (std::string_view(opcode) == "TROWEXPAND" && estimated_cycles > cycles) ||
                             (std::string_view(opcode) == "TDIVS" && (r.dtype == "int16" || r.dtype == "int32"));
#endif
    }
    if (useEstimatedCycles) {
        cycles = estimated_cycles;
        auto& trace = ::pto::mocker::GetMutableTrace();
        if (!trace.executed_pto.empty()) {
            trace.executed_pto.back().total_cycles = cycles;
        }
    }
    r.estimated_cycles = cycles;

    perf_sim::CvSyncRecorder::ApplyPending(opcode, r);
    perf_sim::PtoRecorder::Record(std::move(r));
}

inline void RecordInstr(const char* opcode, auto&& first_tile, auto&&... rest_tiles)
{
    RecordInstrWithOptions(
        opcode, A5TileOpOptions{}, std::forward<decltype(first_tile)>(first_tile),
        std::forward<decltype(rest_tiles)>(rest_tiles)...);
}

// Scalar-stage overload for instructions with no tile arguments
// Scalar-stage ops don't register as data producers (per TileDepTracker design),
// so we skip TrackByAddr and use signal_event=-1.
inline void RecordInstr(const char* opcode)
{
    perf_sim::PipeStage stage = perf_sim::StaticPipeStageLookup(opcode);
    perf_sim::InstrRecord r;
    r.opcode = opcode;
    r.stage = stage;
    r.signal_event = -1;
    r.estimated_cycles = perf_sim::EstimateInstrCycles(opcode, 0, 0, "unknown");
    perf_sim::CvSyncRecorder::ApplyPending(opcode, r);
    perf_sim::PtoRecorder::Record(std::move(r));
}

} // namespace pto::mocker

// RecordInstr variant: first_tile is in the variadic position (MAP_INSTR_IMPL style)
// Accepts any types (tiles or scalars); filters to only tile references before passing to RecordInstr.
template <typename OpcodeStr, typename... Args>
inline void RecordInstrFromFirst(OpcodeStr&& opcode_str, Args&&... args)
{
    // Forward only tile-type arguments to RecordInstr
    ::pto::mocker::RecordInstr(std::forward<OpcodeStr>(opcode_str), std::forward<Args>(args)...);
}

// TPUSH: Record push event to CV ring buffer
// pipe: TPipe<FlagID, DirType, ...> or TMPipe<FlagID, ...> instance
// tile: Tile being pushed
// tile_index: pipe.prod.tileIndex (sequential counter)
template <typename Pipe, typename TileProd>
inline void RecordTPushSync(Pipe& pipe, TileProd& tile, int tile_index)
{
    uint64_t key = perf_sim::MakeCvFifoKey<Pipe>();
    perf_sim::CvSyncRecorder::SetPending(perf_sim::CvSyncKind::Push, key);
    (void)pipe;
    (void)tile;
    (void)tile_index;
}

// TPOP: Record pop event to CV ring buffer
// pipe: TPipe<FlagID, DirType, ...> or TMPipe<FlagID, ...> instance
// tile: Tile being popped
// tile_index: pipe.cons.tileIndex (sequential counter)
template <typename Pipe, typename TileCons>
inline void RecordTPopSync(Pipe& pipe, TileCons& tile, int tile_index)
{
    uint64_t key = perf_sim::MakeCvFifoKey<Pipe>();
    perf_sim::CvSyncRecorder::SetPending(perf_sim::CvSyncKind::Pop, key);
    (void)pipe;
    (void)tile;
    (void)tile_index;
}

namespace pto {

inline uint16_t CostmodelFftsMessage(uint16_t eventId) { return static_cast<uint16_t>(1U + ((eventId & 0xfU) << 8U)); }

template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
PTO_INTERNAL void SYNCALL_IMPL()
{
    ::pto_costmodel_pipe_barrier(PIPE_ALL);
    if constexpr (CoreType == SyncCoreType::AIVOnly) {
        ::ffts_cross_core_sync(PIPE_MTE3, CostmodelFftsMessage(SYNC_AIV_ONLY_ALL));
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 3510))
        ::wait_flag_dev(PIPE_S, SYNC_AIV_ONLY_ALL);
#else
        ::wait_flag_dev(SYNC_AIV_ONLY_ALL);
#endif
    } else if constexpr (CoreType == SyncCoreType::AICOnly) {
        ::ffts_cross_core_sync(PIPE_FIX, CostmodelFftsMessage(SYNC_AIC_FLAG));
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 3510))
        ::wait_flag_dev(PIPE_S, SYNC_AIC_FLAG);
#else
        ::wait_flag_dev(SYNC_AIC_FLAG);
#endif
    } else {
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 3510))
        ::set_intra_block(PIPE_MTE3, SYNC_AIV_FLAG);
        ::wait_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
#else
        ::ffts_cross_core_sync(PIPE_MTE3, CostmodelFftsMessage(SYNC_AIV_FLAG));
        ::wait_flag_dev(SYNC_AIC_AIV_FLAG);
#endif
    }
}

template <SyncCoreType CoreType = SyncCoreType::AIVOnly, typename T>
PTO_INTERNAL void SYNCALL_SOFT_IMPL(T*, int32_t)
{
    SYNCALL_IMPL<CoreType>();
}

template <typename T>
PTO_INTERNAL void SYNCALL_SOFT_AIC_IMPL(T*, int32_t)
{
    SYNCALL_IMPL<SyncCoreType::AICOnly>();
}

template <SyncCoreType CoreType = SyncCoreType::Mix, typename T>
PTO_INTERNAL void SYNCALL_SOFT_MIX_IMPL(T*, int32_t)
{
    SYNCALL_IMPL<CoreType>();
}

} // namespace pto

#endif
