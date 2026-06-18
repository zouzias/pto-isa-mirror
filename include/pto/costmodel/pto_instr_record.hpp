/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COSTMODEL_INSTR_RECORD_HPP
#define PTO_COSTMODEL_INSTR_RECORD_HPP

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "pto/costmodel/perf_sim/costmodel_provider.hpp"
#include "pto/costmodel/perf_sim/latency.hpp"
#include "pto/costmodel/perf_sim/recorder.hpp"
#include "pto/costmodel/perf_sim/tile_dep_tracker.hpp"
#include "pto/costmodel/trace.hpp"

namespace perf_sim = ::pto::perf_sim;

namespace pto::mocker {

inline uint64_t GetCurrentPtoInstrCycles()
{
    auto &trace = GetMutableTrace();
    if (trace.executed_pto.empty()) {
        return 0;
    }

    if (trace.active_pto_stack.size() == 1) {
        FlushAllPendingTails();
    }

    if (trace.active_pto_stack.empty()) {
        return trace.executed_pto.back().total_cycles;
    }
    return trace.executed_pto[trace.active_pto_stack.back()].total_cycles;
}

template <typename T>
inline void InjectTileCycles(T &obj)
{
    if constexpr (requires { obj.SetLastCycle(0.0f); }) {
        obj.SetLastCycle(static_cast<float>(GetCurrentPtoInstrCycles()));
    }
}

inline void AccumulateLastPtoCycles(uint64_t cycles)
{
    auto &trace = GetMutableTrace();
    if (trace.executed_pto.empty()) {
        return;
    }
    if (trace.active_pto_stack.empty()) {
        trace.executed_pto.back().total_cycles += cycles;
        return;
    }
    trace.executed_pto[trace.active_pto_stack.back()].total_cycles += cycles;
}

template <typename T>
inline bool TryTileInfo(int &rows, int &cols, std::string &dtype, T &&tile)
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

template <typename T>
inline void ExtractFirstTileInfo(int &rows, int &cols, std::string &dtype, T &&tile)
{
    TryTileInfo(rows, cols, dtype, std::forward<T>(tile));
}

template <typename T, typename T2, typename... Rest>
inline void ExtractFirstTileInfo(int &rows, int &cols, std::string &dtype, T &&first, T2 &&second, Rest &&...rest)
{
    if (!TryTileInfo(rows, cols, dtype, std::forward<T>(first))) {
        ExtractFirstTileInfo(rows, cols, dtype, std::forward<T2>(second), std::forward<Rest>(rest)...);
    }
}

inline void RecordInstr(const char *opcode, auto &&firstTile, auto &&...restTiles)
{
    perf_sim::PipeStage stage = perf_sim::ResolvePipeStageArgs(opcode, firstTile, restTiles...);
    auto dep = perf_sim::TileDepTracker::TrackByAddr(opcode, stage, firstTile, restTiles...);

    perf_sim::InstrRecord r;
    r.opcode = opcode;
    r.stage = stage;
    r.signal_event = dep.signal_event;
    for (int i = 0; i < dep.wait_count && i < perf_sim::InstrRecord::MAX_WAIT_EVENTS; ++i) {
        r.wait_events[i] = dep.wait_events[i];
    }
    r.wait_count = dep.wait_count;

    ExtractFirstTileInfo(r.rows, r.cols, r.dtype, firstTile, restTiles...);

    uint64_t cycles = GetLastPtoInstrCycles();
    const bool hasTraceCycles = cycles != 0;
    if (cycles == 0) {
        cycles = perf_sim::EstimateInstrCycles(opcode, r.rows, r.cols, r.dtype.empty() ? "unknown" : r.dtype.c_str());
    }
    r.estimated_cycles = cycles;
    if (!hasTraceCycles) {
        AccumulateLastPtoCycles(cycles);
    }

    perf_sim::CvSyncRecorder::ApplyPending(opcode, r);
    perf_sim::PtoRecorder::Record(std::move(r));
}

inline void RecordInstr(const char *opcode)
{
    perf_sim::PipeStage stage = perf_sim::StaticPipeStageLookup(opcode);
    perf_sim::InstrRecord r;
    r.opcode = opcode;
    r.stage = stage;
    r.signal_event = -1;
    uint64_t cycles = GetLastPtoInstrCycles();
    const bool hasTraceCycles = cycles != 0;
    if (cycles == 0) {
        cycles = perf_sim::EstimateInstrCycles(opcode, 0, 0, "unknown");
    }
    r.estimated_cycles = cycles;
    if (!hasTraceCycles) {
        AccumulateLastPtoCycles(cycles);
    }
    perf_sim::CvSyncRecorder::ApplyPending(opcode, r);
    perf_sim::PtoRecorder::Record(std::move(r));
}

template <typename Pipe, typename TileProd>
inline void RecordTPushSync(Pipe &pipe, TileProd &tile, int tileIndex)
{
    uint64_t key = perf_sim::MakeCvFifoKey<Pipe>();
    perf_sim::CvSyncRecorder::SetPending(perf_sim::CvSyncKind::Push, key);
    (void)pipe;
    (void)tile;
    (void)tileIndex;
}

template <typename Pipe, typename TileCons>
inline void RecordTPopSync(Pipe &pipe, TileCons &tile, int tileIndex)
{
    uint64_t key = perf_sim::MakeCvFifoKey<Pipe>();
    perf_sim::CvSyncRecorder::SetPending(perf_sim::CvSyncKind::Pop, key);
    (void)pipe;
    (void)tile;
    (void)tileIndex;
}

inline void RecordPtoInstr(const char *opcode)
{
    {
        PtoInstrScope scope(opcode);
    }
    RecordInstr(opcode);
}

template <typename First, typename... Rest>
inline void RecordPtoInstr(const char *opcode, First &&first, Rest &&...rest)
{
    {
        PtoInstrScope scope(opcode);
    }
    InjectTileCycles(first);
    RecordInstr(opcode, std::forward<First>(first), std::forward<Rest>(rest)...);
}

template <typename First, typename... Rest>
inline void RecordCompletedPtoInstr(const char *opcode, First &&first, Rest &&...rest)
{
    InjectTileCycles(first);
    RecordInstr(opcode, std::forward<First>(first), std::forward<Rest>(rest)...);
}

} // namespace pto::mocker

#endif // PTO_COSTMODEL_INSTR_RECORD_HPP
