/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COSTMODEL_INSTR_IMPL_HPP
#define PTO_COSTMODEL_INSTR_IMPL_HPP

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "pto/costmodel/perf_sim/costmodel_provider.hpp"
#include "pto/costmodel/perf_sim/latency.hpp"
#include "pto/costmodel/perf_sim/recorder.hpp"
#include "pto/costmodel/perf_sim/tile_dep_tracker.hpp"
#include "pto/costmodel/trace.hpp"

#define TADD_IMPL PTO_COSTMODEL_A2A3_TADD_IMPL
#define TABS_IMPL PTO_COSTMODEL_A2A3_TABS_IMPL
#define TAND_IMPL PTO_COSTMODEL_A2A3_TAND_IMPL
#define TOR_IMPL PTO_COSTMODEL_A2A3_TOR_IMPL
#define TSUB_IMPL PTO_COSTMODEL_A2A3_TSUB_IMPL
#define TSUBVIEW_IMPL PTO_COSTMODEL_A2A3_TSUBVIEW_IMPL
#define TMUL_IMPL PTO_COSTMODEL_A2A3_TMUL_IMPL
#define TMIN_IMPL PTO_COSTMODEL_A2A3_TMIN_IMPL
#define TMAX_IMPL PTO_COSTMODEL_A2A3_TMAX_IMPL
#define TEXPANDS_IMPL PTO_COSTMODEL_A2A3_TEXPANDS_IMPL
#define TLOAD_IMPL PTO_COSTMODEL_A2A3_TLOAD_IMPL
#define TPREFETCH_IMPL PTO_COSTMODEL_A2A3_TPREFETCH_IMPL
#define TCMPS_IMPL PTO_COSTMODEL_A2A3_TCMPS_IMPL
#define TCMP_IMPL PTO_COSTMODEL_A2A3_TCMP_IMPL
#define TCONCAT_IMPL PTO_COSTMODEL_A2A3_TCONCAT_IMPL
#define TDIV_IMPL PTO_COSTMODEL_A2A3_TDIV_IMPL
#define TSHL_IMPL PTO_COSTMODEL_A2A3_TSHL_IMPL
#define TSHR_IMPL PTO_COSTMODEL_A2A3_TSHR_IMPL
#define TXOR_IMPL PTO_COSTMODEL_A2A3_TXOR_IMPL
#define TLOG_IMPL PTO_COSTMODEL_A2A3_TLOG_IMPL
#define TDIVS_IMPL PTO_COSTMODEL_A2A3_TDIVS_IMPL
#define TPRELU_IMPL PTO_COSTMODEL_A2A3_TPRELU_IMPL
#define TADDC_IMPL PTO_COSTMODEL_A2A3_TADDC_IMPL
#define TSUBC_IMPL PTO_COSTMODEL_A2A3_TSUBC_IMPL
#define TMATMUL_IMPL PTO_COSTMODEL_A2A3_TMATMUL_IMPL
#define TMATMUL_ACC_IMPL PTO_COSTMODEL_A2A3_TMATMUL_ACC_IMPL
#define TMATMUL_BIAS_IMPL PTO_COSTMODEL_A2A3_TMATMUL_BIAS_IMPL
#define TGEMV_IMPL PTO_COSTMODEL_A2A3_TGEMV_IMPL
#define TGEMV_ACC_IMPL PTO_COSTMODEL_A2A3_TGEMV_ACC_IMPL
#define TGEMV_BIAS_IMPL PTO_COSTMODEL_A2A3_TGEMV_BIAS_IMPL
#define TMRGSORT_IMPL PTO_COSTMODEL_A2A3_TMRGSORT_IMPL
#define TEXTRACT_IMPL PTO_COSTMODEL_A2A3_TEXTRACT_IMPL
#define TIMG2COL_IMPL PTO_COSTMODEL_A2A3_TIMG2COL_IMPL
#define SETFMATRIX_IMPL PTO_COSTMODEL_A2A3_SETFMATRIX_IMPL
#define SET_IMG2COL_RPT_IMPL PTO_COSTMODEL_A2A3_SET_IMG2COL_RPT_IMPL
#define SET_IMG2COL_PADDING_IMPL PTO_COSTMODEL_A2A3_SET_IMG2COL_PADDING_IMPL
#define TINSERT_IMPL PTO_COSTMODEL_A2A3_TINSERT_IMPL
#define TSORT32_IMPL PTO_COSTMODEL_A2A3_TSORT32_IMPL
#define TGATHER_IMPL PTO_COSTMODEL_A2A3_TGATHER_IMPL
#define TCI_IMPL PTO_COSTMODEL_A2A3_TCI_IMPL
#define TTRI_IMPL PTO_COSTMODEL_A2A3_TTRI_IMPL
#define TCVT_IMPL PTO_COSTMODEL_A2A3_TCVT_IMPL
#define TMOV_IMPL PTO_COSTMODEL_A2A3_TMOV_IMPL
#define TROWSUM_IMPL PTO_COSTMODEL_A2A3_TROWSUM_IMPL
#define TCOLSUM_IMPL PTO_COSTMODEL_A2A3_TCOLSUM_IMPL
#define TCOLMAX_IMPL PTO_COSTMODEL_A2A3_TCOLMAX_IMPL
#define TROWMAX_IMPL PTO_COSTMODEL_A2A3_TROWMAX_IMPL
#define TROWMIN_IMPL PTO_COSTMODEL_A2A3_TROWMIN_IMPL
#define TSELS_IMPL PTO_COSTMODEL_A2A3_TSELS_IMPL
#define TSEL_IMPL PTO_COSTMODEL_A2A3_TSEL_IMPL
#define TTRANS_IMPL PTO_COSTMODEL_A2A3_TTRANS_IMPL
#define TMINS_IMPL PTO_COSTMODEL_A2A3_TMINS_IMPL
#define TROWEXPAND_IMPL PTO_COSTMODEL_A2A3_TROWEXPAND_IMPL
#define TROWEXPANDDIV_IMPL PTO_COSTMODEL_A2A3_TROWEXPANDDIV_IMPL
#define TROWEXPANDMUL_IMPL PTO_COSTMODEL_A2A3_TROWEXPANDMUL_IMPL
#define TROWEXPANDSUB_IMPL PTO_COSTMODEL_A2A3_TROWEXPANDSUB_IMPL
#define TROWEXPANDADD_IMPL PTO_COSTMODEL_A2A3_TROWEXPANDADD_IMPL
#define TROWEXPANDMAX_IMPL PTO_COSTMODEL_A2A3_TROWEXPANDMAX_IMPL
#define TROWEXPANDMIN_IMPL PTO_COSTMODEL_A2A3_TROWEXPANDMIN_IMPL
#define TROWEXPANDEXPDIF_IMPL PTO_COSTMODEL_A2A3_TROWEXPANDEXPDIF_IMPL
#define TRSQRT_IMPL PTO_COSTMODEL_A2A3_TRSQRT_IMPL
#define TSQRT_IMPL PTO_COSTMODEL_A2A3_TSQRT_IMPL
#define TEXP_IMPL PTO_COSTMODEL_A2A3_TEXP_IMPL
#define TNOT_IMPL PTO_COSTMODEL_A2A3_TNOT_IMPL
#define TRELU_IMPL PTO_COSTMODEL_A2A3_TRELU_IMPL
#define TGATHERB_IMPL PTO_COSTMODEL_A2A3_TGATHERB_IMPL
#define TADDS_IMPL PTO_COSTMODEL_A2A3_TADDS_IMPL
#define TAXPY_IMPL PTO_COSTMODEL_A2A3_TAXPY_IMPL
#define TSUBS_IMPL PTO_COSTMODEL_A2A3_TSUBS_IMPL
#define TMULS_IMPL PTO_COSTMODEL_A2A3_TMULS_IMPL
#define TMAXS_IMPL PTO_COSTMODEL_A2A3_TMAXS_IMPL
#define TLRELU_IMPL PTO_COSTMODEL_A2A3_TLRELU_IMPL
#define TCOLMIN_IMPL PTO_COSTMODEL_A2A3_TCOLMIN_IMPL
#define TSCATTER_IMPL PTO_COSTMODEL_A2A3_TSCATTER_IMPL
#define TCOLEXPAND_IMPL PTO_COSTMODEL_A2A3_TCOLEXPAND_IMPL
#define TNEG_IMPL PTO_COSTMODEL_A2A3_TNEG_IMPL
#define TCOLEXPANDDIV_IMPL PTO_COSTMODEL_A2A3_TCOLEXPANDDIV_IMPL
#define TCOLEXPANDMUL_IMPL PTO_COSTMODEL_A2A3_TCOLEXPANDMUL_IMPL
#define TCOLEXPANDADD_IMPL PTO_COSTMODEL_A2A3_TCOLEXPANDADD_IMPL
#define TCOLEXPANDMAX_IMPL PTO_COSTMODEL_A2A3_TCOLEXPANDMAX_IMPL
#define TCOLEXPANDMIN_IMPL PTO_COSTMODEL_A2A3_TCOLEXPANDMIN_IMPL
#define TCOLEXPANDSUB_IMPL PTO_COSTMODEL_A2A3_TCOLEXPANDSUB_IMPL
#define TCOLEXPANDEXPDIF_IMPL PTO_COSTMODEL_A2A3_TCOLEXPANDEXPDIF_IMPL
#define TDEQUANT_IMPL PTO_COSTMODEL_A2A3_TDEQUANT_IMPL
#define TREM_IMPL PTO_COSTMODEL_A2A3_TREM_IMPL
#define TFMOD_IMPL PTO_COSTMODEL_A2A3_TFMOD_IMPL
#define TSTORE_IMPL PTO_COSTMODEL_A2A3_TSTORE_IMPL
#define TPUSH_IMPL PTO_COSTMODEL_A2A3_TPUSH_IMPL
#define TPOP_IMPL PTO_COSTMODEL_A2A3_TPOP_IMPL
#define TFREE_IMPL PTO_COSTMODEL_A2A3_TFREE_IMPL
#define TALLOC_IMPL PTO_COSTMODEL_A2A3_TALLOC_IMPL
#include "pto/npu/a2a3/TAdd.hpp"
#include "pto/npu/a2a3/TUnaryOp.hpp"
#include "pto/npu/a2a3/TAnd.hpp"
#include "pto/npu/a2a3/TOr.hpp"
#include "pto/npu/a2a3/TSub.hpp"
#include "pto/npu/a2a3/TSubView.hpp"
#include "pto/npu/a2a3/TMul.hpp"
#include "pto/npu/a2a3/TMin.hpp"
#include "pto/npu/a2a3/TMax.hpp"
#include "pto/npu/a2a3/TExpandS.hpp"
#include "pto/npu/a2a3/TLoad.hpp"
#include "pto/npu/a2a3/TPrefetch.hpp"
#include "pto/npu/a2a3/TCmps.hpp"
#include "pto/npu/a2a3/TCmp.hpp"
#include "pto/npu/a2a3/TConcat.hpp"
#include "pto/npu/a2a3/TDiv.hpp"
#include "pto/npu/a2a3/TShl.hpp"
#include "pto/npu/a2a3/TShr.hpp"
#include "pto/npu/a2a3/TXor.hpp"
#include "pto/npu/a2a3/TDivS.hpp"
#include "pto/npu/a2a3/TPrelu.hpp"
#include "pto/npu/a2a3/TFusedMulAdd.hpp"
#include "pto/npu/a2a3/TMatmul.hpp"
#include "pto/npu/a2a3/TMrgSort.hpp"
#include "pto/npu/a2a3/TExtract.hpp"
#include "pto/npu/a2a3/TImg2col.hpp"
#include "pto/npu/a2a3/SetFmatrix.hpp"
#include "pto/npu/a2a3/SetImg2colRpt.hpp"
#include "pto/npu/a2a3/SetImg2colPadding.hpp"
#include "pto/npu/a2a3/TInsert.hpp"
#include "pto/npu/a2a3/TSort32.hpp"
#include "pto/npu/a2a3/TGather.hpp"
#include "pto/npu/a2a3/TCI.hpp"
#include "pto/npu/a2a3/TTri.hpp"
#include "pto/npu/a2a3/TCvt.hpp"
#include "pto/npu/a2a3/TMov.hpp"
#include "pto/npu/a2a3/TRowSum.hpp"
#include "pto/npu/a2a3/TColSum.hpp"
#include "pto/npu/a2a3/TColMax.hpp"
#include "pto/npu/a2a3/TRowMax.hpp"
#include "pto/npu/a2a3/TRowMin.hpp"
#include "pto/npu/a2a3/TSels.hpp"
#include "pto/npu/a2a3/TSel.hpp"
#include "pto/npu/a2a3/TTrans.hpp"
#include "pto/npu/a2a3/TMins.hpp"
#include "pto/npu/a2a3/TRowExpand.hpp"
#include "pto/npu/a2a3/TRowExpandDiv.hpp"
#include "pto/npu/a2a3/TRowExpandMul.hpp"
#include "pto/npu/a2a3/TRowExpandSub.hpp"
#include "pto/npu/a2a3/TRowExpandAdd.hpp"
#include "pto/npu/a2a3/TRowExpandMax.hpp"
#include "pto/npu/a2a3/TRowExpandMin.hpp"
#include "pto/npu/a2a3/TRowExpandExpdif.hpp"
#include "pto/npu/a2a3/TGatherB.hpp"
#include "pto/npu/a2a3/TAddS.hpp"
#include "pto/npu/a2a3/TAxpy.hpp"
#include "pto/npu/a2a3/TSubS.hpp"
#include "pto/npu/a2a3/TMulS.hpp"
#include "pto/npu/a2a3/TMaxS.hpp"
#include "pto/npu/a2a3/TLRelu.hpp"
#include "pto/npu/a2a3/TColMin.hpp"
#include "pto/npu/a2a3/TScatter.hpp"
#include "pto/npu/a2a3/TColExpand.hpp"
#include "pto/npu/a2a3/TColExpandDiv.hpp"
#include "pto/npu/a2a3/TColExpandMul.hpp"
#include "pto/npu/a2a3/TColExpandAdd.hpp"
#include "pto/npu/a2a3/TColExpandMax.hpp"
#include "pto/npu/a2a3/TColExpandMin.hpp"
#include "pto/npu/a2a3/TColExpandSub.hpp"
#include "pto/npu/a2a3/TColExpandExpdif.hpp"
#include "pto/npu/a2a3/TDequant.hpp"
#include "pto/npu/a2a3/TRem.hpp"
#include "pto/npu/a2a3/TFmod.hpp"
#include "pto/npu/a2a3/TStore.hpp"
#include "pto/npu/a2a3/TPush.hpp"
#include "pto/npu/a2a3/TPop.hpp"
#include "pto/npu/a2a3/TFree.hpp"
#include "pto/npu/a2a3/TAlloc.hpp"
#undef TADD_IMPL
#undef TABS_IMPL
#undef TAND_IMPL
#undef TOR_IMPL
#undef TSUB_IMPL
#undef TSUBVIEW_IMPL
#undef TMUL_IMPL
#undef TMIN_IMPL
#undef TMAX_IMPL
#undef TEXPANDS_IMPL
#undef TLOAD_IMPL
#undef TPREFETCH_IMPL
#undef TCMPS_IMPL
#undef TCMP_IMPL
#undef TCONCAT_IMPL
#undef TDIV_IMPL
#undef TSHL_IMPL
#undef TSHR_IMPL
#undef TXOR_IMPL
#undef TLOG_IMPL
#undef TDIVS_IMPL
#undef TPRELU_IMPL
#undef TADDC_IMPL
#undef TSUBC_IMPL
#undef TMATMUL_IMPL
#undef TMATMUL_ACC_IMPL
#undef TMATMUL_BIAS_IMPL
#undef TGEMV_IMPL
#undef TGEMV_ACC_IMPL
#undef TGEMV_BIAS_IMPL
#undef TMRGSORT_IMPL
#undef TEXTRACT_IMPL
#undef TIMG2COL_IMPL
#undef SETFMATRIX_IMPL
#undef SET_IMG2COL_RPT_IMPL
#undef SET_IMG2COL_PADDING_IMPL
#undef TINSERT_IMPL
#undef TSORT32_IMPL
#undef TGATHER_IMPL
#undef TCI_IMPL
#undef TTRI_IMPL
#undef TCVT_IMPL
#undef TMOV_IMPL
#undef TROWSUM_IMPL
#undef TCOLSUM_IMPL
#undef TCOLMAX_IMPL
#undef TROWMAX_IMPL
#undef TROWMIN_IMPL
#undef TSELS_IMPL
#undef TSEL_IMPL
#undef TTRANS_IMPL
#undef TMINS_IMPL
#undef TROWEXPAND_IMPL
#undef TROWEXPANDDIV_IMPL
#undef TROWEXPANDMUL_IMPL
#undef TROWEXPANDSUB_IMPL
#undef TROWEXPANDADD_IMPL
#undef TROWEXPANDMAX_IMPL
#undef TROWEXPANDMIN_IMPL
#undef TROWEXPANDEXPDIF_IMPL
#undef TRSQRT_IMPL
#undef TSQRT_IMPL
#undef TEXP_IMPL
#undef TNOT_IMPL
#undef TRELU_IMPL
#undef TGATHERB_IMPL
#undef TADDS_IMPL
#undef TAXPY_IMPL
#undef TSUBS_IMPL
#undef TMULS_IMPL
#undef TMAXS_IMPL
#undef TLRELU_IMPL
#undef TCOLMIN_IMPL
#undef TSCATTER_IMPL
#undef TCOLEXPAND_IMPL
#undef TNEG_IMPL
#undef TCOLEXPANDDIV_IMPL
#undef TCOLEXPANDMUL_IMPL
#undef TCOLEXPANDADD_IMPL
#undef TCOLEXPANDMAX_IMPL
#undef TCOLEXPANDMIN_IMPL
#undef TCOLEXPANDSUB_IMPL
#undef TCOLEXPANDEXPDIF_IMPL
#undef TDEQUANT_IMPL
#undef TREM_IMPL
#undef TFMOD_IMPL
#undef TSTORE_IMPL
#undef TPUSH_IMPL
#undef TPOP_IMPL
#undef TFREE_IMPL
#undef TALLOC_IMPL

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

namespace pto {

#define PTO_COSTMODEL_DEFINE_SIMPLE(API)                         \
    template <typename... Args>                                  \
    PTO_INTERNAL void API##_IMPL(Args &&...args)                 \
    {                                                            \
        ::pto::mocker::RecordPtoInstr(#API, std::forward<Args>(args)...); \
    }

#define PTO_COSTMODEL_DEFINE_A2A3(API)                                             \
    template <typename... Args>                                                    \
    PTO_INTERNAL void API##_IMPL(Args &&...args)                                   \
    {                                                                              \
        ::pto::mocker::PtoInstrScope scope(#API);                                  \
        PTO_COSTMODEL_A2A3_##API##_IMPL(std::forward<Args>(args)...);              \
        ::pto::mocker::RecordCompletedPtoInstr(#API, std::forward<Args>(args)...); \
    }

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::mocker::PtoInstrScope scope("TADD");
    PTO_COSTMODEL_A2A3_TADD_IMPL(dst, src0, src1);
    ::pto::mocker::InjectTileCycles(dst);
    ::pto::mocker::RecordInstr("TADD", dst, src0, src1);
}

PTO_COSTMODEL_DEFINE_SIMPLE(TPAIRREDUCESUM)
PTO_COSTMODEL_DEFINE_SIMPLE(TSUBRELUCONV)
PTO_COSTMODEL_DEFINE_SIMPLE(TADDRELUCONV)
PTO_COSTMODEL_DEFINE_A2A3(TABS)
PTO_COSTMODEL_DEFINE_A2A3(TAND)
PTO_COSTMODEL_DEFINE_A2A3(TOR)
PTO_COSTMODEL_DEFINE_A2A3(TSUB)
PTO_COSTMODEL_DEFINE_A2A3(TSUBVIEW)
PTO_COSTMODEL_DEFINE_A2A3(TMUL)
PTO_COSTMODEL_DEFINE_A2A3(TMIN)
PTO_COSTMODEL_DEFINE_A2A3(TMAX)
PTO_COSTMODEL_DEFINE_A2A3(TEXPANDS)
PTO_COSTMODEL_DEFINE_A2A3(TLOAD)
PTO_COSTMODEL_DEFINE_A2A3(TPREFETCH)
PTO_COSTMODEL_DEFINE_A2A3(TCMPS)
PTO_COSTMODEL_DEFINE_A2A3(TCMP)
PTO_COSTMODEL_DEFINE_A2A3(TCONCAT)
PTO_COSTMODEL_DEFINE_A2A3(TDIV)
PTO_COSTMODEL_DEFINE_A2A3(TSHL)
PTO_COSTMODEL_DEFINE_A2A3(TSHR)
PTO_COSTMODEL_DEFINE_A2A3(TXOR)
PTO_COSTMODEL_DEFINE_A2A3(TLOG)
PTO_COSTMODEL_DEFINE_A2A3(TDIVS)
PTO_COSTMODEL_DEFINE_A2A3(TPRELU)
PTO_COSTMODEL_DEFINE_A2A3(TADDC)
PTO_COSTMODEL_DEFINE_A2A3(TSUBC)
PTO_COSTMODEL_DEFINE_SIMPLE(TGEMV_MX)
PTO_COSTMODEL_DEFINE_SIMPLE(TMATMUL_MX)
PTO_COSTMODEL_DEFINE_A2A3(TMATMUL)
PTO_COSTMODEL_DEFINE_A2A3(TMATMUL_ACC)
PTO_COSTMODEL_DEFINE_A2A3(TMATMUL_BIAS)
PTO_COSTMODEL_DEFINE_A2A3(TGEMV)
PTO_COSTMODEL_DEFINE_A2A3(TGEMV_ACC)
PTO_COSTMODEL_DEFINE_A2A3(TGEMV_BIAS)
PTO_COSTMODEL_DEFINE_A2A3(TMRGSORT)
PTO_COSTMODEL_DEFINE_A2A3(TEXTRACT)
PTO_COSTMODEL_DEFINE_A2A3(TIMG2COL)
PTO_COSTMODEL_DEFINE_A2A3(SETFMATRIX)
PTO_COSTMODEL_DEFINE_A2A3(SET_IMG2COL_RPT)
PTO_COSTMODEL_DEFINE_A2A3(SET_IMG2COL_PADDING)
PTO_COSTMODEL_DEFINE_A2A3(TINSERT)
PTO_COSTMODEL_DEFINE_SIMPLE(TFILLPAD_INPLACE)
PTO_COSTMODEL_DEFINE_SIMPLE(TFILLPAD_EXPAND)
PTO_COSTMODEL_DEFINE_A2A3(TSORT32)
PTO_COSTMODEL_DEFINE_A2A3(TGATHER)
PTO_COSTMODEL_DEFINE_A2A3(TCI)
PTO_COSTMODEL_DEFINE_A2A3(TTRI)
PTO_COSTMODEL_DEFINE_SIMPLE(TPARTADD)
PTO_COSTMODEL_DEFINE_SIMPLE(TPARTMUL)
PTO_COSTMODEL_DEFINE_SIMPLE(TPARTMAX)
PTO_COSTMODEL_DEFINE_SIMPLE(TPARTMIN)
PTO_COSTMODEL_DEFINE_SIMPLE(TPARTARGMAX)
PTO_COSTMODEL_DEFINE_SIMPLE(TPARTARGMIN)
PTO_COSTMODEL_DEFINE_SIMPLE(TFUSEDMULADD)
PTO_COSTMODEL_DEFINE_SIMPLE(TMULADDDST)
PTO_COSTMODEL_DEFINE_SIMPLE(TSUBRELU)
PTO_COSTMODEL_DEFINE_SIMPLE(TFUSEDMULADDRELU)
PTO_COSTMODEL_DEFINE_A2A3(TCVT)
PTO_COSTMODEL_DEFINE_A2A3(TMOV)
PTO_COSTMODEL_DEFINE_A2A3(TROWSUM)
PTO_COSTMODEL_DEFINE_SIMPLE(TROWPROD)
PTO_COSTMODEL_DEFINE_A2A3(TCOLSUM)
PTO_COSTMODEL_DEFINE_SIMPLE(TCOLPROD)
PTO_COSTMODEL_DEFINE_A2A3(TCOLMAX)
PTO_COSTMODEL_DEFINE_SIMPLE(TCOLARGMAX)
PTO_COSTMODEL_DEFINE_SIMPLE(TCOLARGMIN)
PTO_COSTMODEL_DEFINE_A2A3(TROWMAX)
PTO_COSTMODEL_DEFINE_SIMPLE(TROWARGMAX)
PTO_COSTMODEL_DEFINE_SIMPLE(TRESHAPE)
PTO_COSTMODEL_DEFINE_A2A3(TROWMIN)
PTO_COSTMODEL_DEFINE_SIMPLE(TROWARGMIN)
PTO_COSTMODEL_DEFINE_A2A3(TSELS)
PTO_COSTMODEL_DEFINE_A2A3(TSEL)
PTO_COSTMODEL_DEFINE_A2A3(TTRANS)
PTO_COSTMODEL_DEFINE_A2A3(TMINS)
PTO_COSTMODEL_DEFINE_A2A3(TROWEXPAND)
PTO_COSTMODEL_DEFINE_A2A3(TROWEXPANDDIV)
PTO_COSTMODEL_DEFINE_A2A3(TROWEXPANDMUL)
PTO_COSTMODEL_DEFINE_A2A3(TROWEXPANDSUB)
PTO_COSTMODEL_DEFINE_A2A3(TROWEXPANDADD)
PTO_COSTMODEL_DEFINE_A2A3(TROWEXPANDMAX)
PTO_COSTMODEL_DEFINE_A2A3(TROWEXPANDMIN)
PTO_COSTMODEL_DEFINE_A2A3(TROWEXPANDEXPDIF)
PTO_COSTMODEL_DEFINE_A2A3(TRSQRT)
PTO_COSTMODEL_DEFINE_A2A3(TSQRT)
PTO_COSTMODEL_DEFINE_A2A3(TEXP)
PTO_COSTMODEL_DEFINE_SIMPLE(TPOW)
PTO_COSTMODEL_DEFINE_SIMPLE(TPOWS)
PTO_COSTMODEL_DEFINE_A2A3(TNOT)
PTO_COSTMODEL_DEFINE_A2A3(TRELU)
PTO_COSTMODEL_DEFINE_A2A3(TGATHERB)
PTO_COSTMODEL_DEFINE_A2A3(TADDS)
PTO_COSTMODEL_DEFINE_A2A3(TAXPY)
PTO_COSTMODEL_DEFINE_A2A3(TSUBS)
PTO_COSTMODEL_DEFINE_A2A3(TMULS)
PTO_COSTMODEL_DEFINE_SIMPLE(TFMODS)
PTO_COSTMODEL_DEFINE_SIMPLE(TREMS)
PTO_COSTMODEL_DEFINE_A2A3(TMAXS)
PTO_COSTMODEL_DEFINE_SIMPLE(TANDS)
PTO_COSTMODEL_DEFINE_SIMPLE(TORS)
PTO_COSTMODEL_DEFINE_SIMPLE(TSHLS)
PTO_COSTMODEL_DEFINE_SIMPLE(TSHRS)
PTO_COSTMODEL_DEFINE_SIMPLE(TXORS)
PTO_COSTMODEL_DEFINE_A2A3(TLRELU)
PTO_COSTMODEL_DEFINE_SIMPLE(TADDSC)
PTO_COSTMODEL_DEFINE_SIMPLE(TSUBSC)
PTO_COSTMODEL_DEFINE_A2A3(TCOLMIN)
PTO_COSTMODEL_DEFINE_A2A3(TSCATTER)
PTO_COSTMODEL_DEFINE_A2A3(TCOLEXPAND)
PTO_COSTMODEL_DEFINE_SIMPLE(MGATHER)
PTO_COSTMODEL_DEFINE_SIMPLE(MSCATTER)
PTO_COSTMODEL_DEFINE_A2A3(TNEG)
PTO_COSTMODEL_DEFINE_A2A3(TCOLEXPANDDIV)
PTO_COSTMODEL_DEFINE_A2A3(TCOLEXPANDMUL)
PTO_COSTMODEL_DEFINE_A2A3(TCOLEXPANDADD)
PTO_COSTMODEL_DEFINE_A2A3(TCOLEXPANDMAX)
PTO_COSTMODEL_DEFINE_A2A3(TCOLEXPANDMIN)
PTO_COSTMODEL_DEFINE_A2A3(TCOLEXPANDSUB)
PTO_COSTMODEL_DEFINE_A2A3(TCOLEXPANDEXPDIF)
PTO_COSTMODEL_DEFINE_A2A3(TDEQUANT)
PTO_COSTMODEL_DEFINE_A2A3(TREM)
PTO_COSTMODEL_DEFINE_A2A3(TFMOD)
PTO_COSTMODEL_DEFINE_SIMPLE(THISTOGRAM)
PTO_COSTMODEL_DEFINE_SIMPLE(TQUANT)
PTO_COSTMODEL_DEFINE_SIMPLE(TINTERLEAVE)
PTO_COSTMODEL_DEFINE_SIMPLE(TDEINTERLEAVE)
PTO_COSTMODEL_DEFINE_SIMPLE(TGET_SCALE_ADDR)

#undef PTO_COSTMODEL_DEFINE_SIMPLE

// TASSIGN maps a tile/tensor to an address; in the cost model it is a
// scalar-stage memory-setup instruction that produces no data (see latency.hpp
// and tile_dep_tracker.hpp), so the perf-sim stub only records it — no real
// memory is touched. The two-parameter (T&, AddrType) signature intentionally
// matches the friend declaration in pto_tile.hpp and the per-arch TAssign.hpp
// definitions, which is the signature the public TASSIGN API resolves to; the
// variadic PTO_COSTMODEL_DEFINE_SIMPLE macro would instead generate a
// forwarding-reference template and leave the call site undefined.
template <typename T, typename AddrType>
PTO_INTERNAL void TASSIGN_IMPL(T &obj, AddrType addr)
{
    if constexpr (is_tile_data_v<T> || is_conv_tile_v<T>) {
        static_assert(std::is_integral_v<AddrType>, "Tile can only be assigned with address of int type.");
        obj.assignData(reinterpret_cast<typename T::TileDType>(static_cast<std::uintptr_t>(addr)));
    } else {
        static_assert(std::is_pointer_v<AddrType>, "GlobalTensor can only be assigned with address of pointer type.");
        obj.SetAddr(addr);
    }
    ::pto::mocker::RecordPtoInstr("TASSIGN", obj);
}

template <Op OpCode>
PTO_INTERNAL void TSYNC_IMPL()
{
    ::pto::mocker::RecordPtoInstr("TSYNC");
}

template <PrintFormat Format, typename... Args>
PTO_INTERNAL void TPRINT_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TPRINT", std::forward<Args>(args)...);
}

template <typename TileData, typename GlobalData, auto... Params, typename... Args>
PTO_INTERNAL void TSTORE_IMPL(Args &&...args)
{
    ::pto::mocker::PtoInstrScope scope("TSTORE");
    PTO_COSTMODEL_A2A3_TSTORE_IMPL<TileData, GlobalData, Params...>(std::forward<Args>(args)...);
    ::pto::mocker::RecordCompletedPtoInstr("TSTORE", std::forward<Args>(args)...);
}

template <typename TileData, typename GlobalData, typename FpTileData, auto... Params, typename... Args>
PTO_INTERNAL void TSTORE_IMPL(Args &&...args)
{
    ::pto::mocker::PtoInstrScope scope("TSTORE");
    PTO_COSTMODEL_A2A3_TSTORE_IMPL<TileData, GlobalData, FpTileData, Params...>(std::forward<Args>(args)...);
    ::pto::mocker::RecordCompletedPtoInstr("TSTORE", std::forward<Args>(args)...);
}

template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TDIV_IMPL(Args &&...args)
{
    ::pto::mocker::PtoInstrScope scope("TDIV");
    PTO_COSTMODEL_A2A3_TDIV_IMPL<PrecisionType>(std::forward<Args>(args)...);
    ::pto::mocker::RecordCompletedPtoInstr("TDIV", std::forward<Args>(args)...);
}

template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TLOG_IMPL(Args &&...args)
{
    ::pto::mocker::PtoInstrScope scope("TLOG");
    PTO_COSTMODEL_A2A3_TLOG_IMPL<PrecisionType>(std::forward<Args>(args)...);
    ::pto::mocker::RecordCompletedPtoInstr("TLOG", std::forward<Args>(args)...);
}

template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TDIVS_IMPL(Args &&...args)
{
    ::pto::mocker::PtoInstrScope scope("TDIVS");
    PTO_COSTMODEL_A2A3_TDIVS_IMPL<PrecisionType>(std::forward<Args>(args)...);
    ::pto::mocker::RecordCompletedPtoInstr("TDIVS", std::forward<Args>(args)...);
}

template <auto Phase, typename... Args>
PTO_INTERNAL void TGEMV_MX_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TGEMV_MX", std::forward<Args>(args)...);
}

template <auto Phase, typename... Args>
PTO_INTERNAL void TMATMUL_MX_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TMATMUL_MX", std::forward<Args>(args)...);
}

template <uint16_t Rounds, typename DstTile, typename... Args>
PTO_INTERNAL void TRANDOM_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TRANDOM", std::forward<Args>(args)...);
}

template <auto Phase, typename... Args>
PTO_INTERNAL void TMATMUL_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TMATMUL", std::forward<Args>(args)...);
}

template <auto Phase, typename... Args>
PTO_INTERNAL void TMATMUL_ACC_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TMATMUL_ACC", std::forward<Args>(args)...);
}

template <auto Phase, typename... Args>
PTO_INTERNAL void TMATMUL_BIAS_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TMATMUL_BIAS", std::forward<Args>(args)...);
}

template <auto Phase, typename... Args>
PTO_INTERNAL void TGEMV_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TGEMV", std::forward<Args>(args)...);
}

template <auto Phase, typename... Args>
PTO_INTERNAL void TGEMV_ACC_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TGEMV_ACC", std::forward<Args>(args)...);
}

template <auto Phase, typename... Args>
PTO_INTERNAL void TGEMV_BIAS_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TGEMV_BIAS", std::forward<Args>(args)...);
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, typename Src3TileData, bool exhausted, typename... Args>
PTO_INTERNAL void TMRGSORT_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TMRGSORT", std::forward<Args>(args)...);
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, bool exhausted, typename... Args>
PTO_INTERNAL void TMRGSORT_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TMRGSORT", std::forward<Args>(args)...);
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData, bool exhausted,
          typename... Args>
PTO_INTERNAL void TMRGSORT_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TMRGSORT", std::forward<Args>(args)...);
}

template <typename DstTileData, typename SrcTileData, auto... Params, typename... Args>
PTO_INTERNAL void TEXTRACT_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TEXTRACT", std::forward<Args>(args)...);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, auto... Params, typename... Args>
PTO_INTERNAL void TEXTRACT_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TEXTRACT", std::forward<Args>(args)...);
}

template <typename TileData, typename ConvTileData, auto FmatrixMode, typename... Args>
PTO_INTERNAL void TIMG2COL_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TIMG2COL", std::forward<Args>(args)...);
}

template <typename ConvTileData, auto FmatrixMode, typename... Args>
PTO_INTERNAL void SETFMATRIX_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("SETFMATRIX", std::forward<Args>(args)...);
}

template <typename ConvTileData, auto FmatrixMode, typename... Args>
PTO_INTERNAL void SET_IMG2COL_RPT_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("SET_IMG2COL_RPT", std::forward<Args>(args)...);
}

template <typename ConvTileData, auto FmatrixMode, typename... Args>
PTO_INTERNAL void SET_IMG2COL_PADDING_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("SET_IMG2COL_PADDING", std::forward<Args>(args)...);
}

template <typename DstTileData, typename SrcTileData, auto... Params, typename... Args>
PTO_INTERNAL void TINSERT_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TINSERT", std::forward<Args>(args)...);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, auto... Params, typename... Args>
PTO_INTERNAL void TINSERT_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TINSERT", std::forward<Args>(args)...);
}

template <auto mode, typename... Args>
PTO_INTERNAL void TINSERT_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TINSERT", std::forward<Args>(args)...);
}

template <typename TileData, auto PadVal, typename... Args>
PTO_INTERNAL void TFILLPAD_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TFILLPAD", std::forward<Args>(args)...);
}

template <typename DstTileData, typename SrcTileData, typename... Args>
PTO_INTERNAL void TFILLPAD_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TFILLPAD", std::forward<Args>(args)...);
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, typename TileDataTmp,
          auto cmpMode, typename... Args>
PTO_INTERNAL void TGATHER_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TGATHER", std::forward<Args>(args)...);
}

template <typename DstTileData, typename SrcTileData, auto maskPattern, auto gatherType, typename... Args>
PTO_INTERNAL void TGATHER_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TGATHER", std::forward<Args>(args)...);
}

template <typename TileData, typename T, int descending, typename... Args>
PTO_INTERNAL void TCI_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TCI", std::forward<Args>(args)...);
}

template <typename TileData, typename TileDataTmp, typename T, int descending, typename... Args>
PTO_INTERNAL void TCI_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TCI", std::forward<Args>(args)...);
}

template <typename TileData, int isUpperOrLower, typename... Args>
PTO_INTERNAL void TTRI_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TTRI", std::forward<Args>(args)...);
}

template <bool NeedSetCtrl, typename... Args>
PTO_INTERNAL void TCVT_IMPL(Args &&...args)
{
    ::pto::mocker::PtoInstrScope scope("TCVT");
    PTO_COSTMODEL_A2A3_TCVT_IMPL<NeedSetCtrl>(std::forward<Args>(args)...);
    ::pto::mocker::RecordCompletedPtoInstr("TCVT", std::forward<Args>(args)...);
}

template <typename DstTileData, typename SrcTileData, auto... Params, typename... Args>
PTO_INTERNAL void TMOV_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TMOV", std::forward<Args>(args)...);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, auto... Params, typename... Args>
PTO_INTERNAL void TMOV_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TMOV", std::forward<Args>(args)...);
}

template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TROWEXPANDDIV_IMPL(Args &&...args)
{
    ::pto::mocker::PtoInstrScope scope("TROWEXPANDDIV");
    PTO_COSTMODEL_A2A3_TROWEXPANDDIV_IMPL<PrecisionType>(std::forward<Args>(args)...);
    ::pto::mocker::RecordCompletedPtoInstr("TROWEXPANDDIV", std::forward<Args>(args)...);
}

template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TRSQRT_IMPL(Args &&...args)
{
    ::pto::mocker::PtoInstrScope scope("TRSQRT");
    PTO_COSTMODEL_A2A3_TRSQRT_IMPL<PrecisionType>(std::forward<Args>(args)...);
    ::pto::mocker::RecordCompletedPtoInstr("TRSQRT", std::forward<Args>(args)...);
}

template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TSQRT_IMPL(Args &&...args)
{
    ::pto::mocker::PtoInstrScope scope("TSQRT");
    PTO_COSTMODEL_A2A3_TSQRT_IMPL<PrecisionType>(std::forward<Args>(args)...);
    ::pto::mocker::RecordCompletedPtoInstr("TSQRT", std::forward<Args>(args)...);
}

template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TEXP_IMPL(Args &&...args)
{
    ::pto::mocker::PtoInstrScope scope("TEXP");
    PTO_COSTMODEL_A2A3_TEXP_IMPL<PrecisionType>(std::forward<Args>(args)...);
    ::pto::mocker::RecordCompletedPtoInstr("TEXP", std::forward<Args>(args)...);
}

template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TPOW_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TPOW", std::forward<Args>(args)...);
}

template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TPOWS_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TPOWS", std::forward<Args>(args)...);
}

template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TFMODS_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TFMODS", std::forward<Args>(args)...);
}

template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TREMS_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TREMS", std::forward<Args>(args)...);
}

template <auto maskPattern, auto ScatterType, typename... Args>
PTO_INTERNAL void TSCATTER_IMPL(Args &&...args)
{
    ::pto::mocker::PtoInstrScope scope("TSCATTER");
    PTO_COSTMODEL_A2A3_TSCATTER_IMPL<maskPattern, ScatterType>(std::forward<Args>(args)...);
    ::pto::mocker::RecordCompletedPtoInstr("TSCATTER", std::forward<Args>(args)...);
}

template <auto CMode, typename... Args>
PTO_INTERNAL void MGATHER_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("MGATHER", std::forward<Args>(args)...);
}

template <auto CMode, auto Mode, typename... Args>
PTO_INTERNAL void MGATHER_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("MGATHER", std::forward<Args>(args)...);
}

template <auto Mode, auto... Rest, typename... Args>
PTO_INTERNAL void MSCATTER_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("MSCATTER", std::forward<Args>(args)...);
}

template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TCOLEXPANDDIV_IMPL(Args &&...args)
{
    ::pto::mocker::PtoInstrScope scope("TCOLEXPANDDIV");
    PTO_COSTMODEL_A2A3_TCOLEXPANDDIV_IMPL<PrecisionType>(std::forward<Args>(args)...);
    ::pto::mocker::RecordCompletedPtoInstr("TCOLEXPANDDIV", std::forward<Args>(args)...);
}

template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TREM_IMPL(Args &&...args)
{
    ::pto::mocker::PtoInstrScope scope("TREM");
    PTO_COSTMODEL_A2A3_TREM_IMPL<PrecisionType>(std::forward<Args>(args)...);
    ::pto::mocker::RecordCompletedPtoInstr("TREM", std::forward<Args>(args)...);
}

template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TFMOD_IMPL(Args &&...args)
{
    ::pto::mocker::PtoInstrScope scope("TFMOD");
    PTO_COSTMODEL_A2A3_TFMOD_IMPL<PrecisionType>(std::forward<Args>(args)...);
    ::pto::mocker::RecordCompletedPtoInstr("TFMOD", std::forward<Args>(args)...);
}

template <typename Pipe, typename TileProd, auto Split, typename... Args>
PTO_INTERNAL void TPUSH_IMPL(Pipe &pipe, TileProd &tile, Args &&...args)
{
    {
        ::pto::mocker::PtoInstrScope scope("TPUSH");
    }
    ::pto::mocker::InjectTileCycles(pipe);
    ::pto::mocker::RecordTPushSync(pipe, tile, pipe.prod.tileIndex);
    ::pto::mocker::RecordInstr("TPUSH", pipe, tile, std::forward<Args>(args)...);
}

template <typename TileData, typename Pipe, typename... Args>
PTO_INTERNAL void TPUSH_IMPL(TileData &tile, Pipe &pipe, Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TPUSH", tile, pipe, std::forward<Args>(args)...);
}

template <typename Pipe, typename TileCons, auto Split, typename... Args>
PTO_INTERNAL void TPOP_IMPL(Pipe &pipe, TileCons &tile, Args &&...args)
{
    {
        ::pto::mocker::PtoInstrScope scope("TPOP");
    }
    ::pto::mocker::InjectTileCycles(pipe);
    ::pto::mocker::RecordTPopSync(pipe, tile, pipe.cons.tileIndex);
    ::pto::mocker::RecordInstr("TPOP", pipe, tile, std::forward<Args>(args)...);
}

template <typename TileData, typename Pipe, typename... Args>
PTO_INTERNAL void TPOP_IMPL(TileData &tile, Pipe &pipe, Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TPOP", tile, pipe, std::forward<Args>(args)...);
}

template <typename Pipe, auto Split, typename... Args>
PTO_INTERNAL void TFREE_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TFREE", std::forward<Args>(args)...);
}

template <typename Pipe, typename... Args>
PTO_INTERNAL void TFREE_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TFREE", std::forward<Args>(args)...);
}

template <typename Pipe, typename GlobalData, auto Split, typename... Args>
PTO_INTERNAL void TALLOC_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TALLOC", std::forward<Args>(args)...);
}

template <typename Pipe, typename GlobalData, auto Split, typename... Args>
PTO_INTERNAL void TFREE_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TFREE", std::forward<Args>(args)...);
}

template <auto byte, typename... Args>
PTO_INTERNAL void THISTOGRAM_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("THISTOGRAM", std::forward<Args>(args)...);
}

template <auto quantType, auto scaleAlg, typename TileDataOut, typename TileDataSrc, typename TileDataExp,
          typename TileDataMax, typename TileDataScaling, typename... Args>
PTO_INTERNAL void TQUANT_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TQUANT", std::forward<Args>(args)...);
}

template <auto quantType, auto storeMode, typename... Args>
PTO_INTERNAL void TQUANT_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TQUANT", std::forward<Args>(args)...);
}

template <auto quantType, typename TileDataOut, typename TileDataSrc, typename TileDataPara, typename... Args>
PTO_INTERNAL void TQUANT_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TQUANT", std::forward<Args>(args)...);
}

template <auto quantType, typename TileDataOut, typename TileDataSrc, typename TileDataPara, typename TileDataTmp,
          typename... Args>
PTO_INTERNAL void TQUANT_IMPL(Args &&...args)
{
    ::pto::mocker::RecordPtoInstr("TQUANT", std::forward<Args>(args)...);
}

} // namespace pto

#endif // PTO_COSTMODEL_INSTR_IMPL_HPP
