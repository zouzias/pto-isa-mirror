/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COSTMODEL_INSTR_TEMPLATE_IMPL_HPP
#define PTO_COSTMODEL_INSTR_TEMPLATE_IMPL_HPP

#include <cstdint>
#include <type_traits>
#include <utility>

namespace pto {

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

#endif // PTO_COSTMODEL_INSTR_TEMPLATE_IMPL_HPP
