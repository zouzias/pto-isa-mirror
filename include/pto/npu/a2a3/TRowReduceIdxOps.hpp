/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef T_ROW_REDUCE_IDX_OPS_HPP
#define T_ROW_REDUCE_IDX_OPS_HPP

#include <pto/common/utils.hpp>
#include <pto/common/type.hpp>

#ifndef B16_REPEAT_MAX
#define B16_REPEAT_MAX 65535
#endif

#define REDUCE_IDX_ONE_STAGE 4096

namespace pto {
template <typename T, typename InstrOp>
struct TRowReduceIdxOp {
    PTO_INTERNAL static void ReduceIdxInstr(__ubuf__ T *dst, __ubuf__ T *src, uint8_t rptTimes, uint16_t dstRptStride,
                                         uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        InstrOp::ReduceIdxInstrImpl(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride);
    }
    PTO_INTERNAL static void ReduceValIdxInstr(__ubuf__ T *dst, __ubuf__ T *src, uint8_t rptTimes, uint16_t dstRptStride,
                                               uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        InstrOp::ReduceValIdxInstrImpl(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride);
    }

    template <bool CntModeEn, int Cols, uint32_t DstStride, uint32_t SrcStride>
    PTO_INTERNAL static void ReduceInstrByModeIdx(__ubuf__ T *dst, __ubuf__ T *src, unsigned rptTimes)
    {
        constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(T);
        if constexpr (DstStride > B16_REPEAT_MAX) {
            for (int i = 0; i < rptTimes; i++) {
                ReduceIdxInstr(dst + i * DstStride, src + i * Cols, 1, 0, 1, 0);
            }
        } else if constexpr (CntModeEn) {
            set_mask_count();
            set_vector_mask(0, (uint32_t)rptTimes * elemPerRpt);
            ReduceIdxInstr(dst, src, 0, DstStride, 1, SrcStride);
            set_mask_norm();
            set_vector_mask(-1, -1);
        } else {
            ReduceIdxInstr(dst, src, rptTimes, DstStride, 1, SrcStride);
        }
    }
};

template <typename InstrOp, typename T, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void ProcReduceIdx4096(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *tmp, int validCol, int validRow)
{
    using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(T);
    constexpr uint8_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint32_t srcRptStride = TileDataIn::Cols / elemPerBlock;

    cce::printf("vsize: %d %d, %d\n", validRow, validCol, CeilDivision(validCol, elemPerRpt));
    for (int i = 0; i < validRow; i++) {
        set_mask_count();
        set_vector_mask(0, (uint32_t)validCol);
        InstrOp::ReduceValIdxInstr(tmp + i * TileDataTmp::Cols, src + i * TileDataIn::Cols,
            CeilDivision(validCol, elemPerRpt), 1, 1, elemPerBlock);
        pipe_barrier(PIPE_V);
        SetContinuousMask(CeilDivision(validCol, elemPerRpt) * 2);
        vreducev2(reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols + TileDataTmp::Cols / 2),
            reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols),
            reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols),
            1, 1, 0x55, 1, 0);
        set_mask_norm();
        set_vector_mask(-1, -1);
    }
    pipe_barrier(PIPE_V);
    set_vector_mask(-1, -1);
    for (int i = 0; i < validRow; i++) {
        for (int j = 0; j < CeilDivision(validCol, elemPerRpt); j++) {
            cce::printf("%f %u ", *(tmp + i * TileDataTmp::Cols + j * 2),
                *(reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols + j * 2 + 1)));
        }
        cce::printf("\n");
    }
}

template <typename InstrOp, typename T, uint32_t DstCols, uint32_t SrcCols>
PTO_INTERNAL void OneRepeatProcIdx(__ubuf__ T *dst, __ubuf__ T *src, int validCol, int validRow)
{
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(T);
    constexpr uint8_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint32_t srcRptStride = SrcCols / elemPerBlock;

    if (validCol == elemPerRpt) {
        InstrOp::template ReduceInstrByModeIdx<true, SrcCols, DstCols, srcRptStride>(dst, src, validRow);
        pipe_barrier(PIPE_V);
        return;
    }

    int remain = validCol % elemPerRpt;
    int rowRptTimes = validRow / REPEAT_MAX;
    unsigned rptTimes;
    SetContinuousMask(remain);
    do {
        rptTimes = (rowRptTimes == 0 ? (validRow % REPEAT_MAX) : REPEAT_MAX);
        InstrOp::template ReduceInstrByModeIdx<false, SrcCols, DstCols, srcRptStride>(dst, src, rptTimes);
        pipe_barrier(PIPE_V);
        rowRptTimes -= 1;
        dst += rptTimes;
        src += rptTimes * SrcCols;
    } while (rowRptTimes >= 0);

    set_vector_mask(-1, -1);
}

template <typename InstrOp, typename T, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TRowReduceIdxInstr(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *tmp, int validCol, int validRow)
{
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(T);
    if (validCol <= elemPerRpt) {
        OneRepeatProcIdx<InstrOp, T, TileDataOut::Cols, TileDataIn::Cols>(
            dst, src, validCol, validRow);
        return;
    } else if (validCol <= REDUCE_IDX_ONE_STAGE) {
        ProcReduceIdx4096<InstrOp, T, TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp, validCol, validRow);
        return;
    }
}

template <typename T>
struct TRowCMaxOp : TRowReduceIdxOp<T, TRowCMaxOp<T>> {
    PTO_INTERNAL static void ReduceIdxInstrImpl(__ubuf__ T *dst, __ubuf__ T *src, uint8_t rptTimes, uint16_t dstRptStride,
                                                uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        vcmax(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, ONLY_INDEX);
    }
    
    PTO_INTERNAL static void ReduceValIdxInstrImpl(__ubuf__ T *dst, __ubuf__ T *src, uint8_t rptTimes, uint16_t dstRptStride,
                                               uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        vcmax(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, VALUE_INDEX);
    }
};

template <typename T, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__tf__ PTO_INTERNAL void TRowIdxMax(typename TileDataOut::TileDType __out__ dstData,
                                 typename TileDataIn::TileDType __in__ srcData,
                                 typename TileDataTmp::TileDType __in__ tmpData, int validCol, int validRow,
                                 unsigned version)
{
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);
    __ubuf__ T *tmp = (__ubuf__ T *)__cce_get_tile_ptr(tmpData);
    TRowReduceIdxInstr<TRowCMaxOp<T>, T, TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp, validCol, validRow);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWCMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    int validCol = src.GetValidCol();
    int validRow = src.GetValidRow();
    TRowIdxMax<typename TileDataIn::DType, TileDataOut, TileDataIn, TileDataTmp>(
        dst.data(), src.data(), tmp.data(), validCol, validRow, VFImplKind::VFIMPL_DEFAULT);
}

template <typename T>
struct TRowCMinOp : TRowReduceIdxOp<T, TRowCMinOp<T>> {
    PTO_INTERNAL static void ReduceIdxInstrImpl(__ubuf__ T *dst, __ubuf__ T *src, uint8_t rptTimes, uint16_t dstRptStride,
                                                uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        vcmin(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, ONLY_INDEX);
    }
    
    PTO_INTERNAL static void ReduceValIdxInstrImpl(__ubuf__ T *dst, __ubuf__ T *src, uint8_t rptTimes, uint16_t dstRptStride,
                                               uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        vcmin(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, VALUE_INDEX);
    }
};

template <typename T, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__tf__ PTO_INTERNAL void TRowIdxMin(typename TileDataOut::TileDType __out__ dstData,
                                 typename TileDataIn::TileDType __in__ srcData,
                                 typename TileDataTmp::TileDType __in__ tmpData, int validCol, int validRow,
                                 unsigned version)
{
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);
    __ubuf__ T *tmp = (__ubuf__ T *)__cce_get_tile_ptr(tmpData);
    TRowReduceIdxInstr<TRowCMinOp<T>, T, TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp, validCol, validRow);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWCMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    int validCol = src.GetValidCol();
    int validRow = src.GetValidRow();
    TRowIdxMin<typename TileDataIn::DType, TileDataOut, TileDataIn, TileDataTmp>(
        dst.data(), src.data(), tmp.data(), validCol, validRow, VFImplKind::VFIMPL_DEFAULT);
}

}
#endif
