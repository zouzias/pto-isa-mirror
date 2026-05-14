/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TLOAD_HPP_310P3
#define TLOAD_HPP_310P3

#include "TFillPad.hpp"

namespace pto {
template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadInstrGm2ub(__ubuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
                                  uint16_t nBurst, uint32_t lenByteBurst, uint32_t gmByteGap, uint32_t ubGap,
                                  uint32_t ubPad)
{
    constexpr uint32_t ELEMS_PER_BLOCK = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    uint32_t lenBurst = (lenByteBurst + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
    uint32_t dstStride = (ubGap + lenBurst) * ELEMS_PER_BLOCK;
    uint32_t srcStride = (gmByteGap + lenByteBurst) / sizeof(typename TileData::DType);

    if (srcStride % ELEMS_PER_BLOCK == 0) {
        copy_gm_to_ubuf(dst, src, 0, nBurst, lenBurst, srcStride / ELEMS_PER_BLOCK - lenBurst, ubGap);
    } else {
        for (int i = 0; i < nBurst; ++i) {
            copy_gm_to_ubuf(dst + i * dstStride, src + i * srcStride, 0, 1, lenBurst, 0, 0);
        }
    }
    if (ubPad != 0) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        auto padValue = GetPadValue<TileData>();
        Handle32BAlignedPad_Other<TileData, TileData>(
            (decltype(getCopyNullPtr<TileData>()))dst, nBurst, lenByteBurst / sizeof(typename TileData::DType), lenBurst * ELEMS_PER_BLOCK, padValue);
        // 对外保持整个指令的结束流水是MTE2
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadNd2nzInstr(__cbuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
                                  uint16_t ndNum, uint16_t nValue, uint16_t dValue, uint16_t srcNdMatrixStride,
                                  uint16_t srcDValue, uint16_t dstNzC0Stride, uint16_t dstNzNStride,
                                  uint16_t dstNzMatrixStride)
{
    // Parameter list:
    // dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
    // dstNzC0Stride, dstNzNStride, dstNzMatrixStride
    PTO_ASSERT(ndNum == 1, "The ndNum must be 1 for 310P3 now.");
    (void)srcNdMatrixStride;
    (void)dstNzMatrixStride;
    constexpr uint32_t ELEMS_PER_BLOCK = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    uint32_t nBurst = ((dValue + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK);
    for (size_t i = 0; i < nValue; i++) {
        copy_gm_to_cbuf(dst + i * dstNzNStride * ELEMS_PER_BLOCK, src + i * srcDValue, 0, 
            nBurst, 1, 0, dstNzC0Stride - 1, (pad_t)0);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadInstrGm2L1(__cbuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
                                  uint16_t nBurst, uint16_t lenBurst, uint16_t gmGap, uint16_t l1Gap)
{
    copy_gm_to_cbuf(dst, src, (uint8_t)0, nBurst, lenBurst, gmGap, l1Gap, (pad_t)0);
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2ubNd2nd(__ubuf__ typename TileData::DType *dstAddr, typename GlobalData::DType *srcAddr,
                                  int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                  int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    static_assert(TileData::Rows < 4096, "Fix: TLOAD Rows>=4095 not supported in 310P3");
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(validRow == gShape0 * gShape1 * gShape2 * gShape3,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    PTO_ASSERT(gShape3 < 4096, "The gshape3 (which equals nBurst) must be less than 4096 for 310P3");
    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    uint16_t nBurst = gShape3;
    uint32_t lenBurst = validCol * sizeof(typename TileData::DType);
    uint64_t gmGapValue = (gStride3 - gShape4) * sizeof(typename TileData::DType);
    uint32_t gmGap = (uint32_t)gmGapValue;
    uint32_t ubGapElement = (TileData::Cols - validCol);
    uint32_t ubGap = (ubGapElement * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint32_t ubPad = 0;
    if constexpr (TileData::PadVal != PadValue::Null) {
        ubPad = ubGapElement % blockSizeElem;
    }
    __ubuf__ typename TileData::DType *dstAddrP = dstAddr;
    typename GlobalData::DType *srcAddrP = srcAddr;
    int64_t dstStride2 = gShape3 * TileData::Cols;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;

    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t srcAddr0 = i * gStride0;
        int64_t dstAddr0 = i * dstStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t srcAddr1 = j * gStride1;
            int64_t dstAddr1 = j * dstStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                srcAddrP = srcAddr + srcAddr0 + srcAddr1 + k * gStride2;
                dstAddrP = dstAddr + dstAddr0 + dstAddr1 + k * dstStride2;
                TLoadInstrGm2ub<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, ubGap, ubPad);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2ubDn2dn(__ubuf__ typename TileData::DType *dstAddr, typename GlobalData::DType *srcAddr,
                                  int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                  int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape2 * gShape4,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    PTO_ASSERT(gShape4 < 4096, "The gshape4 (which equals nBurst) must be less than 4096 for 310P3");
    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    uint16_t nBurst = gShape4;
    uint32_t lenBurst = validRow * sizeof(typename TileData::DType);
    uint64_t gmGapValue = (gStride4 - gShape3) * sizeof(typename TileData::DType);
    uint32_t gmGap = (uint32_t)gmGapValue;
    uint32_t ubGapElement = (TileData::Rows - gShape3);
    uint32_t ubGap = (ubGapElement * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint32_t ubPad = 0;
    if constexpr (TileData::PadVal != PadValue::Null) {
        ubPad = ubGapElement % blockSizeElem;
        set_mov_pad_val(GetPadValue<TileData>());
    }
    typename GlobalData::DType *srcAddrP = srcAddr;
    __ubuf__ typename TileData::DType *dstAddrP = dstAddr;

    int64_t dstStride2 = gShape4 * TileData::Rows;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * dstStride0;
        int64_t srcAddr0 = i * gStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t dstAddr1 = j * dstStride1;
            int64_t srcAddr1 = j * gStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                dstAddrP = dstAddr + dstAddr0 + dstAddr1 + k * dstStride2;
                srcAddrP = srcAddr + srcAddr0 + srcAddr1 + k * gStride2;
                TLoadInstrGm2ub<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, ubGap, ubPad);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckNzFormat(int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int validRow,
                                int validCol)
{
    static_assert(
        GlobalData::staticShape[3] == FRACTAL_NZ_ROW &&
            GlobalData::staticShape[4] == C0_SIZE_BYTE / sizeof(typename TileData::DType),
        "Fix: When TileData is NZ format, the last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
    PTO_ASSERT(validRow == gShape2 * gShape3, "The validRow of TileData must be equal to Shape2 * Shape3 of NZ shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape4,
               "The validCol of TileData must be equal to Shape0 * Shape1 * Shape4 of NZ shape!");
    PTO_ASSERT(gShape1 < 4096, "The gshape1 (which equals nBurst) must be less than 4096 for 310P3");
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2ubNz2nz(__ubuf__ typename TileData::DType *dstAddr, typename GlobalData::DType *srcAddr,
                                  int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                  int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    CheckNzFormat<TileData, GlobalData>(gShape0, gShape1, gShape2, gShape3, gShape4, validRow, validCol);
    uint16_t nBurst = gShape1;
    uint32_t lenBurst = validRow * C0_SIZE_BYTE;
    uint32_t gmGap = (gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType);
    uint32_t ubGap = TileData::Rows - validRow;
    typename GlobalData::DType *srcAddrP = srcAddr;
    __ubuf__ typename TileData::DType *dstAddrP = dstAddr;
    int64_t tileStride = TileData::Rows * gShape1 * gShape4;
    for (uint32_t i = 0; i < gShape0; i++) {
        srcAddrP = srcAddr + i * gStride0;
        dstAddrP = dstAddr + i * tileStride;
        TLoadInstrGm2ub<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, ubGap, 0);
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadGm2ub(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
                                    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                    int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __ubuf__ typename TileData::DType *dstAddr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;
    if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::ND) {
        TLoadGm2ubNd2nd<TileData, GlobalData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                              gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::DN) {
        TLoadGm2ubDn2dn<TileData, GlobalData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                              gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ) {
        TLoadGm2ubNz2nz<TileData, GlobalData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                              gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Nd2nd(__cbuf__ typename TileData::DType *dstAddr, typename GlobalData::DType *srcAddr,
                                  int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                  int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(gShape4 * sizeof(typename TileData::DType) % BLOCK_BYTE_SIZE == 0,
               "The 5th dim of ND shape must be 32 bytes aligned!");
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(validRow == gShape0 * gShape1 * gShape2 * gShape3,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    PTO_ASSERT(gShape3 < 4096, "The gshape3 (which equals nBurst) must be less than 4096 for 310P3");
    uint16_t nBurst = gShape3;
    uint16_t lenBurst = (validCol * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t gmGap = ((gStride3 - gShape4) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t l1Gap = ((TileData::Cols - validCol) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    int64_t dstStride2 = gShape3 * TileData::Cols;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;
    typename GlobalData::DType *srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t srcAddr0 = i * gStride0;
        int64_t dstAddr0 = i * dstStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t dstAddr1 = j * dstStride1;
            int64_t srcAddr1 = j * gStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                srcAddrP = srcAddr + srcAddr0 + srcAddr1 + k * gStride2;
                dstAddrP = dstAddr + dstAddr0 + dstAddr1 + k * dstStride2;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Dn2dn(__cbuf__ typename TileData::DType *dstAddr, typename GlobalData::DType *srcAddr,
                                  int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                  int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(gShape3 * sizeof(typename TileData::DType) % BLOCK_BYTE_SIZE == 0,
               "The 4th dim of DN shape must be 32 bytes aligned!");
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape2 * gShape4,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    PTO_ASSERT(gShape4 < 4096, "The gshape4 (which equals nBurst) must be less than 4096 for 310P3");
    uint16_t nBurst = gShape4;
    uint16_t lenBurst = (validRow * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t gmGap = ((gStride4 - gShape3) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t l1Gap = ((TileData::Rows - gShape3) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;
    typename GlobalData::DType *srcAddrP = srcAddr;

    int64_t dstStride2 = gShape4 * TileData::Rows;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * dstStride0;
        int64_t srcAddr0 = i * gStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t srcAddr1 = j * gStride1;
            int64_t dstAddr1 = j * dstStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                srcAddrP = srcAddr + srcAddr0 + srcAddr1 + k * gStride2;
                dstAddrP = dstAddr + dstAddr0 + dstAddr1 + k * dstStride2;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Nz2nz(__cbuf__ typename TileData::DType *dstAddr, typename GlobalData::DType *srcAddr,
                                  int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                  int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    CheckNzFormat<TileData, GlobalData>(gShape0, gShape1, gShape2, gShape3, gShape4, validRow, validCol);
    uint16_t nBurst = gShape1;
    uint32_t lenBurst = validRow;
    uint32_t gmGap = ((gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint32_t l1Gap = TileData::Rows - validRow;
    typename GlobalData::DType *srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;
    int64_t tileStride = TileData::Rows * gShape1 * gShape4;

    for (uint32_t i = 0; i < gShape0; i++) {
        srcAddrP = srcAddr + i * gStride0;
        dstAddrP = dstAddr + i * tileStride;
        TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, l1Gap);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1VectorInND(__cbuf__ typename TileData::DType *dstAddr, typename GlobalData::DType *srcAddr,
                                       int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                       int gStride1, int gStride2, int gStride3, int gStride4, int validRow,
                                       int validCol)
{
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(validRow == gShape0 * gShape1 * gShape2 * gShape3,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    static_assert(GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1,
                  "Fix: GlobalTensor ony support 2 dim when using vector input!");
    uint16_t nValue = gShape3;
    uint16_t dValue = gShape4;
    uint16_t srcDValue = gStride3;
    typename GlobalData::DType *srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;
    // Parameter list:
    // dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
    // dstNzC0Stride, dstNzNStride, dstNzMatrixStride
    TLoadNd2nzInstr<TileData, GlobalData>(dstAddrP, srcAddrP, 1, nValue, dValue, 0, srcDValue, TileData::Rows, 1, 1);
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1VectorInDn(__cbuf__ typename TileData::DType *dstAddr, typename GlobalData::DType *srcAddr,
                                       int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                       int gStride1, int gStride2, int gStride3, int gStride4, int validRow,
                                       int validCol)
{
    static_assert(GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1,
                  "Fix: GlobalTensor ony support 2 dim when using vector input!");
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape2 * gShape4,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    uint16_t nValue = gShape4;
    uint16_t dValue = gShape3;
    uint16_t srcDValue = gStride3;
    typename GlobalData::DType *srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;
    // Parameter list:
    // dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
    // dstNzC0Stride, dstNzNStride, dstNzMatrixStride
    TLoadNd2nzInstr<TileData, GlobalData>(dstAddrP, srcAddrP, 1, nValue, dValue, 0, srcDValue, TileData::Cols, 1, 1);
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadGm2L1(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
                                    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                    int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;
    if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::ND) {
        if constexpr (TileData::Rows == 1) {
            TLoadGm2L1VectorInND<TileData, GlobalData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4,
                                                       gStride0, gStride1, gStride2, gStride3, gStride4, validRow,
                                                       validCol);
        } else {
            TLoadGm2L1Nd2nd<TileData, GlobalData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4,
                                                  gStride0, gStride1, gStride2, gStride3, gStride4, validRow, validCol);
        }
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::DN) {
        if constexpr (TileData::Cols == 1) {
            TLoadGm2L1VectorInDn<TileData, GlobalData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4,
                                                       gStride0, gStride1, gStride2, gStride3, gStride4, validRow,
                                                       validCol);
        } else {
            TLoadGm2L1Dn2dn<TileData, GlobalData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4,
                                                  gStride0, gStride1, gStride2, gStride3, gStride4, validRow, validCol);
        }
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ) {
        TLoadGm2L1Nz2nz<TileData, GlobalData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                              gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadGm2L1Nd2nz(typename TileData::TileDType __out__ dst,
                                         typename GlobalData::DType __in__ *src, int gShape0, int gShape1, int gShape2,
                                         int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
                                         int gStride3, int gStride4, int validRow, int validCol)
{
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;
    static_assert(GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1,
                  "Fix: GlobalTensor ony support 2 dim when ND2NZ!");
    static_assert(TileData::SFractalSize == 512, "Fix: TileData ony support SFractalSize = 512Bytes!");
    PTO_ASSERT(gShape3 > 0 && gShape3 <= 16384, "The Shape3 of GlobalTensor must be in range of [1, 16384]!");
    PTO_ASSERT(gShape4 > 0 && gShape4 <= 65535, "The Shape4 of GlobalTensor must be must be in range of [1, 65535]!");
    PTO_ASSERT(gStride3 > 0 && gStride3 <= 65535,
               "The Stride3 of GlobalTensor must be must be in range of [1, 65535]!");
    static_assert(TileData::Rows <= 16384, "Fix: The Rows of TileData must be less than 16384!");

    uint16_t nValue = gShape3;
    uint16_t dValue = gShape4;
    uint16_t srcDValue = gStride3;
    // Parameter list:
    // dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
    // dstNzC0Stride, dstNzNStride, dstNzMatrixStride
    TLoadNd2nzInstr<TileData, GlobalData>(dstAddr, srcAddr, 1, nValue, dValue, 0, srcDValue, TileData::Rows, 1, 1);
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadGm2L1Dn2zn(typename TileData::TileDType __out__ dst,
                                         typename GlobalData::DType __in__ *src, int gShape0, int gShape1, int gShape2,
                                         int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
                                         int gStride3, int gStride4, int validRow, int validCol)
{
    static_assert(GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1,
                  "Fix: GlobalTensor ony support 2 dim when DN2ZN!");
    static_assert(TileData::SFractalSize == 512, "Fix: TileData ony support SFractalSize = 512Bytes!");
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;
    PTO_ASSERT(gShape4 > 0 && gShape4 <= 16384, "The Shape4 of GlobalTensor must be in range of [1, 16384]!");
    PTO_ASSERT(gShape3 > 0 && gShape3 <= 65535, "The Shape3 of GlobalTensor must be must be in range of [1, 65535]!");
    PTO_ASSERT(gStride4 > 0 && gStride4 <= 65535,
               "The Stride3 of GlobalTensor must be must be in range of [1, 65535]!");
    static_assert(TileData::Cols <= 16384, "Fix: The Cols of TileData must be less than 16384!");

    uint16_t nValue = gShape4;
    uint16_t dValue = gShape3;
    uint16_t srcDValue = gStride4;
    TLoadNd2nzInstr<TileData, GlobalData>(dstAddr, srcAddr, 1, nValue, dValue, 0, srcDValue, TileData::Cols, 1, 1);
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckNormalTileData(TileData &dst, GlobalData &src)
{
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, int64_t> || std::is_same_v<typename TileData::DType, uint64_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, float>,
        "Fix: Data type must be "
        "int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/float/int64_t/uint64_t!");
    static_assert(TileData::Loc == pto::TileType::Vec || TileData::Loc == pto::TileType::Mat,
                  "Fix: Dst TileType must be Vec or Mat!");
    static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                  "Fix: Source dtype must be same with dst dtype!");

    if constexpr (std::is_same_v<typename TileData::DType, int64_t> ||
                  std::is_same_v<typename TileData::DType, uint64_t>) {
        static_assert(
            (GlobalData::layout == pto::Layout::ND && GetTileLayoutCustom<TileData>() == TileLayoutCustom::ND) ||
                (GlobalData::layout == pto::Layout::DN && GetTileLayoutCustom<TileData>() == TileLayoutCustom::DN),
            "Fix: TLOAD only support ND2ND/DN2DN for b64!");
    }
    PTO_ASSERT(src.GetShape(pto::GlobalTensorDim::DIM_0) > 0 && src.GetShape(pto::GlobalTensorDim::DIM_1) > 0 &&
                   src.GetShape(pto::GlobalTensorDim::DIM_2) > 0 && src.GetShape(pto::GlobalTensorDim::DIM_3) > 0 &&
                   src.GetShape(pto::GlobalTensorDim::DIM_4) > 0 && dst.GetValidRow() > 0 && dst.GetValidCol() > 0,
               "The shape of src and dst must be greater than 0!");
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_TILE_IMPL(TileData &dst, GlobalData &src)
{
    CheckNormalTileData<TileData, GlobalData>(dst, src);
    constexpr bool isSameLayout =
        (GlobalData::layout == pto::Layout::ND && GetTileLayoutCustom<TileData>() == TileLayoutCustom::ND) ||
        (GlobalData::layout == pto::Layout::DN && GetTileLayoutCustom<TileData>() == TileLayoutCustom::DN) ||
        (GlobalData::layout == pto::Layout::NZ && GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ);
    if constexpr (TileData::Loc == pto::TileType::Vec) {
        static_assert(isSameLayout, "Fix: TLOAD(VecTile, GlobalTensor) only support ND2ND/DN2DN/NZ2NZ!");
        TLoadGm2ub<TileData, GlobalData>(
            dst.data(), src.data(), src.GetShape(pto::GlobalTensorDim::DIM_0),
            src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
            src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
            src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
            src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
            src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
    } else if constexpr (TileData::Loc == pto::TileType::Mat) {
        static_assert(
            isSameLayout ||
                (GlobalData::layout == pto::Layout::ND && GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ) ||
                (GlobalData::layout == pto::Layout::DN && GetTileLayoutCustom<TileData>() == TileLayoutCustom::ZN),
            "Fix: TLOAD(MatTile, GlobalTensor) only support ND2ND/DN2DN/NZ2NZ/ND2NZ/DN2ZN!");
        if constexpr (isSameLayout) {
            TLoadGm2L1<TileData, GlobalData>(
                dst.data(), src.data(), src.GetShape(pto::GlobalTensorDim::DIM_0),
                src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
                src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
                src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
                src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
                src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
        } else if constexpr (GlobalData::layout == pto::Layout::ND &&
                             GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ) {
            TLoadGm2L1Nd2nz<TileData, GlobalData>(
                dst.data(), src.data(), src.GetShape(pto::GlobalTensorDim::DIM_0),
                src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
                src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
                src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
                src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
                src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
        } else if constexpr (GlobalData::layout == pto::Layout::DN &&
                             GetTileLayoutCustom<TileData>() == TileLayoutCustom::ZN) {
            TLoadGm2L1Dn2zn<TileData, GlobalData>(
                dst.data(), src.data(), src.GetShape(pto::GlobalTensorDim::DIM_0),
                src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
                src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
                src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
                src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
                src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
        }
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoad5HD(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
                                  int srcN, int srcC1, int srcH, int srcW, int gStride0, int gStride1, int gStride2,
                                  int gStride3, int gStride4, int dstN, int dstC1, int dstH, int dstW)
{
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType *srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;
    constexpr uint32_t maxSupportBurst = 4095;
    // gmGap unit is 32B
    uint32_t gmGap = ((gStride1 - dstH * dstW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    if ((gStride2 == dstW * c0ElemCount || dstH == 1) && // process for W direction all load or H=1
        gmGap <= UINT16_MAX && dstC1 <= maxSupportBurst && dstH * dstW <= UINT16_MAX) {
        uint16_t nBurst = dstC1;
        uint16_t srcGap = gmGap;
        uint16_t lenBurst = dstH * dstW;
        for (uint32_t i = 0; i < dstN; i++) {
            srcAddrP = srcAddr + i * gStride0;
            dstAddrP = dstAddr + i * dstH * dstW * dstC1 * c0ElemCount;
            TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, srcGap, 0);
        }
    } else {
        PTO_ASSERT(dstH <= maxSupportBurst, "Fix: max support dstH is 4095!");
        PTO_ASSERT(dstW <= UINT16_MAX, "Fix: max support dstW is UINT16_MAX!");

        uint16_t nBurst = dstH;
        uint16_t lenBurst = dstW;
        uint16_t srcGap = ((gStride2 - srcW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        uint16_t l1Gap = 0;
        for (uint32_t i = 0; i < dstN; i++) {
            int64_t dstAddr1 = i * dstH * dstW * dstC1 * c0ElemCount;
            int64_t srcAddr1 = i * gStride0;
            for (uint32_t j = 0; j < dstC1; j++) {
                srcAddrP = srcAddr + srcAddr1 + j * gStride1;
                dstAddrP = dstAddr + dstAddr1 + j * dstH * dstW * c0ElemCount;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, srcGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadFractalZ(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
                                       int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4,
                                       int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
                                       int dstShape0, int dstShape1, int dstShape2, int dstShape3)
{
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType *srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;

    if constexpr (TileData::totalDimCount == 4) { // ConvTile layout is [C1HW,N/16,16,C0]
        static_assert(TileData::staticShape[2] == FRACTAL_NZ_ROW && TileData::staticShape[3] == c0ElemCount,
                      "Fix: The TileData last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
        static_assert(GlobalData::staticShape[3] == FRACTAL_NZ_ROW && GlobalData::staticShape[4] == c0ElemCount,
                      "Fix: The GlobalTensor last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");

        uint16_t nBurst = dstShape0;
        uint16_t lenBurst = dstShape1 * dstShape2;
        uint16_t gmGap =
            ((gStride1 - srcShape2 * srcShape3 * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, 0);

    } else { //  [C1,H,W,N,C0]
        PTO_ASSERT(srcShape1 == dstShape1 && srcShape2 == dstShape2,
                   "Fix: layout is Fractal_Z, [srcH,srcW] && [dstH,dstW] should be same!");
        PTO_ASSERT(dstShape3 <= UINT16_MAX, "Fix: max support dstN is UINT16_MAX!");

        uint16_t lenBurst = dstShape3;
        uint16_t gmGap = ((gStride2 - srcShape3 * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        constexpr uint32_t maxSupportBurst = 4095;

        if (dstShape0 * dstShape1 * dstShape2 <= maxSupportBurst) { // if burst <= 4095, only load once
            uint16_t nBurst = dstShape0 * dstShape1 * dstShape2;
            TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, 0);
        } else {
            uint16_t nBurst = dstShape1 * dstShape2;
            for (uint32_t i = 0; i < dstShape0; i++) {
                srcAddrP = srcAddr + i * gStride0;
                dstAddrP = dstAddr + i * dstShape1 * dstShape2 * dstShape3 * c0ElemCount;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, 0);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadNCHW(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
                                   int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4,
                                   int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int dstShape0,
                                   int dstShape1, int dstShape2, int dstShape3)
{
    /* Loading Process: [GM]NCHW -> [UB]NCH_align(W) -> [UB]NCHW -> [UB]NC1HWC0 -> [L1]NC1HWC0 */
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;
    constexpr uint16_t c0ElemCount      = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    constexpr uint16_t ELEMS_PER_BLOCK  = BLOCK_BYTE_SIZE / sizeof(typename GlobalData::DType);
    constexpr uint16_t REPEAT_N_ELEMS   = REPEAT_BYTE / sizeof(typename GlobalData::DType);

    // ConvTile layout is [N,C1,H,W,C0] = [dstShape0, dstShape1, dstShape2, dstShape3, c0ElemCount]
    // GlobalTensor layout is [1,N,C,H,W] = [1, srcShape1, srcShape2, srcShape3, srcShape4]
    PTO_ASSERT(srcShape1 == dstShape0 && srcShape3 == dstShape2 && srcShape4 == dstShape3,
               "Fix: src layout is [1,N,C,H,W],dst layout [N,C1,H,W,C0], srcShape dstShape should be same!");
    PTO_ASSERT(srcShape2 <= dstShape1 * c0ElemCount,
               "Fix: src layout is [1,N,C,H,W],dst layout [N,C1,H,W,C0], srcC should <= dstC1 * dstC0!");

    ///////////////////////////////////
    // AscendC API
    const auto cDstShape1 = dstShape1 * c0ElemCount;
    const uint16_t alignedW = (dstShape3 + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK * ELEMS_PER_BLOCK;
    const uint16_t hwSize = dstShape2 * dstShape3;
    const uint16_t tileNumel = dstShape0 * cDstShape1 * hwSize;
    const uint16_t tileAlignedNumel = dstShape0 * cDstShape1 * dstShape2 * alignedW;

    AscendC::GlobalTensor<typename GlobalData::DType> ascGM;
    ascGM.SetGlobalBuffer(srcAddr);
    AscendC::LocalTensor<typename TileData::DType> ascA1(
        AscendC::TPosition::A1, (uint32_t)((uint64_t)dstAddr), tileNumel);
    AscendC::LocalTensor<typename GlobalData::DType> ascTmpUB(
        AscendC::TPosition::VECIN, 0, (TMP_UB_OFFSET + TMP_UB_SIZE) / sizeof(typename GlobalData::DType));
    auto ascA1UB = ascTmpUB[0], ascUB = ascTmpUB[tileAlignedNumel]; // Both given aligned size.

    // 0.From GM to UB (Copy with W aliged.)
    if (dstShape3 == gStride3 && gStride3 % ELEMS_PER_BLOCK == 0) {
        for (uint32_t i = 0; i < dstShape0; i++) {
            AscendC::DataCopyParams intriParams(
                cDstShape1, hwSize / ELEMS_PER_BLOCK,  (gStride2 - hwSize) / ELEMS_PER_BLOCK, 0);
            AscendC::DataCopy(ascA1UB[i * cDstShape1 * hwSize], ascGM[i * gStride1], intriParams);
        }
    } else {
        for (uint32_t i = 0; i < dstShape0; i++) {
            for (uint32_t j = 0; j < cDstShape1; j++) {
                for (uint32_t k = 0; k < dstShape2; k++) {
                    auto dstOffset = ((i * cDstShape1 + j) * dstShape2 + k) * alignedW;
                    auto srcOffset = i * gStride1 + j * gStride2 + k * gStride3;
                    AscendC::DataCopy(
                        ascA1UB[dstOffset], ascGM[srcOffset], {1, (uint16_t)(alignedW / ELEMS_PER_BLOCK), 0, 0});
                }
            }
        }
    }
    // 1.MEM@UB NCH_alignW -> NC1H_alignWC0
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    AscendC::LocalTensor<uint8_t> ascDummy; // not used
    AscendC::Transpose(ascUB, ascA1UB, ascDummy,
        {(uint16_t)(dstShape0 * dstShape1), (uint16_t)c0ElemCount, (uint16_t)1, (uint16_t)(dstShape2 * alignedW),
        AscendC::TransposeType::TRANSPOSE_NCHW2NHWC});
    // 2.MEM@UB NC1HW_alignWC0 -> NC1HWC0
    pipe_barrier(PIPE_V);
    AscendC::DataCopy(ascA1UB, ascUB, {(uint16_t)(dstShape0 * dstShape1 * c0ElemCount), (uint16_t)dstShape3, (uint16_t)(alignedW - dstShape3), 0});
    // 3.From UB to A1 (Copy as-is.)
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    // AscendC函数会失败，原因是嵌套模板参数认为类型不一致
    copy_ubuf_to_cbuf((__cbuf__ void*)ascA1.GetPhyAddr(), (__ubuf__ void *)(ascA1UB.GetPhyAddr()), 
        0, (uint16_t)1, (uint16_t)((tileNumel + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK), 0, 0);
    // Keep Consistency while from MTE2, to MTE2.
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadNCHW2FractalZ(typename TileData::TileDType __out__ dst,
                                            typename GlobalData::DType __in__ *src, int srcShape0, int srcShape1,
                                            int srcShape2, int srcShape3, int srcShape4, int gStride0, int gStride1,
                                            int gStride2, int gStride3, int gStride4, int dstShape0, int dstShape1,
                                            int dstShape2, int dstShape3)
{
    /*
    kn: align16(O)_align16(C)(HW) -> [transpose]align16(C)_HW_align16(O)-> [transpose5d]C1_HW_align16(O)_C0
     */
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;
    const uint16_t typeSize = sizeof(typename TileData::DType);
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / typeSize;
    typename GlobalData::DType *srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;

    // ConvTile layout is [C1HW,N/16,16,C0] = [dstShape0, dstShape1, dstShape2, dstShape3]
    // GlobalTensor layout is [1,N,C,H,W] = [1, srcShape1, srcShape2, srcShape3, srcShape4]
    PTO_ASSERT(gStride2 == srcShape3 * srcShape4,
               "Fix: src layout is [1,N,C,H,W],dst layout [N,C1,H,W,C0], H*W should be all load");
    PTO_ASSERT(srcShape2 % c0ElemCount == 0, "Fix: expect align 32 bytes for Ci");
    PTO_ASSERT(srcShape1 % c0ElemCount == 0, "Fix: expect align 32 bytes for C0");

    ///////////////////////////////////
    // AscendC API
    const uint32_t dstNumel = TileData::bufferSize / typeSize;
    uint32_t numsOfElemInOneBlk = BLOCK_BYTE_SIZE / typeSize;
    AscendC::GlobalTensor<typename GlobalData::DType> ascGM;
    ascGM.SetGlobalBuffer(srcAddr);
    AscendC::LocalTensor<typename GlobalData::DType> ascTmpUB(AscendC::TPosition::VECOUT, 0, (TMP_UB_OFFSET + TMP_UB_SIZE) / typeSize);  // 总的临时UB的缓存
    auto ascUB0= ascTmpUB[0];  // 用以gm->ub copy, ub->ub,trans5D
    auto ascUB1 = ascTmpUB[dstNumel];  // 用以ub->ub,transpose
    // GM->UB (copy as-is)
    AscendC::DataCopyParams gm2ubParams(1, srcShape2 * gStride2 / numsOfElemInOneBlk, 0, 0);
    for (int i = 0; i < srcShape1; ++i) {
        AscendC::DataCopy(ascUB0[i * srcShape2 * gStride2], ascGM[i * gStride1], gm2ubParams);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    // CoCiHW->CiHWCo (transpose NCHW->NHWC, when N=1,C=Co,HW=CiHW)
    AscendC::LocalTensor<uint8_t> ascDummy; // not used
    AscendC::Transpose(ascUB1, ascUB0, ascDummy,
        {1, (uint16_t)srcShape1, (uint16_t)1, (uint16_t)(srcShape2 * gStride2),
        AscendC::TransposeType::TRANSPOSE_NCHW2NHWC});
    pipe_barrier(PIPE_V);
    // CiHWCo->Ci1HWCo1Co0Ci0 (transpose, NCHW->NHWC, when N=1*Ci1,C=Ci0,HW=HWCo)
    int Ci1 = srcShape2 / c0ElemCount;
    int hwSize = srcShape3 * srcShape4;  // HW->real H, Co->real W
    AscendC::Transpose(ascUB0, ascUB1, ascDummy,
        {(uint16_t)Ci1, (uint16_t)c0ElemCount, (uint16_t)hwSize, (uint16_t)srcShape1,
        AscendC::TransposeType::TRANSPOSE_NCHW2NHWC});
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    // ub->l1, (ci1,HWCo,Ci0)=>(ci1HW,Co1,Co0,Ci0), copy as-is
    copy_ubuf_to_cbuf((__cbuf__ void*)dstAddr, (__ubuf__ void *)(ascUB0.GetPhyAddr()), 0, (uint16_t)(dstShape0 * dstShape1), (uint16_t)(dstShape2 * dstShape3 / numsOfElemInOneBlk), 0, 0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckConvTileData(TileData &dst, GlobalData &src)
{
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, float>,
        "Fix: Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/float!");
    static_assert(TileData::Loc == pto::TileType::Mat, "Fix: Dst TileType must be Mat!");
    static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                  "Fix: Source dtype must be same with dst dtype!");

    constexpr bool isSameLayout =
        (GlobalData::layout == pto::Layout::NC1HWC0 && TileData::layout == pto::Layout::NC1HWC0) ||
        (GlobalData::layout == pto::Layout::FRACTAL_Z && TileData::layout == pto::Layout::FRACTAL_Z) ||
        (GlobalData::layout == pto::Layout::NCHW && TileData::layout == pto::Layout::NC1HWC0) ||
        (GlobalData::layout == pto::Layout::NCHW && TileData::layout == pto::Layout::FRACTAL_Z);
    static_assert(isSameLayout == true, "Fix: Src layout must be NC1HWC0 or FRACTAL_Z or NCHW, and Dst layout must be NC1HWC0 or FRACTAL_Z!");
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_CONVTILE_IMPL(TileData &dst, GlobalData &src)
{
    CheckConvTileData<TileData, GlobalData>(dst, src);
    if constexpr (GlobalData::layout == pto::Layout::NC1HWC0) { // layout is [N,C1,H,W,C0]
        TLoad5HD<TileData, GlobalData>(dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2),
                                       src.GetShape(3), src.GetStride(0), src.GetStride(1), src.GetStride(2),
                                       src.GetStride(3), src.GetStride(4), dst.GetShape(0), dst.GetShape(1),
                                       dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (GlobalData::layout == pto::Layout::FRACTAL_Z) {
        TLoadFractalZ<TileData, GlobalData>(dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2),
                                            src.GetShape(3), src.GetShape(4), src.GetStride(0), src.GetStride(1),
                                            src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
                                            dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (GlobalData::layout == pto::Layout::NCHW &&
                         TileData::layout == pto::Layout::FRACTAL_Z) { // NCHW->[C1HW,N/16,16,C0]
        TLoadNCHW2FractalZ<TileData, GlobalData>(dst.data(), src.data(), src.GetShape(0), src.GetShape(1),
                                                 src.GetShape(2), src.GetShape(3), src.GetShape(4), src.GetStride(0),
                                                 src.GetStride(1), src.GetStride(2), src.GetStride(3), src.GetStride(4),
                                                 dst.GetShape(0), dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (GlobalData::layout == pto::Layout::NCHW &&
                         TileData::layout == pto::Layout::NC1HWC0) { // NCHW->NC1HWC0
        TLoadNCHW<TileData, GlobalData>(dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2),
                                        src.GetShape(3), src.GetShape(4), src.GetStride(0), src.GetStride(1),
                                        src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
                                        dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    } 
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_IMPL(TileData &dst, GlobalData &src)
{
    if constexpr (is_conv_tile_v<TileData>) {
        TLOAD_CONVTILE_IMPL(dst, src);
    } else {
        TLOAD_TILE_IMPL(dst, src);
    }
}
} // namespace pto
#endif // TLOAD_HPP_310P3
