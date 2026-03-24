/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TLOADOP_HPP
#define TLOADOP_HPP

namespace pto {

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadInstrGm2ub(std::vector<CostModelStats> &stats, uint16_t nBurst, uint32_t lenBurst,
                                  uint32_t gmGap, uint32_t ubGap, uint32_t ubPad)
{
    if constexpr (sizeof(typename TileData::DType) == 1) {
        //copy_gm_to_ubuf_align_b8(dst, src, 0, nBurst, lenBurst, 0, ubPad, gmGap, ubGap);
        CostModelStats costModelStats("copy_gm_to_ubuf_align_b8", nBurst, lenBurst, gmGap, ubGap);
        costModelStats.setLeftPaddingNum(0);
        costModelStats.setRightPaddingNum(ubPad);
        stats.emplace_back(costModelStats);
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        //copy_gm_to_ubuf_align_b16(dst, src, 0, nBurst, lenBurst, 0, ubPad, gmGap, ubGap);
        CostModelStats costModelStats("copy_gm_to_ubuf_align_b16", nBurst, lenBurst, gmGap, ubGap);
        costModelStats.setLeftPaddingNum(0);
        costModelStats.setRightPaddingNum(ubPad);
        stats.emplace_back(costModelStats);
    } else if constexpr (sizeof(typename TileData::DType) == 4) {
        //copy_gm_to_ubuf_align_b32(dst, src, 0, nBurst, lenBurst, 0, ubPad, gmGap, ubGap);
        CostModelStats costModelStats("copy_gm_to_ubuf_align_b32", nBurst, lenBurst, gmGap, ubGap);
        costModelStats.setLeftPaddingNum(0);
        costModelStats.setRightPaddingNum(ubPad);
        stats.emplace_back(costModelStats);
    } else if constexpr (sizeof(typename TileData::DType) == 8) {
        //copy_gm_to_ubuf_align_b32(dst, src, 0, nBurst, lenBurst, 0, ubPad * 2, gmGap, ubGap);
        CostModelStats costModelStats("copy_gm_to_ubuf_align_b32", nBurst, lenBurst, gmGap, ubGap);
        costModelStats.setLeftPaddingNum(0);
        costModelStats.setRightPaddingNum(ubPad * 2);
        stats.emplace_back(costModelStats);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadNd2nzInstr(std::vector<CostModelStats> &stats, uint16_t ndNum, uint16_t nValue, uint16_t dValue,
                                  uint16_t srcNdMatrixStride, uint16_t srcDValue, uint16_t dstNzC0Stride,
                                  uint16_t dstNzNStride, uint16_t dstNzMatrixStride)
{
    // Parameter list:
    // stats, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
    // dstNzC0Stride, dstNzNStride, dstNzMatrixStride
    if constexpr (sizeof(typename TileData::DType) == 1) {
        copy_gm_to_cbuf_multi_nd2nz_b8(stats, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                                       dstNzNStride, dstNzMatrixStride);
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        copy_gm_to_cbuf_multi_nd2nz_b16(stats, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                                        dstNzNStride, dstNzMatrixStride);
    } else if constexpr (sizeof(typename TileData::DType) == 4) {
        copy_gm_to_cbuf_multi_nd2nz_b32s(stats, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
                                         dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    } else if constexpr (sizeof(typename TileData::DType) == 8) {
        uint16_t dValueb64 = dValue * 2;
        uint16_t srcDValueb64 = srcDValue * 2;
        copy_gm_to_cbuf_multi_nd2nz_b32s(stats, 0, ndNum, nValue, dValueb64, srcNdMatrixStride, srcDValueb64,
                                         dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadInstrGm2L1(std::vector<CostModelStats> &stats, uint16_t nBurst, uint16_t lenBurst,
                                  uint16_t gmGap, uint16_t l1Gap)
{
    //copy_gm_to_cbuf(dst, src, (uint8_t)0, nBurst, lenBurst, gmGap, l1Gap, (pad_t)0);
    CostModelStats costModelStats("copy_gm_to_cbuf", nBurst, lenBurst, gmGap, l1Gap);
    costModelStats.setPadMode(0);
    stats.emplace_back(costModelStats);
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2ubNd2nd(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                  int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                  int gStride4, int validRow, int validCol)
{
    static_assert(TileData::Rows < 4096, "Fix: TLOAD Rows>=4095 not supported in A2/A3");
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(validRow == gShape0 * gShape1 * gShape2 * gShape3,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    PTO_ASSERT(gShape3 < 4096, "The gshape3 (which equals nBurst) must be less than 4096 for A2/A3");
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
        //set_mov_pad_val(GetPadValue<TileData>());
        // TODO 如何处理？
    }
    int64_t dstStride2 = gShape3 * TileData::Cols;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;

    for (uint32_t i = 0; i < gShape0; i++) {
        for (uint32_t j = 0; j < gShape1; j++) {
            for (uint32_t k = 0; k < gShape2; k++) {
                TLoadInstrGm2ub<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, ubGap, ubPad);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2ubDn2dn(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                  int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                  int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape2 * gShape4,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    PTO_ASSERT(gShape4 < 4096, "The gshape4 (which equals nBurst) must be less than 4096 for A2/A3");
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
        //set_mov_pad_val(GetPadValue<TileData>());
    }

    int64_t dstStride2 = gShape4 * TileData::Rows;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;
    for (uint32_t i = 0; i < gShape0; i++) {
        for (uint32_t j = 0; j < gShape1; j++) {
            for (uint32_t k = 0; k < gShape2; k++) {
                TLoadInstrGm2ub<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, ubGap, ubPad);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2ubNz2nz(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                  int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                  int gStride4, int validRow, int validCol)
{
    uint16_t nBurst = gShape1;
    uint32_t lenBurst = validRow * C0_SIZE_BYTE;
    uint32_t gmGap = (gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType);
    uint32_t ubGap = TileData::Rows - validRow;
    int64_t tileStride = TileData::Rows * gShape1 * gShape4;
    for (uint32_t i = 0; i < gShape0; i++) {
        TLoadInstrGm2ub<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, ubGap, 0);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2ub(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                             int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                             int gStride4, int validRow, int validCol)
{
    if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::ND) {
        TLoadGm2ubNd2nd<TileData, GlobalData>(stats, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                              gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::DN) {
        TLoadGm2ubDn2dn<TileData, GlobalData>(stats, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                              gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ) {
        TLoadGm2ubNz2nz<TileData, GlobalData>(stats, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                              gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Nd2nd(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                  int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                  int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(gShape4 * sizeof(typename TileData::DType) % BLOCK_BYTE_SIZE == 0,
               "The 5th dim of ND shape must be 32 bytes aligned!");
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(validRow == gShape0 * gShape1 * gShape2 * gShape3,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    PTO_ASSERT(gShape3 < 4096, "The gshape3 (which equals nBurst) must be less than 4096 for A2/A3");
    uint16_t nBurst = gShape3;
    uint16_t lenBurst = (validCol * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t gmGap = ((gStride3 - gShape4) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t l1Gap = ((TileData::Cols - validCol) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    int64_t dstStride2 = gShape3 * TileData::Cols;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;

    for (uint32_t i = 0; i < gShape0; i++) {
        for (uint32_t j = 0; j < gShape1; j++) {
            for (uint32_t k = 0; k < gShape2; k++) {
                TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Dn2dn(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                  int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                  int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(gShape3 * sizeof(typename TileData::DType) % BLOCK_BYTE_SIZE == 0,
               "The 4th dim of DN shape must be 32 bytes aligned!");
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape2 * gShape4,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    PTO_ASSERT(gShape4 < 4096, "The gshape4 (which equals nBurst) must be less than 4096 for A2/A3");
    uint16_t nBurst = gShape4;
    uint16_t lenBurst = (validRow * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t gmGap = ((gStride4 - gShape3) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t l1Gap = ((TileData::Rows - gShape3) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    int64_t dstStride2 = gShape4 * TileData::Rows;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;
    for (uint32_t i = 0; i < gShape0; i++) {
        for (uint32_t j = 0; j < gShape1; j++) {
            for (uint32_t k = 0; k < gShape2; k++) {
                TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Nz2nz(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                  int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                  int gStride4, int validRow, int validCol)
{
    uint16_t nBurst = gShape1;
    uint32_t lenBurst = validRow;
    uint32_t gmGap = ((gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint32_t l1Gap = TileData::Rows - validRow;
    int64_t tileStride = TileData::Rows * gShape1 * gShape4;

    for (uint32_t i = 0; i < gShape0; i++) {
        TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, l1Gap);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1VectorInND(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                       int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                       int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(validRow == gShape0 * gShape1 * gShape2 * gShape3,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    static_assert(GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1,
                  "Fix: GlobalTensor ony support 2 dim when using vector input!");
    uint16_t nValue = gShape3;
    uint16_t dValue = gShape4;
    uint16_t srcDValue = gStride3;
    // Parameter list:
    // stats, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
    // dstNzC0Stride, dstNzNStride, dstNzMatrixStride
    TLoadNd2nzInstr<TileData, GlobalData>(stats, 1, nValue, dValue, 0, srcDValue, TileData::Rows, 1, 1);
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1VectorInDn(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                       int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                       int gStride4, int validRow, int validCol)
{
    static_assert(GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1,
                  "Fix: GlobalTensor ony support 2 dim when using vector input!");
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape2 * gShape4,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    uint16_t nValue = gShape4;
    uint16_t dValue = gShape3;
    uint16_t srcDValue = gStride3;
    // Parameter list:
    // dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
    // dstNzC0Stride, dstNzNStride, dstNzMatrixStride
    TLoadNd2nzInstr<TileData, GlobalData>(stats, 1, nValue, dValue, 0, srcDValue, TileData::Cols, 1, 1);
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                             int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                             int gStride4, int validRow, int validCol)
{
    if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::ND) {
        if constexpr (TileData::Rows == 1) {
            TLoadGm2L1VectorInND<TileData, GlobalData>(stats, gShape0, gShape1, gShape2, gShape3, gShape4,
                                                       gStride0, gStride1, gStride2, gStride3, gStride4, validRow,
                                                       validCol);
        } else {
            TLoadGm2L1Nd2nd<TileData, GlobalData>(stats, gShape0, gShape1, gShape2, gShape3, gShape4,
                                                  gStride0, gStride1, gStride2, gStride3, gStride4, validRow, validCol);
        }
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::DN) {
        if constexpr (TileData::Cols == 1) {
            TLoadGm2L1VectorInDn<TileData, GlobalData>(stats, gShape0, gShape1, gShape2, gShape3, gShape4,
                                                       gStride0, gStride1, gStride2, gStride3, gStride4, validRow,
                                                       validCol);
        } else {
            TLoadGm2L1Dn2dn<TileData, GlobalData>(stats, gShape0, gShape1, gShape2, gShape3, gShape4,
                                                  gStride0, gStride1, gStride2, gStride3, gStride4, validRow, validCol);
        }
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ) {
        TLoadGm2L1Nz2nz<TileData, GlobalData>(stats, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                              gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData>
INTERNAL void TLoadGm2L1Nd2nz(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,int gShape3,
                              int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
                              int validRow, int validCol)
{
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
    // stats, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
    // dstNzC0Stride, dstNzNStride, dstNzMatrixStride
    TLoadNd2nzInstr<TileData, GlobalData>(stats, 1, nValue, dValue, 0, srcDValue, TileData::Rows, 1, 1);
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Dn2zn(std::vector<CostModelStats> &stats, int gShape0, int gShape1, int gShape2,
                                  int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                  int gStride4, int validRow, int validCol)
{
    static_assert(GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1,
                  "Fix: GlobalTensor ony support 2 dim when DN2ZN!");
    static_assert(TileData::SFractalSize == 512, "Fix: TileData ony support SFractalSize = 512Bytes!");
    PTO_ASSERT(gShape4 > 0 && gShape4 <= 16384, "The Shape4 of GlobalTensor must be in range of [1, 16384]!");
    PTO_ASSERT(gShape3 > 0 && gShape3 <= 65535, "The Shape3 of GlobalTensor must be must be in range of [1, 65535]!");
    PTO_ASSERT(gStride4 > 0 && gStride4 <= 65535,
               "The Stride3 of GlobalTensor must be must be in range of [1, 65535]!");
    static_assert(TileData::Cols <= 16384, "Fix: The Cols of TileData must be less than 16384!");

    uint16_t nValue = gShape4;
    uint16_t dValue = gShape3;
    uint16_t srcDValue = gStride4;
    TLoadNd2nzInstr<TileData, GlobalData>(stats, 1, nValue, dValue, 0, srcDValue, TileData::Cols, 1, 1);
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_TILE_IMPL(std::vector<CostModelStats> &stats, TileData &dst, GlobalData &src)
{
    constexpr bool isSameLayout =
        (GlobalData::layout == pto::Layout::ND && GetTileLayoutCustom<TileData>() == TileLayoutCustom::ND) ||
        (GlobalData::layout == pto::Layout::DN && GetTileLayoutCustom<TileData>() == TileLayoutCustom::DN) ||
        (GlobalData::layout == pto::Layout::NZ && GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ);
    if constexpr (TileData::Loc == pto::TileType::Vec) {
        static_assert(isSameLayout, "Fix: TLOAD(VecTile, GlobalTensor) only support ND2ND/DN2DN/NZ2NZ!");
        TLoadGm2ub<TileData, GlobalData>(
            stats, src.GetShape(pto::GlobalTensorDim::DIM_0),
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
                stats, src.GetShape(pto::GlobalTensorDim::DIM_0),
                src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
                src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
                src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
                src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
                src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
        } else if constexpr (GlobalData::layout == pto::Layout::ND &&
                             GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ) {
            TLoadGm2L1Nd2nz<TileData, GlobalData>(
                stats, src.GetShape(pto::GlobalTensorDim::DIM_0),
                src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
                src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
                src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
                src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
                src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
        } else if constexpr (GlobalData::layout == pto::Layout::DN &&
                             GetTileLayoutCustom<TileData>() == TileLayoutCustom::ZN) {
            TLoadGm2L1Dn2zn<TileData, GlobalData>(
                stats, src.GetShape(pto::GlobalTensorDim::DIM_0),
                src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
                src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
                src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
                src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
                src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoad5HD(std::vector<CostModelStats> &stats, int srcN, int srcC1, int srcH, int srcW,
                                  int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int dstN,
                                  int dstC1, int dstH, int dstW)
{
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    constexpr uint32_t maxSupportBurst = 4095;
    // gmGap unit is 32B
    uint32_t gmGap = ((gStride1 - dstH * dstW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    if ((gStride2 == dstW * c0ElemCount || dstH == 1) && // process for W direction all load or H=1
        gmGap <= UINT16_MAX && dstC1 <= maxSupportBurst && dstH * dstW <= UINT16_MAX) {
        uint16_t nBurst = dstC1;
        uint16_t srcGap = gmGap;
        uint16_t lenBurst = dstH * dstW;
        for (uint32_t i = 0; i < dstN; i++) {
            TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, srcGap, 0);
        }
    } else {
        PTO_ASSERT(dstH <= maxSupportBurst, "Fix: max support dstH is 4095!");
        PTO_ASSERT(dstW <= UINT16_MAX, "Fix: max support dstW is UINT16_MAX!");

        uint16_t nBurst = dstH;
        uint16_t lenBurst = dstW;
        uint16_t srcGap = ((gStride2 - srcW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        uint16_t l1Gap = 0;
        for (uint32_t i = 0; i < dstN; i++) {
            for (uint32_t j = 0; j < dstC1; j++) {
                TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, srcGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadFractalZ(std::vector<CostModelStats> &stats, int srcShape0, int srcShape1, int srcShape2,
                                int srcShape3, int srcShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                int gStride4, int dstShape0, int dstShape1, int dstShape2, int dstShape3)
{
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);

    if constexpr (TileData::totalDimCount == 4) { // ConvTile layout is [C1HW,N/16,16,C0]
        static_assert(TileData::staticShape[2] == FRACTAL_NZ_ROW && TileData::staticShape[3] == c0ElemCount,
                      "Fix: The TileData last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
        static_assert(GlobalData::staticShape[3] == FRACTAL_NZ_ROW && GlobalData::staticShape[4] == c0ElemCount,
                      "Fix: The GlobalTensor last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");

        uint16_t nBurst = dstShape0;
        uint16_t lenBurst = dstShape1 * dstShape2;
        uint16_t gmGap =
            ((gStride1 - srcShape2 * srcShape3 * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, 0);

    } else { //  [C1,H,W,N,C0]
        PTO_ASSERT(srcShape1 == dstShape1 && srcShape2 == dstShape2,
                   "Fix: layout is Fractal_Z, [srcH,srcW] && [dstH,dstW] should be same!");
        PTO_ASSERT(dstShape3 <= UINT16_MAX, "Fix: max support dstN is UINT16_MAX!");

        uint16_t lenBurst = dstShape3;
        uint16_t gmGap = ((gStride2 - srcShape3 * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        constexpr uint32_t maxSupportBurst = 4095;

        if (dstShape0 * dstShape1 * dstShape2 <= maxSupportBurst) { // if burst <= 4095, only load once
            uint16_t nBurst = dstShape0 * dstShape1 * dstShape2;
            TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, 0);
        } else {
            uint16_t nBurst = dstShape1 * dstShape2;
            for (uint32_t i = 0; i < dstShape0; i++) {
                TLoadInstrGm2L1<TileData, GlobalData>(stats, nBurst, lenBurst, gmGap, 0);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_CONVTILE_IMPL(std::vector<CostModelStats> &stats, TileData &dst, GlobalData &src)
{
    if constexpr (GlobalData::layout == pto::Layout::NC1HWC0) { // layout is NC1HWC0, dst dim4 is c0Size
        TLoad5HD<TileData, GlobalData>(stats, src.GetShape(0), src.GetShape(1), src.GetShape(2),
                                       src.GetShape(3), src.GetStride(0), src.GetStride(1), src.GetStride(2),
                                       src.GetStride(3), src.GetStride(4), dst.GetShape(0), dst.GetShape(1),
                                       dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (GlobalData::layout == pto::Layout::FRACTAL_Z) { // C1HWNC0, dst dim4 is c0Size
        TLoadFractalZ<TileData, GlobalData>(stats, src.GetShape(0), src.GetShape(1), src.GetShape(2),
                                            src.GetShape(3), src.GetShape(4), src.GetStride(0), src.GetStride(1),
                                            src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
                                            dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL std::vector<CostModelStats> runTLoadOp(TileData &dst, GlobalData &src)
{
    std::vector<CostModelStats> stats;
    if constexpr (is_conv_tile_v<TileData>) {
        TLOAD_CONVTILE_IMPL(stats, dst, src);
    } else {
        TLOAD_TILE_IMPL(stats, dst, src);
    }
    return stats;
}
} // namespace pto
#endif