/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSTORE_HPP_A1
#define TSTORE_HPP_A1
#include "common.hpp"
#include "TLoad.hpp"

namespace pto {

template <typename T>
PTO_INTERNAL void SetAtomicAdd();

PTO_INTERNAL void SetAtomicNone();

PTO_INTERNAL AtomicType& static_atomic_var()
{
    static AtomicType atomic_add_ = AtomicType::AtomicNone;
    return atomic_add_;
}

template <typename T>
PTO_INTERNAL void PrepareTailBlock(__ubuf__ T *tail, __ubuf__ T *src, __gm__ T *dst,
    int validRow, int validCol, int srcStride, int dstStride)
{
    constexpr uint32_t ELEMS_PER_BLOCK      = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint32_t ELEMS_PER_I16BLOCK   = BLOCK_BYTE_SIZE / sizeof(int16_t);
    // 构造的 GlobalData 和 TileData 只有类型起作用
    using GlobalData    = GlobalTensor<T, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>>;
    using TileData      = pto::Tile<pto::TileType::Vec, T, 1, ELEMS_PER_BLOCK>;

    uint32_t lenByteBurst   = validCol * sizeof(T);
    uint32_t gmByteGap      = dstStride * sizeof(T) - lenByteBurst;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    TLoadInstrGm2ub<TileData, GlobalData>(tail, dst, validRow, lenByteBurst, 
        gmByteGap, 0, ELEMS_PER_BLOCK - validCol);

    uint32_t validI16Col = (lenByteBurst + sizeof(int16_t) - 1) / sizeof(int16_t);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    set_mask_norm();
    set_vector_mask(0, (1 << ELEMS_PER_I16BLOCK) - 1);
    for (int r = 0; r < validRow; r += REPEAT_MAX) {
        __ubuf__ int16_t *i16Tail = (__ubuf__ int16_t *)(tail + r * ELEMS_PER_BLOCK);
        __ubuf__ int16_t *i16Src = (__ubuf__ int16_t *)(src + r * srcStride);
        uint8_t repeats = min((uint8_t)REPEAT_MAX, (uint8_t)(validRow - r));
        vsub(i16Tail, i16Src, i16Tail, repeats, 0, 0, 0, 1, srcStride / ELEMS_PER_BLOCK, 1);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
    pipe_barrier(PIPE_V);
    TAlign32BPadInstr<T, T>(tail, validRow, validCol, ELEMS_PER_BLOCK, ELEMS_PER_BLOCK - validCol, 0);
}

template <typename T, typename U>
PTO_INTERNAL void CopyUbBlocks2Gm(__gm__ T *dst, __ubuf__ U *src,
    uint32_t validRow, uint32_t validCol, uint32_t srcStride, uint32_t dstStride)
{
    static_assert(std::is_same<T, U>::value, "Types must match");
    constexpr uint32_t ELEMS_PER_BLOCK = BLOCK_BYTE_SIZE / sizeof(U);
    uint16_t lenBurst = validCol / ELEMS_PER_BLOCK;
    if (dstStride % ELEMS_PER_BLOCK == 0) {
        copy_ubuf_to_gm((__gm__ void*)dst, (__ubuf__ void*)src, 0, 
            validRow, lenBurst, srcStride / ELEMS_PER_BLOCK - lenBurst, dstStride / ELEMS_PER_BLOCK - lenBurst);
        return;
    }
    for (int i = 0; i < validRow; ++i) {
        copy_ubuf_to_gm(
            (__gm__ void*)(dst + i * dstStride), (__ubuf__ void*)(src + i * srcStride), 0, 1, lenBurst, 0, 0);
    }
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreUb2gmInstr(typename GlobalData::DType *dst, __ubuf__ typename TileData::DType *src,
                                  uint16_t nBurst, uint32_t lenByteBurst, uint32_t gmByteGap, uint32_t ubGap)
{
    using Tile_DType    = typename TileData::DType;
    using Global_DType  = typename GlobalData::DType;
    constexpr uint32_t ELEMS_PER_BLOCK = BLOCK_BYTE_SIZE / sizeof(Tile_DType);

    uint32_t validCol   = lenByteBurst / sizeof(Tile_DType);
    uint32_t remains    = validCol % ELEMS_PER_BLOCK;
    uint32_t srcStride  = (ubGap + (validCol + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK;
    uint32_t dstStride  = (gmByteGap + lenByteBurst) / sizeof(Tile_DType);

    if (static_atomic_var() == AtomicType::AtomicAdd) {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
        TAlign32BPadInstr<Tile_DType, Tile_DType>(src, nBurst, validCol, srcStride, srcStride - validCol, 0);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        validCol = (remains == 0) ? validCol : (validCol - remains + ELEMS_PER_BLOCK);
        CopyUbBlocks2Gm(dst, src, nBurst, validCol, srcStride, dstStride);
        return;
    }

    auto tail       = (__ubuf__ Tile_DType *)get_imm(TMP_UB_OFFSET);
    auto srcTail    = (__ubuf__ Tile_DType *)(src + validCol - remains);
    auto dstTail    = (__gm__ Global_DType *)(dst + validCol - remains);
    if (remains != 0) {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
        PrepareTailBlock(tail, srcTail, dstTail, nBurst, remains, srcStride, dstStride);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    }
    if (validCol > remains) {
        CopyUbBlocks2Gm(dst, src, nBurst, validCol - remains, srcStride, dstStride);
    }
    if (remains != 0) {
        constexpr uint32_t ELEMS_PER_I16BLOCK      = BLOCK_BYTE_SIZE / sizeof(int16_t);
        uint32_t i16DstStride = (gmByteGap + lenByteBurst) / sizeof(int16_t);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        pipe_barrier(PIPE_MTE3);
        SetAtomicAdd<__gm__ int16_t>();
        CopyUbBlocks2Gm((__gm__ int16_t*)dstTail, (__ubuf__ int16_t*)tail, nBurst, 
            ELEMS_PER_I16BLOCK, ELEMS_PER_I16BLOCK, i16DstStride);
        SetAtomicNone();
    }
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreUb2gmGeneral(typename GlobalData::DType *dstAddr, __ubuf__ typename TileData::DType *srcAddr,
                                int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, 
                                int gStride1, int gStride2, int gStride3, int gStride4, int ubPadShape4)
{
    using Tile_DType    = typename TileData::DType;
    using GM_DType      = typename GlobalData::DType;
    uint16_t nBurst     = gShape3;
    uint32_t lenBurst   = gShape4 * sizeof(Tile_DType);
    uint32_t gmGap      = (gStride3 - gShape4) * sizeof(Tile_DType);
    uint32_t ubGap      = ((ubPadShape4 - gShape4) * sizeof(Tile_DType)) >> SHIFT_BLOCK_BYTE;
    __gm__ GM_DType     *dstGlobalAddr  = dstAddr;
    __ubuf__ Tile_DType *srcTileAddr    = srcAddr;

    int64_t srcStride2 = gShape3 * ubPadShape4;
    int64_t srcStride1 = gShape2 * srcStride2;
    int64_t srcStride0 = gShape1 * srcStride1;
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * gStride0;
        int64_t srcAddr0 = i * srcStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t dstAddr1 = j * gStride1;
            int64_t srcAddr1 = j * srcStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                dstGlobalAddr = dstAddr + dstAddr0 + dstAddr1 + k * gStride2;
                srcTileAddr = srcAddr + srcAddr0 + srcAddr1 + k * srcStride2;
                TStoreUb2gmInstr<GlobalData, TileData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, gmGap, ubGap);
            }
        }
    }
}


template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreUb2gmNd2nd(typename GlobalData::DType *dstAddr, __ubuf__ typename TileData::DType *srcAddr,
                                   int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                   int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(validRow == gShape0 * gShape1 * gShape2 * gShape3,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    PTO_ASSERT(gShape3 < 4096, "The gshape3 (which equals nBurst) must be less than 4096 for A1");
    TStoreUb2gmGeneral<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, 
        gStride0, gStride1, gStride2, gStride3, gStride4, TileData::Cols);
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreUb2gmDn2dn(typename GlobalData::DType *dstAddr, __ubuf__ typename TileData::DType *srcAddr,
                                   int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                   int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape2 * gShape4,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    PTO_ASSERT(gShape4 < 4096, "The gshape4 (which equals nBurst) must be less than 4096 for A1");
    TStoreUb2gmGeneral<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape4, gShape3, 
        gStride0, gStride1, gStride2, gStride4, gStride3, TileData::Rows);
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreUb2gmNz2nz(typename GlobalData::DType *dstAddr, __ubuf__ typename TileData::DType *srcAddr,
                                   int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                   int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    static_assert(GlobalData::staticShape[3] == FRACTAL_NZ_ROW &&
                      GlobalData::staticShape[4] == BLOCK_BYTE_SIZE / sizeof(typename TileData::DType),
                  "When TileData is NZ format, the last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
    PTO_ASSERT(validRow == gShape2 * gShape3, "The validRow of TileData must be equal to Shape2 * Shape3 of NZ shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape4,
               "The validCol of TileData must be equal to Shape0 * Shape1 * Shape4 of NZ shape!");
    PTO_ASSERT(gShape1 < 4096, "The gshape1 (which equals nBurst) must be less than 4096 for A1");
    constexpr uint32_t C0_ELEMS = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    uint32_t newGShape4         = validRow * C0_ELEMS;
    uint32_t ubPadShape4        = TileData::Rows * C0_ELEMS;
    TStoreUb2gmGeneral<GlobalData, TileData>(dstAddr, srcAddr, 1, 1, gShape0, gShape1, newGShape4, 
        gStride0, gStride0, gStride0, gStride1, gStride4, ubPadShape4);    
}

template <typename GlobalData, typename TileData, AtomicType currentAtomicType = AtomicType::AtomicNone>
__tf__ PTO_INTERNAL void TStore(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
                                int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __ubuf__ typename TileData::DType *srcAddr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src);
    typename GlobalData::DType *dstAddr = dst;

    if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        TStoreUb2gmNd2nd<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                               gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        TStoreUb2gmDn2dn<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                               gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
        TStoreUb2gmNz2nz<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                               gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    }
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreMat2GmInstr(typename GlobalData::DType *dstAddr, __cbuf__ typename TileData::DType *srcAddr,
                                    int gShape0, int gShape1, int gShape2, int gStride0, int gStride1, int gStride2,
                                    int64_t srcStride2, uint16_t nBurst, uint16_t lenBurst, uint16_t dstStride,
                                    uint16_t srcStride)
{
    PTO_ASSERT(false, "The cce api copy_cbuf_to_gm is unsupported on A1.");
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreMat2GmNd2Nd(typename GlobalData::DType *dstAddr, __cbuf__ typename TileData::DType *srcAddr,
                                    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                    int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(false, "TStoreMat2GmNd2Nd is unsupported on A1.");
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreMat2GmDn2Dn(typename GlobalData::DType *dstAddr, __cbuf__ typename TileData::DType *srcAddr,
                                    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                    int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(false, "TStoreMat2GmDn2Dn is unsupported on A1.");
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreMat2GmNz2Nz(typename GlobalData::DType *dstAddr, __cbuf__ typename TileData::DType *srcAddr,
                                    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                    int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(false, "TStoreMat2GmNz2Nz is unsupported on A1.");
}

template <typename GlobalData, typename TileData, AtomicType atomicType = AtomicType::AtomicNone>
__tf__ AICORE void TStoreMat(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
                             int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                             int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __cbuf__ typename TileData::DType *srcAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(src);
    typename GlobalData::DType *dstAddr = dst;
    PTO_ASSERT(false, "There is no data path from L1 to GM on the A1.");

    if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        TStoreMat2GmNd2Nd<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                                gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        TStoreMat2GmDn2Dn<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                                gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
        TStoreMat2GmNz2Nz<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                                gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    }
}

template <typename T>
PTO_INTERNAL void SetAtomicAdd()
{
    static_assert((std::is_same<T, __gm__ half>::value) || (std::is_same<T, __gm__ float>::value) ||
                      (std::is_same<T, __gm__ int16_t>::value) || (std::is_same<T, __gm__ int8_t>::value),
                  "Dst and src must be half / float / int16_t / int8_t.");
    static_atomic_var() = AtomicType::AtomicAdd;
    if constexpr (std::is_same<T, __gm__ float>::value) {
        set_atomic_f32();
    } else if constexpr (std::is_same<T, __gm__ half>::value) {
        set_atomic_f16();
    } else if constexpr (std::is_same<T, __gm__ int16_t>::value) {
        set_atomic_s16();
    } else if constexpr (std::is_same<T, __gm__ int8_t>::value) {
        set_atomic_s8();
    }
}

PTO_INTERNAL void SetAtomicNone()
{
    static_atomic_var() = AtomicType::AtomicNone;
    set_atomic_none();
}

template <typename GlobalData, typename TileData, QuantMode_t quantizationMode = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TStoreAccNz2nd(typename GlobalData::DType *dstAddr, __cc__ typename TileData::DType *srcAddr,
                                 int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                 int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(gShape0 == 1 && gShape1 == 1 && gShape2 == 1, "NZ2ND only supports 2D-to-2D conversions.");
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(validRow == gShape3, "The validRow of TileData must be equal to Shape3 of ND shape!");
    PTO_ASSERT(validRow >= 1 && validRow <= 8192, "When GlobalData is ND format, the range of validRow is [1, 8192].");
    uint16_t mSize = validRow;
    uint16_t nSize = validCol;

    uint16_t srcStride = TileData::Rows;
    uint32_t dstD = gStride3;

    uint16_t ndNum = validCol / gShape4;
    constexpr uint8_t c0 = 16;
    uint16_t srcNdStride = TileData::Rows * c0 * gShape4;
    if constexpr (TileData::Compact == CompactMode::Normal) {
        srcStride = (validRow + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
        srcNdStride = srcStride * gShape4 * c0;
    }

    uint16_t dstNdStride = gStride2;

    constexpr uint8_t nz2ndEn = 1;
    constexpr uint8_t unitFlagCtrl = static_cast<uint8_t>(Phase);

    uint64_t xmReg =
        ((nSize & 0xfff) << 4) |                          // Xm[15:4] the n-direction size of the matrix
        (static_cast<uint64_t>(mSize & 0xffff) << 16) |   // Xm[31:16] the m-direction size of the matrix
        (static_cast<uint64_t>(dstD & 0xffffffff) << 32); // Xm[63:32] destination stride between the start addr
    uint64_t xtReg = srcStride |                          // Xt[15:0] the source stride between the start addr
                     (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) |      // Xt[33:32] unit flag control bit
                     (static_cast<uint64_t>(quantizationMode & 0x1f) << 34) | // Xt[38:34] pre-stage quantization mode
                     ((static_cast<uint64_t>(reluPreMode) & 0x7) << 39) |     //  Xt[41:39] relu pre mode
                     (static_cast<uint64_t>(nz2ndEn & 0x1) << 43);            // Xt[43] nz2nd control bit
    uint64_t ndParaSPR = 0;
    ndParaSPR = ndNum |                                               // ND_PARA[15:0] the number of source nd
                (static_cast<uint64_t>(srcNdStride & 0xffff) << 16) | // ND_PARA[31:16] the stride of source nd
                (static_cast<uint64_t>(dstNdStride & 0xffff) << 32);  // ND_PARA[47:32] the stride of destination nd
    set_nd_para(ndParaSPR);
    copy_matrix_cc_to_gm(dstAddr, srcAddr, xmReg, xtReg);
}

template <typename GlobalData, typename TileData, QuantMode_t quantizationMode = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TStoreAccNz2nz(typename GlobalData::DType *dstAddr, __cc__ typename TileData::DType *srcAddr,
                                 int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                 int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape2 * gShape3, "The validRow of TileData must be equal to Shape2 * Shape3 of NZ shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape4,
               "The validCol of TileData must be equal to Shape0 * Shape1 * Shape4 of NZ shape!");
    PTO_ASSERT(
        validRow >= 1 && validRow <= 65535 && validCol % 16 == 0,
        "When GlobalData is NZ format, the range of validRow is [1, 65535] and validCol must be an integer multiple of "
        "16.");

    static_assert(GlobalData::staticShape[3] == FRACTAL_NZ_ROW,
                  "When GlobalData is NZ format, the second-to-last dimension shall be 16.");
    static_assert(
        (std::is_same_v<typename GlobalData::DType, __gm__ float> &&
         (GlobalData::staticShape[4] == 8 || GlobalData::staticShape[4] == 16)) ||
            (std::is_same_v<typename GlobalData::DType, __gm__ int32_t> && GlobalData::staticShape[4] == 16) ||
            (GlobalData::staticShape[4] == BLOCK_BYTE_SIZE / sizeof(typename GlobalData::DType)),
        "When GlobalData is in NZ format: if DstType is float, the last dimension must be either 8 or 16, "
        "and the dimension value is 8 if and only if Channel Split is enabled; if DstType is int32_t, the "
        "last dimension must be exactly 16. In addition, the last dimension must be static and satisfy 32 / "
        "sizeof(DstType).");

    uint16_t mSize = validRow;
    uint16_t nSize = validCol;

    uint32_t c0Size = sizeof(typename GlobalData::DType) * gShape4;
    uint16_t srcStride = TileData::Rows;
    if constexpr (CompactMode::Normal == TileData::Compact) {
        srcStride = (FRACTAL_NZ_ROW + validRow - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
    }
    uint32_t dstStride = gShape2 * gShape3 * c0Size >> SHIFT_BLOCK_BYTE;

    constexpr uint8_t unitFlagCtrl = static_cast<uint8_t>(Phase);
    uint8_t channelSplitEn = 0;
    if (std::is_same_v<typename TileData::DType, float> && std::is_same_v<typename GlobalData::DType, __gm__ float>) {
        if (gShape4 == 8) {
            channelSplitEn = 1;
        }
    }

    uint64_t xmReg =

        (static_cast<uint64_t>(nSize & 0xfff) << 4) |          // Xm[15:4] nSize
        (static_cast<uint64_t>(mSize & 0xffff) << 16) |        // Xm[31:16] mSize
        (static_cast<uint64_t>(dstStride & 0xffffffff) << 32); // Xm[63:32] destination stride between the start addr

    uint64_t xtReg = srcStride | // Xt[15:0] the source stride between the start addr
                     (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) |      // Xt[33:32] unit flag control bit
                     (static_cast<uint64_t>(quantizationMode & 0x1f) << 34) | // Xt[38:34] pre-stage quantization mode
                     ((static_cast<uint64_t>(reluPreMode) & 0x7) << 39) |     //  Xt[41:39] relu pre mode
                     (static_cast<uint64_t>(channelSplitEn & 0x1) << 42);     // Xt[42] channel split control bit

    copy_matrix_cc_to_gm(dstAddr, srcAddr, xmReg, xtReg);
}

template <typename GlobalData, typename TileData, QuantMode_t quantizationMode = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TStoreAccNz2NC1HWC0(typename GlobalData::DType *dstAddr, __cc__ typename TileData::DType *srcAddr,
                                      int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride1,
                                      int gStride3, int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape0 * gShape2 * gShape3,
               "The validRow of TileData must be equal to gShape0 * Shape2 * Shape3 of NC1HWC0 shape!");
    PTO_ASSERT(validCol == gShape1 * gShape4,
               "The validCol of TileData must be equal to Shape1 * Shape4 of NC1HWC0 shape!");
    PTO_ASSERT(validRow >= 1 && validRow <= 65535,
               "When GlobalData is NC1HWC0 format, the range of validRow is [1, 65535].");

    static_assert(
        (std::is_same_v<typename GlobalData::DType, __gm__ float> &&
         (GlobalData::staticShape[4] == 8 || GlobalData::staticShape[4] == 16)) ||
            (std::is_same_v<typename GlobalData::DType, __gm__ int32_t> && GlobalData::staticShape[4] == 16) ||
            (GlobalData::staticShape[4] == BLOCK_BYTE_SIZE / sizeof(typename GlobalData::DType)),
        "When GlobalData is in NC1HWC0 format: if DstType is float, the last dimension must be either 8 or 16, "
        "and the dimension value is 8 if and only if Channel Split is enabled; if DstType is int32_t, the "
        "last dimension must be exactly 16. In addition, the last dimension must be static and satisfy 32 / "
        "sizeof(DstType).");
    uint8_t channelSplitEn = 0;
    if (std::is_same_v<typename TileData::DType, float> && std::is_same_v<typename GlobalData::DType, __gm__ float>) {
        if (gShape4 == 8) {
            channelSplitEn = 1;
        }
    }

    uint16_t mSize = validRow;
    uint16_t nSize = validCol;
    uint32_t c0Size = sizeof(typename GlobalData::DType) * gShape4;
    uint16_t srcStride = TileData::Rows;
    if constexpr (CompactMode::Normal == TileData::Compact) {
        srcStride = CeilAlignment(validRow, FRACTAL_NZ_ROW);
    }
    uint32_t dstStride = gShape0 * gStride1 / gStride3 * c0Size >> SHIFT_BLOCK_BYTE;
    constexpr uint8_t unitFlagCtrl = static_cast<uint8_t>(Phase);
    uint64_t xmReg = (static_cast<uint64_t>(nSize & 0xfff) << 4) | (static_cast<uint64_t>(mSize & 0xffff) << 16) |
                     (static_cast<uint64_t>(dstStride & 0xffffffff) << 32);
    uint64_t xtReg = srcStride | (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) |
                     (static_cast<uint64_t>(quantizationMode & 0x1f) << 34) |
                     ((static_cast<uint64_t>(reluPreMode) & 0x7) << 39) |
                     (static_cast<uint64_t>(channelSplitEn & 0x1) << 42);

    copy_matrix_cc_to_gm(dstAddr, srcAddr, xmReg, xtReg);
}

template <typename GlobalData, typename TileData, QuantMode_t quantPre = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TStoreAccNCHW(typename GlobalData::DType *dstAddr, __cc__ typename TileData::DType *srcAddr,
                                int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride1,
                                int gStride2, int validRow, int validCol)
{
// note: W must not be tiled because only CStride given.
    if constexpr (GlobalData::layout == pto::Layout::NCHW) {
        PTO_ASSERT(validRow == gShape1 * gShape3 * gShape4,
                   "The validRow of TileData must be equal to Shape1 * Shape3 * Shape4 of NCHW shape!");
        PTO_ASSERT(validCol == gShape2, "The validCol of TileData must be equal to Shape2 of NCHW shape!");
    } else { // NCDHW
        PTO_ASSERT(validRow == gShape0 * gShape3 * gShape4,
                   "The validRow of TileData must be equal to Shape0 * Shape3 * Shape4 of NCDHW shape!");
        PTO_ASSERT(validCol == gShape1, "The validCol of TileData must be equal to Shape1 of NCDHW shape!");
    }
    using Tile_DType    = typename TileData::DType;
    using Global_DType  = typename GlobalData::DType;

    uint16_t mSize = validRow;
    uint16_t nSize = validCol;
    uint16_t whSize = mSize / gShape1;
    uint16_t alignWH = TileData::Rows / gShape1;
    constexpr uint16_t DSTTYPE_BYTESIZE = sizeof(Global_DType);
    constexpr uint16_t ELEMS_PER_BLOCK = BLOCK_BYTE_SIZE / DSTTYPE_BYTESIZE;
    uint32_t dstStride = (GlobalData::layout == pto::Layout::NCDHW) ? gStride1 : gStride2;

    ///////////////////////////////////
    // AscendC API
    AscendC::GlobalTensor<Global_DType> ascGM;
    ascGM.SetGlobalBuffer(dstAddr);
    AscendC::LocalTensor<Tile_DType> ascCO1(
        AscendC::TPosition::CO1, (uint32_t)((uint64_t)srcAddr), TileData::Numel);
    AscendC::LocalTensor<Global_DType> ascTmpUB(
        AscendC::TPosition::VECOUT, 0, (TMP_UB_OFFSET + TMP_UB_SIZE) / DSTTYPE_BYTESIZE);
    auto ascUB1= ascTmpUB[0], ascUB2 = ascTmpUB[TileData::Numel];
    AscendC::LocalTensor<uint32_t> offsetUB(
        AscendC::TPosition::VECOUT, 2 * TileData::Numel * DSTTYPE_BYTESIZE, ELEMS_PER_BLOCK);
    // 0.From L0C to UB (Copy as-is.)
    AscendC::DataCopyParams intriParams(1, TileData::Numel / FRACTAL_NZ_ROW / ACC_C0_SIZE, 0, 0);
    AscendC::DataCopyEnhancedParams enhancedParams;
    enhancedParams.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
    AscendC::DataCopy(ascUB1 , ascCO1, intriParams, enhancedParams);
    // 1.MEM@UB NZ->ND(NHWC)(Here HW is align32 bytes.)
    pipe_barrier(PIPE_V);
    constexpr uint16_t C0_N_ELEMS = ACC_C0_SIZE;
    for (int i = 0; i < TileData::Cols / C0_N_ELEMS; i++) {
        AscendC::DataCopy(
            ascUB2[C0_N_ELEMS * i], ascUB1[C0_N_ELEMS * TileData::Rows * i], 
            {TileData::Rows, 1, 0, TileData::Cols / C0_N_ELEMS - 1});
    }
    // 2.MEM@UB NHWC->NCHW(HW aligned)
    pipe_barrier(PIPE_V);     // why not pipe_v barrier
    AscendC::LocalTensor<uint8_t> ascDummy; // not used
    AscendC::Transpose(ascUB1, ascUB2, ascDummy, {
        (uint16_t)gShape1, (uint16_t)TileData::Cols, 1, TileData::Rows, AscendC::TransposeType::TRANSPOSE_NHWC2NCHW});
    // 3.From UB to GM (Copy with unAligned HW)
    using UBTileData = 
        pto::Tile<pto::TileType::Vec, typename decltype(ascUB1)::PrimType, TileData::Rows, TileData::Cols>;
    uint32_t lenByteBurst   = whSize * sizeof(Global_DType);
    uint32_t gmByteGap      = dstStride * sizeof(Global_DType) - lenByteBurst;
    uint32_t ubGap          = (alignWH - whSize) * sizeof(Global_DType) >> SHIFT_BLOCK_BYTE;
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    TStoreUb2gmInstr<GlobalData, UBTileData>(
        ascGM.GetPhyAddr(0), (__ubuf__ typename decltype(ascUB1)::PrimType *)ascUB1.GetPhyAddr(), 
        nSize, lenByteBurst, gmByteGap, ubGap);
}

template <typename GlobalData, typename TileData, QuantMode_t quantizationMode = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
__tf__ AICORE void TStoreAcc(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
                             int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                             int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __cc__ typename TileData::DType *srcAddr = (__cc__ typename TileData::DType *)__cce_get_tile_ptr(src);
    typename GlobalData::DType *dstAddr = dst;

    if constexpr (GlobalData::layout == pto::Layout::ND) {
        TStoreAccNz2nd<GlobalData, TileData, quantizationMode, reluPreMode, Phase>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
            gStride4, validRow, validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::NZ) {
        TStoreAccNz2nz<GlobalData, TileData, quantizationMode, reluPreMode, Phase>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
            gStride4, validRow, validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::NC1HWC0) {
        TStoreAccNz2NC1HWC0<GlobalData, TileData, quantizationMode, reluPreMode, Phase>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride1, gStride3, validRow, validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::NCHW || GlobalData::layout == pto::Layout::NCDHW) {
        TStoreAccNCHW<GlobalData, TileData, quantizationMode, reluPreMode, Phase>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride1, gStride2, validRow, validCol);
    }
}

template <typename GlobalData, typename TileData, typename FpTileData,
          QuantMode_t quantizationMode = QuantMode_t::NoQuant, ReluPreMode reluPreMode = ReluPreMode::NoRelu>
__tf__ AICORE void TStoreAccFp(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
                               typename FpTileData::TileDType __in__ fp, int gShape0, int gShape1, int gShape2,
                               int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                               int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(false, "TStoreAccFp is not supported.");
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckStaticForVecAndMat()
{
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, int64_t> || std::is_same_v<typename TileData::DType, uint64_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, float>,
        "Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/int64_t/uint64_t/half/float!");
    static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                  "Source dtype must be same with dst dtype!");
    static_assert(((GlobalData::layout == pto::Layout::ND) &&
                   (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                      ((GlobalData::layout == pto::Layout::DN) &&
                       (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                      ((GlobalData::layout == pto::Layout::NZ) &&
                       (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) ||
                      (TileData::Rows == 1) || (TileData::Cols == 1),
                  "Src and dst layout must be same, only support ND/DN/NZ or the special case of one row/one column!");
    if constexpr (std::is_same_v<typename TileData::DType, int64_t> ||
                  std::is_same_v<typename TileData::DType, uint64_t>) {
        static_assert((GlobalData::layout == pto::Layout::ND &&
                       (TileData::isRowMajor && TileData::SFractal == SLayout::NoneBox)) ||
                          (GlobalData::layout == pto::Layout::DN &&
                           (!TileData::isRowMajor && TileData::SFractal == SLayout::NoneBox)),
                      "TSTORE(GlobalTensor, VecTile) only support ND2ND/DN2DN for b64!");
    }
}

template <typename TileData, typename GlobalData, bool isQuant>
PTO_INTERNAL void CheckAcc2gm(GlobalData &dst, TileData &src)
{
    static_assert((GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::NZ ||
                   (GlobalData::layout == pto::Layout::NCHW) || GlobalData::layout == pto::Layout::NC1HWC0),
                  "The output data layout must be ND, NZ, NCHW or NC1HWC0.");
    static_assert(std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, float>,
                  "The input data type must be restricted to int32_t/float!");
    if constexpr (!isQuant) {
        static_assert(std::is_same_v<typename GlobalData::DType, __gm__ int32_t> ||
                          std::is_same_v<typename GlobalData::DType, __gm__ float> ||
                          std::is_same_v<typename GlobalData::DType, __gm__ half>,
                      "The output data type must be restricted to int32_t/float/half!");
    } else if constexpr (isQuant) {
        if constexpr (std::is_same_v<typename TileData::DType, float>) {
            // std::is_same<typename GlobalData::DType, __gm__ half>::value---TODO: 可能有问题
            static_assert(std::is_same<typename GlobalData::DType, __gm__ int8_t>::value ||
                              std::is_same<typename GlobalData::DType, __gm__ uint8_t>::value,
                          "The output data type must be restricted to int8_t/uint8_t.");
        } else if constexpr (std::is_same_v<typename TileData::DType, __gm__ int32_t>) {
            static_assert(std::is_same<typename GlobalData::DType, __gm__ int8_t>::value ||
                              std::is_same<typename GlobalData::DType, __gm__ uint8_t>::value ||
                              std::is_same<typename GlobalData::DType, __gm__ half>::value,
                          "The output data type must be restricted to half/int8_t/uint8_t.");
        }
    }
    static_assert(TileData::Cols >= 1 && TileData::Cols <= 4095, "The range of Cols is [1, 4095].");
    static_assert((GlobalData::layout == pto::Layout::ND && TileData::Rows >= 1 && TileData::Rows <= 8192) ||
                      ((GlobalData::layout == pto::Layout::NZ || GlobalData::layout == pto::Layout::NC1HWC0 ||
                      (GlobalData::layout == pto::Layout::NCHW)) &&
                       TileData::Rows >= 1 && TileData::Rows <= 65535 && TileData::Cols % 16 == 0),
                  "When GlobalData is ND format, the range of Rows is [1, 8192]."
                  "When GlobalData is NZ/NC1HWC0/NCHW format, the range of Rows is [1, 65535] and Cols "
                  "must be an integer multiple of 16.");
    PTO_ASSERT(src.GetValidCol() >= 1 && src.GetValidCol() <= 4095, "The range of validCol is [1, 4095].");
    PTO_ASSERT(dst.GetShape(pto::GlobalTensorDim::DIM_0) > 0 && dst.GetShape(pto::GlobalTensorDim::DIM_1) > 0 &&
                   dst.GetShape(pto::GlobalTensorDim::DIM_2) > 0 && dst.GetShape(pto::GlobalTensorDim::DIM_3) > 0 &&
                   dst.GetShape(pto::GlobalTensorDim::DIM_4) > 0 && src.GetValidRow() > 0 && src.GetValidCol() > 0,
               "The shape of src and dst must be greater than 0!");
}

template <typename TileData, typename GlobalData, AtomicType currentAtomicType = AtomicType::AtomicNone,
          STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src)
{
    static_assert(TileData::Loc == pto::TileType::Vec || TileData::Loc == pto::TileType::Acc ||
                      TileData::Loc == pto::TileType::Mat,
                  "Source TileType only suport Vec/Acc/Mat!");
    PTO_ASSERT(dst.GetShape(pto::GlobalTensorDim::DIM_0) > 0 && dst.GetShape(pto::GlobalTensorDim::DIM_1) > 0 &&
                   dst.GetShape(pto::GlobalTensorDim::DIM_2) > 0 && dst.GetShape(pto::GlobalTensorDim::DIM_3) > 0 &&
                   dst.GetShape(pto::GlobalTensorDim::DIM_4) > 0 && src.GetValidRow() > 0 && src.GetValidCol() > 0,
               "The shape of src and dst must be greater than 0!");
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<typename GlobalData::DType>();
    }
    if constexpr (TileData::Loc == pto::TileType::Vec) {
        CheckStaticForVecAndMat<TileData, GlobalData>();
        TStore<GlobalData, TileData>(
            dst.data(), src.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0),
            dst.GetShape(pto::GlobalTensorDim::DIM_1), dst.GetShape(pto::GlobalTensorDim::DIM_2),
            dst.GetShape(pto::GlobalTensorDim::DIM_3), dst.GetShape(pto::GlobalTensorDim::DIM_4),
            dst.GetStride(pto::GlobalTensorDim::DIM_0), dst.GetStride(pto::GlobalTensorDim::DIM_1),
            dst.GetStride(pto::GlobalTensorDim::DIM_2), dst.GetStride(pto::GlobalTensorDim::DIM_3),
            dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(), src.GetValidCol());
    } else if constexpr (TileData::Loc == pto::TileType::Acc) {
        CheckAcc2gm<TileData, GlobalData, false>(dst, src);
        constexpr QuantMode_t quantMode = GetCastPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
        TStoreAcc<GlobalData, TileData, quantMode, ReluPreMode::NoRelu, Phase>(
            dst.data(), src.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0),
            dst.GetShape(pto::GlobalTensorDim::DIM_1), dst.GetShape(pto::GlobalTensorDim::DIM_2),
            dst.GetShape(pto::GlobalTensorDim::DIM_3), dst.GetShape(pto::GlobalTensorDim::DIM_4),
            dst.GetStride(pto::GlobalTensorDim::DIM_0), dst.GetStride(pto::GlobalTensorDim::DIM_1),
            dst.GetStride(pto::GlobalTensorDim::DIM_2), dst.GetStride(pto::GlobalTensorDim::DIM_3),
            dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(), src.GetValidCol());
    } else if constexpr (TileData::Loc == pto::TileType::Mat) {
        CheckStaticForVecAndMat<TileData, GlobalData>();
        TStoreMat<GlobalData, TileData>(
            dst.data(), src.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0),
            dst.GetShape(pto::GlobalTensorDim::DIM_1), dst.GetShape(pto::GlobalTensorDim::DIM_2),
            dst.GetShape(pto::GlobalTensorDim::DIM_3), dst.GetShape(pto::GlobalTensorDim::DIM_4),
            dst.GetStride(pto::GlobalTensorDim::DIM_0), dst.GetStride(pto::GlobalTensorDim::DIM_1),
            dst.GetStride(pto::GlobalTensorDim::DIM_2), dst.GetStride(pto::GlobalTensorDim::DIM_3),
            dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(), src.GetValidCol());
    }
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicNone();
    }
}

template <typename TileData, typename GlobalData, AtomicType currentAtomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src)
{
    static_assert(TileData::Loc == pto::TileType::Acc, "Source TileType only suport Acc!");
    CheckAcc2gm<TileData, GlobalData, false>(dst, src);
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<typename GlobalData::DType>();
    }
    constexpr QuantMode_t quantMode = GetCastPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    TStoreAcc<GlobalData, TileData, quantMode, reluPreMode, Phase>(
        dst.data(), src.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0), dst.GetShape(pto::GlobalTensorDim::DIM_1),
        dst.GetShape(pto::GlobalTensorDim::DIM_2), dst.GetShape(pto::GlobalTensorDim::DIM_3),
        dst.GetShape(pto::GlobalTensorDim::DIM_4), dst.GetStride(pto::GlobalTensorDim::DIM_0),
        dst.GetStride(pto::GlobalTensorDim::DIM_1), dst.GetStride(pto::GlobalTensorDim::DIM_2),
        dst.GetStride(pto::GlobalTensorDim::DIM_3), dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(),
        src.GetValidCol());
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicNone();
    }
}

template <typename TileData, typename GlobalData, AtomicType currentAtomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src, uint64_t preQuantScalar)
{
    static_assert(TileData::Loc == pto::TileType::Acc, "Source TileType only suport Acc!");
    CheckAcc2gm<TileData, GlobalData, true>(dst, src);
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<typename GlobalData::DType>();
    }
    set_quant_pre(preQuantScalar);
    constexpr QuantMode_t quantMode = GetScalarPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    TStoreAcc<GlobalData, TileData, quantMode, reluPreMode, Phase>(
        dst.data(), src.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0), dst.GetShape(pto::GlobalTensorDim::DIM_1),
        dst.GetShape(pto::GlobalTensorDim::DIM_2), dst.GetShape(pto::GlobalTensorDim::DIM_3),
        dst.GetShape(pto::GlobalTensorDim::DIM_4), dst.GetStride(pto::GlobalTensorDim::DIM_0),
        dst.GetStride(pto::GlobalTensorDim::DIM_1), dst.GetStride(pto::GlobalTensorDim::DIM_2),
        dst.GetStride(pto::GlobalTensorDim::DIM_3), dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(),
        src.GetValidCol());
    if constexpr (AtomicType::AtomicAdd == currentAtomicType) {
        SetAtomicNone();
    }
}

template <typename TileData, typename GlobalData, typename FpTileData,
          AtomicType currentAtomicType = AtomicType::AtomicNone, ReluPreMode reluPreMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src, FpTileData &fp)
{
    static_assert(TileData::Loc == pto::TileType::Acc, "Source TileType only suport Acc!");
    CheckAcc2gm<TileData, GlobalData, true>(dst, src);
    if constexpr (AtomicType::AtomicAdd == currentAtomicType) {
        SetAtomicAdd<typename GlobalData::DType>();
    }
    constexpr QuantMode_t quantMode = GetVectorPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    TStoreAccFp<GlobalData, TileData, FpTileData, quantMode, reluPreMode>(
        dst.data(), src.data(), fp.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0),
        dst.GetShape(pto::GlobalTensorDim::DIM_1), dst.GetShape(pto::GlobalTensorDim::DIM_2),
        dst.GetShape(pto::GlobalTensorDim::DIM_3), dst.GetShape(pto::GlobalTensorDim::DIM_4),
        dst.GetStride(pto::GlobalTensorDim::DIM_0), dst.GetStride(pto::GlobalTensorDim::DIM_1),
        dst.GetStride(pto::GlobalTensorDim::DIM_2), dst.GetStride(pto::GlobalTensorDim::DIM_3),
        dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(), src.GetValidCol());
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicNone();
    }
}
} // namespace pto
#endif
