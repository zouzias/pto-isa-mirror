/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifdef PTO_NPU_ARCH_KIRINX90
#include "pto/npu/a2a3/TStore.hpp"
#elif defined(PTO_NPU_ARCH_KIRIN9030)
#ifndef TSTORE_HPP
#define TSTORE_HPP
#include "common.hpp"

namespace pto {
template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetCastPreQuantModeGm()
{
    return QuantMode_t::NoQuant;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetScalarPreQuantModeGm()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (std::is_same<SrcType, int32_t>::value) {
        if constexpr ((std::is_same<DstType, __gm__ int8_t>::value) || (std::is_same<DstType, __gm__ uint8_t>::value)) {
            quantPre = QuantMode_t::REQ8;
        } else if constexpr (std::is_same<DstType, __gm__ half>::value) {
            quantPre = QuantMode_t::DEQF16;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetVectorPreQuantModeGm()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (std::is_same<SrcType, int32_t>::value) {
        if constexpr ((std::is_same<DstType, __gm__ int8_t>::value) || (std::is_same<DstType, __gm__ uint8_t>::value)) {
            quantPre = QuantMode_t::VREQ8;
        } else if constexpr (std::is_same<DstType, __gm__ half>::value) {
            quantPre = QuantMode_t::VDEQF16;
        }
    }
    return quantPre;
}

template <typename T>
PTO_INTERNAL void SetAtomicAdd()
{
    static_assert((std::is_same_v<T, __gm__ int8_t>) || (std::is_same_v<T, __gm__ int16_t>) ||
                      (std::is_same_v<T, __gm__ half>) || (std::is_same_v<T, __gm__ int32_t>) ||
                      (std::is_same_v<T, __gm__ float>),
                  "Dst and src must be half / float / int16_t / int32_t / int8_t.");
    atomic_type_t atomicType = atomic_type_t::ATOMIC_NONE;
    if constexpr (std::is_same_v<T, __gm__ int8_t>) {
        set_atomic_s8();
    } else if (std::is_same_v<T, __gm__ int16_t>) {
        set_atomic_s16();
    } else if (std::is_same_v<T, __gm__ half>) {
        set_atomic_f16();
    } else if (std::is_same_v<T, __gm__ int32_t>) {
        set_atomic_s32();
    } else if (std::is_same_v<T, __gm__ float>) {
        set_atomic_f32();
    }
    set_atomic_add();
}

template <typename SrcTile, typename DstGlobal, bool isQuant>
PTO_INTERNAL void CheckStaticAcc()
{
    static_assert(std::is_same_v<typename SrcTile::DType, int32_t> || std::is_same_v<typename SrcTile::DType, half>,
                  "The input data type must be restricted to int32_t/half!");
    static_assert((DstGlobal::layout == pto::Layout::ND) || (DstGlobal::layout == pto::Layout::NZ),
                  "TSTORE(Acc2GM) only support NZ2ND / NZ2NZ.");
    static_assert(SrcTile::Cols >= 1 && SrcTile::Cols <= 4095, "The range of Cols is [1, 4095].");
    static_assert((DstGlobal::layout == pto::Layout::ND && SrcTile::Rows >= 1 && SrcTile::Rows <= 8192) ||
                      (DstGlobal::layout == pto::Layout::NZ && SrcTile::Rows >= 1 && SrcTile::Rows <= 65535 &&
                       SrcTile::Cols % 16 == 0),
                  "When DstGlobal is ND format, the range of Rows is [1, 8192]."
                  "When DstGlobal is NZ format, the range of Rows is [1, 65535] and Cols"
                  "must be an integer multiple of 16.");
    if constexpr (!isQuant) {
        static_assert(std::is_same_v<typename DstGlobal::DType, __gm__ int32_t> ||
                          std::is_same_v<typename DstGlobal::DType, __gm__ float> ||
                          std::is_same_v<typename DstGlobal::DType, __gm__ half>,
                      "The output data type must be restricted to int32_t/float/half!");
    } else if constexpr (isQuant) {
        if constexpr (std::is_same_v<typename SrcTile::DType, float>) {
            static_assert(std::is_same<typename DstGlobal::DType, __gm__ int8_t>::value ||
                              std::is_same<typename DstGlobal::DType, __gm__ uint8_t>::value ||
                              std::is_same<typename DstGlobal::DType, __gm__ half>::value ||
                              std::is_same<typename DstGlobal::DType, __gm__ float>::value,
                          "The output data type must be restricted to int8_t/uint8_t/half/float.");
        } else if constexpr (std::is_same_v<typename SrcTile::DType, __gm__ int32_t>) {
            static_assert(std::is_same<typename DstGlobal::DType, __gm__ int8_t>::value ||
                              std::is_same<typename DstGlobal::DType, __gm__ uint8_t>::value ||
                              std::is_same<typename DstGlobal::DType, __gm__ half>::value,
                          "The output data type must be restricted to half/int8_t/uint8_t.");
        }
    }
}

template <typename SrcTile, typename DstGlobal>
PTO_INTERNAL void CheckStaticVec()
{
    static_assert(sizeof(typename SrcTile::DType) == sizeof(typename DstGlobal::DType),
                  "Source dtype must be same with dst dtype!");
    static_assert(
        std::is_same_v<typename SrcTile::DType, int8_t> || std::is_same_v<typename SrcTile::DType, uint8_t> ||
            std::is_same_v<typename SrcTile::DType, int16_t> || std::is_same_v<typename SrcTile::DType, uint16_t> ||
            std::is_same_v<typename SrcTile::DType, int32_t> || std::is_same_v<typename SrcTile::DType, uint32_t> ||
            std::is_same_v<typename SrcTile::DType, int64_t> || std::is_same_v<typename SrcTile::DType, uint64_t> ||
            std::is_same_v<typename SrcTile::DType, half> || std::is_same_v<typename SrcTile::DType, float>,
        "Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/int64_t/uint64_t/half/float!");
    static_assert(
        ((DstGlobal::layout == pto::Layout::ND) && (SrcTile::isRowMajor && (SrcTile::SFractal == SLayout::NoneBox))) ||
            ((DstGlobal::layout == pto::Layout::DN) &&
             (!SrcTile::isRowMajor && (SrcTile::SFractal == SLayout::NoneBox))) ||
            ((DstGlobal::layout == pto::Layout::NZ) &&
             (!SrcTile::isRowMajor && (SrcTile::SFractal == SLayout::RowMajor))) ||
            (SrcTile::Rows == 1) || (SrcTile::Cols == 1),
        "Src and dst layout must be same, only support ND/DN/NZ or the special case of one row/one column!");
    static_assert(
        ((DstGlobal::layout == pto::Layout::ND) && (SrcTile::Cols * sizeof(typename SrcTile::DType) % 32 == 0)) ||
        ((DstGlobal::layout == pto::Layout::DN) && (SrcTile::Rows * sizeof(typename SrcTile::DType) % 32 == 0)) ||
        (DstGlobal::layout == pto::Layout::NZ) ||
        ((DstGlobal::layout == pto::Layout::ND) && (SrcTile::Rows * sizeof(typename SrcTile::DType) % 32 == 0) &&
         (SrcTile::Cols == 1)) ||
        ((DstGlobal::layout == pto::Layout::DN) && (SrcTile::Cols * sizeof(typename SrcTile::DType) % 32 == 0) &&
         (SrcTile::Rows == 1)));
}

template <typename DstGlobal, typename SrcTile, QuantMode_t quantPre = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TStoreAccND(typename DstGlobal::DType *dstGlobalAddr, __cc__ typename SrcTile::DType *srcTileAddr,
                              int gShape3, int gShape4, int gStride2, int gStride3, int validRow, int validCol)
{
    uint16_t mSize = validRow;
    uint16_t nSize = validCol;

    uint16_t srcStride = SrcTile::Rows;
    uint32_t dstD = gStride3;

    uint16_t ndNum = validCol / gShape4;
    constexpr uint16_t c0 = 16;
    uint16_t srcNdStride = SrcTile::Rows * gShape4 * c0;
    if constexpr (SrcTile::Compact == CompactMode::Normal) {
        srcStride = (validRow + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
        srcNdStride = srcStride * gShape4 * c0;
    }
    constexpr uint8_t unitFlagCtrl = static_cast<uint8_t>(Phase);
    constexpr uint8_t nz2ndEn = 1;
    uint16_t dstNdStride = gStride2;

    uint64_t xtReg = srcStride | // Xt[15:0] the source stride between the start addr
                     (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) | // Xt[33:32] unit flag control bit
                     (((quantPre >> SHIFT_BLOCK_BYTE) & 0x1) << 29) |
                     (static_cast<uint64_t>(quantPre & 0x1f) << 34) | // Xt[29], Xt[38:34] pre-stage quantization mode
                     ((static_cast<uint64_t>(reluPreMode) & 0x7) << 39) | //  Xt[41:39] relu pre mode
                     (static_cast<uint64_t>(nz2ndEn & 0x1) << 43);        //  Xt[43] nz2nd control bit
    uint64_t xmReg =
        ((nSize & 0xfff) << 4) |                          // Xm[15:4] the n-direction size of the matrix
        (static_cast<uint64_t>(mSize & 0xffff) << 16) |   // Xm[31:16] the m-direction size of the matrix
        (static_cast<uint64_t>(dstD & 0xffffffff) << 32); // Xm[63:32] destination stride between the start addr
    uint64_t config =
        ndNum |                                               // ND_PARA[15:0] the number of source nd
        (static_cast<uint64_t>(srcNdStride & 0xffff) << 16) | // ND_PARA[31:16] the stride of source nd
        (static_cast<uint64_t>(dstNdStride & 0xffff) << 32);  // ND_PARA[47:32] the stride of destination nd
    set_loop3_para(config);
    copy_matrix_cc_to_gm(dstGlobalAddr, srcTileAddr, xmReg, xtReg);
}

template <typename DstGlobal, typename SrcTile, QuantMode_t quantPre = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TStoreAccNZ(typename DstGlobal::DType *dstAddr, __cc__ typename SrcTile::DType *srcAddr,
                              typename DstGlobal::DType *dstGlobalAddr, __cc__ typename SrcTile::DType *srcTileAddr,
                              int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                              int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape2 * gShape3, "The validRow of SrcTile must be equal to Shape2 * Shape3 of NZ shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape4,
               "The validCol of SrcTile must be equal to Shape0 * Shape1 * Shape4 of NZ shape!");
    static_assert(DstGlobal::staticShape[3] == FRACTAL_NZ_ROW,
                  "When DstGlobal is NZ format, the second-to-last dimension shall be 16.");
    static_assert((std::is_same_v<typename DstGlobal::DType, __gm__ int32_t> && DstGlobal::staticShape[4] == 16) ||
                      (DstGlobal::staticShape[4] == BLOCK_BYTE_SIZE / sizeof(typename DstGlobal::DType)) ||
                      (std::is_same_v<typename DstGlobal::DType, __gm__ float> &&
                       (DstGlobal::staticShape[4] == 8 || DstGlobal::staticShape[4] == 16)),
                  "When DstGlobal is in NZ format: if DstType is float, the last dimension must be either 8 or 16, "
                  "and the dimension value is 8 if and only if Channel Split is enabled; if DstType is int32_t, the "
                  "last dimension must be exactly 16. In addition, the last dimension must be static and satisfy 32 / "
                  "sizeof(DstType).");

    uint16_t mSize = validRow;
    uint16_t nSize = validCol;
    uint16_t srcStride = SrcTile::Rows;
    if constexpr (CompactMode::Normal == SrcTile::Compact) {
        srcStride = (FRACTAL_NZ_ROW - 1 + validRow) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
    }
    constexpr uint8_t unitFlagCtrl = static_cast<uint8_t>(Phase);
    uint8_t channelSplitEn = 0;

    uint16_t c0Size = 16;
    if constexpr (sizeof(typename SrcTile::DType) == 1) {
        c0Size = 32;
    } else if constexpr (std::is_same_v<typename SrcTile::DType, float> &&
                         std::is_same_v<typename DstGlobal::DType, __gm__ float>) {
        if (gShape4 == 8) {
            c0Size = 8;
            channelSplitEn = 1;
        }
    }
    uint32_t dstStride = (gShape2 * gShape3 + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW * c0Size;
    if constexpr (sizeof(typename DstGlobal::DType) == 1) {
        dstStride <<= 1;
    }
    uint64_t xmReg =
        ((static_cast<uint64_t>(nSize & 0xfff) << 4) |           // Xm[15:4] the n-direction size of the matrix
         (static_cast<uint64_t>(mSize & 0xffff) << 16) |         // Xm[31:16] the m-direction size of the matrix
         (static_cast<uint64_t>(dstStride & 0xffffffff) << 32)); // Xm[63:32] destination stride between the start addr
    uint64_t xtReg = srcStride |                                 // Xt[15:0] the source stride between the start addr
                     (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) | // Xt[33:32] unit flag control bit
                     (((quantPre >> SHIFT_BLOCK_BYTE) & 0x1) << 29) |
                     (static_cast<uint64_t>(quantPre & 0x1f) << 34) | // Xt[29], Xt[38:34] pre-stage quantization mode
                     ((static_cast<uint64_t>(reluPreMode) & 0x7) << 39) | //  Xt[41:39] relu pre mode
                     (static_cast<uint64_t>(channelSplitEn & 0x1) << 42); // Xt[42] channel split control bit
    copy_matrix_cc_to_gm(dstAddr, srcAddr, xmReg, xtReg);
}

template <typename DstGlobal, typename SrcTile, typename FpTileData, QuantMode_t quantPre = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu>
__tf__ AICORE void TStoreAccFp(typename DstGlobal::DType __out__ *dst, typename SrcTile::TileDType __in__ src,
                               typename FpTileData::TileDType __in__ fp, int gShape0, int gShape1, int gShape2,
                               int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                               int gStride4, int validRow, int validCol)
{
    __cc__ typename SrcTile::DType *srcAddr = (__cc__ typename SrcTile::DType *)__cce_get_tile_ptr(src);
    typename DstGlobal::DType *dstAddr = dst;
    __fbuf__ typename FpTileData::DType *dstAddrFp = (__fbuf__ typename FpTileData::DType *)__cce_get_tile_ptr(fp);
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
    if constexpr (DstGlobal::layout == pto::Layout::NZ) {
        __cc__ typename SrcTile::DType *srcTileAddr = srcAddr;
        typename DstGlobal::DType *dstGlobalAddr = dstAddr;
        TStoreAccNZ<DstGlobal, SrcTile, quantPre, reluPreMode>(dstAddr, srcAddr, dstGlobalAddr, srcTileAddr, gShape0,
                                                               gShape1, gShape2, gShape3, gShape4, gStride0, validRow,
                                                               validCol);
    } else if constexpr (DstGlobal::layout == pto::Layout::ND) {
        TStoreAccND<DstGlobal, SrcTile, quantPre, reluPreMode>(dstAddr, srcAddr, gShape3, gShape4, gStride2, gStride3,
                                                               validRow, validCol);
    }
}

template <typename DstGlobal, typename SrcTile, QuantMode_t quantPre = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
__tf__ AICORE void TStoreAcc(typename DstGlobal::DType __out__ *dst, typename SrcTile::TileDType __in__ src,
                             int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                             int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __cc__ typename SrcTile::DType *srcAddr = (__cc__ typename SrcTile::DType *)__cce_get_tile_ptr(src);
    typename DstGlobal::DType *dstAddr = dst;
    if constexpr (DstGlobal::layout == pto::Layout::ND) {
        TStoreAccND<DstGlobal, SrcTile, quantPre, reluPreMode, Phase>(dstAddr, srcAddr, gShape3, gShape4, gStride2,
                                                                      gStride3, validRow, validCol);
    } else if constexpr (DstGlobal::layout == pto::Layout::NZ) {
        __cc__ typename SrcTile::DType *srcTileAddr = srcAddr;
        typename DstGlobal::DType *dstGlobalAddr = dstAddr;
        TStoreAccNZ<DstGlobal, SrcTile, quantPre, reluPreMode, Phase>(dstAddr, srcAddr, dstGlobalAddr, srcTileAddr,
                                                                      gShape0, gShape1, gShape2, gShape3, gShape4,
                                                                      gStride0, validRow, validCol);
    }
}

template <typename SrcTile, typename DstGlobal>
PTO_INTERNAL void TStoreInstr(typename DstGlobal::DType *dst, __ubuf__ typename SrcTile::DType *src, uint32_t nBurst,
                              uint32_t lenBurst, uint64_t burstDstStride, uint32_t burstSrcStride)
{
    copy_ubuf_to_gm_align_v2(dst, src, 0, nBurst, lenBurst, burstDstStride, burstSrcStride);
}

template <typename DstGlobal, typename SrcTile>
PTO_INTERNAL void TStoreVecND(typename DstGlobal::DType *dstAddr, __ubuf__ typename SrcTile::DType *srcAddr,
                              int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                              int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    typename DstGlobal::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename SrcTile::DType *srcTileAddr = srcAddr;
    uint32_t loop1SrcStride = GetByteSize<typename SrcTile::DType>(gShape3 * SrcTile::Cols);
    uint32_t loop1DstStride = GetByteSize<typename SrcTile::DType>(gStride2);
    uint32_t loop2SrcStride = GetByteSize<typename SrcTile::DType>(gShape2 * gShape3 * SrcTile::Cols);
    uint32_t loop2DstStride = GetByteSize<typename SrcTile::DType>(gStride1);

    uint64_t loopSizeConfig = (gShape2 & 0x1FFFFF) | ((static_cast<uint64_t>(gShape1) & 0x3FFFFF) << 21);
    set_loop_size_ubtoout(loopSizeConfig);

    uint64_t loop1Config = ((uint64_t)loop1DstStride) | (((uint64_t)loop1SrcStride) << 40);
    set_loop1_stride_ubtoout(loop1Config);

    uint64_t loop2Config = ((uint64_t)loop2DstStride) | (((uint64_t)loop2SrcStride) << 40);
    set_loop2_stride_ubtoout(loop2Config);

    uint32_t nBurst = gShape3;
    uint32_t lenBurst = GetByteSize<typename SrcTile::DType>(validCol);
    uint64_t srcStride0 = gShape1 * gShape2 * gShape3 * SrcTile::Cols;
    uint64_t burstDstStride = GetByteSize<typename SrcTile::DType>(gStride3);
    uint32_t burstSrcStride = GetByteSize<typename SrcTile::DType>(SrcTile::Cols);
    for (uint32_t k = 0; k < gShape0; k++) {
        dstGlobalAddr = dstAddr + k * gStride0;
        srcTileAddr = srcAddr + k * srcStride0;
        TStoreInstr<SrcTile, DstGlobal>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, burstDstStride, burstSrcStride);
    }
}
template <typename DstGlobal, typename SrcTile>
PTO_INTERNAL void TStoreVecDN(typename DstGlobal::DType *dstAddr, __ubuf__ typename SrcTile::DType *srcAddr,
                              int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                              int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    typename DstGlobal::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename SrcTile::DType *srcTileAddr = srcAddr;
    uint32_t loop1SrcStride = GetByteSize<typename SrcTile::DType>(SrcTile::Rows * gShape4);
    uint32_t loop1DstStride = GetByteSize<typename SrcTile::DType>(gStride2);
    uint32_t loop2SrcStride = GetByteSize<typename SrcTile::DType>(gShape2 * SrcTile::Rows * gShape4);
    uint32_t loop2DstStride = GetByteSize<typename SrcTile::DType>(gStride1);

    uint64_t loop1Config = ((uint64_t)loop1DstStride) | (((uint64_t)loop1SrcStride) << 40);
    set_loop1_stride_ubtoout(loop1Config);

    uint64_t loop2Config = ((uint64_t)loop2DstStride) | (((uint64_t)loop2SrcStride) << 40);
    set_loop2_stride_ubtoout(loop2Config);

    uint64_t loopSizeConfig = (gShape2 & 0x1FFFFF) | ((static_cast<uint64_t>(gShape1) & 0x3FFFFF) << 21);
    set_loop_size_ubtoout(loopSizeConfig);

    uint64_t srcStride0 = gShape1 * gShape2 * gShape4 * SrcTile::Rows;
    uint32_t nBurst = gShape4;
    uint32_t lenBurst = GetByteSize<typename SrcTile::DType>(validRow);
    uint64_t burstDstStride = GetByteSize<typename SrcTile::DType>(gStride4);
    uint32_t burstSrcStride = GetByteSize<typename SrcTile::DType>(SrcTile::Rows);
    for (uint32_t k = 0; k < gShape0; k++) {
        dstGlobalAddr = dstAddr + k * gStride0;
        srcTileAddr = srcAddr + k * srcStride0;
        TStoreInstr<SrcTile, DstGlobal>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, burstDstStride, burstSrcStride);
    }
}

template <typename DstGlobal, typename SrcTile>
PTO_INTERNAL void TStoreVecNZ(typename DstGlobal::DType *dstAddr, __ubuf__ typename SrcTile::DType *srcAddr,
                              int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                              int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    static_assert(
        (std::is_same_v<typename DstGlobal::DType, __gm__ int32_t> && DstGlobal::staticShape[4] == 16) ||
            (DstGlobal::staticShape[4] == BLOCK_BYTE_SIZE / sizeof(typename DstGlobal::DType)) ||
            (std::is_same_v<typename DstGlobal::DType, __gm__ float> &&
             (DstGlobal::staticShape[4] == 8 || DstGlobal::staticShape[4] == 16)),
        "When DstGlobal is in NZ format: if DstType is float, the last dimension must be either 8 or 16, \n"
        "and the dimension value is 8 if and only if Channel Split is enabled; if DstType is int32_t, the \n"
        "last dimension must be exactly 16. In addition, the last dimension must be static and satisfy 32 / \n"
        "sizeof(DstType).");
    static_assert(DstGlobal::staticShape[3] == FRACTAL_NZ_ROW,
                  "When DstGlobal is NZ format, the second-to-last dimension shall be 16.");
    PTO_ASSERT(validRow == gShape2 * gShape3, "The validRow of SrcTile must be equal to Shape2 * Shape3 of NZ shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape4,
               "The validCol of SrcTile must be equal to Shape0 * Shape1 * Shape4 of NZ shape!");
    typename DstGlobal::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename SrcTile::DType *srcTileAddr = srcAddr;
    uint32_t nBurst = gShape1;
    uint32_t lenBurst = validRow * C0_SIZE_BYTE;
    uint64_t burstDstStride = GetByteSize<typename SrcTile::DType>(gStride1);
    uint32_t burstSrcStride = SrcTile::Rows * C0_SIZE_BYTE;
    int64_t tileStride = gShape1 * SrcTile::Rows * gShape4;
    for (uint32_t k = 0; k < gShape0; k++) {
        dstGlobalAddr = dstAddr + k * gStride0;
        srcTileAddr = srcAddr + k * tileStride;
        TStoreInstr<SrcTile, DstGlobal>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, burstDstStride, burstSrcStride);
    }
}
template <typename DstGlobal, typename SrcTile>
__tf__ AICORE OP_NAME(TSTORE)
    OP_TYPE(memory) void TStore(typename DstGlobal::DType __out__ *dst, typename SrcTile::TileDType __in__ src,
                                int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __ubuf__ typename SrcTile::DType *srcAddr = (__ubuf__ typename SrcTile::DType *)__cce_get_tile_ptr(src);
    typename DstGlobal::DType *dstAddr = dst;

    if constexpr (SrcTile::isRowMajor & (SrcTile::SFractal == SLayout::NoneBox)) {
        TStoreVecND<DstGlobal, SrcTile>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                        gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (!SrcTile::isRowMajor & (SrcTile::SFractal == SLayout::NoneBox)) {
        TStoreVecDN<DstGlobal, SrcTile>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                        gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (!SrcTile::isRowMajor & (SrcTile::SFractal == SLayout::RowMajor)) {
        TStoreVecNZ<DstGlobal, SrcTile>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                        gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    }
}

template <typename SrcTile, typename DstGlobal, AtomicType atomicType = AtomicType::AtomicNone,
          STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(DstGlobal &dst, SrcTile &src)
{
    static_assert(SrcTile::Loc == pto::TileType::Vec || SrcTile::Loc == pto::TileType::Acc,
                  "Source TileType only suport Vec/Acc!");

    if constexpr (atomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<typename DstGlobal::DType>();
    }

    if constexpr (SrcTile::Loc == pto::TileType::Acc) {
        using L0cT = typename SrcTile::DType;
        using DstT = typename DstGlobal::DType;
        CheckStaticAcc<SrcTile, DstGlobal, false>();

        constexpr QuantMode_t quantPre = GetCastPreQuantModeGm<L0cT, DstT>();
        TStoreAcc<DstGlobal, SrcTile, quantPre, ReluPreMode::NoRelu, Phase>(
            dst.data(), src.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0),
            dst.GetShape(pto::GlobalTensorDim::DIM_1), dst.GetShape(pto::GlobalTensorDim::DIM_2),
            dst.GetShape(pto::GlobalTensorDim::DIM_3), dst.GetShape(pto::GlobalTensorDim::DIM_4),
            dst.GetStride(pto::GlobalTensorDim::DIM_0), dst.GetStride(pto::GlobalTensorDim::DIM_1),
            dst.GetStride(pto::GlobalTensorDim::DIM_2), dst.GetStride(pto::GlobalTensorDim::DIM_3),
            dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(), src.GetValidCol());
    } else if constexpr (SrcTile::Loc == pto::TileType::Vec) {
        CheckStaticVec<SrcTile, DstGlobal>();
        TStore<DstGlobal, SrcTile>(
            dst.data(), src.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0),
            dst.GetShape(pto::GlobalTensorDim::DIM_1), dst.GetShape(pto::GlobalTensorDim::DIM_2),
            dst.GetShape(pto::GlobalTensorDim::DIM_3), dst.GetShape(pto::GlobalTensorDim::DIM_4),
            dst.GetStride(pto::GlobalTensorDim::DIM_0), dst.GetStride(pto::GlobalTensorDim::DIM_1),
            dst.GetStride(pto::GlobalTensorDim::DIM_2), dst.GetStride(pto::GlobalTensorDim::DIM_3),
            dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(), src.GetValidCol());
    }

    if constexpr (atomicType == AtomicType::AtomicAdd) {
        set_atomic_none();
    }
}

template <typename SrcTile, typename DstGlobal, AtomicType atomicType = AtomicType::AtomicNone, ReluPreMode reluPreMode,
          STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(DstGlobal &dst, SrcTile &src)
{
    static_assert(SrcTile::Loc == pto::TileType::Acc, "Source TileType only suport Acc!");
    using L0cT = typename SrcTile::DType;
    using DstT = typename DstGlobal::DType;
    CheckStaticAcc<SrcTile, DstGlobal, false>();

    if constexpr (atomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<DstT>();
    }

    constexpr QuantMode_t quantPre = GetCastPreQuantModeGm<L0cT, DstT>();
    TStoreAcc<DstGlobal, SrcTile, quantPre, reluPreMode, Phase>(
        dst.data(), src.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0), dst.GetShape(pto::GlobalTensorDim::DIM_1),
        dst.GetShape(pto::GlobalTensorDim::DIM_2), dst.GetShape(pto::GlobalTensorDim::DIM_3),
        dst.GetShape(pto::GlobalTensorDim::DIM_4), dst.GetStride(pto::GlobalTensorDim::DIM_0),
        dst.GetStride(pto::GlobalTensorDim::DIM_1), dst.GetStride(pto::GlobalTensorDim::DIM_2),
        dst.GetStride(pto::GlobalTensorDim::DIM_3), dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(),
        src.GetValidCol());

    if constexpr (atomicType == AtomicType::AtomicAdd) {
        set_atomic_none();
    }
}

template <typename SrcTile, typename DstGlobal, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(DstGlobal &dst, SrcTile &src, uint64_t preQuantScalar)
{
    static_assert(SrcTile::Loc == pto::TileType::Acc, "Source TileType only suport Acc!");

    using L0cT = typename SrcTile::DType;
    using DstT = typename DstGlobal::DType;
    CheckStaticAcc<SrcTile, DstGlobal, true>();

    if constexpr (atomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<DstT>();
    }

    constexpr QuantMode_t quantPre = GetScalarPreQuantModeGm<L0cT, DstT>();
    set_quant_pre(preQuantScalar);
    TStoreAcc<DstGlobal, SrcTile, quantPre, reluPreMode, Phase>(
        dst.data(), src.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0), dst.GetShape(pto::GlobalTensorDim::DIM_1),
        dst.GetShape(pto::GlobalTensorDim::DIM_2), dst.GetShape(pto::GlobalTensorDim::DIM_3),
        dst.GetShape(pto::GlobalTensorDim::DIM_4), dst.GetStride(pto::GlobalTensorDim::DIM_0),
        dst.GetStride(pto::GlobalTensorDim::DIM_1), dst.GetStride(pto::GlobalTensorDim::DIM_2),
        dst.GetStride(pto::GlobalTensorDim::DIM_3), dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(),
        src.GetValidCol());

    if constexpr (AtomicType::AtomicAdd == atomicType) {
        set_atomic_none();
    }
}

template <typename SrcTile, typename DstGlobal, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(DstGlobal &dst, SrcTile &src, FpTileData &fp)
{
    static_assert(SrcTile::Loc == pto::TileType::Acc, "Source TileType only suport Acc!");
    using DstT = typename DstGlobal::DType;
    using L0cT = typename SrcTile::DType;
    CheckStaticAcc<SrcTile, DstGlobal, true>();

    if constexpr (AtomicType::AtomicAdd == atomicType) {
        SetAtomicAdd<DstT>();
    }

    constexpr QuantMode_t quantPre = GetVectorPreQuantModeGm<L0cT, DstT>();
    TStoreAccFp<DstGlobal, SrcTile, FpTileData, quantPre, reluPreMode>(
        dst.data(), src.data(), fp.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0),
        dst.GetShape(pto::GlobalTensorDim::DIM_1), dst.GetShape(pto::GlobalTensorDim::DIM_2),
        dst.GetShape(pto::GlobalTensorDim::DIM_3), dst.GetShape(pto::GlobalTensorDim::DIM_4),
        dst.GetStride(pto::GlobalTensorDim::DIM_0), dst.GetStride(pto::GlobalTensorDim::DIM_1),
        dst.GetStride(pto::GlobalTensorDim::DIM_2), dst.GetStride(pto::GlobalTensorDim::DIM_3),
        dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(), src.GetValidCol());

    if constexpr (atomicType == AtomicType::AtomicAdd) {
        set_atomic_none();
    }
}
} // namespace pto
#endif // TSTORE_HPP
#endif // PTO_NPU_ARCH_KIRIN9030