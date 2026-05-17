/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MGATHER_HPP
#define MGATHER_HPP

#include <pto/common/utils.hpp>
#include <pto/common/constants.hpp>
#include "common.hpp"

namespace pto {

enum class GatherOOB : uint8_t
{
    Undefined = 0,
    Clamp = 1,
    Wrap = 2,
    Zero = 3
};

#ifndef PTO_COALESCE_ENUM_DEFINED
#define PTO_COALESCE_ENUM_DEFINED
enum class Coalesce : uint8_t
{
    Row = 0,
    Elem = 1
};
#endif

template <typename T>
struct IsValidMGatherDType {
    static constexpr bool value = std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
                                  std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
                                  std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                                  std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float>;
};

template <typename Tile>
struct IsMGatherNDTile {
    static constexpr bool value = Tile::isRowMajor && (Tile::SFractal == SLayout::NoneBox);
};

template <typename Tile>
struct IsMGatherNZTile {
    static constexpr bool value =
        !Tile::isRowMajor && (Tile::SFractal == SLayout::RowMajor) && (Tile::SFractalSize == TileConfig::fractalABSize);
};

template <GatherOOB Oob>
AICORE PTO_INLINE uint32_t mgather_remap(uint32_t idx, uint32_t cap, uint32_t &doRead)
{
    if constexpr (Oob == GatherOOB::Undefined) {
        doRead = 1u;
        return idx;
    } else if constexpr (Oob == GatherOOB::Clamp) {
        doRead = 1u;
        return (idx >= cap) ? (cap - 1u) : idx;
    } else if constexpr (Oob == GatherOOB::Wrap) {
        doRead = 1u;
        return idx % cap;
    } else {
        doRead = (idx < cap) ? 1u : 0u;
        return idx;
    }
}

template <typename T>
AICORE PTO_INLINE void MGatherRowDma(__ubuf__ T *dst, __gm__ T *src, uint32_t lenBytes)
{
    if constexpr (sizeof(T) == 1) {
        copy_gm_to_ubuf_align_b8(dst, src, 0, 1, lenBytes, 0, 0, 0, 0);
    } else if constexpr (sizeof(T) == 2) {
        copy_gm_to_ubuf_align_b16(dst, src, 0, 1, lenBytes, 0, 0, 0, 0);
    } else if constexpr (sizeof(T) == 4) {
        copy_gm_to_ubuf_align_b32(dst, src, 0, 1, lenBytes, 0, 0, 0, 0);
    }
}

template <typename T>
AICORE PTO_INLINE void MGatherRowMultiDma(__ubuf__ T *dst, __gm__ T *src, uint16_t nBurst, uint32_t lenBytes,
                                          uint32_t gmGapBytes, uint32_t ubGapBlocks)
{
    if constexpr (sizeof(T) == 1) {
        copy_gm_to_ubuf_align_b8(dst, src, 0, nBurst, lenBytes, 0, 0, gmGapBytes, ubGapBlocks);
    } else if constexpr (sizeof(T) == 2) {
        copy_gm_to_ubuf_align_b16(dst, src, 0, nBurst, lenBytes, 0, 0, gmGapBytes, ubGapBlocks);
    } else if constexpr (sizeof(T) == 4) {
        copy_gm_to_ubuf_align_b32(dst, src, 0, nBurst, lenBytes, 0, 0, gmGapBytes, ubGapBlocks);
    }
}

template <typename T>
AICORE PTO_INLINE void MGatherElemDma(__ubuf__ T *dst, __gm__ T *src)
{
    if constexpr (sizeof(T) == 1) {
        copy_gm_to_ubuf_align_b8(dst, src, 0, 1, sizeof(T), 0, 0, 0, 0);
    } else if constexpr (sizeof(T) == 2) {
        copy_gm_to_ubuf_align_b16(dst, src, 0, 1, sizeof(T), 0, 0, 0, 0);
    } else if constexpr (sizeof(T) == 4) {
        copy_gm_to_ubuf_align_b32(dst, src, 0, 1, sizeof(T), 0, 0, 0, 0);
    }
}

template <typename DstTile, typename T>
AICORE PTO_INLINE void MGatherZeroFillUB(__ubuf__ T *dstPtr)
{
    using U = std::conditional_t<sizeof(T) == 4, uint32_t, uint16_t>;
    __ubuf__ U *dst = (__ubuf__ U *)dstPtr;
    constexpr uint32_t numel = DstTile::Numel * sizeof(T) / sizeof(U);
    set_mask_count();
    set_vector_mask(0, numel);
    vector_dup(dst, (U)0, 0, 1, 1, 8, 8);
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename DstTile, typename T>
AICORE PTO_INLINE uint32_t MGatherNZUbOffset(uint32_t r, uint32_t c)
{
    constexpr uint32_t kC0 = C0_SIZE_BYTE / sizeof(T);
    const uint32_t blockCol = c / kC0;
    const uint32_t colInBlock = c - blockCol * kC0;
    return blockCol * (uint32_t)DstTile::Rows * kC0 + r * kC0 + colInBlock;
}

template <typename T>
AICORE PTO_INLINE uint64_t MGatherNZGmOffset(uint32_t logicalRow, uint32_t logicalCol, int gShape0, int gShape1,
                                             int gStride0, int gStride1, int gStride2, int gStride3, int gStride4)
{
    constexpr uint32_t kC0 = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t kFRow = FRACTAL_NZ_ROW;
    const uint32_t blockColCombined = logicalCol / kC0;
    const uint32_t colInBlock = logicalCol - blockColCombined * kC0;
    const uint32_t blockRow = logicalRow / kFRow;
    const uint32_t rowInBlock = logicalRow - blockRow * kFRow;
    const uint32_t blockColOuter0 = (gShape0 == 1) ? 0u : (blockColCombined / (uint32_t)gShape1);
    const uint32_t blockColOuter1 = (gShape0 == 1) ? blockColCombined : (blockColCombined - blockColOuter0 * gShape1);
    return (uint64_t)blockColOuter0 * (uint64_t)gStride0 + (uint64_t)blockColOuter1 * (uint64_t)gStride1 +
           (uint64_t)blockRow * (uint64_t)gStride2 + (uint64_t)rowInBlock * (uint64_t)gStride3 +
           (uint64_t)colInBlock * (uint64_t)gStride4;
}

template <typename T>
AICORE PTO_INLINE void MGatherZeroSlice(__ubuf__ T *dstSlice, __ubuf__ int32_t *idxSlice, uint32_t count,
                                        uint32_t tableSize)
{
    set_vector_mask(0, count);
    vmuls(idxSlice, idxSlice, static_cast<int32_t>(-1), 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vadds(idxSlice, idxSlice, static_cast<int32_t>(tableSize), 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vrelu(idxSlice, idxSlice, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vmins(idxSlice, idxSlice, static_cast<int32_t>(1), 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
        vmul(reinterpret_cast<__ubuf__ int32_t *>(dstSlice), reinterpret_cast<__ubuf__ int32_t *>(dstSlice), idxSlice,
             1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    } else if constexpr (std::is_same_v<T, float>) {
        __ubuf__ float *idxF = reinterpret_cast<__ubuf__ float *>(idxSlice);
        vconv_s322f32(idxF, idxSlice, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(reinterpret_cast<__ubuf__ float *>(dstSlice), reinterpret_cast<__ubuf__ float *>(dstSlice), idxF, 1, 1, 1,
             1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    } else if constexpr (std::is_same_v<T, half>) {
        __ubuf__ float *idxF = reinterpret_cast<__ubuf__ float *>(idxSlice);
        __ubuf__ half *idxH = reinterpret_cast<__ubuf__ half *>(idxSlice);
        vconv_s322f32(idxF, idxSlice, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f16(idxH, idxF, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(reinterpret_cast<__ubuf__ half *>(dstSlice), reinterpret_cast<__ubuf__ half *>(dstSlice), idxH, 1, 1, 1, 1,
             8, 8, 8);
        pipe_barrier(PIPE_V);
    } else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>) {
        __ubuf__ float *idxF = reinterpret_cast<__ubuf__ float *>(idxSlice);
        __ubuf__ int16_t *idxS = reinterpret_cast<__ubuf__ int16_t *>(idxSlice);
        vconv_s322f32(idxF, idxSlice, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322s16r(idxS, idxF, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(reinterpret_cast<__ubuf__ int16_t *>(dstSlice), reinterpret_cast<__ubuf__ int16_t *>(dstSlice), idxS, 1, 1,
             1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }
}

template <GatherOOB Oob, typename T, typename TIdx, typename DstTile, typename IdxTile>
__tf__ AICORE void MGatherRowImpl(typename DstTile::TileDType __out__ dst, __gm__ T *tablePtr,
                                  typename IdxTile::TileDType __in__ indices, uint32_t validRow, uint32_t validCol,
                                  uint32_t tableRows, uint32_t tableRowStride)
{
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ TIdx *idxPtr = (__ubuf__ TIdx *)__cce_get_tile_ptr(indices);

    if constexpr (Oob == GatherOOB::Zero) {
        MGatherZeroFillUB<DstTile, T>(dstPtr);
        PtoSetWaitFlag<PIPE_V, PIPE_MTE2>();
    }

    PtoSetWaitFlag<PIPE_V, PIPE_S>();

    const uint32_t lenBytes = validCol * sizeof(T);
    constexpr uint32_t kRowStride = DstTile::RowStride;

    for (uint32_t r = 0; r < validRow; r++) {
        uint32_t rawIdx = static_cast<uint32_t>(idxPtr[r]);
        uint32_t doRead;
        uint32_t safeIdx = mgather_remap<Oob>(rawIdx, tableRows, doRead);
        if (doRead) {
            __gm__ T *srcRow = tablePtr + static_cast<uint64_t>(safeIdx) * tableRowStride;
            __ubuf__ T *dstRow = dstPtr + r * kRowStride;
            MGatherRowDma<T>(dstRow, srcRow, lenBytes);
        }
    }

    PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
}

template <GatherOOB Oob, typename T, typename TIdx, typename DstTile, typename IdxTile>
__tf__ AICORE void MGatherRowNzImpl(typename DstTile::TileDType __out__ dst, __gm__ T *tablePtr,
                                    typename IdxTile::TileDType __in__ indices, uint32_t validRow, int gShape0,
                                    int gShape1, int gShape2, int gStride0, int gStride1, int gStride2, int gStride3)
{
    constexpr uint32_t kC0 = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t kFRow = FRACTAL_NZ_ROW;
    constexpr uint32_t kFractalRowBytes = kC0 * sizeof(T);

    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ TIdx *idxPtr = (__ubuf__ TIdx *)__cce_get_tile_ptr(indices);

    if constexpr (Oob == GatherOOB::Zero) {
        MGatherZeroFillUB<DstTile, T>(dstPtr);
        PtoSetWaitFlag<PIPE_V, PIPE_MTE2>();
    }

    PtoSetWaitFlag<PIPE_V, PIPE_S>();

    const uint32_t tableLogicalRows = (uint32_t)gShape2 * kFRow;
    const uint32_t gmGapBytes = ((uint32_t)gStride1 - kC0) * (uint32_t)sizeof(T);
    constexpr uint32_t ubGapBlocks = (uint32_t)DstTile::Rows - 1u;
    const int64_t tileOuterStrideElem = (int64_t)gShape1 * (int64_t)DstTile::Rows * (int64_t)kC0;

    for (uint32_t r = 0; r < validRow; r++) {
        uint32_t rawIdx = static_cast<uint32_t>(idxPtr[r]);
        uint32_t doRead;
        uint32_t safeIdx = mgather_remap<Oob>(rawIdx, tableLogicalRows, doRead);
        if (doRead) {
            const uint32_t srcBlockRow = safeIdx / kFRow;
            const uint32_t srcRowInBlock = safeIdx - srcBlockRow * kFRow;
            const uint32_t dstBlockRow = r / kFRow;
            const uint32_t dstRowInBlock = r - dstBlockRow * kFRow;

            for (uint32_t i = 0; i < (uint32_t)gShape0; i++) {
                __gm__ T *srcAddr = tablePtr + (int64_t)i * (int64_t)gStride0 +
                                    (int64_t)srcBlockRow * (int64_t)gStride2 +
                                    (int64_t)srcRowInBlock * (int64_t)gStride3;
                __ubuf__ T *dstAddr = dstPtr + (int64_t)i * tileOuterStrideElem +
                                      (int64_t)dstBlockRow * (int64_t)kFRow * (int64_t)kC0 +
                                      (int64_t)dstRowInBlock * (int64_t)kC0;
                MGatherRowMultiDma<T>(dstAddr, srcAddr, (uint16_t)gShape1, kFractalRowBytes, gmGapBytes, ubGapBlocks);
            }
        }
    }

    PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
}

template <GatherOOB Oob, typename T, typename TIdx, typename DstTile, typename IdxTile>
__tf__ AICORE void MGatherElemImpl(typename DstTile::TileDType __out__ dst, __gm__ T *tablePtr,
                                   typename IdxTile::TileDType __in__ indices, uint32_t validRow, uint32_t validCol,
                                   uint32_t tableSize)
{
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ TIdx *idxPtr = (__ubuf__ TIdx *)__cce_get_tile_ptr(indices);

    if constexpr (Oob == GatherOOB::Zero) {
        MGatherZeroFillUB<DstTile, T>(dstPtr);
        PtoSetWaitFlag<PIPE_V, PIPE_MTE2>();
    }

    PtoSetWaitFlag<PIPE_V, PIPE_S>();

    constexpr uint32_t kDstRowStride = DstTile::RowStride;
    constexpr uint32_t kIdxRowStride = IdxTile::RowStride;

    for (uint32_t r = 0; r < validRow; r++) {
        const uint32_t idxRowOff = r * kIdxRowStride;
        const uint32_t dstRowOff = r * kDstRowStride;
        for (uint32_t c = 0; c < validCol; c++) {
            const uint32_t idxOff = idxRowOff + c;
            const uint32_t dstOff = dstRowOff + c;
            uint32_t rawIdx = static_cast<uint32_t>(idxPtr[idxOff]);
            uint32_t doRead;
            uint32_t safeIdx = mgather_remap<Oob>(rawIdx, tableSize, doRead);
            if constexpr (Oob == GatherOOB::Zero) {
                if (!doRead) {
                    safeIdx = 0;
                }
                doRead = 1u;
            }
            if (doRead) {
                __gm__ T *srcElem = tablePtr + safeIdx;
                __ubuf__ T *dstElem = dstPtr + dstOff;
                MGatherElemDma<T>(dstElem, srcElem);
            }
        }
    }

    if constexpr (Oob == GatherOOB::Zero) {
        PtoSetWaitFlag<PIPE_S, PIPE_V>();
        PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
        set_mask_count();
        for (uint32_t r = 0; r < validRow; r++) {
            __ubuf__ int32_t *idxRow = reinterpret_cast<__ubuf__ int32_t *>(idxPtr) + r * kIdxRowStride;
            __ubuf__ T *dstRow = dstPtr + r * kDstRowStride;
            MGatherZeroSlice<T>(dstRow, idxRow, validCol, tableSize);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
        PtoSetWaitFlag<PIPE_V, PIPE_S>();
    }

    PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
}

template <GatherOOB Oob, typename T, typename TIdx, typename DstTile, typename IdxTile>
__tf__ AICORE void MGatherElemNzImpl(typename DstTile::TileDType __out__ dst, __gm__ T *tablePtr,
                                     typename IdxTile::TileDType __in__ indices, uint32_t validRow, uint32_t validCol,
                                     uint32_t tableSize, int gShape0, int gShape1, int gStride0, int gStride1,
                                     int gStride2, int gStride3, int gStride4, uint32_t nLogicalCols)
{
    constexpr uint32_t kC0 = C0_SIZE_BYTE / sizeof(T);
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ TIdx *idxPtr = (__ubuf__ TIdx *)__cce_get_tile_ptr(indices);

    if constexpr (Oob == GatherOOB::Zero) {
        MGatherZeroFillUB<DstTile, T>(dstPtr);
        PtoSetWaitFlag<PIPE_V, PIPE_MTE2>();
    }

    PtoSetWaitFlag<PIPE_V, PIPE_S>();

    constexpr uint32_t kIdxRowStride = IdxTile::RowStride;
    const uint32_t nColBlocks = (validCol + kC0 - 1u) / kC0;
    const uint32_t kDstColBlockStride = (uint32_t)DstTile::Rows * kC0;

    for (uint32_t bcol = 0; bcol < nColBlocks; bcol++) {
        const uint32_t cBase = bcol * kC0;
        const uint32_t cLimit = (cBase + kC0 < validCol) ? (cBase + kC0) : validCol;
        const uint32_t kInBlock = cLimit - cBase;
        __ubuf__ T *dstBlockBase = dstPtr + (uint64_t)bcol * (uint64_t)kDstColBlockStride;
        for (uint32_t r = 0; r < validRow; r++) {
            const uint32_t idxRowOff = r * kIdxRowStride;
            __ubuf__ T *dstRowBase = dstBlockBase + (uint64_t)r * (uint64_t)kC0;
            for (uint32_t cInner = 0; cInner < kInBlock; cInner++) {
                const uint32_t c = cBase + cInner;
                const uint32_t idxOff = idxRowOff + c;
                uint32_t rawIdx = static_cast<uint32_t>(idxPtr[idxOff]);
                uint32_t doRead;
                uint32_t safeIdx = mgather_remap<Oob>(rawIdx, tableSize, doRead);
                if constexpr (Oob == GatherOOB::Zero) {
                    if (!doRead) {
                        safeIdx = 0;
                    }
                    doRead = 1u;
                }
                if (doRead) {
                    const uint32_t logicalRow = safeIdx / nLogicalCols;
                    const uint32_t logicalCol = safeIdx - logicalRow * nLogicalCols;
                    const uint64_t srcOff = MGatherNZGmOffset<T>(logicalRow, logicalCol, gShape0, gShape1, gStride0,
                                                                 gStride1, gStride2, gStride3, gStride4);
                    __gm__ T *srcElem = tablePtr + srcOff;
                    __ubuf__ T *dstElem = dstRowBase + cInner;
                    MGatherElemDma<T>(dstElem, srcElem);
                }
            }
        }
    }

    if constexpr (Oob == GatherOOB::Zero) {
        PtoSetWaitFlag<PIPE_S, PIPE_V>();
        PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
        set_mask_count();
        for (uint32_t bcol = 0; bcol < nColBlocks; bcol++) {
            const uint32_t cBase = bcol * kC0;
            const uint32_t cLimit = (cBase + kC0 < validCol) ? (cBase + kC0) : validCol;
            const uint32_t kInBlock = cLimit - cBase;
            for (uint32_t r = 0; r < validRow; r++) {
                __ubuf__ T *dstSlice =
                    dstPtr + (uint64_t)bcol * (uint64_t)kDstColBlockStride + (uint64_t)r * (uint64_t)kC0;
                __ubuf__ int32_t *idxSlice = reinterpret_cast<__ubuf__ int32_t *>(idxPtr) + r * kIdxRowStride + cBase;
                MGatherZeroSlice<T>(dstSlice, idxSlice, kInBlock, tableSize);
            }
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
        PtoSetWaitFlag<PIPE_V, PIPE_S>();
    }

    PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
}

template <GatherOOB Oob, typename T, typename TIdx, typename DstTile, typename IdxTile>
__tf__ AICORE void MGatherElemScalarImpl(typename DstTile::TileDType __out__ dst, __gm__ T *tablePtr,
                                         typename IdxTile::TileDType __in__ indices, uint32_t tableSize)
{
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ TIdx *idxPtr = (__ubuf__ TIdx *)__cce_get_tile_ptr(indices);

    PtoSetWaitFlag<PIPE_V, PIPE_S>();

    uint32_t rawIdx = static_cast<uint32_t>(idxPtr[0]);
    uint32_t doRead;
    uint32_t safeIdx = mgather_remap<Oob>(rawIdx, tableSize, doRead);
    if (doRead) {
        dstPtr[0] = tablePtr[safeIdx];
    } else {
        dstPtr[0] = static_cast<T>(0);
    }

    PtoSetWaitFlag<PIPE_S, PIPE_V>();
}

template <typename T>
struct IsMGatherZeroVecCoreSupportedDType {
    static constexpr bool value = std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
                                  std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                                  std::is_same_v<T, half> || std::is_same_v<T, float>;
};

template <Coalesce Mode, GatherOOB Oob, typename DstTile, typename GlobalTable, typename IdxTile>
PTO_INTERNAL void MGatherCheck()
{
    using T = typename DstTile::DType;
    using TIdx = typename IdxTile::DType;

    static_assert(IsValidMGatherDType<T>::value,
                  "MGATHER A2/A3 data type must be int8/uint8/int16/uint16/int32/uint32/half/bfloat16/float.");
    static_assert(std::is_same_v<TIdx, int32_t> || std::is_same_v<TIdx, uint32_t>,
                  "MGATHER A2/A3 index type must be int32_t or uint32_t.");
    static_assert(std::is_same_v<typename GlobalTable::DType, __gm__ T>,
                  "MGATHER A2/A3 source table must be a GM GlobalTensor with element type matching the destination.");
    static_assert(DstTile::Loc == TileType::Vec, "MGATHER A2/A3 destination must be a Vec tile (UB).");
    static_assert(IdxTile::Loc == TileType::Vec, "MGATHER A2/A3 indices must be a Vec tile (UB).");

    static_assert(IdxTile::isRowMajor, "MGATHER A2/A3 index tile must be BLayout::RowMajor.");
    static_assert(IdxTile::SFractal == SLayout::NoneBox, "MGATHER A2/A3 index tile must be ND (SLayout::NoneBox).");

    constexpr bool kIsTableND = (GlobalTable::layout == Layout::ND);
    constexpr bool kIsTableNZ = (GlobalTable::layout == Layout::NZ);
    constexpr bool kIsDstND = IsMGatherNDTile<DstTile>::value;
    constexpr bool kIsDstNZ = IsMGatherNZTile<DstTile>::value;

    static_assert(kIsTableND || kIsTableNZ, "MGATHER A2/A3 table must use Layout::ND or Layout::NZ.");
    static_assert((kIsTableND && kIsDstND) || (kIsTableNZ && kIsDstNZ),
                  "MGATHER A2/A3 layout pairing must be either:\n"
                  "  (a) GM Layout::ND + UB tile (BLayout::RowMajor + SLayout::NoneBox), or\n"
                  "  (b) GM Layout::NZ + UB tile (BLayout::ColMajor + SLayout::RowMajor + SFractalSize=512).");

    static_assert(DstTile::Cols * sizeof(T) % BLOCK_BYTE_SIZE == 0,
                  "MGATHER A2/A3 destination tile padded Cols*sizeof(T) must be 32-byte aligned.");

    if constexpr (kIsTableNZ) {
        static_assert(GlobalTable::staticShape[3] == FRACTAL_NZ_ROW,
                      "MGATHER A2/A3 NZ table requires staticShape[3] == FRACTAL_NZ_ROW (16).");
        static_assert(GlobalTable::staticShape[4] == C0_SIZE_BYTE / sizeof(T),
                      "MGATHER A2/A3 NZ table requires staticShape[4] == 32 / sizeof(T).");
        static_assert(DstTile::Cols % (C0_SIZE_BYTE / sizeof(T)) == 0,
                      "MGATHER A2/A3 NZ destination tile Cols must be a multiple of C0 (= 32 / sizeof(T)).");
        static_assert(DstTile::Rows % FRACTAL_NZ_ROW == 0,
                      "MGATHER A2/A3 NZ destination tile Rows must be a multiple of FRACTAL_NZ_ROW (16).");
    }

    constexpr int kDstValidR = DstTile::ValidRow;
    constexpr int kDstValidC = DstTile::ValidCol;
    constexpr int kIdxValidR = IdxTile::ValidRow;
    constexpr int kIdxValidC = IdxTile::ValidCol;

    if constexpr (Mode == Coalesce::Row) {
        if constexpr (kDstValidR > 0 && kIdxValidR > 0 && kIdxValidC > 0) {
            static_assert(kIdxValidR == 1 && kIdxValidC == kDstValidR,
                          "MGATHER A2/A3 Coalesce::Row requires index tile valid shape [1, R].");
        }
    } else {
        if constexpr (kDstValidR > 0 && kIdxValidR > 0) {
            static_assert(kIdxValidR == kDstValidR,
                          "MGATHER A2/A3 Coalesce::Elem requires index tile ValidRow == destination ValidRow.");
        }
        if constexpr (kDstValidC > 0 && kIdxValidC > 0) {
            static_assert(kIdxValidC == kDstValidC,
                          "MGATHER A2/A3 Coalesce::Elem requires index tile ValidCol == destination ValidCol.");
        }
        if constexpr (Oob == GatherOOB::Zero) {
            static_assert(IsMGatherZeroVecCoreSupportedDType<T>::value,
                          "MGATHER A2/A3 Coalesce::Elem with GatherOOB::Zero requires a dtype supported by the "
                          "vec-core post-pass (int16/uint16/int32/uint32/half/float). int8/uint8/bfloat16 are not "
                          "supported because vmul has no native instruction for those dtypes.");
            static_assert(std::is_same_v<TIdx, int32_t>,
                          "MGATHER A2/A3 Coalesce::Elem with GatherOOB::Zero requires int32_t indices (the vec-core "
                          "valid-mask computation operates in signed int32 lanes).");
        }
    }
}

template <Coalesce Mode = Coalesce::Row, GatherOOB Oob = GatherOOB::Undefined, typename DstTile, typename GlobalTable,
          typename IdxTile>
PTO_INTERNAL void MGATHER_IMPL(DstTile &dst, GlobalTable &table, IdxTile &indices)
{
    using T = typename DstTile::DType;
    using TIdx = typename IdxTile::DType;

    MGatherCheck<Mode, Oob, DstTile, GlobalTable, IdxTile>();

    __gm__ T *tablePtr = reinterpret_cast<__gm__ T *>(table.data());

    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();

    constexpr bool kIsTableNZ = (GlobalTable::layout == Layout::NZ);

    if constexpr (kIsTableNZ) {
        const int gShape0 = static_cast<int>(table.GetShape(GlobalTensorDim::DIM_0));
        const int gShape1 = static_cast<int>(table.GetShape(GlobalTensorDim::DIM_1));
        const int gShape2 = static_cast<int>(table.GetShape(GlobalTensorDim::DIM_2));
        const int gStride0 = static_cast<int>(table.GetStride(GlobalTensorDim::DIM_0));
        const int gStride1 = static_cast<int>(table.GetStride(GlobalTensorDim::DIM_1));
        const int gStride2 = static_cast<int>(table.GetStride(GlobalTensorDim::DIM_2));
        const int gStride3 = static_cast<int>(table.GetStride(GlobalTensorDim::DIM_3));
        const int gStride4 = static_cast<int>(table.GetStride(GlobalTensorDim::DIM_4));

        if constexpr (Mode == Coalesce::Row) {
            MGatherRowNzImpl<Oob, T, TIdx, DstTile, IdxTile>(dst.data(), tablePtr, indices.data(), validRow, gShape0,
                                                             gShape1, gShape2, gStride0, gStride1, gStride2, gStride3);
        } else {
            constexpr uint32_t kC0 = C0_SIZE_BYTE / sizeof(T);
            const uint32_t nLogicalCols = static_cast<uint32_t>(gShape0 * gShape1) * kC0;
            const uint32_t tableSize = static_cast<uint32_t>(gShape2 * FRACTAL_NZ_ROW) * nLogicalCols;
            if constexpr (DstTile::ValidRow == 1 && DstTile::ValidCol == 1) {
                MGatherElemScalarImpl<Oob, T, TIdx, DstTile, IdxTile>(dst.data(), tablePtr, indices.data(), tableSize);
            } else {
                MGatherElemNzImpl<Oob, T, TIdx, DstTile, IdxTile>(dst.data(), tablePtr, indices.data(), validRow,
                                                                  validCol, tableSize, gShape0, gShape1, gStride0,
                                                                  gStride1, gStride2, gStride3, gStride4, nLogicalCols);
            }
        }
    } else {
        if constexpr (Mode == Coalesce::Row) {
            const uint32_t tableRows =
                static_cast<uint32_t>(table.GetShape(GlobalTensorDim::DIM_0) * table.GetShape(GlobalTensorDim::DIM_1) *
                                      table.GetShape(GlobalTensorDim::DIM_2) * table.GetShape(GlobalTensorDim::DIM_3));
            const uint32_t tableRowStride = static_cast<uint32_t>(table.GetStride(GlobalTensorDim::DIM_3));
            MGatherRowImpl<Oob, T, TIdx, DstTile, IdxTile>(dst.data(), tablePtr, indices.data(), validRow, validCol,
                                                           tableRows, tableRowStride);
        } else {
            const uint32_t tableSize =
                static_cast<uint32_t>(table.GetShape(GlobalTensorDim::DIM_0) * table.GetShape(GlobalTensorDim::DIM_1) *
                                      table.GetShape(GlobalTensorDim::DIM_2) * table.GetShape(GlobalTensorDim::DIM_3) *
                                      table.GetShape(GlobalTensorDim::DIM_4));
            if constexpr (DstTile::ValidRow == 1 && DstTile::ValidCol == 1) {
                MGatherElemScalarImpl<Oob, T, TIdx, DstTile, IdxTile>(dst.data(), tablePtr, indices.data(), tableSize);
            } else {
                MGatherElemImpl<Oob, T, TIdx, DstTile, IdxTile>(dst.data(), tablePtr, indices.data(), validRow,
                                                                validCol, tableSize);
            }
        }
    }
}

} // namespace pto

#endif // MGATHER_HPP
