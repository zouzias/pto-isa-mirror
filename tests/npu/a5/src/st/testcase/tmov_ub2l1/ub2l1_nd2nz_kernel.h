/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef UB2L1_ND2NZ_KERNEL_H_
#define UB2L1_ND2NZ_KERNEL_H_

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>

#include "ub2l1_nd2nz_types.h"

namespace PtoTestCommon {

using namespace pto;

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void readbackNzMat(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t vectorId,
    uint16_t blockCount, uint32_t srcByteOffset = 0)
{
    __cbuf__ T* srcMatAddr = (__cbuf__ T*)((__cbuf__ uint8_t*)__cce_get_tile_ptr(src) + srcByteOffset);
    __ubuf__ T* dstUbAddr = __cce_get_tile_ptr(dst);
    copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcMatAddr, vectorId, 1, blockCount, 0, 0);
}

template <typename MatTile, typename UbTile>
__tf__ PTO_INTERNAL void initNzMat(
    typename MatTile::TileDType __out__ dst, typename UbTile::TileDType __in__ src, uint32_t byteCount,
    uint32_t dstByteOffset)
{
    copy_ubuf_to_cbuf(
        (__cbuf__ uint8_t*)__cce_get_tile_ptr(dst) + dstByteOffset, (__ubuf__ void*)__cce_get_tile_ptr(src), 0, 1,
        byteCount / BLOCK_BYTE_SIZE, 0, 0);
}

template <
    typename T, int ElementBits, int SrcRows, int SrcCols, int DstRows, int DstCols, int ValidRows, int ValidCols,
    Nd2NzOperation Operation, int IndexRow, int IndexCol, bool Dynamic, int SrcValidRows, int SrcValidCols>
__global__ AICORE void runUbToL1Nd2Nz(
    __gm__ uint8_t* out, __gm__ uint8_t* input, uint32_t validRows, uint32_t validCols, uint32_t srcValidRows,
    uint32_t srcValidCols)
{
    constexpr uint32_t L1_READY = 0;
    constexpr uint32_t UB_READY = 1;
    constexpr uint32_t L1_INITIALIZED = 2;
    constexpr uint32_t INIT_ACK = 3;
    constexpr uint32_t AIV1_EVENT_OFFSET = 16;
    constexpr uint32_t DST_BYTES = DstRows * DstCols * ElementBits / 8;
    static_assert(SrcRows * SrcCols * ElementBits / 8 <= 256 * 1024, "ST source exceeds A5 UB capacity.");
    static_assert(DST_BYTES <= 512 * 1024, "ST destination exceeds A5 L1 capacity.");
    constexpr uint32_t MAX_READBACK_BYTES = 128 * 1024;
    constexpr uint32_t READBACK_BYTES = DST_BYTES > MAX_READBACK_BYTES ? MAX_READBACK_BYTES : DST_BYTES;
    constexpr uint32_t READBACK_COUNT = (DST_BYTES + READBACK_BYTES - 1) / READBACK_BYTES;
    constexpr bool IS_INSERT = Operation == Nd2NzOperation::INSERT || Operation == Nd2NzOperation::DUAL_INSERT;
    using SrcTile = Tile<
        TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor, Dynamic ? DYNAMIC : SrcValidRows,
        Dynamic ? DYNAMIC : SrcValidCols>;
    using DstTile = Tile<
        TileType::Mat, T, DstRows, DstCols, BLayout::ColMajor, Dynamic ? DYNAMIC : (IS_INSERT ? DstRows : ValidRows),
        Dynamic ? DYNAMIC : (IS_INSERT ? DstCols : ValidCols), SLayout::RowMajor>;
    using RawSrc = Tile<TileType::Vec, uint8_t, SrcRows, SrcCols * ElementBits / 8>;
    using RawDst = Tile<TileType::Vec, uint8_t, 1, READBACK_BYTES, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using SrcGlobal = GlobalTensor<
        uint8_t, Shape<1, 1, 1, SrcRows, SrcCols * ElementBits / 8>,
        pto::Stride<1, 1, 1, SrcCols * ElementBits / 8, 1>>;
    using DstGlobal = GlobalTensor<uint8_t, Shape<1, 1, 1, 1, DYNAMIC>, pto::Stride<1, 1, 1, READBACK_BYTES, 1>>;
    SrcTile src;
    DstTile dst;
    if constexpr (Dynamic) {
        src.SetValidShape(srcValidRows, srcValidCols);
        dst.SetValidShape(IS_INSERT ? DstRows : validRows, IS_INSERT ? DstCols : validCols);
    }
    RawSrc rawSrc;
    RawDst rawDst(1, READBACK_BYTES);
    TASSIGN(src, 0);
    TASSIGN(dst, 0);
    TASSIGN(rawSrc, 0);
    TASSIGN(rawDst, 0);
    SrcGlobal srcGlobal(
        input +
        (Operation == Nd2NzOperation::DUAL_INSERT ? get_subblockid() * SrcRows * SrcCols * ElementBits / 8 : 0));
#if defined(__DAV_VEC__)
    // Initialize L1 once; the dual-AIV case writes disjoint row windows.
    if (get_subblockid() == 0) {
        for (uint32_t chunk = 0; chunk < READBACK_COUNT; ++chunk) {
            const uint32_t offset = chunk * READBACK_BYTES;
            const uint32_t bytes = min(READBACK_BYTES, DST_BYTES - offset);
            rawDst.SetValidShape(1, bytes);
            DstGlobal dstGlobal(out + offset, Shape<1, 1, 1, 1, DYNAMIC>(bytes));
            TLOAD(rawDst, dstGlobal);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            initNzMat<DstTile, RawDst>(dst.data(), rawDst.data(), bytes, offset);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        if constexpr (Operation == Nd2NzOperation::DUAL_INSERT) {
            set_intra_block(PIPE_MTE3, L1_INITIALIZED);
        }
    }
    if constexpr (Operation == Nd2NzOperation::DUAL_INSERT) {
        wait_intra_block(PIPE_MTE2, INIT_ACK);
    }
    if (Operation == Nd2NzOperation::DUAL_INSERT || get_subblockid() == 0) {
        TLOAD(rawSrc, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        if constexpr (Operation == Nd2NzOperation::EXTRACT) {
            TEXTRACT(dst, src, IndexRow, IndexCol);
        } else if constexpr (IS_INSERT) {
            TINSERT(
                dst, src, IndexRow + (Operation == Nd2NzOperation::DUAL_INSERT ? get_subblockid() * validRows : 0),
                IndexCol);
        } else {
            TMOV(dst, src);
        }
        set_intra_block(PIPE_MTE3, L1_READY);
        if (get_subblockid() == 0) {
            for (uint32_t chunk = 0; chunk < READBACK_COUNT; ++chunk) {
                const uint32_t offset = chunk * READBACK_BYTES;
                const uint32_t bytes = min(READBACK_BYTES, DST_BYTES - offset);
                rawDst.SetValidShape(1, bytes);
                DstGlobal dstGlobal(out + offset, Shape<1, 1, 1, 1, DYNAMIC>(bytes));
                wait_intra_block(PIPE_MTE3, UB_READY);
                TSTORE(dstGlobal, rawDst);
                if (chunk + 1 < READBACK_COUNT) {
                    set_intra_block(PIPE_MTE3, L1_READY);
                }
            }
        }
    }
#endif
#if defined(__DAV_CUBE__)
    if constexpr (Operation == Nd2NzOperation::DUAL_INSERT) {
        wait_intra_block(PIPE_MTE1, L1_INITIALIZED);
        set_intra_block(PIPE_MTE1, INIT_ACK);
        set_intra_block(PIPE_MTE1, INIT_ACK + AIV1_EVENT_OFFSET);
    }
    wait_intra_block(PIPE_MTE1, L1_READY);
    if constexpr (Operation == Nd2NzOperation::DUAL_INSERT) {
        wait_intra_block(PIPE_MTE1, L1_READY + AIV1_EVENT_OFFSET);
    }
    // Wait for GM stores before reusing the bounded UB readback buffer.
    for (uint32_t chunk = 0; chunk < READBACK_COUNT; ++chunk) {
        if (chunk > 0) {
            wait_intra_block(PIPE_MTE1, L1_READY);
        }
        const uint32_t offset = chunk * READBACK_BYTES;
        const uint32_t bytes = min(READBACK_BYTES, DST_BYTES - offset);
        readbackNzMat<uint8_t, RawDst, DstTile>(rawDst.data(), dst.data(), 0, bytes / BLOCK_BYTE_SIZE, offset);
        set_intra_block(PIPE_MTE1, UB_READY);
    }
#endif
}

} // namespace PtoTestCommon

#endif
