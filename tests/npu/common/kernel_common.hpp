/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef TESTS_NPU_COMMON_KERNEL_COMMON_HPP
#define TESTS_NPU_COMMON_KERNEL_COMMON_HPP

#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

namespace pto::test {

template <typename TileData>
__tf__ PTO_INTERNAL void TfCreateCbufMatrix(typename TileData::TileDType __out__ tile, int64_t repeatBit, int n)
{
    create_cbuf_matrix((__cbuf__ uint16_t *)__cce_get_tile_ptr(tile), repeatBit, n);
}

template <typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void TfCopyCbufToUbuf(typename TileDataDst::TileDType __out__ dst,
                                          typename TileDataSrc::TileDType __in__ src, int vecCore, int blockCount,
                                          int blockLen, int srcStride, int dstStride)
{
    copy_cbuf_to_ubuf((__ubuf__ void *)__cce_get_tile_ptr(dst), (__cbuf__ void *)__cce_get_tile_ptr(src), vecCore,
                      blockCount, blockLen, srcStride, dstStride);
}

template <typename DstTileData, typename SrcTileData, uint8_t syncID>
AICORE inline void MovL1ToUbuf(DstTileData &dstTile, SrcTileData &srcTile)
{
#if defined(__DAV_CUBE__)
    uint16_t blockCount = 1;
    uint16_t blockLen = DstTileData::Rows * DstTileData::Cols * sizeof(typename SrcTileData::DType) / BLOCK_BYTE_SIZE;
    if constexpr (std::is_same<typename SrcTileData::DType, float4_e1m2x2_t>::value ||
                  std::is_same<typename SrcTileData::DType, float4_e2m1x2_t>::value) {
        blockLen = DstTileData::Rows * DstTileData::Cols / B4_C0_SIZE;
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    TfCopyCbufToUbuf<DstTileData, SrcTileData>(dstTile.data(), srcTile.data(), 0, blockCount, blockLen, 0, 0);
    TfCopyCbufToUbuf<DstTileData, SrcTileData>(dstTile.data(), srcTile.data(), 1, blockCount, blockLen, 0, 0);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
#endif
    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
#endif
}

template <typename SrcTile, typename DstTile, typename GlobalData, typename ScalarT, typename ComputeFn>
PTO_INTERNAL void RunVecBinaryWithManualEvents(DstTile &dstTile, SrcTile &srcTile, GlobalData &dstGlobal,
                                               GlobalData &srcGlobal, ScalarT scalar, ComputeFn &&compute)
{
    TLOAD(dstTile, dstGlobal);
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    compute(dstTile, srcTile, scalar);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

} // namespace pto::test

#endif
