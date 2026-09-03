/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

#define LOGSIZE 128
#define DEBUGLOG
#ifdef DEBUGLOG
#define LOG(x) *(gLog++) = x;
#else
#define LOG(x)
#endif

inline AICORE uint64_t get_syscnt()
{
    uint64_t syscnt;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

template <typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void tf_copy_cbuf_to_ubuf(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src, int vec_core,
    int block_count, int block_len, int src_stride, int dst_stride)
{
    copy_cbuf_to_ubuf(
        (__ubuf__ void*)__cce_get_tile_ptr(dst), (__cbuf__ void*)__cce_get_tile_ptr(src), vec_core, block_count,
        block_len, src_stride, dst_stride);
}

template <typename T, int M, int K, int baseM, int baseK, bool useNotAlloc>
AICORE inline void runTLOAD_MIX_ND2ND(__gm__ T* out, __gm__ T* src0, __gm__ uint64_t* gLog)
{
#ifdef DEBUGLOG
    gLog += block_idx * LOGSIZE;
#endif
    using NDValidShape = TileShape2D<T, -1, -1, Layout::ND>;
    using NDWholeShape = BaseShape2D<T, -1, -1, Layout::ND>;
    NDValidShape ndValidShape(M, K);
    NDWholeShape ndWholeShape(M, K);
    using GlobalDataSrc0 = GlobalTensor<T, NDValidShape, NDWholeShape, Layout::ND>;
    GlobalDataSrc0 src0Global(src0, ndValidShape, ndWholeShape);

    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, baseM, baseK>,
        pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, T, baseM, baseK, BLayout::RowMajor, M, K, SLayout::NoneBox>;
    using TileUBData = Tile<TileType::Vec, T, baseM, baseK, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(baseM, baseK);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    volatile uint64_t t0 = 0, t1 = 0;
#if defined(__DAV_CUBE__)
    t0 = get_syscnt();
    for (int i = 0; i < 5; ++i) {
        if constexpr (useNotAlloc) {
            TLOAD<TLoadL2Hint::NotAllocKeep>(aMatTile, src0Global);
        } else {
            TLOAD<TLoadL2Hint::NormalFirstVictim>(aMatTile, src0Global);
        }
    }
    t1 = get_syscnt();

    uint8_t syncID = 0;
    uint16_t blockCount = 1;
    uint16_t blockLen = baseM * baseK * sizeof(T) / 32;
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    tf_copy_cbuf_to_ubuf<TileUBData, TileMatAData>(srcTile.data(), aMatTile.data(), 0, blockCount, blockLen, 0, 0);
    tf_copy_cbuf_to_ubuf<TileUBData, TileMatAData>(srcTile.data(), aMatTile.data(), 1, blockCount, blockLen, 0, 0);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
#endif
    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
    LOG(t0);
    LOG(t1 - t0);
#endif

#if defined(__DAV_VEC__)
    uint8_t syncID = 0;
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, srcTile);
#endif
    out = dstGlobal.data();
}

template <bool useNotAlloc>
__global__ AICORE void TLoadL2HintMatKernel(__gm__ uint8_t* out, __gm__ uint8_t* src, __gm__ uint64_t* gLog)
{
    // A5_TLOAD_MAT_*: ND float 128x256 GM->L1 via mix ND2ND
    runTLOAD_MIX_ND2ND<float, 128, 256, 128, 256, useNotAlloc>(
        reinterpret_cast<__gm__ float*>(out), reinterpret_cast<__gm__ float*>(src), gLog);
}

template <int32_t testKey>
void LaunchTLoadL2HintMat(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream)
{
    if constexpr (testKey == 1) {
        TLoadL2HintMatKernel<false><<<1, nullptr, stream>>>(out, src, gLog);
    } else if constexpr (testKey == 2) {
        TLoadL2HintMatKernel<true><<<1, nullptr, stream>>>(out, src, gLog);
    }
}

template void LaunchTLoadL2HintMat<1>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void LaunchTLoadL2HintMat<2>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
