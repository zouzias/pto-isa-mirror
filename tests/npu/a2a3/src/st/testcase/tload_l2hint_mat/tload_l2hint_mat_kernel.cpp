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
#include <pto/npu/a2a3/l2_cache_hint.hpp>
#include <iostream>

using namespace std;
using namespace pto;

#define LOGSIZE 128
#define DEBUGLOG
#ifdef DEBUGLOG
#define LOG(x) *(gLog++) = x;
#else
#define LOG(x)
#endif

// Golden reload only for sim that does not model L2 disable offset.
// Real A2A3 NPU uses runtime g_opL2CacheHintCfg.l2Cacheoffset — do not mask NotAlloc.
inline AICORE uint64_t get_syscnt()
{
    uint64_t syscnt;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

template <typename GlobalData, typename TileDataSrc>
__tf__ PTO_INTERNAL void tf_copy_cbuf_to_gm(
    typename GlobalData::DType __out__* dst, typename TileDataSrc::TileDType __in__ src, uint8_t n, uint16_t nBurst,
    uint16_t lenBurst, uint16_t l1Gap, uint16_t gmGap)
{
    copy_cbuf_to_gm(dst, __cce_get_tile_ptr(src), n, nBurst, lenBurst, l1Gap, gmGap);
}

template <typename GlobalData, typename TileData>
AICORE inline void TSTORE_MAT2GM(GlobalData& dst, TileData& src)
{
    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);

    uint32_t validRow = src.GetValidRow();
    uint32_t validCol = src.GetValidCol();

    if constexpr (GlobalData::layout == pto::Layout::ND && GetTileLayoutCustom<TileData>() == TileLayoutCustom::ND) {
        uint16_t nBurst = validRow;
        uint16_t lenBurst = (validCol + blockSizeElem - 1) / blockSizeElem;
        uint16_t l1Gap = (TileData::Cols - validCol) / blockSizeElem;
        uint16_t gmGap = 0;
        tf_copy_cbuf_to_gm<GlobalData, TileData>(dst.data(), src.data(), (uint8_t)0, nBurst, lenBurst, l1Gap, gmGap);
    }
}

template <
    typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1,
    int gWholeShape2, int gWholeShape3, int gWholeShape4, bool useNotAlloc>
AICORE inline void RunTLoadND2ND(__gm__ T __out__* out, __gm__ T __in__* src, __gm__ uint64_t* gLog)
{
#ifdef DEBUGLOG
    gLog += block_idx * LOGSIZE;
#endif
    constexpr int gStride[5] = {
        gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape3 * gWholeShape4, gWholeShape4, 1};
    constexpr int blockSize = 32 / sizeof(T);
    constexpr int validRow = gShape0 * gShape1 * gShape2 * gShape3;
    constexpr int validCol = gShape4;
    constexpr int Rows = gShape0 * gShape1 * gShape2 * gShape3;
    constexpr int Cols = (gShape4 + blockSize - 1) / blockSize * blockSize;

    using DynShapeDim5 = Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Mat, T, Rows, Cols, BLayout::RowMajor, -1, -1>;

    TileData srcTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    volatile uint64_t t0, t1;
    t0 = get_syscnt();
    for (int i = 0; i < 5; ++i) {
        if constexpr (useNotAlloc) {
            TLOAD<TLoadL2Hint::NotAllocKeep>(srcTile, srcGlobal);
        } else {
            TLOAD<TLoadL2Hint::NormalFirstVictim>(srcTile, srcGlobal);
        }
    }
    t1 = get_syscnt();
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE_MAT2GM<GlobalData, TileData>(dstGlobal, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
#endif
    LOG(t0);
    LOG(t1 - t0);
    LOG(g_opL2CacheHintCfg.l2Cacheoffset);
}

template <bool useNotAlloc>
__global__ AICORE void TLoadL2HintKernel(__gm__ float* out, __gm__ float* src, __gm__ uint64_t* gLog)
{
    // A2A3_TLOAD_MAT_*: ND float 1,1,1,128,256
    RunTLoadND2ND<float, 1, 1, 1, 128, 256, 1, 1, 1, 128, 256, useNotAlloc>(out, src, gLog);
}

template <int32_t testKey>
void LaunchTLoadL2Hint(float* out, float* src, uint64_t* gLog, void* stream)
{
    if constexpr (testKey == 1) {
        TLoadL2HintKernel<false><<<1, nullptr, stream>>>(out, src, gLog);
    } else if constexpr (testKey == 2) {
        TLoadL2HintKernel<true><<<1, nullptr, stream>>>(out, src, gLog);
    }
}

template void LaunchTLoadL2Hint<1>(float* out, float* src, uint64_t* gLog, void* stream);
template void LaunchTLoadL2Hint<2>(float* out, float* src, uint64_t* gLog, void* stream);
