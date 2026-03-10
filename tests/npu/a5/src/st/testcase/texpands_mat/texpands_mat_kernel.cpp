/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <limits>
#include <algorithm>

using namespace std;
using namespace pto;

template <typename T, int Rows, int Cols>
AICORE inline void runTexpands_Tile(__gm__ T *out, T value)
{
    using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, Rows, Cols>,
                                    pto::Stride<1 * Rows * Cols, 1 * Rows * Cols, Rows * Cols, Cols, 1>>;
    GlobalData dstGlobal(out);

    using TileData = Tile<TileType::Mat, T, Rows, Cols, BLayout::RowMajor, Rows, Cols, SLayout::NoneBox>;
    TileData MatTile;
    TASSIGN(MatTile, 0x0);

    using TileUBData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, Rows, Cols>;
    TileUBData srcTile;
    TASSIGN(srcTile, 0x0);

    __cbuf__ T *srcMatAddr = MatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    uint8_t syncID = 0;

#if defined(__DAV_CUBE__)
    TEXPANDS<TileData>(MatTile, value);  //MTE2
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    uint16_t blockCount = 1;
    uint16_t blockLen = Rows * Cols * sizeof(T) / 32;
    // L1 -> UB : AIC
    copy_cbuf_to_ubuf((__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 0, blockCount, blockLen, 0,
                      0); // move to vector
                          // core0
    copy_cbuf_to_ubuf((__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 1, blockCount, blockLen, 0,
                      0); // move to vector
                          // core1
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
#endif

#if defined(__DAV_VEC__)
    // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, int N, int C1, int H, int W, int C0>
AICORE inline void runTexpands_ConvTile(__gm__ T *out, T value)
{
    constexpr int elementSize = N * C1 * H * W * C0;
    constexpr int bufferSizeA = elementSize * sizeof(T);
    constexpr int reshapeRow = N;
    constexpr int reshapeCol = C1 * H * W * C0;
    using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, reshapeRow, reshapeCol>,
                                    pto::Stride<reshapeRow * reshapeCol, reshapeRow * reshapeCol, reshapeRow * reshapeCol, reshapeCol, 1>>;
    GlobalData dstGlobal(out);

    
    using TileData = ConvTile<TileType::Mat, T, bufferSizeA, Layout::NC1HWC0,
                                  pto::ConvTileShape<N, C1, H, W, C0>>;
    TileData MatTile;
    TASSIGN(MatTile, 0x0);

    using TileUBData = Tile<TileType::Vec, T, N, C1 * H * W * C0, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(N, C1 * H * W * C0);
    TASSIGN(srcTile, 0x0);
    __cbuf__ T *srcMatAddr = MatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    uint8_t syncID = 0;

    
#if defined(__DAV_CUBE__)
    TEXPANDS<TileData>(MatTile, value);  //MTE2
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // L1 -> UB : AIC
    uint16_t blockCount = 1;
    uint16_t blockLen = elementSize * sizeof(T) / 32;
    copy_cbuf_to_ubuf((__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 0, blockCount, blockLen, 0,
                      0); // move to vector
                          // core0
    copy_cbuf_to_ubuf((__ubuf__ void *)srcUbAddr, (__cbuf__ void *)srcMatAddr, 1, blockCount, blockLen, 0,
                      0); // move to vector
                          // core1
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
#endif

#if defined(__DAV_VEC__)
    // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <typename T, int format, int shape0, int shape1, int shape2, int shape3, int shape4>
__global__ AICORE void TEXPANDS_KERNEL(__gm__ uint8_t *out, T value)
{
    if constexpr (format == 0) { // Tile
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            runTexpands_Tile<bfloat16_t, shape0, shape1>(reinterpret_cast<__gm__ bfloat16_t *>(out), bfloat16_t(0));
        } else {
            runTexpands_Tile<T, shape0, shape1>(reinterpret_cast<__gm__ T *>(out), value);
        }
    } else if constexpr (format == 1) { // convTile
        runTexpands_ConvTile<T, shape0, shape1, shape2, shape3, shape4>(reinterpret_cast<__gm__ T *>(out), value);
    }
}

template <int32_t testKey>
void launchTEXPANDS(uint8_t *out, void *stream)
{
    if constexpr (testKey == 1) {
        TEXPANDS_KERNEL<half, 0, 128, 128, 0, 0, 0><<<1, nullptr, stream>>>(out, half(2));
    } else if constexpr (testKey == 2) {
        TEXPANDS_KERNEL<int16_t, 0, 32, 64, 0, 0, 0><<<1, nullptr, stream>>>(out, int16_t(5));
    } else if constexpr (testKey == 3) {
        TEXPANDS_KERNEL<float, 0, 32, 32, 0, 0, 0><<<1, nullptr, stream>>>(out, float(3));
    } else if constexpr (testKey == 4) {
        TEXPANDS_KERNEL<int8_t, 0, 32, 32, 0, 0, 0><<<1, nullptr, stream>>>(out, int8_t(1));
    } else if constexpr (testKey == 5) {
        // uint16_t represent bfloat16
        TEXPANDS_KERNEL<uint16_t, 0, 256, 256, 0, 0, 0><<<1, nullptr, stream>>>(out, 0);
    } else if constexpr (testKey == 6) {
        TEXPANDS_KERNEL<half, 1, 1, 16, 7, 7, 16><<<1, nullptr, stream>>>(out, half(3));
    } else if constexpr (testKey == 7) {
        TEXPANDS_KERNEL<float, 1, 2, 3, 2, 3, 8><<<1, nullptr, stream>>>(out, float(4));
    }
}

template void launchTEXPANDS<1>(uint8_t *out, void *stream);
template void launchTEXPANDS<2>(uint8_t *out, void *stream);
template void launchTEXPANDS<3>(uint8_t *out, void *stream);
template void launchTEXPANDS<4>(uint8_t *out, void *stream);
template void launchTEXPANDS<5>(uint8_t *out, void *stream);
template void launchTEXPANDS<6>(uint8_t *out, void *stream);
template void launchTEXPANDS<7>(uint8_t *out, void *stream);