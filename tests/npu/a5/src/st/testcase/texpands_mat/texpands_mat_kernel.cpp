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


template <typename T, int Rows, int Cols>
AICORE inline void runTSetValue_Tile(__gm__ T *out, __gm__ T *src0, T value)
{
    // 动态写法
    using NDValidShape = TileShape2D<T, -1, -1, Layout::ND>;
    using NDWholeShape = BaseShape2D<T, -1, -1, Layout::ND>;
    NDValidShape ndValidShape(Rows, Cols);
    NDWholeShape ndWholeShape(Rows, Cols);
    using GlobalDataSrc0 = GlobalTensor<T, NDValidShape, NDWholeShape, Layout::ND>;
    GlobalDataSrc0 src0Global(src0, ndValidShape, ndWholeShape);

    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, Rows, Cols>,
                     pto::Stride<1 * Rows * Cols, 1 * Rows * Cols, Rows * Cols, Cols, 1>, Layout::ND>;

    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, T, Rows, Cols, BLayout::RowMajor, Rows, Cols, SLayout::NoneBox>;
    using TileUBData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    TileUBData srcTile(Rows, Cols);
    TASSIGN(srcTile, 0x0);

    TileMatAData aMatTile;
    TASSIGN(aMatTile, 0x0);

    __cbuf__ T *srcMatAddr = aMatTile.data();
    __ubuf__ T *srcUbAddr = srcTile.data();
    __gm__ T *outAddr = dstGlobal.data();

    /*************************************TLOAD****************************************/
    TLOAD<TileMatAData, GlobalDataSrc0>(aMatTile, src0Global);
    uint8_t syncID = 0;
    TSET_VALUE<TileMatAData>(aMatTile, value);  //MTE2

    // L1 -> UB : AIC
#if defined(__DAV_CUBE__)
    uint16_t blockCount = 1;
    uint16_t blockLen = Rows * Cols * sizeof(T) / 32;
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
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
    wait_intra_block(PIPE_MTE3,
                     syncID); // veccore0 id0 correspond cubecore id is id0,  veccore1 id0 correspond cubecore id is 16
    TSTORE(dstGlobal, srcTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}


template <typename T, int format, int shape0, int shape1, int shape2, int shape3, int shape4>
__global__ AICORE void TSETVALUE_KERNEL(__gm__ uint8_t *out, __gm__ uint8_t *src0, T value)
{
    if constexpr (format == 0) { // Tile
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            runTSetValue_Tile<bfloat16_t, shape0, shape1>(reinterpret_cast<__gm__ bfloat16_t *>(out), reinterpret_cast<__gm__ bfloat16_t *>(src0), 7);
        } else {
            runTSetValue_Tile<T, shape0, shape1>(reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), value);
        }
    } else if constexpr (format == 1) { // convTile
        // runTSetValue_ConvTile<T, shape0, shape1, shape2, shape3, shape4>(reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src0), value);
    }
}

template <int32_t testKey>
void launchTSETVALUE(uint8_t *out, uint8_t *src0, void *stream)
{
    if constexpr (testKey == 1) {
        TSETVALUE_KERNEL<half, 0, 128, 128, 0, 0, 0><<<1, nullptr, stream>>>(out, src0, half(2));
    } else if constexpr (testKey == 2) {
        TSETVALUE_KERNEL<int16_t, 0, 32, 64, 0, 0, 0><<<1, nullptr, stream>>>(out, src0, int16_t(5));
    } else if constexpr (testKey == 3) {
        TSETVALUE_KERNEL<float, 0, 32, 32, 0, 0, 0><<<1, nullptr, stream>>>(out, src0, float(3));
    } else if constexpr (testKey == 4) {
        TSETVALUE_KERNEL<int8_t, 0, 32, 32, 0, 0, 0><<<1, nullptr, stream>>>(out, src0, int8_t(1));
    } else if constexpr (testKey == 5) {
        // uint16_t represent bfloat16
        TSETVALUE_KERNEL<uint16_t, 0, 256, 256, 0, 0, 0><<<1, nullptr, stream>>>(out, src0, 7);
    } else if constexpr (testKey == 6) {
        TSETVALUE_KERNEL<half, 1, 2, 32, 14, 14, 8><<<1, nullptr, stream>>>(out, src0, half(3));
    } else if constexpr (testKey == 7) {
        TSETVALUE_KERNEL<float, 1, 2, 32, 14, 14, 8><<<1, nullptr, stream>>>(out, src0, float(4));
    }
}

template void launchTSETVALUE<1>(uint8_t *out, uint8_t *src0, void *stream);
template void launchTSETVALUE<2>(uint8_t *out, uint8_t *src0, void *stream);
template void launchTSETVALUE<3>(uint8_t *out, uint8_t *src0, void *stream);
template void launchTSETVALUE<4>(uint8_t *out, uint8_t *src0, void *stream);
template void launchTSETVALUE<5>(uint8_t *out, uint8_t *src0, void *stream);
template void launchTSETVALUE<6>(uint8_t *out, uint8_t *src0, void *stream);
template void launchTSETVALUE<7>(uint8_t *out, uint8_t *src0, void *stream);
