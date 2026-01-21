
/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <iostream>
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include <pto/npu/a5/custom/TInsertCustom.hpp>


using namespace std;
using namespace pto;

template <typename T, uint32_t Rows, uint32_t Cols, TInsertMode Mode = TInsertMode::NZ>
AICORE void runTInsertCustom(__gm__ T *out, __gm__ T *src)
{
    using SrcShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, Rows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 = pto::Stride<Cols / c0Size * c0Size * Rows, Rows * c0Size,
        FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using TmpVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using DstVecTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTile = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcVecTile srcTile(Rows, Cols);
    TmpVecTile tmpTile(Rows, Cols);
    DstVecTile dstTile(Rows, Cols);
    MatTile matTile(Rows, Cols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x10000);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(matTile, 0x0);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;

    constexpr uint32_t alignedRow = ((Rows + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    constexpr uint32_t burstNum = Cols / c0Size;
    constexpr uint16_t burstLen = (alignedRow * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;

    constexpr uint16_t dstGap = (Mode == TInsertMode::NZ) ? 0 : 1;

    __cbuf__ T *matAddr = matTile.data();
    __ubuf__ T *dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMOV(tmpTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TINSERT_CUSTOM<Mode>(matTile, tmpTile);
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    wait_intra_block(PIPE_MTE1, syncId);
    wait_intra_block(PIPE_MTE1, syncId + eventIdNum);

    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);

    copy_cbuf_to_ubuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, 0, burstNum, burstLen, dstGap, 0);
    copy_cbuf_to_ubuf((__ubuf__ void *)dstUbAddr, (__cbuf__ void *)matAddr, 1, burstNum, burstLen, dstGap, 0);

    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);

    set_intra_block(PIPE_MTE1, syncId);
    set_intra_block(PIPE_MTE1, syncId + eventIdNum);
#endif

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    TSTORE(dstGlobal, dstTile);
#endif
    out = dstGlobal.data();
}

template <typename T, uint32_t Rows, uint32_t Cols, TInsertMode Mode = TInsertMode::NZ>
__global__ AICORE void launchTInsertCustomKernel(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    runTInsertCustom<T, Rows, Cols, Mode>(
        reinterpret_cast<__gm__ T *>(out), reinterpret_cast<__gm__ T *>(src));
}

template <int32_t testKey>
void launchTInsertCustom(uint64_t *out, uint64_t *src, void* stream)
{
    cout << "launchTInsertCustom start!" << endl;
    if constexpr (testKey == 1) {
        launchTInsertCustomKernel<float, 16, 32, TInsertMode::NZ><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTInsertCustomKernel<float, 16, 32, TInsertMode::NZ_PLUS_1><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTInsertCustomKernel<float, 32, 64, TInsertMode::NZ_PLUS_1><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTInsertCustomKernel<int32_t, 32, 32, TInsertMode::NZ_PLUS_1><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTInsertCustomKernel<float, 32, 32, TInsertMode::SPLIT2_NZ_PLUS_1><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 6) {
        launchTInsertCustomKernel<float, 32, 32, TInsertMode::SPLIT4_NZ_PLUS_1><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 7) {
        launchTInsertCustomKernel<float, 64, 64, TInsertMode::SPLIT4_NZ_PLUS_1><<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTInsertCustom<1>(uint64_t *out, uint64_t *src, void* stream);
template void launchTInsertCustom<2>(uint64_t *out, uint64_t *src, void* stream);
template void launchTInsertCustom<3>(uint64_t *out, uint64_t *src, void* stream);
template void launchTInsertCustom<4>(uint64_t *out, uint64_t *src, void* stream);
template void launchTInsertCustom<5>(uint64_t *out, uint64_t *src, void* stream);
template void launchTInsertCustom<6>(uint64_t *out, uint64_t *src, void* stream);
template void launchTInsertCustom<7>(uint64_t *out, uint64_t *src, void* stream);