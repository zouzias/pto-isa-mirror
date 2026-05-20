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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t DstRows, uint32_t DstCols, uint32_t IdxRow,
          uint32_t IdxCol>
__global__ AICORE void RunTExtractNDVec(__gm__ T *out, __gm__ T *srcIn, __gm__ T *dstInitIn)
{
    using SrcShape = pto::Shape<1, 1, 1, SrcRows, SrcCols>;
    using SrcStride = pto::Stride<SrcRows * SrcCols, SrcRows * SrcCols, SrcRows * SrcCols, SrcCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    using DstShape = pto::Shape<1, 1, 1, DstRows, DstCols>;
    using DstStride = pto::Stride<DstRows * DstCols, DstRows * DstCols, DstRows * DstCols, DstCols, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;

    using SrcVec = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor>;
    using DstVec = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::RowMajor>;

    SrcVec srcTile;
    DstVec dstTile;

    TASSIGN(srcTile, 0x0);
    constexpr uint32_t srcSize = SrcRows * SrcCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcSize + 0xFF) / 0x100) * 0x100;
    TASSIGN(dstTile, dstAssignAddr);

    SrcGlobal srcGlobal(srcIn);
    DstGlobal dstInitGlobal(dstInitIn);
    DstGlobal outGlobal(out);

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    TLOAD(dstTile, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TEXTRACT(dstTile, srcTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstTile);
#endif
}

template <int32_t testKey>
void launchTExtractNDVec(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream)
{
    if constexpr (testKey == 1) {
        RunTExtractNDVec<float, 32, 512, 32, 512, 0, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstInitIn));
    } else if constexpr (testKey == 2) {
        RunTExtractNDVec<float, 512, 32, 512, 32, 0, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstInitIn));
    } else if constexpr (testKey == 3) {
        RunTExtractNDVec<float, 128, 128, 128, 128, 0, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstInitIn));
    }
}

template void launchTExtractNDVec<1>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);
template void launchTExtractNDVec<2>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);
template void launchTExtractNDVec<3>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);
