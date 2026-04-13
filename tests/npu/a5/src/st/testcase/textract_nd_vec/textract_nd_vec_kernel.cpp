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
        RunTExtractNDVec<float, 16, 16, 8, 8, 0, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstInitIn));
    } else if constexpr (testKey == 2) {
        RunTExtractNDVec<float, 16, 16, 8, 8, 4, 8><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstInitIn));
    } else if constexpr (testKey == 3) {
        RunTExtractNDVec<half, 32, 32, 16, 16, 8, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(srcIn), reinterpret_cast<half *>(dstInitIn));
    } else if constexpr (testKey == 4) {
        RunTExtractNDVec<int8_t, 64, 64, 32, 32, 0, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(srcIn), reinterpret_cast<int8_t *>(dstInitIn));
    } else if constexpr (testKey == 5) {
        RunTExtractNDVec<half, 32, 48, 16, 16, 4, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(srcIn), reinterpret_cast<half *>(dstInitIn));
    } else if constexpr (testKey == 6) {
        RunTExtractNDVec<float, 16, 24, 8, 8, 3, 8><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstInitIn));
    } else if constexpr (testKey == 7) {
        RunTExtractNDVec<float, 16, 24, 8, 8, 0, 3><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstInitIn));
    } else if constexpr (testKey == 8) {
        RunTExtractNDVec<half, 16, 48, 8, 16, 2, 5><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(srcIn), reinterpret_cast<half *>(dstInitIn));
    } else if constexpr (testKey == 9) {
        RunTExtractNDVec<int8_t, 64, 64, 32, 32, 0, 7><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(srcIn), reinterpret_cast<int8_t *>(dstInitIn));
    }
}

template void launchTExtractNDVec<1>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);
template void launchTExtractNDVec<2>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);
template void launchTExtractNDVec<3>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);
template void launchTExtractNDVec<4>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);
template void launchTExtractNDVec<5>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);
template void launchTExtractNDVec<6>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);
template void launchTExtractNDVec<7>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);
template void launchTExtractNDVec<8>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);
template void launchTExtractNDVec<9>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);

template <typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t IdxRow, uint32_t IdxCol>
__global__ AICORE void RunTExtractNDVecScalar(__gm__ T *out, __gm__ T *srcIn, __gm__ T *dstInitIn)
{
    constexpr uint32_t MinAlignedCols = 32 / sizeof(T);
    using SrcShape = pto::Shape<1, 1, 1, SrcRows, SrcCols>;
    using SrcStride = pto::Stride<SrcRows * SrcCols, SrcRows * SrcCols, SrcRows * SrcCols, SrcCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    using DstShape = pto::Shape<1, 1, 1, 1, MinAlignedCols>;
    using DstStride = pto::Stride<MinAlignedCols, MinAlignedCols, MinAlignedCols, MinAlignedCols, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;

    using SrcVec = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor>;
    using DstLoadVec = Tile<TileType::Vec, T, 1, MinAlignedCols, BLayout::RowMajor>;
    using DstExtractVec = Tile<TileType::Vec, T, 1, MinAlignedCols, BLayout::RowMajor, 1, 1>;

    SrcVec srcTile;
    DstLoadVec dstLoad;
    DstExtractVec dstExtract;

    TASSIGN(srcTile, 0x0);
    constexpr uint32_t srcSize = SrcRows * SrcCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcSize + 0xFF) / 0x100) * 0x100;
    TASSIGN(dstLoad, dstAssignAddr);
    TASSIGN(dstExtract, dstAssignAddr);

    SrcGlobal srcGlobal(srcIn);
    DstGlobal dstInitGlobal(dstInitIn);
    DstGlobal outGlobal(out);

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    TLOAD(dstLoad, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TEXTRACT(dstExtract, srcTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstLoad);
#endif
}

template <int32_t testKey>
void launchTExtractNDVecScalar(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream)
{
    if constexpr (testKey == 1) {
        RunTExtractNDVecScalar<float, 16, 16, 5, 7><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(srcIn), reinterpret_cast<float *>(dstInitIn));
    } else if constexpr (testKey == 2) {
        RunTExtractNDVecScalar<half, 32, 32, 10, 15><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(srcIn), reinterpret_cast<half *>(dstInitIn));
    } else if constexpr (testKey == 3) {
        RunTExtractNDVecScalar<int8_t, 64, 64, 20, 30><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(srcIn), reinterpret_cast<int8_t *>(dstInitIn));
    }
}

template void launchTExtractNDVecScalar<1>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);
template void launchTExtractNDVecScalar<2>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);
template void launchTExtractNDVecScalar<3>(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);

template <typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t DstRows, uint32_t DstCols, uint32_t IdxRow,
          uint32_t IdxCol>
__global__ AICORE void RunTExtractNZVecToVec(__gm__ T *out, __gm__ T *src)
{
    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));

    using SrcShapeDim5 = pto::Shape<1, 1, 1, SrcRows, SrcCols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, SrcCols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    using OutShapeDim5 = pto::Shape<1, DstCols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 =
        pto::Stride<DstCols / c0Size * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcNDTile = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor, -1, -1>;
    using SrcNZTile = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::ColMajor, SrcRows, SrcCols, SLayout::RowMajor>;
    using DstNZTile = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, DstRows, DstCols, SLayout::RowMajor>;

    SrcNDTile srcNDTile(SrcRows, SrcCols);
    SrcNZTile srcNZTile;
    DstNZTile dstNZTile;

    TASSIGN(srcNDTile, 0x0);
    TASSIGN(srcNZTile, 0x10000);
    TASSIGN(dstNZTile, 0x20000);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);

#if defined(__DAV_VEC__)
    TLOAD(srcNDTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMOV(srcNZTile, srcNDTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TEXTRACT(dstNZTile, srcNZTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstNZTile);
#endif
}

template <int32_t testKey>
void launchTExtractNZVecToVec(uint64_t *out, uint64_t *src, void *stream)
{
    if constexpr (testKey == 1) {
        RunTExtractNZVecToVec<float, 16, 32, 16, 32, 0, 0>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src));
    } else if constexpr (testKey == 2) {
        RunTExtractNZVecToVec<float, 32, 32, 16, 32, 16, 0>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src));
    } else if constexpr (testKey == 3) {
        RunTExtractNZVecToVec<half, 32, 64, 16, 32, 16, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src));
    }
}

template void launchTExtractNZVecToVec<1>(uint64_t *out, uint64_t *src, void *stream);
template void launchTExtractNZVecToVec<2>(uint64_t *out, uint64_t *src, void *stream);
template void launchTExtractNZVecToVec<3>(uint64_t *out, uint64_t *src, void *stream);
