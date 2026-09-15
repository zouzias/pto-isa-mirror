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

using namespace pto;

AICORE inline void test_copy_cbuf_to_ubuf(
    __ubuf__ void* dst, __cbuf__ void* src, uint8_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcGap,
    uint16_t dstGap)
{
    constexpr uint32_t CBUF_UB_BURST_UNIT = 32;
    uint16_t loop2SrcStride = static_cast<uint16_t>(lenBurst + srcGap);
    uint64_t fixpNzPara =
        static_cast<uint64_t>(1) | (static_cast<uint64_t>(loop2SrcStride) << 16) | (static_cast<uint64_t>(1) << 32);
    set_fixp_nz_para(fixpNzPara);
    __ubuf__ uint8_t* dstP = reinterpret_cast<__ubuf__ uint8_t*>(dst);
    __cbuf__ uint8_t* srcP = reinterpret_cast<__cbuf__ uint8_t*>(src);
    uint32_t srcStep = (lenBurst + srcGap) * CBUF_UB_BURST_UNIT;
    uint32_t dstStep = (lenBurst + dstGap) * CBUF_UB_BURST_UNIT;
    for (uint16_t i = 0; i < nBurst; ++i) {
        pto_copy_cbuf_to_ubuf(
            reinterpret_cast<__ubuf__ void*>(dstP + i * dstStep), reinterpret_cast<__cbuf__ void*>(srcP + i * srcStep),
            sid, 1, lenBurst, 0, 0);
    }
    set_fixp_nz_para(0);
}

template <
    typename TileMatAData, typename TileMatBData, typename RightTile, typename AccTile, typename GlobalDataSrc0,
    typename GlobalDataSrc1>
AICORE inline void LoadMatmulOnly(
    TileMatAData& aMatTile, TileMatBData& bMatTile, RightTile& bTile, AccTile& cTile, GlobalDataSrc0& src0Global,
    GlobalDataSrc1& src1Global)
{
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aMatTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
}

template <typename AType, typename CType, int M, int K, int N>
__global__ AICORE void RunTInsertAcc2Mat(__gm__ CType* out, __gm__ AType* src0, __gm__ AType* src1)
{
    using GlobalDataSrc0 = GlobalTensor<AType, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<AType, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, AType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using DstMatTile = Tile<TileType::Mat, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    DstMatTile dstMatTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);
    TASSIGN<(M * K + K * N) * sizeof(AType)>(dstMatTile);

    using RightTile = TileRight<AType, K, N, K, N>;
    using AccTile = TileAcc<CType, M, N, M, N>;
    RightTile bTile;
    AccTile cTile;
    TASSIGN<0x0>(bTile);
    TASSIGN<(M * K + K * N + M * N) * sizeof(AType)>(cTile);

    using DstVecTile = Tile<TileType::Vec, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    DstVecTile dstVecTile;
    TASSIGN<0x0>(dstVecTile);

    static constexpr uint16_t sGRows = 16;
    static constexpr uint16_t sGCols = 512 / (sGRows * sizeof(CType));
    static constexpr uint16_t kGRows = (M + sGRows - 1) / sGRows;
    static constexpr uint16_t kGCols = (N + sGCols - 1) / sGCols;
    using ShapeDim5 = Shape<1, kGCols, kGRows, sGRows, sGCols>;
    using StridDim5 =
        pto::Stride<kGCols * kGRows * sGCols * sGRows, kGRows * sGCols * sGRows, sGCols * sGRows, sGCols, 1>;
    using NZOutputGlobalData = GlobalTensor<CType, ShapeDim5, StridDim5, Layout::NZ>;
    NZOutputGlobalData dstGlobal(out);

    LoadMatmulOnly(aMatTile, bMatTile, bTile, cTile, src0Global, src1Global);

    TINSERT(dstMatTile, cTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));

    pipe_barrier(PIPE_FIX);

    constexpr uint32_t c0Size = 512 / (16 * sizeof(CType));
    constexpr uint16_t burstLen = M * c0Size * sizeof(CType) / 32;
    constexpr uint16_t burstNum = N / c0Size;
    __ubuf__ CType* dstUbAddr = dstVecTile.data();
    __cbuf__ CType* srcMatAddr = dstMatTile.data();
    test_copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcMatAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstVecTile);
}

template <typename AType, typename CType, int M, int K, int N>
__global__ AICORE void RunTInsertMat2Mat(__gm__ CType* out, __gm__ AType* src0, __gm__ AType* src1)
{
    using GlobalDataSrc0 = GlobalTensor<AType, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<AType, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, AType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using DstMatTile = Tile<TileType::Mat, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    DstMatTile dstMatTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);
    TASSIGN<(M * K + K * N) * sizeof(AType)>(dstMatTile);

    using RightTile = TileRight<AType, K, N, K, N>;
    using SrcMatTile = Tile<TileType::Mat, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;
    RightTile bTile;
    SrcMatTile cMatTile;
    TASSIGN<0x0>(bTile);
    TASSIGN<(M * K + K * N + M * N) * sizeof(AType)>(cMatTile);

    using DstVecTile = Tile<TileType::Vec, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    DstVecTile dstVecTile;
    TASSIGN<0x0>(dstVecTile);

    static constexpr uint16_t sGRows = 16;
    static constexpr uint16_t sGCols = 512 / (sGRows * sizeof(CType));
    static constexpr uint16_t kGRows = (M + sGRows - 1) / sGRows;
    static constexpr uint16_t kGCols = (N + sGCols - 1) / sGCols;
    using ShapeDim5 = Shape<1, kGCols, kGRows, sGRows, sGCols>;
    using StridDim5 =
        pto::Stride<kGCols * kGRows * sGCols * sGRows, kGRows * sGCols * sGRows, sGCols * sGRows, sGCols, 1>;
    using NZOutputGlobalData = GlobalTensor<CType, ShapeDim5, StridDim5, Layout::NZ>;
    NZOutputGlobalData dstGlobal(out);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cMatTile, aMatTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TINSERT(dstMatTile, cMatTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));

    pipe_barrier(PIPE_FIX);

    constexpr uint32_t c0Size = 512 / (16 * sizeof(CType));
    constexpr uint16_t burstLen = M * c0Size * sizeof(CType) / 32;
    constexpr uint16_t burstNum = N / c0Size;
    __ubuf__ CType* dstUbAddr = dstVecTile.data();
    __cbuf__ CType* srcMatAddr = dstMatTile.data();
    test_copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcMatAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstVecTile);
}

// Forward declaration - SdkL1CopyBlock is defined later in this file
AICORE inline void SdkL1CopyBlock(__cbuf__ int16_t* dst, __cbuf__ int16_t* src, uint64_t copyBytes);

template <typename T, int M, int N, bool useFixpNzPara>
__global__ AICORE void RunCbufToCbufTest(__gm__ T* out, __gm__ T* src)
{
    constexpr int c0 = 32 / sizeof(T);
    using SrcShape = pto::Shape<1, N / c0, M / 16, 16, c0>;
    using SrcStride = pto::Stride<M * N, (M / 16) * 16 * c0, 16 * c0, c0, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride, Layout::NZ>;
    using DstGlobal = GlobalTensor<T, SrcShape, SrcStride, Layout::NZ>;
    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out);

    using SrcMatTile = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    using DstMatTile = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    SrcMatTile srcMatTile;
    DstMatTile dstMatTile;
    TASSIGN<0x0>(srcMatTile);
    TASSIGN<M * N * sizeof(T)>(dstMatTile);

    using DstVecTile = Tile<TileType::Vec, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    DstVecTile dstVecTile;
    TASSIGN<0x0>(dstVecTile);

    TLOAD(srcMatTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);

    // Use SDK -pattern
    // is silently
    constexpr uint64_t BLOCK_BYTES = 512;
    constexpr uint64_t totalBytes = static_cast<uint64_t>(M) * N * sizeof(T);
    constexpr uint64_t numBlocks = totalBytes / BLOCK_BYTES;
    __cbuf__ int16_t* srcBase = reinterpret_cast<__cbuf__ int16_t*>(srcMatTile.data());
    __cbuf__ int16_t* dstBase = reinterpret_cast<__cbuf__ int16_t*>(dstMatTile.data());
    if constexpr (numBlocks == 1) {
        SdkL1CopyBlock(dstBase, srcBase, totalBytes);
    } else {
        for (uint64_t i = 0; i < numBlocks; i++) {
            SdkL1CopyBlock(
                dstBase + (i * BLOCK_BYTES) / sizeof(int16_t), srcBase + (i * BLOCK_BYTES) / sizeof(int16_t),
                BLOCK_BYTES);
        }
    }
    (void)useFixpNzPara;

    constexpr uint32_t c0Size = 512 / (16 * sizeof(T));
    constexpr uint16_t burstLen = M * c0Size * sizeof(T) / 32;
    constexpr uint16_t burstNum = N / c0Size;
    __ubuf__ T* dstUbAddr = dstVecTile.data();
    __cbuf__ T* srcCbAddr = dstMatTile.data();
    test_copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcCbAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstVecTile);
}

template <typename T, int M, int N, bool withTinsert>
__global__ AICORE void RunTInsertMat2MatLoad(__gm__ T* out, __gm__ T* src)
{
    constexpr int c0 = 32 / sizeof(T);
    using Shape5 = pto::Shape<1, N / c0, M / 16, 16, c0>;
    using Stride5 = pto::Stride<M * N, (M / 16) * 16 * c0, 16 * c0, c0, 1>;
    using GlobalData = GlobalTensor<T, Shape5, Stride5, Layout::NZ>;
    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    using SrcMatTile = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    using DstMatTile = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    SrcMatTile srcMatTile;
    DstMatTile dstMatTile;
    TASSIGN<0x0>(srcMatTile);
    TASSIGN<M * N * sizeof(T)>(dstMatTile);

    using DstVecTile = Tile<TileType::Vec, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    DstVecTile dstVecTile;
    TASSIGN<0x0>(dstVecTile);

    TLOAD(srcMatTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    constexpr uint32_t c0Size = 512 / (16 * sizeof(T));
    constexpr uint16_t burstLen = M * c0Size * sizeof(T) / 32;
    constexpr uint16_t burstNum = N / c0Size;
    __ubuf__ T* dstUbAddr = dstVecTile.data();

    if constexpr (withTinsert) {
        TINSERT(dstMatTile, srcMatTile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
        pipe_barrier(PIPE_FIX);
        test_copy_cbuf_to_ubuf(
            (__ubuf__ void*)dstUbAddr, (__cbuf__ void*)dstMatTile.data(), 0, burstNum, burstLen, 0, 0);
    } else {
        test_copy_cbuf_to_ubuf(
            (__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcMatTile.data(), 0, burstNum, burstLen, 0, 0);
    }

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstVecTile);
}

//
//
//
AICORE inline void SdkL1CopyBlock(__cbuf__ int16_t* dst, __cbuf__ int16_t* src, uint64_t copyBytes)
{
    constexpr uint64_t C0_BYTE_SIZE = 32;
    uint64_t copy_times = copyBytes / C0_BYTE_SIZE;

    //
    //
    //
    uint64_t xm = C0_BYTE_SIZE | (1ULL << 40) | (static_cast<uint64_t>(fixp_trans_mode_t::LOOP_ENHANCE) << 61);
    uint64_t xt = C0_BYTE_SIZE | (1ULL << 40);
    uint64_t config = C0_BYTE_SIZE | (1ULL << 40);
    uint64_t nz_para_config = copy_times | (1ULL << 16) | (1ULL << 32) | (1ULL << 48);

    pipe_barrier(PIPE_ALL);
    set_loopenhance_para(config);
    set_fixp_nz_para(nz_para_config);
    fix_cbuf_to_cbuf(dst, src, xm, xt);
    pipe_barrier(PIPE_ALL);
    set_fixp_nz_para(0);
}

//
//
//
//
//
//
//
template <typename T, int M, int N>
__global__ AICORE void RunCbufToCbufSdkMode(__gm__ T* out, __gm__ T* src)
{
    constexpr int c0 = 32 / sizeof(T);
    using SrcShape = pto::Shape<1, N / c0, M / 16, 16, c0>;
    using SrcStride = pto::Stride<M * N, (M / 16) * 16 * c0, 16 * c0, c0, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride, Layout::NZ>;
    using DstGlobal = GlobalTensor<T, SrcShape, SrcStride, Layout::NZ>;
    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out);

    using SrcMatTile = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    using DstMatTile = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    SrcMatTile srcMatTile;
    DstMatTile dstMatTile;
    TASSIGN<0x0>(srcMatTile);
    TASSIGN<M * N * sizeof(T)>(dstMatTile);

    using DstVecTile = Tile<TileType::Vec, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    DstVecTile dstVecTile;
    TASSIGN<0x0>(dstVecTile);

    TLOAD(srcMatTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);

    //
    constexpr uint64_t BLOCK_BYTES = 512;
    constexpr uint64_t totalBytes = static_cast<uint64_t>(M) * N * sizeof(T);
    constexpr uint64_t numBlocks = totalBytes / BLOCK_BYTES;
    __cbuf__ int16_t* srcBase = reinterpret_cast<__cbuf__ int16_t*>(srcMatTile.data());
    __cbuf__ int16_t* dstBase = reinterpret_cast<__cbuf__ int16_t*>(dstMatTile.data());

    if constexpr (numBlocks == 1) {
        SdkL1CopyBlock(dstBase, srcBase, totalBytes);
    } else {
        for (uint64_t i = 0; i < numBlocks; i++) {
            SdkL1CopyBlock(
                dstBase + (i * BLOCK_BYTES) / sizeof(int16_t), srcBase + (i * BLOCK_BYTES) / sizeof(int16_t),
                BLOCK_BYTES);
        }
    }

    constexpr uint32_t c0Size = 512 / (16 * sizeof(T));
    constexpr uint16_t burstLen = M * c0Size * sizeof(T) / 32;
    constexpr uint16_t burstNum = N / c0Size;
    __ubuf__ T* dstUbAddr = dstVecTile.data();
    __cbuf__ T* srcCbAddr = dstMatTile.data();
    test_copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcCbAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstVecTile);
}

//
//
//
template <typename AType, typename CType, int M, int K, int N>
__global__ AICORE void RunAccDirectToUb(__gm__ CType* out, __gm__ AType* src0, __gm__ AType* src1)
{
    using GlobalDataSrc0 = GlobalTensor<AType, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<AType, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, AType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);

    using RightTile = TileRight<AType, K, N, K, N>;
    using AccTile = TileAcc<CType, M, N, M, N>;
    RightTile bTile;
    AccTile cTile;
    TASSIGN<0x0>(bTile);
    TASSIGN<(M * K + K * N + M * N) * sizeof(AType)>(cTile);

    using DstVecTile = Tile<TileType::Vec, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    DstVecTile dstVecTile;
    TASSIGN<0x0>(dstVecTile);

    static constexpr uint16_t sGRows = 16;
    static constexpr uint16_t sGCols = 512 / (sGRows * sizeof(CType));
    static constexpr uint16_t kGRows = (M + sGRows - 1) / sGRows;
    static constexpr uint16_t kGCols = (N + sGCols - 1) / sGCols;
    using ShapeDim5 = Shape<1, kGCols, kGRows, sGRows, sGCols>;
    using StridDim5 =
        pto::Stride<kGCols * kGRows * sGRows * sGCols, kGRows * sGRows * sGCols, sGRows * sGCols, sGCols, 1>;
    using NZOutputGlobalData = GlobalTensor<CType, ShapeDim5, StridDim5, Layout::NZ>;
    NZOutputGlobalData dstGlobal(out);

    LoadMatmulOnly(aMatTile, bMatTile, bTile, cTile, src0Global, src1Global);

    // fix_cbuf_to_ubuf NORMAL_DMA
    constexpr int32_t c0Size = 32 / sizeof(CType);
    constexpr uint32_t copyBytes = c0Size * sizeof(CType);
    constexpr uint32_t innerRows = AccTile::InnerRows;
    constexpr uint32_t innerCols = AccTile::InnerCols;
    constexpr uint32_t innerNumel = AccTile::InnerNumel;
    constexpr uint32_t blockNumRow = AccTile::Rows / innerRows;
    constexpr uint32_t blockNumCol = AccTile::Cols / innerCols;

    __cbuf__ CType* srcData = cTile.data();
    __ubuf__ CType* dstAddr = dstVecTile.data();
    __ubuf__ uint8_t* dstBytes = reinterpret_cast<__ubuf__ uint8_t*>(dstAddr);
    __cbuf__ uint8_t* srcBytes = reinterpret_cast<__cbuf__ uint8_t*>(srcData);

    SetFixpNzPara(1, 1, 1, 0);
    for (uint32_t row = 0; row < static_cast<uint32_t>(M); ++row) {
        uint32_t srcBlockRow = row / innerRows;
        uint32_t srcInnerRow = row % innerRows;
        for (uint32_t colBlk = 0; colBlk < static_cast<uint32_t>(N) / c0Size; ++colBlk) {
            uint32_t srcCol = colBlk * c0Size;
            uint32_t srcBlockCol = srcCol / innerCols;
            uint32_t srcInnerCol = srcCol % innerCols;
            uint32_t srcElemOff =
                (blockNumRow * srcBlockCol + srcBlockRow) * innerNumel + srcInnerRow * innerCols + srcInnerCol;
            uint32_t srcByteOff = srcElemOff * sizeof(CType);
            uint32_t dstByteOff = (row * N + colBlk * c0Size) * sizeof(CType);
            fix_cbuf_to_ubuf(
                dstBytes + dstByteOff, srcBytes + srcByteOff, copyBytes, copyBytes, fixp_trans_mode_t::NORMAL_DMA,
                static_cast<uint64_t>(0), 1);
        }
    }
    SetFixpNzPara(0, 0, 0, 0);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstVecTile);
}

// matmul to Acc tile -> Acc CBuf -> UBuf (fix_cbuf_to_ubuf) -> Mat CBuf (copy_ubuf_to_cbuf) -> UBuf -> TSTORE
template <typename AType, typename CType, int M, int K, int N>
__global__ AICORE void RunAccViaUb(__gm__ CType* out, __gm__ AType* src0, __gm__ AType* src1)
{
    using GlobalDataSrc0 = GlobalTensor<AType, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<AType, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, AType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using DstMatTile = Tile<TileType::Mat, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    DstMatTile dstMatTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);
    TASSIGN<(M * K + K * N) * sizeof(AType)>(dstMatTile);

    using RightTile = TileRight<AType, K, N, K, N>;
    using AccTile = TileAcc<CType, M, N, M, N>;
    RightTile bTile;
    AccTile cTile;
    TASSIGN<0x0>(bTile);
    TASSIGN<(M * K + K * N + M * N) * sizeof(AType)>(cTile);

    // UBuf relay buffer Acc -> UB -> Mat
    using RelayTile = Tile<TileType::Vec, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    RelayTile relayTile;
    TASSIGN<0x0>(relayTile);

    using DstVecTile = Tile<TileType::Vec, CType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    DstVecTile dstVecTile;
    TASSIGN<M * N * sizeof(CType)>(dstVecTile);

    static constexpr uint16_t sGRows = 16;
    static constexpr uint16_t sGCols = 512 / (sGRows * sizeof(CType));
    static constexpr uint16_t kGRows = (M + sGRows - 1) / sGRows;
    static constexpr uint16_t kGCols = (N + sGCols - 1) / sGCols;
    using ShapeDim5 = Shape<1, kGCols, kGRows, sGRows, sGCols>;
    using StridDim5 =
        pto::Stride<kGCols * kGRows * sGRows * sGCols, kGRows * sGRows * sGCols, sGRows * sGCols, sGCols, 1>;
    using NZOutputGlobalData = GlobalTensor<CType, ShapeDim5, StridDim5, Layout::NZ>;
    NZOutputGlobalData dstGlobal(out);

    LoadMatmulOnly(aMatTile, bMatTile, bTile, cTile, src0Global, src1Global);

    // copy acc tile from CBuf to UBuf (relayTile) in NZ format using fix_cbuf_to_ubuf
    constexpr int32_t c0Size = 32 / sizeof(CType);
    constexpr uint32_t copyBytes = c0Size * sizeof(CType);
    constexpr uint32_t innerRows = AccTile::InnerRows;
    constexpr uint32_t innerCols = AccTile::InnerCols;
    constexpr uint32_t innerNumel = AccTile::InnerNumel;
    constexpr uint32_t blockNumRow = AccTile::Rows / innerRows;
    constexpr uint32_t blockNumCol = AccTile::Cols / innerCols;

    __cbuf__ CType* srcData = cTile.data();
    __ubuf__ CType* relayAddr = relayTile.data();
    __ubuf__ uint8_t* relayBytes = reinterpret_cast<__ubuf__ uint8_t*>(relayAddr);
    __cbuf__ uint8_t* srcBytes = reinterpret_cast<__cbuf__ uint8_t*>(srcData);

    // Copy in NZ format: same offsets in source CBuf and destination UBuf
    SetFixpNzPara(1, 1, 1, 0);
    for (uint32_t row = 0; row < static_cast<uint32_t>(M); ++row) {
        uint32_t srcBlockRow = row / innerRows;
        uint32_t srcInnerRow = row % innerRows;
        for (uint32_t colBlk = 0; colBlk < static_cast<uint32_t>(N) / c0Size; ++colBlk) {
            uint32_t srcBlockCol = colBlk; // c0Size == innerCols for half
            uint32_t srcElemOff = (blockNumRow * srcBlockCol + srcBlockRow) * innerNumel + srcInnerRow * innerCols;
            uint32_t byteOff = srcElemOff * sizeof(CType);
            fix_cbuf_to_ubuf(
                relayBytes + byteOff, srcBytes + byteOff, copyBytes, copyBytes, fixp_trans_mode_t::NORMAL_DMA,
                static_cast<uint64_t>(0), 1);
        }
    }
    SetFixpNzPara(0, 0, 0, 0);

    // copy from UB to Mat CBuf
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    constexpr uint32_t matC0Size = 512 / (16 * sizeof(CType));
    constexpr uint16_t burstLen = M * matC0Size * sizeof(CType) / 32;
    constexpr uint16_t burstNum = N / matC0Size;
    __ubuf__ uint8_t* ubSrc = reinterpret_cast<__ubuf__ uint8_t*>(relayAddr);
    __cbuf__ uint8_t* cbDst = reinterpret_cast<__cbuf__ uint8_t*>(dstMatTile.data());
    constexpr uint32_t CBUF_UB_BURST_UNIT = 32;
    uint32_t srcStep = burstLen * CBUF_UB_BURST_UNIT;
    uint32_t dstStep = (M * matC0Size) * sizeof(CType);
    for (uint16_t i = 0; i < burstNum; ++i) {
        copy_ubuf_to_cbuf(
            reinterpret_cast<__cbuf__ void*>(cbDst + i * dstStep),
            reinterpret_cast<__ubuf__ void*>(ubSrc + i * srcStep), 0, 1, burstLen, 0, 0);
    }

    set_flag(PIPE_MTE3, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_FIX, EVENT_ID0);

    // copy mat cbuf to ub for tstore
    __ubuf__ CType* dstUbAddr = dstVecTile.data();
    __cbuf__ CType* srcMatAddr = dstMatTile.data();
    test_copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcMatAddr, 0, burstNum, burstLen, 0, 0);
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstVecTile);
}

template <int32_t testKey>
void launchTInsertAcc2Mat(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream)
{
    if constexpr (testKey == 1) {
        RunTInsertAcc2Mat<half, half, 16, 16, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (testKey == 2) {
        RunTInsertAcc2Mat<half, half, 32, 32, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (testKey == 3) {
        RunTInsertMat2Mat<half, half, 16, 16, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (testKey == 4) {
        RunTInsertMat2Mat<half, half, 32, 32, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (testKey == 5) {
        RunCbufToCbufTest<half, 16, 16, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0));
    } else if constexpr (testKey == 6) {
        RunCbufToCbufTest<half, 16, 16, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0));
    } else if constexpr (testKey == 7) {
        RunTInsertMat2MatLoad<half, 16, 16, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0));
    } else if constexpr (testKey == 8) {
        RunTInsertMat2MatLoad<half, 16, 16, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0));
    } else if constexpr (testKey == 9) {
        RunCbufToCbufSdkMode<half, 16, 16>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0));
    } else if constexpr (testKey == 10) {
        RunCbufToCbufSdkMode<half, 32, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0));
    } else if constexpr (testKey == 11) {
        RunTInsertMat2MatLoad<half, 32, 32, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0));
    } else if constexpr (testKey == 12) {
        RunAccDirectToUb<half, half, 16, 16, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else if constexpr (testKey == 13) {
        RunAccViaUb<half, half, 16, 16, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    }
}

template void launchTInsertAcc2Mat<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<2>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<3>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<4>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<5>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<6>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<7>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<8>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<9>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<10>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<11>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<12>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void launchTInsertAcc2Mat<13>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
