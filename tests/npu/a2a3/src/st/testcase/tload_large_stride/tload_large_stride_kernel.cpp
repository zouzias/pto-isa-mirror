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
using namespace pto;

template <typename TileData>
__tf__ PTO_INTERNAL void StoreRaw(__gm__ uint8_t* __out__ dst, typename TileData::TileDType __in__ src)
{
    constexpr int blocks = TileData::Rows * TileData::Cols * sizeof(typename TileData::DType) / 32;
    if constexpr (TileData::Loc == TileType::Vec) {
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(PTO_NPU_ARCH_KIRINDEV0000)
        pto_copy_ubuf_to_gm_align_v2(dst, (__ubuf__ uint8_t*)__cce_get_tile_ptr(src), 0, 1, blocks * 32, 0, 0, 0);
#else
        copy_ubuf_to_gm(dst, (__ubuf__ uint8_t*)__cce_get_tile_ptr(src), 0, 1, blocks, 0, 0);
#endif
#if defined(PTO_NPU_ARCH_A2A3)
    } else {
        copy_cbuf_to_gm(dst, (__cbuf__ uint8_t*)__cce_get_tile_ptr(src), 0, 1, blocks, 0, 0);
#endif
    }
}

#if !defined(PTO_NPU_ARCH_A2A3) && !defined(PTO_TLOAD_VEC_ONLY)
template <typename MatTile, typename VecTile>
__tf__ PTO_INTERNAL void CopyMatToVec(
    typename VecTile::TileDType __out__ dst, typename MatTile::TileDType __in__ src, int core)
{
    copy_cbuf_to_ubuf(
        (__ubuf__ void*)__cce_get_tile_ptr(dst), (__cbuf__ void*)__cce_get_tile_ptr(src), core, 1,
        MatTile::Rows * MatTile::Cols * sizeof(typename MatTile::DType) / 32, 0, 0);
}
#endif

#if defined(PTO_NPU_ARCH_A5)
AICORE inline void LoadWideConv(__gm__ uint8_t* out, __gm__ uint8_t* src, int64_t burstStride, int64_t batchStride)
{
    using Global = GlobalTensor<uint8_t, Shape<2, 2, 1, 1, 32>, pto::Stride<-1, -1, 32, 32, 1>, Layout::NC1HWC0>;
    Global input(src, {}, pto::Stride<-1, -1, 32, 32, 1>(batchStride, burstStride, 32, 32, 1));
    using MatT = ConvTile<TileType::Mat, uint8_t, 128, Layout::NC1HWC0, ConvTileShape<2, 2, 1, 1, 32>>;
    using RawMatT = Tile<TileType::Mat, uint8_t, 4, 32, BLayout::RowMajor>;
    using VecT = Tile<TileType::Vec, uint8_t, 4, 32, BLayout::RowMajor>;
    MatT mat;
    RawMatT raw;
    VecT vec;
    TASSIGN(mat, 0);
    TASSIGN(raw, 0);
    TASSIGN(vec, 0);
#if defined(__DAV_CUBE__)
    TLOAD(mat, input);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    CopyMatToVec<RawMatT, VecT>(vec.data(), raw.data(), 0);
    CopyMatToVec<RawMatT, VecT>(vec.data(), raw.data(), 1);
    set_intra_block(PIPE_MTE1, 0);
    set_intra_block(PIPE_MTE1, 16);
#else
    wait_intra_block(PIPE_MTE3, 0);
    if (get_subblockid() == 0) {
        StoreRaw<VecT>(out, vec.data());
    }
#endif
}
#endif

template <bool Mat, int Format, uint64_t Gap, bool Dynamic, bool Outer, int Axis = 0, int AddressBits = 32>
__global__ AICORE void LargeStrideKernel(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int64_t burstStride, int64_t batchStride)
{
    if constexpr (Format == 4) {
#if defined(PTO_NPU_ARCH_A5)
        LoadWideConv(out, src, burstStride, batchStride);
#endif
    } else {
        constexpr int bytes = Format == 2 ? 512 : 32;
        constexpr int64_t burst = Gap + bytes;
        constexpr int64_t batch = Outer ? (1LL << AddressBits) + 4096 : 2 * bytes;
        constexpr int64_t bs = Dynamic ? -1 : burst;
        constexpr int64_t os = Dynamic ? -1 : batch;
        constexpr bool nd = Format == 0 || Format == 3;
        constexpr int validCols = Format == 3 ? 31 : 32;
        constexpr auto loc = Mat ? TileType::Mat : TileType::Vec;
        using ShapeT = std::conditional_t<
            nd, Shape<Axis == 0 ? 2 : 1, Axis == 1 ? 2 : 1, Axis == 2 ? 2 : 1, 2, validCols>,
            std::conditional_t<
                Format == 1, Shape<Axis == 0 ? 2 : 1, Axis == 1 ? 2 : 1, Axis == 2 ? 2 : 1, 32, 2>,
                Shape<2, 2, 1, 16, 32>>>;
        using StrideT = std::conditional_t<
            nd, pto::Stride<Axis == 0 ? os : 1, Axis == 1 ? os : 1, Axis == 2 ? os : 1, bs, 1>,
            std::conditional_t<
                Format == 1, pto::Stride<Axis == 0 ? os : 1, Axis == 1 ? os : 1, Axis == 2 ? os : 1, 1, bs>,
                pto::Stride<os, bs, 512, 32, 1>>>;
        constexpr auto layout = nd ? Layout::ND : (Format == 1 ? Layout::DN : Layout::NZ);
        using Global = GlobalTensor<uint8_t, ShapeT, StrideT, layout>;
        StrideT strides(
            Axis == 0 ? batchStride : 1, Format == 2 ? burstStride : (Axis == 1 ? batchStride : 1),
            Format == 2 ? 512 : (Axis == 2 ? batchStride : 1), nd ? burstStride : (Format == 2 ? 32 : 1),
            Format == 1 ? burstStride : 1);
        Global input(src, {}, strides);
        using TileT = std::conditional_t<
            nd, Tile<loc, uint8_t, 4, 64, BLayout::RowMajor, 4, validCols, SLayout::NoneBox, 512, PadValue::Zero>,
            std::conditional_t<
                Format == 1, Tile<loc, uint8_t, 64, 4, BLayout::ColMajor, 32, 4>,
                Tile<loc, uint8_t, 32, 128, BLayout::ColMajor, 16, 128, SLayout::RowMajor, 512>>>;
        TileT tile;
        TASSIGN(tile, 0);
        if constexpr (Mat) {
#ifndef PTO_TLOAD_VEC_ONLY
#if defined(PTO_NPU_ARCH_A2A3)
#if defined(__DAV_CUBE__)
            TLOAD(tile, input);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            StoreRaw<TileT>(out, tile.data());
#endif
#else
            using VecT = Tile<TileType::Vec, uint8_t, 1, TileT::Rows * TileT::Cols, BLayout::RowMajor>;
            VecT vec;
            TASSIGN(vec, 0);
#if !defined(PTO_NPU_ARCH_A5) || defined(__DAV_CUBE__)
            TLOAD(tile, input);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            CopyMatToVec<TileT, VecT>(vec.data(), tile.data(), 0);
#if defined(PTO_NPU_ARCH_A5)
            CopyMatToVec<TileT, VecT>(vec.data(), tile.data(), 1);
            set_intra_block(PIPE_MTE1, 0);
            set_intra_block(PIPE_MTE1, 16);
#else
            set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
            StoreRaw<VecT>(out, vec.data());
#endif
#endif
#if defined(PTO_NPU_ARCH_A5) && defined(__DAV_VEC__)
            wait_intra_block(PIPE_MTE3, 0);
            if (get_subblockid() == 0) {
                StoreRaw<VecT>(out, vec.data());
            }
#endif
#endif
#endif
        } else {
#if (!defined(PTO_NPU_ARCH_A2A3) && !defined(PTO_NPU_ARCH_A5)) || defined(__DAV_VEC__)
            TLOAD(tile, input);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5)
            if (get_subblockid() == 0)
#endif
            {
                StoreRaw<TileT>(out, tile.data());
            }
#endif
        }
    }
}

template <bool Mat, int Format, uint64_t Gap, bool Dynamic, bool Outer, int Axis = 0, int AddressBits = 32>
void LaunchLargeStride(uint8_t* out, uint8_t* src, void* stream)
{
    constexpr int bytes = Format == 2 ? 512 : 32;
    LargeStrideKernel<Mat, Format, Gap, Dynamic, Outer, Axis, AddressBits>
        <<<1, nullptr, stream>>>(out, src, Gap + bytes, Outer ? (1LL << AddressBits) + 4096 : 2 * bytes);
}

#ifndef PTO_TLOAD_VEC_ONLY
template void LaunchLargeStride<true, 0, 65535ULL * 32, false, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 0, 65536ULL * 32, true, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 1, 65535ULL * 32, false, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 1, 65536ULL * 32, true, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 2, 65535ULL * 32, false, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 2, 65536ULL * 32, true, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 0, 1ULL << 32, false, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 1, 1ULL << 32, true, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 2, 1ULL << 32, false, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 0, 64, true, true>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 1, 64, false, true>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 2, 1024, true, true>(uint8_t*, uint8_t*, void*);
#endif
template void LaunchLargeStride<false, 0, (1ULL << 32) - 32, false, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 0, 1ULL << 32, true, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 1, (1ULL << 32) - 32, false, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 1, 1ULL << 32, true, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 2, (1ULL << 32) - 32, false, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 2, 1ULL << 32, true, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 0, 64, true, true>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 1, 64, false, true>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 2, 1024, true, true>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 3, 64, false, false>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 3, 1ULL << 32, true, false>(uint8_t*, uint8_t*, void*);

#ifndef PTO_TLOAD_VEC_ONLY
template <typename T, bool DN, int64_t RowStride, bool Dynamic>
__global__ AICORE void ConversionStrideKernel(__gm__ uint8_t* out, __gm__ T* src, int64_t stride)
{
    using ShapeT = std::conditional_t<DN, Shape<1, 1, 1, 16, 2>, Shape<1, 1, 1, 2, 16>>;
    constexpr int64_t rowStride = Dynamic ? -1 : RowStride;
    using StrideT = std::conditional_t<DN, pto::Stride<1, 1, 1, 1, rowStride>, pto::Stride<1, 1, 1, rowStride, 1>>;
    using Global = GlobalTensor<T, ShapeT, StrideT, DN ? Layout::DN : Layout::ND>;
    Global input(src, {}, StrideT(1, 1, 1, DN ? 1 : stride, DN ? stride : 1));
    using MatT = Tile<
        TileType::Mat, T, 16, 16, DN ? BLayout::RowMajor : BLayout::ColMajor, DN ? 16 : 2, DN ? 2 : 16,
        DN ? SLayout::ColMajor : SLayout::RowMajor, 512>;
    MatT mat;
    TASSIGN(mat, 0);
#if defined(PTO_NPU_ARCH_A2A3)
#if defined(__DAV_CUBE__)
    TLOAD(mat, input);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    StoreRaw<MatT>(out, mat.data());
#endif
#else
    using VecT = Tile<TileType::Vec, uint8_t, 1, 16 * 16 * sizeof(T), BLayout::RowMajor>;
    VecT vec;
    TASSIGN(vec, 0);
#if !defined(PTO_NPU_ARCH_A5) || defined(__DAV_CUBE__)
    TLOAD(mat, input);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    CopyMatToVec<MatT, VecT>(vec.data(), mat.data(), 0);
#if defined(PTO_NPU_ARCH_A5)
    CopyMatToVec<MatT, VecT>(vec.data(), mat.data(), 1);
    set_intra_block(PIPE_MTE1, 0);
    set_intra_block(PIPE_MTE1, 16);
#else
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    StoreRaw<VecT>(out, vec.data());
#endif
#endif
#if defined(PTO_NPU_ARCH_A5) && defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, 0);
    if (get_subblockid() == 0) {
        StoreRaw<VecT>(out, vec.data());
    }
#endif
#endif
}

template <typename T, bool DN, int64_t RowStride, bool Dynamic>
void LaunchConversionStride(uint8_t* out, uint8_t* src, void* stream)
{
    ConversionStrideKernel<T, DN, RowStride, Dynamic>
        <<<1, nullptr, stream>>>(out, reinterpret_cast<T*>(src), RowStride);
}
template void LaunchConversionStride<float, false, 32767, false>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<float, false, 32768, false>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<float, false, 32768, true>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<float, false, 65535, true>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<float, true, 32767, false>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<float, true, 32768, false>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<float, true, 32768, true>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<float, true, 65535, true>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<int32_t, false, 32767, false>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<int32_t, false, 32768, false>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<int32_t, false, 32768, true>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<int32_t, false, 65535, true>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<int32_t, true, 32767, false>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<int32_t, true, 32768, false>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<int32_t, true, 32768, true>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<int32_t, true, 65535, true>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<uint32_t, false, 32767, false>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<uint32_t, false, 32768, false>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<uint32_t, false, 32768, true>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<uint32_t, false, 65535, true>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<uint32_t, true, 32767, false>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<uint32_t, true, 32768, false>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<uint32_t, true, 32768, true>(uint8_t*, uint8_t*, void*);
template void LaunchConversionStride<uint32_t, true, 65535, true>(uint8_t*, uint8_t*, void*);
#endif

#ifdef PTO_TLOAD_WIDE_STRIDE
#ifndef PTO_TLOAD_VEC_ONLY
template void LaunchLargeStride<true, 0, ((1ULL << 40) - 32) - 32, false, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 0, ((1ULL << 40) + 0) - 32, true, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 0, ((1ULL << 40) + 64) - 32, false, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 1, ((1ULL << 40) - 32) - 32, false, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 1, ((1ULL << 40) + 0) - 32, true, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 1, ((1ULL << 40) + 64) - 32, false, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 2, ((1ULL << 40) - 32) - 512, false, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 2, ((1ULL << 40) + 0) - 512, true, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 2, ((1ULL << 40) + 64) - 512, false, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 0, 64, true, true, 1, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 0, 64, true, true, 2, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 1, 64, true, true, 1, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<true, 1, 64, true, true, 2, 40>(uint8_t*, uint8_t*, void*);
#endif
template void LaunchLargeStride<false, 0, ((1ULL << 40) - 32) - 32, false, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 0, ((1ULL << 40) + 0) - 32, true, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 0, ((1ULL << 40) + 64) - 32, false, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 1, ((1ULL << 40) - 32) - 32, false, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 1, ((1ULL << 40) + 0) - 32, true, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 1, ((1ULL << 40) + 64) - 32, false, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 2, ((1ULL << 40) - 32) - 512, false, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 2, ((1ULL << 40) + 0) - 512, true, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 2, ((1ULL << 40) + 64) - 512, false, false, 0, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 0, 64, true, true, 1, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 0, 64, true, true, 2, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 1, 64, true, true, 1, 40>(uint8_t*, uint8_t*, void*);
template void LaunchLargeStride<false, 1, 64, true, true, 2, 40>(uint8_t*, uint8_t*, void*);
#endif
#ifdef PTO_TLOAD_CONV_WIDE
template void LaunchLargeStride<true, 4, 64, true, true, 0, 40>(uint8_t*, uint8_t*, void*);
#endif

#ifdef PTO_TLOAD_VECTOR_LENGTH
template <typename MatT>
__tf__ PTO_INTERNAL void InitializeVectorMat(typename MatT::TileDType __out__ dst, __gm__ uint8_t* src)
{
    // Include an adjacent block to detect writes beyond the padded vector.
    constexpr int blocks = MatT::Rows * MatT::Cols * sizeof(typename MatT::DType) / 32 + 1;
    pto_copy_gm_to_cbuf((__cbuf__ uint8_t*)__cce_get_tile_ptr(dst), src, 0, 1, blocks, 0, 0);
    pipe_barrier(PIPE_MTE2);
}

template <typename MatT>
__tf__ PTO_INTERNAL void StoreVectorMat(__gm__ uint8_t* __out__ dst, typename MatT::TileDType __in__ src)
{
    constexpr int bytes = MatT::Rows * MatT::Cols * sizeof(typename MatT::DType) + 32;
    auto srcAddr = (__cbuf__ uint8_t*)__cce_get_tile_ptr(src);
#if defined(PTO_NPU_ARCH_A2A3)
    copy_cbuf_to_gm(dst, srcAddr, 0, 1, bytes / 32, 0, 0);
#else
    auto buffer = (__ubuf__ uint8_t*)0;
    for (int offset = 0; offset < bytes; offset += 1024) {
        int count = bytes - offset < 1024 ? bytes - offset : 1024;
        copy_cbuf_to_ubuf(buffer, srcAddr + offset, 0, 1, count / 32, 0, 0);
        set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm(dst + offset, buffer, 0, 1, count / 32, 0, 0);
        set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    }
#endif
}

template <typename T, bool DN, int Length, bool Dynamic>
__global__ AICORE void VectorLengthKernel(__gm__ uint8_t* out, __gm__ T* src, int length)
{
#if !defined(PTO_NPU_ARCH_A2A3) || defined(__DAV_CUBE__)
    constexpr int capacity = (Length * sizeof(T) + 31) / 32 * 32 / sizeof(T);
    constexpr int validLength = Dynamic ? -1 : Length;
    using MatT = Tile<
        TileType::Mat, T, DN ? capacity : 1, DN ? 1 : capacity, DN ? BLayout::ColMajor : BLayout::RowMajor,
        DN ? validLength : 1, DN ? 1 : validLength, SLayout::NoneBox>;
    using ShapeT = Shape<1, 1, 1, DN ? validLength : 1, DN ? 1 : validLength>;
    using StrideT = pto::Stride<1, 1, 1, DN ? 1 : -1, DN ? -1 : 1>;
    using Global = GlobalTensor<T, ShapeT, StrideT, DN ? Layout::DN : Layout::ND>;
    MatT mat;
    if constexpr (Dynamic) {
        if constexpr (DN) {
            mat.SetValidRow(length);
        } else {
            mat.SetValidCol(length);
        }
    }
    TASSIGN(mat, 0);
    Global input(
        src, ShapeT(1, 1, 1, DN ? length : 1, DN ? 1 : length), StrideT(1, 1, 1, DN ? 1 : length, DN ? length : 1));
    InitializeVectorMat<MatT>(mat.data(), out);
    TLOAD(mat, input);
#if defined(PTO_NPU_ARCH_A2A3)
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    StoreVectorMat<MatT>(out, mat.data());
#else
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    StoreVectorMat<MatT>(out, mat.data());
#endif
#endif
}

template <typename T, bool DN, int Length, bool Dynamic>
void LaunchVectorLength(uint8_t* out, uint8_t* src, void* stream)
{
    VectorLengthKernel<T, DN, Length, Dynamic><<<1, nullptr, stream>>>(out, reinterpret_cast<T*>(src), Length);
}
template void LaunchVectorLength<uint8_t, false, 65535, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint8_t, false, 65536, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint8_t, false, 65536, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint8_t, false, 65537, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint16_t, false, 65535, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint16_t, false, 65536, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint16_t, false, 65536, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint16_t, false, 65537, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<float, false, 65535, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<float, false, 65536, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<float, false, 65536, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<float, false, 65537, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, false, 16383, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, false, 16384, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, false, 16384, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, false, 16385, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, false, 32767, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, false, 32768, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, false, 32768, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, false, 32769, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint64_t, false, 32768, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint8_t, true, 31, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint8_t, true, 32, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint8_t, true, 33, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint16_t, true, 15, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint16_t, true, 16, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint16_t, true, 17, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<float, true, 7, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<float, true, 8, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<float, true, 9, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, true, 3, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, true, 4, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, true, 5, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, true, 16383, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, true, 16384, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, true, 16384, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, false, 64, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, false, 3, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<float, false, 64, false>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<float, false, 7, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint8_t, false, 65, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint8_t, true, 65, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint16_t, false, 65, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<uint16_t, true, 65, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<float, false, 65, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<float, true, 65, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, false, 65, true>(uint8_t*, uint8_t*, void*);
template void LaunchVectorLength<int64_t, true, 65, true>(uint8_t*, uint8_t*, void*);
#endif
