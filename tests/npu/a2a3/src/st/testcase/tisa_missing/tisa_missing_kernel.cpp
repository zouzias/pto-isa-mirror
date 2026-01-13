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
#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

using namespace pto;

namespace {
constexpr uint32_t kRows = 64;
constexpr uint32_t kCols = 64;
constexpr uint32_t kTileElems = kRows * kCols;
} // namespace

using GlobalShape = Shape<1, 1, 1, kRows, kCols>;
using GlobalStride = Stride<1, 1, 1, kCols, 1>;

using VecI32 = Tile<TileType::Vec, int32_t, kRows, kCols, BLayout::RowMajor, -1, -1>;
using VecF32 = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;

template <typename T, typename TileT>
__tf__ AICORE void ScalarLoadGmToVecTile(typename TileT::TileDType __out__ dstTile, __gm__ T *src)
{
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstTile);
    for (uint32_t i = 0; i < kTileElems; ++i) {
        dst[i] = src[i];
    }
}

template <typename T, typename TileT, typename... WaitEvents>
__tf__ AICORE void StoreVecTile(__gm__ T *out, uint32_t outIdx, TileT &tile, WaitEvents &...events)
{
    using GlobalT = GlobalTensor<T, GlobalShape, GlobalStride>;
    GlobalT outG(out + outIdx * kTileElems);
    SetFullVecMaskByDType<T>();
    Event<Op::TSTORE_VEC, Op::SCALAR> evStoreDone;
    evStoreDone = TSTORE(outG, tile, events...);
    TSYNC(evStoreDone);
    // Ensure the UB tile is not reused before the MTE3 store fully completes.
    pipe_barrier(PIPE_ALL);
}

template <typename TileT>
__tf__ AICORE void UbGatherScatter(typename TileT::TileDType __out__ gathered, typename TileT::TileDType __in__ memSrc,
    typename TileT::TileDType __in__ idx, typename TileT::TileDType __in__ scatterSrc,
    typename TileT::TileDType memDst, uint32_t numel)
{
    __ubuf__ int32_t *idxPtr = (__ubuf__ int32_t *)__cce_get_tile_ptr(idx);
    __ubuf__ int32_t *scatterPtr = (__ubuf__ int32_t *)__cce_get_tile_ptr(scatterSrc);
    __ubuf__ int32_t *memSrcPtr = (__ubuf__ int32_t *)__cce_get_tile_ptr(memSrc);
    __ubuf__ int32_t *memDstPtr = (__ubuf__ int32_t *)__cce_get_tile_ptr(memDst);
    __ubuf__ int32_t *gatherPtr = (__ubuf__ int32_t *)__cce_get_tile_ptr(gathered);

    for (uint32_t i = 0; i < numel; ++i) {
        const uint32_t j = static_cast<uint32_t>(idxPtr[i]);
        gatherPtr[i] = memSrcPtr[j];
        memDstPtr[j] = scatterPtr[i];
    }
}

__global__ AICORE void run_int_ops(__gm__ int32_t __out__ *out, __gm__ int32_t __in__ *src0, __gm__ int32_t __in__ *src1)
{
    using GlobalI32 = GlobalTensor<int32_t, GlobalShape, GlobalStride>;
    constexpr uint64_t tileSize = kRows * kCols * sizeof(int32_t);
    static_assert(tileSize * 3 <= 184 * 1024, "UB size overflow, should be less than 192KB.");

    SetFullVecMaskByDType<int32_t>();

    VecI32 a(kRows, kCols);
    VecI32 b(kRows, kCols);
    VecI32 dst(kRows, kCols);
    TASSIGN(a, 0x0);
    TASSIGN(b, tileSize);
    TASSIGN(dst, tileSize * 2);

    GlobalI32 aG(src0);
    GlobalI32 bG(src1);

    // These "missing ISA" ops are implemented via TF loops (scalar pipeline),
    // so the load->compute dependency must target `Op::SCALAR` (not `Op::VECTOR`).
    TLOAD(a, aG);
    Event<Op::TLOAD, Op::SCALAR> evLoadToScalar;
    evLoadToScalar = TLOAD(b, bG);

    uint32_t outIdx = 0;

    {
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TAND(dst, a, b, evLoadToScalar);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TOR(dst, a, b);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TXOR(dst, a, b);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TNOT(dst, a);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TSHL(dst, a, b);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TSHR(dst, a, b);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TREM(dst, a, b);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        constexpr int32_t kRemScalar = 7;
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TREMS(dst, a, kRemScalar);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        constexpr int32_t kLogicScalar = 0x0F0F0F0F;
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TANDS(dst, a, kLogicScalar);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        constexpr int32_t kLogicScalar = 0x0F0F0F0F;
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TORS(dst, a, kLogicScalar);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        constexpr int32_t kLogicScalar = 0x0F0F0F0F;
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TXORS(dst, a, kLogicScalar);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
}

__global__ AICORE void run_float_ops(
    __gm__ float __out__ *out, __gm__ float __in__ *src0, __gm__ float __in__ *src1, __gm__ float __in__ *src2)
{
    using GlobalF32 = GlobalTensor<float, GlobalShape, GlobalStride>;
    constexpr uint64_t tileSize = kRows * kCols * sizeof(float);
    static_assert(tileSize * 4 <= 184 * 1024, "UB size overflow, should be less than 192KB.");

    SetFullVecMaskByDType<float>();

    VecF32 x(kRows, kCols);
    VecF32 y(kRows, kCols);
    VecF32 z(kRows, kCols);
    VecF32 dst(kRows, kCols);
    TASSIGN(x, 0x0);
    TASSIGN(y, tileSize);
    TASSIGN(z, tileSize * 2);
    TASSIGN(dst, tileSize * 3);

    GlobalF32 xG(src0);
    GlobalF32 yG(src1);
    GlobalF32 zG(src2);

    // These "missing ISA" ops are implemented via TF loops (scalar pipeline),
    // so the load->compute dependency must target `Op::SCALAR` (not `Op::VECTOR`).
    TLOAD(x, xG);
    TLOAD(y, yG);
    Event<Op::TLOAD, Op::SCALAR> evLoadToScalar;
    evLoadToScalar = TLOAD(z, zG);

    uint32_t outIdx = 0;

    {
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TNEG(dst, x, evLoadToScalar);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TRELU(dst, x);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        constexpr float kLRelu = 0.1f;
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TLRELU(dst, x, kLRelu);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TPRELU(dst, x, y);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TADDC(dst, x, y, z);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TSUBC(dst, x, y, z);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        constexpr float kBias = 1.25f;
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TADDSC(dst, x, kBias, y);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        constexpr float kBias = 1.25f;
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TSUBSC(dst, x, kBias, y);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        constexpr float kBias = 1.25f;
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TSUBS(dst, x, kBias);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        constexpr float kBias = 1.25f;
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TMAXS(dst, x, kBias);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TREM(dst, x, y);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
    {
        constexpr float kRemScalar = 1.3f;
        Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
        evCompute = TREMS(dst, x, kRemScalar);
        StoreVecTile(out, outIdx++, dst, evCompute);
    }
}

__global__ AICORE void run_mgather_mscatter(__gm__ int32_t __out__ *out, __gm__ int32_t __in__ *memSrc,
    __gm__ int32_t __in__ *idx, __gm__ int32_t __in__ *scatterSrc, __gm__ int32_t __in__ *memDst)
{
    using GlobalI32 = GlobalTensor<int32_t, GlobalShape, GlobalStride>;
    constexpr uint64_t tileSize = kRows * kCols * sizeof(int32_t);
    static_assert(tileSize * 5 <= 184 * 1024, "UB size overflow, should be less than 192KB.");

    SetFullVecMaskByDType<int32_t>();

    VecI32 idxTile(kRows, kCols);
    VecI32 scatterTile(kRows, kCols);
    VecI32 memSrcTile(kRows, kCols);
    VecI32 memDstTile(kRows, kCols);
    VecI32 tmp(kRows, kCols);
    TASSIGN(idxTile, 0x0);
    TASSIGN(scatterTile, tileSize);
    TASSIGN(memSrcTile, tileSize * 2);
    TASSIGN(memDstTile, tileSize * 3);
    TASSIGN(tmp, tileSize * 4);

    GlobalI32 memSrcG(memSrc);
    GlobalI32 idxG(idx);
    GlobalI32 scatterSrcG(scatterSrc);
    GlobalI32 memDstG(memDst);

    // Avoid extremely slow random GM access in TF fallback implementations:
    // load all sources into UB first, then do gather/scatter in UB, and finally
    // store results to GM sequentially.
    TLOAD(idxTile, idxG);
    TLOAD(scatterTile, scatterSrcG);
    TLOAD(memSrcTile, memSrcG);
    Event<Op::TLOAD, Op::SCALAR> evLoadToScalar;
    evLoadToScalar = TLOAD(memDstTile, memDstG);
    TSYNC(evLoadToScalar);
    UbGatherScatter<VecI32>(tmp.data(), memSrcTile.data(), idxTile.data(), scatterTile.data(), memDstTile.data(), kTileElems);

    Event<Op::SCALAR, Op::TSTORE_VEC> evScalarToStore;
    evScalarToStore = RecordEvent{};
    StoreVecTile(out, 0, tmp, evScalarToStore);

    evScalarToStore = RecordEvent{};
    StoreVecTile(out, 1, memDstTile, evScalarToStore);
}

void LaunchTisaMissingIntOps(int32_t *out, int32_t *src0, int32_t *src1, void *stream)
{
    run_int_ops<<<1, nullptr, stream>>>(out, src0, src1);
}

void LaunchTisaMissingFloatOps(float *out, float *src0, float *src1, float *src2, void *stream)
{
    run_float_ops<<<1, nullptr, stream>>>(out, src0, src1, src2);
}

void LaunchTisaMissingMGatherMScatter(
    int32_t *out, int32_t *memSrc, int32_t *idx, int32_t *scatterSrc, int32_t *memDstInit, void *stream)
{
    run_mgather_mscatter<<<1, nullptr, stream>>>(out, memSrc, idx, scatterSrc, memDstInit);
}
