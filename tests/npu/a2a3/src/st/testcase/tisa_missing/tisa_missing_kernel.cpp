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

template <typename T, typename TileT>
template <typename... WaitEvents>
__tf__ AICORE void StoreVecTile(__gm__ T *out, uint32_t outIdx, TileT &tile, WaitEvents &...events)
{
    using GlobalT = GlobalTensor<T, GlobalShape, GlobalStride>;
    GlobalT outG(out + outIdx * kTileElems);
    SetFullVecMaskByDType<T>();
    Event<Op::TSTORE_VEC, Op::SCALAR> evStoreDone;
    evStoreDone = TSTORE(outG, tile, events...);
    TSYNC(evStoreDone);
}

__global__ AICORE void run_int_ops(__gm__ int32_t __out__ *out, __gm__ int32_t __in__ *src0, __gm__ int32_t __in__ *src1)
{
    using GlobalI32 = GlobalTensor<int32_t, GlobalShape, GlobalStride>;
    constexpr uint64_t tileSize = kRows * kCols * sizeof(int32_t);
    static_assert(tileSize * 3 <= 184 * 1024, "UB size overflow, should be less than 192KB.");

    VecI32 a(kRows, kCols);
    VecI32 b(kRows, kCols);
    VecI32 dst(kRows, kCols);
    TASSIGN(a, 0x0);
    TASSIGN(b, tileSize);
    TASSIGN(dst, tileSize * 2);

    GlobalI32 aG(src0);
    GlobalI32 bG(src1);

    TLOAD(a, aG);
    Event<Op::TLOAD, Op::VECTOR> evLoadToVec;
    evLoadToVec = TLOAD(b, bG);

    uint32_t outIdx = 0;

    Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
    evCompute = TAND(dst, a, b, evLoadToVec);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TOR(dst, a, b);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TXOR(dst, a, b);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TNOT(dst, a);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TSHL(dst, a, b);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TSHR(dst, a, b);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TREM(dst, a, b);
    StoreVecTile(out, outIdx++, dst, evCompute);

    constexpr int32_t kRemScalar = 7;
    evCompute = TREMS(dst, a, kRemScalar);
    StoreVecTile(out, outIdx++, dst, evCompute);

    constexpr int32_t kLogicScalar = 0x0F0F0F0F;
    evCompute = TANDS(dst, a, kLogicScalar);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TORS(dst, a, kLogicScalar);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TXORS(dst, a, kLogicScalar);
    StoreVecTile(out, outIdx++, dst, evCompute);
}

__global__ AICORE void run_float_ops(
    __gm__ float __out__ *out, __gm__ float __in__ *src0, __gm__ float __in__ *src1, __gm__ float __in__ *src2)
{
    using GlobalF32 = GlobalTensor<float, GlobalShape, GlobalStride>;
    constexpr uint64_t tileSize = kRows * kCols * sizeof(float);
    static_assert(tileSize * 4 <= 184 * 1024, "UB size overflow, should be less than 192KB.");

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

    TLOAD(x, xG);
    TLOAD(y, yG);
    Event<Op::TLOAD, Op::VECTOR> evLoadToVec;
    evLoadToVec = TLOAD(z, zG);

    uint32_t outIdx = 0;

    Event<Op::VECTOR, Op::TSTORE_VEC> evCompute;
    evCompute = TNEG(dst, x, evLoadToVec);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TRELU(dst, x);
    StoreVecTile(out, outIdx++, dst, evCompute);

    constexpr float kLRelu = 0.1f;
    evCompute = TLRELU(dst, x, kLRelu);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TPRELU(dst, x, y);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TADDC(dst, x, y, z);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TSUBC(dst, x, y, z);
    StoreVecTile(out, outIdx++, dst, evCompute);

    constexpr float kBias = 1.25f;
    evCompute = TADDSC(dst, x, kBias, y);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TSUBSC(dst, x, kBias, y);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TSUBS(dst, x, kBias);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TMAXS(dst, x, kBias);
    StoreVecTile(out, outIdx++, dst, evCompute);

    evCompute = TREM(dst, x, y);
    StoreVecTile(out, outIdx++, dst, evCompute);

    constexpr float kRemScalar = 1.3f;
    evCompute = TREMS(dst, x, kRemScalar);
    StoreVecTile(out, outIdx++, dst, evCompute);
}

__global__ AICORE void run_mgather_mscatter(__gm__ int32_t __out__ *out, __gm__ int32_t __in__ *memSrc,
    __gm__ int32_t __in__ *idx, __gm__ int32_t __in__ *scatterSrc, __gm__ int32_t __in__ *memDst)
{
    using GlobalI32 = GlobalTensor<int32_t, GlobalShape, GlobalStride>;
    constexpr uint64_t tileSize = kRows * kCols * sizeof(int32_t);
    static_assert(tileSize * 3 <= 184 * 1024, "UB size overflow, should be less than 192KB.");

    VecI32 idxTile(kRows, kCols);
    VecI32 srcTile(kRows, kCols);
    VecI32 tmp(kRows, kCols);
    TASSIGN(idxTile, 0x0);
    TASSIGN(srcTile, tileSize);
    TASSIGN(tmp, tileSize * 2);

    GlobalI32 memSrcG(memSrc);
    GlobalI32 idxG(idx);
    GlobalI32 scatterSrcG(scatterSrc);
    GlobalI32 memDstG(memDst);

    TLOAD(idxTile, idxG);
    Event<Op::TLOAD, Op::SCALAR> evLoadToScalar;
    evLoadToScalar = TLOAD(srcTile, scatterSrcG);

    Event<Op::SCALAR, Op::TSTORE_VEC> evGatherToStore;
    evGatherToStore = MGATHER(tmp, memSrcG, idxTile, evLoadToScalar);
    StoreVecTile(out, 0, tmp, evGatherToStore);

    MSCATTER(memDstG, srcTile, idxTile, evLoadToScalar);
    ScalarLoadGmToVecTile<int32_t, VecI32>(tmp.data(), memDstG.data());

    Event<Op::SCALAR, Op::TSTORE_VEC> evScalarToStore;
    evScalarToStore = RecordEvent{};
    StoreVecTile(out, 1, tmp, evScalarToStore);
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
