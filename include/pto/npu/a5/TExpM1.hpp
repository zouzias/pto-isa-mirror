/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEXPM1_HPP
#define TEXPM1_HPP

#include <cmath>
#include <pto/common/utils.hpp>
#include <pto/common/constants.hpp>
#include "common.hpp"
#include "utils.hpp"

namespace pto {

namespace texpm1_cfg {
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t NUM_WARPS = 32;
constexpr uint32_t TOTAL_THREADS = WARP_SIZE * NUM_WARPS;
} // namespace texpm1_cfg

constexpr float INV_LN2_APPROX = 1.4426950216293335f;
constexpr float LN2_HALF_APPROX = 0.4099999964237213f;
constexpr float LN2_APPROX = 0.693145751953215f;
constexpr float ONE_MINUS_LN2_APPROX = 0.000001428606765330187f;
constexpr float FLOAT_128 = 128.0f;
constexpr float FLOAT_NEG_ONE = -1.0f;
constexpr float C5 = 0.008382412604987621f;
constexpr float C4 = 0.0013879507314413786f;
constexpr float C3 = 0.04166783019900322f;
constexpr float C2 = 0.1666639745235443f;
constexpr float C1 = 0.4999999403953552f;
constexpr float FLOAT_INF = INFINITY;
constexpr float FLOAT_NEG_25 = -25.0f;
constexpr float FLOAT_2 = 2.0f;

AICORE PTO_INLINE float Expm1Taylor(float f1)
{
    float f0 = __expf(f1) - 1;
    float f2 = f1 * INV_LN2_APPROX; // f1 / ln2
    float f3 = __floorf(f2);
    float f4 = __fabsf(f1);
    bool p1 = f4 < LN2_HALF_APPROX;
    float f5 = p1 ? 0.0f : f3; // f5 = k
    float f6 = -f5;
    constexpr float f7 = LN2_APPROX;
    float f8 = __fma(f6, f7, f1);
    constexpr float f9 = ONE_MINUS_LN2_APPROX; // ln2_lo
    float f10 = __fma(f6, f9, f8);             // r = x - k * ln2_hi - k * ln2_lo
    bool p2 = f5 == FLOAT_128;
    float f11 = f5 + FLOAT_NEG_ONE;
    float f12 = p2 ? f11 : f5; // f5 - 1
    // f21 = p(r) = 1/2 + r * (1/6 + r * (1/24 + r * (1/120 + r * 1/720)))
    constexpr float f13 = C5;
    constexpr float f14 = C4;
    float f15 = __fma(f14, f10, f13);
    constexpr float f16 = C3;
    float f17 = __fma(f15, f10, f16);
    constexpr float f18 = C2;
    float f19 = __fma(f17, f10, f18);
    constexpr float f20 = C1;
    float f21 = __fma(f19, f10, f20); // p(r)

    float f22 = f10 * f21;            // r * p(r)
    float f23 = __fma(f22, f10, f10); // expm1(r) = r + r^2 * p(r)
    float f24 = __powf(2, f12);
    float f25 = f24 + FLOAT_NEG_ONE;
    float f26 = __fma(f23, f24, f25);
    float f27 = f26 + f26;
    float f28 = p2 ? f27 : f26;
    bool p3 = f12 > FLOAT_128;
    float f29 = p3 ? FLOAT_INF : f28;
    bool p4 = f12 < FLOAT_NEG_25;
    float f30 = p4 ? FLOAT_NEG_ONE : f28;
    float f31 = f1 + f1;
    bool p5 = f1 == 0.0f;
    float f32 = p5 ? f31 : f30;

    return __fabsf(f1) > FLOAT_2 ? f0 : f32;
}

template <typename T, typename U, uint32_t DstRowStride, uint32_t SrcRowStride>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_texpm1_elem_kernel(__ubuf__ T *__restrict__ dst, __ubuf__ U *__restrict__ src, unsigned validRow,
                                 unsigned validCol)
{
    const uint32_t tx = __cce_simt_get_TID_X();
    const uint32_t ty = __cce_simt_get_TID_Y();

#pragma unroll(1)
    for (uint32_t row = ty; row < validRow; row += texpm1_cfg::NUM_WARPS) {
        __ubuf__ U *srcRow = src + row * SrcRowStride;
        __ubuf__ T *dstRow = dst + row * DstRowStride;

#pragma unroll(4)
        for (uint32_t col = tx; col < validCol; col += texpm1_cfg::WARP_SIZE) {
            float f1 = srcRow[col];
            dstRow[col] = Expm1Taylor(f1);
        }
    }
}

template <typename DstTile, typename SrcTile>
__tf__ AICORE void TExpM1(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __out__ src,
                          unsigned validRow, unsigned validCol)
{
    using T = typename DstTile::DType;
    using U = typename SrcTile::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ U *srcPtr = (__ubuf__ U *)__cce_get_tile_ptr(src);

    cce::async_invoke<simt_texpm1_elem_kernel<T, U, DstTile::RowStride, SrcTile::RowStride>>(
        cce::dim3{texpm1_cfg::WARP_SIZE, texpm1_cfg::NUM_WARPS}, dstPtr, srcPtr, validRow, validCol);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TEXPM1_IMPL(DstTile &dst, SrcTile &src)
{
    using T = typename DstTile::DType;
    using U = typename SrcTile::DType;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>,
                  "Fix: TEXPM1 dst has invalid data type.");
    static_assert(std::is_same_v<U, int32_t> || std::is_same_v<U, int16_t> || std::is_same_v<U, float> ||
                      std::is_same_v<U, half> || std::is_same_v<U, bfloat16_t>,
                  "Fix: TEXPM1 src has invalid data type.");
    static_assert(DstTile::isRowMajor && SrcTile::isRowMajor, "Fix: TEXPM1 only support row major layout.");
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TEXPM1: Number of columns of src and dst must be the same.");
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TEXPM1: Number of rows of src and dst must be the same.");
    TExpM1<DstTile, SrcTile>(dst.data(), src.data(), dstValidRow, dstValidCol);
}
} // namespace pto

#endif // TEXPM1_HPP
