/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A2A3_T_EXTRA_OPS_HPP
#define PTO_NPU_A2A3_T_EXTRA_OPS_HPP

#include <cstdint>
#include <cstddef>
#include <type_traits>

#include <pto/common/pto_tile.hpp>
#include <pto/common/event.hpp>

namespace pto {
/**
 * Extra elementwise helpers.
 *
 * These ops are implemented as explicit TF loops (rather than vector intrinsics)
 * to cover cases not directly supported by A2/A3 vector instructions. They are
 * intended for correctness and portability across supported tile layouts; use
 * them sparingly in performance-critical paths.
 */

template <typename TileData>
PTO_INTERNAL constexpr uint32_t GetTileElementOffset(uint32_t r, uint32_t c)
{
    if constexpr (TileData::SFractal == SLayout::NoneBox) {
        if constexpr (TileData::isRowMajor) {
            return r * TileData::Cols + c;
        } else {
            return c * TileData::Rows + r;
        }
    } else {
        constexpr uint32_t subTilesPerCol = TileData::Rows / TileData::InnerRows;
        constexpr uint32_t subTilesPerRow = TileData::Cols / TileData::InnerCols;

        const uint32_t subTileR = r / TileData::InnerRows;
        const uint32_t subTileC = c / TileData::InnerCols;
        const uint32_t innerR = r % TileData::InnerRows;
        const uint32_t innerC = c % TileData::InnerCols;

        uint32_t subTileIndex = 0;
        if constexpr (TileData::isRowMajor) {
            subTileIndex = subTileR * subTilesPerRow + subTileC;
        } else {
            subTileIndex = subTileC * subTilesPerCol + subTileR;
        }

        uint32_t innerIndex = 0;
        if constexpr (TileData::SFractal == SLayout::RowMajor) {
            innerIndex = innerR * TileData::InnerCols + innerC;
        } else if constexpr (TileData::SFractal == SLayout::ColMajor) {
            innerIndex = innerC * TileData::InnerRows + innerR;
        } else {
            static_assert(pto::always_false_v<TileData>, "Invalid tile sub-fractal layout");
        }

        return subTileIndex * TileData::InnerNumel + innerIndex;
    }
}

template <typename TileData>
PTO_INTERNAL void CheckVecTile()
{
    static_assert(TileData::Loc == TileType::Vec, "Only Vec tiles are supported for this op on A2/A3");
}

template <typename T>
inline constexpr bool kExtraOpsHalfLike = std::is_same_v<T, half> || std::is_same_v<T, float16_t>;

PTO_INTERNAL void ExtraOpsWaitMte2ToScalar()
{
    PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
}

PTO_INTERNAL void ExtraOpsWaitScalarToMte3()
{
    PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
}

template <typename DType>
PTO_INTERNAL DType FpRemainder(DType x, DType y)
{
    const float yf = static_cast<float>(y);
    if (yf == 0.0f) {
        return static_cast<DType>(0);
    }
    const float xf = static_cast<float>(x);
    const float q = xf / yf;
    const int64_t qi = static_cast<int64_t>(q); // trunc toward zero
    const float rf = xf - static_cast<float>(qi) * yf;
    return static_cast<DType>(rf);
}

template <typename T>
PTO_INTERNAL T FromUnsigned(std::make_unsigned_t<T> v)
{
    if constexpr (!std::is_integral_v<T> || !std::is_signed_v<T>) {
        return static_cast<T>(v);
    } else {
        using U = std::make_unsigned_t<T>;
        constexpr uint32_t bits = sizeof(T) * 8U;
        constexpr U signBit = U(1) << (bits - 1U);
        if ((v & signBit) == 0) {
            return static_cast<T>(v);
        }
        const U magnitude = static_cast<U>(~v + 1U);
        if (magnitude == signBit) {
            // -2^(bits-1)
            return static_cast<T>(static_cast<int64_t>(-1) * (static_cast<int64_t>(1) << (bits - 1U)));
        }
        return static_cast<T>(-static_cast<int64_t>(magnitude));
    }
}

template <typename TileData>
__tf__ PTO_INTERNAL void TREM_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::TileDType __in__ src1, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            const T a = src0Ptr[off];
            const T b = src1Ptr[off];
            if constexpr (std::is_integral_v<T>) {
                dstPtr[off] = (b == static_cast<T>(0)) ? static_cast<T>(0) : static_cast<T>(a % b);
            } else {
                dstPtr[off] = FpRemainder<T>(a, b);
            }
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TREM_IMPL(TileData &dst, TileData &src0, TileData &src1)
{
    CheckVecTile<TileData>();
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TREM_TF<TileData>(dst.data(), src0.data(), src1.data(), validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TSHL_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::TileDType __in__ src1, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    using U = std::make_unsigned_t<T>;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

    constexpr uint32_t bits = sizeof(T) * 8U;
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            const uint32_t sh = static_cast<uint32_t>(static_cast<U>(src1Ptr[off])) & (bits - 1U);
            const U a = static_cast<U>(src0Ptr[off]);
            const U out = static_cast<U>(a << sh);
            dstPtr[off] = FromUnsigned<T>(out);
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TSHL_IMPL(TileData &dst, TileData &src0, TileData &src1)
{
    using T = typename TileData::DType;
    CheckVecTile<TileData>();
    static_assert(std::is_integral_v<T>, "TSHL: intended for integral element types");
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TSHL_TF<TileData>(dst.data(), src0.data(), src1.data(), validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TSHR_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::TileDType __in__ src1, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    using U = std::make_unsigned_t<T>;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

    constexpr uint32_t bits = sizeof(T) * 8U;
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            const U a = static_cast<U>(src0Ptr[off]);
            const uint32_t sh = static_cast<uint32_t>(static_cast<U>(src1Ptr[off])) & (bits - 1U);
            U out = static_cast<U>(a >> sh);
            if constexpr (std::is_signed_v<T>) {
                constexpr U signBit = U(1) << (bits - 1U);
                if ((a & signBit) != 0 && sh != 0) {
                    const U mask = static_cast<U>(~U(0)) << (bits - sh);
                    out = static_cast<U>(out | mask);
                }
            }
            dstPtr[off] = FromUnsigned<T>(out);
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TSHR_IMPL(TileData &dst, TileData &src0, TileData &src1)
{
    using T = typename TileData::DType;
    CheckVecTile<TileData>();
    static_assert(std::is_integral_v<T>, "TSHR: intended for integral element types");
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TSHR_TF<TileData>(dst.data(), src0.data(), src1.data(), validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TAND_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::TileDType __in__ src1, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            dstPtr[off] = static_cast<T>(src0Ptr[off] & src1Ptr[off]);
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TAND_IMPL(TileData &dst, TileData &src0, TileData &src1)
{
    using T = typename TileData::DType;
    CheckVecTile<TileData>();
    static_assert(std::is_integral_v<T>, "TAND: intended for integral element types");
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TAND_TF<TileData>(dst.data(), src0.data(), src1.data(), validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TOR_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::TileDType __in__ src1, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            dstPtr[off] = static_cast<T>(src0Ptr[off] | src1Ptr[off]);
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TOR_IMPL(TileData &dst, TileData &src0, TileData &src1)
{
    using T = typename TileData::DType;
    CheckVecTile<TileData>();
    static_assert(std::is_integral_v<T>, "TOR: intended for integral element types");
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TOR_TF<TileData>(dst.data(), src0.data(), src1.data(), validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TXOR_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::TileDType __in__ src1, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            dstPtr[off] = static_cast<T>(src0Ptr[off] ^ src1Ptr[off]);
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TXOR_IMPL(TileData &dst, TileData &src0, TileData &src1)
{
    using T = typename TileData::DType;
    CheckVecTile<TileData>();
    static_assert(std::is_integral_v<T>, "TXOR: intended for integral element types");
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TXOR_TF<TileData>(dst.data(), src0.data(), src1.data(), validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TNEG_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src,
    uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            if constexpr (kExtraOpsHalfLike<T>) {
                const float v = static_cast<float>(srcPtr[off]);
                dstPtr[off] = static_cast<T>(-v);
            } else {
                dstPtr[off] = static_cast<T>(-srcPtr[off]);
            }
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TNEG_IMPL(TileData &dst, TileData &src)
{
    CheckVecTile<TileData>();
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TNEG_TF<TileData>(dst.data(), src.data(), validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TNOT_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src,
    uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            dstPtr[off] = static_cast<T>(~srcPtr[off]);
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TNOT_IMPL(TileData &dst, TileData &src)
{
    using T = typename TileData::DType;
    CheckVecTile<TileData>();
    static_assert(std::is_integral_v<T>, "TNOT: intended for integral element types");
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TNOT_TF<TileData>(dst.data(), src.data(), validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TRELU_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src,
    uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            const T v = srcPtr[off];
            if constexpr (kExtraOpsHalfLike<T>) {
                const float vf = static_cast<float>(v);
                dstPtr[off] = static_cast<T>((vf > 0.0f) ? vf : 0.0f);
            } else {
                dstPtr[off] = (v > static_cast<T>(0)) ? v : static_cast<T>(0);
            }
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TRELU_IMPL(TileData &dst, TileData &src)
{
    CheckVecTile<TileData>();
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TRELU_TF<TileData>(dst.data(), src.data(), validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TPRELU_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::TileDType __in__ src1, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            const T x = src0Ptr[off];
            const T a = src1Ptr[off];
            if constexpr (kExtraOpsHalfLike<T>) {
                const float xf = static_cast<float>(x);
                const float af = static_cast<float>(a);
                dstPtr[off] = static_cast<T>((xf > 0.0f) ? xf : (xf * af));
            } else {
                dstPtr[off] = (x > static_cast<T>(0)) ? x : static_cast<T>(x * a);
            }
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TPRELU_IMPL(TileData &dst, TileData &src0, TileData &src1)
{
    CheckVecTile<TileData>();
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TPRELU_TF<TileData>(dst.data(), src0.data(), src1.data(), validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TADDC_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::TileDType __in__ src1, typename TileData::TileDType __in__ src2, uint32_t validRow,
    uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    __ubuf__ T *src2Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src2);
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            if constexpr (kExtraOpsHalfLike<T>) {
                const float a = static_cast<float>(src0Ptr[off]);
                const float b = static_cast<float>(src1Ptr[off]);
                const float d = static_cast<float>(src2Ptr[off]);
                dstPtr[off] = static_cast<T>(a + b + d);
            } else {
                dstPtr[off] = static_cast<T>(src0Ptr[off] + src1Ptr[off] + src2Ptr[off]);
            }
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TADDC_IMPL(TileData &dst, TileData &src0, TileData &src1, TileData &src2)
{
    CheckVecTile<TileData>();
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TADDC_TF<TileData>(dst.data(), src0.data(), src1.data(), src2.data(), validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TSUBC_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::TileDType __in__ src1, typename TileData::TileDType __in__ src2, uint32_t validRow,
    uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    __ubuf__ T *src2Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src2);
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            if constexpr (kExtraOpsHalfLike<T>) {
                const float a = static_cast<float>(src0Ptr[off]);
                const float b = static_cast<float>(src1Ptr[off]);
                const float d = static_cast<float>(src2Ptr[off]);
                dstPtr[off] = static_cast<T>(a - b + d);
            } else {
                dstPtr[off] = static_cast<T>(src0Ptr[off] - src1Ptr[off] + src2Ptr[off]);
            }
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TSUBC_IMPL(TileData &dst, TileData &src0, TileData &src1, TileData &src2)
{
    CheckVecTile<TileData>();
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TSUBC_TF<TileData>(dst.data(), src0.data(), src1.data(), src2.data(), validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TSUBS_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::DType scalar, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    const float sf = kExtraOpsHalfLike<T> ? static_cast<float>(scalar) : 0.0f;
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            if constexpr (kExtraOpsHalfLike<T>) {
                const float v = static_cast<float>(src0Ptr[off]);
                dstPtr[off] = static_cast<T>(v - sf);
            } else {
                dstPtr[off] = static_cast<T>(src0Ptr[off] - scalar);
            }
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TSUBS_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar)
{
    CheckVecTile<TileData>();
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TSUBS_TF<TileData>(dst.data(), src0.data(), scalar, validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TREMS_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::DType scalar, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            const T a = src0Ptr[off];
            if constexpr (std::is_integral_v<T>) {
                dstPtr[off] = (scalar == static_cast<T>(0)) ? static_cast<T>(0) : static_cast<T>(a % scalar);
            } else {
                dstPtr[off] = FpRemainder<T>(a, scalar);
            }
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TREMS_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar)
{
    CheckVecTile<TileData>();
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TREMS_TF<TileData>(dst.data(), src0.data(), scalar, validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TMAXS_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::DType scalar, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    const float sf = kExtraOpsHalfLike<T> ? static_cast<float>(scalar) : 0.0f;
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            const T v = src0Ptr[off];
            if constexpr (kExtraOpsHalfLike<T>) {
                const float vf = static_cast<float>(v);
                dstPtr[off] = static_cast<T>((vf > sf) ? vf : sf);
            } else {
                dstPtr[off] = (v > scalar) ? v : scalar;
            }
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TMAXS_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar)
{
    CheckVecTile<TileData>();
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TMAXS_TF<TileData>(dst.data(), src0.data(), scalar, validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TANDS_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::DType scalar, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            dstPtr[off] = static_cast<T>(src0Ptr[off] & scalar);
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TANDS_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar)
{
    using T = typename TileData::DType;
    CheckVecTile<TileData>();
    static_assert(std::is_integral_v<T>, "TANDS: intended for integral element types");
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TANDS_TF<TileData>(dst.data(), src0.data(), scalar, validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TORS_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::DType scalar, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            dstPtr[off] = static_cast<T>(src0Ptr[off] | scalar);
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TORS_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar)
{
    using T = typename TileData::DType;
    CheckVecTile<TileData>();
    static_assert(std::is_integral_v<T>, "TORS: intended for integral element types");
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TORS_TF<TileData>(dst.data(), src0.data(), scalar, validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TXORS_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::DType scalar, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            dstPtr[off] = static_cast<T>(src0Ptr[off] ^ scalar);
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TXORS_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar)
{
    using T = typename TileData::DType;
    CheckVecTile<TileData>();
    static_assert(std::is_integral_v<T>, "TXORS: intended for integral element types");
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TXORS_TF<TileData>(dst.data(), src0.data(), scalar, validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TLRELU_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::DType scalar, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    const float sf = kExtraOpsHalfLike<T> ? static_cast<float>(scalar) : 0.0f;
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            const T v = src0Ptr[off];
            if constexpr (kExtraOpsHalfLike<T>) {
                const float vf = static_cast<float>(v);
                dstPtr[off] = static_cast<T>((vf > 0.0f) ? vf : (vf * sf));
            } else {
                dstPtr[off] = (v > static_cast<T>(0)) ? v : static_cast<T>(v * scalar);
            }
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TLRELU_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar)
{
    CheckVecTile<TileData>();
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TLRELU_TF<TileData>(dst.data(), src0.data(), scalar, validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TADDSC_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::DType scalar, typename TileData::TileDType __in__ src1, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    const float sf = kExtraOpsHalfLike<T> ? static_cast<float>(scalar) : 0.0f;
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            if constexpr (kExtraOpsHalfLike<T>) {
                const float a = static_cast<float>(src0Ptr[off]);
                const float b = static_cast<float>(src1Ptr[off]);
                dstPtr[off] = static_cast<T>(a + sf + b);
            } else {
                dstPtr[off] = static_cast<T>(src0Ptr[off] + scalar + src1Ptr[off]);
            }
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TADDSC_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar, TileData &src1)
{
    CheckVecTile<TileData>();
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TADDSC_TF<TileData>(dst.data(), src0.data(), scalar, src1.data(), validRow, validCol);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TSUBSC_TF(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::DType scalar, typename TileData::TileDType __in__ src1, uint32_t validRow, uint32_t validCol)
{
    using T = typename TileData::DType;
    ExtraOpsWaitMte2ToScalar();
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    const float sf = kExtraOpsHalfLike<T> ? static_cast<float>(scalar) : 0.0f;
    for (uint32_t r = 0; r < validRow; ++r) {
        for (uint32_t c = 0; c < validCol; ++c) {
            const uint32_t off = GetTileElementOffset<TileData>(r, c);
            if constexpr (kExtraOpsHalfLike<T>) {
                const float a = static_cast<float>(src0Ptr[off]);
                const float b = static_cast<float>(src1Ptr[off]);
                dstPtr[off] = static_cast<T>(a - sf + b);
            } else {
                dstPtr[off] = static_cast<T>(src0Ptr[off] - scalar + src1Ptr[off]);
            }
        }
    }
    ExtraOpsWaitScalarToMte3();
}

template <typename TileData>
PTO_INTERNAL void TSUBSC_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar, TileData &src1)
{
    CheckVecTile<TileData>();
    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    TSUBSC_TF<TileData>(dst.data(), src0.data(), scalar, src1.data(), validRow, validCol);
}

} // namespace pto

#endif
