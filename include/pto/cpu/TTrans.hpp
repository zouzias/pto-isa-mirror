/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TTRANS_HPP
#define TTRANS_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include <type_traits>
#include <cstdint>

namespace pto {

// Target layout formats defined in your system architecture
template <bool ignored = true>
enum class Layout {
    NCHW,
    NC1HWC0,
    FRACTAL_Z
};

template <typename DstTileData, typename SrcTileData>
inline void TTRANS_NCHW2NC1HWC0(DstTileData &dst, SrcTileData &src)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;

    const auto *src_ptr = reinterpret_cast<const SrcDType *>(src.data());
    auto *dst_ptr = reinterpret_cast<DstDType *>(dst.data());

    constexpr int64_t C0 = 32 / sizeof(SrcDType);

    int64_t N = src.GetShape(0);
    int64_t C = src.GetShape(1);
    int64_t H = src.GetShape(2);
    int64_t W = src.GetShape(3);
    int64_t C1 = (C + C0 - 1) / C0;
    size_t Size = N * C1 * H * W * C0;

    std::fill(dst.data(), dst.data() + Size, 0);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            size_t r = c / C0;
            size_t cl = c % C0;
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    size_t srcIndex = w + W * h + W * H * c + W * H * C * n;
                    size_t dstIndex = W * H * C1 * C0 * n + C0 * H * W * r + C0 * W * h + C0 * w + cl;
                    dst_ptr[dstIndex] = src_ptr[srcIndex];
                }
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData>
inline void TTRANS_GNCHW2NC1HWC0(DstTileData &dst, SrcTileData &src)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;

    const auto *src_ptr = reinterpret_cast<const SrcDType *>(src.data());
    auto *dst_ptr = reinterpret_cast<DstDType *>(dst.data());

    constexpr int64_t C0 = 32 / sizeof(SrcDType);

    int64_t G = src.GetShape(0);
    int64_t N = src.GetShape(1);
    int64_t C = src.GetShape(2);
    int64_t H = src.GetShape(3);
    int64_t W = src.GetShape(4);
    int64_t C1 = (C + C0 - 1) / C0;
    size_t Size = G * N * C1 * H * W * C0;

    std::fill(dst.data(), dst.data() + Size, 0);

    for (int64_t g = 0; g < G; ++g) {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                size_t r = c / C0;
                size_t cl = c % C0;
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        size_t srcIndex = w + W * h + W * H * c + W * H * C * n + W * H * C * N * g;
                        size_t dstIndex =
                            W * H * C1 * C0 * N * g + W * H * C1 * C0 * n + C0 * H * W * r + C0 * W * h + C0 * w + cl;
                        dst_ptr[dstIndex] = src_ptr[srcIndex];
                    }
                }
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData>
inline void TTRANS_NC1HWC02C1HWN1N0C0(DstTileData &dst, SrcTileData &src)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;

    const auto *src_ptr = reinterpret_cast<const SrcDType *>(src.data());
    auto *dst_ptr = reinterpret_cast<DstDType *>(dst.data());

    int64_t C1HW = dst.GetShape(0);
    int64_t N1 = dst.GetShape(1);
    int64_t N0 = dst.GetShape(2);
    int64_t C0 = dst.GetShape(3);

    int64_t N = src.GetShape(0);
    int64_t C1 = src.GetShape(1);
    int64_t H = src.GetShape(2);
    int64_t W = src.GetShape(3);

    size_t Size = C1HW * N1 * N0 * C0;

    std::fill(dst.data(), dst.data() + Size, 0);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c1 = 0; c1 < C1; ++c1) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    for (size_t c0 = 0; c0 < C0; c0++) {
                        size_t srcIndex = c0 + C0 * w + C0 * W * h + C0 * W * H * c1 + C0 * W * H * C1 * n;
                        size_t c1hw_idx = c1 * (H * W) + h * W + w;
                        size_t n1 = n / N0;
                        size_t n0 = n % N0;
                        size_t dstIndex = c0 + C0 * n0 + C0 * N0 * n1 + C0 * N0 * N1 * c1hw_idx;

                        dst_ptr[dstIndex] = src_ptr[srcIndex];
                    }
                }
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData>
inline void TTRANS_GNC1HWC02C1HWN1N0C0(DstTileData &dst, SrcTileData &src)
{
    using SrcDType = typename SrcTileData::DType;
    using DstDType = typename DstTileData::DType;

    const auto *src_ptr = reinterpret_cast<const SrcDType *>(src.data());
    auto *dst_ptr = reinterpret_cast<DstDType *>(dst.data());

    int64_t GC1HW = dst.GetShape(0);
    int64_t N1 = dst.GetShape(1);
    int64_t N0 = dst.GetShape(2);
    int64_t C0 = dst.GetShape(3);

    int64_t G = src.GetShape(0);
    int64_t N = src.GetShape(1);
    int64_t C1 = src.GetShape(2);
    int64_t H = src.GetShape(3);
    int64_t W = src.GetShape(4);

    size_t Size = GC1HW * N1 * N0 * C0;

    std::fill(dst.data(), dst.data() + Size, 0);

    for (int64_t g = 0; g < G; ++g) {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c1 = 0; c1 < C1; ++c1) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        for (size_t c0 = 0; c0 < C0; c0++) {
                            size_t srcIndex = c0 + C0 * w + C0 * W * h + C0 * W * H * c1 + C0 * W * H * C1 * n +
                                              C0 * W * H * C1 * N * g;
                            size_t c1hw_idx = g * C1 * H * W + c1 * (H * W) + h * W + w;
                            size_t n1 = n / N0;
                            size_t n0 = n % N0;
                            size_t dstIndex = c0 + C0 * n0 + C0 * N0 * n1 + C0 * N0 * N1 * c1hw_idx;

                            dst_ptr[dstIndex] = src_ptr[srcIndex];
                        }
                    }
                }
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData>
void TTrans_Impl(typename DstTileData::TileDType dst, typename SrcTileData::TileDType src, unsigned validRow,
                 unsigned validCol)
{
    for (size_t c = 0; c < validCol; c++) {
        size_t subTileSrcC = c / SrcTileData::InnerCols;
        size_t innerSrcC = c % SrcTileData::InnerCols;
        size_t subTileDstC = c / DstTileData::InnerCols;
        size_t innerDstC = c % DstTileData::InnerCols;

        for (size_t r = 0; r < validRow; r++) {
            size_t srcTileIdx, dstTileIdx;
            if constexpr (SrcTileData::SFractal == SLayout::NoneBox)
                srcTileIdx = GetTileElementOffsetPlain<SrcTileData>(r, c);
            else {
                size_t subTileR = r / SrcTileData::InnerRows;
                size_t innerR = r % SrcTileData::InnerRows;
                srcTileIdx = GetElementOffsetSubfractals<SrcTileData>(subTileSrcC, innerSrcC, subTileR, innerR);
            }

            if constexpr (DstTileData::SFractal == SLayout::NoneBox)
                dstTileIdx = GetTileElementOffsetPlain<DstTileData>(c, r);
            else {
                size_t subTileR = r / DstTileData::InnerRows;
                size_t innerR = r % DstTileData::InnerRows;
                dstTileIdx = GetElementOffsetSubfractals<DstTileData>(subTileR, innerR, subTileDstC, innerDstC);
            }
            dst[dstTileIdx] = src[srcTileIdx];
        }
    }
}

template <typename DstTileData, typename SrcTileData, typename TmpTileData>
PTO_INTERNAL void TTRANS_IMPL(DstTileData &dst, SrcTileData &src, TmpTileData &tmp)
{
    // Validate matching element widths at compilation
    static_assert(sizeof(typename SrcTileData::DType) == sizeof(typename DstTileData::DType),
                  "Data type sizes between source and destination tiles must match.");

    if constexpr (is_conv_tile_v<SrcTileData> && is_conv_tile_v<DstTileData>) {
        constexpr Layout src_layout = SrcTileData::layout;
        constexpr Layout dst_layout = DstTileData::layout;

        if constexpr (src_layout == Layout::NCHW && dst_layout == Layout::NC1HWC0) {
            TTRANS_NCHW2NC1HWC0(dst, src);
        } else if (src_layout == Layout::NC1HWC0 && dst_layout == Layout::FRACTAL_Z) {
            TTRANS_NC1HWC02C1HWN1N0C0(dst, src);
        } else if constexpr (src_layout == Layout::GNCHW && dst_layout == Layout::GNC1HWC0) {
            TTRANS_GNCHW2NC1HWC0(dst, src);
        } else if (src_layout == Layout::GNC1HWC0 && dst_layout == Layout::FRACTAL_Z) {
            TTRANS_GNC1HWC02C1HWN1N0C0(dst, src);
        }
    } else if constexpr (is_tile_data_v<SrcTileData> && is_tile_data_v<DstTileData>) {
        static_assert(SrcTileData::ValidRow == DstTileData::ValidCol && SrcTileData::ValidCol == DstTileData::ValidRow,
                      "Hardware matrix tiles transpose dimension sizes must mirror match.");
        unsigned validRow = src.GetValidRow();
        unsigned validCol = src.GetValidCol();
        TTrans_Impl<DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol);
    }
}

} // namespace pto

#endif // TTRANS_HPP