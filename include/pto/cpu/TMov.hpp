/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMOV_HPP
#define TMOV_HPP

#include <cassert>
#include <algorithm>
#include <pto/common/constants.hpp>
#include <pto/common/cpu_stub.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    assert(src.GetValidRow() == dst.GetValidRow() && src.GetValidRow() == dst.GetValidRow());
    for (size_t c = 0; c < src.GetValidCol(); c++) {
        size_t subTileSrcC = c / SrcTileData::InnerCols;
        size_t innerSrcC = c % SrcTileData::InnerCols;
        size_t subTileDstC = c / DstTileData::InnerCols;
        size_t innerDstC = c % DstTileData::InnerCols;

        for (size_t r = 0; r < src.GetValidRow(); r++) {
            size_t srcTileIdx;
            size_t dstTileIdx;
            if constexpr (SrcTileData::SFractal == SLayout::NoneBox) {
                srcTileIdx = GetTileElementOffsetPlain<SrcTileData>(r, c);
            } else {
                size_t subTileR = r / SrcTileData::InnerRows;
                size_t innerR = r % SrcTileData::InnerRows;
                srcTileIdx = GetTileElementOffsetSubfractals<SrcTileData>(subTileR, innerR, subTileSrcC, innerSrcC);
            }

            if constexpr (DstTileData::SFractal == SLayout::NoneBox) {
                dstTileIdx = GetTileElementOffsetPlain<DstTileData>(r, c);
            } else {
                size_t subTileR = r / DstTileData::InnerRows;
                size_t innerR = r % DstTileData::InnerRows;
                dstTileIdx = GetTileElementOffsetSubfractals<DstTileData>(subTileR, innerR, subTileDstC, innerDstC);
            }
            dst.data()[dstTileIdx] = src.data()[srcTileIdx];
        }
    }
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    TMOV_IMPL(dst, src);
    if constexpr (reluMode == ReluPreMode::NormalRelu) {
        const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
        const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                auto &v = dst.data()[GetTileElementOffset<DstTileData>(r, c)];
                if (v < static_cast<typename DstTileData::DType>(0)) {
                    v = static_cast<typename DstTileData::DType>(0);
                }
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    if constexpr (mode == AccToVecMode::DualModeSplitM) {
        // CPU simulator: both Vec subblocks execute sequentially on the same host memory.
        // Overwrite approach (write at offset 0) is correct because subblock N's data
        // is always consumed before subblock N+1 overwrites it.
        // The kernel code must use two separate runningOTile tiles (one per subblock)
        // for the O accumulation, since each subblock's accumulated result must persist
        // independently across tiles.
        const size_t dstRows = dst.GetValidRow();
        const size_t dstCols = dst.GetValidCol();
        const size_t halfRows = dstRows;
        const uint32_t subblock_id = get_subblockid();
        const size_t srcRowOffset = static_cast<size_t>(subblock_id) * halfRows;
        for (size_t c = 0; c < dstCols; c++) {
            for (size_t dr = 0; dr < halfRows; dr++) {
                size_t sr = srcRowOffset + dr;
                size_t srcIdx;
                size_t dstIdx;
                if constexpr (SrcTileData::SFractal == SLayout::NoneBox) {
                    srcIdx = GetTileElementOffsetPlain<SrcTileData>(sr, c);
                } else {
                    size_t subTileSrcR = sr / SrcTileData::InnerRows;
                    size_t innerSrcR = sr % SrcTileData::InnerRows;
                    size_t subTileSrcC = c / SrcTileData::InnerCols;
                    size_t innerSrcC = c % SrcTileData::InnerCols;
                    srcIdx = GetTileElementOffsetSubfractals<SrcTileData>(subTileSrcR, innerSrcR, subTileSrcC, innerSrcC);
                }
                if constexpr (DstTileData::SFractal == SLayout::NoneBox) {
                    dstIdx = GetTileElementOffsetPlain<DstTileData>(dr, c);
                } else {
                    size_t subTileDstR = dr / DstTileData::InnerRows;
                    size_t innerDstR = dr % DstTileData::InnerRows;
                    size_t subTileDstC = c / DstTileData::InnerCols;
                    size_t innerDstC = c % DstTileData::InnerCols;
                    dstIdx = GetTileElementOffsetSubfractals<DstTileData>(subTileDstR, innerDstR, subTileDstC, innerDstC);
                }
                dst.data()[dstIdx] = src.data()[srcIdx];
            }
        }
        if constexpr (reluMode == ReluPreMode::NormalRelu) {
            for (size_t dr = 0; dr < halfRows; ++dr) {
                for (size_t c = 0; c < dstCols; ++c) {
                    auto &v = dst.data()[GetTileElementOffset<DstTileData>(dr, c)];
                    if (v < static_cast<typename DstTileData::DType>(0)) {
                        v = static_cast<typename DstTileData::DType>(0);
                    }
                }
            }
        }
    } else if constexpr (mode == AccToVecMode::DualModeSplitN) {
        // CPU simulator: same overwrite logic as DualModeSplitM.
        const size_t dstRows = dst.GetValidRow();
        const size_t dstCols = dst.GetValidCol();
        const size_t halfCols = dstCols;
        const uint32_t subblock_id = get_subblockid();
        const size_t srcColOffset = static_cast<size_t>(subblock_id) * halfCols;
        for (size_t dc = 0; dc < halfCols; dc++) {
            size_t sc = srcColOffset + dc;
            for (size_t r = 0; r < dstRows; r++) {
                size_t srcIdx;
                size_t dstIdx;
                if constexpr (SrcTileData::SFractal == SLayout::NoneBox) {
                    srcIdx = GetTileElementOffsetPlain<SrcTileData>(r, sc);
                } else {
                    size_t subTileSrcR = r / SrcTileData::InnerRows;
                    size_t innerSrcR = r % SrcTileData::InnerRows;
                    size_t subTileSrcC = sc / SrcTileData::InnerCols;
                    size_t innerSrcC = sc % SrcTileData::InnerCols;
                    srcIdx = GetTileElementOffsetSubfractals<SrcTileData>(subTileSrcR, innerSrcR, subTileSrcC, innerSrcC);
                }
                if constexpr (DstTileData::SFractal == SLayout::NoneBox) {
                    dstIdx = GetTileElementOffsetPlain<DstTileData>(r, dc);
                } else {
                    size_t subTileDstR = r / DstTileData::InnerRows;
                    size_t innerDstR = r % DstTileData::InnerRows;
                    size_t subTileDstC = dc / DstTileData::InnerCols;
                    size_t innerDstC = dc % DstTileData::InnerCols;
                    dstIdx = GetTileElementOffsetSubfractals<DstTileData>(subTileDstR, innerDstR, subTileDstC, innerDstC);
                }
                dst.data()[dstIdx] = src.data()[srcIdx];
            }
        }
        if constexpr (reluMode == ReluPreMode::NormalRelu) {
            for (size_t r = 0; r < dstRows; ++r) {
                for (size_t dc = 0; dc < halfCols; ++dc) {
                    auto &v = dst.data()[GetTileElementOffset<DstTileData>(r, dc)];
                    if (v < static_cast<typename DstTileData::DType>(0)) {
                        v = static_cast<typename DstTileData::DType>(0);
                    }
                }
            }
        }
    } else {
        TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src);
    }
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp)
{
    (void)fp;
    TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp)
{
    (void)mode;
    (void)fp;
    TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar)
{
    (void)preQuantScalar;
    TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src);
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar)
{
    (void)mode;
    (void)preQuantScalar;
    TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src);
}
} // namespace pto
#endif // TMOV_HPP
