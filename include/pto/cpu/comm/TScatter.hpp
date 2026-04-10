/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TSCATTER_HPP
#define PTO_COMM_TSCATTER_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TSCATTER_IMPL: Scatter operation - root distributes data to all ranks
//
// The calling NPU (root) splits local source data along DIM_3 (row dimension)
// and distributes the portions to each rank in the parallel group.
// This is the inverse of TGATHER.
//
// The source tensor has shape (D0, D1, D2, N*H, W). Each rank r receives
// rows [r*H, (r+1)*H), i.e., per-rank data of shape (D0, D1, D2, H, W).
//
// When the per-rank data exceeds the UB tile capacity in rows and/or columns,
// the transfer is automatically chunked via 2D sliding:
//   - Outer dimensions (DIM_0, DIM_1, DIM_2) are iterated explicitly.
//   - DIM_3 (rows) is split into tileValidRow-sized chunks.
//   - DIM_4 (cols) is split into tileValidCol-sized chunks.
//
// Constraints for chunked mode:
//   - If TileData has static ValidRow, per-rank DIM_3 must be divisible by ValidRow.
//   - If TileData has static ValidCol, DIM_4 must be divisible by ValidCol.
//   - All destination tensors in the ParallelGroup are assumed to have the same shape/strides.
// ============================================================================

template <typename GlobalDataDst, typename T>
void Scatter(
    typename GlobalDataDst::DType *dst, 
    std::vector<T> src, 
    long int dstShape[], 
    long int dstStride[],
    long int srcOffset
)
{
    int srcIndex = 0;
    for (size_t i = 0; i < dstShape[0]; i++) {
        for (size_t j = 0; j < dstShape[1]; j++) {
            for (size_t k = 0; k < dstShape[2]; k++) {
                for (size_t l = 0; l < dstShape[3]; l++) {
                    for (size_t m = 0; m < dstShape[4]; m++) {
                        int dstIndex = i * dstStride[0] + j * dstStride[1] + 
                                k * dstStride[2] + l * dstStride[3] + m * dstStride[4];
                        dst[dstIndex] = src[srcIndex++ + srcOffset];
                    }
                }
            }
        }
    }
}


template <typename ParallelGroupType, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TSCATTER_IMPL(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &stagingTileData)
{
    using GlobalDstData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    T* srcArray = srcGlobalData.data();

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TSCATTER: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TSCATTER: TileData element type must match GlobalData element type");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TSCATTER: src/dst layout mismatch");

    const int nranks = parallelGroup.GetSize();
    const int rootIdx = parallelGroup.GetRootIdx();

    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");
    PTO_ASSERT(rootIdx >= 0 && rootIdx < nranks, "rootIdx must be in range [0, nranks)!");

    constexpr size_t numDims = 5;
    long int srcShape[numDims] = {
        srcGlobalData.GetShape(0), 
        srcGlobalData.GetShape(1), 
        srcGlobalData.GetShape(2), 
        srcGlobalData.GetShape(3), 
        srcGlobalData.GetShape(4)
    };
    long int srcStride[numDims] = {
        srcGlobalData.GetStride(0), 
        srcGlobalData.GetStride(1), 
        srcGlobalData.GetStride(2), 
        srcGlobalData.GetStride(3),
        srcGlobalData.GetStride(4)
    };

    int srcSize = srcShape[0] * srcShape[1] * srcShape[2] * srcShape[3] * srcShape[4];
    std::vector<T> buffer(srcSize);

    int bufferIndx = 0;
    for (size_t i = 0; i < srcShape[0]; i++) {
        for (size_t j = 0; j < srcShape[1]; j++) {
            for (size_t k = 0; k < srcShape[2]; k++) {
                for (size_t l = 0; l < srcShape[3]; l++) {
                    for (size_t m = 0; m < srcShape[4]; m++) {
                        int index = i * srcStride[0] + j * srcStride[1] + 
                            k * srcStride[2] + l * srcStride[3] + m * srcStride[4];
                        buffer[bufferIndx++] = srcArray[index];
                    }
                }
            }
        }
    }

    size_t elementsPerRank = srcSize / nranks;

    for (int r = 0; r < nranks; ++r)
    {
        GlobalDstData& dstGlobalData = parallelGroup[r];
        long int dstShape[numDims] = {
            dstGlobalData.GetShape(0), 
            dstGlobalData.GetShape(1), 
            dstGlobalData.GetShape(2), 
            dstGlobalData.GetShape(3), 
            dstGlobalData.GetShape(4)
        };
        long int dstStride[numDims] = {
            dstGlobalData.GetStride(0), 
            dstGlobalData.GetStride(1), 
            dstGlobalData.GetStride(2), 
            dstGlobalData.GetStride(3),
            dstGlobalData.GetStride(4)
        };

        long int currentSrcOffset = r * elementsPerRank;
        Scatter<GlobalDstData, T>(
            dstGlobalData.data(),
            buffer,
            dstShape, dstStride,
            currentSrcOffset
        );
    }
    
}

// ============================================================================
// TSCATTER_IMPL (ping-pong): Scatter with double buffering
//
// Uses two staging tiles (pingTile, pongTile) to overlap TLOAD of the next
// chunk (MTE2) with TSTORE of the current chunk (MTE3).
//
// Timeline without ping-pong:
//   [TLOAD chunk0] -> [TSTORE chunk0] -> [TLOAD chunk1] -> [TSTORE chunk1] -> ...
//
// Timeline with ping-pong:
//   [TLOAD chunk0] -> [TSTORE chunk0 | TLOAD chunk1] -> [TSTORE chunk1 | TLOAD chunk2] -> ...
//
// Constraints: same as TSCATTER_IMPL for chunked mode.
// ============================================================================

template <typename ParallelGroupType, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TSCATTER_IMPL(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData, TileData &pingTile,
                                TileData &pongTile)
{
    using GlobalDstData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TSCATTER: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TSCATTER: TileData element type must match GlobalData element type");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TSCATTER: src/dst layout mismatch");

    const int nranks = parallelGroup.GetSize();
    const int rootIdx = parallelGroup.GetRootIdx();

    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");
    PTO_ASSERT(rootIdx >= 0 && rootIdx < nranks, "rootIdx must be in range [0, nranks)!");

    pto::comm::TSCATTER_IMPL(parallelGroup, srcGlobalData, pingTile);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TSCATTER_HPP
