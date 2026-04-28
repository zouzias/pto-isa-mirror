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
#include <pto/common/pto_tile.hpp>
#include "acl/acl.h"

using namespace pto;

/**
 * Grad_w reduction kernel: grad_w_partial -> sum -> multiply with weights -> accumulate
 * Data flow: gm -> ub (TLOAD) -> vector (TCOLSUM, TCVT, TMUL, TADD) -> ub -> gm (TSTORE)
 */
template <typename T, typename W, int kTRows_, int kTCols_, int num_blocks, int vRows, int vCols>
__global__ AICORE void runEngramGradWReduce(
    __gm__ T __out__ *grad_weight_hidden,
    __gm__ T __out__ *grad_weight_embed,
    __gm__ T __in__ *grad_w_partial,
    __gm__ W __in__ *weight_hidden,
    __gm__ W __in__ *weight_embed)
{
    // Shape definitions for grad_w_partial: (num_blocks, vCols)
    using DynShapePartial = Shape<1, 1, 1, num_blocks, vCols>;
    using DynStridePartial = Stride<1, 1, 1, vCols, 1>;
    using GlobalPartial = GlobalTensor<T, DynShapePartial, DynStridePartial>;

    // Shape for reduced result: (1, vCols) - single row after sum
    using DynShapeReduced = Shape<1, 1, 1, 1, vCols>;
    using DynStrideReduced = Stride<1, 1, 1, vCols, 1>;
    using GlobalReduced = GlobalTensor<T, DynShapeReduced, DynStrideReduced>;

    // Shape for weights: (1, vCols) - single row
    using DynShapeWeight = Shape<1, 1, 1, 1, vCols>;
    using DynStrideWeight = Stride<1, 1, 1, vCols, 1>;
    using GlobalWeight = GlobalTensor<W, DynShapeWeight, DynStrideWeight>;

    // Shape for output gradients: (1, vCols)
    using GlobalGrad = GlobalTensor<T, DynShapeReduced, DynStrideReduced>;

    // Tile definitions
    // Source tile for grad_w_partial: multiple rows to reduce
    using SrcTileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, num_blocks, vCols>;
    // Reduced tile: single row after TCOLSUM
    using ReducedTileData = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, 1, vCols>;
    // Weight tiles (bfloat16)
    using WeightTileData = Tile<TileType::Vec, W, 1, kTCols_, BLayout::RowMajor, 1, vCols>;
    // Float tiles for converted weights
    using FloatTileData = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, 1, vCols>;
    // Temporary tile for multiplication result
    using TmpTileData = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, 1, vCols>;
    // Output tiles
    using OutTileData = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, 1, vCols>;

    // Create tiles with default constructor
    SrcTileData srcTile;
    ReducedTileData reducedTile;
    WeightTileData whTile;
    WeightTileData weTile;
    FloatTileData whFloatTile;
    FloatTileData weFloatTile;
    TmpTileData tmpWhTile;
    TmpTileData tmpWeTile;
    OutTileData gradWhTile;
    OutTileData gradWeTile;

    // Assign buffer addresses using template syntax
    size_t srcTileSize = sizeof(T) * SrcTileData::Numel;
    size_t reducedTileSize = sizeof(T) * ReducedTileData::Numel;
    size_t weightTileSize = sizeof(W) * WeightTileData::Numel;
    size_t floatTileSize = sizeof(T) * FloatTileData::Numel;
    size_t tmpTileSize = sizeof(T) * TmpTileData::Numel;
    size_t outTileSize = sizeof(T) * OutTileData::Numel;

    TASSIGN<0x0>(srcTile);
    TASSIGN<srcTileSize>(reducedTile);
    TASSIGN<srcTileSize + reducedTileSize>(whTile);
    TASSIGN<srcTileSize + reducedTileSize + weightTileSize>(weTile);
    TASSIGN<srcTileSize + reducedTileSize + 2 * weightTileSize>(whFloatTile);
    TASSIGN<srcTileSize + reducedTileSize + 2 * weightTileSize + floatTileSize>(weFloatTile);
    TASSIGN<srcTileSize + reducedTileSize + 2 * weightTileSize + 2 * floatTileSize>(tmpWhTile);
    TASSIGN<srcTileSize + reducedTileSize + 2 * weightTileSize + 2 * floatTileSize + tmpTileSize>(tmpWeTile);
    TASSIGN<srcTileSize + reducedTileSize + 2 * weightTileSize + 2 * floatTileSize + 2 * tmpTileSize>(gradWhTile);
    TASSIGN<srcTileSize + reducedTileSize + 2 * weightTileSize + 2 * floatTileSize + 2 * tmpTileSize + outTileSize>(gradWeTile);

    // Initialize global tensors with just pointer
    GlobalPartial partialGlobal(grad_w_partial);
    GlobalWeight whGlobal(weight_hidden);
    GlobalWeight weGlobal(weight_embed);
    GlobalGrad gradWhGlobal(grad_weight_hidden);
    GlobalGrad gradWeGlobal(grad_weight_embed);

    // Step 1: Load grad_w_partial
    TLOAD(srcTile, partialGlobal);

    // Step 3: Load weights (bfloat16)
    TLOAD(whTile, whGlobal);
    TLOAD(weTile, weGlobal);

    // Step 5: Load existing gradients for accumulation
    TLOAD(gradWhTile, gradWhGlobal);
    TLOAD(gradWeTile, gradWeGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    // Step 2: Reduce using TCOLSUM (sum over rows)
    TCOLSUM(reducedTile, srcTile);

    // Step 4: Convert bfloat16 to float
    TCVT(whFloatTile, whTile, RoundMode::CAST_RINT);
    TCVT(weFloatTile, weTile, RoundMode::CAST_RINT);

    // Step 6: Multiply grad_w_sum with weights
    // grad_wh += grad_w_sum * weight_embed
    TMUL(tmpWhTile, reducedTile, weFloatTile);
    // grad_we += grad_w_sum * weight_hidden
    TMUL(tmpWeTile, reducedTile, whFloatTile);

    // Step 7: Accumulate using TADD
    TADD(gradWhTile, gradWhTile, tmpWhTile);
    TADD(gradWeTile, gradWeTile, tmpWeTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
#endif

    // Step 8: Store results
    TSTORE(gradWhGlobal, gradWhTile);
    TSTORE(gradWeGlobal, gradWeTile);
}

template <bool isBf16, int kTRows_, int kTCols_, int num_blocks, int vRows, int vCols>
void LaunchEngramGradWReduce(
    void *grad_weight_hidden,
    void *grad_weight_embed,
    void *grad_w_partial,
    void *weight_hidden,
    void *weight_embed,
    void *stream)
{
    using T = float;
    using W = bfloat16_t;

    runEngramGradWReduce<T, W, kTRows_, kTCols_, num_blocks, vRows, vCols>
        <<<1, nullptr, stream>>>((T *)grad_weight_hidden, (T *)grad_weight_embed, (T *)grad_w_partial,
                                 (W *)weight_hidden, (W *)weight_embed);
}

// Template instantiations for test configurations
template void LaunchEngramGradWReduce<true, 4, 128, 4, 4, 128>(
    void *grad_weight_hidden, void *grad_weight_embed, void *grad_w_partial,
    void *weight_hidden, void *weight_embed, void *stream);

template void LaunchEngramGradWReduce<true, 8, 256, 8, 8, 256>(
    void *grad_weight_hidden, void *grad_weight_embed, void *grad_w_partial,
    void *weight_hidden, void *weight_embed, void *stream);

template void LaunchEngramGradWReduce<true, 16, 512, 16, 16, 512>(
    void *grad_weight_hidden, void *grad_weight_embed, void *grad_w_partial,
    void *weight_hidden, void *weight_embed, void *stream);