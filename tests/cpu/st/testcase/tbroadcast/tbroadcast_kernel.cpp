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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
AICORE void runTBroadcast(__gm__ T __out__ *dst0, __gm__ T __out__ *dst1, __gm__ T __in__ *src)
{
    constexpr size_t total_count = kGRows_ * kGCols_;
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GTensor = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, kTRows_, kTCols_, pto::BLayout::RowMajor, -1, -1>;
    int my_rank = 0;

    ShapeDyn fullShape(1, 1, 1, kGRows_, kGCols_);
    StrideDyn fullStride(total_count, total_count, total_count, kGCols_, 1);

    GTensor srcG(src, fullShape, fullStride);

    GTensor tensors[2];
    int nranks = 2;
    tensors[0] = GTensor(dst0, fullShape, fullStride);
    tensors[1] = GTensor(dst1, fullShape, fullStride);

    pto::comm::ParallelGroup<GTensor> group(tensors, nranks, my_rank);

    TileData stagingTile(kTRows_, kTCols_);
    TBROADCAST(group, srcG, stagingTile);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTBroadcast(T *dst0, T *dst1, T *src, void *stream)
{
    runTBroadcast<T, kGRows_, kGCols_, kTRows_, kTCols_>(dst0, dst1, src);
}

template void LaunchTBroadcast<float, 16, 16, 16, 16>(float *dst0, float *dst1, float *src, void *stream);