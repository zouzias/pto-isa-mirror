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
#include <vector>

using namespace pto;

// TLOAD with validShape < physicalShape on NC1HWC0 layout.
// Physical: [N, C1, H, W] with C0 derived from dtype.
// Valid:    [VN, VC1, VH, VW] — only valid portion is loaded from global to tile.
template <typename T, int N, int C1, int H, int W, int VN, int VC1, int VH>
void runTloadValid(__gm__ T* out, __gm__ T* src)
{
    constexpr int C0 = 32 / sizeof(T);
    constexpr uint32_t totalElements = N * C1 * H * W * C0;
    constexpr uint32_t bufferSize = totalElements * sizeof(T);

    using Stride5HD = Stride<(int64_t)C1 * H * W * C0, (int64_t)H * W * C0, (int64_t)W * C0, (int64_t)C0, 1>;
    using GShape = Shape<N, C1, H, W, C0>;
    GlobalTensor<T, GShape, Stride5HD, Layout::NC1HWC0> srcGlobal(src);

    using PhysShape = ConvTileShape<N, C1, H, W>;
    using ValidShape = ConvTileValidShape<VN, VC1, VH, W>;
    using MyTile = ConvTile<TileType::Mat, T, bufferSize, Layout::NC1HWC0, PhysShape, ValidShape>;
    MyTile convTile;

    std::vector<T> localBuffer(totalElements, 0);
    convTile.data() = reinterpret_cast<typename MyTile::TileDType>(localBuffer.data());

    TLOAD(convTile, srcGlobal);

    for (uint32_t i = 0; i < totalElements; i++) {
        out[i] = convTile.data()[i];
    }
}

extern "C" __global__ AICORE void launch_tload_valid(__gm__ uint8_t* o, __gm__ uint8_t* s)
{
    runTloadValid<half, 4, 2, 4, 4, 2, 1, 2>((__gm__ half*)o, (__gm__ half*)s);
}

template <int32_t testKey>
void launchTVALIDSHAPE(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream)
{
    if constexpr (testKey == 1) {
        launch_tload_valid(out, src);
    }
}

template void launchTVALIDSHAPE<1>(uint8_t*, uint8_t*, uint64_t*, void*);
