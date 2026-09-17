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

using namespace pto;

template <typename T, int kRows, int kCols>
AICORE void runTROWEXPAND(__gm__ T __out__* out, __gm__ T __in__* src)
{
    using ShapeMat = Shape<1, 1, 1, kRows, kCols>;
    using StrideMat = Stride<1, 1, 1, kCols, 1>;
    using GlobalMat = GlobalTensor<T, ShapeMat, StrideMat>;

    using TileMat = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1>;
    TileMat srcTile(kRows, kCols);
    TileMat dstTile(kRows, kCols);

    GlobalMat srcGlobal(src);
    GlobalMat dstGlobal(out);

    TASSIGN(srcTile, 0);
    TASSIGN(dstTile, kRows * kCols * sizeof(typename TileMat::DType));

    TLOAD(srcTile, srcGlobal);
    TROWEXPAND(dstTile, srcTile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kRows, int kCols>
void LaunchTROWEXPAND(T* out, T* src, void* stream)
{
    (void)stream;
    runTROWEXPAND<T, kRows, kCols>(out, src);
}

template void LaunchTROWEXPAND<float, 64, 64>(float* out, float* src, void* stream);
template void LaunchTROWEXPAND<int64_t, 64, 64>(int64_t* out, int64_t* src, void* stream);
template void LaunchTROWEXPAND<uint64_t, 64, 64>(uint64_t* out, uint64_t* src, void* stream);
