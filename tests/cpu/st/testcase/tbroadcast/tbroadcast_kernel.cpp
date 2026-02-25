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

using namespace pto;

template <int kTRows_, int kTCols_, int kTNumProc_>
AICORE void runTBroadcast(__gm__ float __out__ *out, __gm__ float __in__ *src)
{
    using TileTSrc = Tile<TileType::Vec, float, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using TileTDst = Tile<TileType::Vec, float, kTNumProc_ * kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using SrcShape = Shape<1, 1, 1, kTRows_, kTCols_>;
    using SrcStride = Stride<1, 1, 1, kTCols_, 1>;
    using SrcGTf = GlobalTensor<float, SrcShape, SrcStride>;
    using DstShape = Shape<1, 1, 1, kTNumProc_ * kTRows_, kTCols_>;
    using DstStride = Stride<1, 1, 1, kTCols_, 1>;
    using DstGTf = GlobalTensor<float, DstShape, DstStride>;

    TileTSrc srcTile(kTRows_, kTCols_);
    TileTDst dstTile(kTNumProc_ * kTRows_, kTCols_);

    SrcGTf srcGlobal(src);
    DstGTf dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    TEXPANDS(dstTile, 0.0f);
    TBROADCAST(dstTile, srcTile, kTNumProc_);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <int kTRows_, int kTCols_, int kTNumProc_>
void LaunchTBroadcast(float *out, float *src, void *stream)
{
    runTBroadcast<kTRows_, kTCols_, kTNumProc_>(out, src);
}

template void LaunchTBroadcast<16, 16, 2>(float *out, float *src, void *stream);