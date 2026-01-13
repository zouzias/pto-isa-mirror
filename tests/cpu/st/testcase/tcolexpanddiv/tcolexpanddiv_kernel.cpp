/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <type_traits>

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
using namespace pto;

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col>
AICORE void runTColExpandDiv(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using Src0Shape = Shape<1, 1, 1, dstRow, dstCol>;
    using Src0Stride = Stride<1, 1, 1, dstCol, 1>;
    using Src1Shape = Shape<1, 1, 1, src1Row, src1Col>;
    using Src1Stride = Stride<1, 1, 1, src1Col, 1>;

    using Src0Global = GlobalTensor<T, Src0Shape, Src0Stride>;
    using Src1Global = GlobalTensor<T, Src1Shape, Src1Stride>;

    using Src0Tile = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, -1, -1>;
    using Src1Tile = Tile<TileType::Vec, T, src1Row, src1Col, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, -1, -1>;

    Src0Tile src0Tile(dstRow, dstCol);
    Src1Tile src1Tile(src1Row, src1Col);
    DstTile dstTile(dstRow, dstCol);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    Src0Global src0Global(src0);
    Src1Global src1Global(src1);
    Src0Global dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TCOLEXPANDDIV(dstTile, src0Tile, src1Tile);
    TSTORE(dstGlobal, dstTile);

    out = dstGlobal.data();
}

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col>
void LaunchTColExpandDiv(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTColExpandDiv<half, dstRow, dstCol, src1Row, src1Col>(
            (half *)out, (half *)src0, (half *)src1);
    } else {
        runTColExpandDiv<T, dstRow, dstCol, src1Row, src1Col>(out, src0, src1);
    }
}

template void LaunchTColExpandDiv<float, 32, 64, 1, 64>(float *out, float *src0, float *src1, void *stream);
template void LaunchTColExpandDiv<float, 8, 32, 1, 32>(float *out, float *src0, float *src1, void *stream);
template void LaunchTColExpandDiv<aclFloat16, 16, 64, 1, 64>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);
template void LaunchTColExpandDiv<aclFloat16, 4, 128, 1, 128>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);
