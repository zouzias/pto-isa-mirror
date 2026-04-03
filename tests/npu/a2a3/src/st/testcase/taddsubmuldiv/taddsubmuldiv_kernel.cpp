/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to License for details. You may not use this file except in compliance with License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

namespace TAddSubMulDiv2 {

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTAddSubMulDiv2(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1,
                                                __gm__ T __in__ *src2, __gm__ T __in__ *src3, __gm__ T __in__ *src4)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData src2Tile(vRows, vCols);
    TileData src3Tile(vRows, vCols);
    TileData src4Tile(vRows, vCols);
    TileData tmpTile(vRows, vCols);
    TileData dstTile(vRows, vCols);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, sizeof(T) * TileData::Numel);
    TASSIGN(src2Tile, 2 * sizeof(T) * TileData::Numel);
    TASSIGN(src3Tile, 3 * sizeof(T) * TileData::Numel);
    TASSIGN(src4Tile, 4 * sizeof(T) * TileData::Numel);
    TASSIGN(tmpTile, 5 * sizeof(T) * TileData::Numel);
    TASSIGN(dstTile, 6 * sizeof(T) * TileData::Numel);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData src2Global(src2);
    GlobalData src3Global(src3);
    GlobalData src4Global(src4);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TADD> event0;
    Event<Op::TADD, Op::TSUB> event1;
    Event<Op::TSUB, Op::TMUL> event2;
    Event<Op::TMUL, Op::TDIV> event3;
    Event<Op::TDIV, Op::TSTORE_VEC> event4;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(src1Tile, src1Global);
    event1 = TADD(tmpTile, src0Tile, src1Tile, event0);
    event2 = TSUB(tmpTile, tmpTile, src2Tile, event1);
    event3 = TMUL(tmpTile, tmpTile, src3Tile, event2);
    event4 = TDIV(dstTile, tmpTile, src4Tile, event3);
    TSTORE(dstGlobal, dstTile, event4);
    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void launchTAddSubMulDiv2(T *out, T *src0, T *src1, T *src2, T *src3, T *src4, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTAddSubMulDiv2<half, kTRows_, kTCols_, vRows, vCols>
            <<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1, (half *)src2, (half *)src3, (half *)src4);
    } else {
        runTAddSubMulDiv2<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1, src2, src3, src4);
    }
}

template void launchTAddSubMulDiv2<float, 64, 64, 64, 64>(float *out, float *src0, float *src1, float *src2,
                                                                  float *src3, float *src4, void *stream);
template void launchTAddSubMulDiv2<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
                                                                        aclFloat16 *src2, aclFloatFloat16 *src3, aclFloat16 *src4, void *stream);

}  // namespace TAddSubMulDiv2