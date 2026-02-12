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
#include "acl/acl.h"

using namespace pto;
// dst(m,n) = (src(m,n) - offset(m,1)) * scale(m,1)
// fp32        int8/16    fp32           fp32
template <typename dstDType, typename srcDType, int dstRows, int dstCols, int srcRows, int srcCols, int dstValidRows,
          int dstValidCols, int paraRows, int paraCols>
__global__ AICORE void runTDequant(__gm__ dstDType __out__ *out, __gm__ srcDType __in__ *src,
                                   __gm__ dstDType __in__ *scale, __gm__ dstDType __in__ *offset)
{
    using DstGlobalData = GlobalTensor<dstDType, Shape<1, 1, 1, dstRows, dstCols>, pto::Stride<1, 1, 1, dstCols, 1>>;
    using SrcGlobalData = GlobalTensor<srcDType, Shape<1, 1, 1, srcRows, srcCols>, pto::Stride<1, 1, 1, srcCols, 1>>;
    using ParaGlobalData = GlobalTensor<dstDType, Shape<1, 1, 1, paraRows, paraCols>, pto::Stride<1, 1, 1, paraCols, 1>>;
    GlobalData dstGlobal(dst);
    GlobalData srcGlobal(src);
    GlobalData scaleGlobal(scale);
    GlobalData offsetGlobal(offset);

    using DstTileData = Tile<TileType::Vec, dstDType, dstRows, dstCols, BLayout::RowMajor, -1, -1>;
    DstTileData dstTile(dstValidRows, dstValidCols);
    using SrcTileData = Tile<TileType::Vec, srcDType, srcRows, srcCols, BLayout::RowMajor, -1, -1>;
    SrcTileData srcTile(dstValidRows, dstValidCols);
    using ParaTileData = Tile<TileType::Vec, dstDType, paraRows, paraCols, BLayout::RowMajor, -1, -1>;
    ParaTileData scaleTile(dstValidRows, 1);
    ParaTileData offsetTile(dstValidRows, 1);
    TASSIGN(dstTile, 0x0);
    TASSIGN(srcTile, dstRows * dstCols * sizeof(dstDType));
    TASSIGN(scaleTile, dstRows * dstCols * sizeof(dstDType) + srcRows * srcCols * sizeof(srcDType));
    TASSIGN(offsetTile, dstRows * dstCols * sizeof(dstDType) + srcRows * srcCols * sizeof(srcDType) +
                        paraRows * paraCols * sizeof(dstDType));

    Event<Op::TLOAD, Op::TDEQUANT> event0;
    Event<Op::TDEQUANT, Op::TSTORE_VEC> event1;

    TLOAD(srcTile, srcGlobal);
    TLOAD(scaleTile, scaleGlobal);
    event0 = TLOAD(offsetTile, offsetGlobal);
    event1 = TDEQUANT(dstTile, srcTile, scaleTile, offsetTile, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename dstDType, typename srcDType, int dstRows, int dstCols, int srcRows, int srcCols, int dstValidRows,
          int dstValidCols, int paraRows, int paraCols>
void LaunchTDequant(dstDType *out, srcDType *src, dstDType *scale, dstDType *offset, void *stream)
{
    runTDequant<dstDType, srcDType, dstRows, dstCols, srcRows, srcCols, dstValidRows, dstValidCols, paraRows, paraCols>
        <<<1, nullptr, stream>>>(out, src, scale, offset);
}

// dstDType, srcDType, dstRows, dstCols, srcRows, srcCols, dstValidRows, dstValidCols, paraRows, paraCols
template void LaunchTDequant<float, int8_t, 32, 32, 32, 32, 32, 32, 32, 32>(float *out, int8_t *src, float *scale,
                                                                            float *offset, void *stream);
template void LaunchTDequant<float, int16_t, 32, 32, 32, 32, 32, 32, 32, 32>(float *out, int16_t *src, float *scale,
                                                                            float *offset, void *stream);