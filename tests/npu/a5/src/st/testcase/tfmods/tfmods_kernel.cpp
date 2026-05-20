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
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTFModS(__gm__ T *out, __gm__ T *src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using srcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileRow, dstTileCol));
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, row * col * sizeof(T));
// causes issues in automode as the tile returned from the TLOAD tfcall appears unused and this tload may not finish
// before the second tload
#ifndef __PTO_AUTO__
    TLOAD(dstTile, dstGlobal);
#endif
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TFMODS(dstTile, srcTile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTFMODSCase1(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTFModS<float, 32, 512, 32, 32, 512, 512>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTFMODSCase2(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTFModS<float, 512, 32, 512, 512, 32, 32>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTFMODSCase3(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTFModS<float, 128, 128, 128, 128, 128, 128>(out, src, scalar);
}

template <uint32_t caseId>
void launchTFMODSTestCase(void *out, void *src, float scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTFMODSCase1<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 2: {
            launchTFMODSCase2<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 3: {
            launchTFMODSCase3<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        default: {
        }
    }
}

template void launchTFMODSTestCase<1>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTFMODSTestCase<2>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTFMODSTestCase<3>(void *out, void *src, float scalar, aclrtStream stream);