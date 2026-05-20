/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

template <typename T, int srcRow, int srcValidRow, int dstRow, int col, int validCol>
PTO_INTERNAL void runTColProd(__gm__ T __out__ *out, __gm__ T __in__ *src, bool isBinary)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(srcValidRow, validCol), DynDim2Stride(srcRow, col));
    GlobalData dstGlobal(out, DynDim2Shape(dstRow, validCol), DynDim2Stride(dstRow, col));

    using SrcTileData = Tile<TileType::Vec, T, srcRow, col, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, dstRow, col, BLayout::RowMajor, -1, -1>;
    SrcTileData srcTile(srcValidRow, validCol);
    DstTileData dstTile(dstRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, srcRow * col * sizeof(T));

    // 搬运数据
    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TCOLPROD(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTCOLPRODCase01(__gm__ float *out, __gm__ float *src)
{
    runTColProd<float, 32, 32, 1, 512, 512>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase02(__gm__ float *out, __gm__ float *src)
{
    runTColProd<float, 512, 512, 1, 32, 32>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase03(__gm__ float *out, __gm__ float *src)
{
    runTColProd<float, 128, 128, 1, 128, 128>(out, src, false);
}

template <uint32_t caseId>
void launchTCOLPRODTestCase(void *out, void *src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTCOLPRODCase01<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 2: {
            launchTCOLPRODCase02<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 3: {
            launchTCOLPRODCase03<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        default: {
        }
    }
}

template void launchTCOLPRODTestCase<1>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<2>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<3>(void *out, void *src, aclrtStream stream);