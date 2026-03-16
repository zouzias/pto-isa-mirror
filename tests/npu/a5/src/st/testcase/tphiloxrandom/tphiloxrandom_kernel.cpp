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
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <typename T, int rows, int cols, int validRows, int validCols>
PTO_INTERNAL void runTPhiloxRandom(__gm__ T *out)
{
    using DynDim2Shape = Shape<1, 1, 1, rows, cols>;
    using DynDim2Stride = pto::Stride<rows * cols, rows * cols, rows * cols, cols, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData dstGlobal(out);

    using DstTileData = Tile<TileType::Vec, T, rows, cols>;
    DstTileData dstTile;
    TASSIGN(dstTile, 0x0);

    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

    __VEC_SCOPE__
    {
        uint32_t sReg = 256;
        MaskReg pReg = CreatePredicate<T>(sReg);
        __ubuf__ T *dst = dstTile.data();
        RegTensor<uint32_t> ctr0, ctr1, ctr2, ctr3, vZero, incIdx;
        vci((RegTensor<int32_t> &)incIdx, 0, INC_ORDER);

        
        vdup(ctr0, 0, pReg, MODE_ZEROING);
        vdup(ctr1, 1, pReg, MODE_ZEROING);
        vdup(ctr2, 2, pReg, MODE_ZEROING);
        vdup(ctr3, 3, pReg, MODE_ZEROING);
        vdup(vZero, 0, pReg, MODE_ZEROING);

        MaskReg pd;
        vaddc(pd, ctr0, ctr0, (RegTensor<uint32_t> &)incIdx, pReg);
        vaddcs(pd, ctr1, ctr1, vZero, pd, pReg);
        vaddcs(pd, ctr2, ctr2, vZero, pd, pReg);
        vaddcs(pd, ctr3, ctr3, vZero, pd, pReg);
        for (uint16_t i = 1; i < 10; ++i) {
            vaddcs(pd, ctr0, ctr0, vZero, pd, pReg);
            vaddcs(pd, ctr1, ctr1, vZero, pd, pReg);
            vaddcs(pd, ctr2, ctr2, vZero, pd, pReg);
            vaddcs(pd, ctr3, ctr3, vZero, pd, pReg);
        }

        vsts(ctr0, dst, 64, NORM_B32, pReg, POST_UPDATE);
        vsts(ctr1, dst, 64, NORM_B32, pReg, POST_UPDATE);
        vsts(ctr2, dst, 64, NORM_B32, pReg, POST_UPDATE);
        vsts(ctr3, dst, 64, NORM_B32, pReg, POST_UPDATE);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

extern "C" __global__ AICORE void launchTPHILOXRANDOMCase01(__gm__ uint32_t *out)
{
    runTPhiloxRandom<uint32_t, 4, 64, 4, 64>(out);
}

template <uint32_t caseId>
void launchTPHILOXRANDOMTestCase(void *out, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTPHILOXRANDOMCase01<<<1, nullptr, stream>>>((uint32_t *)out);
            break;
        }
        default: {
        }
    }
}

template void launchTPHILOXRANDOMTestCase<1>(void *out, aclrtStream stream);