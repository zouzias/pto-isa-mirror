/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#include <acl/acl.h>

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace std;
using namespace pto;

template <typename SrcTile, typename DstTile, typename GlobalData, typename ScalarT, typename ComputeFn>
PTO_INTERNAL void RunVecBinaryWithManualEvents(DstTile &dstTile, SrcTile &srcTile, GlobalData &dstGlobal,
                                               GlobalData &srcGlobal, ScalarT scalar, ComputeFn &&compute)
{
    TLOAD(dstTile, dstGlobal);
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    compute(dstTile, srcTile, scalar);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTSubS(__gm__ T *out, __gm__ T *src, T scalar)
{
    using Shape2D = Shape<1, 1, 1, -1, -1>;
    using Stride2D = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, Shape2D, Stride2D>;
    using SrcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;

    const Shape2D dynShape(validRow, validCol);
    GlobalData dstGlobal(out, dynShape, Stride2D(dstTileRow, dstTileCol));
    GlobalData srcGlobal(src, dynShape, Stride2D(row, col));
    SrcTileData srcTile(validRow, validCol);
    DstTileData dstTile(validRow, validCol);
    TASSIGN<0x0>(srcTile);
    TASSIGN<0x28000>(dstTile);
    RunVecBinaryWithManualEvents(dstTile, srcTile, dstGlobal, srcGlobal, scalar,
                                 [](auto &dst, auto &srcTileRef, auto value) { TSUBS(dst, srcTileRef, value); });
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTSUBSCase1(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTSubS<float, 32, 128, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase2(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTSubS<half, 63, 128, 63, 63, 64, 64>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase3(__gm__ int32_t *out, __gm__ int32_t *src, int32_t scalar)
{
    runTSubS<int32_t, 31, 256, 31, 31, 128, 128>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase4(__gm__ int16_t *out, __gm__ int16_t *src, int16_t scalar)
{
    runTSubS<int16_t, 15, 192, 15, 15, 192, 192>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase5(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTSubS<float, 7, 512, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase6(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTSubS<float, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}

template <uint32_t caseId>
void launchTSUBSTestCase(void *out, void *src, float scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTSUBSCase1<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 2: {
            launchTSUBSCase2<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 3: {
            launchTSUBSCase3<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src, scalar);
            break;
        }
        case 4: {
            launchTSUBSCase4<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src, scalar);
            break;
        }
        case 5: {
            launchTSUBSCase5<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 6: {
            launchTSUBSCase6<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        default: {
        }
    }
}

template void launchTSUBSTestCase<1>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<2>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<3>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<4>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<5>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<6>(void *out, void *src, float scalar, aclrtStream stream);
