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

#include <pto/pto-inst.hpp>

#include "tests/npu/common/kernel_common.hpp"

using namespace std;
using namespace pto;
using namespace pto::test;

template <typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTMuls(__gm__ T *out, __gm__ T *src, T scalar)
{
    using Shape2D = Shape<1, 1, 1, -1, -1>;
    using Stride2D = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, Shape2D, Stride2D>;
    using DstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    using SrcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;

    const Shape2D dynShape(validRow, validCol);
    DstTileData dstTile(validRow, validCol);
    SrcTileData srcTile(validRow, validCol);
    GlobalData srcGlobal(src, dynShape, Stride2D(row, col));
    GlobalData dstGlobal(out, dynShape, Stride2D(dstTileRow, dstTileCol));
    TASSIGN<0x0>(srcTile);
    TASSIGN<SrcTileData::Numel * sizeof(T)>(dstTile);
    RunVecBinaryWithManualEvents(dstTile, srcTile, dstGlobal, srcGlobal, scalar,
                                 [](auto &dst, auto &srcTileRef, auto value) { TMULS(dst, srcTileRef, value); });
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTMULSCase1(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTMuls<float, 32, 128, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase2(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTMuls<half, 63, 128, 63, 63, 64, 64>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTMULSCase3(__gm__ int32_t *out, __gm__ int32_t *src, int32_t scalar)
{
    runTMuls<int32_t, 31, 256, 31, 31, 128, 128>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase4(__gm__ int16_t *out, __gm__ int16_t *src, int16_t scalar)
{
    runTMuls<int16_t, 15, 192, 15, 15, 192, 192>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase5(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTMuls<float, 7, 512, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase6(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTMuls<float, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}

template <uint32_t caseId>
void launchTMULSTestCase(void *out, void *src, float scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTMULSCase1<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 2: {
            launchTMULSCase2<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 3: {
            launchTMULSCase3<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src, scalar);
            break;
        }
        case 4: {
            launchTMULSCase4<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src, scalar);
            break;
        }
        case 5: {
            launchTMULSCase5<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 6: {
            launchTMULSCase6<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        default: {
        }
    }
}

template void launchTMULSTestCase<1>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<2>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<3>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<4>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<5>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<6>(void *out, void *src, float scalar, aclrtStream stream);
