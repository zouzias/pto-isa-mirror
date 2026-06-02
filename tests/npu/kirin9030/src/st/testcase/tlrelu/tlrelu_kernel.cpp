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

template <typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTLRelu(__gm__ T *out, __gm__ T *src, T scalar)
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
    TASSIGN<0x0>(srcTile);
    TASSIGN<srcTileData::Numel * sizeof(T)>(dstTile);
    TLOAD(dstTile, dstGlobal);
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLRELU(dstTile, srcTile, scalar);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTLRELUCase1(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTLRelu<float, 32, 128, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase2(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTLRelu<half, 63, 128, 63, 63, 64, 64>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase3(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTLRelu<float, 7, 512, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase4(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTLRelu<float, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}

// Group A: FP32 (block_size=64, align_unit=8) - 8 systematic cases, col 32B aligned
extern "C" __global__ AICORE void launchTLRELUCase5(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTLRelu<float, 8, 64, 8, 8, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase6(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTLRelu<float, 8, 64, 8, 8, 48, 48>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase7(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTLRelu<float, 8, 64, 8, 8, 56, 56>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase8(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTLRelu<float, 12, 64, 12, 8, 48, 48>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase9(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTLRelu<float, 4, 96, 4, 4, 96, 96>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase10(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTLRelu<float, 4, 96, 4, 4, 72, 72>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase11(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTLRelu<float, 4, 96, 4, 4, 80, 80>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase12(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTLRelu<float, 8, 96, 8, 4, 80, 80>(out, src, scalar);
}

// Group B: FP16 (block_size=128, align_unit=16) - 8 systematic cases, col 32B aligned
extern "C" __global__ AICORE void launchTLRELUCase13(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTLRelu<half, 8, 64, 8, 8, 64, 64>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase14(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTLRelu<half, 8, 64, 8, 8, 48, 48>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase15(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTLRelu<half, 8, 64, 8, 8, 32, 32>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase16(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTLRelu<half, 12, 64, 12, 8, 48, 48>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase17(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTLRelu<half, 2, 144, 2, 2, 144, 144>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase18(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTLRelu<half, 2, 144, 2, 2, 128, 128>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase19(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTLRelu<half, 2, 144, 2, 2, 112, 112>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTLRELUCase20(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTLRelu<half, 4, 144, 4, 2, 112, 112>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}

template <uint32_t caseId>
void launchTLRELUTestCase(void *out, void *src, float scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTLRELUCase1<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 2: {
            launchTLRELUCase2<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 3: {
            launchTLRELUCase3<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 4: {
            launchTLRELUCase4<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        // Group A: FP32
        case 5: {
            launchTLRELUCase5<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 6: {
            launchTLRELUCase6<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 7: {
            launchTLRELUCase7<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 8: {
            launchTLRELUCase8<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 9: {
            launchTLRELUCase9<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 10: {
            launchTLRELUCase10<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 11: {
            launchTLRELUCase11<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 12: {
            launchTLRELUCase12<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        // Group B: FP16
        case 13: {
            launchTLRELUCase13<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 14: {
            launchTLRELUCase14<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 15: {
            launchTLRELUCase15<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 16: {
            launchTLRELUCase16<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 17: {
            launchTLRELUCase17<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 18: {
            launchTLRELUCase18<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 19: {
            launchTLRELUCase19<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 20: {
            launchTLRELUCase20<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        default: {
        }
    }
}

template void launchTLRELUTestCase<1>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<2>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<3>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<4>(void *out, void *src, float scalar, aclrtStream stream);

// Group A: FP32
template void launchTLRELUTestCase<5>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<6>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<7>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<8>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<9>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<10>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<11>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<12>(void *out, void *src, float scalar, aclrtStream stream);

// Group B: FP16
template void launchTLRELUTestCase<13>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<14>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<15>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<16>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<17>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<18>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<19>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTLRELUTestCase<20>(void *out, void *src, float scalar, aclrtStream stream);