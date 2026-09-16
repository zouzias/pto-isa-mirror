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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

// TMOV ND→NZ kernel for hifloat8_t.
// Sizes must be pre-aligned: kRows % 16 == 0 and kCols % 32 == 0.
template <int kRows, int kCols>
__global__ AICORE void runTMOV_nd2nz(__gm__ hifloat8_t __out__* out, __gm__ hifloat8_t __in__* src)
{
    using T = hifloat8_t;
    constexpr int c0 = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T)); // 32

    // Input GM: ND row-major
    using SrcShape = Shape<1, 1, 1, kRows, kCols>;
    using SrcStride = pto::Stride<1, 1, 1, kCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    // Output GM: NZ fractal [C1, N1, 16, c0]
    constexpr int C1 = kCols / c0;
    constexpr int N1 = kRows / FRACTAL_NZ_ROW;
    using DstShape = Shape<1, C1, N1, FRACTAL_NZ_ROW, c0>;
    using DstStride = pto::Stride<C1 * kRows * c0, kRows * c0, FRACTAL_NZ_ROW * c0, c0, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride, Layout::NZ>;

    // UB tiles: src ND, dst NZ
    using SrcTile = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T, kRows, kCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcTile srcTile(kRows, kCols);
    DstTile dstTile(kRows, kCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out);

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TMOV<DstTile, SrcTile>(dstTile, srcTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

// TMOV ND→NZ kernel for float4_e1m2x2_t (2×f4e1m2 packed per byte).
// kRows/kCols are in f4 element (nibble) units. kCols is padded up to the
// 64-element (32-byte) NZ panel width: the output GM always holds full panels,
// and the per-row column pad is zero-filled so the last panel is deterministic.
template <int kRows, int kCols>
__global__ AICORE void runTMOV_nd2nz_f4(__gm__ float4_e1m2x2_t __out__* out, __gm__ float4_e1m2x2_t __in__* src)
{
    using T = float4_e1m2x2_t;
    constexpr int c0 = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T)) * 2; // 64 nibbles = 32 bytes
    constexpr int alignedCols = (kCols + c0 - 1) / c0 * c0;
    constexpr int C1 = alignedCols / c0;
    constexpr int N1 = kRows / FRACTAL_NZ_ROW;

    // Input GM: ND row-major [kRows, alignedCols] nibbles. The valid data is
    // kCols wide; the trailing pad up to the NZ panel width is zero-filled in
    // GM (NZ always stores full panels, so the feed is pre-padded).
    using SrcShape = Shape<1, 1, 1, kRows, alignedCols>;
    using SrcStride = pto::Stride<1, 1, 1, alignedCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    // Output GM: NZ fractal [C1, N1, 16, c0] over the padded width
    using DstShape = Shape<1, C1, N1, FRACTAL_NZ_ROW, c0>;
    using DstStride = pto::Stride<C1 * kRows * c0, kRows * c0, FRACTAL_NZ_ROW * c0, c0, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride, Layout::NZ>;

    // UB tiles sized to the padded width
    using SrcTile = Tile<TileType::Vec, T, kRows, alignedCols, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T, kRows, alignedCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcTile srcTile(kRows, alignedCols);
    DstTile dstTile(kRows, alignedCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out);

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TMOV<DstTile, SrcTile>(dstTile, srcTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

// Host-visible launch wrappers (uint8_t* since hifloat8_t is opaque on host)
template <int kRows, int kCols>
void launchTMOV_nd2nz(uint8_t* out, uint8_t* src, void* stream)
{
    runTMOV_nd2nz<kRows, kCols><<<1, nullptr, stream>>>((hifloat8_t*)out, (hifloat8_t*)src);
}

template <int kRows, int kCols>
void launchTMOV_nd2nz_f4(uint8_t* out, uint8_t* src, void* stream)
{
    runTMOV_nd2nz_f4<kRows, kCols><<<1, nullptr, stream>>>((float4_e1m2x2_t*)out, (float4_e1m2x2_t*)src);
}

template void launchTMOV_nd2nz<32, 32>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz<32, 64>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz<64, 64>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz_f4<16, 8160>(uint8_t*, uint8_t*, void*);
