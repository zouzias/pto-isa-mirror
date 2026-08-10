/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

// TMOV ND->ZN kernel. ND [K,N] row-major -> ZN [K/K0, N/16, 16, K0] where
// K0 = 32B/sizeof(T) (16 for B16, 8 for B32). This is the Right/NT cube operand layout
// (BLayout::RowMajor + SLayout::ColMajor) — the within-fractal transpose of ND->NZ.
// Sizes must be pre-aligned: K % K0 == 0 and N % 16 == 0.
//
// The within-fractal transpose writes a continuous byte sequence into UB (fractal 0 at
// offset 0, fractal 1 at offset 512B, ...). TSTORE only accepts ND/DN/NZ tile signatures,
// not the ZN signature, but the bytes are layout-agnostic once in UB: we alias the dst
// region with a plain ND-view tile ([K,N] RowMajor+NoneBox) and store it flat into a
// matching ND-view GM. The GM bytes are then exactly the ZN byte sequence the golden
// (gen_data.py nd_to_zn) computes.
template <typename T, int kRows, int kCols>
__global__ AICORE void runTMOV_nd2zn(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr int k0 = BLOCK_BYTE_SIZE / sizeof(T); // 16 for B16, 8 for B32

    // Input GM: ND row-major
    using SrcShape = Shape<1, 1, 1, kRows, kCols>;
    using SrcStride = pto::Stride<1, 1, 1, kCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    // Output GM: ND [K,N] row-major (a flat view of the ZN byte sequence).
    using DstShape = Shape<1, 1, 1, kRows, kCols>;
    using DstStride = pto::Stride<1, 1, 1, kCols, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;

    // UB tiles — two views of the dst region:
    //   ZnTile (RowMajor + ColMajor): drives TMOV, triggers the ND->ZN transpose path.
    //   NdTile (RowMajor + NoneBox) : drives TSTORE as a flat ND copy; same bytes.
    using SrcTile = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1>;
    using ZnTile = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1, SLayout::ColMajor>;
    using NdTile = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(kRows, kCols);
    ZnTile znTile(kRows, kCols);
    NdTile ndTile(kRows, kCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(znTile, 0x10000); // dst region for the transpose
    TASSIGN(ndTile, 0x10000); // same UB offset — alias of znTile

    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out);

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    // ND -> ZN within-fractal transpose, writes the continuous ZN byte sequence into UB.
    TMOV<ZnTile, SrcTile>(znTile, srcTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    // Flat ND copy of the (identical) bytes out to GM.
    TSTORE(dstGlobal, ndTile);
    out = dstGlobal.data();
}

// Host-visible launch wrappers. hifloat8_t is opaque on host, so the B8 wrapper uses
// uint8_t* (matching the tmov_nd2nz convention); half uses uint16_t*; int32/float use uint32_t*.
template <int kRows, int kCols>
void launchTMOV_nd2zn_hif8(uint8_t* out, uint8_t* src, void* stream)
{
    runTMOV_nd2zn<hifloat8_t, kRows, kCols><<<1, nullptr, stream>>>((hifloat8_t*)out, (hifloat8_t*)src);
}

template <int kRows, int kCols>
void launchTMOV_nd2zn_half(uint16_t* out, uint16_t* src, void* stream)
{
    runTMOV_nd2zn<half, kRows, kCols><<<1, nullptr, stream>>>((half*)out, (half*)src);
}

template <int kRows, int kCols>
void launchTMOV_nd2zn_b32(uint32_t* out, uint32_t* src, void* stream)
{
    runTMOV_nd2zn<int32_t, kRows, kCols><<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
}

template void launchTMOV_nd2zn_hif8<32, 32>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2zn_hif8<32, 64>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2zn_hif8<64, 64>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2zn_hif8<128, 128>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2zn_half<32, 32>(uint16_t*, uint16_t*, void*);
template void launchTMOV_nd2zn_half<32, 64>(uint16_t*, uint16_t*, void*);
template void launchTMOV_nd2zn_half<64, 64>(uint16_t*, uint16_t*, void*);
template void launchTMOV_nd2zn_half<128, 128>(uint16_t*, uint16_t*, void*);
template void launchTMOV_nd2zn_b32<32, 32>(uint32_t*, uint32_t*, void*);
template void launchTMOV_nd2zn_b32<32, 64>(uint32_t*, uint32_t*, void*);
template void launchTMOV_nd2zn_b32<64, 64>(uint32_t*, uint32_t*, void*);
template void launchTMOV_nd2zn_b32<128, 128>(uint32_t*, uint32_t*, void*);
