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
#include <pto/common/debug.h>
#include "acl/acl.h"

using namespace pto;

/*
 * TTRANS ConvTile ValidShape coverage on NPU (a2a3).
 *
 * Device side facts pinned down by this file (see TTransImplConvTile in
 * include/pto/npu/a2a3/TTrans.hpp):
 *   - TTRANS derives every axis of the transform from the physical shape
 *     (GetShape), so a ValidShape configured on the src or dst tile does not
 *     change the result: dst holds the full physical transform. This matches
 *     the CPU simulator implementation (include/pto/cpu/TTrans.hpp).
 *
 * format 0: NCHW(N, C, H, W) -> NC1HWC0(N, C1, H, W, C0)
 *           phys (N, C, H, W), valid (VN, VC, VH, VW)
 * format 1: NC1HWC0(N, C1, H, W, C0) -> NCHW(N, C, H, W)
 *           phys (N, C1, H, W, C0), valid (VN, VC1, VH, VW, VC0)
 * format 2: GNCHW(G, N, C, H, W) -> GNC1HWC0(G, N, C1, H, W, C0)
 *           phys (G, N, C, H, W), valid (VG, VN, VC, VH, VW)
 *
 * NPU constraints honored here: H * W * sizeof(T) must be a multiple of 32 B and
 * the src channel dim is read as the padded C1 * C0, so C is zero padded in the
 * input data (see gen_data.py).
 */

// NCHW -> NC1HWC0, per axis valid shape
template <typename T, int N, int C, int H, int W, int VN, int VC, int VH, int VW>
__global__ AICORE void runTTRANSConvValid_NCHW2NC1HWC0(__gm__ T __out__* out, __gm__ T __in__* src)
{
    static_assert(VN >= 1 && VN <= N, "valid N must be in [1, N]");
    static_assert(VC >= 1 && VC <= C, "valid C must be in [1, C]");
    static_assert(VH >= 1 && VH <= H, "valid H must be in [1, H]");
    static_assert(VW >= 1 && VW <= W, "valid W must be in [1, W]");

    constexpr int C0 = static_cast<int>(BLOCK_BYTE_SIZE / sizeof(T));
    constexpr int C1 = (C + C0 - 1) / C0;
    constexpr int CPad = C1 * C0;
    constexpr int C1V = (VC + C0 - 1) / C0;
    constexpr int C0V = VC - (C1V - 1) * C0;
    static_assert(C1V <= C1, "valid C1 must not exceed physical C1");
    static_assert(C0V >= 1 && C0V <= C0, "valid C0 must be in [1, C0]");
    static_assert(H * W * sizeof(T) % BLOCK_BYTE_SIZE == 0, "expect align for H * W");

    constexpr int elemNum = N * C1 * H * W * C0;

    using ShapeDim5 = Shape<1, 1, 1, 1, elemNum>;
    using StrideDim5 = pto::Stride<elemNum, elemNum, elemNum, elemNum, 1>;
    using GlobalData = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using FlatTileData = Tile<TileType::Vec, T, 1, elemNum, BLayout::RowMajor, 1, elemNum>;
    FlatTileData src0Tile;
    TASSIGN(src0Tile, 0x0);

    using SrcConvTile = ConvTile<
        TileType::Vec, T, elemNum, Layout::NCHW, ConvTileShape<N, CPad, H, W>, ConvTileValidShape<VN, VC, VH, VW>>;
    SrcConvTile srcTile;
    static_assert(srcTile.totalDimCount == 4);
    TASSIGN(srcTile, 0x0);

    using DstConvTile = ConvTile<
        TileType::Vec, T, elemNum, Layout::NC1HWC0, ConvTileShape<N, C1, H, W, C0>,
        ConvTileValidShape<VN, C1V, VH, VW, C0V>>;
    DstConvTile dstTile;
    static_assert(dstTile.totalDimCount == 5);
    TASSIGN(dstTile, 0x0 + elemNum * sizeof(T));

    FlatTileData dst0Tile;
    TASSIGN(dst0Tile, 0x0 + elemNum * sizeof(T));

    constexpr int tmpTileH = H * W;
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? 32 : 16;
    constexpr int tmpTileW = (C0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    using TmpTileData = Tile<TileType::Vec, T, tmpTileH, tmpTileW, BLayout::RowMajor, tmpTileH, tmpTileW>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + elemNum * sizeof(T) * 2);

    // TTRANS writes the full physical dst (valid shape is ignored), so the
    // pre-zeroing below has no effect on the result and only guards the tail in
    // case a future device revision honors the valid shape.
    __ubuf__ T* dstPtr = dstTile.data();
    for (int i = 0; i < elemNum; i++) {
        dstPtr[i] = static_cast<T>(0);
    }
    pipe_barrier(PIPE_ALL);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    TTRANS(dstTile, srcTile, tmpTile);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dst0Tile);
}

// NC1HWC0 -> NCHW, per axis valid shape
template <typename T, int N, int C1, int H, int W, int C0, int VN, int VC1, int VH, int VW, int VC0>
__global__ AICORE void runTTRANSConvValid_NC1HWC02NCHW(__gm__ T __out__* out, __gm__ T __in__* src)
{
    static_assert(VN >= 1 && VN <= N, "valid N must be in [1, N]");
    static_assert(VC1 >= 1 && VC1 <= C1, "valid C1 must be in [1, C1]");
    static_assert(VH >= 1 && VH <= H, "valid H must be in [1, H]");
    static_assert(VW >= 1 && VW <= W, "valid W must be in [1, W]");
    static_assert(VC0 >= 1 && VC0 <= C0, "valid C0 must be in [1, C0]");
    static_assert(C0 == static_cast<int>(BLOCK_BYTE_SIZE / sizeof(T)), "C0 must match the dtype block size");
    static_assert(H * W * sizeof(T) % BLOCK_BYTE_SIZE == 0, "expect align for H * W");

    constexpr int C = C1 * C0;
    constexpr int VC = (VC1 - 1) * C0 + VC0;
    constexpr int elemNum = N * C1 * H * W * C0;

    using ShapeDim5 = Shape<1, 1, 1, 1, elemNum>;
    using StrideDim5 = pto::Stride<elemNum, elemNum, elemNum, elemNum, 1>;
    using GlobalData = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using FlatTileData = Tile<TileType::Vec, T, 1, elemNum, BLayout::RowMajor, 1, elemNum>;
    FlatTileData src0Tile;
    TASSIGN(src0Tile, 0x0);

    using SrcConvTile = ConvTile<
        TileType::Vec, T, elemNum, Layout::NC1HWC0, ConvTileShape<N, C1, H, W, C0>,
        ConvTileValidShape<VN, VC1, VH, VW, VC0>>;
    SrcConvTile srcTile;
    static_assert(srcTile.totalDimCount == 5);
    TASSIGN(srcTile, 0x0);

    using DstConvTile = ConvTile<
        TileType::Vec, T, elemNum, Layout::NCHW, ConvTileShape<N, C, H, W>, ConvTileValidShape<VN, VC, VH, VW>>;
    DstConvTile dstTile;
    static_assert(dstTile.totalDimCount == 4);
    TASSIGN(dstTile, 0x0 + elemNum * sizeof(T));

    FlatTileData dst0Tile;
    TASSIGN(dst0Tile, 0x0 + elemNum * sizeof(T));

    constexpr int tmpTileH = H * W;
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? 32 : 16;
    constexpr int tmpTileW = (C0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    using TmpTileData = Tile<TileType::Vec, T, tmpTileH, tmpTileW, BLayout::RowMajor, tmpTileH, tmpTileW>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + elemNum * sizeof(T) * 2);

    // TTRANS writes the full physical dst (valid shape is ignored), so the
    // pre-zeroing below has no effect on the result and only guards the tail in
    // case a future device revision honors the valid shape.
    __ubuf__ T* dstPtr = dstTile.data();
    for (int i = 0; i < elemNum; i++) {
        dstPtr[i] = static_cast<T>(0);
    }
    pipe_barrier(PIPE_ALL);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    TTRANS(dstTile, srcTile, tmpTile);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dst0Tile);
}

// GNCHW -> GNC1HWC0, per axis valid shape (valid is ignored by this path on NPU)
template <typename T, int G, int N, int C, int H, int W, int VG, int VN, int VC, int VH, int VW>
__global__ AICORE void runTTRANSConvValid_GNCHW2GNC1HWC0(__gm__ T __out__* out, __gm__ T __in__* src)
{
    static_assert(VG >= 1 && VG <= G, "valid G must be in [1, G]");
    static_assert(VN >= 1 && VN <= N, "valid N must be in [1, N]");
    static_assert(VC >= 1 && VC <= C, "valid C must be in [1, C]");
    static_assert(VH >= 1 && VH <= H, "valid H must be in [1, H]");
    static_assert(VW >= 1 && VW <= W, "valid W must be in [1, W]");

    constexpr int C0 = static_cast<int>(BLOCK_BYTE_SIZE / sizeof(T));
    constexpr int C1 = (C + C0 - 1) / C0;
    constexpr int CPad = C1 * C0;
    constexpr int C1V = (VC + C0 - 1) / C0;
    constexpr int C0V = VC - (C1V - 1) * C0;
    static_assert(C1V <= C1, "valid C1 must not exceed physical C1");
    static_assert(C0V >= 1 && C0V <= C0, "valid C0 must be in [1, C0]");
    static_assert(H * W * sizeof(T) % BLOCK_BYTE_SIZE == 0, "expect align for H * W");

    constexpr int elemNum = G * N * C1 * H * W * C0;

    using ShapeDim5 = Shape<1, 1, 1, 1, elemNum>;
    using StrideDim5 = pto::Stride<elemNum, elemNum, elemNum, elemNum, 1>;
    using GlobalData = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using FlatTileData = Tile<TileType::Vec, T, 1, elemNum, BLayout::RowMajor, 1, elemNum>;
    FlatTileData src0Tile;
    TASSIGN(src0Tile, 0x0);

    using SrcConvTile = ConvTile<
        TileType::Vec, T, elemNum, Layout::GNCHW, ConvTileShape<G, N, CPad, H, W>,
        ConvTileValidShape<VG, VN, VC, VH, VW>>;
    SrcConvTile srcTile;
    static_assert(srcTile.totalDimCount == 5);
    TASSIGN(srcTile, 0x0);

    using DstConvTile = ConvTile<
        TileType::Vec, T, elemNum, Layout::GNC1HWC0, ConvTileShape<G, N, C1, H, W, C0>,
        ConvTileValidShape<VG, VN, C1V, VH, VW, C0V>>;
    DstConvTile dstTile;
    static_assert(dstTile.totalDimCount == 6);
    TASSIGN(dstTile, 0x0 + elemNum * sizeof(T));

    FlatTileData dst0Tile;
    TASSIGN(dst0Tile, 0x0 + elemNum * sizeof(T));

    constexpr int tmpTileH = H * W;
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? 32 : 16;
    constexpr int tmpTileW = (C0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    using TmpTileData = Tile<TileType::Vec, T, tmpTileH, tmpTileW, BLayout::RowMajor, tmpTileH, tmpTileW>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + elemNum * sizeof(T) * 2);

    __ubuf__ T* dstPtr = dstTile.data();
    for (int i = 0; i < elemNum; i++) {
        dstPtr[i] = static_cast<T>(0);
    }
    pipe_barrier(PIPE_ALL);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    TTRANS(dstTile, srcTile, tmpTile);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dst0Tile);
}

/*
 * ConvTile ValidShape API probe: dumps GetShape / GetValidShape of static,
 * default and dynamic ConvTiles to GM so the host side can assert them.
 * The dump order must stay in sync with case_api_validshape_probe in main.cpp.
 */
__global__ AICORE void runTTRANSConvValidShapeApi(__gm__ int32_t __out__* out)
{
    int idx = 0;

    // A: NCHW, static valid on every axis
    {
        using CT = ConvTile<
            TileType::Vec, half, 8 * 20 * 7 * 9, Layout::NCHW, ConvTileShape<8, 20, 7, 9>,
            ConvTileValidShape<3, 7, 5, 4>>;
        CT tile;
        for (int d = 0; d < CT::totalDimCount; d++) {
            out[idx++] = static_cast<int32_t>(tile.GetShape(d));
            out[idx++] = static_cast<int32_t>(tile.GetValidShape(d));
        }
    }

    // B: NC1HWC0, static valid on every axis
    {
        using CT = ConvTile<
            TileType::Vec, half, 8 * 2 * 7 * 9 * 16, Layout::NC1HWC0, ConvTileShape<8, 2, 7, 9, 16>,
            ConvTileValidShape<3, 1, 5, 4, 7>>;
        CT tile;
        for (int d = 0; d < CT::totalDimCount; d++) {
            out[idx++] = static_cast<int32_t>(tile.GetShape(d));
            out[idx++] = static_cast<int32_t>(tile.GetValidShape(d));
        }
    }

    // C: FRACTAL_Z, static valid on every axis
    {
        using CT = ConvTile<
            TileType::Vec, half, 126 * 4 * 2 * 16, Layout::FRACTAL_Z, ConvTileShape<126, 4, 2, 16>,
            ConvTileValidShape<63, 3, 2, 16>>;
        CT tile;
        for (int d = 0; d < CT::totalDimCount; d++) {
            out[idx++] = static_cast<int32_t>(tile.GetShape(d));
            out[idx++] = static_cast<int32_t>(tile.GetValidShape(d));
        }
    }

    // D: GNCHW, static valid on every axis
    {
        using CT = ConvTile<
            TileType::Vec, half, 2 * 4 * 20 * 7 * 9, Layout::GNCHW, ConvTileShape<2, 4, 20, 7, 9>,
            ConvTileValidShape<1, 3, 7, 5, 4>>;
        CT tile;
        for (int d = 0; d < CT::totalDimCount; d++) {
            out[idx++] = static_cast<int32_t>(tile.GetShape(d));
            out[idx++] = static_cast<int32_t>(tile.GetValidShape(d));
        }
    }

    // E: GNC1HWC0 (6D), static valid on every axis
    {
        using CT = ConvTile<
            TileType::Vec, half, 2 * 4 * 2 * 7 * 9 * 16, Layout::GNC1HWC0, ConvTileShape<2, 4, 2, 7, 9, 16>,
            ConvTileValidShape<1, 3, 1, 5, 4, 7>>;
        CT tile;
        for (int d = 0; d < CT::totalDimCount; d++) {
            out[idx++] = static_cast<int32_t>(tile.GetShape(d));
            out[idx++] = static_cast<int32_t>(tile.GetValidShape(d));
        }
    }

    // F: NCHW without an explicit valid shape -> valid defaults to physical
    {
        using CT = ConvTile<TileType::Vec, half, 8 * 20 * 7 * 9, Layout::NCHW, ConvTileShape<8, 20, 7, 9>>;
        CT tile;
        for (int d = 0; d < CT::totalDimCount; d++) {
            out[idx++] = static_cast<int32_t>(tile.GetShape(d));
            out[idx++] = static_cast<int32_t>(tile.GetValidShape(d));
        }
    }

    // G: dynamic physical + dynamic valid axes
    {
        using CT = ConvTile<
            TileType::Vec, half, 16 * 4 * 8 * 8, Layout::NCHW, ConvTileShape<DYNAMIC, 4, DYNAMIC, 8>,
            ConvTileValidShape<DYNAMIC, 4, DYNAMIC, 8>>;
        CT tile(int64_t(16), int64_t(8));
        // SetDynamicShape resets every dynamic valid axis to its physical value
        out[idx++] = static_cast<int32_t>(tile.GetShape(0));
        out[idx++] = static_cast<int32_t>(tile.GetShape(2));
        out[idx++] = static_cast<int32_t>(tile.GetValidShape(0));
        out[idx++] = static_cast<int32_t>(tile.GetValidShape(2));

        tile.SetAllValidShape(int64_t(10), int64_t(5));
        out[idx++] = static_cast<int32_t>(tile.GetValidShape(0));
        out[idx++] = static_cast<int32_t>(tile.GetValidShape(1));
        out[idx++] = static_cast<int32_t>(tile.GetValidShape(2));
        out[idx++] = static_cast<int32_t>(tile.GetValidShape(3));

        tile.SetValidShape(0, int64_t(8));
        out[idx++] = static_cast<int32_t>(tile.GetValidShape(0));

        tile.SetDynamicShape(int64_t(12), int64_t(6));
        out[idx++] = static_cast<int32_t>(tile.GetShape(0));
        out[idx++] = static_cast<int32_t>(tile.GetShape(2));
        out[idx++] = static_cast<int32_t>(tile.GetValidShape(0));
        out[idx++] = static_cast<int32_t>(tile.GetValidShape(2));
    }

    pipe_barrier(PIPE_ALL);
}

void LaunchTTRANSConvValidShapeApi(int32_t* out, void* stream)
{
    runTTRANSConvValidShapeApi<<<1, nullptr, stream>>>(out);
}

/*
 * format 0: p = (N, C, H, W, -),     v = (VN, VC, VH, VW, -)
 * format 1: p = (N, C1, H, W, C0),   v = (VN, VC1, VH, VW, VC0)
 * format 2: p = (G, N, C, H, W),     v = (VG, VN, VC, VH, VW)
 */
template <
    typename T, int format, int p0, int p1, int p2, int p3, int p4, int v0, int v1, int v2, int v3, int v4>
void LaunchTTRANSConvValidShape(T* out, T* src, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        if constexpr (format == 0) {
            runTTRANSConvValid_NCHW2NC1HWC0<half, p0, p1, p2, p3, v0, v1, v2, v3><<<1, nullptr, stream>>>(
                (half*)(out), (half*)(src));
        } else if constexpr (format == 1) {
            runTTRANSConvValid_NC1HWC02NCHW<half, p0, p1, p2, p3, p4, v0, v1, v2, v3, v4><<<1, nullptr, stream>>>(
                (half*)(out), (half*)(src));
        } else if constexpr (format == 2) {
            runTTRANSConvValid_GNCHW2GNC1HWC0<half, p0, p1, p2, p3, p4, v0, v1, v2, v3, v4><<<1, nullptr, stream>>>(
                (half*)(out), (half*)(src));
        }
    } else {
        if constexpr (format == 0) {
            runTTRANSConvValid_NCHW2NC1HWC0<T, p0, p1, p2, p3, v0, v1, v2, v3><<<1, nullptr, stream>>>(out, src);
        } else if constexpr (format == 1) {
            runTTRANSConvValid_NC1HWC02NCHW<T, p0, p1, p2, p3, p4, v0, v1, v2, v3, v4><<<1, nullptr, stream>>>(
                out, src);
        } else if constexpr (format == 2) {
            runTTRANSConvValid_GNCHW2GNC1HWC0<T, p0, p1, p2, p3, p4, v0, v1, v2, v3, v4><<<1, nullptr, stream>>>(
                out, src);
        } else {
            static_assert(format == 0 || format == 1 || format == 2, "unsupported TTRANS validshape format");
        }
    }
}

/*-------------------- NCHW -> NC1HWC0 --------------------*/

// N axis unequal: VN = 2 (N = 4), other axes keep physical -> tail batches stay zero
template void LaunchTTRANSConvValidShape<aclFloat16, 0, 4, 20, 4, 16, 1, 2, 20, 4, 16, 1>(
    aclFloat16* out, aclFloat16* src, void* stream);
// C axis unequal + unaligned: VC = 7 (C = 20, C0 = 16) -> C1V = 1, C0V = 7
template void LaunchTTRANSConvValidShape<aclFloat16, 0, 4, 20, 4, 16, 1, 4, 7, 4, 16, 1>(
    aclFloat16* out, aclFloat16* src, void* stream);
// H axis unequal: VH = 5 (H = 8)
template void LaunchTTRANSConvValidShape<aclFloat16, 0, 4, 16, 8, 16, 1, 4, 16, 5, 16, 1>(
    aclFloat16* out, aclFloat16* src, void* stream);
// W axis unequal: VW = 4 (W = 16)
template void LaunchTTRANSConvValidShape<aclFloat16, 0, 4, 16, 4, 16, 1, 4, 16, 4, 4, 1>(
    aclFloat16* out, aclFloat16* src, void* stream);
// every axis unequal + unaligned at once
template void LaunchTTRANSConvValidShape<aclFloat16, 0, 4, 20, 8, 16, 1, 2, 7, 5, 4, 1>(
    aclFloat16* out, aclFloat16* src, void* stream);
// float32 (C0 = 8): C = 20 -> C1 = 3, VC = 7 -> C1V = 1, C0V = 7
template void LaunchTTRANSConvValidShape<float, 0, 4, 20, 4, 16, 1, 2, 7, 3, 4, 1>(
    float* out, float* src, void* stream);
// boundary: valid == physical on every axis
template void LaunchTTRANSConvValidShape<aclFloat16, 0, 4, 20, 4, 16, 1, 4, 20, 4, 16, 1>(
    aclFloat16* out, aclFloat16* src, void* stream);
// boundary: valid == 1 on every axis
template void LaunchTTRANSConvValidShape<aclFloat16, 0, 4, 20, 4, 16, 1, 1, 1, 1, 1, 1>(
    aclFloat16* out, aclFloat16* src, void* stream);

/*-------------------- NC1HWC0 -> NCHW --------------------*/

// N axis unequal: VN = 3 (N = 4) -> tail batch stays zero
template void LaunchTTRANSConvValidShape<aclFloat16, 1, 4, 2, 4, 16, 16, 3, 2, 4, 16, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
// C1 axis unequal: VC1 = 1 (C1 = 2) -> VC = 16
template void LaunchTTRANSConvValidShape<aclFloat16, 1, 4, 2, 4, 16, 16, 4, 1, 4, 16, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
// C0 axis unequal: VC0 = 7 (C0 = 16)
template void LaunchTTRANSConvValidShape<aclFloat16, 1, 4, 2, 4, 16, 16, 4, 2, 4, 16, 7>(
    aclFloat16* out, aclFloat16* src, void* stream);
// H axis unequal: VH = 5 (H = 8)
template void LaunchTTRANSConvValidShape<aclFloat16, 1, 4, 2, 8, 16, 16, 4, 2, 5, 16, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
// W axis unequal: VW = 4 (W = 16)
template void LaunchTTRANSConvValidShape<aclFloat16, 1, 4, 2, 4, 16, 16, 4, 2, 4, 4, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
// every axis unequal + unaligned at once
template void LaunchTTRANSConvValidShape<aclFloat16, 1, 4, 2, 8, 16, 16, 2, 1, 5, 4, 7>(
    aclFloat16* out, aclFloat16* src, void* stream);
// int32 (C0 = 8): every axis unequal + unaligned
template void LaunchTTRANSConvValidShape<int32_t, 1, 4, 2, 4, 8, 8, 2, 1, 3, 4, 5>(
    int32_t* out, int32_t* src, void* stream);
// boundary: valid == physical on every axis
template void LaunchTTRANSConvValidShape<aclFloat16, 1, 4, 2, 4, 16, 16, 4, 2, 4, 16, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);

/*-------------------- GNCHW -> GNC1HWC0 --------------------*/

// G axis unequal: VG = 1 (G = 2)
template void LaunchTTRANSConvValidShape<aclFloat16, 2, 2, 2, 20, 4, 16, 1, 2, 20, 4, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
// every axis unequal + unaligned at once
template void LaunchTTRANSConvValidShape<aclFloat16, 2, 2, 2, 20, 4, 16, 1, 1, 7, 3, 4>(
    aclFloat16* out, aclFloat16* src, void* stream);
