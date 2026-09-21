/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdlib>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/debug.h>
#include <pto/common/pto_tile.hpp>

using namespace pto;

/*
 * TTRANS with a per-axis ConvTile ValidShape, so that every axis can be configured
 * independently as "valid < physical" (unequal) and/or not aligned to C0 (unaligned).
 *
 * format 0: NCHW(N, C, H, W) -> NC1HWC0(N, C1, H, W, C0)
 *   src physical: (N, C, H, W)        src valid: (VN, VC, VH, VW)
 *   dst physical: (N, C1, H, W, C0)   dst valid: (VN, C1V, VH, VW, C0V)
 *   C0 = 32 / DTypeSize, C1 = ceil(C / C0), C1V = ceil(VC / C0), C0V = VC - (C1V - 1) * C0
 *
 * format 1: NC1HWC0(N, C1, H, W, C0) -> NCHW(N, C, H, W)
 *   src physical: (N, C1, H, W, C0)   src valid: (VN, VC1, VH, VW, VC0)
 *   dst physical: (N, C, H, W)        dst valid: (VN, VC, VH, VW)
 *   C = C1 * C0, VC = (VC1 - 1) * C0 + VC0
 *
 * format 2: GNCHW(G, N, C, H, W) -> GNC1HWC0(G, N, C1, H, W, C0)
 *   src physical: (G, N, C, H, W)        src valid: (VG, VN, VC, VH, VW)
 *   dst physical: (G, N, C1, H, W, C0)   dst valid: (VG, VN, C1V, VH, VW, C0V)
 */

template <typename T, int N, int C, int H, int W, int VN, int VC, int VH, int VW>
__global__ AICORE void runTTRANSConv_NCHW2NC1HWC0_AxisValid(__gm__ T __out__* out, __gm__ T __in__* src)
{
    static_assert(VN >= 1 && VN <= N, "valid N must be in [1, N]");
    static_assert(VC >= 1 && VC <= C, "valid C must be in [1, C]");
    static_assert(VH >= 1 && VH <= H, "valid H must be in [1, H]");
    static_assert(VW >= 1 && VW <= W, "valid W must be in [1, W]");

    constexpr size_t DTypeSize = GetTypeSize<T>();
    constexpr size_t C0 = 32 / DTypeSize;
    constexpr size_t C1 = (C + C0 - 1) / C0;
    constexpr size_t C1V = (VC + C0 - 1) / C0;
    constexpr size_t C0V = VC - (C1V - 1) * C0;
    static_assert(C1V <= C1, "valid C1 must not exceed physical C1");
    static_assert(C0V >= 1 && C0V <= C0, "valid C0 must be in [1, C0]");

    constexpr size_t srcElemNum = N * C * H * W;
    constexpr size_t dstElemNum = N * C1 * H * W * C0;

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + srcElemNum * DTypeSize);

    using SrcConvTile = ConvTile<
        TileType::Vec, T, srcElemNum * DTypeSize, Layout::NCHW, ConvTileShape<N, C, H, W>,
        ConvTileValidShape<VN, VC, VH, VW>>;
    using DstConvTile = ConvTile<
        TileType::Vec, T, dstElemNum * DTypeSize, Layout::NC1HWC0, ConvTileShape<N, C1, H, W, C0>,
        ConvTileValidShape<VN, C1V, VH, VW, C0V>>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(srcTile.totalDimCount == 4);
    static_assert(dstTile.totalDimCount == 5);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    std::fill((uint8_t*)dstTile.data(), (uint8_t*)dstTile.data() + dstElemNum * DTypeSize, 0);

    // Not used internally, just for placeholder
    using TmpTileData = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, 1, 32>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + (srcElemNum + dstElemNum) * DTypeSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

template <typename T, int N, int C1, int H, int W, int C0, int VN, int VC1, int VH, int VW, int VC0>
__global__ AICORE void runTTRANSConv_NC1HWC02NCHW_AxisValid(__gm__ T __out__* out, __gm__ T __in__* src)
{
    static_assert(C0 == 32 / GetTypeSize<T>(), "C0 must match the dtype block size");
    static_assert(VN >= 1 && VN <= N, "valid N must be in [1, N]");
    static_assert(VC1 >= 1 && VC1 <= C1, "valid C1 must be in [1, C1]");
    static_assert(VH >= 1 && VH <= H, "valid H must be in [1, H]");
    static_assert(VW >= 1 && VW <= W, "valid W must be in [1, W]");
    static_assert(VC0 >= 1 && VC0 <= C0, "valid C0 must be in [1, C0]");

    constexpr size_t DTypeSize = GetTypeSize<T>();
    constexpr size_t C = C1 * C0;
    constexpr size_t VC = (VC1 - 1) * C0 + VC0;
    static_assert(VC >= 1 && VC <= C, "derived valid C must be in [1, C]");

    constexpr size_t srcElemNum = N * C1 * H * W * C0;
    constexpr size_t dstElemNum = N * C * H * W;

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + srcElemNum * DTypeSize);

    using SrcConvTile = ConvTile<
        TileType::Vec, T, srcElemNum * DTypeSize, Layout::NC1HWC0, ConvTileShape<N, C1, H, W, C0>,
        ConvTileValidShape<VN, VC1, VH, VW, VC0>>;
    using DstConvTile = ConvTile<
        TileType::Vec, T, dstElemNum * DTypeSize, Layout::NCHW, ConvTileShape<N, C, H, W>,
        ConvTileValidShape<VN, VC, VH, VW>>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(srcTile.totalDimCount == 5);
    static_assert(dstTile.totalDimCount == 4);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    std::fill((uint8_t*)dstTile.data(), (uint8_t*)dstTile.data() + dstElemNum * DTypeSize, 0);

    // Not used internally, just for placeholder
    using TmpTileData = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, 1, 32>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + (srcElemNum + dstElemNum) * DTypeSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

template <typename T, int G, int N, int C, int H, int W, int VG, int VN, int VC, int VH, int VW>
__global__ AICORE void runTTRANSConv_GNCHW2GNC1HWC0_AxisValid(__gm__ T __out__* out, __gm__ T __in__* src)
{
    static_assert(VG >= 1 && VG <= G, "valid G must be in [1, G]");
    static_assert(VN >= 1 && VN <= N, "valid N must be in [1, N]");
    static_assert(VC >= 1 && VC <= C, "valid C must be in [1, C]");
    static_assert(VH >= 1 && VH <= H, "valid H must be in [1, H]");
    static_assert(VW >= 1 && VW <= W, "valid W must be in [1, W]");

    constexpr size_t DTypeSize = GetTypeSize<T>();
    constexpr size_t C0 = (32 / DTypeSize) * (IsTwinType<T>() ? 2 : 1);
    constexpr size_t C1 = (C + C0 - 1) / C0;
    constexpr size_t C1V = (VC + C0 - 1) / C0;
    constexpr size_t C0V = VC - (C1V - 1) * C0;
    static_assert(C1V <= C1, "valid C1 must not exceed physical C1");
    static_assert(C0V >= 1 && C0V <= C0, "valid C0 must be in [1, C0]");

    constexpr size_t srcElemNum = G * N * C * H * W;
    constexpr size_t dstElemNum = G * N * C1 * H * W * C0;

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + srcElemNum * DTypeSize);

    using SrcConvTile = ConvTile<
        TileType::Vec, T, srcElemNum * DTypeSize, Layout::GNCHW, ConvTileShape<G, N, C, H, W>,
        ConvTileValidShape<VG, VN, VC, VH, VW>>;
    using DstConvTile = ConvTile<
        TileType::Vec, T, dstElemNum * DTypeSize, Layout::GNC1HWC0, ConvTileShape<G, N, C1, H, W, C0>,
        ConvTileValidShape<VG, VN, C1V, VH, VW, C0V>>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(srcTile.totalDimCount == 5);
    static_assert(dstTile.totalDimCount == 6);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    std::fill((uint8_t*)dstTile.data(), (uint8_t*)dstTile.data() + dstElemNum * DTypeSize, 0);

    // Not used internally, just for placeholder
    using TmpTileData = Tile<TileType::Vec, T, 1, 64, BLayout::RowMajor, 1, 64>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + (srcElemNum + dstElemNum) * DTypeSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

/*
 * format 0: p = (N, C, H, W, -), v = (VN, VC, VH, VW, -)
 * format 1: p = (N, C1, H, W, C0), v = (VN, VC1, VH, VW, VC0)
 * format 2: p = (G, N, C, H, W), v = (VG, VN, VC, VH, VW)
 */
template <
    typename T, int format, int p0, int p1, int p2, int p3, int p4, int v0, int v1, int v2, int v3, int v4>
void LaunchTTRANSConvAxisValid(T* out, T* src, void* stream)
{
    if constexpr (format == 0) {
        runTTRANSConv_NCHW2NC1HWC0_AxisValid<T, p0, p1, p2, p3, v0, v1, v2, v3>(out, src);
    } else if constexpr (format == 1) {
        runTTRANSConv_NC1HWC02NCHW_AxisValid<T, p0, p1, p2, p3, p4, v0, v1, v2, v3, v4>(out, src);
    } else if constexpr (format == 2) {
        runTTRANSConv_GNCHW2GNC1HWC0_AxisValid<T, p0, p1, p2, p3, p4, v0, v1, v2, v3, v4>(out, src);
    } else {
        static_assert(format == 0 || format == 1 || format == 2, "unsupported TTRANS validshape format");
    }
}

/*-------------------- NCHW -> NC1HWC0 --------------------*/

// N axis unequal + unaligned: VN = 3 (N = 8)
template void LaunchTTRANSConvAxisValid<half, 0, 8, 8, 4, 4, 1, 3, 8, 4, 4, 1>(half* out, half* src, void* stream);
// C axis unequal + unaligned to C0: VC = 7 (C = 20, C0 = 16)
template void LaunchTTRANSConvAxisValid<half, 0, 4, 20, 4, 4, 1, 4, 7, 4, 4, 1>(half* out, half* src, void* stream);
// H axis unequal + unaligned: VH = 5 (H = 7)
template void LaunchTTRANSConvAxisValid<half, 0, 4, 16, 7, 4, 1, 4, 16, 5, 4, 1>(half* out, half* src, void* stream);
// W axis unequal + unaligned: VW = 4 (W = 9)
template void LaunchTTRANSConvAxisValid<half, 0, 4, 16, 4, 9, 1, 4, 16, 4, 4, 1>(half* out, half* src, void* stream);
// all axes unequal + unaligned at once
template void LaunchTTRANSConvAxisValid<half, 0, 8, 20, 7, 9, 1, 3, 7, 5, 4, 1>(half* out, half* src, void* stream);
// all axes unequal + unaligned, C0 = 8 so C1 = 3
template void LaunchTTRANSConvAxisValid<float, 0, 8, 20, 7, 9, 1, 3, 7, 5, 4, 1>(
    float* out, float* src, void* stream);
// boundary: valid == physical on every axis
template void LaunchTTRANSConvAxisValid<half, 0, 8, 20, 7, 9, 1, 8, 20, 7, 9, 1>(half* out, half* src, void* stream);
// boundary: valid == 1 on every axis
template void LaunchTTRANSConvAxisValid<half, 0, 8, 20, 7, 9, 1, 1, 1, 1, 1, 1>(half* out, half* src, void* stream);

/*-------------------- NC1HWC0 -> NCHW --------------------*/

// N axis unequal + unaligned: VN = 5 (N = 8)
template void LaunchTTRANSConvAxisValid<half, 1, 8, 2, 4, 4, 16, 5, 2, 4, 4, 16>(half* out, half* src, void* stream);
// C1 axis unequal: VC1 = 1 (C1 = 2)
template void LaunchTTRANSConvAxisValid<half, 1, 8, 2, 4, 4, 16, 8, 1, 4, 4, 16>(half* out, half* src, void* stream);
// C0 axis unequal + unaligned: VC0 = 7 (C0 = 16)
template void LaunchTTRANSConvAxisValid<half, 1, 8, 2, 4, 4, 16, 8, 2, 4, 4, 7>(half* out, half* src, void* stream);
// H axis unequal + unaligned: VH = 5 (H = 7)
template void LaunchTTRANSConvAxisValid<half, 1, 4, 2, 7, 4, 16, 4, 2, 5, 4, 16>(half* out, half* src, void* stream);
// W axis unequal + unaligned: VW = 4 (W = 9)
template void LaunchTTRANSConvAxisValid<half, 1, 4, 2, 4, 9, 16, 4, 2, 4, 4, 16>(half* out, half* src, void* stream);
// all axes unequal + unaligned at once
template void LaunchTTRANSConvAxisValid<half, 1, 8, 2, 7, 9, 16, 3, 1, 5, 4, 7>(half* out, half* src, void* stream);
// all axes unequal + unaligned, C0 = 8
template void LaunchTTRANSConvAxisValid<int32_t, 1, 8, 2, 7, 9, 8, 3, 1, 5, 4, 5>(
    int32_t* out, int32_t* src, void* stream);
// boundary: valid == physical on every axis
template void LaunchTTRANSConvAxisValid<half, 1, 8, 2, 7, 9, 16, 8, 2, 7, 9, 16>(half* out, half* src, void* stream);

/*-------------------- GNCHW -> GNC1HWC0 --------------------*/

// G axis unequal: VG = 1 (G = 2), other axes keep physical
template void LaunchTTRANSConvAxisValid<half, 2, 2, 4, 20, 5, 7, 1, 4, 20, 5, 7>(half* out, half* src, void* stream);
// all axes unequal + unaligned at once: (VG, VN, VC, VH, VW) = (1, 3, 7, 4, 5)
template void LaunchTTRANSConvAxisValid<half, 2, 2, 4, 20, 5, 7, 1, 3, 7, 4, 5>(half* out, half* src, void* stream);
