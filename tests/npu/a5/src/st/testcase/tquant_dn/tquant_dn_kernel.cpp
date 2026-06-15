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

using namespace pto;

#ifndef PTO_CEIL
#define PTO_CEIL(x, y) ((((x) + (y)-1) / (y)) * (y))
#endif

namespace TQuantDNTest {

template <typename T, int M, int N, int N_pad>
__global__ AICORE void runTQuantDN(__gm__ T __in__ *src_gm, __gm__ int8_t __out__ *fp8_nz_gm,
                                   __gm__ uint8_t __out__ *e8_zz_gm)
{
    constexpr uint32_t grpSize = 32;
    constexpr uint32_t hatM = M / grpSize;
    constexpr uint32_t paddedCols = N_pad;
    constexpr uint32_t groupedColsValid = paddedCols / 32;
    constexpr uint32_t numGroups = hatM * groupedColsValid;
    constexpr uint32_t numGroupsFlat = M * groupedColsValid;
    constexpr uint32_t numGroupsFlatAligned = PTO_CEIL(numGroupsFlat, 32);
    constexpr uint32_t paddedColsAlign32 = PTO_CEIL(paddedCols, 32);
    constexpr uint32_t paddedRows16 = PTO_CEIL(M, FRACTAL_NZ_ROW);
    constexpr uint32_t virtualRow = paddedRows16 + 1;

    constexpr uint32_t groupedColsE8Static = PTO_CEIL(groupedColsValid, 32);
    constexpr uint32_t groupedColsB16Static = PTO_CEIL(groupedColsValid, 16);
    constexpr uint32_t hatMAlign32 = PTO_CEIL(hatM, 32);

    using SrcTile = Tile<TileType::Vec, T, M, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                         PadValue::Zero>;
    using DstFP8Tile = Tile<TileType::Vec, int8_t, M, paddedCols, BLayout::RowMajor, M, paddedCols, SLayout::NoneBox,
                            512, PadValue::Zero>;
    using MaxTile = Tile<TileType::Vec, T, hatM, groupedColsB16Static, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                         512, PadValue::Zero>;
    using ScalingTile = Tile<TileType::Vec, T, hatM, groupedColsB16Static, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                             512, PadValue::Zero>;
    using E8NdTile = Tile<TileType::Vec, uint8_t, hatM, groupedColsE8Static, BLayout::RowMajor, -1, -1,
                          SLayout::NoneBox, 512, PadValue::Zero>;

    using E8DnTile = Tile<TileType::Vec, uint8_t, hatMAlign32, paddedColsAlign32, BLayout::ColMajor, -1, -1,
                          SLayout::NoneBox, 512, PadValue::Zero>;

    using E8ZzTile = Tile<TileType::Vec, uint8_t, paddedRows16, groupedColsValid, BLayout::RowMajor, -1, -1,
                          SLayout::RowMajor, 32, PadValue::Zero>;
    using E8StoreTile = Tile<TileType::Vec, uint8_t, 1, numGroupsFlatAligned, BLayout::RowMajor, -1, -1,
                             SLayout::NoneBox, 512, PadValue::Zero>;

    using Fp8NZTile = Tile<TileType::Vec, int8_t, virtualRow, paddedCols, BLayout::ColMajor, M, paddedCols,
                           SLayout::RowMajor, 512, PadValue::Null, CompactMode::RowPlusOne>;

    constexpr uint32_t colBlkCount = paddedCols / 16;
    constexpr uint32_t hatP = hatM / 2;
    constexpr uint32_t tmpBufSize =
        (BLOCK_SIZE / sizeof(uint16_t) + (colBlkCount > hatP ? colBlkCount : hatP) * (hatP > colBlkCount ? hatP : colBlkCount) +
         BLOCK_SIZE / sizeof(uint16_t)) *
        sizeof(uint16_t);
    constexpr uint32_t tmpBufSizeAligned = PTO_CEIL(tmpBufSize, 32);

    using TmpTile = Tile<TileType::Vec, uint8_t, 1, tmpBufSizeAligned, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                         512, PadValue::Zero>;

    SrcTile srcTile(M, paddedCols);
    DstFP8Tile fp8Tile;
    MaxTile maxPerGpTile(hatM, groupedColsValid);
    ScalingTile scalingTile(hatM, groupedColsValid);
    E8NdTile e8Tile(hatM, groupedColsValid);
    E8DnTile e8DnTile(hatM, paddedColsAlign32);
    E8ZzTile e8ZzTile(paddedRows16, groupedColsValid);
    E8StoreTile e8StoreTile(1, numGroupsFlatAligned);
    Fp8NZTile fp8TileNZ;
    TmpTile tmpTile(1, tmpBufSizeAligned);

    using SrcGlobal = GlobalTensor<T, Shape<1, 1, 1, M, N_pad>, pto::Stride<1, 1, 1, N_pad, 1>>;
    SrcGlobal srcGlobal(src_gm);

    using DstE8Global = GlobalTensor<uint8_t, Shape<1, 1, 1, 1, numGroupsFlatAligned>,
                                     pto::Stride<1, 1, 1, numGroupsFlatAligned, 1>>;
    DstE8Global e8Global(e8_zz_gm);

    using DstFp8GlobalNZ = GlobalTensor<int8_t, TileShape2D<int8_t, M, paddedCols, Layout::NZ>,
                                        BaseShape2D<int8_t, M, paddedCols, Layout::NZ>, Layout::NZ>;
    DstFp8GlobalNZ fp8GlobalNZ((__gm__ int8_t *)fp8_nz_gm);

    constexpr uint32_t srcTileBytes = M * paddedCols * sizeof(T);
    constexpr uint32_t maxTileBytes = hatM * groupedColsB16Static * sizeof(T);
    constexpr uint32_t scalingTileBytes = hatM * groupedColsB16Static * sizeof(T);
    constexpr uint32_t e8TileBytes = hatM * groupedColsE8Static;
    constexpr uint32_t e8DnTileBytes = hatMAlign32 * paddedColsAlign32;
    constexpr uint32_t fp8TileBytes = M * paddedCols;

    constexpr uint32_t srcTileAddr = 0x0;
    constexpr uint32_t maxTileAddr = PTO_CEIL(srcTileAddr + srcTileBytes, 0x20);
    constexpr uint32_t scalingTileAddr = PTO_CEIL(maxTileAddr + maxTileBytes, 0x20);
    constexpr uint32_t e8TileAddr = PTO_CEIL(scalingTileAddr + scalingTileBytes, 0x20);
    constexpr uint32_t e8DnTileAddr = PTO_CEIL(e8TileAddr + e8TileBytes, 0x20);
    constexpr uint32_t fp8TileAddr = 0x0;
    constexpr uint32_t C0_SIZE_B = 32;
    constexpr uint32_t nColGroupsNZ = paddedCols / C0_SIZE_B;
    constexpr uint32_t fp8NZTileBytes = (nColGroupsNZ > 1)
                                            ? (nColGroupsNZ - 1) * (paddedRows16 + 1) * C0_SIZE_B + paddedRows16 * C0_SIZE_B
                                            : paddedRows16 * C0_SIZE_B;
    constexpr uint32_t fp8NZTileAddr = PTO_CEIL(fp8TileAddr + fp8TileBytes, 0x20);
    constexpr uint32_t workTileEnd = e8DnTileAddr + e8DnTileBytes;
    constexpr uint32_t fp8NZEnd = fp8NZTileAddr + fp8NZTileBytes;
    constexpr uint32_t zzTmpStart = PTO_CEIL(workTileEnd > fp8NZEnd ? workTileEnd : fp8NZEnd, 0x20);
    constexpr uint32_t e8ZzTileAddr = zzTmpStart;
    constexpr uint32_t e8StoreTileAddr = zzTmpStart;
    constexpr uint32_t tmpTileAddr = PTO_CEIL(e8ZzTileAddr + numGroupsFlatAligned, 0x20);
    constexpr uint32_t layoutEnd = PTO_CEIL(tmpTileAddr + tmpBufSizeAligned, 0x100);
    static_assert(layoutEnd <= 0x40000, "UB layout exceeds 256 KB.");

    TASSIGN(srcTile, srcTileAddr);
    TASSIGN(maxPerGpTile, maxTileAddr);
    TASSIGN(scalingTile, scalingTileAddr);
    TASSIGN(e8Tile, e8TileAddr);
    TASSIGN(e8DnTile, e8DnTileAddr);
    TASSIGN(e8ZzTile, e8ZzTileAddr);
    TASSIGN(e8StoreTile, e8StoreTileAddr);
    TASSIGN(fp8Tile, fp8TileAddr);
    TASSIGN(fp8TileNZ, fp8NZTileAddr);
    TASSIGN(tmpTile, tmpTileAddr);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TQUANT<QuantType::MXFP8, QuantScaleAlg::OCP>(fp8Tile, srcTile, &e8Tile, &maxPerGpTile, &scalingTile, &e8DnTile);

    TMOV(fp8TileNZ, fp8Tile);
    TMOV(e8ZzTile, e8DnTile, tmpTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(e8Global, e8StoreTile);
    TSTORE(fp8GlobalNZ, fp8TileNZ);
}

template <int M, int N, int N_pad>
void LaunchTQuantDN(uint16_t *src, int8_t *fp8_nz, uint8_t *e8_zz, void *stream)
{
    runTQuantDN<bfloat16_t, M, N, N_pad><<<1, nullptr, stream>>>((bfloat16_t *)src, fp8_nz, e8_zz);
}

template void LaunchTQuantDN<128, 128, 128>(uint16_t *, int8_t *, uint8_t *, void *);
template void LaunchTQuantDN<64, 128, 128>(uint16_t *, int8_t *, uint8_t *, void *);
template void LaunchTQuantDN<64, 256, 256>(uint16_t *, int8_t *, uint8_t *, void *);
template void LaunchTQuantDN<128, 256, 256>(uint16_t *, int8_t *, uint8_t *, void *);
template void LaunchTQuantDN<64, 64, 64>(uint16_t *, int8_t *, uint8_t *, void *);
template void LaunchTQuantDN<128, 64, 64>(uint16_t *, int8_t *, uint8_t *, void *);

} // namespace TQuantDNTest
