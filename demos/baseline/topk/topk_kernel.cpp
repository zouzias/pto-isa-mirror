/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <int Cols>
PTO_INTERNAL int32_t FillMrgArray(int32_t *mrgArray, int blockLen)
{
    int32_t arrayCount = 0;
    int32_t tmpInner = Cols;
    for (int32_t i = blockLen; i >= 64; i /= 4) {
        int32_t count;
        for (count = 0; count < tmpInner / i; count++) {
            mrgArray[arrayCount++] = i;
        }
        tmpInner -= count * i;
    }
    return arrayCount;
}

template <typename DstTileData, typename SrcTileData, typename TmpTileData, typename T, int Cols, int topk>
PTO_INTERNAL void SortTailBlock(DstTileData &dstTile, SrcTileData &srcTile, int blockLen, uint64_t tmpAddr)
{
    TmpTileData tmp1Tile(1, Cols);
    TASSIGN(tmp1Tile, tmpAddr);

    int32_t mrgArray[15] = {0};
    int32_t arrayCount = FillMrgArray<Cols>(mrgArray, blockLen);
    uint16_t mrgSortedLen = 0;
    MrgSortExecutedNumList executedNumList;
    for (int32_t i = 0; i < arrayCount - 1; ++i) {
        mrgSortedLen += static_cast<uint16_t>(mrgArray[i]);
        uint64_t tmpMrgSortedLen = mrgSortedLen;
        uint64_t tmpMrgArray = mrgArray[i + 1];
        if (tmpMrgSortedLen > topk) {
            tmpMrgSortedLen = topk;
        }
        if (tmpMrgArray > topk) {
            tmpMrgArray = topk;
        }

        SrcTileData src0Tile(1, tmpMrgSortedLen);
        SrcTileData src1Tile(1, tmpMrgArray);
        SrcTileData curDstTile(1, tmpMrgSortedLen + tmpMrgArray);
        TASSIGN(src0Tile, (uint64_t)srcTile.data());
        TASSIGN(src1Tile, (uint64_t)srcTile.data() + mrgSortedLen * sizeof(T));
        TASSIGN(curDstTile, (uint64_t)srcTile.data());
        TMRGSORT<DstTileData, TmpTileData, SrcTileData, SrcTileData, 0>(
            curDstTile, executedNumList, tmp1Tile, src0Tile, src1Tile);
        pipe_barrier(PIPE_V);
    }
}

template <typename DstTileData, typename SrcTileData, int kTRows_, int kTCols_, int valid_row, int valid_col, int dtopk>
PTO_INTERNAL void MrgsortSingleRow(DstTileData &dstTile, SrcTileData &srcTile, uint64_t tmpAddr)
{
    using T = typename SrcTileData::DType;
    constexpr uint32_t TYPE_COEF = sizeof(float) / sizeof(T);
    uint32_t blockLen = 64 * TYPE_COEF;

    // Merge sort data for every 4 blockLen lengths.
    for (; blockLen * 4 <= valid_col; blockLen *= 4) {
        uint16_t cols = valid_col / (blockLen * 4) * (blockLen * 4);
        SrcTileData srcSortedTile(1, cols);
        SrcTileData tmpSortedTile(1, cols);
        TASSIGN(srcSortedTile, (uint64_t)srcTile.data());
        TASSIGN(tmpSortedTile, tmpAddr);
        TMRGSORT<SrcTileData, SrcTileData>(tmpSortedTile, srcSortedTile, blockLen);
        pipe_barrier(PIPE_V);
        TMOV(srcSortedTile, tmpSortedTile);
        pipe_barrier(PIPE_V);
    }

    // sort tail block
    if (blockLen < valid_col) {
        SortTailBlock<DstTileData, SrcTileData, SrcTileData, T, valid_col, dtopk>(srcTile, srcTile, blockLen, tmpAddr);
        SrcTileData tmpMovTile(1, dtopk);
        TASSIGN(tmpMovTile, (uint64_t)srcTile.data());
        pipe_barrier(PIPE_V);
        TMOV(dstTile, tmpMovTile);
    } else {
        SrcTileData tmpMovTile(1, dtopk);
        TASSIGN(tmpMovTile, (uint64_t)srcTile.data());
        pipe_barrier(PIPE_V);
        TMOV(dstTile, tmpMovTile);
    }
}

template <typename T, typename DstTileData, typename SrcTileData, typename RowTile, int kTRows_, int kTCols_,
    int validRow, int validCol, int topk>
PTO_INTERNAL void MrgsortSingleTile(DstTileData &dstTile, SrcTileData &srcTile, uint64_t tmpAddr)
{
    for (int i = 0; i < validRow; i++) {
        RowTile rowSrcTile(1, validCol);
        TASSIGN(rowSrcTile, (uint64_t)srcTile.data() + i * kTCols_ * sizeof(T));
        RowTile rowDstTile(1, validCol);
        TASSIGN(rowDstTile, (uint64_t)dstTile.data() + i * DstTileData::Cols * sizeof(T));
        MrgsortSingleRow<RowTile, RowTile, 1, kTCols_, 1, validCol, topk>(rowDstTile, rowSrcTile, tmpAddr);
    }
}

template <typename T, typename DstTileData, typename RowTile, typename Src0TileData, typename Src1TileData,
    typename Src2TileData, typename Src3TileData, int listNum>
PTO_INTERNAL void MrgsortAmongTiles(
    DstTileData &dst, Src0TileData &src0, Src1TileData &src1, Src2TileData &src2, Src3TileData &src3)
{
    int validRow = dst.GetValidRow();
    int validCol = dst.GetValidCol();
    constexpr int tmpCols =
        (listNum == 2) ?
            (Src0TileData::Cols + Src1TileData::Cols) :
            ((listNum == 3) ? (Src0TileData::Cols + Src1TileData::Cols + Src2TileData::Cols) :
                              (Src0TileData::Cols + Src1TileData::Cols + Src2TileData::Cols + Src3TileData::Cols));
    int tmpCol = src0.GetValidCol() + src1.GetValidCol();
    if constexpr (listNum >= 3) {
        tmpCol += src2.GetValidCol();
    }
    if constexpr (listNum == 4) {
        tmpCol += src3.GetValidCol();
    }
    for (int i = 0; i < validRow; i++) {
        RowTile tmpTile(1, tmpCol);
        TASSIGN(tmpTile, (uint64_t)dst.data() + DstTileData::Rows * DstTileData::Cols * sizeof(T));

        RowTile rowDstTile(1, validCol);
        TASSIGN(rowDstTile, (uint64_t)dst.data() + i * DstTileData::Cols * sizeof(T));

        RowTile rowSrc0Tile(1, src0.GetValidCol());
        TASSIGN(rowSrc0Tile, (uint64_t)src0.data() + i * Src0TileData::Cols * sizeof(T));

        RowTile rowSrc1Tile(1, src1.GetValidCol());
        TASSIGN(rowSrc1Tile, (uint64_t)src1.data() + i * Src1TileData::Cols * sizeof(T));

        MrgSortExecutedNumList executedNumList;
        if constexpr (listNum == 2) {
            TMRGSORT<RowTile, RowTile, RowTile, RowTile, 0>(
                rowDstTile, executedNumList, tmpTile, rowSrc0Tile, rowSrc1Tile);
        } else if constexpr (listNum == 3) {
            RowTile rowSrc2Tile(1, src2.GetValidCol());
            TASSIGN(rowSrc2Tile, (uint64_t)src2.data() + i * Src2TileData::Cols * sizeof(T));

            TMRGSORT<RowTile, RowTile, RowTile, RowTile, RowTile, 0>(
                rowDstTile, executedNumList, tmpTile, rowSrc0Tile, rowSrc1Tile, rowSrc2Tile);
        } else if constexpr (listNum == 4) {
            RowTile rowSrc2Tile(1, src2.GetValidCol());
            TASSIGN(rowSrc2Tile, (uint64_t)src2.data() + i * Src2TileData::Cols * sizeof(T));

            RowTile rowSrc3Tile(1, src3.GetValidCol());
            TASSIGN(rowSrc3Tile, (uint64_t)src3.data() + i * Src3TileData::Cols * sizeof(T));

            TMRGSORT<RowTile, RowTile, RowTile, RowTile, RowTile, RowTile, 0>(
                rowDstTile, executedNumList, tmpTile, rowSrc0Tile, rowSrc1Tile, rowSrc2Tile, rowSrc3Tile);
        }
    }
}

template <typename T, typename DstTileData, typename SrcTileData, typename RowTile, int kTRows_, int kTCols_,
    int validRow, int validCol>
PTO_INTERNAL void CreateIndexAndSort32(DstTileData &dst, SrcTileData &src, int start, uint64_t idxAddr)
{
    using indexT = uint32_t;
    constexpr int TYPE_COEF = sizeof(float) / sizeof(T);
    using TileIndex_dst = Tile<TileType::Vec, indexT, 1, kTCols_, BLayout::RowMajor, -1, -1>;
    TileIndex_dst idxTile(1, validCol);
    TASSIGN(idxTile, idxAddr);
    TCI<TileIndex_dst, indexT, 0>(idxTile, start);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    for (size_t i = 0; i < validRow; ++i) {
        RowTile dstRowTile(1, validCol * 2 * TYPE_COEF);
        RowTile srcRowTile(1, validCol);
        RowTile tmpTile(1, validCol);
        TASSIGN(dstRowTile, (uint64_t)dst.data() + i * DstTileData::Cols * sizeof(T));
        TASSIGN(srcRowTile, (uint64_t)src.data() + i * SrcTileData::Cols * sizeof(T));
        TASSIGN(tmpTile, (uint64_t)idxTile.data() + kTCols_ * sizeof(indexT));
        TSORT32(dstRowTile, srcRowTile, idxTile, tmpTile);
        pipe_barrier(PIPE_V);
    }
}

template <typename T, typename DstTileData, typename SrcTileData, typename RowTile, bool isIndex>
PTO_INTERNAL void ExtractDataOrIndex(DstTileData &dstTile, SrcTileData &srcTile)
{
    for (size_t i = 0; i < srcTile.GetValidRow(); ++i) {
        RowTile rowTile(1, srcTile.GetValidCol());
        TASSIGN(rowTile, (uint64_t)srcTile.data() + i * SrcTileData::Cols * sizeof(T));
        if constexpr (isIndex == false) {
            RowTile rowDTile(1, dstTile.GetValidCol());
            TASSIGN(rowDTile, (uint64_t)dstTile.data() + i * DstTileData::Cols * sizeof(T));
            if constexpr (std::is_same_v<T, half>) {
                TGATHER<RowTile, RowTile, MaskPattern::P0001>(rowDTile, rowTile);
            } else {
                TGATHER<RowTile, RowTile, MaskPattern::P0101>(rowDTile, rowTile);
            }
        } else {
            using indexT = uint32_t;
            using CopySrcTileData = Tile<TileType::Vec, indexT, 1, DstTileData::Cols * 2, BLayout::RowMajor, -1, -1>;
            CopySrcTileData copyTile(1, dstTile.GetValidCol() * 2);
            TASSIGN(copyTile, (uint64_t)rowTile.data());

            using IndexRowTileData = Tile<TileType::Vec, indexT, 1, DstTileData::Cols, BLayout::RowMajor, -1, -1>;
            IndexRowTileData rowITile(1, dstTile.GetValidCol());
            TASSIGN(rowITile, (uint64_t)dstTile.data() + i * DstTileData::Cols * sizeof(indexT));

            TGATHER<IndexRowTileData, CopySrcTileData, MaskPattern::P1010>(rowITile, copyTile);
        }
    }
}

template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int topk, int start, int blockDim>
AICORE inline void runTOPKMulti(__gm__ T *origOut, __gm__ uint32_t *origIndex, __gm__ T *origSrc)
{
    using indexT = uint32_t;

    constexpr int totalRow = gShape0 * gShape1 * gShape2 * gShape3;
    constexpr int validRow = gShape0 * gShape1 * gShape2 * gShape3 / blockDim;
    constexpr int validCol = gShape4;
    __gm__ T *src = origSrc + get_block_idx() * validRow * gWholeShape4;
    __gm__ T *out = origOut + get_block_idx() * validRow * topk;
    __gm__ uint32_t *index = origIndex + get_block_idx() * validRow * topk;
    constexpr int Rows = gWholeShape0 * gWholeShape1 * gWholeShape2 * gWholeShape3 / blockDim;
    constexpr int Cols = gWholeShape4;

    // 2 tiles
    constexpr int tileNumTwo = 2;
    constexpr int PerCols = validCol / tileNumTwo;
    static_assert(validCol % (32 * tileNumTwo) == 0, "expect validCol % (32 * tileNumTwo) == 0.");
    using DynShapeDim5 = Shape<1, 1, 1, validRow, PerCols>;
    using DynStridDim5 = Stride<Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    GlobalData src0Global(src); // ND2ND
    GlobalData src1Global(src + PerCols);

    constexpr int TYPE_COEF = sizeof(float) / sizeof(T);
    constexpr int PerDstCols = PerCols * 2 * TYPE_COEF;
    using DstTileData = Tile<TileType::Vec, T, validRow, PerDstCols, BLayout::RowMajor, validRow, PerDstCols>;
    DstTileData sort32DstTile[tileNumTwo];
    constexpr uint32_t sort32DstSize = validRow * PerDstCols * sizeof(T) * tileNumTwo;
    TASSIGN(sort32DstTile[0], 0x0);
    TASSIGN(sort32DstTile[1], 0x0 + sort32DstSize / tileNumTwo);

    using SrcTileData = Tile<TileType::Vec, T, validRow, PerCols, BLayout::RowMajor, validRow, PerCols>;
    SrcTileData srcTile[tileNumTwo];
    uint32_t srcSize = validRow * PerCols * sizeof(T) * tileNumTwo;
    TASSIGN(srcTile[0], 0x0 + sort32DstSize);
    TASSIGN(srcTile[1], 0x0 + sort32DstSize + srcSize / tileNumTwo);

    using SingleRowTileData = Tile<TileType::Vec, T, 1, PerDstCols, BLayout::RowMajor, -1, -1>;
    // after sort32, new colsize: old_col * 2 * TYPE_COEF

    TLOAD(srcTile[0], src0Global);
    uint64_t idxAddr = 0x0 + sort32DstSize + srcSize;
    CreateIndexAndSort32<T, DstTileData, SrcTileData, SingleRowTileData,
        validRow, PerCols, validRow, PerCols>(sort32DstTile[0], srcTile[0], 0, idxAddr);

    TLOAD(srcTile[1], src1Global);
    idxAddr += srcSize;
    CreateIndexAndSort32<T, DstTileData, SrcTileData, SingleRowTileData,
        validRow, PerCols, validRow, PerCols>(sort32DstTile[1], srcTile[1], PerCols, idxAddr);

    pipe_barrier(PIPE_V);
    DstTileData perMrgDstTile[tileNumTwo];
    TASSIGN(perMrgDstTile[0], 0x0 + sort32DstSize);
    TASSIGN(perMrgDstTile[1], 0x0 + sort32DstSize + sort32DstSize / 2);
    uint64_t tmpAddr = 0x0 + sort32DstSize * 2;
    MrgsortSingleTile<T, DstTileData, DstTileData, SingleRowTileData,
        validRow, PerDstCols, validRow, PerDstCols, PerDstCols>(perMrgDstTile[0], sort32DstTile[0], tmpAddr);

    pipe_barrier(PIPE_V);
    MrgsortSingleTile<T, DstTileData, DstTileData, SingleRowTileData,
        validRow, PerDstCols, validRow, PerDstCols, PerDstCols>(perMrgDstTile[1], sort32DstTile[1], tmpAddr);

    using TiledMrgDstTileData = Tile<TileType::Vec, T, validRow, validCol * 2 * TYPE_COEF, BLayout::RowMajor,
                                     validRow, topk * 2 * TYPE_COEF>;
    using DstRowTile = Tile<TileType::Vec, T, 1, validCol * 2 * TYPE_COEF, BLayout::RowMajor, -1, -1>;
    TiledMrgDstTileData mrgDstTile;
    TASSIGN(mrgDstTile, 0x0 + sort32DstSize * 2);
    // mrgsort between two tiles
    pipe_barrier(PIPE_V);

    MrgsortAmongTiles<T, TiledMrgDstTileData, DstRowTile, DstTileData, DstTileData,
        DstTileData, DstTileData, 2>(mrgDstTile, perMrgDstTile[0], perMrgDstTile[1], perMrgDstTile[1], perMrgDstTile[1]);

    using DstDataTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    DstDataTileData dTile(validRow, topk);
    TASSIGN(dTile, 0x0);
    using DstIndexTileData = Tile<TileType::Vec, indexT, Rows, Cols, BLayout::RowMajor, -1, -1>;
    DstIndexTileData iTile(validRow, topk);
    TASSIGN(iTile, 0x0 + Rows * Cols * sizeof(T));

    pipe_barrier(PIPE_V);
    ExtractDataOrIndex<T, DstDataTileData, TiledMrgDstTileData, DstRowTile, 0>(dTile, mrgDstTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    pipe_barrier(PIPE_V);
    ExtractDataOrIndex<T, DstIndexTileData, TiledMrgDstTileData, DstRowTile, 1>(iTile, mrgDstTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);

    using DstShapeDim5 = Shape<1, 1, 1, validRow, topk>;
    using DstStridDim5 = Stride<validRow * topk, validRow * topk, validRow * topk, topk, 1>;
    using DstDataGlobalData = GlobalTensor<T, DstShapeDim5, DstStridDim5>;
    DstDataGlobalData dstDataGlobal(out);

    using DstIdxGlobalData = GlobalTensor<indexT, DstShapeDim5, DstStridDim5>;
    DstIdxGlobalData dstIdxGlobal(index);

    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstDataGlobal, dTile);

    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(dstIdxGlobal, iTile);

    out = dstDataGlobal.data();
    index = dstIdxGlobal.data();
}

template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int topk, int start, int blockDim>
__global__ AICORE void Topk(__gm__ uint8_t *out, __gm__ uint8_t *index, __gm__ uint8_t *src)
{
    using indexT = uint32_t;
    if constexpr (std::is_same_v<T, uint16_t>) {
        runTOPKMulti<half, gShape0, gShape1, gShape2, gShape3, gShape4,
            gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4,
            topk, start, blockDim>(reinterpret_cast<__gm__ half *>(out),
            reinterpret_cast<__gm__ indexT *>(index), reinterpret_cast<__gm__ half *>(src));
    } else {
        runTOPKMulti<float, gShape0, gShape1, gShape2, gShape3, gShape4,
            gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4,
            topk, start, blockDim>(reinterpret_cast<__gm__ float *>(out),
            reinterpret_cast<__gm__ indexT *>(index), reinterpret_cast<__gm__ float *>(src));
    }
}

template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int topk, int start>
void launchTopk(uint8_t *out, uint8_t *index, uint8_t *src, void *stream)
{
    constexpr int blockDim = 4;
    Topk<T, gShape0, gShape1, gShape2, gShape3, gShape4,
        gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4,
        topk, start, blockDim><<<blockDim, nullptr, stream>>>(out, index, src);
}

template void launchTopk<float, 1, 1, 1, 8, 1024, 1, 1, 1, 8, 1280, 1000, 0>(uint8_t *out, uint8_t *index, uint8_t *src, void *stream);