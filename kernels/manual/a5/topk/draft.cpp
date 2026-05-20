/**
 * Radix-select TopK (2-byte key) for Ascend A5 with pto-isa.
 *
 * All pto-isa ops (including TASSIGN / TLOAD) live only in five `Phase*` functions — no callees that emit
 * T-instructions. `RunRadixTopKDraft` only constructs tile objects and calls the phases. Tiled 8192 / 256:
 * Phase1/3 stream `TLOAD` + `THISTOGRAM`; Phase5 per-tile `TGATHER` + per-tile six-arg `TCONCAT_IMPL` into `gtSeg`/`eqSeg`.
 *
 * 1) **Phase1** — TASSIGN, tile `TLOAD` + `THISTOGRAM<BYTE_1>`, cumulative `chistMSB`
 * 2) **Phase2** — `TCMPS`/`TCI`/`TSELS`, raw MSB + `WinnerBinU8` path, `TGATHER`+`TSUB` remainK
 * 3) **Phase3** — `TCVT` idx; tile `TLOAD` + `THISTOGRAM<BYTE_0>`, `chistLSB`
 * 4) **Phase4** — LSB winner + `TOR` packed threshold; `TASSIGN` `packedThrU` @ `kRemainUbOut`
 * 5) **Phase5** — per-tile `TGATHER` + six-arg `TCONCAT_IMPL` (`NeetCntDstIdx`): `idx*Out` → `idx*Acc` per loop, final five-arg merge + `TSTORE`
 *
 * Current example: N = 8192, TopK = 512 (see `scripts/gen_data.py`).
 */

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/npu/a5/TCmps.hpp>
#include <pto/npu/a5/Tci.hpp>
#include <pto/npu/a5/TSels.hpp>
#include <pto/npu/a5/THistogram.hpp>
#include <pto/npu/a5/TGather.hpp>
#include <pto/npu/a5/TConcat.hpp>

using namespace pto;

namespace topk_radix_detail {

constexpr int kN = 8192;
constexpr int kTileCols = 256;
constexpr int kBinNum = 256;
#define PTO_DIV_ROUNDUP(x, y) (((x) + (y)-1) / (y))
#define PTO_CEIL(x, y) (PTO_DIV_ROUNDUP(x, y) * (y))

template <int ValidCols>
using InTileU16 = Tile<TileType::Vec, uint16_t, 1, ValidCols, BLayout::RowMajor, -1, -1>;
using HistTile = Tile<TileType::Vec, uint32_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
using WinnerLaneTile = Tile<TileType::Vec, uint32_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
using WinnerBinTile = Tile<TileType::Vec, uint32_t, 1, 32, BLayout::RowMajor, -1, -1>;
using MaskCmpTile = Tile<TileType::Vec, uint8_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
using IdxU32Tile = Tile<TileType::Vec, uint32_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
using TmpSelsTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
using RowMinDstTile = Tile<TileType::Vec, uint32_t, 1, 16, BLayout::RowMajor, -1, -1>;
using RowMinTmpTile = Tile<TileType::Vec, uint32_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
using SelMaskRowTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
using GatherIdxU32Tile = Tile<TileType::Vec, uint32_t, 1, 32, BLayout::RowMajor, -1, -1>;
using RemainKTile = Tile<TileType::Vec, uint32_t, 1, 32, BLayout::RowMajor, -1, -1>;
using PackedU16Tile = Tile<TileType::Vec, uint16_t, 1, 32, BLayout::RowMajor, -1, -1>;

constexpr uint64_t kWinnerUbMask = 0x23000;
constexpr uint64_t kWinnerUbIdx = 0x23400;
constexpr uint64_t kWinnerUbGather = 0x23800;
constexpr uint64_t kWinnerUbTmp = 0x23C00;
constexpr uint64_t kWinnerUbRowMinDst = 0x24000;
constexpr uint64_t kWinnerUbRowMinTmp = 0x24400;
constexpr uint64_t kWinnerUbSelMask = 0x24800;
constexpr uint64_t kWinnerUbSelZero = 0x24840;
constexpr uint64_t kWinnerUbU32One = 0x24880;
constexpr uint64_t kWinnerUbSelOut = 0x24900;
constexpr uint64_t kWinnerUbTselTmp = 0x24A00;
constexpr uint64_t kRemainUbTopk = 0x24E00;
constexpr uint64_t kRemainUbN = 0x24E80;
constexpr uint64_t kRemainUbCw = 0x24F00;
constexpr uint64_t kRemainUbSumAbove = 0x24F80;
constexpr uint64_t kRemainUbOut = 0x25000;
constexpr uint64_t kMsbWinnerSavedUb = 0x24D80;
constexpr uint16_t kIdxAlignedRows = PTO_CEIL(1 * sizeof(uint8_t), BLOCK_BYTE_SIZE);
using IdxFilterTile = Tile<TileType::Vec, uint8_t, kIdxAlignedRows, 1, BLayout::ColMajor, -1, -1>;

template <int ValidCols>
using GatherSrcI16 = Tile<TileType::Vec, int16_t, 1, ValidCols, BLayout::RowMajor, -1, -1>;
using GatherChunkU32 = Tile<TileType::Vec, uint32_t, 1, kTileCols, BLayout::RowMajor, -1, -1>;
constexpr int kGatherConcatRows =
    (1 * static_cast<int>(sizeof(uint32_t)) < 32) ? (32 / static_cast<int>(sizeof(uint32_t))) : 1;
using GatherConcatCountTile = Tile<TileType::Vec, uint32_t, kGatherConcatRows, 1, BLayout::ColMajor, -1, -1>;

// --- Five phases: no T-instruction callees below this line. ---

AICORE inline void Phase1_LoadAndHistogramMsb(__gm__ uint16_t *src, InTileU16<kTileCols> &inTile, HistTile &tileHist,
                                              HistTile &chistMSB, IdxFilterTile &idxFilter)
{
    TASSIGN(inTile, 0x00000);
    TASSIGN(tileHist, 0x10000);
    TASSIGN(chistMSB, 0x14000);
    TASSIGN(idxFilter, 0x1C000);

    chistMSB.SetValidRow(1);
    chistMSB.SetValidCol(kBinNum);
    TEXPANDS(chistMSB, 0u);

    constexpr int kLoop = (kN + kTileCols - 1) / kTileCols;
    for (int i = 0; i < kLoop; ++i) {
        int base = i * kTileCols;
        int valid = (base + kTileCols <= kN) ? kTileCols : (kN - base);
        if (valid <= 0) {
            break;
        }

        if (i != 0) {
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        }
        inTile.SetValidCol(valid);
        using SrcGlobal = GlobalTensor<uint16_t, pto::Shape<1, 1, 1, 1, kTileCols>,
                                       pto::Stride<kTileCols, kTileCols, kTileCols, kTileCols, 1>>;
        SrcGlobal srcGlobal(src + base);
        TLOAD(inTile, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        THISTOGRAM<pto::HistByte::BYTE_1>(tileHist, inTile, idxFilter);
        TADD(chistMSB, chistMSB, tileHist);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    }
}

template <int TopK>
AICORE inline void Phase2_WinnerMsbAndRemainK(HistTile &chistMSB, WinnerBinTile &msbWinnerBin, RemainKTile &remainKTile,
                                               WinnerBinTile &msbWinnerSaved)
{
    constexpr uint32_t kThrMsb = static_cast<uint32_t>(kN - TopK);
    constexpr uint32_t kSelsFalse = 0xffffffffu;
    WinnerLaneTile msbWinnerLanes(1, kBinNum);

    {
        MaskCmpTile maskTile(1, kBinNum);
        IdxU32Tile indexTile(1, kBinNum);
        TmpSelsTile tmpSelsTile(1, 32);
        TASSIGN(maskTile, kWinnerUbMask);
        TASSIGN(indexTile, kWinnerUbIdx);
        TASSIGN(msbWinnerLanes, kWinnerUbGather);
        TASSIGN(tmpSelsTile, kWinnerUbTmp);
        maskTile.SetValidCol(kBinNum);
        indexTile.SetValidCol(kBinNum);
        msbWinnerLanes.SetValidCol(kBinNum);
        tmpSelsTile.SetValidCol(32);
        chistMSB.SetValidCol(kBinNum);
        TCMPS(maskTile, chistMSB, kThrMsb, CmpMode::GE);
        TCI<IdxU32Tile, IdxU32Tile, uint32_t, 0>(indexTile, static_cast<uint32_t>(0), indexTile);
        TSELS(msbWinnerLanes, maskTile, indexTile, tmpSelsTile, kSelsFalse);
    }

    TASSIGN(msbWinnerSaved, kMsbWinnerSavedUb);
    msbWinnerSaved.SetValidRow(1);
    msbWinnerSaved.SetValidCol(32);
    {
        RowMinDstTile rowMinDst(1, 16);
        RowMinTmpTile rowMinTmp(1, kBinNum);
        TASSIGN(rowMinDst, kWinnerUbRowMinDst);
        TASSIGN(rowMinTmp, kWinnerUbRowMinTmp);
        rowMinDst.SetValidRow(1);
        rowMinDst.SetValidCol(1);
        rowMinTmp.SetValidRow(1);
        rowMinTmp.SetValidCol(kBinNum);
        TROWMIN(rowMinDst, msbWinnerLanes, rowMinTmp);
        GatherIdxU32Tile gatherIdx(1, 32);
        TmpSelsTile gatherTmp(1, 32);
        TASSIGN(gatherIdx, kWinnerUbRowMinTmp);
        TASSIGN(gatherTmp, kWinnerUbTselTmp);
        gatherIdx.SetValidRow(1);
        gatherIdx.SetValidCol(32);
        gatherTmp.SetValidCol(32);
        TEXPANDS(gatherIdx, 0u);
        TGATHER(msbWinnerSaved, rowMinDst, gatherIdx, gatherTmp);
    }

    {
        RowMinDstTile rowMinDst(1, 16);
        RowMinTmpTile rowMinTmp(1, kBinNum);
        RowMinDstTile zeroTile(1, 16);
        RowMinDstTile selOut(1, 16);
        SelMaskRowTile selMask(1, 32);
        TmpSelsTile tselTmp(1, 32);
        TASSIGN(rowMinDst, kWinnerUbRowMinDst);
        TASSIGN(rowMinTmp, kWinnerUbRowMinTmp);
        TASSIGN(zeroTile, kWinnerUbSelZero);
        TASSIGN(selOut, kWinnerUbSelOut);
        TASSIGN(selMask, kWinnerUbSelMask);
        TASSIGN(tselTmp, kWinnerUbTselTmp);
        rowMinDst.SetValidRow(1);
        rowMinDst.SetValidCol(1);
        rowMinTmp.SetValidRow(1);
        rowMinTmp.SetValidCol(kBinNum);
        zeroTile.SetValidRow(1);
        zeroTile.SetValidCol(1);
        selOut.SetValidRow(1);
        selOut.SetValidCol(1);
        selMask.SetValidRow(1);
        selMask.SetValidCol(1);
        tselTmp.SetValidCol(32);
        TROWMIN(rowMinDst, msbWinnerLanes, rowMinTmp);
        {
            RowMinDstTile uOne(1, 16);
            TASSIGN(uOne, kWinnerUbU32One);
            uOne.SetValidRow(1);
            uOne.SetValidCol(1);
            TEXPANDS(uOne, 1u);
            TSUB(rowMinDst, rowMinDst, uOne);
        }
        constexpr uint32_t kCmp256 = 256u;
        TCMPS(selMask, rowMinDst, kCmp256, CmpMode::GT);
        TSEL(selOut, selMask, zeroTile, rowMinDst, tselTmp);
        TASSIGN(msbWinnerBin, kWinnerUbTmp);
        msbWinnerBin.SetValidRow(1);
        msbWinnerBin.SetValidCol(32);
        GatherIdxU32Tile gatherIdx(1, 32);
        TmpSelsTile gatherTmp(1, 32);
        TASSIGN(gatherIdx, kWinnerUbRowMinTmp);
        TASSIGN(gatherTmp, kWinnerUbTselTmp);
        gatherIdx.SetValidRow(1);
        gatherIdx.SetValidCol(32);
        gatherTmp.SetValidCol(32);
        TEXPANDS(gatherIdx, 0u);
        TGATHER(msbWinnerBin, selOut, gatherIdx, gatherTmp);
    }

    {
        using U32x32 = Tile<TileType::Vec, uint32_t, 1, 32, BLayout::RowMajor, -1, -1>;
        U32x32 thrMsbT(1, 32);
        U32x32 cwT(1, 32);
        U32x32 cwFixT(1, 32);
        TmpSelsTile gatherTmp(1, 32);
        SelMaskRowTile cwSelMask(1, 32);
        TmpSelsTile cwSelTmp(1, 32);
        TASSIGN(thrMsbT, kRemainUbTopk);
        TASSIGN(cwT, kRemainUbCw);
        TASSIGN(cwFixT, kRemainUbN);
        TASSIGN(remainKTile, kRemainUbOut);
        TASSIGN(gatherTmp, kWinnerUbRowMinTmp);
        TASSIGN(cwSelMask, kWinnerUbSelMask);
        TASSIGN(cwSelTmp, kWinnerUbTselTmp);
        thrMsbT.SetValidRow(1);
        thrMsbT.SetValidCol(32);
        cwT.SetValidRow(1);
        cwT.SetValidCol(32);
        cwFixT.SetValidRow(1);
        cwFixT.SetValidCol(32);
        remainKTile.SetValidRow(1);
        remainKTile.SetValidCol(32);
        gatherTmp.SetValidCol(32);
        cwSelMask.SetValidRow(1);
        cwSelMask.SetValidCol(32);
        cwSelTmp.SetValidCol(32);
        constexpr uint32_t kThrMsbU = static_cast<uint32_t>(kN - TopK);
        TEXPANDS(thrMsbT, kThrMsbU);
        TGATHER(cwT, chistMSB, msbWinnerBin, gatherTmp);
        TEXPANDS(remainKTile, 0u);
        TCMPS(cwSelMask, msbWinnerSaved, static_cast<uint32_t>(0), CmpMode::EQ);
        TSEL(cwFixT, cwSelMask, remainKTile, cwT, cwSelTmp);
        TSUB(remainKTile, thrMsbT, cwFixT);
        // PIPE_V/PIPE_S sync removed: test whether Phase3 TCMPS(GT) vs remainK is safe without scalar fence
    }
}

AICORE inline void Phase3_HistogramLsb(__gm__ uint16_t *src, InTileU16<kTileCols> &inTile, HistTile &tileHist,
                                        HistTile &chistLSB, IdxFilterTile &idxFilter, WinnerBinTile &msbWinnerSaved)
{
    TASSIGN(inTile, 0x00000);
    TASSIGN(tileHist, 0x10000);
    TASSIGN(chistLSB, 0x18000);
    TASSIGN(idxFilter, 0x1C000);

    idxFilter.SetValidRow(1);
    idxFilter.SetValidCol(1);
    msbWinnerSaved.SetValidRow(1);
    msbWinnerSaved.SetValidCol(1);
    TCVT(idxFilter, msbWinnerSaved, RoundMode::CAST_TRUNC);

    chistLSB.SetValidRow(1);
    chistLSB.SetValidCol(kBinNum);
    TEXPANDS(chistLSB, 0u);

    constexpr int kLoop = (kN + kTileCols - 1) / kTileCols;
    for (int i = 0; i < kLoop; ++i) {
        int base = i * kTileCols;
        int valid = (base + kTileCols <= kN) ? kTileCols : (kN - base);
        if (valid <= 0) {
            break;
        }

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        inTile.SetValidCol(valid);
        using SrcGlobal2 = GlobalTensor<uint16_t, pto::Shape<1, 1, 1, 1, kTileCols>,
                                        pto::Stride<kTileCols, kTileCols, kTileCols, kTileCols, 1>>;
        SrcGlobal2 srcGlobal(src + base);
        TLOAD(inTile, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        THISTOGRAM<pto::HistByte::BYTE_0>(tileHist, inTile, idxFilter);
        TADD(chistLSB, chistLSB, tileHist);
        if (i + 1 < kLoop) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        }
    }
}

AICORE inline void Phase4_WinnerLsbRemainKAndPackedThresholdTor(HistTile &chistLSB, RemainKTile &remainKTile,
                                                                 WinnerBinTile &lsbWinnerBin, WinnerBinTile &msbWinnerSaved,
                                                                 PackedU16Tile &packedThrU)
{
    constexpr uint32_t kSelsFalse2 = 0xffffffffu;
    WinnerLaneTile lsbWinnerLanes(1, kBinNum);
    {
        MaskCmpTile maskTile(1, kBinNum);
        IdxU32Tile indexTile(1, kBinNum);
        TmpSelsTile tmpSelsTile(1, 32);
        TASSIGN(maskTile, kWinnerUbMask);
        TASSIGN(indexTile, kWinnerUbIdx);
        TASSIGN(lsbWinnerLanes, kWinnerUbGather);
        TASSIGN(tmpSelsTile, kWinnerUbTmp);
        maskTile.SetValidCol(kBinNum);
        indexTile.SetValidCol(kBinNum);
        lsbWinnerLanes.SetValidCol(kBinNum);
        tmpSelsTile.SetValidCol(32);
        chistLSB.SetValidCol(kBinNum);
        remainKTile.SetValidRow(1);
        remainKTile.SetValidCol(32);
        TCMPS(maskTile, chistLSB, remainKTile, CmpMode::GT);
        TCI<IdxU32Tile, IdxU32Tile, uint32_t, 0>(indexTile, static_cast<uint32_t>(0), indexTile);
        TSELS(lsbWinnerLanes, maskTile, indexTile, tmpSelsTile, kSelsFalse2);
    }

    TASSIGN(lsbWinnerBin, kWinnerUbTmp);
    lsbWinnerBin.SetValidRow(1);
    lsbWinnerBin.SetValidCol(32);
    {
        RowMinDstTile rowMinDst(1, 16);
        RowMinTmpTile rowMinTmp(1, kBinNum);
        TASSIGN(rowMinDst, kWinnerUbRowMinDst);
        TASSIGN(rowMinTmp, kWinnerUbRowMinTmp);
        rowMinDst.SetValidRow(1);
        rowMinDst.SetValidCol(1);
        rowMinTmp.SetValidRow(1);
        rowMinTmp.SetValidCol(kBinNum);
        TROWMIN(rowMinDst, lsbWinnerLanes, rowMinTmp);
        GatherIdxU32Tile gatherIdx(1, 32);
        TmpSelsTile gatherTmp(1, 32);
        TASSIGN(gatherIdx, kWinnerUbRowMinTmp);
        TASSIGN(gatherTmp, kWinnerUbTselTmp);
        gatherIdx.SetValidRow(1);
        gatherIdx.SetValidCol(32);
        gatherTmp.SetValidCol(32);
        TEXPANDS(gatherIdx, 0u);
        TGATHER(lsbWinnerBin, rowMinDst, gatherIdx, gatherTmp);
    }

    {
        PackedU16Tile msbU(1, 32);
        PackedU16Tile hiU(1, 32);
        PackedU16Tile lsbU(1, 32);
        PackedU16Tile outU(1, 32);
        TASSIGN(msbU, kRemainUbTopk);
        TASSIGN(hiU, kRemainUbSumAbove);
        TASSIGN(lsbU, kRemainUbCw);
        TASSIGN(outU, kRemainUbOut);
        msbWinnerSaved.SetValidRow(1);
        msbWinnerSaved.SetValidCol(1);
        lsbWinnerBin.SetValidRow(1);
        lsbWinnerBin.SetValidCol(1);
        msbU.SetValidRow(1);
        msbU.SetValidCol(1);
        hiU.SetValidRow(1);
        hiU.SetValidCol(1);
        lsbU.SetValidRow(1);
        lsbU.SetValidCol(1);
        outU.SetValidRow(1);
        outU.SetValidCol(1);
        TCVT(msbU, msbWinnerSaved, RoundMode::CAST_TRUNC);
        constexpr uint16_t kShift8 = 8u;
        TSHLS(hiU, msbU, kShift8);
        TCVT(lsbU, lsbWinnerBin, RoundMode::CAST_TRUNC);
        TOR(outU, hiU, lsbU);
        // PIPE_V/PIPE_S sync removed: test whether Phase5 TGATHER vs packedThrU is safe without scalar fence
    }

    TASSIGN(packedThrU, kRemainUbOut);
    packedThrU.SetValidRow(1);
    packedThrU.SetValidCol(1);
}

template <int TopK>
AICORE inline void Phase5_TgatherGtEqTconcatAndStore(__gm__ uint16_t *src, __gm__ uint32_t *outIdx,
                                                     PackedU16Tile &packedThrU)
{
    constexpr int kLoop = (kN + kTileCols - 1) / kTileCols;
    constexpr int cmpVCol = (kTileCols + 7) / 8;
    constexpr int cmpCol = (cmpVCol + 31) / 32 * 32;
    using GatherSrcTile = GatherSrcI16<kTileCols>;
    using GatherDstTile = Tile<TileType::Vec, uint32_t, 1, kTileCols, BLayout::RowMajor, -1, -1>;
    using TmpCmpTile = Tile<TileType::Vec, uint8_t, 1, cmpCol, BLayout::RowMajor, -1, -1>;
    using ConcatTile = GatherConcatCountTile;
    constexpr uint64_t kGatherTileSrcUbBytes = static_cast<uint64_t>(kTileCols) * sizeof(uint16_t);
    constexpr uint64_t kGatherTileDstUbBytes = static_cast<uint64_t>(kTileCols) * sizeof(uint32_t);
    constexpr uint64_t kGatherUbSrcGt = 0x20000;
    constexpr uint64_t kGatherUbSrcEq = kGatherUbSrcGt + kGatherTileSrcUbBytes;
    constexpr uint64_t kChunkGtDst = kGatherUbSrcEq + kGatherTileSrcUbBytes;
    constexpr uint64_t kChunkEqDst = kChunkGtDst + kGatherTileDstUbBytes;
    constexpr uint64_t kChunkConcatGt = kChunkEqDst + kGatherTileDstUbBytes;
    constexpr uint64_t kChunkConcatEq = kChunkConcatGt + 64u;
    constexpr uint64_t kGatherUbTmpGt = 0x21880;
    constexpr uint64_t kGatherUbTmpEq = kGatherUbTmpGt + static_cast<uint64_t>(cmpCol);

    constexpr uint64_t kUbGtSeg = 0x28000;
    constexpr uint64_t kUbEqSeg = kUbGtSeg + static_cast<uint64_t>(TopK) * sizeof(uint32_t);
    constexpr uint64_t kUbSegTmp = kUbEqSeg + static_cast<uint64_t>(TopK) * sizeof(uint32_t);
    constexpr uint64_t kUbMerged = kUbSegTmp + static_cast<uint64_t>(TopK) * sizeof(uint32_t);
    constexpr uint64_t kUbIdxGtAcc = kUbMerged + static_cast<uint64_t>(2 * TopK) * sizeof(uint32_t);
    constexpr uint64_t kUbIdxGtOut = kUbIdxGtAcc + 64u;
    constexpr uint64_t kUbIdxEqAcc = kUbIdxGtOut + 64u;
    constexpr uint64_t kUbIdxEqOut = kUbIdxEqAcc + 64u;

    using SegIdxTile = Tile<TileType::Vec, uint32_t, 1, TopK, BLayout::RowMajor, -1, -1>;
    using MergedIdxTile = Tile<TileType::Vec, uint32_t, 1, 2 * TopK, BLayout::RowMajor, -1, -1>;

    SegIdxTile gtSeg(1, TopK);
    SegIdxTile eqSeg(1, TopK);
    SegIdxTile segTmp(1, TopK);
    MergedIdxTile mergedIdx(1, 2 * TopK);
    GatherConcatCountTile idxGtAcc(1, 1);
    GatherConcatCountTile idxGtOut(1, 1);
    GatherConcatCountTile idxEqAcc(1, 1);
    GatherConcatCountTile idxEqOut(1, 1);

    TASSIGN(gtSeg, kUbGtSeg);
    TASSIGN(eqSeg, kUbEqSeg);
    TASSIGN(segTmp, kUbSegTmp);
    TASSIGN(mergedIdx, kUbMerged);
    TASSIGN(idxGtAcc, kUbIdxGtAcc);
    TASSIGN(idxGtOut, kUbIdxGtOut);
    TASSIGN(idxEqAcc, kUbIdxEqAcc);
    TASSIGN(idxEqOut, kUbIdxEqOut);

    gtSeg.SetValidRow(1);
    gtSeg.SetValidCol(TopK);
    eqSeg.SetValidRow(1);
    eqSeg.SetValidCol(TopK);
    segTmp.SetValidRow(1);
    segTmp.SetValidCol(TopK);
    mergedIdx.SetValidRow(1);
    mergedIdx.SetValidCol(2 * TopK);
    idxGtAcc.SetValidRow(1);
    idxGtAcc.SetValidCol(1);
    idxGtOut.SetValidRow(1);
    idxGtOut.SetValidCol(1);
    idxEqAcc.SetValidRow(1);
    idxEqAcc.SetValidCol(1);
    idxEqOut.SetValidRow(1);
    idxEqOut.SetValidCol(1);

    TEXPANDS(gtSeg, 0u);
    TEXPANDS(eqSeg, 0u);

    __ubuf__ uint32_t *idxGtAccBytes = reinterpret_cast<__ubuf__ uint32_t *>(idxGtAcc.data());
    __ubuf__ uint32_t *idxGtOutBytes = reinterpret_cast<__ubuf__ uint32_t *>(idxGtOut.data());
    __ubuf__ uint32_t *idxEqAccBytes = reinterpret_cast<__ubuf__ uint32_t *>(idxEqAcc.data());
    __ubuf__ uint32_t *idxEqOutBytes = reinterpret_cast<__ubuf__ uint32_t *>(idxEqOut.data());
    idxGtAccBytes[0] = 0u;
    idxEqAccBytes[0] = 0u;

    GatherSrcTile srcG(1, kTileCols);
    GatherDstTile dstG(1, kTileCols);
    ConcatTile concatG(1, 1);
    TmpCmpTile tmpG(1, cmpVCol);
    GatherSrcTile srcE(1, kTileCols);
    GatherDstTile dstE(1, kTileCols);
    ConcatTile concatE(1, 1);
    TmpCmpTile tmpE(1, cmpVCol);

    TASSIGN(srcG, kGatherUbSrcGt);
    TASSIGN(dstG, kChunkGtDst);
    TASSIGN(concatG, kChunkConcatGt);
    TASSIGN(tmpG, kGatherUbTmpGt);
    TASSIGN(srcE, kGatherUbSrcEq);
    TASSIGN(dstE, kChunkEqDst);
    TASSIGN(concatE, kChunkConcatEq);
    TASSIGN(tmpE, kGatherUbTmpEq);

    srcG.SetValidRow(1);
    srcG.SetValidCol(kTileCols);
    dstG.SetValidRow(1);
    dstG.SetValidCol(kTileCols);
    concatG.SetValidRow(1);
    concatG.SetValidCol(1);
    tmpG.SetValidRow(1);
    tmpG.SetValidCol(cmpVCol);
    srcE.SetValidRow(1);
    srcE.SetValidCol(kTileCols);
    dstE.SetValidRow(1);
    dstE.SetValidCol(kTileCols);
    concatE.SetValidRow(1);
    concatE.SetValidCol(1);
    tmpE.SetValidRow(1);
    tmpE.SetValidCol(cmpVCol);
    segTmp.SetValidCol(TopK);

    using TileGm = GlobalTensor<int16_t, pto::Shape<1, 1, 1, 1, kTileCols>,
                                pto::Stride<kTileCols, kTileCols, kTileCols, kTileCols, 1>>;

    for (int i = 0; i < kLoop; ++i) {
        TileGm ggm(reinterpret_cast<__gm__ int16_t *>(src) + i * kTileCols);
        if (i > 0) {
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
        TLOAD(srcG, ggm);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TGATHER<GatherDstTile, GatherSrcTile, PackedU16Tile, ConcatTile, TmpCmpTile, CmpMode::GT>(
            dstG, srcG, packedThrU, concatG, tmpG, i * kTileCols);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TCONCAT_IMPL(segTmp, gtSeg, dstG, idxGtOut, idxGtAcc, concatG);
        TMOV(gtSeg, segTmp);
        TMOV(idxGtAcc, idxGtOut);
    }

    const uint32_t kTopKU = static_cast<uint32_t>(TopK);
    const uint32_t gtCount = idxGtAccBytes[0] / sizeof(uint32_t);
    const uint32_t eqCap = (kTopKU > gtCount) ? (kTopKU - gtCount) : 0u;
    segTmp.SetValidCol(static_cast<int>(eqCap));

    for (int i = 0; i < kLoop; ++i) {
        TileGm egm(reinterpret_cast<__gm__ int16_t *>(src) + i * kTileCols);
        if (i > 0) {
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
        TLOAD(srcE, egm);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TGATHER<GatherDstTile, GatherSrcTile, PackedU16Tile, ConcatTile, TmpCmpTile, CmpMode::EQ>(
            dstE, srcE, packedThrU, concatE, tmpE, i * kTileCols);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TCONCAT_IMPL(segTmp, eqSeg, dstE, idxEqOut, idxEqAcc, concatE);
        TMOV(eqSeg, segTmp);
        TMOV(idxEqAcc, idxEqOut);
    }

    TCONCAT_IMPL(mergedIdx, gtSeg, eqSeg, idxGtAcc, idxEqAcc);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);

    mergedIdx.SetValidRow(1);
    mergedIdx.SetValidCol(TopK);
    using OutShape = pto::Shape<1, 1, 1, 1, TopK>;
    using OutStride = pto::Stride<TopK, TopK, TopK, TopK, 1>;
    GlobalTensor<uint32_t, OutShape, OutStride> outGlobal(outIdx);
    TSTORE(outGlobal, mergedIdx);
}

template <int TopK>
__global__ AICORE void RunRadixTopKDraft(__gm__ uint16_t *src, __gm__ uint32_t *outIdx)
{
    static_assert(TopK > 0, "TopK must be positive.");
    static_assert(TopK <= kN, "TopK cannot exceed N.");

    InTileU16<kTileCols> inTile(1, kTileCols);
    HistTile tileHist(1, kBinNum);
    HistTile chistMSB(1, kBinNum);
    HistTile chistLSB(1, kBinNum);
    IdxFilterTile idxFilter(1, 1);
    WinnerBinTile msbWinnerBin(1, 32);
    RemainKTile remainKTile(1, 32);
    WinnerBinTile msbWinnerSaved(1, 32);
    WinnerBinTile lsbWinnerBin(1, 32);
    PackedU16Tile packedThrU(1, 32);

    Phase1_LoadAndHistogramMsb(src, inTile, tileHist, chistMSB, idxFilter);
    Phase2_WinnerMsbAndRemainK<TopK>(chistMSB, msbWinnerBin, remainKTile, msbWinnerSaved);
    Phase3_HistogramLsb(src, inTile, tileHist, chistLSB, idxFilter, msbWinnerSaved);
    Phase4_WinnerLsbRemainKAndPackedThresholdTor(chistLSB, remainKTile, lsbWinnerBin, msbWinnerSaved, packedThrU);
    Phase5_TgatherGtEqTconcatAndStore<TopK>(src, outIdx, packedThrU);
}

} // namespace topk_radix_detail

template <int TopK>
void LaunchRadixTopKDraft(uint16_t *src, uint32_t *outIdx, void *stream)
{
    topk_radix_detail::RunRadixTopKDraft<TopK><<<1, nullptr, stream>>>(src, outIdx);
}

template void LaunchRadixTopKDraft<512>(uint16_t *src, uint32_t *outIdx, void *stream);
