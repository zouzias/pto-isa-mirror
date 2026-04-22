/**
 * Radix-select TopK (2-byte key) for Ascend A5 with pto-isa.
 *
 * This file lives under `kernels/manual/a5/topk_ub/` (local UB-focused variant). The upstream
 * manual scaffold is `kernels/manual/a5/topk/` — keep that directory aligned with cann/pto-isa.
 *
 * All pto-isa ops (including `TASSIGN` / `TLOAD`) live only in five `Phase*` functions — no nested helpers inside them.
 * `RunRadixTopKDraft` only constructs tile objects and calls the phases in order.
 * 1) **`TASSIGN` UB + `TLOAD` + Histogram (MSB)** — `THISTOGRAM<BYTE_1>` → `chistMSB`.
 * 2) **Winner MSB + remainK** — `TCMPS`/`TCI`/`TSELS`, raw MSB broadcast, `WinnerBinU8` path, `TGATHER`+`TSUB` remainK.
 * 3) **Histogram (LSB)** — `TCVT` idx from saved MSB; `THISTOGRAM<BYTE_0>` → `chistLSB`.
 * 4) **Winner LSB + remainK + packed threshold** — `TCMPS`/`TSELS`/`TROWMIN`/`TGATHER`; `TCVT`/`TSHLS`/`TOR` for compare key.
 * 5) **Two full-width `TGATHER` (GT/EQ) + 五參數 `TCONCAT_IMPL`（`TConcatIdx`）+ `TSTORE`**.
 *
 * Notes:
 * - ISA/pto 名稱為 **`TGATHER`**（模擬器 log 中常見 `VGATHER` 類指令為其下層實現）。
 * - Input keys are uint16 (sortable). TGATHER compare path uses int16_t tiles so the
 *   A5 dispatch selects TGather_b16_gt/eq (see include/pto/npu/a5/TGather.hpp); keys and
 *   threshold share the same bit pattern via reinterpret_cast.
 * - Output index order is unspecified (no sorting required); length is TopK (caller may trim if duplicates in EQ).
 *
 * Current example shape: N = 65536 (64K) keys, TopK = 512 (see scripts/gen_data.py).
 * Input is loaded once to UB (1×N uint16); MSB/LSB histograms and full-width TGATHER read that buffer only.
 * UB layout uses ~0xC01000 bytes (keys 128 KiB + GT/EQ index buffers 256 KiB each + merged); verify A5 UB limits.
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

constexpr int kN = 65536;
constexpr int kBinNum = 256;
#define PTO_DIV_ROUNDUP(x, y) (((x) + (y)-1) / (y))
#define PTO_CEIL(x, y) (PTO_DIV_ROUNDUP(x, y) * (y))

template <int ValidCols>
using InTileU16 = Tile<TileType::Vec, uint16_t, 1, ValidCols, BLayout::RowMajor, -1, -1>;
using HistTile = Tile<TileType::Vec, uint32_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
// TSELS output: per-bin lane values (same shape as hist).
using WinnerLaneTile = Tile<TileType::Vec, uint32_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
// Broadcast winner: 1×32 u32 lanes (TGATHER from row min / TSEL path).
using WinnerBinTile = Tile<TileType::Vec, uint32_t, 1, 32, BLayout::RowMajor, -1, -1>;
using MaskCmpTile = Tile<TileType::Vec, uint8_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
using IdxU32Tile = Tile<TileType::Vec, uint32_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
using TmpSelsTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
// TROWMIN dst / TSEL src: 1×16 u32 (32B row align); valid col 1. Tmp: same shape as WinnerLaneTile.
using RowMinDstTile = Tile<TileType::Vec, uint32_t, 1, 16, BLayout::RowMajor, -1, -1>;
using RowMinTmpTile = Tile<TileType::Vec, uint32_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
using SelMaskRowTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
// TGATHER(selOut, idx): idx all 0 → dst lanes = selOut[0].
using GatherIdxU32Tile = Tile<TileType::Vec, uint32_t, 1, 32, BLayout::RowMajor, -1, -1>;
using RemainKTile = Tile<TileType::Vec, uint32_t, 1, 32, BLayout::RowMajor, -1, -1>;
// packedThreshold = (msb<<8)|lsb via TSHL + TOR (reuse kRemainUb* after remainKTile V ops are done).
using PackedU16Tile = Tile<TileType::Vec, uint16_t, 1, 32, BLayout::RowMajor, -1, -1>;

// Histogram tiles sit after full keys [0, kN*sizeof(uint16)) = [0, 0x20000). Winner scratch starts at 0x24000
// (past idxFilter at 0x23000). Gather GT/EQ buffers are high UB (0x40000 / 0x80000) — see RunRadixTopKDraft.
constexpr uint64_t kUbTileHist = 0x20000;
constexpr uint64_t kUbChistMSB = 0x21000;
constexpr uint64_t kUbChistLSB = 0x22000;
constexpr uint64_t kUbIdxFilter = 0x23000;
// Scratch for MSB/LSB winner lanes (TCMPS / TCI / TSELS).
constexpr uint64_t kWinnerUbMask = 0x24000;
constexpr uint64_t kWinnerUbIdx = 0x24400;
constexpr uint64_t kWinnerUbGather = 0x24800;
constexpr uint64_t kWinnerUbTmp = 0x24C00;
// TROWMIN / TADDS / TCMPS / TSEL scratch (after TSELS; tmp must not alias outLanes at kWinnerUbGather).
constexpr uint64_t kWinnerUbRowMinDst = 0x25000;
constexpr uint64_t kWinnerUbRowMinTmp = 0x25400; // 1×256×4 bytes
constexpr uint64_t kWinnerUbSelMask = 0x25800;   // TCMPS mask (packed), 1×32 u8
constexpr uint64_t kWinnerUbSelZero = 0x25840;   // u32 0 for TSEL src0
constexpr uint64_t kWinnerUbU32One = 0x25880;    // 1×16 u32 scratch: scalar 1 for TSUB (avoid TADDS on u32 in older CANN)
constexpr uint64_t kWinnerUbSelOut = 0x25900;    // TSEL dst (1×16 u32)
constexpr uint64_t kWinnerUbTselTmp = 0x25A00;   // TSEL tmp; TGATHER tmp (WinnerBinU8 outBin uses kWinnerUbTmp)
// After TROWMIN..TSEL, reuse kWinnerUbRowMinTmp for TGATHER index tile (1×32 u32).
// MSB remainK tile pipeline: TEXPANDS(TopK), TEXPANDS(N), TGATHER(C[w]), TSUB(N,Cw), TSUB(TopK,sumAbove).
constexpr uint64_t kRemainUbTopk = 0x25E00;
constexpr uint64_t kRemainUbN = 0x25E80;
constexpr uint64_t kRemainUbCw = 0x25F00;
constexpr uint64_t kRemainUbSumAbove = 0x25F80;
constexpr uint64_t kRemainUbOut = 0x26000;
// Copy of msbWinnerBin before LSB WinnerBinU8FromSelsMin overwrites kWinnerUbTmp (128 B, 1×32 u32).
constexpr uint64_t kMsbWinnerSavedUb = 0x25D80;
// Match tests/npu/a5/src/st/testcase/thistogram/thistogram_kernel.cpp idx tile layout.
constexpr uint16_t kIdxAlignedRows = PTO_CEIL(1 * sizeof(uint8_t), BLOCK_BYTE_SIZE);
using IdxFilterTile = Tile<TileType::Vec, uint8_t, kIdxAlignedRows, 1, BLayout::ColMajor, -1, -1>;

template <int ValidCols>
using GatherSrcI16 = Tile<TileType::Vec, int16_t, 1, ValidCols, BLayout::RowMajor, -1, -1>;

// Keys already in UB at kUbFullKeys (same layout as GM); slice [base, base+validCols) for TGATHER.
constexpr uint64_t kUbFullKeys = 0x00000;

// Full-width compare-gather dst (1×kN indices); keys already contiguous in UB at kUbFullKeys.
using GatherFullU32 = Tile<TileType::Vec, uint32_t, 1, kN, BLayout::RowMajor, -1, -1>;
constexpr int kGatherConcatRows =
    (1 * static_cast<int>(sizeof(uint32_t)) < 32) ? (32 / static_cast<int>(sizeof(uint32_t))) : 1;
using GatherConcatCountTile = Tile<TileType::Vec, uint32_t, kGatherConcatRows, 1, BLayout::ColMajor, -1, -1>;

// --- Five phases: every TLOAD / THISTOGRAM / TCMPS / TGATHER / … appears only below (no callees). ---

AICORE inline void Phase1_LoadAndHistogramMsb(__gm__ uint16_t *src, InTileU16<kN> &fullInTile, HistTile &tileHist,
                                                HistTile &chistMSB, HistTile &chistLSB, IdxFilterTile &idxFilter)
{
    TASSIGN(fullInTile, kUbFullKeys);
    TASSIGN(tileHist, kUbTileHist);
    TASSIGN(chistMSB, kUbChistMSB);
    TASSIGN(chistLSB, kUbChistLSB);
    TASSIGN(idxFilter, kUbIdxFilter);

    fullInTile.SetValidCol(kN);
    using SrcGlobal = GlobalTensor<uint16_t, pto::Shape<1, 1, 1, 1, kN>, pto::Stride<kN, kN, kN, kN, 1>>;
    SrcGlobal srcGlobal(src);
    TLOAD(fullInTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    chistMSB.SetValidRow(1);
    chistMSB.SetValidCol(kBinNum);
    TEXPANDS(chistMSB, 0u);
    THISTOGRAM<pto::HistByte::BYTE_1>(tileHist, fullInTile, idxFilter);
    TMOV(chistMSB, tileHist);
}

template <int TopK>
AICORE inline void Phase2_WinnerMsbAndRemainK(HistTile &chistMSB, WinnerBinTile &msbWinnerBin, RemainKTile &remainKTile,
                                               WinnerBinTile &msbWinnerSaved)
{
    constexpr uint32_t kThrMsb = static_cast<uint32_t>(kN - TopK);
    constexpr uint32_t kSelsFalse = 0xffffffffu;
    WinnerLaneTile msbWinnerLanes(1, kBinNum);

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
        TmpSelsTile gatherTmp(1, 32);
        TASSIGN(thrMsbT, kRemainUbTopk);
        TASSIGN(cwT, kRemainUbCw);
        TASSIGN(remainKTile, kRemainUbOut);
        TASSIGN(gatherTmp, kWinnerUbRowMinTmp);
        thrMsbT.SetValidRow(1);
        thrMsbT.SetValidCol(32);
        cwT.SetValidRow(1);
        cwT.SetValidCol(32);
        remainKTile.SetValidRow(1);
        remainKTile.SetValidCol(32);
        gatherTmp.SetValidCol(32);
        constexpr uint32_t kThrMsbU = static_cast<uint32_t>(kN - TopK);
        TEXPANDS(thrMsbT, kThrMsbU);
        TGATHER(cwT, chistMSB, msbWinnerBin, gatherTmp);
        TSUB(remainKTile, thrMsbT, cwT);
    }
}

AICORE inline void Phase3_HistogramLsb(InTileU16<kN> &fullInTile, HistTile &tileHist, HistTile &chistLSB,
                                       IdxFilterTile &idxFilter, WinnerBinTile &msbWinnerSaved)
{
    idxFilter.SetValidRow(1);
    idxFilter.SetValidCol(1);
    msbWinnerSaved.SetValidRow(1);
    msbWinnerSaved.SetValidCol(1);
    TCVT(idxFilter, msbWinnerSaved, RoundMode::CAST_TRUNC);

    chistLSB.SetValidRow(1);
    chistLSB.SetValidCol(kBinNum);
    TEXPANDS(chistLSB, 0u);
    THISTOGRAM<pto::HistByte::BYTE_0>(tileHist, fullInTile, idxFilter);
    TMOV(chistLSB, tileHist);
}

AICORE inline void Phase4_WinnerLsbRemainKAndPackedThresholdTor(HistTile &chistLSB, RemainKTile &remainKTile,
                                                                  WinnerBinTile &lsbWinnerBin,
                                                                  WinnerBinTile &msbWinnerSaved, PackedU16Tile &outU)
{
    constexpr uint32_t kSelsFalse = 0xffffffffu;
    WinnerLaneTile lsbWinnerLanes(1, kBinNum);
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
    TSELS(lsbWinnerLanes, maskTile, indexTile, tmpSelsTile, kSelsFalse);

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

    PackedU16Tile msbU(1, 32);
    PackedU16Tile hiU(1, 32);
    PackedU16Tile lsbU(1, 32);
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
    set_flag(PIPE_V, PIPE_S, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
}

template <int TopK>
AICORE inline void Phase5_TgatherGtEqTconcatAndStore(PackedU16Tile &packedThrU, __gm__ uint32_t *outIdx)
{
    constexpr uint64_t kFullGatherGtDst = 0x30000;
    constexpr uint64_t kFullGatherEqDst = 0x38000;
    constexpr uint64_t kChunkConcatGt = 0x28000;
    constexpr uint64_t kChunkConcatEq = 0x28040;
    constexpr uint64_t kUbMerged = 0x0;
    constexpr uint64_t kGatherUbTmp = 0x29000;
    constexpr int cmpVCol = (kN + 7) / 8;
    constexpr int cmpCol = (cmpVCol + 31) / 32 * 32;
    using DstTile = Tile<TileType::Vec, uint32_t, 1, kN, BLayout::RowMajor, -1, -1>;
    using TmpGatherTile = Tile<TileType::Vec, uint8_t, 1, cmpCol, BLayout::RowMajor, -1, -1>;

    GatherFullU32 gtChunk(1, kN);
    GatherFullU32 eqChunk(1, kN);
    GatherConcatCountTile idxGtCnt(1, 1);
    GatherConcatCountTile idxEqCnt(1, 1);
    TASSIGN(gtChunk, kFullGatherGtDst);
    TASSIGN(eqChunk, kFullGatherEqDst);
    TASSIGN(idxGtCnt, kChunkConcatGt);
    TASSIGN(idxEqCnt, kChunkConcatEq);

    using MergedIdxTile = Tile<TileType::Vec, uint32_t, 1, 2 * TopK, BLayout::RowMajor, -1, -1>;
    MergedIdxTile mergedIdx(1, 2 * TopK);
    TASSIGN(mergedIdx, kUbMerged);
    mergedIdx.SetValidRow(1);
    mergedIdx.SetValidCol(2 * TopK);
    idxGtCnt.SetValidRow(1);
    idxGtCnt.SetValidCol(1);
    idxEqCnt.SetValidRow(1);
    idxEqCnt.SetValidCol(1);

    GatherSrcI16<kN> srcGt(1, kN);
    TmpGatherTile tmpGt(1, cmpVCol);
    TASSIGN(tmpGt, kGatherUbTmp);
    srcGt.SetValidCol(kN);
    TASSIGN(srcGt, kUbFullKeys);
    TGATHER<DstTile, GatherSrcI16<kN>, PackedU16Tile, GatherConcatCountTile, TmpGatherTile, CmpMode::GT>(
        gtChunk, srcGt, packedThrU, idxGtCnt, tmpGt, 0);

    GatherSrcI16<kN> srcEq(1, kN);
    TmpGatherTile tmpEq(1, cmpVCol);
    TASSIGN(tmpEq, kGatherUbTmp);
    srcEq.SetValidCol(kN);
    TASSIGN(srcEq, kUbFullKeys);
    TGATHER<DstTile, GatherSrcI16<kN>, PackedU16Tile, GatherConcatCountTile, TmpGatherTile, CmpMode::EQ>(
        eqChunk, srcEq, packedThrU, idxEqCnt, tmpEq, 0);

    TCONCAT_IMPL(mergedIdx, gtChunk, eqChunk, idxGtCnt, idxEqCnt);

    mergedIdx.SetValidCol(TopK);
    using OutShape = pto::Shape<1, 1, 1, 1, TopK>;
    using OutStride = pto::Stride<TopK, TopK, TopK, TopK, 1>;
    GlobalTensor<uint32_t, OutShape, OutStride> outGlobal(outIdx);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(outGlobal, mergedIdx);
}

template <int TopK>
__global__ AICORE void RunRadixTopKDraft(__gm__ uint16_t *src, __gm__ uint32_t *outIdx)
{
    static_assert(TopK > 0, "TopK must be positive.");
    static_assert(TopK <= kN, "TopK cannot exceed N.");

    InTileU16<kN> fullInTile(1, kN);
    HistTile tileHist(1, kBinNum);
    HistTile chistMSB(1, kBinNum);
    HistTile chistLSB(1, kBinNum);
    IdxFilterTile idxFilter(1, 1);

    Phase1_LoadAndHistogramMsb(src, fullInTile, tileHist, chistMSB, chistLSB, idxFilter);

    WinnerBinTile msbWinnerBin(1, 32);
    RemainKTile remainKTile(1, 32);
    WinnerBinTile msbWinnerSaved(1, 32);
    WinnerBinTile lsbWinnerBin(1, 32);
    Phase2_WinnerMsbAndRemainK<TopK>(chistMSB, msbWinnerBin, remainKTile, msbWinnerSaved);

    Phase3_HistogramLsb(fullInTile, tileHist, chistLSB, idxFilter, msbWinnerSaved);

    PackedU16Tile packedThrU(1, 32);
    TASSIGN(packedThrU, kRemainUbOut);
    Phase4_WinnerLsbRemainKAndPackedThresholdTor(chistLSB, remainKTile, lsbWinnerBin, msbWinnerSaved, packedThrU);

    Phase5_TgatherGtEqTconcatAndStore<TopK>(packedThrU, outIdx);
}

} // namespace topk_radix_detail

template <int TopK>
void LaunchRadixTopKDraft(uint16_t *src, uint32_t *outIdx, void *stream)
{
    topk_radix_detail::RunRadixTopKDraft<TopK><<<1, nullptr, stream>>>(src, outIdx);
}

template void LaunchRadixTopKDraft<512>(uint16_t *src, uint32_t *outIdx, void *stream);
