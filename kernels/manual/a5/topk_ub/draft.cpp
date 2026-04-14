/**
 * Radix-select TopK (2-byte key) for Ascend A5 with pto-isa.
 *
 * This file lives under `kernels/manual/a5/topk_ub/` (local UB-focused variant). The upstream
 * manual scaffold is `kernels/manual/a5/topk/` — keep that directory aligned with cann/pto-isa.
 *
 * Pipeline:
 * 1) THISTOGRAM<true>  (MSB) over full input -> chistMSB
 * 2) Winner MSB: smallest b with C[b] >= (N - TopK). idxFilter / packed MSB use raw min bin (TROWMIN); RemainK uses
 *    WinnerBinU8 (TADDS -1 on b) so TGATHER reads C[winner-1].
 * 3) remainK = (N - TopK) - C[w] with w = post-TADDS bin (same as Python remain_k uses C[winner-1]).
 * 4) THISTOGRAM<false> (LSB, MSB filter) -> chistLSB
 * 5) Winner LSB: TCMPS GE(chistLSB, remainKTile), TCI, TSELS, TROWMIN → lsbWinnerBin
 * 6) GatherCmpToTile (GT then EQ) on full 1×N UB keys; TCONCAT(gtChunk,eqChunk) with counts; TSTORE TopK indices to GM.
 *
 * Notes:
 * - Input keys are uint16 (sortable). TGATHER compare path uses int16_t tiles so the
 *   A5 dispatch selects TGather_b16_gt/eq (see include/pto/npu/a5/TGather.hpp); keys and
 *   threshold share the same bit pattern via reinterpret_cast.
 * - Output index order is unspecified (no sorting required); length is TopK (caller may trim if duplicates in EQ).
 *
 * Current example shape: N = 1 * 2048 keys, TopK = 512 (see scripts/gen_data.py).
 * Input is loaded once to UB (1×N uint16); MSB/LSB histograms and per-chunk TGATHER read that buffer only.
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

constexpr int kN = 2048;
constexpr int kBinNum = 256;
#define PTO_DIV_ROUNDUP(x, y) (((x) + (y)-1) / (y))
#define PTO_CEIL(x, y) (PTO_DIV_ROUNDUP(x, y) * (y))

template <int ValidCols>
using InTileU16 = Tile<TileType::Vec, uint16_t, 1, ValidCols, BLayout::RowMajor, -1, -1>;
using HistTile = Tile<TileType::Vec, uint32_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
// TSELS output: per-bin lane values (same shape as hist); primary output of FindWinnerBucketDescending.
using WinnerLaneTile = Tile<TileType::Vec, uint32_t, 1, kBinNum, BLayout::RowMajor, -1, -1>;
// Broadcast winner value: 1×32 u32 lanes (each = selOut[0] after TGATHER); output of WinnerBinU8FromSelsMin.
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

// Scratch for FindWinnerBucketDescending (TCMPS / TCI / TSELS). Disjoint from 0x10000–0x1C000
// histogram tiles and from 0x20000+ gather scratch used after winner selection.
constexpr uint64_t kWinnerUbMask = 0x23000;
constexpr uint64_t kWinnerUbIdx = 0x23400;
constexpr uint64_t kWinnerUbGather = 0x23800;
constexpr uint64_t kWinnerUbTmp = 0x23C00;
// TROWMIN / TADDS / TCMPS / TSEL scratch (after TSELS; tmp must not alias outLanes at kWinnerUbGather).
constexpr uint64_t kWinnerUbRowMinDst = 0x24000;
constexpr uint64_t kWinnerUbRowMinTmp = 0x24400; // 1×256×4 bytes
constexpr uint64_t kWinnerUbSelMask = 0x24800;   // TCMPS mask (packed), 1×32 u8
constexpr uint64_t kWinnerUbSelZero = 0x24840;   // u32 0 for TSEL src0
constexpr uint64_t kWinnerUbSelOut = 0x24900;    // TSEL dst (1×16 u32)
constexpr uint64_t kWinnerUbTselTmp = 0x24A00;   // TSEL tmp; TGATHER tmp (WinnerBinU8 outBin uses kWinnerUbTmp)
// After TROWMIN..TSEL, reuse kWinnerUbRowMinTmp for TGATHER index tile (1×32 u32).
// MSB remainK tile pipeline: TEXPANDS(TopK), TEXPANDS(N), TGATHER(C[w]), TSUB(N,Cw), TSUB(TopK,sumAbove).
constexpr uint64_t kRemainUbTopk = 0x24E00;
constexpr uint64_t kRemainUbN = 0x24E80;
constexpr uint64_t kRemainUbCw = 0x24F00;
constexpr uint64_t kRemainUbSumAbove = 0x24F80;
constexpr uint64_t kRemainUbOut = 0x25000;
// Copy of msbWinnerBin before LSB WinnerBinU8FromSelsMin overwrites kWinnerUbTmp (128 B, 1×32 u32).
constexpr uint64_t kMsbWinnerSavedUb = 0x24D80;
// Match tests/npu/a5/src/st/testcase/thistogram/thistogram_kernel.cpp idx tile layout.
constexpr uint16_t kIdxAlignedRows = PTO_CEIL(1 * sizeof(uint8_t), BLOCK_BYTE_SIZE);
using IdxFilterTile = Tile<TileType::Vec, uint8_t, kIdxAlignedRows, 1, BLayout::ColMajor, -1, -1>;

template <int ValidCols>
AICORE inline void LoadTileU16(InTileU16<ValidCols> &inTile, __gm__ uint16_t *src, int base, int validCols)
{
    inTile.SetValidCol(validCols);
    using SrcGlobal = GlobalTensor<uint16_t, pto::Shape<1, 1, 1, 1, ValidCols>,
                                   pto::Stride<ValidCols, ValidCols, ValidCols, ValidCols, 1>>;
    SrcGlobal srcGlobal(src + base);
    TLOAD(inTile, srcGlobal);
}

template <int ValidCols>
using GatherSrcI16 = Tile<TileType::Vec, int16_t, 1, ValidCols, BLayout::RowMajor, -1, -1>;

// Keys already in UB at kUbFullKeys (same layout as GM); slice [base, base+validCols) for TGATHER.
constexpr uint64_t kUbFullKeys = 0x00000;

template <int ValidCols>
AICORE inline void AssignGatherChunkUb(GatherSrcI16<ValidCols> &inTile, int base, int validCols)
{
    inTile.SetValidCol(validCols);
    TASSIGN(inTile, kUbFullKeys + static_cast<uint64_t>(base) * sizeof(uint16_t));
}

AICORE inline void ZeroHist(HistTile &dst)
{
    dst.SetValidRow(1);
    dst.SetValidCol(kBinNum);
    TEXPANDS(dst, 0u);
}

// Full-width compare-gather dst (1×kN indices); keys already contiguous in UB at kUbFullKeys.
using GatherFullU32 = Tile<TileType::Vec, uint32_t, 1, kN, BLayout::RowMajor, -1, -1>;
// Must match TGATHER ConcatTile in GatherCmpToTile (count = first u32 at ubConcat).
constexpr int kGatherConcatRows =
    (1 * static_cast<int>(sizeof(uint32_t)) < 32) ? (32 / static_cast<int>(sizeof(uint32_t))) : 1;
using GatherConcatCountTile = Tile<TileType::Vec, uint32_t, kGatherConcatRows, 1, BLayout::ColMajor, -1, -1>;

// THISTOGRAM: ascending cumulative C[b]. Pipeline: TCMPS(GE,thr), TCI, TSELS -> outLanes; thr from caller.
AICORE inline void FindWinnerBucketDescending(HistTile &histTile, uint32_t thr, WinnerLaneTile &outLanes)
{
    MaskCmpTile maskTile(1, kBinNum);
    IdxU32Tile indexTile(1, kBinNum);
    TmpSelsTile tmpSelsTile(1, 32);

    TASSIGN(maskTile, kWinnerUbMask);
    TASSIGN(indexTile, kWinnerUbIdx);
    TASSIGN(outLanes, kWinnerUbGather);
    TASSIGN(tmpSelsTile, kWinnerUbTmp);

    maskTile.SetValidCol(kBinNum);
    indexTile.SetValidCol(kBinNum);
    outLanes.SetValidCol(kBinNum);
    tmpSelsTile.SetValidCol(32);
    histTile.SetValidCol(kBinNum);

    TCMPS(maskTile, histTile, thr, CmpMode::GE);
    TCI<IdxU32Tile, IdxU32Tile, uint32_t, 0>(indexTile, static_cast<uint32_t>(0), indexTile);
    constexpr uint32_t kSelsFalse = 0xffffffffu;
    TSELS(outLanes, maskTile, indexTile, tmpSelsTile, kSelsFalse);
}

// LSB: per-bin C[b] vs remainK (broadcast); GE mask → TCI bin ids → TSELS lanes → caller TROWMIN for min idx.
AICORE inline void LsbHistGeRemainKToLanes(HistTile &chistLSB, RemainKTile &remainKTile, WinnerLaneTile &outLanes)
{
    MaskCmpTile maskTile(1, kBinNum);
    IdxU32Tile indexTile(1, kBinNum);
    TmpSelsTile tmpSelsTile(1, 32);

    TASSIGN(maskTile, kWinnerUbMask);
    TASSIGN(indexTile, kWinnerUbIdx);
    TASSIGN(outLanes, kWinnerUbGather);
    TASSIGN(tmpSelsTile, kWinnerUbTmp);

    maskTile.SetValidCol(kBinNum);
    indexTile.SetValidCol(kBinNum);
    outLanes.SetValidCol(kBinNum);
    tmpSelsTile.SetValidCol(32);
    chistLSB.SetValidCol(kBinNum);
    remainKTile.SetValidRow(1);
    remainKTile.SetValidCol(32);

    TCMPS(maskTile, chistLSB, remainKTile, CmpMode::GT);
    TCI<IdxU32Tile, IdxU32Tile, uint32_t, 0>(indexTile, static_cast<uint32_t>(0), indexTile);
    constexpr uint32_t kSelsFalse = 0xffffffffu;
    TSELS(outLanes, maskTile, indexTile, tmpSelsTile, kSelsFalse);
}

// TROWMIN + TGATHER broadcast. Caller must TASSIGN(outBin, ...) before call (do not force kWinnerUbTmp here).
AICORE inline void WinnerLsbBinU32RowMinBroadcast(WinnerLaneTile &outLanes, WinnerBinTile &outBin)
{
    RowMinDstTile rowMinDst(1, 16);
    RowMinTmpTile rowMinTmp(1, kBinNum);

    TASSIGN(rowMinDst, kWinnerUbRowMinDst);
    TASSIGN(rowMinTmp, kWinnerUbRowMinTmp);

    rowMinDst.SetValidRow(1);
    rowMinDst.SetValidCol(1);
    rowMinTmp.SetValidRow(1);
    rowMinTmp.SetValidCol(kBinNum);

    TROWMIN(rowMinDst, outLanes, rowMinTmp);

    outBin.SetValidRow(1);
    outBin.SetValidCol(32);

    GatherIdxU32Tile gatherIdx(1, 32);
    TmpSelsTile gatherTmp(1, 32);
    TASSIGN(gatherIdx, kWinnerUbRowMinTmp);
    TASSIGN(gatherTmp, kWinnerUbTselTmp);
    gatherIdx.SetValidRow(1);
    gatherIdx.SetValidCol(32);
    gatherTmp.SetValidCol(32);

    TEXPANDS(gatherIdx, 0u);
    // Broadcast rowMinDst[0] → outBin[*] (idx 0 repeats first element).
    TGATHER(outBin, rowMinDst, gatherIdx, gatherTmp);
}

// TROWMIN → TADDS(-1) → TCMPS(GT,256) → TSEL(mask ? 0 : value) → TEXPANDS(0) idx → TGATHER → outBin (u32×32).
// TSEL: mask true (value > 256) → src0 (zero); else → src1 (post-TADDS rowMinDst).
AICORE inline void WinnerBinU8FromSelsMin(WinnerLaneTile &outLanes, WinnerBinTile &outBin)
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

    TROWMIN(rowMinDst, outLanes, rowMinTmp);
    TADDS(rowMinDst, rowMinDst, static_cast<uint32_t>(-1));
    constexpr uint32_t kCmp256 = 256u;
    TCMPS(selMask, rowMinDst, kCmp256, CmpMode::GT);
    TSEL(selOut, selMask, zeroTile, rowMinDst, tselTmp);

    TASSIGN(outBin, kWinnerUbTmp);
    outBin.SetValidRow(1);
    outBin.SetValidCol(32);

    GatherIdxU32Tile gatherIdx(1, 32);
    TmpSelsTile gatherTmp(1, 32);
    TASSIGN(gatherIdx, kWinnerUbRowMinTmp);
    TASSIGN(gatherTmp, kWinnerUbTselTmp);
    gatherIdx.SetValidRow(1);
    gatherIdx.SetValidCol(32);
    gatherTmp.SetValidCol(32);

    TEXPANDS(gatherIdx, 0u);
    TGATHER(outBin, selOut, gatherIdx, gatherTmp);
}

// (msb<<8)|lsb in uint16 bit pattern: TCVT u32→u16 from winner bins, TSHLS<<8 on MSB, TOR with LSB.
// UB: kRemainUbTopk, SumAbove, Cw, Out — only after no more vector use of remainKTile at kRemainUbOut.
AICORE inline uint16_t PackedThresholdU16ViaShlOr(WinnerBinTile &msbWinnerBin, WinnerBinTile &lsbWinnerBin)
{
    PackedU16Tile msbU(1, 32);
    PackedU16Tile hiU(1, 32);
    PackedU16Tile lsbU(1, 32);
    PackedU16Tile outU(1, 32);

    TASSIGN(msbU, kRemainUbTopk);
    TASSIGN(hiU, kRemainUbSumAbove);
    TASSIGN(lsbU, kRemainUbCw);
    TASSIGN(outU, kRemainUbOut);

    msbWinnerBin.SetValidRow(1);
    msbWinnerBin.SetValidCol(1);
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

    TCVT(msbU, msbWinnerBin, RoundMode::CAST_TRUNC);
    constexpr uint16_t kShift8 = 8u;
    TSHLS(hiU, msbU, kShift8);
    TCVT(lsbU, lsbWinnerBin, RoundMode::CAST_TRUNC);
    TOR(outU, hiU, lsbU);
    set_flag(PIPE_V, PIPE_S, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID1);

    __ubuf__ const uint16_t *po = reinterpret_cast<__ubuf__ const uint16_t *>(outU.data());
    return po[0];
}

AICORE inline void LsbWinnerBinFromChistTiles(HistTile &chistLSB, RemainKTile &remainKTile, WinnerBinTile &lsbWinnerBin)
{
    WinnerLaneTile lsbWinnerLanes(1, kBinNum);
    LsbHistGeRemainKToLanes(chistLSB, remainKTile, lsbWinnerLanes);

    TASSIGN(lsbWinnerBin, kWinnerUbTmp);
    lsbWinnerBin.SetValidRow(1);
    lsbWinnerBin.SetValidCol(32);
    WinnerLsbBinU32RowMinBroadcast(lsbWinnerLanes, lsbWinnerBin);
}

// remainK = thr_msb - C[w]; w = TGATHER index from WinnerBinU8FromSelsMin (TROWMIN bin + TADDS(-1)), not raw winner.
template <int TopK>
AICORE inline void RemainKMsbFromTiles(HistTile &chistMSB, WinnerBinTile &winnerBinU32, RemainKTile &remainKTile)
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

    TGATHER(cwT, chistMSB, winnerBinU32, gatherTmp);

    TSUB(remainKTile, thrMsbT, cwT);
}

// THISTOGRAM<false> idx filter: MSB byte = raw winner (TROWMIN min bin), not WinnerBinU8FromSelsMin (-1) tile.
AICORE inline void FillIdxMsbFromWinnerBin(IdxFilterTile &idxTile, WinnerBinTile &winnerMsbRawU32)
{
    idxTile.SetValidRow(1);
    idxTile.SetValidCol(1);
    winnerMsbRawU32.SetValidRow(1);
    winnerMsbRawU32.SetValidCol(1);
    TCVT(idxTile, winnerMsbRawU32, RoundMode::CAST_TRUNC);
}

// Full 1×ValidCols UB slice at gmBase; lane index offset 0. ConcatTile at ubConcat holds match count (TGATHER).
template <CmpMode mode, int ValidCols>
AICORE inline void GatherCmpToTile(uint16_t threshold, int gmBase, int validCols, uint64_t ubDst, uint64_t ubConcat)
{
    using DstTile = Tile<TileType::Vec, uint32_t, 1, ValidCols, BLayout::RowMajor, -1, -1>;
    GatherSrcI16<ValidCols> srcTile(1, validCols);
    DstTile dstTile(1, ValidCols);
    using ConcatTile = GatherConcatCountTile;
    constexpr int cmpVCol = (ValidCols + 7) / 8;
    constexpr int cmpCol = (cmpVCol + 31) / 32 * 32;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, cmpCol, BLayout::RowMajor, -1, -1>;

    ConcatTile concatTile(1, 1);
    TmpTile tmpTile(1, cmpVCol);

    constexpr uint64_t kGatherUbTmp = 0x21880;
    TASSIGN(dstTile, ubDst);
    TASSIGN(concatTile, ubConcat);
    TASSIGN(tmpTile, kGatherUbTmp);

    AssignGatherChunkUb<ValidCols>(srcTile, gmBase, validCols);

    int16_t kBits;
    {
        uint16_t t = threshold;
        kBits = *reinterpret_cast<const int16_t *>(&t);
    }
    if constexpr (mode == CmpMode::GT) {
        TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::GT, 0u>(dstTile, srcTile, kBits,
                                                                                         concatTile, tmpTile);
    } else {
        TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::EQ, 0u>(dstTile, srcTile, kBits,
                                                                                        concatTile, tmpTile);
    }
}

// After MSB cumulative hist: raw MSB → msbWinnerSaved (idx + packed MSB); (-1) bin → msbWinnerBin for RemainK TGATHER.
template <int TopK>
AICORE inline void MsbWinnerRemainKAndSaveTile(HistTile &chistMSB, WinnerBinTile &msbWinnerBin, RemainKTile &remainKTile,
                                               WinnerBinTile &msbWinnerSaved)
{
    constexpr uint32_t kThrMsb = static_cast<uint32_t>(kN - TopK);
    WinnerLaneTile msbWinnerLanes(1, kBinNum);
    FindWinnerBucketDescending(chistMSB, kThrMsb, msbWinnerLanes);

    TASSIGN(msbWinnerSaved, kMsbWinnerSavedUb);
    msbWinnerSaved.SetValidRow(1);
    msbWinnerSaved.SetValidCol(32);
    // Same as LSB path: TROWMIN + broadcast, no TADDS(-1) — raw MSB winner bin for THISTOGRAM<false> / packed high byte.
    WinnerLsbBinU32RowMinBroadcast(msbWinnerLanes, msbWinnerSaved);

    WinnerBinU8FromSelsMin(msbWinnerLanes, msbWinnerBin);
    RemainKMsbFromTiles<TopK>(chistMSB, msbWinnerBin, remainKTile);
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

    TASSIGN(fullInTile, kUbFullKeys);
    TASSIGN(tileHist, 0x10000);
    TASSIGN(chistMSB, 0x14000);
    TASSIGN(chistLSB, 0x18000);
    TASSIGN(idxFilter, 0x1C000);

    // Single GM load of full input; histograms and gather use UB only afterward.
    LoadTileU16(fullInTile, src, 0, kN);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Pass 1: MSB histogram (full width)
    ZeroHist(chistMSB);
    THISTOGRAM<pto::HistByte::BYTE_1>(tileHist, fullInTile, idxFilter);
    TMOV(chistMSB, tileHist);

    WinnerBinTile msbWinnerBin(1, 32);
    RemainKTile remainKTile(1, 32);
    WinnerBinTile msbWinnerSaved(1, 32);
    WinnerBinTile lsbWinnerBin(1, 32);
    MsbWinnerRemainKAndSaveTile<TopK>(chistMSB, msbWinnerBin, remainKTile, msbWinnerSaved);

    // Pass 2: LSB histogram (THISTOGRAM<false>, idx = raw MSB winner byte). Reuse full in-UB buffer.
    FillIdxMsbFromWinnerBin(idxFilter, msbWinnerSaved);
    ZeroHist(chistLSB);
    THISTOGRAM<pto::HistByte::BYTE_0>(tileHist, fullInTile, idxFilter);
    TMOV(chistLSB, tileHist);
    // LSB winner path overwrites kWinnerUbTmp; msbWinnerSaved already holds raw MSB (kMsbWinnerSavedUb) for packedThreshold.
    LsbWinnerBinFromChistTiles(chistLSB, remainKTile, lsbWinnerBin);
    uint16_t packedThreshold = PackedThresholdU16ViaShlOr(msbWinnerSaved, lsbWinnerBin);

    // Compare-gather: full 1×kN TGATHER dst (8 KiB each). Reuse UB after MSB/LSB winner scratch (≥0x23000).
    constexpr uint64_t kFullGatherGtDst = 0x23000;
    constexpr uint64_t kFullGatherEqDst = 0x25000;
    constexpr uint64_t kChunkConcatGt = 0x21000;
    constexpr uint64_t kChunkConcatEq = 0x21040;

    GatherFullU32 gtChunk(1, kN);
    GatherFullU32 eqChunk(1, kN);
    GatherConcatCountTile idxGtCnt(1, 1);
    GatherConcatCountTile idxEqCnt(1, 1);

    TASSIGN(gtChunk, kFullGatherGtDst);
    TASSIGN(eqChunk, kFullGatherEqDst);
    TASSIGN(idxGtCnt, kChunkConcatGt);
    TASSIGN(idxEqCnt, kChunkConcatEq);

    // mergedIdx after reserved 2×TopK u32 at 0x28000 (legacy layout hole).
    constexpr uint64_t kUbMerged =
        static_cast<uint64_t>(0x28000) + static_cast<uint64_t>(2 * TopK) * sizeof(uint32_t);

    using MergedIdxTile = Tile<TileType::Vec, uint32_t, 1, 2 * TopK, BLayout::RowMajor, -1, -1>;

    MergedIdxTile mergedIdx(1, 2 * TopK);

    TASSIGN(mergedIdx, kUbMerged);

    mergedIdx.SetValidRow(1);
    mergedIdx.SetValidCol(2 * TopK);
    idxGtCnt.SetValidRow(1);
    idxGtCnt.SetValidCol(1);
    idxEqCnt.SetValidRow(1);
    idxEqCnt.SetValidCol(1);

    const uint32_t kTopKU = static_cast<uint32_t>(TopK);
    GatherCmpToTile<CmpMode::GT, kN>(packedThreshold, 0, kN, kFullGatherGtDst, kChunkConcatGt);

    GatherCmpToTile<CmpMode::EQ, kN>(packedThreshold, 0, kN, kFullGatherEqDst, kChunkConcatEq);

    TCONCAT_IMPL(mergedIdx, gtChunk, eqChunk, idxGtCnt, idxEqCnt);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);

    mergedIdx.SetValidRow(1);
    mergedIdx.SetValidCol(TopK);
    using OutShape = pto::Shape<1, 1, 1, 1, TopK>;
    using OutStride = pto::Stride<TopK, TopK, TopK, TopK, 1>;
    GlobalTensor<uint32_t, OutShape, OutStride> outGlobal(outIdx);
    TSTORE(outGlobal, mergedIdx);
}

} // namespace topk_radix_detail

template <int TopK>
void LaunchRadixTopKDraft(uint16_t *src, uint32_t *outIdx, void *stream)
{
    topk_radix_detail::RunRadixTopKDraft<TopK><<<1, nullptr, stream>>>(src, outIdx);
}

template void LaunchRadixTopKDraft<512>(uint16_t *src, uint32_t *outIdx, void *stream);
