/**
 * Radix-select TopK (2-byte key) for Ascend A5 with pto-isa.
 *
 * Pipeline:
 * 1) THISTOGRAM<true>  (MSB) over full input -> chistMSB
 * 2) Winner MSB: smallest b with C[b] >= (N - TopK). idxFilter / packed MSB use raw min bin (TROWMIN); RemainK uses
 *    WinnerBinU8 (TADDS -1 on b) so TGATHER reads C[winner-1].
 * 3) remainK = (N - TopK) - C[w] with w = post-TADDS bin (same as Python remain_k uses C[winner-1]).
 * 4) THISTOGRAM<false> (LSB, MSB filter) -> chistLSB
 * 5) Winner LSB: TCMPS GE(chistLSB, remainKTile), TCI, TSELS, TROWMIN → lsbWinnerBin
 * 6) Two passes: each tile GatherCmpToTile → +gmBase; running list appended with scalar UB copy into gtSeg / eqSeg (capped).
 *    Final TCONCAT(gtSeg,eqSeg); TSTORE merged indices to GM.
 *
 * Notes:
 * - Input keys are uint16 (sortable). TGATHER compare path uses int16_t tiles so the
 *   A5 dispatch selects TGather_b16_gt/eq (see include/pto/npu/a5/TGather.hpp); keys and
 *   threshold share the same bit pattern via reinterpret_cast.
 * - Output index order is unspecified (no sorting required); length is TopK (caller may trim if duplicates in EQ).
 *
 * Current example shape: N = 1 * 2048 keys, TopK = 512 (see scripts/gen_data.py).
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
constexpr int kTileCols = 256;
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

template <int ValidCols>
AICORE inline void LoadTileI16Gather(GatherSrcI16<ValidCols> &inTile, __gm__ uint16_t *src, int base)
{
    using SrcGlobal = GlobalTensor<int16_t, pto::Shape<1, 1, 1, 1, ValidCols>,
                                    pto::Stride<ValidCols, ValidCols, ValidCols, ValidCols, 1>>;
    SrcGlobal srcGlobal(reinterpret_cast<__gm__ int16_t *>(src) + base);
    TLOAD(inTile, srcGlobal);
}

AICORE inline void AddHistInUb(HistTile &dst, const HistTile &src) { TADD(dst, dst, src); }

AICORE inline void ZeroHist(HistTile &dst)
{
    // Clear 256 x u32 bins in UB (scalar store).
    __ubuf__ uint32_t *p = reinterpret_cast<__ubuf__ uint32_t *>(dst.data());
    for (int i = 0; i < kBinNum; ++i) {
        p[i] = 0;
    }
}

using GatherChunkU32 = Tile<TileType::Vec, uint32_t, 1, kTileCols, BLayout::RowMajor, -1, -1>;

// Append chunk[0..cnt) after acc[0..accLen); accLen updated; total length capped at cap (scalar UB stores).
AICORE inline void AppendIndicesScalar(__ubuf__ uint32_t *acc, uint32_t &accLen, uint32_t cap,
                                       const __ubuf__ uint32_t *chunk, uint32_t cnt)
{
    if (cap == 0 || cnt == 0) {
        return;
    }
    uint32_t room = (cap > accLen) ? (cap - accLen) : 0u;
    uint32_t take = (cnt < room) ? cnt : room;
    if (take == 0) {
        return;
    }
    for (uint32_t j = 0; j < take; ++j) {
        acc[accLen + j] = chunk[j];
    }
    accLen += take;
}

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
    TCI<IdxU32Tile, uint32_t, 0>(indexTile, static_cast<uint32_t>(0));
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
    TCI<IdxU32Tile, uint32_t, 0>(indexTile, static_cast<uint32_t>(0));
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

// TGATHER offset = tileIndex * ValidCols (template arg must be compile-time; use switch on tile index i).
template <CmpMode mode, int ValidCols>
AICORE inline uint32_t GatherCmpToTile(__gm__ uint16_t *src, uint16_t threshold, int gmBase, int validCols,
                                       uint64_t ubDst, uint64_t ubConcat, int tileIndex)
{
    using DstTile = Tile<TileType::Vec, uint32_t, 1, ValidCols, BLayout::RowMajor, -1, -1>;
    GatherSrcI16<ValidCols> srcTile(1, validCols);
    DstTile dstTile(1, ValidCols);
    constexpr int concatRow =
        (1 * static_cast<int>(sizeof(uint32_t)) < 32) ? (32 / static_cast<int>(sizeof(uint32_t))) : 1;
    using ConcatTile = Tile<TileType::Vec, uint32_t, concatRow, 1, BLayout::ColMajor, -1, -1>;
    constexpr int cmpVCol = (ValidCols + 7) / 8;
    constexpr int cmpCol = (cmpVCol + 31) / 32 * 32;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, cmpCol, BLayout::RowMajor, -1, -1>;

    ConcatTile concatTile(1, 1);
    TmpTile tmpTile(1, cmpVCol);

    constexpr uint64_t kGatherUbSrc = 0x20000;
    constexpr uint64_t kGatherUbTmp = 0x21880;
    TASSIGN(srcTile, kGatherUbSrc);
    TASSIGN(dstTile, ubDst);
    TASSIGN(concatTile, ubConcat);
    TASSIGN(tmpTile, kGatherUbTmp);

    LoadTileI16Gather(srcTile, src, gmBase);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    int16_t kBits;
    {
        uint16_t t = threshold;
        kBits = *reinterpret_cast<const int16_t *>(&t);
    }
    if constexpr (mode == CmpMode::GT) {
        switch (tileIndex) {
        case 0:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::GT, 0u>(dstTile, srcTile, kBits,
                                                                                            concatTile, tmpTile);
            break;
        case 1:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::GT, static_cast<uint32_t>(ValidCols)>(
                dstTile, srcTile, kBits, concatTile, tmpTile);
            break;
        case 2:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::GT,
                    static_cast<uint32_t>(ValidCols * 2)>(dstTile, srcTile, kBits, concatTile, tmpTile);
            break;
        case 3:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::GT,
                    static_cast<uint32_t>(ValidCols * 3)>(dstTile, srcTile, kBits, concatTile, tmpTile);
            break;
        case 4:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::GT,
                    static_cast<uint32_t>(ValidCols * 4)>(dstTile, srcTile, kBits, concatTile, tmpTile);
            break;
        case 5:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::GT,
                    static_cast<uint32_t>(ValidCols * 5)>(dstTile, srcTile, kBits, concatTile, tmpTile);
            break;
        case 6:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::GT,
                    static_cast<uint32_t>(ValidCols * 6)>(dstTile, srcTile, kBits, concatTile, tmpTile);
            break;
        case 7:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::GT,
                    static_cast<uint32_t>(ValidCols * 7)>(dstTile, srcTile, kBits, concatTile, tmpTile);
            break;
        default:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::GT, 0u>(dstTile, srcTile, kBits,
                                                                                            concatTile, tmpTile);
            break;
        }
    } else {
        switch (tileIndex) {
        case 0:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::EQ, 0u>(dstTile, srcTile, kBits,
                                                                                           concatTile, tmpTile);
            break;
        case 1:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::EQ, static_cast<uint32_t>(ValidCols)>(
                dstTile, srcTile, kBits, concatTile, tmpTile);
            break;
        case 2:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::EQ,
                    static_cast<uint32_t>(ValidCols * 2)>(dstTile, srcTile, kBits, concatTile, tmpTile);
            break;
        case 3:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::EQ,
                    static_cast<uint32_t>(ValidCols * 3)>(dstTile, srcTile, kBits, concatTile, tmpTile);
            break;
        case 4:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::EQ,
                    static_cast<uint32_t>(ValidCols * 4)>(dstTile, srcTile, kBits, concatTile, tmpTile);
            break;
        case 5:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::EQ,
                    static_cast<uint32_t>(ValidCols * 5)>(dstTile, srcTile, kBits, concatTile, tmpTile);
            break;
        case 6:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::EQ,
                    static_cast<uint32_t>(ValidCols * 6)>(dstTile, srcTile, kBits, concatTile, tmpTile);
            break;
        case 7:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::EQ,
                    static_cast<uint32_t>(ValidCols * 7)>(dstTile, srcTile, kBits, concatTile, tmpTile);
            break;
        default:
            TGATHER<DstTile, GatherSrcI16<ValidCols>, ConcatTile, TmpTile, CmpMode::EQ, 0u>(dstTile, srcTile, kBits,
                                                                                           concatTile, tmpTile);
            break;
        }
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID1);

    __ubuf__ uint32_t *cntPtr = reinterpret_cast<__ubuf__ uint32_t *>(concatTile.data());
    __ubuf__ uint32_t *dstLane = reinterpret_cast<__ubuf__ uint32_t *>(dstTile.data());
    uint32_t cnt = cntPtr[0];
    if (tileIndex < 0 || tileIndex > 7) {
        const uint32_t add = static_cast<uint32_t>(tileIndex) * static_cast<uint32_t>(ValidCols);
        for (uint32_t j = 0; j < cnt; ++j) {
            dstLane[j] += add;
        }
    }
    return cnt;
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

    constexpr int kLoop = (kN + kTileCols - 1) / kTileCols;

    InTileU16<kTileCols> inTile(1, kTileCols);
    HistTile tileHist(1, kBinNum);
    HistTile chistMSB(1, kBinNum);
    HistTile chistLSB(1, kBinNum);
    IdxFilterTile idxFilter(1, 1);

    TASSIGN(inTile, 0x00000);
    TASSIGN(tileHist, 0x10000);
    TASSIGN(chistMSB, 0x14000);
    TASSIGN(chistLSB, 0x18000);
    TASSIGN(idxFilter, 0x1C000);

    // Pass 1: MSB histogram
    ZeroHist(chistMSB);
    bool firstMsb = true;
    for (int i = 0; i < kLoop; ++i) {
        int base = i * kTileCols;
        int valid = (base + kTileCols <= kN) ? kTileCols : (kN - base);
        if (valid <= 0) {
            break;
        }

        if (i != 0) {
            // Next TLOAD (MTE2) must not overwrite inTile until THISTOGRAM/TMOV/TADD (V) finished.
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        }
        LoadTileU16(inTile, src, base, valid);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        THISTOGRAM<true>(tileHist, inTile, idxFilter);
        if (firstMsb) {
            TMOV(chistMSB, tileHist);
            firstMsb = false;
        } else {
            AddHistInUb(chistMSB, tileHist);
        }
        // Always signal V finished on inTile: next MSB iter waits EVENT_ID2; after last MSB tile,
        // LSB i==0 waits the same so first LSB TLOAD cannot overwrite before last MSB THISTOGRAM completes.
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    }

    WinnerBinTile msbWinnerBin(1, 32);
    RemainKTile remainKTile(1, 32);
    WinnerBinTile msbWinnerSaved(1, 32);
    WinnerBinTile lsbWinnerBin(1, 32);
    MsbWinnerRemainKAndSaveTile<TopK>(chistMSB, msbWinnerBin, remainKTile, msbWinnerSaved);

    // Pass 2: LSB histogram (THISTOGRAM<false>, idx = raw MSB winner byte). LsbWinner: GE(chist, remainK) + ...
    FillIdxMsbFromWinnerBin(idxFilter, msbWinnerSaved);
    ZeroHist(chistLSB);
    bool firstLsb = true;
    for (int i = 0; i < kLoop; ++i) {
        int base = i * kTileCols;
        int valid = (base + kTileCols <= kN) ? kTileCols : (kN - base);
        if (valid <= 0) {
            break;
        }

        // Same EVENT_ID2 ping-pong as MSB: i==0 waits MSB last tile's set; i>0 waits previous LSB tile.
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        LoadTileU16(inTile, src, base, valid);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        THISTOGRAM<false>(tileHist, inTile, idxFilter);
        if (firstLsb) {
            TMOV(chistLSB, tileHist);
            firstLsb = false;
        } else {
            AddHistInUb(chistLSB, tileHist);
        }
        if (i + 1 < kLoop) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        }
    }
    // LSB winner path overwrites kWinnerUbTmp; msbWinnerSaved already holds raw MSB (kMsbWinnerSavedUb) for packedThreshold.
    LsbWinnerBinFromChistTiles(chistLSB, remainKTile, lsbWinnerBin);
    uint16_t packedThreshold = PackedThresholdU16ViaShlOr(msbWinnerSaved, lsbWinnerBin);

    // Compare-gather scratch: separate UB for GT vs EQ dst/concat (GatherCmpToTile).
    constexpr uint64_t kChunkGtDst = 0x20800;
    constexpr uint64_t kChunkEqDst = 0x20C00;
    constexpr uint64_t kChunkConcatGt = 0x21000;
    constexpr uint64_t kChunkConcatEq = 0x21040;

    using IdxCountTile = Tile<TileType::Vec, uint32_t, 1, 32, BLayout::RowMajor, -1, -1>;

    GatherChunkU32 gtChunk(1, kTileCols);
    GatherChunkU32 eqChunk(1, kTileCols);

    TASSIGN(gtChunk, kChunkGtDst);
    TASSIGN(eqChunk, kChunkEqDst);

    gtChunk.SetValidRow(1);
    gtChunk.SetValidCol(kTileCols);
    eqChunk.SetValidRow(1);
    eqChunk.SetValidCol(kTileCols);

    // UB layout (post-histogram): GT segment, EQ segment, merged idx, TCONCAT counts.
    constexpr uint64_t kUbGtSeg = 0x28000;
    constexpr uint64_t kUbEqSeg = kUbGtSeg + static_cast<uint64_t>(TopK) * sizeof(uint32_t);
    constexpr uint64_t kUbMerged = kUbEqSeg + static_cast<uint64_t>(TopK) * sizeof(uint32_t);
    constexpr uint64_t kUbIdxConcat0 = kUbMerged + static_cast<uint64_t>(2 * TopK) * sizeof(uint32_t);
    constexpr uint64_t kUbIdxConcat1 = kUbIdxConcat0 + 128u;

    using SegIdxTile = Tile<TileType::Vec, uint32_t, 1, TopK, BLayout::RowMajor, -1, -1>;
    using MergedIdxTile = Tile<TileType::Vec, uint32_t, 1, 2 * TopK, BLayout::RowMajor, -1, -1>;

    SegIdxTile gtSeg(1, TopK);
    SegIdxTile eqSeg(1, TopK);
    MergedIdxTile mergedIdx(1, 2 * TopK);
    IdxCountTile idxGtCnt(1, 32);
    IdxCountTile idxEqCnt(1, 32);

    TASSIGN(gtSeg, kUbGtSeg);
    TASSIGN(eqSeg, kUbEqSeg);
    TASSIGN(mergedIdx, kUbMerged);
    TASSIGN(idxGtCnt, kUbIdxConcat0);
    TASSIGN(idxEqCnt, kUbIdxConcat1);

    gtSeg.SetValidRow(1);
    gtSeg.SetValidCol(TopK);
    eqSeg.SetValidRow(1);
    eqSeg.SetValidCol(TopK);
    mergedIdx.SetValidRow(1);
    mergedIdx.SetValidCol(2 * TopK);
    idxGtCnt.SetValidRow(1);
    idxGtCnt.SetValidCol(1);
    idxEqCnt.SetValidRow(1);
    idxEqCnt.SetValidCol(1);

    TEXPANDS(gtSeg, 0u);
    TEXPANDS(eqSeg, 0u);
    __ubuf__ uint32_t *gtPtr = reinterpret_cast<__ubuf__ uint32_t *>(gtSeg.data());
    __ubuf__ uint32_t *eqPtr = reinterpret_cast<__ubuf__ uint32_t *>(eqSeg.data());
    __ubuf__ uint32_t *gtLane = reinterpret_cast<__ubuf__ uint32_t *>(gtChunk.data());
    __ubuf__ uint32_t *eqLane = reinterpret_cast<__ubuf__ uint32_t *>(eqChunk.data());

    const uint32_t kTopKU = static_cast<uint32_t>(TopK);
    uint32_t gtCount = 0;
    for (int i = 0; i < kLoop; ++i) {
        int base = i * kTileCols;
        int valid = (base + kTileCols <= kN) ? kTileCols : (kN - base);
        if (valid <= 0) {
            break;
        }
        uint32_t cntG =
            GatherCmpToTile<CmpMode::GT, kTileCols>(src, packedThreshold, base, valid, kChunkGtDst, kChunkConcatGt, i);
        AppendIndicesScalar(gtPtr, gtCount, kTopKU, gtLane, cntG);
    }

    const uint32_t eqCap = (kTopKU > gtCount) ? (kTopKU - gtCount) : 0u;
    uint32_t eqCount = 0;
    for (int i = 0; i < kLoop; ++i) {
        int base = i * kTileCols;
        int valid = (base + kTileCols <= kN) ? kTileCols : (kN - base);
        if (valid <= 0) {
            break;
        }
        uint32_t cntE =
            GatherCmpToTile<CmpMode::EQ, kTileCols>(src, packedThreshold, base, valid, kChunkEqDst, kChunkConcatEq, i);
        AppendIndicesScalar(eqPtr, eqCount, eqCap, eqLane, cntE);
    }

    __ubuf__ uint32_t *ig = reinterpret_cast<__ubuf__ uint32_t *>(idxGtCnt.data());
    __ubuf__ uint32_t *ie = reinterpret_cast<__ubuf__ uint32_t *>(idxEqCnt.data());
    ig[0] = gtCount;
    ie[0] = eqCount;

    TCONCAT_IMPL(mergedIdx, gtSeg, eqSeg, idxGtCnt, idxEqCnt);
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
