/**
 * topk_kernel.cpp - auto-mode A3 prototype, values-only top-K via merge sort.
 *
 * Adapted (with minimal edits) from
 *   tests/npu/a2a3/src/st/testcase/tmrgsort/tmrgsort_kernel.cpp
 *
 * specifically the `RunTMrgsortTopk<float, 1, 1280, 1, 1280, 512>` shape,
 * which is in `ALL_TESTCASES` and known to build under auto mode. We
 * extract just that function plus its two helpers (FillMrgArray and
 * SortTailBlock) into a standalone project, and wrap it in a topk-style
 * launchTopk(uint8_t*, ...) host wrapper so main.cpp can use the same
 * tests/common/test_common.h harness as the add_tile_array project.
 *
 * v1 limitations (deliberate; documented in README.md):
 *   - Values-only top-K. Indices are NOT produced. Adding indices requires
 *     prepending TSORT32 (which emits the (val, idx) packed format) plus a
 *     TGATHER pair to extract value/index slots — that is a v2 task.
 *   - Input must be pre-sorted in 64-block descending order (the Python
 *     reference does this; in v2 the in-kernel TSORT32 will replace this
 *     precondition).
 *   - Single AICORE (`<<<1, nullptr, stream>>>`); no `block_idx` work
 *     distribution.
 *   - No double-buffering, no TPipe/TPUSH/TPOP, no manual ping-pong.
 *   - No set_flag/wait_flag/pipe_barrier at kernel scope (auto-sync
 *     handles ordering); the existing `#ifndef __PTO_AUTO__` guards in
 *     the source are preserved in case the file is later compiled in
 *     manual mode for cross-checking.
 */

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <int kTCols_>
PTO_INTERNAL int32_t FillMrgArray(int32_t *mrgArray, int blockLen)
{
    int32_t arrayCount = 0;
    int32_t tmpInner = kTCols_;
    for (int32_t i = blockLen; i >= 64; i /= 4) {
        int32_t count;
        for (count = 0; count < tmpInner / i; count++) {
            mrgArray[arrayCount++] = i;
        }
        tmpInner -= count * i;
    }
    return arrayCount;
}

template <typename GlobalData, typename DstGlobalData, typename DstTileData, typename TileData, typename TmpTileData,
          typename T, int kTCols_, int topk>
PTO_INTERNAL void SortTailBlock(DstGlobalData &dstGlobal, DstTileData &dstTile, TileData &srcTile, int blockLen)
{
    // tmp1Tile is independent scratch for TMRGSORT. No TASSIGN: this file is
    // auto-mode-only and the auto allocator owns placement.
    TmpTileData tmp1Tile(1, kTCols_);

    int32_t mrgArray[15] = {0};
    int32_t arrayCount = FillMrgArray<kTCols_>(mrgArray, blockLen);
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

        TileData src0Tile(1, tmpMrgSortedLen);
        TileData src1Tile(1, tmpMrgArray);
        // Semantic aliasing: src0Tile is the (tmpMrgSortedLen)-wide prefix view
        // of srcTile (the already-merged region from RunTopk's main loop);
        // src1Tile is the (tmpMrgArray)-wide TSUBVIEW at offset mrgSortedLen
        // (the next tail run). srcTile owns the data on loop entry.
        TRESHAPE(src0Tile, srcTile);
        TSUBVIEW(src1Tile, srcTile, 0, mrgSortedLen);

        // Independent destination — NOT aliased to srcTile.
        // The previous attempt aliased the merge destination back onto
        // srcTile (via TRESHAPE(curDstTile, srcTile)) so TMRGSORT would
        // write in place. That produced an interleaved output (top values
        // at even positions, lower values at odd positions): a likely
        // read/write aliasing hazard between TMRGSORT's reads of src0Tile /
        // src1Tile (both views of srcTile) and its writes to a destination
        // that is also a view of srcTile with overlapping range.
        // Using independent storage for the merge destination avoids the
        // overlap; we then promote the topK prefix back via TMOV.
        // tailDstTile owns the merged data after TMRGSORT returns.
        TileData tailDstTile(1, tmpMrgSortedLen + tmpMrgArray);
        TMRGSORT<TileData, TmpTileData, TileData, TileData, 0>(tailDstTile, executedNumList, tmp1Tile, src0Tile,
                                                                src1Tile);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif
        // Semantic aliasing: tailTopKTile is the (topK)-wide prefix view of
        // tailDstTile (which holds the merged descending run). tailDstTile
        // owns the data; tailTopKTile is a smaller-valid view used only as
        // the source of the TMOV below — no lifetime conflict.
        TileData tailTopKTile(1, topk);
        TRESHAPE(tailTopKTile, tailDstTile);
        // Promote the topK prefix into dstTile. dstTile aliases srcTile's
        // prefix (via the outer TRESHAPE in RunTopk), so this TMOV also
        // updates srcTile[0..topK-1] for any subsequent loop iterations
        // that re-derive src0Tile from srcTile. The final TSTORE after the
        // loop reads dstTile's topK and writes it to dstGlobal.
        TMOV(dstTile, tailTopKTile);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int topk>
__global__ AICORE void RunTopk(__gm__ uint8_t *out_raw, __gm__ uint8_t *src_raw)
{
    // Host launchers cannot apply the __gm__ qualifier via reinterpret_cast,
    // so the kernel entry takes raw __gm__ uint8_t* and casts inside the
    // kernel body where __gm__ is a valid type qualifier
    // (see docs_for_ai/compile_error_logbook.md §E11).
    __gm__ T *out = reinterpret_cast<__gm__ T *>(out_raw);
    __gm__ T *src = reinterpret_cast<__gm__ T *>(src_raw);

    using GlobalData = GlobalTensor<T, Shape<1, 1, 1, kTRows_, kTCols_>, pto::Stride<1, 1, 1, kGCols_, 1>>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstGlobalData = GlobalTensor<T, Shape<1, 1, 1, kTRows_, topk>, pto::Stride<1, 1, 1, kGCols_, 1>>;
    using DstTileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;

    // srcTile and tmpTile are independent storage; the auto allocator owns
    // placement. No TASSIGN here — this file is auto-mode-only.
    TileData srcTile(1, kTCols_);
    DstTileData dstTile(1, topk);
    TmpTileData tmpTile(1, kTCols_);
    // Semantic aliasing: dstTile is the (topK)-wide prefix view of srcTile.
    // After in-place merge sort, srcTile[0..topK-1] is the descending top-K;
    // dstTile is the tile shape that matches the GM output stride for TSTORE.
    // NOT memory reuse — srcTile owns the data; dstTile is a smaller-valid
    // view of the same buffer.
    TRESHAPE(dstTile, srcTile);

    GlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    uint32_t blockLen = 64;
    // Merge sort data for every 4 blockLen lengths.
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    for (; blockLen * 4 <= kTCols_; blockLen *= 4) {
        uint16_t cols = kTCols_ / (blockLen * 4) * (blockLen * 4);
        TileData srcSortedTile(1, cols);
        TmpTileData tmpSortedTile(1, cols);
        TRESHAPE(srcSortedTile, srcTile);
        TRESHAPE(tmpSortedTile, tmpTile);
        TMRGSORT<TmpTileData, TileData>(tmpSortedTile, srcSortedTile, blockLen);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif
        TMOV(srcSortedTile, tmpSortedTile);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif
    }

    // sort tail block
    if (blockLen < kTCols_) {
        SortTailBlock<GlobalData, DstGlobalData, DstTileData, TileData, TmpTileData, T, kTCols_, topk>(
            dstGlobal, dstTile, srcTile, blockLen);
    } else {
        TmpTileData tmpMovTile(1, topk);
        TRESHAPE(tmpMovTile, tmpTile);
        TMOV(dstTile, tmpMovTile);
#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
        TSTORE(dstGlobal, dstTile);
    }
}

template <typename T>
void launchTopk(uint8_t *out, uint8_t *src, void *stream)
{
    // Shape mirrors the auto-clean instantiation
    //   `LanchTMrgsortTopK<float, 1, 1280, 1, 1280, 512>` from
    //   tests/npu/a2a3/src/st/testcase/tmrgsort/tmrgsort_kernel.cpp:357
    constexpr int kGRows_ = 1;
    constexpr int kGCols_ = 1280;
    constexpr int kTRows_ = 1;
    constexpr int kTCols_ = 1280;
    constexpr int topk    = 512;
    // Pass raw uint8_t* directly. The kernel applies __gm__ + reinterpret
    // internally; host-side casts to __gm__ pointers are rejected by bisheng.
    RunTopk<T, kGRows_, kGCols_, kTRows_, kTCols_, topk>
        <<<1, nullptr, stream>>>(out, src);
}

template void launchTopk<float>(uint8_t *out, uint8_t *src, void *stream);
