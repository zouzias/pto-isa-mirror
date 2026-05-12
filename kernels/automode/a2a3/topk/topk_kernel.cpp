/**
 * topk_kernel.cpp - auto-mode A3 prototype, full top-K with values + indices.
 *
 * Adapted from kernels/manual/a2a3/topk/topk_kernel.cpp's runTOPK pipeline,
 * conservatively ported for auto mode. The previous values-only shortcut
 * (which assumed pre-sorted 64-element blocks and skipped TSORT32 / index
 * tracking) has been removed.
 *
 * Pipeline (single AICORE, single row, no buffering):
 *   1. TLOAD  src + idx
 *   2. TSORT32(packed, src, idx, scratch)        — per 32-element block sort,
 *                                                   emits (val, idx) packed format
 *   3. main TMRGSORT loop (4-way self-merge)     — blockLen = 64*TYPE_COEF, *= 4
 *   4. SortTailBlock for non-power-of-4 residual — independent dst + TMOV-back
 *   5. TGATHER  P0101 → outVal     (float-only mask; half is v2)
 *   6. TGATHER  P1010 → outIdx     (via TRESHAPE type-pun: float-buf as uint32)
 *   7. TSTORE  outVal + outIdx
 *
 * v1 limitations (deliberate; documented in README.md):
 *   - Single AICORE; no block_idx work split.
 *   - Single row (kRows=1); no row loop.
 *   - Float dtype only (TYPE_COEF=1). Half (TYPE_COEF=2) mask patterns differ;
 *     deferred to v2.
 *   - No double / multi-buffering; no TPipe / TPUSH / TPOP.
 *   - No unguarded set_flag / wait_flag / pipe_barrier in kernel scope.
 *   - SortTailBlock writes to an INDEPENDENT destination (mrgScratchTile)
 *     and TMOVs the merged run back to sort32DstTile prefix; this avoids
 *     the in-place TMRGSORT-with-dst-aliased-to-src pattern from the
 *     manual (in-place worked there with manual sync; auto-mode behavior
 *     of in-place merge is the open question we are sidestepping for v1).
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

template <typename T, int kCols, int kTopK>
__global__ AICORE void RunTopk(__gm__ uint8_t *outVal_raw, __gm__ uint8_t *outIdx_raw,
                                __gm__ uint8_t *src_raw,    __gm__ uint8_t *idx_raw)
{
    using indexT = uint32_t;
    constexpr int TYPE_COEF = sizeof(float) / sizeof(T);
    // Packed (val, idx) widths per the manual TopK convention:
    //   kPackedCols  ↔ manual `dstCols  = validCol * 2 * TYPE_COEF`
    //   kPackedTopK  ↔ manual `dtopk    = topk     * 2 * TYPE_COEF`
    // Everything POST-TSORT32 (main merge loop, SortTailBlock, final TGATHER
    // prefix) operates on packed widths, not source-element widths.
    constexpr int kPackedCols = kCols * 2 * TYPE_COEF;
    constexpr int kPackedTopK = kTopK * 2 * TYPE_COEF;

    // Host launchers cannot apply __gm__ via reinterpret_cast (E11). Cast
    // inside the kernel where __gm__ is a valid type qualifier.
    __gm__ T      *outVal = reinterpret_cast<__gm__ T *>(outVal_raw);
    __gm__ indexT *outIdx = reinterpret_cast<__gm__ indexT *>(outIdx_raw);
    __gm__ T      *src    = reinterpret_cast<__gm__ T *>(src_raw);
    __gm__ indexT *idx    = reinterpret_cast<__gm__ indexT *>(idx_raw);

    using SrcGlobal    = GlobalTensor<T,      Shape<1, 1, 1, 1, kCols>, Stride<1, 1, 1, kCols, 1>>;
    using IdxGlobal    = GlobalTensor<indexT, Shape<1, 1, 1, 1, kCols>, Stride<1, 1, 1, kCols, 1>>;
    using OutValGlobal = GlobalTensor<T,      Shape<1, 1, 1, 1, kTopK>, Stride<1, 1, 1, kTopK, 1>>;
    using OutIdxGlobal = GlobalTensor<indexT, Shape<1, 1, 1, 1, kTopK>, Stride<1, 1, 1, kTopK, 1>>;

    using SrcTile       = Tile<TileType::Vec, T,      1, kCols,       BLayout::RowMajor, -1, -1>;
    using IdxTile       = Tile<TileType::Vec, indexT, 1, kCols,       BLayout::RowMajor, -1, -1>;
    using PackedTile    = Tile<TileType::Vec, T,      1, kPackedCols, BLayout::RowMajor, -1, -1>;
    using PackedIdxTile = Tile<TileType::Vec, indexT, 1, kPackedCols, BLayout::RowMajor, -1, -1>;
    using OutValTile    = Tile<TileType::Vec, T,      1, kTopK,       BLayout::RowMajor, -1, -1>;
    using OutIdxTile    = Tile<TileType::Vec, indexT, 1, kTopK,       BLayout::RowMajor, -1, -1>;

    // Independent storage tiles. Auto allocator places each in non-overlapping
    // UB based on liveness; no TASSIGN with numeric addresses.
    SrcTile     srcTile(1, kCols);
    IdxTile     idxTile(1, kCols);
    SrcTile     sort32TmpTile(1, kCols);     // TSORT32 internal scratch (content irrelevant)
    PackedTile  sort32DstTile(1, kPackedCols);
    PackedTile  mrgScratchTile(1, kPackedCols);
    OutValTile  outValTile(1, kTopK);
    OutIdxTile  outIdxTile(1, kTopK);

    SrcGlobal     srcGlobal(src);
    IdxGlobal     idxGlobal(idx);
    OutValGlobal  outValGlobal(outVal);
    OutIdxGlobal  outIdxGlobal(outIdx);

    // ============================================================
    // Phase 1: TSORT32 — per-32-block in-place sort, emit (val, idx) packed.
    // ============================================================
    TLOAD(srcTile, srcGlobal);
    TLOAD(idxTile, idxGlobal);
    TSORT32(sort32DstTile, srcTile, idxTile, sort32TmpTile);

    // ============================================================
    // Phase 2: main merge loop — 4-way self-merge with ping-pong between
    // sort32DstTile (post-TSORT32 source) and mrgScratchTile (independent
    // destination buffer). After each TMRGSORT, TMOV copies the merged
    // result back to sort32DstTile prefix so the next iteration's view of
    // sort32DstTile sees fresh merged data.
    // ============================================================
    // blockLen and cols are in PACKED-element units (manual maps to valid_col
    // = dstCols once MrgsortSingleRow operates on the post-TSORT32 buffer).
    uint32_t blockLen = 64 * TYPE_COEF;
    for (; blockLen * 4 <= kPackedCols; blockLen *= 4) {
        uint16_t cols = kPackedCols / (blockLen * 4) * (blockLen * 4);
        // Semantic prefix views (NOT memory reuse):
        //   srcSortedView is the cols-wide prefix view of sort32DstTile,
        //     which owns the data on iteration entry.
        //   tmpSortedView is the cols-wide prefix view of mrgScratchTile,
        //     an INDEPENDENT buffer used as the merge destination — not
        //     overlapping with sort32DstTile's storage.
        PackedTile srcSortedView(1, cols);
        PackedTile tmpSortedView(1, cols);
        // Same-type prefix slice — TSUBVIEW(..., 0, 0) is the canonical form.
        // TRESHAPE is reserved for true reshape / type-pun cases (see Phase 5).
        TSUBVIEW(srcSortedView, sort32DstTile, 0, 0);
        TSUBVIEW(tmpSortedView, mrgScratchTile, 0, 0);
        TMRGSORT<PackedTile, PackedTile>(tmpSortedView, srcSortedView, blockLen);
        // Promote merged result back into sort32DstTile prefix so the next
        // iteration's srcSortedView reads the fresh merged data. Lifetime:
        // tmpSortedView is read once here (TMOV source), then becomes dead.
        TMOV(srcSortedView, tmpSortedView);
    }

    // ============================================================
    // Phase 3: SortTailBlock — handle non-power-of-4 residuals.
    // Mirrors manual SortTailBlock structurally but writes to an INDEPENDENT
    // destination (mrgScratchTile via curDstView) and TMOV-backs to
    // sort32DstTile prefix. The manual's in-place merge (curDstTile aliased
    // onto srcTile.data()) is NOT used — that pattern produced interleaved
    // output in the previous values-only port.
    // ============================================================
    // Tail-block bound, mrgArray planning, and per-iter clip values are all
    // in PACKED-element units. Manual mapping:
    //   kPackedCols  ↔ Cols  (manual SortTailBlock template arg)
    //   kPackedTopK  ↔ topk  (manual SortTailBlock template arg, = dtopk)
    if (blockLen < kPackedCols) {
        PackedTile tmp1Tile(1, kPackedCols);  // TMRGSORT scratch (manual: SrcTileData=RowTile, packed width)
        int32_t mrgArray[15] = {0};
        int32_t arrayCount = FillMrgArray<kPackedCols>(mrgArray, blockLen);
        uint16_t mrgSortedLen = 0;
        MrgSortExecutedNumList executedNumList;
        for (int32_t i = 0; i < arrayCount - 1; ++i) {
            mrgSortedLen += static_cast<uint16_t>(mrgArray[i]);
            uint64_t tmpMrgSortedLen = mrgSortedLen;
            uint64_t tmpMrgArray = mrgArray[i + 1];
            if (tmpMrgSortedLen > kPackedTopK) tmpMrgSortedLen = kPackedTopK;
            if (tmpMrgArray > kPackedTopK) tmpMrgArray = kPackedTopK;

            // Semantic alias views into sort32DstTile (which owns the data on
            // iteration entry):
            //   src0View = tmpMrgSortedLen-wide prefix view  (already-merged region)
            //   src1View = tmpMrgArray-wide offset view at offset mrgSortedLen
            //              (the next tail run produced by the previous merge)
            PackedTile src0View(1, tmpMrgSortedLen);
            PackedTile src1View(1, tmpMrgArray);
            // src0View is a same-type prefix slice (offset 0); src1View is a
            // same-type slice at column offset mrgSortedLen. Both expressed
            // via TSUBVIEW for semantic clarity.
            TSUBVIEW(src0View, sort32DstTile, 0, 0);
            TSUBVIEW(src1View, sort32DstTile, 0, mrgSortedLen);

            // Independent destination view: curDstView aliases mrgScratchTile
            // (NOT sort32DstTile). TMRGSORT reads src0View / src1View from
            // sort32DstTile and writes the merged run into mrgScratchTile —
            // no read/write overlap on the same buffer.
            PackedTile curDstView(1, tmpMrgSortedLen + tmpMrgArray);
            // Same-type prefix slice of the independent destination buffer.
            TSUBVIEW(curDstView, mrgScratchTile, 0, 0);
            // All four roles (Dst, Tmp, Src0, Src1) are PackedTile; matches the
            // manual SortTailBlock instantiation `<DstTileData, SrcTileData,
            // SrcTileData, ...>` where SrcTileData = RowTile (packed width).
            TMRGSORT<PackedTile, PackedTile, PackedTile, PackedTile, 0>(
                curDstView, executedNumList, tmp1Tile, src0View, src1View);

            // Semantic prefix view of sort32DstTile, target of TMOV-back from
            // curDstView. After this TMOV, sort32DstTile[0..len-1] holds the
            // merged run, so the next loop iteration's src0View sees it.
            PackedTile copyBackView(1, tmpMrgSortedLen + tmpMrgArray);
            // Same-type prefix slice of sort32DstTile (TMOV destination).
            TSUBVIEW(copyBackView, sort32DstTile, 0, 0);
            TMOV(copyBackView, curDstView);
        }
    }

    // ============================================================
    // Phase 4: TGATHER values from the (val, idx) packed buffer.
    // ============================================================
    // Semantic prefix view: sortedTopKView is the kPackedTopK-wide prefix of
    // sort32DstTile (= manual `dtopk = topk * 2 * TYPE_COEF`). After phases
    // 2-3, sort32DstTile[0..kPackedTopK-1] holds the descending-sorted top-K
    // (val, idx) pairs. sort32DstTile owns the data; sortedTopKView is read
    // once by TGATHER. Mask P0101 picks val slots for float (TYPE_COEF=1);
    // P0001 for half (v2).
    PackedTile sortedTopKView(1, kPackedTopK);
    // Same-type prefix slice (kPackedTopK-wide prefix of sort32DstTile).
    TSUBVIEW(sortedTopKView, sort32DstTile, 0, 0);
    if constexpr (std::is_same_v<T, half>) {
        TGATHER<OutValTile, PackedTile, MaskPattern::P0001>(outValTile, sortedTopKView);
    } else {
        TGATHER<OutValTile, PackedTile, MaskPattern::P0101>(outValTile, sortedTopKView);
    }

    // ============================================================
    // Phase 5: TGATHER indices via TRESHAPE type-pun (float-buf as uint32).
    // ============================================================
    // Semantic alias view (DIFFERENT element type — a sanctioned type-pun):
    // sortedTopKIdxView is the same prefix of the same UB buffer that holds
    // the packed (val, idx) data, but viewed as uint32 elements so TGATHER
    // P1010 can pick the idx slots. Pattern lifted from include/pto/npu/a2a3/
    // TQuant.hpp's auto branch (TRESHAPE_IMPL between half and int32 element
    // types) — Inferred to apply at the kernel-level TRESHAPE wrapper here.
    // sort32DstTile owns the bytes; sortedTopKIdxView reads them once.
    PackedIdxTile sortedTopKIdxView(1, kPackedTopK);
    TRESHAPE(sortedTopKIdxView, sort32DstTile);
    TGATHER<OutIdxTile, PackedIdxTile, MaskPattern::P1010>(outIdxTile, sortedTopKIdxView);

    // ============================================================
    // Phase 6: TSTORE values + indices to GM.
    // ============================================================
    TSTORE(outValGlobal, outValTile);
    TSTORE(outIdxGlobal, outIdxTile);
}

template <typename T>
void launchTopk(uint8_t *outVal, uint8_t *outIdx, uint8_t *src, uint8_t *idx, void *stream)
{
    constexpr int kCols = 1280;
    constexpr int kTopK = 512;
    // Pass raw uint8_t* directly. The kernel applies __gm__ + reinterpret
    // internally; host-side casts to __gm__ pointers are rejected by bisheng (E11).
    RunTopk<T, kCols, kTopK><<<1, nullptr, stream>>>(outVal, outIdx, src, idx);
}

template void launchTopk<float>(uint8_t *outVal, uint8_t *outIdx, uint8_t *src, uint8_t *idx, void *stream);
