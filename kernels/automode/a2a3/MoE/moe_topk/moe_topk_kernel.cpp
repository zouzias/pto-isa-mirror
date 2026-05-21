/**
 * moe_topk_kernel.cpp - auto-mode A3 prototype.
 *
 * Per-row top-K selection over MoE router logits.
 * Input:  logits (kRows, kCols) = (256, 32) float32, one row per token.
 *         idx    (1, kCols)     = (1,  32)  uint32   identity row [0..31].
 * Output: outVal (kRows, kTopK) = (256, 2)  float32  top-K values, descending.
 *         outIdx (kRows, kTopK) = (256, 2)  uint32   matching expert indices.
 *
 * Pipeline per row:
 *   1. TLOAD  logits row + idx (identity)
 *   2. TSORT32 -> sort32DstTile:  one 32-element block -> 64 packed (val,idx)
 *   3. TSUBVIEW sortedTopKView: first kPackedTopK=4 elements of sort32DstTile
 *   4. TGATHER P0101 -> outValTile (top-2 float values)
 *   5. TRESHAPE + TGATHER P1010 -> outIdxTile (top-2 uint32 indices)
 *   6. TSTORE outVal + outIdx
 *
 * With kCols=32 and kTopK=2 (float, TYPE_COEF=1):
 *   kPackedCols = 32 * 2 = 64
 *   blockLen    = 64 * 1 = 64
 *   Main merge loop: blockLen*4 (=256) > kPackedCols (=64) -> skipped entirely.
 *   Tail block:     blockLen (=64) == kPackedCols (=64) -> also skipped.
 *   Result: TSORT32 alone fully sorts the 32-element row.
 *
 * Pattern source: kernels/automode/a2a3/topk/topk_kernel.cpp (confirmed-built).
 * Tiles declared inside the row loop (topk lesson: inside-loop declarations give
 * per-iter liveness analysis and prevent cross-iter UB aliasing).
 *
 * Limitations (v1):
 *   - Single AICORE; no block_idx work split.
 *   - Float dtype only (TYPE_COEF=1).
 *   - kCols must be exactly 32 (one TSORT32 block); merge loop/tail omitted.
 *   - kTopK=2: outValTile/outIdxTile width=2 (8 bytes); alignment risk noted.
 *   - No double-buffering.
 */

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

#include "generated_cases.h"

using namespace pto;

template <typename T, int kRows, int kCols, int kTopK>
__global__ AICORE void RunMoeTopk(__gm__ uint8_t *outVal_raw, __gm__ uint8_t *outIdx_raw,
                                   __gm__ uint8_t *src_raw,    __gm__ uint8_t *idx_raw)
{
    using indexT = uint32_t;
    constexpr int TYPE_COEF = sizeof(float) / sizeof(T);
    // Packed (val, idx) widths: everything post-TSORT32 uses packed widths.
    constexpr int kPackedCols = kCols * 2 * TYPE_COEF;  // 32*2*1 = 64
    constexpr int kPackedTopK = kTopK * 2 * TYPE_COEF;  //  2*2*1 =  4

    // pto_tile.hpp:1510 asserts Cols * sizeof(DType) % 32 == 0 (32-byte alignment).
    // kTopK=2 gives 2*4=8 bytes which fails. Use kGatherWidth=8 (exactly 32 bytes)
    // for the OutValTile/OutIdxTile Cols template parameter; the runtime valid
    // region stays (1, kTopK=2) via the dynamic constructor, so only kTopK values
    // are written by TGATHER and stored to GM.
    constexpr int kGatherWidth = 8;  // min 32-byte-aligned tile for float/uint32

    __gm__ T      *outVal = reinterpret_cast<__gm__ T *>(outVal_raw);
    __gm__ indexT *outIdx = reinterpret_cast<__gm__ indexT *>(outIdx_raw);
    __gm__ T      *src    = reinterpret_cast<__gm__ T *>(src_raw);
    __gm__ indexT *idx    = reinterpret_cast<__gm__ indexT *>(idx_raw);

    using SrcGlobal    = GlobalTensor<T,      Shape<1, 1, 1, 1, kCols>, Stride<1, 1, 1, kCols, 1>>;
    using IdxGlobal    = GlobalTensor<indexT, Shape<1, 1, 1, 1, kCols>, Stride<1, 1, 1, kCols, 1>>;
    using OutValGlobal = GlobalTensor<T,      Shape<1, 1, 1, 1, kTopK>, Stride<1, 1, 1, kTopK, 1>>;
    using OutIdxGlobal = GlobalTensor<indexT, Shape<1, 1, 1, 1, kTopK>, Stride<1, 1, 1, kTopK, 1>>;

    using SrcTile       = Tile<TileType::Vec, T,      1, kCols,        BLayout::RowMajor, -1, -1>;
    using IdxTile       = Tile<TileType::Vec, indexT, 1, kCols,        BLayout::RowMajor, -1, -1>;
    using PackedTile    = Tile<TileType::Vec, T,      1, kPackedCols,  BLayout::RowMajor, -1, -1>;
    using PackedIdxTile = Tile<TileType::Vec, indexT, 1, kPackedCols,  BLayout::RowMajor, -1, -1>;
    // Cols=kGatherWidth (not kTopK) satisfies the 32-byte alignment assertion.
    // Dynamic valid region allows construction with (1, kTopK) at runtime.
    using OutValTile    = Tile<TileType::Vec, T,      1, kGatherWidth, BLayout::RowMajor, -1, -1>;
    using OutIdxTile    = Tile<TileType::Vec, indexT, 1, kGatherWidth, BLayout::RowMajor, -1, -1>;

    for (int row = 0; row < kRows; ++row) {
        // pipe_barrier at row-loop start: hardware-confirmed requirement for
        // cross-iter auto-sync gap (see topk/topk_kernel.cpp for full rationale).
        pipe_barrier(PIPE_ALL);

        SrcGlobal     srcGlobal(src + row * kCols);
        IdxGlobal     idxGlobal(idx);
        OutValGlobal  outValGlobal(outVal + row * kTopK);
        OutIdxGlobal  outIdxGlobal(outIdx + row * kTopK);

        // Tiles inside loop: per-iter liveness isolation (topk lesson).
        SrcTile     srcTile(1, kCols);
        IdxTile     idxTile(1, kCols);
        SrcTile     sort32TmpTile(1, kCols);
        PackedTile  sort32DstTile(1, kPackedCols);
        OutValTile  outValTile(1, kTopK);
        OutIdxTile  outIdxTile(1, kTopK);

        // Phase 1: TSORT32 - one 32-element block -> 64 packed (val,idx).
        // With kCols=32 this is a single block sort; no merge loop follows.
        TLOAD(srcTile, srcGlobal);
        TLOAD(idxTile, idxGlobal);
        TSORT32(sort32DstTile, srcTile, idxTile, sort32TmpTile);

        // Phase 2: merge loop - skipped because blockLen=64 == kPackedCols=64
        // (blockLen*4 = 256 > kPackedCols = 64 on first check).

        // Phase 3: tail block - also skipped (blockLen == kPackedCols).

        // Phase 4: TGATHER values - prefix view of kPackedTopK=4 packed elems.
        // After TSORT32, sort32DstTile[0..3] holds the 2 highest (val,idx) pairs.
        // P0101 picks positions 0,2 (val slots) for float (TYPE_COEF=1).
        PackedTile sortedTopKView(1, kPackedTopK);
        TSUBVIEW(sortedTopKView, sort32DstTile, 0, 0);
        if constexpr (std::is_same_v<T, half>) {
            TGATHER<OutValTile, PackedTile, MaskPattern::P0001>(outValTile, sortedTopKView);
        } else {
            TGATHER<OutValTile, PackedTile, MaskPattern::P0101>(outValTile, sortedTopKView);
        }

        // Phase 5: TGATHER indices via TRESHAPE type-pun (float buf as uint32).
        // P1010 picks positions 1,3 (idx slots).
        PackedIdxTile sortedTopKIdxView(1, kPackedTopK);
        TRESHAPE(sortedTopKIdxView, sort32DstTile);
        TGATHER<OutIdxTile, PackedIdxTile, MaskPattern::P1010>(outIdxTile, sortedTopKIdxView);

        // Phase 6: TSTORE results.
        TSTORE(outValGlobal, outValTile);
        TSTORE(outIdxGlobal, outIdxTile);
    }
}

template <typename T>
void launchMoeTopk(uint8_t *outVal, uint8_t *outIdx, uint8_t *src, uint8_t *idx, void *stream)
{
    constexpr int kRows = kMoeT;
    constexpr int kCols = kMoeE;
    constexpr int kTopK = (kMoeTopK >= 2) ? kMoeTopK : 2;  // moe_topk standalone tests K >= 2
    RunMoeTopk<T, kRows, kCols, kTopK><<<1, nullptr, stream>>>(outVal, outIdx, src, idx);
}

template void launchMoeTopk<float>(uint8_t *outVal, uint8_t *outIdx, uint8_t *src, uint8_t *idx, void *stream);
