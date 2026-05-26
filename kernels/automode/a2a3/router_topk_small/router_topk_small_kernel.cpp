/**
 * router_topk_small_kernel.cpp - auto-mode A3 prototype.
 *
 * SKELETON. Tile types, loop shape, and host wrapper are final. The body of
 * the K-pass argmax loop is marked TODO(body) and will be filled in the
 * implementation pass.
 *
 * Algorithm (current design — to be ratified by build):
 *
 *   For each pass k = 0..K-1:
 *     TROWARGMAX(rowVal[T, 1], rowIdx[T, 1], scratch[T, E], tmp[T, E])
 *     // scratch starts as a copy of scores; rowVal/rowIdx hold the max-of-row
 *     // and its column index for this pass.
 *     copy rowVal -> outVal[:, k]     (TSUBVIEW dst into outVal at col k)
 *     copy rowIdx -> outIdx[:, k]
 *     suppress the just-picked column per row in scratch (TODO — see below)
 *
 *   The suppression step (writing -inf at the per-row picked index) has no
 *   ready-made TROW-scoped instruction in pto_instr.hpp; candidate
 *   implementations to evaluate at body-implementation time:
 *
 *     (a) TSCATTER with rowIdx as the scatter-index tile and a constant-(-inf)
 *         value tile. Requires per-row scatter from a vec source.
 *     (b) Pre-build a Tile<Vec, uint8, T, E> ramp `colId` (each row holds
 *         0..E-1) once; for each pass compute `mask = (colId == rowIdx[t])`
 *         and do `scratch -= MASK_BIG * mask`. Pure vec ops, no scatter.
 *     (c) Pre-sort each row with TSORT32 (E ≤ 32 fits in one TSORT32 dst),
 *         then TGATHER the first K (val, idx) pairs. Reuses the §A12 TopK
 *         shape; argmax avoided entirely.
 *
 *   Decision pending (see README §Open algorithmic choice).
 *
 * Risk acknowledged: TROWARGMAX is implemented in
 * include/pto/npu/a2a3/TRowReduceIdxOps.hpp, which is flagged by
 * docs_for_ai/auto_mode_bad_patterns.md §2.7 as having a PR-852 sync bug
 * (PtoSetWaitFlag inside __tf__ body without __PTO_AUTO__ guard). PR-852 is
 * NOT merged on this branch. If the first run produces wrong values or
 * indices, that is a likely suspect.
 *
 * Auto-mode constraints honored:
 *   - Single AICORE; no block_idx work split.
 *   - Static tile shapes; no SetValidRow / partial stores.
 *   - All tiles declared once outside the loop; auto allocator pins each.
 *   - No TASSIGN literal addresses, no `#ifndef __PTO_AUTO__` manual-sync,
 *     no Tile::data() in kernel code, no *_IMPL calls, no raw CCE intrinsics,
 *     no Event<>, no TPipe/TPUSH/TPOP, no double buffering.
 *
 * Host boundary: uint8_t* for typed FP32 + uint32 buffers; non-template
 * `…Fp32` wrapper hides any compiler-only types from main.cpp.
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>
#include "generated_cases.h"

using namespace pto;

namespace router_topk_small_cfg {

// Static configuration. Must match scripts/gen_data.py and main.cpp.
constexpr unsigned kT = kRouterTopkSmallT;    // rows
constexpr unsigned kE = kRouterTopkSmallE;    // columns (experts)
constexpr unsigned kK = kRouterTopkSmallK;    // top-K (compile-time; <= 10 by design)

static_assert(kK <= 10, "router_topk_small is specialised for small K (≤ 10).");
static_assert(kK <= kE, "K cannot exceed E.");

}  // namespace router_topk_small_cfg

// ============================================================================
// Kernel entry.
//   In  : scores       [T, E] float32.
//   Out : topk_values  [T, K] float32 (per row, descending).
//   Out : topk_indices [T, K] uint32  (per row, descending; columns of E).
// ============================================================================
template <typename T_, typename TIdx_>
__global__ AICORE void runRouterTopkSmall(
    __gm__ uint8_t *topk_values_raw,
    __gm__ uint8_t *topk_indices_raw,
    __gm__ uint8_t *scores_raw)
{
    using namespace router_topk_small_cfg;

    __gm__ T_    *scores       = reinterpret_cast<__gm__ T_    *>(scores_raw);
    __gm__ T_    *topk_values  = reinterpret_cast<__gm__ T_    *>(topk_values_raw);
    __gm__ TIdx_ *topk_indices = reinterpret_cast<__gm__ TIdx_ *>(topk_indices_raw);

    // GM layout note: 32-byte UB-burst alignment ([pto_tile.hpp:1510-1522])
    // requires RowMajor Cols*sizeof(DType) % 32 == 0 or ColMajor
    // Rows*sizeof(DType) % 32 == 0. For T=256, sizeof=4 (uint32 / float),
    // a RowMajor [kT, 1] tile holds 4 bytes per row — fails. The §A4 trowsum
    // fix is: declare reduction outputs as ColMajor and TRESHAPE to a
    // RowMajor [Cols, Rows] view before TSTORE. We do the same here, which
    // *transposes* the GM layout: GM holds [kK, kT] instead of [kT, kK].
    // Python golden + comparator are updated accordingly.
    using ScoresGlobal = GlobalTensor<T_,    Shape<1, 1, 1, kT, kE>, Stride<1, 1, 1, kE, 1>>;
    using OutValGlobal = GlobalTensor<T_,    Shape<1, 1, 1, kK, kT>, Stride<1, 1, 1, kT, 1>>;
    using OutIdxGlobal = GlobalTensor<TIdx_, Shape<1, 1, 1, kK, kT>, Stride<1, 1, 1, kT, 1>>;

    using ScoresTile  = Tile<TileType::Vec, T_,    kT, kE, BLayout::RowMajor, -1, -1>;
    using ScratchTile = Tile<TileType::Vec, T_,    kT, kE, BLayout::RowMajor, -1, -1>;  // workspace; suppression target
    using TmpTile     = Tile<TileType::Vec, T_,    kT, kE, BLayout::RowMajor, -1, -1>;  // TROWARGMAX scratch
    // ColMajor [kT, 1] for the per-pass argmax outputs (mirrors §A4 trowsum).
    using RowValTile  = Tile<TileType::Vec, T_,    kT, 1,  BLayout::ColMajor, kT, 1>;
    using RowIdxTile  = Tile<TileType::Vec, TIdx_, kT, 1,  BLayout::ColMajor, kT, 1>;
    // ColMajor [kT, kK] for the K-pass accumulators. ColMajor means a
    // column-k subview is shape [kT, 1] ColMajor — alignment-clean — so the
    // per-pass `rowValTile -> outValTile[:, k]` copy via TSUBVIEW works.
    using OutValTile  = Tile<TileType::Vec, T_,    kT, kK, BLayout::ColMajor, kT, kK>;
    using OutIdxTile  = Tile<TileType::Vec, TIdx_, kT, kK, BLayout::ColMajor, kT, kK>;
    // RowMajor [kK, kT] reshape views used as the TSTORE source.
    using OutValTileND = Tile<TileType::Vec, T_,    kK, kT, BLayout::RowMajor, kK, kT>;
    using OutIdxTileND = Tile<TileType::Vec, TIdx_, kK, kT, BLayout::RowMajor, kK, kT>;

    ScoresTile    scoresTile(kT, kE);
    ScratchTile   scratchTile(kT, kE);
    TmpTile       tmpTile(kT, kE);
    RowValTile    rowValTile;
    RowIdxTile    rowIdxTile;
    OutValTile    outValTile;
    OutIdxTile    outIdxTile;
    OutValTileND  outValTileND;
    OutIdxTileND  outIdxTileND;

    ScoresGlobal scoresGlobal(scores);
    OutValGlobal outValGlobal(topk_values);
    OutIdxGlobal outIdxGlobal(topk_indices);

    // Phase 1: load scores into the scratch workspace (which we mutate
    // across K passes by suppressing picked positions).
    TLOAD(scratchTile, scoresGlobal);

    // Phase 2: K passes of (argmax + write to col k + suppress).
    //
    // TODO(body): finalise the suppression strategy among options (a)/(b)/(c)
    //             in the file-header comment, then implement here. The
    //             skeleton below sketches the (b) ramp-mask form so reviewers
    //             can see the intended shape.
    for (unsigned k = 0; k < kK; ++k) {
        TROWARGMAX(rowValTile, rowIdxTile, scratchTile, tmpTile);

        // TODO(body): TSUBVIEW outValColK / outIdxColK := outValTile / outIdxTile
        //             at (row=0, col=k) with shape [kT, 1] ColMajor. Then
        //             TMOV(outValColK, rowValTile); TMOV(outIdxColK, rowIdxTile).

        // TODO(body): suppress scratchTile[t, rowIdxTile[t]] := -inf
        //             per option (a) TSCATTER, (b) ramp-mask, or (c) precomputed sort.
        (void)k;  // skeleton placeholder
    }

    // Phase 3: TRESHAPE the ColMajor [kT, kK] accumulators into RowMajor
    // [kK, kT] views, then TSTORE. The GM layout is [kK, kT] — Python
    // golden + comparator handle the transpose.
    TRESHAPE(outValTileND, outValTile);
    TRESHAPE(outIdxTileND, outIdxTile);
    TSTORE(outValGlobal, outValTileND);
    TSTORE(outIdxGlobal, outIdxTileND);
}

// ----------------------------------------------------------------------------
// Templated host launcher.
// ----------------------------------------------------------------------------
template <typename T_, typename TIdx_>
void launchRouterTopkSmall(uint8_t *topk_values,
                           uint8_t *topk_indices,
                           uint8_t *scores,
                           void    *stream)
{
    runRouterTopkSmall<T_, TIdx_><<<1, nullptr, stream>>>(
        topk_values, topk_indices, scores);
}

template void launchRouterTopkSmall<float, uint32_t>(
    uint8_t *topk_values, uint8_t *topk_indices, uint8_t *scores, void *stream);

// Non-template host-boundary wrapper.
extern "C" void launchRouterTopkSmallFp32(uint8_t *topk_values,
                                          uint8_t *topk_indices,
                                          uint8_t *scores,
                                          void    *stream)
{
    launchRouterTopkSmall<float, uint32_t>(topk_values, topk_indices, scores, stream);
}
