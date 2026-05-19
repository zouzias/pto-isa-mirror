/**
 * gather_kernel.cpp - auto-mode A3 prototype (v3).
 *
 * Unpack-and-accumulate per-expert FFN outputs back into per-token rows, with
 * softmax routing weights for kTopK > 1. Generic over kTopK in {1, 2, 4, 8, 16}.
 *
 * v3 changes over v2:
 *   - kTopK == 1: replaces the per-row TLOAD/TSTORE loop with a chunked
 *     bulk-TSCATTER row reorder. Idx tile built on-chip from A_id via
 *     TMULS + TROWEXPAND + TCI + TCOLEXPANDADD. Cuts 256 PIPE_ALL barriers
 *     and 512 single-row DMAs down to a handful of bulk DMAs.
 *   - kTopK > 1: drops the reordered_scratch GM round-trip. Builds the
 *     inverse map r_inv[k, t] = r in UB via a one-shot TSCATTER, then fuses
 *     pass-2 (reorder) and pass-3 (weighted combine) into a single H-chunked
 *     loop where the reorder is a UB-to-UB TGATHER per k.
 *
 * Inputs   (GM): B                [kT*kTopK + 16, kH] fp32
 *                A_id             [kT*kTopK + 16]     int32
 *                rank_id          [kT*kTopK + 16]     int32 (only kTopK > 1)
 *                outVal           [kT, kPadded]       fp32  (only kTopK > 1;
 *                                                            cols kTopK..kPadded-1
 *                                                            host-padded with -1e30)
 * Scratch  (GM): weights_scratch  [kT, kPadded]       fp32  (only kTopK > 1)
 * Outputs  (GM): C                [kT, kH]            fp32
 *
 *   kPadded = max(8, kTopK) — softmax tile column padding for 32-byte UB alignment.
 *
 * The reordered_scratch GM buffer used by v2 is GONE.
 *
 * Auto-mode constraints honored:
 *   - Single AICORE (<<<1, nullptr, stream>>>).
 *   - Static tiles declared once outside loops.
 *   - GlobalTensor reconstructed per H-chunk iter with runtime offset.
 *   - No TASSIGN, no Tile::data() in kernel, no *_IMPL calls, no raw CCE
 *     intrinsics, no Event<>, no manual sync, no TPipe/TPUSH/TPOP, no double
 *     buffering, no A5-only ops.
 *
 * Notable risks (unconfirmed):
 *   - First in-tree auto-mode A3 use of indexed TGATHER and TSCATTER. The
 *     mask-pattern TGATHER variant is exercised by the TopK kernel; the
 *     indexed forms invoked here are not.
 *   - Per-k uses TSUBVIEW(rInvFlat, k, 0) + TRESHAPE to view a (1, kT)
 *     prefix as (kT, 1) ColMajor; same-byte-count layout reinterpret in the
 *     style of the TopK kernel's float<->uint32 TRESHAPE, but for layout
 *     change rather than dtype change.
 *   - kChunkH adapts with kTopK to keep bChunkBig under the UB ceiling.
 *
 * Pattern sources:
 *   - Softmax (kTopK > 1): tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp:54-60
 *   - GlobalTensor-with-runtime-offset / static tile / no manual sync:
 *     kernels/automode/a2a3/moe_top1_unpermute/moe_top1_unpermute_kernel.cpp
 *   - Weighted combine via TROWEXPANDMUL + TADD: identical math to v2's pass 3.
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace gather_cfg {

// v3 shape — must match scripts/gen_data.py and main.cpp.
constexpr unsigned kT    = 256;
constexpr unsigned kH    = 64;
constexpr unsigned kTopK = 1;

constexpr unsigned kPackedRows   = kT * kTopK;
constexpr unsigned kOverspillPad = 16;
constexpr unsigned kAlloc        = kPackedRows + kOverspillPad;

// Softmax tile column padding for 32-byte UB alignment.
// fp32 needs Cols * 4 % 32 == 0  ->  Cols % 8 == 0.
constexpr unsigned kPadded = (kTopK < 8) ? 8 : kTopK;

// H-chunk picked so that bChunk (kPackedRows × kChunkH × 4 B) stays under
// the ~192 KB UB ceiling. fp32 32-B alignment requires kChunkH % 8 == 0.
constexpr unsigned kChunkH = (kTopK <= 2) ? 32 : (kTopK <= 4) ? 16 : 8;
static_assert(kH % kChunkH == 0, "kH must be a multiple of kChunkH");
static_assert((kChunkH * sizeof(float)) % 32 == 0,
              "kChunkH must satisfy 32-B UB alignment for fp32");

}  // namespace gather_cfg

template <typename T>
__global__ AICORE void runGather(
    __gm__ T       __out__   *C,
    __gm__ T       __in__    *B,
    __gm__ int32_t __in__    *A_id,
    __gm__ int32_t __in__    *rank_id,
    __gm__ T       __in__    *outVal,
    __gm__ T       __out__   *weights_scratch)
{
    using namespace gather_cfg;

    // ---- Common per-chunk view of C ------------------------------------
    using CChunkShape  = Shape <1, 1, 1, kT, kChunkH>;
    using CChunkStride = Stride<kT * kH, kT * kH, kT * kH, kH, 1>;
    using CChunkGlobal = GlobalTensor<T, CChunkShape, CChunkStride>;

    // [0..kChunkH) ramp built on-chip (TCI requires Rows == 1).
    using RampRow = Tile<TileType::Vec, int32_t, 1, kChunkH,
                         BLayout::RowMajor, 1, kChunkH>;

    if constexpr (kTopK == 1) {
        // ================================================================
        // Plan A: bulk TSCATTER row reorder.
        //   idx[r, h] = A_id[r] * kChunkH + h
        //   cChunk[idx[r, h]] = bChunk[r, h]    (UB-to-UB)
        //   Then one bulk TSTORE per chunk to C.
        // ================================================================
        (void)rank_id;
        (void)outVal;
        (void)weights_scratch;

        // B at kTopK==1 has the same (kT, kH) shape as C.
        using BChunkShape  = CChunkShape;
        using BChunkStride = CChunkStride;
        using BChunkGlobal = GlobalTensor<T, BChunkShape, BChunkStride>;

        using ChunkTile = Tile<TileType::Vec, T, kT, kChunkH,
                               BLayout::RowMajor, kT, kChunkH>;
        using IdxTile   = Tile<TileType::Vec, int32_t, kT, kChunkH,
                               BLayout::RowMajor, kT, kChunkH>;

        using AIdShape  = Shape <1, 1, 1, kT, 1>;
        using AIdStride = Stride<kT, kT, kT, 1, 1>;
        using AIdGlobal = GlobalTensor<int32_t, AIdShape, AIdStride>;
        using AIdCol    = Tile<TileType::Vec, int32_t, kT, 1,
                               BLayout::ColMajor, kT, 1>;

        AIdCol    aIdCol;
        AIdCol    baseCol;
        IdxTile   baseTile;
        IdxTile   idxTile;
        RampRow   rampRow;
        ChunkTile bChunk;
        ChunkTile cChunk;

        AIdGlobal aIdGlobal(A_id);

        // ---- Hoisted index construction (reused across all H-chunks) ----
        TLOAD(aIdCol, aIdGlobal);                                  // (kT, 1)
        TMULS(baseCol, aIdCol, static_cast<int32_t>(kChunkH));     // A_id * kChunkH
        TROWEXPAND(baseTile, baseCol);                             // (kT, kChunkH) row-broadcast
        TCI<RampRow, int32_t, /*descending=*/0>(rampRow, 0);       // (1, kChunkH) = [0..kChunkH)
        TCOLEXPANDADD(idxTile, baseTile, rampRow);                 // idx[r, h] = base[r] + h

        // ---- Per H-chunk bulk reorder ----------------------------------
        for (unsigned col = 0; col < kH; col += kChunkH) {
            BChunkGlobal bGlobal(B + col);
            CChunkGlobal cGlobal(C + col);

            TLOAD(bChunk, bGlobal);
            TSCATTER(cChunk, bChunk, idxTile);
            TSTORE(cGlobal, cChunk);
        }
    } else {
        // ================================================================
        // Plan B2: r_inv built in UB + per-k indexed TGATHER + weighted combine.
        // ================================================================

        // B at kTopK > 1 has shape (kPackedRows, kH).
        using BBigShape  = Shape <1, 1, 1, kPackedRows, kChunkH>;
        using BBigStride = Stride<kPackedRows * kH, kPackedRows * kH,
                                  kPackedRows * kH, kH, 1>;
        using BBigGlobal = GlobalTensor<T, BBigShape, BBigStride>;
        using BBigTile   = Tile<TileType::Vec, T, kPackedRows, kChunkH,
                                BLayout::RowMajor, kPackedRows, kChunkH>;

        // A_id and rank_id loaded as (1, kPackedRows) row-major.
        using PackedRowShape  = Shape <1, 1, 1, 1, kPackedRows>;
        using PackedRowStride = Stride<kPackedRows, kPackedRows, kPackedRows,
                                       kPackedRows, 1>;
        using PackedRowGlobal = GlobalTensor<int32_t, PackedRowShape, PackedRowStride>;
        using PackedRow       = Tile<TileType::Vec, int32_t, 1, kPackedRows,
                                     BLayout::RowMajor, 1, kPackedRows>;

        // r_inv layout: (kTopK, kT) row-major. Slot encoding:
        //   slot[r] = rank_id[r] * kT + A_id[r]
        //   r_inv[rank_id[r], A_id[r]] = r
        // Per-k slice is row k → (1, kT) row-major prefix view, which we
        // TRESHAPE to (kT, 1) ColMajor (same kT contiguous int32s) so that
        // TROWEXPAND can broadcast across kChunkH columns.
        using RInvFlat = Tile<TileType::Vec, int32_t, kTopK, kT,
                              BLayout::RowMajor, kTopK, kT>;
        using RInvKRow = Tile<TileType::Vec, int32_t, 1, kT,
                              BLayout::RowMajor, 1, kT>;
        using RInvKCol = Tile<TileType::Vec, int32_t, kT, 1,
                              BLayout::ColMajor, kT, 1>;

        // Softmax tiles (same composition as v2).
        using SoftmaxShape  = Shape <1, 1, 1, kT, kPadded>;
        using SoftmaxStride = Stride<kT * kPadded, kT * kPadded, kT * kPadded, kPadded, 1>;
        using SoftmaxGlobal = GlobalTensor<T, SoftmaxShape, SoftmaxStride>;
        using ValTile   = Tile<TileType::Vec, T, kT, kPadded,
                               BLayout::RowMajor, kT, kPadded>;
        using BcastTile = Tile<TileType::Vec, T, kT, 1,
                               BLayout::ColMajor, kT, 1>;

        // Per-k weight column from GM scratch via strided GlobalTensor.
        using WeightShape  = Shape <1, 1, 1, kT, 1>;
        using WeightStride = Stride<kT * kPadded, kT * kPadded, kT * kPadded, kPadded, 1>;
        using WeightGlobal = GlobalTensor<T, WeightShape, WeightStride>;
        using WeightTile   = Tile<TileType::Vec, T, kT, 1,
                                  BLayout::ColMajor, kT, 1>;

        // Per-chunk tiles.
        using ChunkTile = Tile<TileType::Vec, T, kT, kChunkH,
                               BLayout::RowMajor, kT, kChunkH>;
        using IdxTile   = Tile<TileType::Vec, int32_t, kT, kChunkH,
                               BLayout::RowMajor, kT, kChunkH>;

        // ----------------------------------------------------------------
        // Pass 0: build r_inv[k, t] = r in UB.
        // ----------------------------------------------------------------
        PackedRow iotaRow;
        PackedRow aIdRow;
        PackedRow rIdRow;
        PackedRow scaledRow;
        PackedRow slotRow;
        RInvFlat  rInvFlat;

        PackedRowGlobal aIdRowGlobal(A_id);
        PackedRowGlobal rIdRowGlobal(rank_id);

        TCI<PackedRow, int32_t, /*descending=*/0>(iotaRow, 0);   // [0..kPackedRows)
        TLOAD(aIdRow, aIdRowGlobal);
        TLOAD(rIdRow, rIdRowGlobal);
        TMULS(scaledRow, rIdRow, static_cast<int32_t>(kT));      // rank_id * kT
        TADD (slotRow, scaledRow, aIdRow);                       // + A_id
        TSCATTER(rInvFlat, iotaRow, slotRow);                    // rInvFlat[slot[r]] = r

        // ----------------------------------------------------------------
        // Pass 1: softmax(outVal) -> weights_scratch.
        //   weights_scratch stays as a GM buffer because the per-k weight
        //   column needs a strided TLOAD (RowStride is hard-coded to Cols
        //   in the Tile type system, so UB-side per-k slicing via TSUBVIEW
        //   would not give a (kT, 1) ColMajor view of column k of a
        //   (kT, kPadded) RowMajor tile).
        // ----------------------------------------------------------------
        ValTile   valTile;
        BcastTile maxTile;
        ValTile   tmpTile;
        ValTile   expTile;
        BcastTile sumTile;
        ValTile   weightTile;

        SoftmaxGlobal outValGlobal (outVal);
        SoftmaxGlobal weightsGlobal(weights_scratch);

        TLOAD(valTile, outValGlobal);
        TROWMAX(maxTile, valTile, tmpTile);
        TROWEXPANDSUB(tmpTile, valTile, maxTile);
        TEXP(expTile, tmpTile);
        TROWSUM(sumTile, expTile, tmpTile);
        TROWEXPANDDIV(weightTile, expTile, sumTile);
        TSTORE(weightsGlobal, weightTile);

        // ----------------------------------------------------------------
        // Pass 2: per H-chunk fused reorder + weighted combine.
        //   For each chunk:
        //     bChunkBig <- B[:, col:col+kChunkH]    (kPackedRows × kChunkH)
        //     accChunk  <- 0                        (kT × kChunkH)
        //     for k in 0..kTopK-1:
        //       rInvK = view (kT, 1) of rInvFlat row k
        //       idxK[t, h] = rInvK[t] * kChunkH + h
        //       gathK = bChunkBig[idxK]             (UB-side TGATHER)
        //       weightK = weights_scratch[:, k]     (GM strided TLOAD)
        //       accChunk += weightK * gathK         (TROWEXPANDMUL + TADD)
        //     TSTORE accChunk -> C[:, col:col+kChunkH]
        // ----------------------------------------------------------------
        RampRow    rampRow;
        BBigTile   bChunkBig;
        ChunkTile  accChunk;
        ChunkTile  gathK;
        ChunkTile  scaledK;
        IdxTile    baseK;
        IdxTile    idxK;
        IdxTile    tmpK;
        RInvKRow   rInvKRow;
        RInvKCol   rInvKCol;
        RInvKCol   rInvKColScaled;
        WeightTile weightK;

        TCI<RampRow, int32_t, /*descending=*/0>(rampRow, 0);     // (1, kChunkH)

        for (unsigned col = 0; col < kH; col += kChunkH) {
            BBigGlobal   bGlobal(B + col);
            CChunkGlobal cGlobal(C + col);

            TLOAD(bChunkBig, bGlobal);
            TEXPANDS(accChunk, static_cast<T>(0));

            for (unsigned k = 0; k < kTopK; ++k) {
                TSUBVIEW(rInvKRow, rInvFlat, static_cast<uint16_t>(k), static_cast<uint16_t>(0));
                TRESHAPE(rInvKCol, rInvKRow);

                TMULS(rInvKColScaled, rInvKCol, static_cast<int32_t>(kChunkH));
                TROWEXPAND(baseK, rInvKColScaled);
                TCOLEXPANDADD(idxK, baseK, rampRow);

                TGATHER(gathK, bChunkBig, idxK, tmpK);

                WeightGlobal weightGlobal(weights_scratch + k);
                TLOAD(weightK, weightGlobal);

                TROWEXPANDMUL(scaledK, gathK, weightK);
                TADD(accChunk, accChunk, scaledK);
            }
            TSTORE(cGlobal, accChunk);
        }
    }
}

template <typename T>
void launchGather(T *C, T *B,
                  int32_t *A_id, int32_t *rank_id,
                  T *outVal, T *weights_scratch,
                  void *stream)
{
    runGather<T><<<1, nullptr, stream>>>(C, B, A_id, rank_id, outVal, weights_scratch);
}

template void launchGather<float>(float *C, float *B,
                                  int32_t *A_id, int32_t *rank_id,
                                  float *outVal, float *weights_scratch,
                                  void *stream);
