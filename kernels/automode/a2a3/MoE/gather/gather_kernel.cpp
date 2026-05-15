/**
 * gather_kernel.cpp - auto-mode A3 prototype.
 *
 * Unpack-and-accumulate per-expert FFN outputs back into per-token rows, with
 * softmax routing weights for kTopK > 1. Generic over kTopK in {1, 2, 4, 8, 16}.
 *
 * Inputs   (GM): B                [kT*kTopK + 16, kH] fp32  (first kT*kTopK rows consulted)
 *                A_id             [kT*kTopK + 16]     int32 (first kT*kTopK consulted)
 *                rank_id          [kT*kTopK + 16]     int32 (only used when kTopK > 1)
 *                outVal           [kT, kPadded]       fp32  (only used when kTopK > 1;
 *                                                            cols kTopK..kPadded-1 host-padded
 *                                                            with -1e30 so exp() underflows to 0)
 * Scratch  (GM): weights_scratch  [kT, kPadded]       fp32  (only used when kTopK > 1)
 * Outputs  (GM): C                [kT, kH]            fp32  (zero-initialized by host)
 *
 *   kPadded = max(8, kTopK) — softmax tile column padding for 32-byte UB alignment.
 *
 * ===========================================================================
 * Algorithm overview
 * ===========================================================================
 *
 *   if constexpr (kTopK == 1):
 *       // Fast path: softmax of a single value is always 1.0, so the weighted
 *       // gather degenerates to an unweighted permute / accumulation. Match v1
 *       // behavior bit-for-bit; skip the softmax entirely.
 *       for r in [0, kPackedRows):
 *           t = A_id[r]
 *           TLOAD bTile from B[r]
 *           TLOAD cTile from C[t]
 *           TADD  sumTile = cTile + bTile
 *           TSTORE C[t] = sumTile
 *
 *   else:
 *       // Pass 1: softmax(outVal) -> weights_scratch
 *       //   Composition mirrors tfa/pto_macro_fa_softmax.hpp lines 54-60:
 *       //   row-max -> broadcast-subtract -> exp -> row-sum -> broadcast-divide.
 *       TLOAD          valTile     from outVal[:, :kPadded]
 *       TROWMAX        maxTile     <- valTile                  // (kT, 8) per-row max
 *       TROWEXPANDSUB  tmpTile     <- valTile - maxTile        // broadcast subtract
 *       TEXP           expTile     <- exp(tmpTile)
 *       TROWSUM        sumTile     <- expTile                  // (kT, 8) per-row sum
 *       TROWEXPANDDIV  weightTile  <- expTile / sumTile        // softmax
 *       TSTORE         weights_scratch <- weightTile
 *
 *       // Pass 2: weighted scatter-add
 *       for r in [0, kPackedRows):
 *           t = A_id[r]
 *           k = rank_id[r]
 *           w = weights_scratch[t * kPadded + k]   // GM scalar read
 *           TLOAD  bTile      from B[r]
 *           TMULS  scaledTile = bTile * w
 *           TLOAD  cTile      from C[t]
 *           TADD   sumTile    = cTile + scaledTile
 *           TSTORE C[t]       = sumTile
 *
 * ===========================================================================
 * Tile-based softmax composition (kTopK > 1 only)
 * ===========================================================================
 *
 * The five-instruction softmax recipe is lifted from the manual-mode Flash
 * Attention softmax macro [tests/npu/a2a3/src/st/testcase/tfa/
 * pto_macro_fa_softmax.hpp:54-60]. Differences:
 *   - We add the final TROWEXPANDDIV step (FA uses online-softmax rescaling).
 *   - No manual `pipe_barrier(PIPE_V)` between stages — auto-mode is expected
 *     to insert the RAW edges; if it doesn't, that's the first thing to suspect
 *     if the kernel produces wrong numbers.
 *
 * The TROWEXPANDSUB / TROWEXPANDDIV broadcast-tile contract requires src1 to
 * have `validCol == 32 / sizeof(T) == 8` for fp32 row-major. TROWMAX/TROWSUM
 * emit exactly that shape (a per-row scalar in lane 0, replicated across all 8
 * lanes of a 32-byte block). See include/pto/npu/a2a3/TRowExpandSub.hpp:70-71.
 *
 * ===========================================================================
 * Why outVal cols kTopK..kPadded-1 are host-padded with -1e30
 * ===========================================================================
 *
 * For kTopK in {1, 2, 4}, kPadded = 8 (32-byte alignment for fp32 UB).
 * The valid softmax inputs are outVal[:, 0..kTopK). Cols kTopK..7 are
 * "padding" — but TROWMAX, TROWEXPANDSUB, TEXP, TROWSUM all read the full
 * tile width. If padding values were random, they'd corrupt the per-row
 * max (and through it, the softmax denominator and the weights).
 *
 * Filling padding with -1e30 (a very large negative) makes the softmax
 * pipeline neutralize them automatically:
 *     -1e30 - real_max  ≈ -inf
 *     exp(-inf)         = 0
 *     0 contributes nothing to the row sum.
 *     0 / real_sum      = 0
 * So padding cols of weights_scratch end up at 0.0, which is what we want —
 * rank_id from scatter only takes values in [0, kTopK), so we never look up
 * a padding column at gather time.
 *
 * The host (scripts/gen_data.py for standalone, or a small pad step in the
 * end-to-end driver) is responsible for the -1e30 fill. Both pass the same
 * pre-padded outVal blob to this kernel.
 *
 * ===========================================================================
 * Auto-mode constraints honored (mirror moe_top1_unpermute / FA softmax macro)
 * ===========================================================================
 *
 *   - Single AICORE (<<<1, nullptr, stream>>>).
 *   - Static row + softmax tiles declared once outside any loop.
 *   - pipe_barrier(PIPE_ALL) at the start of each row iteration in pass 2
 *     (same hardware-confirmed cross-iter auto-sync guard used in topk_kernel
 *     and the v1 gather).
 *   - GlobalTensor reconstructed per iteration with (base + runtime offset)
 *     for pass-2 row addresses.
 *   - No `TASSIGN`, no `Tile::data()` in kernel, no `*_IMPL` calls, no raw CCE
 *     intrinsics, no `Event<>`, no manual sync, no `TPipe`/`TPUSH`/`TPOP`,
 *     no double buffering, no A5-only ops.
 *
 * Limitations (v1):
 *   - Single AICORE; no block_idx parallelism.
 *   - fp32 only.
 *   - Static (kT, kPadded) softmax tiles — UB budget is ~96 KB at kT=256
 *     kPadded=16 (well under 192 KB) but ~192 KB at kT=512 kPadded=16
 *     (right at the UB limit; if it fails there, we chunk).
 *   - Trusts auto-mode RAW dependency analysis through the
 *     TROWMAX -> TROWEXPANDSUB -> TEXP -> TROWSUM -> TROWEXPANDDIV chain.
 *     The manual-mode FA macro adds `pipe_barrier(PIPE_V)` between phases;
 *     if auto-mode mis-handles this, weights will be wrong.
 *
 * Pattern sources:
 *   - tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp (softmax recipe)
 *   - kernels/automode/a2a3/moe_top1_unpermute/moe_top1_unpermute_kernel.cpp
 *     (row TLOAD->TADD->TSTORE skeleton, §11.5)
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace gather_cfg {

// v1 shape — must match scripts/gen_data.py and main.cpp.
constexpr unsigned kT    = 256;
constexpr unsigned kH    = 64;
constexpr unsigned kTopK = 1;

constexpr unsigned kPackedRows   = kT * kTopK;
constexpr unsigned kOverspillPad = 16;
constexpr unsigned kAlloc        = kPackedRows + kOverspillPad;

// Softmax tile column padding for 32-byte UB alignment.
// fp32 needs Cols * 4 % 32 == 0  ->  Cols % 8 == 0.
constexpr unsigned kPadded = (kTopK < 8) ? 8 : kTopK;

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

    using RowShape  = Shape <1, 1, 1, 1, kH>;
    using RowStride = Stride<1, 1, 1, kH, 1>;
    using RowGlobal = GlobalTensor<T, RowShape, RowStride>;

    using RowTile = Tile<TileType::Vec, T,
                         1, kH,
                         BLayout::RowMajor,
                         1, kH>;

    if constexpr (kTopK == 1) {
        // ============================================================
        // Fast path: softmax(single value) = 1.0, so weighted gather
        // degenerates to unweighted permute / accumulation.
        // Matches v1 gather byte-for-byte. outVal / rank_id /
        // weights_scratch params are unused on this path.
        // ============================================================
        (void)rank_id;
        (void)outVal;
        (void)weights_scratch;

        RowTile bTile;
        RowTile cTile;
        RowTile sumTile;

        for (unsigned r = 0; r < kPackedRows; ++r) {
            pipe_barrier(PIPE_ALL);

            int32_t t = A_id[r];                              // GM scalar read

            size_t src_off = static_cast<size_t>(r) * kH;
            size_t dst_off = static_cast<size_t>(t) * kH;

            RowGlobal bGlobal(B + src_off);
            RowGlobal cGlobal(C + dst_off);

            TLOAD(bTile, bGlobal);
            TLOAD(cTile, cGlobal);
            TADD (sumTile, cTile, bTile);
            TSTORE(cGlobal, sumTile);
        }
    } else {
        // ============================================================
        // Pass 1: softmax(outVal) -> weights_scratch.
        // Pure-tile recipe; see comment block at top of file for sourcing.
        // ============================================================
        using SoftmaxShape  = Shape <1, 1, 1, kT, kPadded>;
        using SoftmaxStride = Stride<kT * kPadded, kT * kPadded, kT * kPadded, kPadded, 1>;
        using SoftmaxGlobal = GlobalTensor<T, SoftmaxShape, SoftmaxStride>;

        using ValTile   = Tile<TileType::Vec, T, kT, kPadded, BLayout::RowMajor, kT, kPadded>;
        using BcastTile = Tile<TileType::Vec, T, kT, 8,       BLayout::RowMajor, kT, 8>;

        ValTile   valTile;
        BcastTile maxTile;
        ValTile   tmpTile;
        ValTile   expTile;
        BcastTile sumTile;
        ValTile   weightTile;

        SoftmaxGlobal outValGlobal (outVal);
        SoftmaxGlobal weightsGlobal(weights_scratch);

        TLOAD(valTile, outValGlobal);                  // (kT, kPadded) <- host-padded GM
        TROWMAX(maxTile, valTile, tmpTile);            // (kT, 8) broadcast max
        TROWEXPANDSUB(tmpTile, valTile, maxTile);      // val - max
        TEXP(expTile, tmpTile);                        // exp(val - max)
        TROWSUM(sumTile, expTile, tmpTile);            // (kT, 8) broadcast sum
        TROWEXPANDDIV(weightTile, expTile, sumTile);   // exp(...) / sum
        TSTORE(weightsGlobal, weightTile);             // -> GM scratch

        // ============================================================
        // Pass 2: weighted scatter-add. Per packed row:
        //   C[t] += weights_scratch[t * kPadded + k] * B[r]
        // ============================================================
        RowTile bTile;
        RowTile cTile;
        RowTile scaledTile;
        RowTile sumRowTile;

        for (unsigned r = 0; r < kPackedRows; ++r) {
            pipe_barrier(PIPE_ALL);

            int32_t t = A_id[r];                                   // GM scalar read
            int32_t k = rank_id[r];                                // GM scalar read
            T       w = weights_scratch[t * kPadded + k];          // GM scalar read

            size_t src_off = static_cast<size_t>(r) * kH;
            size_t dst_off = static_cast<size_t>(t) * kH;

            RowGlobal bGlobal(B + src_off);
            RowGlobal cGlobal(C + dst_off);

            TLOAD (bTile, bGlobal);
            TMULS (scaledTile, bTile, w);
            TLOAD (cTile, cGlobal);
            TADD  (sumRowTile, cTile, scaledTile);
            TSTORE(cGlobal, sumRowTile);
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
