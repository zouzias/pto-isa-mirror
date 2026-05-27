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
 * Outputs  (GM): C                [kT, kH]            fp32  (zero-initialized by host; only required for kTopK > 1)
 *
 *   kPadded = max(8, kTopK) — softmax tile column padding for 32-byte UB alignment.
 *
 * ===========================================================================
 * Algorithm overview
 * ===========================================================================
 *
 *   if constexpr (kTopK == 1):
 *       // Fast path: softmax of a single value is always 1.0, and each token
 *       // has exactly one packed row. The gather degenerates to a row reorder
 *       // and A_id is a permutation of [0, kT) -> output rows are disjoint.
 *       for r in [0, kPackedRows):
 *           t = A_id[r]
 *           TLOAD bTile from B[r]
 *           TSTORE C[t] = bTile
 *
 *   else:
 *       // Pass 1: softmax(outVal) -> weights_scratch
 *       //   Composition mirrors tfa/pto_macro_fa_softmax.hpp lines 54-60:
 *       //   row-max -> broadcast-subtract -> exp -> row-sum -> broadcast-divide.
 *       TLOAD          valTile     from outVal[:, :kPadded]
 *       TROWMAX        maxTile     <- valTile                  // (kT, 1) per-row max
 *       TROWEXPANDSUB  tmpTile     <- valTile - maxTile        // broadcast subtract
 *       TEXP           expTile     <- exp(tmpTile)
 *       TROWSUM        sumTile     <- expTile                  // (kT, 1) per-row sum
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
 * Parallelism (v2): vec-subcore SPMD via (block_idx, subblockid)
 * ===========================================================================
 *
 * Vec arch on A3: each AI core hosts 2 independent vec subcores; we launch
 * `kBlockDim` AI cores and use `get_subblockid()` to distinguish the two
 * vec subcores inside one AI core. The unique subcore id is:
 *
 *     uid = block_idx * 2 + subblockid          // [0, kBlockDim*2)
 *
 * `kTotalSubcores = kBlockDim * 2` and is the number of work slices. The
 * launch dim is chosen so `kTotalSubcores <= min(kT, kMaxVecSubcores=48)`.
 *
 *   - kTopK == 1: split the packed-rows axis [0, kPackedRows) across all
 *     subcores. A_id is a permutation when kTopK==1, so output rows written
 *     by different subcores are disjoint. No cross-subcore sync needed.
 *
 *   - kTopK > 1: split the token axis [0, kT) across all subcores.
 *       - Pass 1: each subcore computes softmax for its token slice and
 *         writes the corresponding rows of weights_scratch. Slices are
 *         disjoint, so weights_scratch writes don't race.
 *       - Pass 2: each subcore scans all r in [0, kPackedRows) but gates
 *         on `tStart <= A_id[r] < tEnd`. The full scan keeps the loop bound
 *         compile-time and avoids needing a reverse (t -> r) map; for matched
 *         iters it does the TLOAD/TMULS/TLOAD/TADD/TSTORE chain. C[t] writes
 *         are partitioned by t-range across subcores, so no cross-subcore
 *         race on C either.
 *
 * ===========================================================================
 * Pipeline (v2): MultiBuffered<2> on kTopK == 1 row loop
 * ===========================================================================
 *
 * For the kTopK == 1 fast path, the per-row body is TLOAD(b) -> TSTORE(C[t]).
 * Adjacent iterations have disjoint B source rows and disjoint C dest rows
 * (A_id is a permutation), so we wrap the inner-row body in
 * `MultiBuffered<2>::loop<Range<kRowsPerSubcoreMax>>` to ping-pong bTile across
 * two L1 lanes. This overlaps MTE2 (TLOAD of iter N+1) with MTE3 (TSTORE of
 * iter N). The Range upper bound is the worst-case subcore slice size; the
 * lambda gates on `i >= myRows` for subcores with a shorter slice.
 *
 * The kTopK > 1 pass-2 loop is NOT multi-buffered: it does a read-modify-write
 * on C[t] and within one subcore the same `t` appears `kTopK` times across
 * the scanned `r` iterations. Ping-pong on (b, c)Tile would let iter N+1's
 * TLOAD(C[t]) issue before iter N's TSTORE(C[t]) committed, racing the RMW
 * when adjacent gated iterations alias on `t`. Keeping pass 2 single-buffered
 * defers ordering to auto-mode's default per-tile dataflow analysis (the same
 * behavior the v1 kernel relied on).
 *
 * ===========================================================================
 * Host/AICORE separation
 * ===========================================================================
 *
 * `gather_cfg::kBlockDim` is a plain `constexpr unsigned` (no AICORE-qualified
 * helpers), so the host launchGather can read it directly to size the launch
 * grid. The kernel uses the same constant — no duplicate host mirror needed.
 *
 * ===========================================================================
 * Auto-mode constraints honored
 * ===========================================================================
 *
 *   - Static row tile (Tile<Vec, T, 1, kH, RowMajor, 1, kH>) declared inside
 *     the multi-buffered lambda (kTopK == 1) so each lane gets its own buffer.
 *   - Dynamic-valid softmax tiles (kTopK > 1 pass 1) sized for the worst-case
 *     per-subcore slice and constructed with the runtime row count.
 *   - GlobalTensor reconstructed per iteration with (base + runtime offset)
 *     for pass-2 row addresses.
 *   - No `TASSIGN`, no `Tile::data()` in kernel, no `*_IMPL` calls, no raw CCE
 *     intrinsics, no `Event<>`, no manual sync, no `TPipe`/`TPUSH`/`TPOP`.
 *     The only sanctioned pipeline construct is the `MultiBuffered` helper.
 *
 * Limitations:
 *   - fp32 only.
 *   - kT==0 is rejected by static_assert.
 *
 * Pattern sources:
 *   - tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp (softmax recipe)
 *   - kernels/automode/a2a3/MoE/router_matmul/router_matmul_kernel.cpp
 *     (MultiBuffered<2>::loop + block_idx SPMD pattern)
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "generated_cases.h"
#include "multiBuffer.hpp"

using namespace pto;
using namespace pto_auto;

// Number of pipeline lanes for the multi-buffered row loop on the kTopK==1
// fast path. 2 = ping-pong; MultiBuffered emits a `#pragma pto v_loop_barrier`
// between lanes so the auto-mode compiler allocates one buffer per lane.
constexpr int kNumBuffers = 2;

namespace gather_cfg {

// Input-shape constants pulled from generated_cases.h (single-case per binary).
constexpr unsigned kT    = kMoeT;
constexpr unsigned kH    = kMoeH;
constexpr unsigned kTopK = kMoeTopK;

static_assert(kT > 0, "kT must be > 0.");

constexpr unsigned kPackedRows   = kT * kTopK;
constexpr unsigned kOverspillPad = 16;
constexpr unsigned kAlloc        = kPackedRows + kOverspillPad;

// Softmax tile column padding for 32-byte UB alignment.
// fp32 needs Cols * 4 % 32 == 0  ->  Cols % 8 == 0.
constexpr unsigned kPadded = (kTopK < 8) ? 8 : kTopK;

// A3 vec-subcore SPMD. Each AI core hosts 2 vec subcores; we launch
// kBlockDim AI cores and use (block_idx, subblockid) to address the
// kTotalSubcores work slices.
constexpr unsigned kMaxAICores            = 24;
constexpr unsigned kVecSubcoresPerAICore  = 2;
constexpr unsigned kMaxVecSubcores        = kMaxAICores * kVecSubcoresPerAICore;  // 48

// We split work along the token axis (or packed-rows axis for kTopK == 1).
// Both reduce to "at most kT independent slices", so we cap at min(kT, 48)
// and round up to a pair so kBlockDim is integral. If the rounding leaves an
// extra (kTotalSubcores > kT), the trailing subcore's slice is empty and its
// loop bodies are gated out at runtime.
constexpr unsigned kDesiredSubcores =
    (kT < kMaxVecSubcores) ? kT : kMaxVecSubcores;
constexpr unsigned kBlockDim =
    (kDesiredSubcores + kVecSubcoresPerAICore - 1) / kVecSubcoresPerAICore;
constexpr unsigned kTotalSubcores = kBlockDim * kVecSubcoresPerAICore;

// Worst-case per-subcore slice size — used as the compile-time upper bound
// for the multi-buffered Range and for the static allocation of dynamic-valid
// softmax tiles.
constexpr unsigned kRowsPerSubcoreMax =
    (kPackedRows + kTotalSubcores - 1) / kTotalSubcores;
constexpr unsigned kTokensPerSubcoreMax =
    (kT + kTotalSubcores - 1) / kTotalSubcores;

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

    // Unique vec-subcore id across the full launch. kBlockDim AI cores × 2
    // vec subcores per core = kTotalSubcores slices.
    const unsigned bid = static_cast<unsigned>(get_block_idx());
    const unsigned vid = static_cast<unsigned>(get_subblockid());
    const unsigned uid = bid * kVecSubcoresPerAICore + vid;

    if constexpr (kTopK == 1) {
        // ============================================================
        // Fast path: softmax(single value) = 1.0, and each token has exactly
        // one packed row. This is a pure row reorder; C does not need to be
        // read because there is nothing to accumulate. A_id is a permutation
        // when kTopK == 1, so subcores writing disjoint r-ranges also write
        // disjoint C[t] rows — no cross-subcore race on C.
        // ============================================================
        (void)rank_id;
        (void)outVal;
        (void)weights_scratch;

        const unsigned rStart = (uid * kPackedRows) / kTotalSubcores;
        const unsigned rEnd   = ((uid + 1) * kPackedRows) / kTotalSubcores;
        const unsigned myRows = rEnd - rStart;

        // Multi-buffered row loop. bTile is declared inside the lambda so
        // each ping-pong lane owns its own L1 buffer; MTE2 (TLOAD) of lane
        // N+1 overlaps MTE3 (TSTORE) of lane N. Range upper bound is the
        // worst-case per-subcore slice; shorter slices are gated out.
        MultiBuffered<kNumBuffers> row_db;
        row_db.loop<Range<kRowsPerSubcoreMax>>([&](auto ctx) {
            const unsigned i = static_cast<unsigned>(ctx.iter);
            if (i >= myRows) return;
            const unsigned r = rStart + i;

            int32_t t = A_id[r];                              // GM scalar read

            size_t src_off = static_cast<size_t>(r) * kH;
            size_t dst_off = static_cast<size_t>(t) * kH;

            RowGlobal bGlobal(B + src_off);
            RowGlobal cGlobal(C + dst_off);

            RowTile bTile;
            TLOAD(bTile, bGlobal);
            TSTORE(cGlobal, bTile);
        });
    } else {
        // ============================================================
        // kTopK > 1 path. SPMD split on the token axis kT:
        //   Pass 1: each subcore computes softmax for tokens [tStart, tEnd)
        //           and writes its slice of weights_scratch.
        //   Pass 2: each subcore scans r in [0, kPackedRows) and processes
        //           only the rows whose A_id falls in [tStart, tEnd).
        // C[t] writes are partitioned by t-range across subcores, so cross-
        // subcore C races do not occur. Pass 2 is NOT multi-buffered: within
        // one subcore the same t appears kTopK times across r, and pinging
        // the (b,c)Tile lanes would let iter N+1 issue TLOAD(C[t]) before
        // iter N's TSTORE(C[t]) committed.
        // ============================================================
        const unsigned tStart = (uid * kT) / kTotalSubcores;
        const unsigned tEnd   = ((uid + 1) * kT) / kTotalSubcores;
        const unsigned myT    = tEnd - tStart;

        // Per-subcore softmax tile: allocate for the worst-case slice size,
        // use dynamic valid rows for the actual runtime count.
        using SoftmaxShape  = Shape <1, 1, 1, DYNAMIC, kPadded>;
        using SoftmaxStride = Stride<kT * kPadded, kT * kPadded, kT * kPadded, kPadded, 1>;
        using SoftmaxGlobal = GlobalTensor<T, SoftmaxShape, SoftmaxStride>;

        using ValTile   = Tile<TileType::Vec, T, kTokensPerSubcoreMax, kPadded,
                               BLayout::RowMajor, DYNAMIC, kPadded>;
        using BcastTile = Tile<TileType::Vec, T, kTokensPerSubcoreMax, 1,
                               BLayout::ColMajor, DYNAMIC, 1>;

        // Pass 1: softmax slice. Guarded by myT > 0 because the rounding-up
        // of kBlockDim can leave a tail subcore with an empty slice.
        if (myT > 0) {
            ValTile   valTile(myT);
            BcastTile maxTile(myT);
            ValTile   tmpTile(myT);
            ValTile   expTile(myT);
            BcastTile sumTile(myT);
            ValTile   weightTile(myT);

            size_t softOff = static_cast<size_t>(tStart) * kPadded;
            SoftmaxShape softShape(myT);
            SoftmaxGlobal outValGlobal (outVal          + softOff, softShape);
            SoftmaxGlobal weightsGlobal(weights_scratch + softOff, softShape);

            TLOAD(valTile, outValGlobal);                  // (myT, kPadded) <- host-padded GM
            TROWMAX(maxTile, valTile, tmpTile);            // (myT, 1) row max
            TROWEXPANDSUB(tmpTile, valTile, maxTile);      // val - max
            TEXP(expTile, tmpTile);                        // exp(val - max)
            TROWSUM(sumTile, expTile, tmpTile);            // (myT, 1) row sum
            TROWEXPANDDIV(weightTile, expTile, sumTile);   // exp(...) / sum
            TSTORE(weightsGlobal, weightTile);             // -> GM scratch slice
        }

        // ============================================================
        // Pass 2: weighted scatter-add over the subcore's t-range.
        // Each subcore scans the full packed-rows axis and gates on the
        // runtime check `tStart <= A_id[r] < tEnd`. Matched rows do the
        // TLOAD/TMULS/TLOAD/TADD/TSTORE chain into C[t]. No multi-buffer
        // here — see comment block above on the cross-iter RMW hazard.
        // ============================================================
        RowTile bTile;
        RowTile cTile;
        RowTile scaledTile;
        RowTile sumRowTile;

        for (unsigned r = 0; r < kPackedRows; ++r) {
            int32_t t = A_id[r];                                   // GM scalar read
            if (static_cast<unsigned>(t) < tStart || static_cast<unsigned>(t) >= tEnd) {
                continue;
            }
            int32_t k = rank_id[r];                                // GM scalar read
            T       w = weights_scratch[static_cast<size_t>(t) * kPadded + k];  // GM scalar read

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
    runGather<T><<<gather_cfg::kBlockDim, nullptr, stream>>>(
        C, B, A_id, rank_id, outVal, weights_scratch);
}

template void launchGather<float>(float *C, float *B,
                                  int32_t *A_id, int32_t *rank_id,
                                  float *outVal, float *weights_scratch,
                                  void *stream);
