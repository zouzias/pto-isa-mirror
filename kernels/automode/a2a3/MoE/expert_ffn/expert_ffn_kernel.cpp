/**
 * expert_ffn_kernel.cpp - auto-mode A3 prototype.
 *
 * Per-token element-wise expert FFN:
 *   output[t] = relu(x_packed[t] * w1[e]) * w2[e]
 * where * is element-wise multiplication (Hadamard product), and e is the
 * expert assigned to token t. This is a gated activation pattern: w1 and w2
 * are per-expert feature scale vectors of shape (kE, kD).
 *
 * Tokens are pre-sorted (packed) by expert assignment using the same
 * packed-token convention as moe_segmented_gemm_relu. expert_count[e] and
 * expert_start[e] give each expert's slice in x_packed / output.
 *
 * All operands are float32. No cube path (no TMATMUL); pure Vec pipeline.
 *
 * Pipeline per token t assigned to expert e:
 *   TLOAD  w1[e]          -- expert gate weights (reloaded per token,
 *   TLOAD  w2[e]             same as moe_segmented_gemm_relu bMatTile reload)
 *   TLOAD  x_packed[t]    -- token features
 *   TMUL   h    = x * w1  -- element-wise gating
 *   TMAXS  relu = max(h, 0.0f)   -- ReLU activation
 *   TMUL   out  = relu * w2      -- output gate
 *   TSTORE output[t]
 *
 * Tiles declared OUTSIDE both loops (moe_segmented_gemm_relu / add_tile_array
 * pattern). Inside-inner-loop placement (v1) produced 94% error — suspected
 * cross-iter auto-sync gap in nested loop context; pipe_barrier(PIPE_ALL) at
 * the start of the inner token loop guards this.
 *
 * Limitations (v1):
 *   - Single AICORE; no block_idx work split.
 *   - One token per Vec tile (kD=64 elements; no row-batching).
 *   - w1/w2 reloaded every token iteration (correctness-first).
 *   - All float32; no FP16 path.
 *   - No double-buffering.
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace expert_ffn_cfg {
constexpr unsigned kD          = 64;  // feature dimension
constexpr unsigned kNumExperts = 32;  // number of experts
}  // namespace expert_ffn_cfg

__global__ AICORE void runExpertFfn(
    __gm__ uint8_t *output_raw,
    __gm__ uint8_t *x_packed_raw,
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ uint8_t *w1_raw,
    __gm__ uint8_t *w2_raw)
{
    using namespace expert_ffn_cfg;
    using T = float;

    __gm__ T *output   = reinterpret_cast<__gm__ T *>(output_raw);
    __gm__ T *x_packed = reinterpret_cast<__gm__ T *>(x_packed_raw);
    __gm__ T *w1       = reinterpret_cast<__gm__ T *>(w1_raw);
    __gm__ T *w2       = reinterpret_cast<__gm__ T *>(w2_raw);

    using TokenGlobal = GlobalTensor<T, Shape<1, 1, 1, 1, kD>, Stride<1, 1, 1, kD, 1>>;
    using WGlobal     = GlobalTensor<T, Shape<1, 1, 1, 1, kD>, Stride<1, 1, 1, kD, 1>>;

    // Vec tile: (1, kD) float32, static valid region.
    using VecTile = Tile<TileType::Vec, T, 1, kD, BLayout::RowMajor, 1, kD>;

    // Tiles declared OUTSIDE both loops (moe_segmented_gemm_relu / add_tile_array
    // pattern, confirmed-working for nested loops). Inside-inner-loop placement
    // caused 94% error in v1: suspected cross-iter auto-sync gap in nested context.
    VecTile xTile;
    VecTile w1Tile;
    VecTile w2Tile;
    VecTile hTile;
    VecTile reluTile;
    VecTile outTile;

    for (unsigned e = 0; e < kNumExperts; ++e) {
        int32_t start = expert_start[e];
        int32_t count = expert_count[e];

        // w1/w2 global views constructed once per expert; reused across tokens.
        // Same pattern as moe_segmented_gemm_relu bGlobal (confirmed-working).
        WGlobal w1Global(w1 + static_cast<size_t>(e) * kD);
        WGlobal w2Global(w2 + static_cast<size_t>(e) * kD);

        for (int32_t t = 0; t < count; ++t) {
            // pipe_barrier at token-loop start: guards cross-iter auto-sync gap.
            // Same hardware-confirmed requirement as topk row loop.
            pipe_barrier(PIPE_ALL);

            size_t tokOff = static_cast<size_t>(start + t) * kD;
            TokenGlobal xGlobal(x_packed + tokOff);
            TokenGlobal outGlobal(output   + tokOff);

            // Reload w1/w2 every iteration from the same expert view
            // (same w1[e], w2[e] data; matches moe_segmented_gemm_relu bMatTile reload).
            TLOAD(w1Tile, w1Global);
            TLOAD(w2Tile, w2Global);
            TLOAD(xTile,  xGlobal);

            TMUL(hTile,    xTile,   w1Tile);   // h    = x * w1
            TMAXS(reluTile, hTile,  0.0f);     // relu = max(h, 0)
            TMUL(outTile, reluTile, w2Tile);   // out  = relu * w2

            TSTORE(outGlobal, outTile);
        }
    }
}

void launchExpertFfn(uint8_t *output, uint8_t *x_packed,
                     int32_t *expert_count, int32_t *expert_start,
                     uint8_t *w1, uint8_t *w2, void *stream)
{
    runExpertFfn<<<1, nullptr, stream>>>(output, x_packed, expert_count, expert_start, w1, w2);
}

// extern "C" wrapper for consistency with the other MoE kernels.
extern "C" void launchExpertFfnFloat(uint8_t *output, uint8_t *x_packed,
                                      int32_t *expert_count, int32_t *expert_start,
                                      uint8_t *w1, uint8_t *w2, void *stream)
{
    launchExpertFfn(output, x_packed, expert_count, expert_start, w1, w2, stream);
}
