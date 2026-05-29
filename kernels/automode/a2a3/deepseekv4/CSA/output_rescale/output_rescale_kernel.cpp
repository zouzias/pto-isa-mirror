// --------------------------------------------------------------------------------
// output_rescale — sparse_attn final /sum_exp + attn_sink tail (auto-mode A3, vector).
//
// Op (kernel.py:345-350), runs ONCE per (b, m) after the pipelined loop:
//   for i in T.Parallel(h):
//       sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
//   for i, j in T.Parallel(h, d):
//       acc_o[i, j] /= sum_exp[i]
//   T.copy(acc_o, o_shared)                # FP32 -> BF16 cast
//   T.copy(o_shared, o[by, bx, :, :])      # GM write
//
// This leaf represents the TAIL of the pipelined loop, i.e. it runs AFTER
// the last `t` iteration. The driver provides:
//   - acc_o    : the post-loop FP32 accumulator
//   - scores_max, sum_exp : the post-loop running statistics
//   - attn_sink : per-head learnable bias (kernel.py:298)
//
// Output: BF16 `o[H, D]` ready to be written into `o[by, bx, :, :]`.
//
// DESIGN / SKELETON ONLY — pseudocode body below. Real implementation is
// deferred. Do not claim compile or runtime success until run.sh produces
// `test data success` on the compiler server.
// --------------------------------------------------------------------------------

#include <cstdint>
#include "common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
#include "generated_cases.h"

using namespace pto;

extern "C" __global__ __aicore__ void output_rescale_kernel(
    __gm__ uint8_t *o,            // BF16 [H, D]  — out (final o tile for (by, bx))
    __gm__ uint8_t *acc_o,        // FP32 [H, D]  — in  (post-loop accumulator)
    __gm__ uint8_t *scores_max,   // FP32 [H]     — in  (final running max)
    __gm__ uint8_t *sum_exp,      // FP32 [H]     — in/out (sink-adjusted denom)
    __gm__ uint8_t *attn_sink)    // FP32 [H]     — in  (learnable per-head bias)
{
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 vector finalize.
    // Pattern anchored on add_tile_array (vector TLOAD/TSTORE) + scalar
    // per-head reduce_one ops (exp, division).
    //
    //   GlobalTensor<float>     gAcc   (acc_o,      {kCsaH, kCsaD});
    //   GlobalTensor<float>     gMax   (scores_max, {kCsaH});
    //   GlobalTensor<float>     gSum   (sum_exp,    {kCsaH});
    //   GlobalTensor<float>     gSink  (attn_sink,  {kCsaH});
    //   GlobalTensor<bfloat16_t>gO     (o,          {kCsaH, kCsaD});
    //
    //   Tile<float, MemUB> accTile;       // [H, D]
    //   Tile<float, MemUB> maxV, sumV, sinkV;  // [H]
    //   Tile<bfloat16_t, MemUB> oTile;    // [H, D]
    //
    //   TLOAD(accTile, gAcc);
    //   TLOAD(maxV,   gMax);
    //   TLOAD(sumV,   gSum);
    //   TLOAD(sinkV,  gSink);
    //
    //   // 1) sum_exp += exp(attn_sink - scores_max)
    //   sumV = sumV + exp(sinkV - maxV);
    //
    //   // 2) acc_o /= sum_exp (per-head broadcast division along D).
    //   for i in [0, kCsaH):
    //     for j in [0, kCsaD):
    //       accTile[i, j] = accTile[i, j] / sumV[i];
    //
    //   // 3) BF16 cast for the GM write.
    //   oTile = cast<bfloat16_t>(accTile);
    //
    //   // 4) Spill.
    //   TSTORE(gO,   oTile);
    //   TSTORE(gSum, sumV);   // sink-adjusted denom (for debug only)
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on add_tile_array_kernel.cpp for vector tile call shape; the
    // per-head broadcast division is the only non-elementwise step — use a
    // broadcast/reshape primitive from pto/pto-inst.hpp (TBROADCAST or
    // explicit per-row loop, whichever the auto-mode pass prefers).
    (void)o; (void)acc_o; (void)scores_max; (void)sum_exp; (void)attn_sink;
}

// Host launcher.
void launch_output_rescale(uint8_t *o, uint8_t *acc_o, uint8_t *scores_max,
                           uint8_t *sum_exp, uint8_t *attn_sink, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch output_rescale_kernel.
    (void)o; (void)acc_o; (void)scores_max; (void)sum_exp; (void)attn_sink;
    (void)stream;
}
