// --------------------------------------------------------------------------------
// online_softmax — sparse_attn FA running max/exp/rescale/sum (auto-mode A3, vector).
//
// Op (kernel.py:331-340), per pipelined inner-block iteration:
//   scores_max_prev = scores_max
//   reduce_max(acc_s -> scores_max, dim=1, clear=False)
//   scores_scale = exp(scores_max_prev - scores_max)
//   acc_s = exp(acc_s - scores_max)
//   acc_s_cast = (BF16) acc_s
//   reduce_sum(acc_s -> scores_sum, dim=1)
//   sum_exp = sum_exp * scores_scale + scores_sum
//
// In/out semantics:
//   - acc_s (FP32, [H, BLOCK]) is updated in place (post-exp).
//   - scores_max (FP32, [H])   is updated in place (running max).
//   - sum_exp   (FP32, [H])    is updated in place (running denom).
//   - acc_s_cast (BF16, [H, BLOCK])    is written out (downstream PV).
//   - scores_scale (FP32, [H])        is written out (consumed by pv_matmul
//                                      to rescale acc_o BEFORE the next gemm).
//
// Mask handling: upstream qk_matmul does NOT apply -inf to masked columns.
// We fold the mask in here by passing the topk_idxs slice for this block;
// where idx == -1, we override acc_s[:, j] with a large negative bias before
// the running-max so the resulting exp underflows to ~0.
// (Pure -inf would be cleaner but FP32 -inf can poison the running max if
// every column in this block is masked; large-negative is robust.)
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

// Tile constants. UB-resident; the entire [H, BLOCK] FP32 tile + a few
// [H] FP32 vectors fit in UB by construction (H<=16, BLOCK<=64 in defaults).

extern "C" __global__ __aicore__ void online_softmax_kernel(
    __gm__ uint8_t *acc_s,         // FP32 [H, BLOCK]  — in/out (rewritten as exp())
    __gm__ uint8_t *acc_s_cast,    // BF16 [H, BLOCK]  — out
    __gm__ uint8_t *scores_max,    // FP32 [H]         — in/out (running max)
    __gm__ uint8_t *sum_exp,       // FP32 [H]         — in/out (running denom)
    __gm__ uint8_t *scores_scale,  // FP32 [H]         — out (consumed by pv_matmul)
    __gm__ uint8_t *topk_idxs,     // INT32 [BLOCK]    — in (slice for this block)
    uint32_t topk_len,             // = kCsaS
    uint32_t t_block)              // current pipelined-block index
{
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 vector online softmax.
    // Pattern anchored on add_tile_array (vector TLOAD/TSTORE) +
    // reduce primitives (TREDUCE_MAX / TREDUCE_SUM in pto/pto-inst.hpp).
    //
    //   GlobalTensor<float>     gS    (acc_s,        {kCsaH,     kCsaBlock});
    //   GlobalTensor<bfloat16_t>gSC   (acc_s_cast,   {kCsaH,     kCsaBlock});
    //   GlobalTensor<float>     gMax  (scores_max,   {kCsaH});
    //   GlobalTensor<float>     gSum  (sum_exp,      {kCsaH});
    //   GlobalTensor<float>     gScl  (scores_scale, {kCsaH});
    //   GlobalTensor<int32_t>   gIdx  (topk_idxs,    {kCsaBlock});
    //
    //   Tile<float,   MemUB> sTile;        // [H, BLOCK]
    //   Tile<bfloat16_t, MemUB> sCastTile; // [H, BLOCK]
    //   Tile<float,   MemUB> mPrev, m, scale, sumLocal, sumPrev;
    //   Tile<int32_t, MemUB> idxTile;      // [BLOCK]
    //
    //   TLOAD(sTile,    gS);
    //   TLOAD(mPrev,    gMax);
    //   TLOAD(sumPrev,  gSum);
    //   TLOAD(idxTile,  gIdx);
    //
    //   // 1) Apply mask: where idx[j]==-1, replace sTile[:, j] with -1e30f.
    //   for j in [0, kCsaBlock):
    //     if (idxTile[j] == -1 || t_block*kCsaBlock + j >= topk_len) {
    //       for i in [0, kCsaH): sTile[i, j] = -1.0e30f;
    //     }
    //
    //   // 2) Running max along block dim, NOT clearing prior:
    //   TREDUCE_MAX(sTile, m, axis=1);            // m = max over BLOCK
    //   m = max(m, mPrev);                         // running max
    //
    //   // 3) Rescale factor for acc_o (consumed by pv_matmul).
    //   scale = exp(mPrev - m);
    //
    //   // 4) Subtract new max + exp.
    //   for i, j: sTile[i, j] = exp(sTile[i, j] - m[i]);
    //
    //   // 5) Cast to BF16 for downstream PV matmul.
    //   sCastTile = cast<bfloat16_t>(sTile);
    //
    //   // 6) Reduce sum, update running denom.
    //   TREDUCE_SUM(sTile, sumLocal, axis=1);
    //   sumLocal = sumPrev * scale + sumLocal;
    //
    //   // 7) Spill all updated tiles back to GM.
    //   TSTORE(gS,   sTile);
    //   TSTORE(gSC,  sCastTile);
    //   TSTORE(gMax, m);
    //   TSTORE(gSum, sumLocal);
    //   TSTORE(gScl, scale);
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on add_tile_array_kernel.cpp for vector tile call shape; use
    // TREDUCE_* primitives from pto/pto-inst.hpp for the row reductions.
    (void)acc_s; (void)acc_s_cast; (void)scores_max; (void)sum_exp;
    (void)scores_scale; (void)topk_idxs; (void)topk_len; (void)t_block;
}

// Host launcher.
void launch_online_softmax(uint8_t *acc_s, uint8_t *acc_s_cast,
                           uint8_t *scores_max, uint8_t *sum_exp,
                           uint8_t *scores_scale, uint8_t *topk_idxs,
                           uint32_t topk_len, uint32_t t_block, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch online_softmax_kernel.
    (void)acc_s; (void)acc_s_cast; (void)scores_max; (void)sum_exp;
    (void)scores_scale; (void)topk_idxs; (void)topk_len; (void)t_block; (void)stream;
}
