// --------------------------------------------------------------------------------
// biased_softmax — Compressor APE-biased softmax (auto-mode A3, vector).
//
//   tmp[b, sb, r, n]            = score[b, sb, r, n] + ape[r, n]
//   softmax_score[b, sb, :, n]  = softmax_along_dim2(tmp[b, sb, :, n])
//     B    = kCompB
//     SB   = kCompS / kCompRatio                       (compressed seq len)
//     R    = (2 * kCompRatio) if kCompOverlap else kCompRatio
//     N    = kCompCoff * kCompHeadDim
//
// Reference: deepseek/model.py:339 (`score + self.ape`) and model.py:343
// (`(kv * score.softmax(dim=2))`).  The overlap_transform `-inf` mask
// (model.py:342) is intentionally NOT modelled here — see README §Status.
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

// Shape aliases (from ../build/generated_cases.h).
//   kCompB, kCompS, kCompHeadDim, kCompRatio, kCompOverlap, kCompCoff
// R = 2 * kCompRatio when overlap; N = kCompCoff * kCompHeadDim.

extern "C" __global__ __aicore__ void biased_softmax_kernel(
    __gm__ uint8_t *out,    // FP32, (B, SB, R, N) row-major
    __gm__ uint8_t *score,  // FP32, (B, SB, R, N) row-major
    __gm__ uint8_t *ape)    // FP32, (R, N) row-major
{
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 vector biased softmax.
    // Pattern anchors on add_tile_array_kernel.cpp (vec auto baseline) plus
    // the sanctioned TReduce* + TBroadcast pattern; see
    // docs_for_ai/auto_mode_bad_patterns.md for "vector reduction" rules.
    //
    //   GlobalTensor<float> gScore (score, {B, SB, R, N});
    //   GlobalTensor<float> gApe   (ape,   {R, N});
    //   GlobalTensor<float> gOut   (out,   {B, SB, R, N});
    //
    //   // ape is reused across (B, SB) — load once into UB.
    //   Tile<float, MemUB> tApe;
    //   TLOAD(tApe, gApe);
    //
    //   for (b = 0; b < B; ++b) {
    //     for (sb = 0; sb < SB; ++sb) {
    //       Tile<float, MemUB> tScore;       // [R, N]
    //       Tile<float, MemUB> tBiased;      // [R, N]
    //       Tile<float, MemUB> tRowMax;      // [R, 1]   reduction-max along N? NO — along R
    //       Tile<float, MemUB> tRowSum;      // [R, 1]
    //
    //       TLOAD(tScore, gScore[b, sb, :, :]);
    //       TADD(tBiased, tScore, tApe);     // score + ape (ape broadcast across (B, SB))
    //
    //       // softmax along the R axis (axis 2 in the python view):
    //       //   max over R for each (n) column → reduce-axis-0 of [R, N]
    //       //   exp(x - max)
    //       //   sum over R for each (n) column
    //       //   divide by sum
    //       TReduceMax(tRowMax, tBiased, /*axis=*/0);   // [1, N]
    //       TSub      (tBiased, tBiased, tRowMax);      // broadcast along R
    //       TExp      (tBiased, tBiased);
    //       TReduceSum(tRowSum, tBiased, /*axis=*/0);   // [1, N]
    //       TDiv      (tBiased, tBiased, tRowSum);      // broadcast along R
    //
    //       TSTORE(gOut[b, sb, :, :], tBiased);
    //     }
    //   }
    // ----------------------------------------------------------------------
    // NOTE: the actual API names (TReduceMax, TExp, TSub, TDiv) must be
    // confirmed against include/pto/ headers before implementation; this
    // skeleton uses placeholder names.
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on add_tile_array_kernel.cpp for the baseline auto-mode call
    // shape and on any in-tree softmax (search `softmax` under kernels/automode).
    (void)out; (void)score; (void)ape;
}

// Host launcher (kept minimal — the real launcher will go through pto-launch
// helpers similar to MoE/router_matmul/router_matmul_kernel.cpp).
void launch_biased_softmax(uint8_t *out, uint8_t *score, uint8_t *ape,
                           uint64_t B, uint64_t SB, uint64_t R, uint64_t N,
                           void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch
    // biased_softmax_kernel.
    (void)out; (void)score; (void)ape;
    (void)B; (void)SB; (void)R; (void)N; (void)stream;
}
