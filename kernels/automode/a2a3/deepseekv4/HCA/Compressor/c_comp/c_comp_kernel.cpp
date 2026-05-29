// --------------------------------------------------------------------------------
// c_comp — Compressor weighted-sum reduction (auto-mode A3, vector).
//
//   kv_comp[b, sb, n] = sum_r kv[b, sb, r, n] * softmax_score[b, sb, r, n]
//     B    = kCompB
//     SB   = kCompS / kCompRatio
//     R    = (2 * kCompRatio) if kCompOverlap else kCompRatio
//     N    = kCompCoff * kCompHeadDim
//
// Reference: deepseek/model.py:343
//   `kv = (kv * score.softmax(dim=2)).sum(dim=2)`
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

extern "C" __global__ __aicore__ void c_comp_kernel(
    __gm__ uint8_t *out,            // FP32, (B, SB, N) row-major
    __gm__ uint8_t *kv,             // FP32, (B, SB, R, N) row-major
    __gm__ uint8_t *softmax_score)  // FP32, (B, SB, R, N) row-major
{
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 vector weighted-sum reduction.
    // Pattern anchors on add_tile_array_kernel.cpp + a fused
    // TMul / TReduceSum sequence.
    //
    //   GlobalTensor<float> gKv  (kv,            {B, SB, R, N});
    //   GlobalTensor<float> gS   (softmax_score, {B, SB, R, N});
    //   GlobalTensor<float> gOut (out,           {B, SB, N});
    //
    //   for (b = 0; b < B; ++b) {
    //     for (sb = 0; sb < SB; ++sb) {
    //       Tile<float, MemUB> tKv;       // [R, N]
    //       Tile<float, MemUB> tScore;    // [R, N]
    //       Tile<float, MemUB> tProd;     // [R, N]
    //       Tile<float, MemUB> tAcc;      // [1, N]
    //
    //       TLOAD(tKv,    gKv[b, sb, :, :]);
    //       TLOAD(tScore, gS [b, sb, :, :]);
    //       TMul (tProd, tKv, tScore);
    //       TReduceSum(tAcc, tProd, /*axis=*/0);   // sum along R, keep N
    //       TSTORE(gOut[b, sb, :], tAcc);
    //     }
    //   }
    // ----------------------------------------------------------------------
    // NOTE: API names (TMul, TReduceSum) must be confirmed against
    // include/pto/ headers before implementation.
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on add_tile_array_kernel.cpp (baseline auto-mode call shape)
    // and on any in-tree reduce kernel under kernels/automode/ for the
    // sanctioned TReduceSum form.
    (void)out; (void)kv; (void)softmax_score;
}

// Host launcher (kept minimal — the real launcher will go through pto-launch
// helpers similar to MoE/router_matmul/router_matmul_kernel.cpp).
void launch_c_comp(uint8_t *out, uint8_t *kv, uint8_t *softmax_score,
                   uint64_t B, uint64_t SB, uint64_t R, uint64_t N,
                   void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch c_comp_kernel.
    (void)out; (void)kv; (void)softmax_score;
    (void)B; (void)SB; (void)R; (void)N; (void)stream;
}
