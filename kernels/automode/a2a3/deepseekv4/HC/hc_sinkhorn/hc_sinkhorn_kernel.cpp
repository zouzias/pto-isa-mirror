// --------------------------------------------------------------------------------
// hc_sinkhorn - split + sigmoid + softmax + Sinkhorn iterations (auto-mode A3, vector).
//
// Per-row of `mixes [N, MIX_HC]` (N = B*S, MIX_HC = (2+HC_MULT)*HC_MULT):
//   pre[i, :HC_MULT]              = sigmoid(mixes[i, :HC_MULT]      * hc_scale[0]
//                                           + hc_base[:HC_MULT])    + eps
//   post[i, :HC_MULT]             = 2 * sigmoid(mixes[i, HC_MULT:2*HC_MULT]
//                                              * hc_scale[1]
//                                              + hc_base[HC_MULT:2*HC_MULT])
//   comb_frag[HC_MULT, HC_MULT]   = (mixes[i, 2*HC_MULT:] reshape [HC_MULT, HC_MULT])
//                                   * hc_scale[2] + hc_base[2*HC_MULT:].reshape(...)
//
//   row-softmax + eps -> col-normalize -> (sinkhorn_iters - 1) x
//       (row-normalize, col-normalize)
//   comb[i, :, :] = comb_frag
//
// Reference: deepseek/kernel.py:371-427 (hc_split_sinkhorn_kernel).
//
// DESIGN / SKELETON ONLY - pseudocode body below. Real implementation is
// deferred. Do not claim compile or runtime success until run.sh produces
// `test data success` on the compiler server.
//
// With HC_MULT=4 (model default), comb is 4x4 and the entire per-row working
// set fits comfortably in UB. The Sinkhorn loop has no inter-iteration sync
// requirement under auto mode.
// --------------------------------------------------------------------------------

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "generated_cases.h"

using namespace pto;

namespace hc_sinkhorn_cfg {

constexpr unsigned kB     = static_cast<unsigned>(kHcB);
constexpr unsigned kS     = static_cast<unsigned>(kHcS);
constexpr unsigned kHc    = static_cast<unsigned>(kHcMult);
constexpr unsigned kMixHc = static_cast<unsigned>(kHcMixHc);
constexpr unsigned kIters = static_cast<unsigned>(kHcSinkhornIters);
constexpr unsigned kN     = kB * kS;

// Per-row scalar epsilon. kHcEps comes from generated_cases.h as a static
// constexpr float; replicate locally to keep this TU self-contained.
constexpr float kEps = kHcEps;

// Static assert: kMixHc must equal (2 + kHc) * kHc by model construction.
static_assert(kMixHc == (2u + kHc) * kHc,
              "MIX_HC must equal (2 + HC_MULT) * HC_MULT");

}  // namespace hc_sinkhorn_cfg

extern "C" __global__ AICORE void hc_sinkhorn_kernel(
    __gm__ uint8_t *pre_raw,        // FP32 [N, HC_MULT]
    __gm__ uint8_t *post_raw,       // FP32 [N, HC_MULT]
    __gm__ uint8_t *comb_raw,       // FP32 [N, HC_MULT, HC_MULT]
    __gm__ uint8_t *mixes_raw,      // FP32 [N, MIX_HC]
    __gm__ uint8_t *hc_scale_raw,   // FP32 [3]
    __gm__ uint8_t *hc_base_raw)    // FP32 [MIX_HC]
{
    using namespace hc_sinkhorn_cfg;

    // ----------------------------------------------------------------------
    // PSEUDOCODE - auto-mode A3 vector kernel mirroring the TileLang
    // hc_split_sinkhorn_kernel (kernel.py:371-427). Pattern anchors on
    // add_tile_array_kernel.cpp (TLOAD / vector-op / TSTORE) for the
    // smallest-known-good auto-mode vector shape.
    //
    //   __gm__ float *pre      = reinterpret_cast<__gm__ float *>(pre_raw);
    //   __gm__ float *post     = reinterpret_cast<__gm__ float *>(post_raw);
    //   __gm__ float *comb     = reinterpret_cast<__gm__ float *>(comb_raw);
    //   __gm__ float *mixes    = reinterpret_cast<__gm__ float *>(mixes_raw);
    //   __gm__ float *hc_scale = reinterpret_cast<__gm__ float *>(hc_scale_raw);
    //   __gm__ float *hc_base  = reinterpret_cast<__gm__ float *>(hc_base_raw);
    //
    //   // GlobalTensor views; per-row tiles loaded into UB inside the loop.
    //   GlobalTensor<float> gMixes   (mixes,    {kN, kMixHc});
    //   GlobalTensor<float> gPre     (pre,      {kN, kHc});
    //   GlobalTensor<float> gPost    (post,     {kN, kHc});
    //   GlobalTensor<float> gComb    (comb,     {kN, kHc, kHc});
    //   GlobalTensor<float> gScale   (hc_scale, {3});
    //   GlobalTensor<float> gBase    (hc_base,  {kMixHc});
    //
    //   // Hoist scale + base outside the row loop (one TLOAD each).
    //   Tile<float, MemUB> ubScale;   // [3]
    //   Tile<float, MemUB> ubBase;    // [kMixHc]
    //   TLOAD(ubScale, gScale);
    //   TLOAD(ubBase,  gBase);
    //
    //   for (unsigned i = 0; i < kN; ++i) {
    //     Tile<float, MemUB> ubMixes;    // [kMixHc]
    //     Tile<float, MemUB> ubPre;      // [kHc]
    //     Tile<float, MemUB> ubPost;     // [kHc]
    //     Tile<float, MemUB> ubComb;     // [kHc, kHc]
    //     Tile<float, MemUB> ubRowMax;   // [kHc]
    //     Tile<float, MemUB> ubRowSum;   // [kHc]
    //     Tile<float, MemUB> ubColSum;   // [kHc]
    //
    //     TLOAD(ubMixes, gMixes[i, :]);
    //
    //     // ---- pre = sigmoid(mixes[:hc] * scale[0] + base[:hc]) + eps -----
    //     // ubPre = ubMixes[0:kHc] * scale[0] + ubBase[0:kHc]
    //     // ubPre = TSIGMOID(ubPre)
    //     // ubPre = ubPre + eps
    //
    //     // ---- post = 2 * sigmoid(mixes[hc:2*hc] * scale[1] + base[hc:2*hc])
    //     // ubPost = ubMixes[kHc:2*kHc] * scale[1] + ubBase[kHc:2*kHc]
    //     // ubPost = TSIGMOID(ubPost)
    //     // ubPost = 2.0f * ubPost
    //
    //     // ---- comb_frag[j,k] = mixes[2*hc + j*hc + k] * scale[2] + base[...]
    //     // ubComb = ubMixes[2*kHc:] reshape [kHc, kHc]
    //     // ubComb = ubComb * scale[2] + ubBase[2*kHc:]  (broadcasted appropriately)
    //
    //     // ---- comb = softmax(comb, dim=1) + eps --------------------------
    //     // ubRowMax = TREDUCE_MAX(ubComb, dim=1)
    //     // ubComb   = TEXP(ubComb - ubRowMax[:, None])
    //     // ubRowSum = TREDUCE_SUM(ubComb, dim=1)
    //     // ubComb   = ubComb / ubRowSum[:, None] + eps
    //
    //     // ---- comb = comb / (comb.sum(-2) + eps) -------------------------
    //     // ubColSum = TREDUCE_SUM(ubComb, dim=0)
    //     // ubComb   = ubComb / (ubColSum[None, :] + eps)
    //
    //     // ---- (sinkhorn_iters - 1) extra row+col normalize iterations ----
    //     for (unsigned it = 0; it + 1 < kIters; ++it) {
    //       // ubRowSum = TREDUCE_SUM(ubComb, dim=1)
    //       // ubComb   = ubComb / (ubRowSum[:, None] + eps)
    //       // ubColSum = TREDUCE_SUM(ubComb, dim=0)
    //       // ubComb   = ubComb / (ubColSum[None, :] + eps)
    //     }
    //
    //     // ---- writeback ---------------------------------------------------
    //     // TSTORE(gPre[i, :],     ubPre);
    //     // TSTORE(gPost[i, :],    ubPost);
    //     // TSTORE(gComb[i, :, :], ubComb);
    //   }
    // ----------------------------------------------------------------------
    // TODO(deepseekv4-hc): replace pseudocode above with real auto-mode body.
    // Inner-row math is small (HC_MULT=4 lanes) — verify whether TREDUCE_*
    // intrinsics support 4-wide reductions on A3 vector path; if not, plan
    // for a tree-reduce expansion (cross-check docs_for_ai/qualifier_reference.md).
    (void)pre_raw;
    (void)post_raw;
    (void)comb_raw;
    (void)mixes_raw;
    (void)hc_scale_raw;
    (void)hc_base_raw;
}

extern "C" void launch_hc_sinkhorn(uint8_t *pre, uint8_t *post, uint8_t *comb,
                                   uint8_t *mixes, uint8_t *hc_scale, uint8_t *hc_base,
                                   void *stream)
{
    // TODO(deepseekv4-hc): wire ICache, AICORE config, then launch
    // hc_sinkhorn_kernel<<<1, nullptr, stream>>>(...).
    (void)pre; (void)post; (void)comb;
    (void)mixes; (void)hc_scale; (void)hc_base; (void)stream;
}
