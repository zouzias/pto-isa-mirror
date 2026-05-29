// --------------------------------------------------------------------------------
// gate_score_topk — DeepSeek-V4 score-branch routing (auto-mode A3, vector).
//
// Computes:
//   biased   = original_scores + bias              (model.py:575-576, optional)
//   indices  = topk(biased, N_ACTIVATED, dim=-1)   (model.py:580)
//   weights  = original_scores.gather(1, indices)  (model.py:581)
//   weights /= weights.sum(-1, keepdim=True)       (model.py:582-583, skipped if softmax)
//   weights *= route_scale                          (model.py:584)
//
// DESIGN / SKELETON ONLY — pseudocode body below. Real implementation is
// deferred. Do not claim compile or runtime success until run.sh produces
// `test data success` on the compiler server.
//
// Pattern sources:
//   - MoE/moe_topk/moe_topk_kernel.cpp           (TSORT32 → TSUBVIEW → TGATHER)
//   - MoE/gather/gather_kernel.cpp               (TROWSUM → TROWEXPANDDIV normalize)
//   - MoE/scatter/scatter_kernel.cpp             (per-row scalar gather pattern)
// --------------------------------------------------------------------------------

#include <cstdint>
#include "common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
#include "generated_cases.h"

using namespace pto;

namespace gate_score_topk_cfg {
constexpr unsigned kT          = kDsmoeT;
constexpr unsigned kNRouted    = kDsmoeNRouted;
constexpr unsigned kNActivated = kDsmoeNActivated;

// TSORT32 requires a 32-wide block; pad N_ROUTED up.
constexpr unsigned kPaddedNRouted =
    (kNRouted < 32) ? 32 : ((kNRouted + 31) / 32) * 32;

// route_scale from ModelArgs (default 1.0). Configurable at build time
// via -DDSMOE_ROUTE_SCALE=<float>.
#ifndef DSMOE_ROUTE_SCALE
#define DSMOE_ROUTE_SCALE 1.0f
#endif
constexpr float kRouteScale = static_cast<float>(DSMOE_ROUTE_SCALE);

// Set to 1 to skip the post-gather normalize (softmax variant of gate_softmax).
#ifndef DSMOE_SCORE_IS_SOFTMAX
#define DSMOE_SCORE_IS_SOFTMAX 0
#endif
}  // namespace gate_score_topk_cfg

extern "C" __global__ __aicore__ void gate_score_topk_kernel(
    __gm__ uint8_t *indices_out,      // INT32, (T, N_ACTIVATED) — out
    __gm__ uint8_t *weights_out,      // FP32 , (T, N_ACTIVATED) — out
    __gm__ uint8_t *original_scores,  // FP32 , (T, N_ROUTED)    — in (un-biased)
    __gm__ uint8_t *bias)             // FP32 , (N_ROUTED,)      — in (optional; host fills 0 if none)
{
    using namespace gate_score_topk_cfg;
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 vector top-K + gather + normalize.
    //
    //   GlobalTensor<float>   gOrig   (original_scores, {kT, kNRouted});
    //   GlobalTensor<float>   gBias   (bias,            {kNRouted});
    //   GlobalTensor<int32_t> gIdx    (indices_out,     {kT, kNActivated});
    //   GlobalTensor<float>   gWeight (weights_out,     {kT, kNActivated});
    //
    //   Tile<float,   MemUB, Shape{1, kPaddedNRouted}>  rowOrig;     // padded row
    //   Tile<float,   MemUB, Shape{1, kPaddedNRouted}>  rowBiased;   // padded row
    //   Tile<float,   MemUB, Shape{1, kNRouted}>        biasTile;    // broadcast
    //   Tile<float,   MemUB, Shape{1, kPaddedNRouted * 2}> sortDst;  // packed (val,idx)
    //   Tile<int32_t, MemUB, Shape{1, kNActivated}>     idxTile;
    //   Tile<float,   MemUB, Shape{1, kNActivated}>     wTile;
    //
    //   // Load bias once (broadcast).
    //   TLOAD(biasTile, gBias);
    //
    //   for (t = 0; t < kT; ++t) {
    //       pipe_barrier(PIPE_ALL);                     // cross-iter auto-sync guard
    //       TLOAD(rowOrig, gOrig[t, :]);                // un-biased
    //       // Zero-pad cols [kNRouted, kPaddedNRouted) with -FLT_MAX so they
    //       // never top-K.
    //       TADD (rowBiased, rowOrig, biasTile);        // biased view for selection
    //       // (host pads the bias-tail too)
    //
    //       // TSORT32 → TSUBVIEW → TGATHER (P0101 for val, P1010 for idx) —
    //       // see MoE/moe_topk_kernel.cpp.
    //       TSORT32(sortDst, rowBiased, /*identityIdx*/...);
    //       TGATHER<wTile,   ...P0101>(wTile,   sortDst);
    //       TGATHER<idxTile, ...P1010>(idxTile, sortDst);
    //
    //       // The values in wTile correspond to *biased* scores at the top
    //       // positions, but the model wants the *un-biased* values.
    //       // Re-gather from rowOrig using idxTile:
    //       for (k = 0; k < kNActivated; ++k) {
    //           int32_t e = idxTile[k];
    //           wTile[k] = rowOrig[e];          // scalar UB read
    //       }
    //
    //       // Normalize (skipped if score_func == "softmax").
    //   #if !DSMOE_SCORE_IS_SOFTMAX
    //       float s = 0.0f;
    //       for (k = 0; k < kNActivated; ++k) s += wTile[k];
    //       float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
    //       TMULS(wTile, wTile, inv);
    //   #endif
    //
    //       // route_scale.
    //       TMULS(wTile, wTile, kRouteScale);
    //
    //       TSTORE(gIdx   [t, :], idxTile);
    //       TSTORE(gWeight[t, :], wTile);
    //   }
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on MoE/moe_topk_kernel.cpp for the TSORT32 / TGATHER core
    // and MoE/scatter_kernel.cpp for the per-iter pipe_barrier guard.
    (void)indices_out; (void)weights_out; (void)original_scores; (void)bias;
}

// Host launcher (kept minimal).
void launch_gate_score_topk(uint8_t *indices_out, uint8_t *weights_out,
                            uint8_t *original_scores, uint8_t *bias,
                            uint64_t T, uint64_t N, uint64_t K, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch gate_score_topk_kernel.
    (void)indices_out; (void)weights_out; (void)original_scores; (void)bias;
    (void)T; (void)N; (void)K; (void)stream;
}
