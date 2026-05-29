// --------------------------------------------------------------------------------
// gate_hash_routing — DeepSeek-V4 hash-table routing (auto-mode A3, vector).
//
// Computes:
//   indices = tid2eid[input_ids]                   (model.py:577-578)
//   weights = original_scores.gather(1, indices)   (model.py:581)
//   weights /= weights.sum(-1, keepdim=True)       (model.py:582-583, if not softmax)
//   weights *= route_scale                          (model.py:584)
//
// Distinct from gate_score_topk: pure gather, NO topk, NO reduce, NO sort.
// See README.md for the contrast.
//
// DESIGN / SKELETON ONLY — pseudocode body below. Real implementation is
// deferred. Do not claim compile or runtime success until run.sh produces
// `test data success` on the compiler server.
//
// Pattern sources:
//   - MoE/scatter/scatter_kernel.cpp   (per-iter pipe_barrier + GM scalar reads)
//   - add_tile_array/                  (vector tile baseline)
// --------------------------------------------------------------------------------

#include <cstdint>
#include "common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
#include "generated_cases.h"

using namespace pto;

namespace gate_hash_routing_cfg {
constexpr unsigned kT          = kDsmoeT;
constexpr unsigned kNRouted    = kDsmoeNRouted;
constexpr unsigned kNActivated = kDsmoeNActivated;
constexpr unsigned kVocab      = kDsmoeVocab;

#ifndef DSMOE_ROUTE_SCALE
#define DSMOE_ROUTE_SCALE 1.0f
#endif
constexpr float kRouteScale = static_cast<float>(DSMOE_ROUTE_SCALE);

#ifndef DSMOE_SCORE_IS_SOFTMAX
#define DSMOE_SCORE_IS_SOFTMAX 0
#endif
}  // namespace gate_hash_routing_cfg

extern "C" __global__ __aicore__ void gate_hash_routing_kernel(
    __gm__ uint8_t *indices_out,      // INT32, (T, N_ACTIVATED) — out
    __gm__ uint8_t *weights_out,      // FP32 , (T, N_ACTIVATED) — out
    __gm__ uint8_t *input_ids,        // INT32, (T,)            — in
    __gm__ uint8_t *tid2eid,          // INT32, (VOCAB, N_ACTIVATED) — in (lookup table)
    __gm__ uint8_t *original_scores)  // FP32 , (T, N_ROUTED)    — in
{
    using namespace gate_hash_routing_cfg;
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 vector indirect gather.
    //
    //   __gm__ int32_t *gIds   = reinterpret_cast<__gm__ int32_t *>(input_ids);
    //   __gm__ int32_t *gTab   = reinterpret_cast<__gm__ int32_t *>(tid2eid);
    //   __gm__ int32_t *gIdxOut= reinterpret_cast<__gm__ int32_t *>(indices_out);
    //   __gm__ float   *gOrig  = reinterpret_cast<__gm__ float   *>(original_scores);
    //   __gm__ float   *gWOut  = reinterpret_cast<__gm__ float   *>(weights_out);
    //
    //   Tile<int32_t, MemUB, Shape{1, kNActivated}>  idxTile;
    //   Tile<float,   MemUB, Shape{1, kNActivated}>  wTile;
    //
    //   for (t = 0; t < kT; ++t) {
    //       pipe_barrier(PIPE_ALL);             // cross-iter auto-sync guard
    //
    //       int32_t tid = gIds[t];              // GM scalar read
    //
    //       // 1) Lookup: indices[t, k] = tid2eid[tid, k]
    //       for (k = 0; k < kNActivated; ++k) {
    //           idxTile[k] = gTab[tid * kNActivated + k];   // GM scalar read
    //       }
    //
    //       // 2) Gather original_scores[t, indices[t, k]] for each k
    //       for (k = 0; k < kNActivated; ++k) {
    //           int32_t e = idxTile[k];
    //           wTile[k]  = gOrig[t * kNRouted + e];        // GM scalar read
    //       }
    //
    //       // 3) Normalize (skipped if score_func == "softmax")
    //   #if !DSMOE_SCORE_IS_SOFTMAX
    //       float s = 0.0f;
    //       for (k = 0; k < kNActivated; ++k) s += wTile[k];
    //       float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
    //       TMULS(wTile, wTile, inv);
    //   #endif
    //
    //       // 4) route_scale
    //       TMULS(wTile, wTile, kRouteScale);
    //
    //       // 5) Store this row
    //       for (k = 0; k < kNActivated; ++k) {
    //           gIdxOut[t * kNActivated + k] = idxTile[k];
    //           gWOut  [t * kNActivated + k] = wTile  [k];
    //       }
    //   }
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on MoE/scatter_kernel.cpp pass 3 (per-iter pipe_barrier +
    // GM scalar reads) for the call shape.
    (void)indices_out; (void)weights_out;
    (void)input_ids; (void)tid2eid; (void)original_scores;
}

// Host launcher (kept minimal).
void launch_gate_hash_routing(uint8_t *indices_out, uint8_t *weights_out,
                              uint8_t *input_ids, uint8_t *tid2eid,
                              uint8_t *original_scores,
                              uint64_t T, uint64_t N, uint64_t K, uint64_t V,
                              void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch gate_hash_routing_kernel.
    (void)indices_out; (void)weights_out; (void)input_ids;
    (void)tid2eid; (void)original_scores;
    (void)T; (void)N; (void)K; (void)V; (void)stream;
}
