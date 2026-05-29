// --------------------------------------------------------------------------------
// index_score — Indexer relu(einsum) * weights, reduce over heads
//                (auto-mode A3, vector).
//
// score[b, s, t] = sum_h ( relu(sum_d q[b, s, h, d] * kv[b, t, d]) * weights[b, s, h] )
//
// Reference: deepseek/model.py:421-422 (Indexer.forward)
//   index_score = einsum("bshd,btd->bsht", q, kv_cache)
//   index_score = (relu(index_score) * weights.unsqueeze(-1)).sum(dim=2)
//
// Assumption (see README §References): the einsum is small-D enough for the
// prototype shape that streaming a per-(s, t) inner product through UB is
// acceptable. A follow-up may split into einsum_cube + reduce_vector.
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

// Vector tile constants (see README.md memory-budget plan).
constexpr uint32_t kTileT = 16;  // T-axis inner block

// Shape constants from the family manifest (../build/generated_cases.h):
//   kIdxB, kIdxS, kIdxT, kIdxNHeads, kIdxHeadDim, ...

extern "C" __global__ __aicore__ void index_score_kernel(
    __gm__ uint8_t *score,        // FP32, (B, S, T) row-major
    __gm__ uint8_t *q,            // BF16, (B, S, H, D)
    __gm__ uint8_t *kv_cache,     // BF16, (B, T, D)
    __gm__ uint8_t *weights)      // FP32, (B, S, H)
{
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 vector leaf.
    // Pattern anchor: kernels/automode/a2a3/add_tile_array (Vec auto baseline).
    //
    //   GlobalTensor<bfloat16_t> gQ  (q,        {B*S, H, D});
    //   GlobalTensor<bfloat16_t> gKV (kv_cache, {B, T, D});
    //   GlobalTensor<float>      gW  (weights,  {B*S, H});
    //   GlobalTensor<float>      gO  (score,    {B*S, T});
    //
    //   for (bs = 0; bs < B*S; ++bs) {
    //     // Pull the per-(b, s) query [H, D] and the [H] weights once.
    //     Tile<bfloat16_t, MemUB> q_sh;       // [H, D]
    //     Tile<float,      MemUB> w_sh;       // [H]
    //     TLOAD(q_sh, gQ[bs, :, :]);
    //     TLOAD(w_sh, gW[bs, :]);
    //
    //     // Determine which batch index this bs belongs to so we can pull
    //     // the matching kv_cache slab.
    //     int b = bs / kIdxS;
    //
    //     for (tTile = 0; tTile < ceildiv(T, kTileT); ++tTile) {
    //       Tile<bfloat16_t, MemUB> kv_tile;          // [kTileT, D]
    //       Tile<float,      MemUB> partial;          // [H, kTileT]
    //       Tile<float,      MemUB> score_tile;       // [kTileT]
    //
    //       TLOAD(kv_tile,
    //             gKV[b, tTile*kTileT : (tTile+1)*kTileT, :]);
    //
    //       // 1) Einsum: partial[h, t] = sum_d q_sh[h, d] * kv_tile[t, d]
    //       //    Express as H inner-products of length D, vector path:
    //       //    for h in 0..H:
    //       //      for t in 0..kTileT:
    //       //        partial[h, t] = TDOT(q_sh[h, :], kv_tile[t, :]);
    //       //    Auto-mode pass should pipeline these via UB-resident MACs.
    //       TCLEAR(partial);
    //       for (int h = 0; h < kIdxNHeads; ++h) {
    //         for (int tt = 0; tt < kTileT; ++tt) {
    //           // FP32 accumulate over D using vector MAC.
    //           partial[h, tt] = TDOT_BF16_FP32(q_sh.row(h), kv_tile.row(tt));
    //         }
    //       }
    //
    //       // 2) relu in place
    //       TRELU(partial);                  // max(partial, 0)
    //
    //       // 3) scale by per-head weights: partial[h, t] *= w_sh[h]
    //       //    broadcast across t — vector mul with H broadcast.
    //       TMUL_BROADCAST(partial, w_sh);   // along H axis
    //
    //       // 4) reduce-sum across H: score_tile[t] = sum_h partial[h, t]
    //       TREDUCE_SUM_ALONG(score_tile, partial, /*axis=*/0);
    //
    //       TSTORE(gO[bs, tTile*kTileT : (tTile+1)*kTileT], score_tile);
    //     }
    //   }
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on add_tile_array_kernel.cpp (single-AICORE Vec auto baseline)
    // for the loop / Tile shape. The TDOT / TRELU / TREDUCE names above are
    // placeholders — substitute the actual auto-mode vector primitives once
    // verified against include/pto/pto-inst.hpp.
    (void)score; (void)q; (void)kv_cache; (void)weights;
}

// Host launcher (minimal scaffold).
void launch_index_score(uint8_t *score, uint8_t *q, uint8_t *kv_cache,
                        uint8_t *weights,
                        uint64_t B, uint64_t S, uint64_t T,
                        uint64_t H, uint64_t D, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch the kernel.
    (void)score; (void)q; (void)kv_cache; (void)weights;
    (void)B; (void)S; (void)T; (void)H; (void)D; (void)stream;
}
