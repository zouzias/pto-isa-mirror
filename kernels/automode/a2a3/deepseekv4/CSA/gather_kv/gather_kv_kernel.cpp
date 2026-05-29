// --------------------------------------------------------------------------------
// gather_kv — sparse_attn KV gather stage (auto-mode A3, vector).
//
// Op (kernel.py:322-325):
//   for i in T.Parallel(block):
//       idxs[i] = T.if_then_else(t*block + i < topk,
//                                topk_idxs[by, bx, t*block + i],
//                                -1)
//   for i, j in T.Parallel(block, d):
//       kv_shared[i, j] = T.if_then_else(idxs[i] != -1,
//                                        kv[by, idxs[i], j],
//                                        0)
//
// This leaf scaffolds the gather for ONE (b, m, t) tuple. The outer loop over
// pipelined blocks (T.Pipelined at kernel.py:321) lives in the FA driver, not
// here. The compile-time block index `t` is baked into the binary via the
// generated_cases manifest.
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

// Tile constants. BLOCK / D from the family manifest; for this prototype the
// whole [BLOCK, D] tile is a single UB-resident vector tile (BLOCK ≤ 64,
// D ≤ 128 in defaults).
//   kCsaBlock = block (rows gathered per pipelined step)
//   kCsaD     = head dim
// Assumption: a single [kCsaBlock, kCsaD] BF16 tile fits in UB.

extern "C" __global__ __aicore__ void gather_kv_kernel(
    __gm__ uint8_t *kv_gathered,   // BF16, (BLOCK, D) row-major — out
    __gm__ uint8_t *kv,            // BF16, (T_total, D)         — full KV cache for batch row
    __gm__ uint8_t *topk_idxs,     // INT32, (TOPK,)             — indices for this (b, m)
    uint32_t t_block,              // current pipelined block index
    uint32_t topk_len,             // = kCsaS
    uint32_t kv_rows)              // = T_total (full cache length)
{
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 vector gather, INT32-indexed BF16 rows.
    // Pattern anchored on add_tile_array (vector TLOAD/TSTORE) + a future
    // gather primitive (MGATHER on A5, on A3 we emulate via scalar loop).
    //
    //   GlobalTensor<int32_t>    gIdx (topk_idxs,   {kCsaS});
    //   GlobalTensor<bfloat16_t> gKv  (kv,          {kv_rows, kCsaD});
    //   GlobalTensor<bfloat16_t> gOut (kv_gathered, {kCsaBlock, kCsaD});
    //
    //   Tile<int32_t,    MemUB> idxTile;       // [kCsaBlock]
    //   Tile<bfloat16_t, MemUB> outTile;       // [kCsaBlock, kCsaD]
    //
    //   // 1) Load this block's slice of topk_idxs and clamp out-of-range
    //   //    positions to -1 (sentinel).
    //   for (i = 0; i < kCsaBlock; ++i) {
    //       int32_t global_i = t_block * kCsaBlock + i;
    //       idxTile[i] = (global_i < topk_len) ? gIdx[global_i] : -1;
    //   }
    //
    //   // 2) Gather each row; out-of-range positions become zero rows.
    //   for (i = 0; i < kCsaBlock; ++i) {
    //       int32_t idx = idxTile[i];
    //       if (idx == -1) {
    //           // Zero the row (vector fill).
    //           TFILL(outTile.row(i), bfloat16_t{0});
    //       } else {
    //           // Bulk-copy one D-wide row from gKv into outTile row i.
    //           TLOAD(outTile.row(i), gKv[idx, :]);
    //       }
    //   }
    //
    //   // 3) Spill the gathered block to GM.
    //   TSTORE(gOut, outTile);
    //
    // Note: on A3 there is no MGATHER (see docs_for_ai/a3_a5_differences.md
    // §MGATHER A5-only). Index-driven row gathers must be emulated with a
    // scalar loop + per-row DMA. On A5 this whole block collapses to one
    // MGATHER call.
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on add_tile_array_kernel.cpp for vector tile call shape.
    (void)kv_gathered; (void)kv; (void)topk_idxs;
    (void)t_block; (void)topk_len; (void)kv_rows;
}

// Host launcher.
void launch_gather_kv(uint8_t *kv_gathered, uint8_t *kv, uint8_t *topk_idxs,
                      uint32_t t_block, uint32_t topk_len, uint32_t kv_rows,
                      void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch gather_kv_kernel.
    (void)kv_gathered; (void)kv; (void)topk_idxs;
    (void)t_block; (void)topk_len; (void)kv_rows; (void)stream;
}
