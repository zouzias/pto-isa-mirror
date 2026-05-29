// --------------------------------------------------------------------------------
// qk_matmul — sparse_attn Q @ K^T * softmax_scale (auto-mode A3, cube).
//
// Op (kernel.py:328-330):
//   T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=FullRow)
//   for i, j in T.Parallel(h, block):
//       acc_s[i, j] *= scale
//
// We split the upstream mask-fill (kernel.py:326-327: acc_s = -inf where
// idx==-1) into the vector-side online_softmax leaf; this leaf assumes a
// fresh acc_s = 0 and does only the GEMM + scale. (Note: this means -inf
// initial fill is NOT applied here; the masking still happens, but later in
// online_softmax via subtracting a large negative bias per masked column —
// see online_softmax README for the deferred behavior.)
//
// Reference: deepseek/kernel.py:326-330 (sparse_attn inner block).
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

// Cube tile constants. For the prototype the whole problem fits in one
// fractal tile because H, BLOCK, D are small (<= 128 in defaults).
// All sizes are multiples of 16 — A3 cube fractal is 16x16x16.
//   M-dim = kCsaH      (heads)              -> kTileM
//   N-dim = kCsaBlock  (gathered KV rows)   -> kTileN
//   K-dim = kCsaD      (head dim)           -> kTileK
// Assumption: kCsaH, kCsaBlock, kCsaD are all multiples of 16 after the
// head-padding in kernel.py:359 and DEFAULT_CASES tiny shape (H=4 is < 16;
// real binaries should bump H to 16 via the --cases override).

extern "C" __global__ __aicore__ void qk_matmul_kernel(
    __gm__ uint8_t *acc_s,        // FP32, (H, BLOCK)  — out (acc_s after scale)
    __gm__ uint8_t *q,            // BF16, (H, D)      — in  (q_shared)
    __gm__ uint8_t *kv_block,     // BF16, (BLOCK, D)  — in  (kv_shared)
    float softmax_scale)          // multiplied in after GEMM
{
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 cube GEMM, BF16 x BF16^T -> FP32, then *scale.
    // Pattern mirrors the cube body of z_gemm + a vector tail for the
    // post-scale multiply.
    //
    //   GlobalTensor<bfloat16_t> gQ  (q,        {kCsaH,     kCsaD});
    //   GlobalTensor<bfloat16_t> gKv (kv_block, {kCsaBlock, kCsaD});
    //   GlobalTensor<float>      gS  (acc_s,    {kCsaH,     kCsaBlock});
    //
    //   Tile<bfloat16_t, MemL1>  qL1, kL1;
    //   TileLeft<bfloat16_t,  MemL0A> qL0A;       // [H, D] fractal panel
    //   TileRight<bfloat16_t, MemL0B> kL0B;       // [BLOCK, D] fractal panel
    //                                              // (acts as K^T via TileRight
    //                                              //  + transpose_B semantics)
    //   Tile<float, MemL0C>      cL0C;            // [H, BLOCK] accumulator
    //
    //   TLOAD(qL1, gQ);
    //   TLOAD(kL1, gKv);
    //   TLOAD(qL0A, qL1);
    //   TLOAD(kL0B, kL1);
    //   TCLEAR(cL0C);
    //   TMATMUL(cL0C, qL0A, kL0B);                 // FP32 accumulator
    //   // Apply softmax_scale on the way out — vector op on a cube-only core
    //   // is not allowed, so the scale step is fused into the next leaf
    //   // (online_softmax) as `acc_s *= scale` instead of here.
    //   TSTORE(gS, cL0C);
    //
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on z_gemm_kernel.cpp for the cube call shape. The scale multiply
    // is conceptually part of this stage but lives on the vector side; pass
    // softmax_scale through to the next leaf via the generated_cases header
    // or a side-band constant.
    (void)acc_s; (void)q; (void)kv_block; (void)softmax_scale;
}

// Host launcher.
void launch_qk_matmul(uint8_t *acc_s, uint8_t *q, uint8_t *kv_block,
                      float softmax_scale, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch qk_matmul_kernel.
    (void)acc_s; (void)q; (void)kv_block; (void)softmax_scale; (void)stream;
}
