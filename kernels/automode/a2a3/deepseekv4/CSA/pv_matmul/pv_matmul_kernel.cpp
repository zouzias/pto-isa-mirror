// --------------------------------------------------------------------------------
// pv_matmul — sparse_attn acc_o += acc_s_cast @ V (auto-mode A3, cube).
//
// Op (kernel.py:341-343):
//   for i, j in T.Parallel(h, d):
//       acc_o[i, j] *= scores_scale[i]       // vector rescale — NOT here
//   T.gemm(acc_s_cast, kv_shared, acc_o, policy=FullRow)
//
// This leaf is the cube body only: a [H, BLOCK] BF16 x [BLOCK, D] BF16 GEMM
// that accumulates into a [H, D] FP32 acc_o. The vector-side rescale
// `acc_o *= scores_scale[h]` is performed by the FA driver BEFORE invoking
// this kernel (because this is a cube-only AICORE binary, it cannot do
// vector ops). The driver pipeline is:
//
//   online_softmax  -> emits scores_scale, acc_s_cast
//   (driver)        -> acc_o *= scores_scale   (vector pass over [H, D])
//   pv_matmul       -> acc_o += acc_s_cast @ kv_block
//
// Reference: deepseek/kernel.py:341-343 (sparse_attn inner block).
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
// fractal tile because H, BLOCK, D are small.
// All sizes are multiples of 16 — A3 cube fractal is 16x16x16.
//   M-dim = kCsaH      (heads)   -> kTileM
//   N-dim = kCsaD      (head dim)-> kTileN
//   K-dim = kCsaBlock  (block)   -> kTileK
// Assumption: matches the head-padded shape from kernel.py:359.

extern "C" __global__ __aicore__ void pv_matmul_kernel(
    __gm__ uint8_t *acc_o,       // FP32 [H, D]      — in/out (RMW accumulator)
    __gm__ uint8_t *acc_s_cast,  // BF16 [H, BLOCK]  — in  (from online_softmax)
    __gm__ uint8_t *kv_block)    // BF16 [BLOCK, D]  — in  (same as qk_matmul's kv_block)
{
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 cube GEMM, BF16 x BF16 -> FP32 (RMW).
    // Pattern mirrors the cube body of z_gemm; the only difference is the
    // RMW semantics on acc_o (load existing FP32 into L0C before TMATMUL).
    //
    //   GlobalTensor<float>      gO  (acc_o,      {kCsaH, kCsaD});
    //   GlobalTensor<bfloat16_t> gSC (acc_s_cast, {kCsaH, kCsaBlock});
    //   GlobalTensor<bfloat16_t> gV  (kv_block,   {kCsaBlock, kCsaD});
    //
    //   Tile<bfloat16_t, MemL1>  scL1, vL1;
    //   TileLeft<bfloat16_t,  MemL0A> scL0A;       // [H, BLOCK] panel
    //   TileRight<bfloat16_t, MemL0B> vL0B;        // [BLOCK, D] panel
    //   Tile<float, MemL0C>      oL0C;             // [H, D] accumulator
    //
    //   TLOAD(scL1, gSC);
    //   TLOAD(vL1,  gV);
    //   // RMW: load prior acc_o into L0C as the initial accumulator value.
    //   TLOAD(oL0C, gO);
    //   TLOAD(scL0A, scL1);
    //   TLOAD(vL0B,  vL1);
    //   TMATMUL(oL0C, scL0A, vL0B);                // FP32 accumulator
    //   TSTORE(gO, oL0C);
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on z_gemm_kernel.cpp for the cube call shape. The RMW pattern
    // on L0C will need a TLOAD-prior step (z_gemm uses TCLEAR; we instead
    // load gO into oL0C before the matmul).
    (void)acc_o; (void)acc_s_cast; (void)kv_block;
}

// Host launcher.
void launch_pv_matmul(uint8_t *acc_o, uint8_t *acc_s_cast, uint8_t *kv_block,
                      void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch pv_matmul_kernel.
    (void)acc_o; (void)acc_s_cast; (void)kv_block; (void)stream;
}
