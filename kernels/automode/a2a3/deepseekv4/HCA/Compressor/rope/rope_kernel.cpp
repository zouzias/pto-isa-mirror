// --------------------------------------------------------------------------------
// rope — Compressor apply_rotary_emb (auto-mode A3, vector).
//
// Rotates the last ROPE_DIM dims of kv_comp by precomputed (cos, sin) pairs.
// Implements the real-arithmetic form of the python:
//     model.py:243   x = torch.view_as_real(x * freqs_cis).flatten(-2)
//
//     re_out = re_in * cs - im_in * sn
//     im_out = re_in * sn + im_in * cs
//
// for each (b, sb, k) with k in [0, ROPE_DIM/2), where
// (cs, sn) = (freqs_cos[sb, k], freqs_sin[sb, k]).
//
//     B    = kCompB
//     SB   = kCompS / kCompRatio
//     RD   = kCompRopeDim
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
//   kCompB, kCompS, kCompRopeDim, kCompRatio
// SB = kCompS / kCompRatio; RD = kCompRopeDim.

extern "C" __global__ __aicore__ void rope_kernel(
    __gm__ uint8_t *out,         // FP32, (B, SB, RD) row-major (rotated tail)
    __gm__ uint8_t *kv_tail,     // FP32, (B, SB, RD) row-major (input tail)
    __gm__ uint8_t *freqs_cos,   // FP32, (SB, RD/2)
    __gm__ uint8_t *freqs_sin)   // FP32, (SB, RD/2)
{
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 vector rotary embed.
    // Pattern anchors on add_tile_array_kernel.cpp (baseline auto-mode
    // call shape) + a TGather-style strided pair read.
    //
    //   GlobalTensor<float> gIn  (kv_tail,   {B, SB, RD});
    //   GlobalTensor<float> gCos (freqs_cos, {SB, RD/2});
    //   GlobalTensor<float> gSin (freqs_sin, {SB, RD/2});
    //   GlobalTensor<float> gOut (out,       {B, SB, RD});
    //
    //   for (b = 0; b < B; ++b) {
    //     for (sb = 0; sb < SB; ++sb) {
    //       Tile<float, MemUB> tIn;   // [RD]
    //       Tile<float, MemUB> tCs;   // [RD/2]
    //       Tile<float, MemUB> tSn;   // [RD/2]
    //       Tile<float, MemUB> tRe;   // [RD/2]   even-indexed entries of tIn
    //       Tile<float, MemUB> tIm;   // [RD/2]   odd-indexed entries of tIn
    //       Tile<float, MemUB> tReOut;
    //       Tile<float, MemUB> tImOut;
    //
    //       TLOAD(tIn, gIn [b, sb, :]);
    //       TLOAD(tCs, gCos[sb, :]);
    //       TLOAD(tSn, gSin[sb, :]);
    //
    //       // Split interleaved (re, im) pairs into two contiguous halves.
    //       // (Sanctioned API for stride-2 gather on UB tiles TBD.)
    //       TGatherEven(tRe, tIn);    // tRe[k] = tIn[2*k]
    //       TGatherOdd (tIm, tIn);    // tIm[k] = tIn[2*k + 1]
    //
    //       // re_out = re_in * cs - im_in * sn
    //       Tile<float, MemUB> tTmp1, tTmp2;
    //       TMul(tTmp1, tRe, tCs);
    //       TMul(tTmp2, tIm, tSn);
    //       TSub(tReOut, tTmp1, tTmp2);
    //
    //       // im_out = re_in * sn + im_in * cs
    //       TMul(tTmp1, tRe, tSn);
    //       TMul(tTmp2, tIm, tCs);
    //       TAdd(tImOut, tTmp1, tTmp2);
    //
    //       // Interleave back into (re, im) layout and store.
    //       Tile<float, MemUB> tOut;
    //       TScatterInterleave(tOut, tReOut, tImOut);  // tOut[2*k]=tReOut[k], tOut[2*k+1]=tImOut[k]
    //       TSTORE(gOut[b, sb, :], tOut);
    //     }
    //   }
    // ----------------------------------------------------------------------
    // NOTE: API names (TGatherEven / TGatherOdd / TScatterInterleave, TMul,
    // TAdd, TSub) must be confirmed against include/pto/ headers before
    // implementation; this skeleton uses placeholder names.
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on add_tile_array_kernel.cpp + any in-tree rope / rotary kernel.
    (void)out; (void)kv_tail; (void)freqs_cos; (void)freqs_sin;
}

// Host launcher (kept minimal — the real launcher will go through pto-launch
// helpers similar to MoE/router_matmul/router_matmul_kernel.cpp).
void launch_rope(uint8_t *out, uint8_t *kv_tail,
                 uint8_t *freqs_cos, uint8_t *freqs_sin,
                 uint64_t B, uint64_t SB, uint64_t RD, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch rope_kernel.
    (void)out; (void)kv_tail; (void)freqs_cos; (void)freqs_sin;
    (void)B; (void)SB; (void)RD; (void)stream;
}
