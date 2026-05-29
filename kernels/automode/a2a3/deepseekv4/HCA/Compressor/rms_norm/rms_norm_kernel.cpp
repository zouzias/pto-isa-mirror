// --------------------------------------------------------------------------------
// rms_norm — Compressor RMSNorm along HEAD_DIM (auto-mode A3, vector).
//
//   var      = mean_h (x^2)
//   x_hat    = x * rsqrt(var + eps)
//   out      = weight * x_hat
//
//     B   = kCompB
//     SB  = kCompS / kCompRatio
//     D   = kCompHeadDim
//     eps = 1e-6f                              (model.py:185 default)
//
// Reference: deepseek/model.py:192-197 (RMSNorm.forward)
//            deepseek/model.py:363    (call site inside Compressor)
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

// RMSNorm epsilon. ModelArgs.norm_eps = 1e-6 (model.py:185 default).
constexpr float kRmsNormEps = 1e-6f;

// Shape aliases (from ../build/generated_cases.h).
//   kCompB, kCompS, kCompHeadDim, kCompRatio

extern "C" __global__ __aicore__ void rms_norm_kernel(
    __gm__ uint8_t *out,     // FP32, (B, SB, D)
    __gm__ uint8_t *x,       // FP32, (B, SB, D)
    __gm__ uint8_t *weight)  // FP32, (D,)
{
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 vector RMSNorm.
    // Pattern anchors on add_tile_array_kernel.cpp (baseline auto-mode
    // call shape) + sanctioned TReduceSum / TMul / TRsqrt.
    //
    //   GlobalTensor<float> gIn  (x,      {B, SB, D});
    //   GlobalTensor<float> gW   (weight, {D});
    //   GlobalTensor<float> gOut (out,    {B, SB, D});
    //
    //   // weight is reused across (B, SB); load once.
    //   Tile<float, MemUB> tW;
    //   TLOAD(tW, gW);
    //
    //   for (b = 0; b < B; ++b) {
    //     for (sb = 0; sb < SB; ++sb) {
    //       Tile<float, MemUB> tIn;     // [D]
    //       Tile<float, MemUB> tSq;     // [D]
    //       Tile<float, MemUB> tVar;    // [1]
    //       Tile<float, MemUB> tInv;    // [1]
    //       Tile<float, MemUB> tOut;    // [D]
    //
    //       TLOAD(tIn, gIn[b, sb, :]);
    //       TMul (tSq, tIn, tIn);                 // x^2
    //       TReduceSum(tVar, tSq, /*axis=*/0);    // sum over D → [1]
    //       // mean = sum / D, then rsqrt(mean + eps).
    //       // Implement mean+eps via scalar broadcast on UB.
    //       TScale     (tVar, tVar, 1.0f / D);    // divide by D
    //       TAddScalar (tVar, tVar, kRmsNormEps); // + eps
    //       TRsqrt     (tInv, tVar);              // [1]
    //       TMul       (tOut, tIn,   tInv);       // broadcast scalar
    //       TMul       (tOut, tOut,  tW);         // pointwise * weight
    //       TSTORE     (gOut[b, sb, :], tOut);
    //     }
    //   }
    // ----------------------------------------------------------------------
    // NOTE: API names (TReduceSum, TRsqrt, TScale, TAddScalar) must be
    // confirmed against include/pto/ headers before implementation; this
    // skeleton uses placeholder names.
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on add_tile_array_kernel.cpp + any in-tree rms_norm/layernorm
    // kernel under kernels/automode/ for the sanctioned reduce+rsqrt form.
    (void)out; (void)x; (void)weight;
}

// Host launcher (kept minimal — the real launcher will go through pto-launch
// helpers similar to MoE/router_matmul/router_matmul_kernel.cpp).
void launch_rms_norm(uint8_t *out, uint8_t *x, uint8_t *weight,
                     uint64_t B, uint64_t SB, uint64_t D, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch rms_norm_kernel.
    (void)out; (void)x; (void)weight;
    (void)B; (void)SB; (void)D; (void)stream;
}
