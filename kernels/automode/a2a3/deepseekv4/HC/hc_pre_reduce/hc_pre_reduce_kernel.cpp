// --------------------------------------------------------------------------------
// hc_pre_reduce - collapse HC copies into a single residual via `pre` weights
//                 (auto-mode A3, vector).
//
//   y[b, s, d] = sum_{h=0..HC_MULT-1} pre[b, s, h] * x[b, s, h, d]
//     pre : (B, S, HC_MULT)         FP32
//     x   : (B, S, HC_MULT, DIM)    FP32
//     y   : (B, S, DIM)             FP32
//
// Reference: deepseek/model.py:681 (Block.hc_pre final line).
//
// DESIGN / SKELETON ONLY - pseudocode body below. Real implementation is
// deferred. Do not claim compile or runtime success until run.sh produces
// `test data success` on the compiler server.
//
// The reduction is over a tiny static axis (HC_MULT = 4 in the model
// default); it expands to a 4-FMA chain per (token, dim_tile), not a true
// vector reduce.
// --------------------------------------------------------------------------------

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "generated_cases.h"

using namespace pto;

namespace hc_pre_reduce_cfg {

constexpr unsigned kB    = static_cast<unsigned>(kHcB);
constexpr unsigned kS    = static_cast<unsigned>(kHcS);
constexpr unsigned kDim  = static_cast<unsigned>(kHcDim);
constexpr unsigned kHc   = static_cast<unsigned>(kHcMult);

constexpr unsigned kN    = kB * kS;     // token count

// DIM tile along the innermost dim. 256 is a guess; pick a multiple of 16
// and small enough to keep the (HC_MULT + 1) UB tiles within the per-token
// UB budget. Cross-check docs_for_ai/tile_type_reference.md for the canonical
// vector tile alignment on A3.
constexpr unsigned kDimTile = 256;

}  // namespace hc_pre_reduce_cfg

extern "C" __global__ AICORE void hc_pre_reduce_kernel(
    __gm__ uint8_t *y_raw,     // FP32 (B, S, DIM)
    __gm__ uint8_t *pre_raw,   // FP32 (B, S, HC_MULT)
    __gm__ uint8_t *x_raw)     // FP32 (B, S, HC_MULT, DIM)
{
    using namespace hc_pre_reduce_cfg;

    // ----------------------------------------------------------------------
    // PSEUDOCODE - auto-mode A3 vector reduction over the HC_MULT axis.
    // Pattern anchors on add_tile_array_kernel.cpp (TLOAD / vector ops /
    // TSTORE) for the auto-mode call shape.
    //
    //   __gm__ float *y   = reinterpret_cast<__gm__ float *>(y_raw);
    //   __gm__ float *pre = reinterpret_cast<__gm__ float *>(pre_raw);
    //   __gm__ float *x   = reinterpret_cast<__gm__ float *>(x_raw);
    //
    //   GlobalTensor<float> gY  (y,   {kN, kDim});
    //   GlobalTensor<float> gPre(pre, {kN, kHc});
    //   GlobalTensor<float> gX  (x,   {kN, kHc, kDim});
    //
    //   // Number of DIM tiles per token (no tail in default shapes; the
    //   // tiny case has DIM=64 which is < kDimTile, so kDimTile must be
    //   // capped at min(kDim, kDimTile) below for the tiny case to fit).
    //   constexpr unsigned kDimT = (kDim < kDimTile) ? kDim : kDimTile;
    //   constexpr unsigned kDimTiles = (kDim + kDimT - 1) / kDimT;
    //
    //   for (unsigned t = 0; t < kN; ++t) {
    //     Tile<float, MemUB> ubPre;          // [kHc] - per-token pre weights
    //     TLOAD(ubPre, gPre[t, :]);          // 4 scalars
    //
    //     for (unsigned dT = 0; dT < kDimTiles; ++dT) {
    //       const unsigned d0 = dT * kDimT;
    //
    //       Tile<float, MemUB> ubX_h;        // [kDimT] - one HC slice
    //       Tile<float, MemUB> ubY;          // [kDimT] - accumulator
    //
    //       // h = 0: initialize
    //       TLOAD(ubX_h, gX[t, 0, d0:d0+kDimT]);
    //       TMUL(ubY, ubX_h, ubPre[0] /* broadcast scalar */);
    //
    //       // h = 1..HC_MULT-1: FMA accumulate
    //       #pragma unroll
    //       for (unsigned h = 1; h < kHc; ++h) {
    //         TLOAD(ubX_h, gX[t, h, d0:d0+kDimT]);
    //         TFMA(ubY, ubX_h, ubPre[h] /* broadcast */, ubY);
    //       }
    //
    //       TSTORE(gY[t, d0:d0+kDimT], ubY);
    //     }
    //   }
    // ----------------------------------------------------------------------
    // TODO(deepseekv4-hc): replace pseudocode above with real auto-mode body.
    // Verify the scalar-broadcast multiply / FMA shape on A3 vector — the
    // exact intrinsic may be TMUL_S / TFMA_S or rely on a Tile<scalar> view.
    // Cross-check docs_for_ai/qualifier_reference.md for the auto-mode-safe
    // broadcast pattern.
    (void)y_raw;
    (void)pre_raw;
    (void)x_raw;
}

extern "C" void launch_hc_pre_reduce(uint8_t *y, uint8_t *pre, uint8_t *x,
                                     void *stream)
{
    // TODO(deepseekv4-hc): wire ICache, AICORE config, then
    // hc_pre_reduce_kernel<<<1, nullptr, stream>>>(y, pre, x).
    (void)y; (void)pre; (void)x; (void)stream;
}
