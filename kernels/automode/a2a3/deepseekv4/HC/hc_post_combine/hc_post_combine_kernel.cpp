// --------------------------------------------------------------------------------
// hc_post_combine - expand single-stream residual back into HC_MULT copies
//                   (auto-mode A3, vector).
//
//   y[b, s, h, d] = post[b, s, h] * x[b, s, d]
//                 + sum_{h2=0..HC_MULT-1} comb[b, s, h2, h] * residual[b, s, h2, d]
//
//   x        : (B, S, DIM)             FP32
//   residual : (B, S, HC_MULT, DIM)    FP32
//   post     : (B, S, HC_MULT)         FP32
//   comb     : (B, S, HC_MULT, HC_MULT)FP32
//   y        : (B, S, HC_MULT, DIM)    FP32
//
// Reference: deepseek/model.py:685-686 (Block.hc_post).
//
// DESIGN / SKELETON ONLY - pseudocode body below. Real implementation is
// deferred. Do not claim compile or runtime success until run.sh produces
// `test data success` on the compiler server.
// --------------------------------------------------------------------------------

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "generated_cases.h"

using namespace pto;

namespace hc_post_combine_cfg {

constexpr unsigned kB   = static_cast<unsigned>(kHcB);
constexpr unsigned kS   = static_cast<unsigned>(kHcS);
constexpr unsigned kDim = static_cast<unsigned>(kHcDim);
constexpr unsigned kHc  = static_cast<unsigned>(kHcMult);

constexpr unsigned kN   = kB * kS;     // token count

// Smaller DIM tile than hc_pre_reduce because per-token UB holds:
//   ubX (DIM_TILE), ubRes (HC_MULT, DIM_TILE), ubY (HC_MULT, DIM_TILE).
constexpr unsigned kDimTile = 128;

}  // namespace hc_post_combine_cfg

extern "C" __global__ AICORE void hc_post_combine_kernel(
    __gm__ uint8_t *y_raw,        // FP32 (B, S, HC_MULT, DIM)
    __gm__ uint8_t *x_raw,        // FP32 (B, S, DIM)
    __gm__ uint8_t *residual_raw, // FP32 (B, S, HC_MULT, DIM)
    __gm__ uint8_t *post_raw,     // FP32 (B, S, HC_MULT)
    __gm__ uint8_t *comb_raw)     // FP32 (B, S, HC_MULT, HC_MULT)
{
    using namespace hc_post_combine_cfg;

    // ----------------------------------------------------------------------
    // PSEUDOCODE - auto-mode A3 vector kernel implementing the per-token
    //   y[h, d] = post[h] * x[d] + sum_{h2} comb[h2, h] * residual[h2, d]
    // expansion. Pattern anchors on add_tile_array_kernel.cpp.
    //
    //   __gm__ float *y    = reinterpret_cast<__gm__ float *>(y_raw);
    //   __gm__ float *x    = reinterpret_cast<__gm__ float *>(x_raw);
    //   __gm__ float *res  = reinterpret_cast<__gm__ float *>(residual_raw);
    //   __gm__ float *post = reinterpret_cast<__gm__ float *>(post_raw);
    //   __gm__ float *comb = reinterpret_cast<__gm__ float *>(comb_raw);
    //
    //   GlobalTensor<float> gY   (y,    {kN, kHc, kDim});
    //   GlobalTensor<float> gX   (x,    {kN, kDim});
    //   GlobalTensor<float> gRes (res,  {kN, kHc, kDim});
    //   GlobalTensor<float> gPost(post, {kN, kHc});
    //   GlobalTensor<float> gComb(comb, {kN, kHc, kHc});
    //
    //   constexpr unsigned kDimT = (kDim < kDimTile) ? kDim : kDimTile;
    //   constexpr unsigned kDimTiles = (kDim + kDimT - 1) / kDimT;
    //
    //   for (unsigned t = 0; t < kN; ++t) {
    //     Tile<float, MemUB> ubPost;       // [kHc]
    //     Tile<float, MemUB> ubComb;       // [kHc, kHc]
    //     TLOAD(ubPost, gPost[t, :]);
    //     TLOAD(ubComb, gComb[t, :, :]);
    //
    //     for (unsigned dT = 0; dT < kDimTiles; ++dT) {
    //       const unsigned d0 = dT * kDimT;
    //
    //       Tile<float, MemUB> ubX;                  // [kDimT]
    //       Tile<float, MemUB> ubResH;               // [kDimT] - one h2 slice
    //       Tile<float, MemUB> ubY[kHc];             // [kHc, kDimT] accumulator
    //
    //       TLOAD(ubX, gX[t, d0:d0+kDimT]);
    //
    //       // ---- Initialize ubY[h] = post[h] * x[:DIM_TILE] -------------
    //       #pragma unroll
    //       for (unsigned h = 0; h < kHc; ++h) {
    //         TMUL(ubY[h], ubX, ubPost[h] /* broadcast scalar */);
    //       }
    //
    //       // ---- Accumulate sum_{h2} comb[h2, h] * residual[h2, :] -----
    //       #pragma unroll
    //       for (unsigned h2 = 0; h2 < kHc; ++h2) {
    //         TLOAD(ubResH, gRes[t, h2, d0:d0+kDimT]);
    //         #pragma unroll
    //         for (unsigned h = 0; h < kHc; ++h) {
    //           TFMA(ubY[h], ubResH, ubComb[h2, h] /* broadcast */, ubY[h]);
    //         }
    //       }
    //
    //       // ---- Writeback -------------------------------------------
    //       #pragma unroll
    //       for (unsigned h = 0; h < kHc; ++h) {
    //         TSTORE(gY[t, h, d0:d0+kDimT], ubY[h]);
    //       }
    //     }
    //   }
    // ----------------------------------------------------------------------
    // TODO(deepseekv4-hc): replace pseudocode above with real auto-mode body.
    // The static HC_MULT=4 means the (h, h2) double loop expands to 4 * 4 =
    // 16 broadcast-FMAs per DIM_TILE — fully unrollable. Verify whether the
    // auto-mode allocator is happy with an array `Tile<float, MemUB> ubY[kHc]`
    // of size kHc; cross-check docs_for_ai/auto_mode_bad_patterns.md.
    (void)y_raw;
    (void)x_raw;
    (void)residual_raw;
    (void)post_raw;
    (void)comb_raw;
}

extern "C" void launch_hc_post_combine(uint8_t *y, uint8_t *x, uint8_t *residual,
                                       uint8_t *post, uint8_t *comb, void *stream)
{
    // TODO(deepseekv4-hc): wire ICache, AICORE config, then
    // hc_post_combine_kernel<<<1, nullptr, stream>>>(...).
    (void)y; (void)x; (void)residual;
    (void)post; (void)comb; (void)stream;
}
