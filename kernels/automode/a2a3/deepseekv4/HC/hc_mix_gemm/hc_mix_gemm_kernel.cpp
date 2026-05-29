// --------------------------------------------------------------------------------
// hc_mix_gemm - Hyper-Connections "mixes" projection (auto-mode A3, cube).
//
//   mixes[M, N] = (x_flat[M, K] @ hc_fn[N, K]^T) * rsqrt[M]
//     M = B * S                  (token count after collapsing B,S)
//     K = HC_MULT * DIM          (HC-flattened hidden dim, 16384 default)
//     N = MIX_HC = (2 + HC_MULT) * HC_MULT  (24 for HC_MULT=4)
//
// Reference: deepseek/model.py:677-679 (Block.hc_pre — the F.linear + rsqrt
//            step before hc_split_sinkhorn).
//
// DESIGN / SKELETON ONLY - pseudocode body below. Real implementation is
// deferred. Do not claim compile or runtime success until run.sh produces
// `test data success` on the compiler server.
//
// Weights are FP32 in the DeepSeek-V4-Pro checkpoint (model.py:666-672
// `with set_dtype(torch.float32)`) so this leaf is FP32 throughout
// (no BF16 quantization staging in v1).
// --------------------------------------------------------------------------------

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "generated_cases.h"

using namespace pto;

namespace hc_mix_gemm_cfg {

// Shape constants from the family manifest (../build/generated_cases.h).
constexpr unsigned kB      = static_cast<unsigned>(kHcB);
constexpr unsigned kS      = static_cast<unsigned>(kHcS);
constexpr unsigned kDim    = static_cast<unsigned>(kHcDim);
constexpr unsigned kHcM    = static_cast<unsigned>(kHcMult);
constexpr unsigned kMixHc  = static_cast<unsigned>(kHcMixHc);

constexpr unsigned kM      = kB * kS;          // token count
constexpr unsigned kK      = kHcM * kDim;      // flattened HC inner dim
constexpr unsigned kN      = kMixHc;           // output features per token

// Cube tile constants (tentative — see README.md memory-budget plan).
// Keep multiples of 16 to respect the cube fractal granularity (16x16x16).
// N = MIX_HC = 24 is small; the implementer may need to pad to 32 to honor
// fractal alignment. See README "Auto-mode constraints" for the assumption.
constexpr unsigned kTileM = 16;
constexpr unsigned kTileN = 32;  // N_padded; pad MIX_HC=24 -> 32 for cube fractal.
constexpr unsigned kTileK = 64;

}  // namespace hc_mix_gemm_cfg

extern "C" __global__ AICORE void hc_mix_gemm_kernel(
    __gm__ uint8_t *mixes_raw,   // FP32, (M, N) row-major  (output)
    __gm__ uint8_t *x_raw,       // FP32, (M, K) row-major  (x_flat)
    __gm__ uint8_t *hc_fn_raw,   // FP32, (N, K) row-major  (hc_fn weight)
    __gm__ uint8_t *rsqrt_raw)   // FP32, (M,)              (precomputed rsqrt)
{
    using namespace hc_mix_gemm_cfg;

    // ----------------------------------------------------------------------
    // PSEUDOCODE - auto-mode A3 cube GEMM, FP32 x FP32 -> FP32, with a
    // per-row vector epilogue (* rsqrt[m]) before GM store.
    //
    // Pattern mirrors MoE/router_matmul/router_matmul_kernel.cpp (cube path
    // with bMatTile reload per m0) and z_gemm_kernel.cpp.
    //
    //   __gm__ float *mixes = reinterpret_cast<__gm__ float *>(mixes_raw);
    //   __gm__ float *x     = reinterpret_cast<__gm__ float *>(x_raw);
    //   __gm__ float *hc_fn = reinterpret_cast<__gm__ float *>(hc_fn_raw);
    //   __gm__ float *rsqrt = reinterpret_cast<__gm__ float *>(rsqrt_raw);
    //
    //   GlobalTensor<float> gX     (x,     {M, K});       // (M, K)
    //   GlobalTensor<float> gWk    (hc_fn, {N, K});       // (N, K) - treat as Wkv^T
    //   GlobalTensor<float> gMix   (mixes, {M, N});       // (M, N)
    //   GlobalTensor<float> gRsqrt (rsqrt, {M});          // (M,)   - vector input
    //
    //   for (mTile = 0; mTile < ceildiv(M, kTileM); ++mTile) {
    //     Tile<float, MemL1>   atile;        // L1 A panel  [kTileM, K_l1]
    //     Tile<float, MemL1>   btile;        // L1 B panel  [N_padded, K_l1]
    //     Tile<float, MemL0C>  ctile;        // L0C accumulator [kTileM, kTileN]
    //     TileLeft<float,  MemL0A>  aL0A;    // A fractal
    //     TileRight<float, MemL0B>  bL0B;    // B fractal (hc_fn^T view)
    //
    //     for (kTile = 0; kTile < ceildiv(K, kTileK); ++kTile) {
    //       TLOAD(atile, gX [mTile*kTileM : (mTile+1)*kTileM,
    //                        kTile*kTileK : (kTile+1)*kTileK]);
    //       TLOAD(btile, gWk[0 : kTileN,
    //                        kTile*kTileK : (kTile+1)*kTileK]);
    //
    //       if (kTile == 0) {
    //         TMATMUL    (ctile, atile, btile);   // zero-init L0C
    //       } else {
    //         TMATMUL_ACC(ctile, atile, btile);   // accumulate
    //       }
    //     }
    //
    //     // ---- Vector epilogue (per-row * rsqrt[m]) ---------------------
    //     // Stage ctile (L0C) -> UB; load rsqrt slice; broadcast-multiply
    //     // each row by its rsqrt scalar; then TSTORE to GM.
    //     //
    //     //   Tile<float, MemUB> ubMix;
    //     //   Tile<float, MemUB> ubRsqrt;
    //     //   TLOAD(ubMix,   ctile);
    //     //   TLOAD(ubRsqrt, gRsqrt[mTile*kTileM : (mTile+1)*kTileM]);
    //     //   TMUL(ubMix, ubMix, ubRsqrt /* row-broadcast */);
    //     //   TSTORE(gMix[mTile*kTileM : (mTile+1)*kTileM, 0:kN], ubMix);
    //     //
    //     // Note: N=MIX_HC=24 is the *valid* output column count; the
    //     // padded fractal width is kTileN=32 (assume MIX_HC mod 16 != 0).
    //     // Auto-mode valid-region handles trimming to kN on the GM write.
    //   }
    //
    // ----------------------------------------------------------------------
    // TODO(deepseekv4-hc): replace pseudocode above with real auto-mode body.
    // Anchor on MoE/router_matmul/router_matmul_kernel.cpp for the cube
    // reload + TMATMUL_ACC pattern and add_tile_array_kernel.cpp for the
    // baseline auto-mode call shape (TLOAD/TSTORE).
    (void)mixes_raw;
    (void)x_raw;
    (void)hc_fn_raw;
    (void)rsqrt_raw;
}

// Host launcher — minimal stub. The real launcher will go through
// pto-launch helpers similar to MoE/router_matmul/router_matmul_kernel.cpp.
extern "C" void launch_hc_mix_gemm(uint8_t *mixes, uint8_t *x, uint8_t *hc_fn,
                                   uint8_t *rsqrt, void *stream)
{
    // TODO(deepseekv4-hc): wire ICache, AICORE config, then launch the
    // kernel with <<<1, nullptr, stream>>>.
    (void)mixes; (void)x; (void)hc_fn; (void)rsqrt; (void)stream;
}
