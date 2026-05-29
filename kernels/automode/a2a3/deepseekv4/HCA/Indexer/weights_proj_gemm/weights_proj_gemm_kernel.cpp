// --------------------------------------------------------------------------------
// weights_proj_gemm — Indexer per-head weights projection (auto-mode A3, cube).
//
// weights[M, N] = (X[M, K] @ W_wproj[N, K]^T) * scale
//   M = B * S
//   K = DIM                 (4096 default)
//   N = N_HEADS             (64 default, small)
//   scale = softmax_scale * n_heads^-0.5     (model.py:395, model.py:419)
//
// Reference: deepseek/model.py:419 (Indexer.forward, weights = wproj(x) * scale)
//            deepseek/model.py:395 (softmax_scale = head_dim^-0.5)
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

constexpr uint32_t kTileM = 64;
constexpr uint32_t kTileN = 64;  // n_heads default 64 fits in one N tile
constexpr uint32_t kTileK = 64;

// Shape constants from the family manifest (../build/generated_cases.h):
//   kIdxB, kIdxS, kIdxDim, kIdxNHeads, kIdxHeadDim, ...

extern "C" __global__ __aicore__ void weights_proj_gemm_kernel(
    __gm__ uint8_t *out,         // FP32, (M, N) row-major
    __gm__ uint8_t *X,           // BF16, (M, K) row-major
    __gm__ uint8_t *W_wproj,     // BF16, (N, K) row-major
    float           scale)        // softmax_scale * n_heads^-0.5
{
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 cube GEMM with post-multiply scale.
    //
    //   GlobalTensor<bfloat16_t> gX (X,       {M, K});
    //   GlobalTensor<bfloat16_t> gW (W_wproj, {N, K});
    //   GlobalTensor<float>      gO (out,     {M, N});
    //
    //   // W_wproj is small (N = n_heads <= 64). Preload once outside the M loop.
    //   Tile<bfloat16_t, MemL1>  btile;
    //   TLOAD(btile, gW[:, :]);
    //
    //   for (mTile = 0; mTile < ceildiv(M, kTileM); ++mTile) {
    //     Tile<bfloat16_t, MemL1>   atile;
    //     Tile<float,      MemL0C>  ctile;
    //     TileLeft<bfloat16_t,  MemL0A>  aL0A;
    //     TileRight<bfloat16_t, MemL0B>  bL0B;
    //
    //     TLOAD(atile, gX[mTile*kTileM : (mTile+1)*kTileM, :]);
    //
    //     // N fits in one N tile (kTileN >= n_heads in default shape),
    //     // so there's no nTile loop. For larger n_heads, mirror z_gemm.
    //     TCLEAR(ctile);
    //     for (kTile = 0; kTile < ceildiv(K, kTileK); ++kTile) {
    //       TLOAD(aL0A, atile.slice(:, kTile*kTileK : (kTile+1)*kTileK));
    //       TLOAD(bL0B, btile.slice(:, kTile*kTileK : (kTile+1)*kTileK));
    //       TMATMUL(ctile, aL0A, bL0B);
    //     }
    //
    //     // Post-scale: either (a) fuse into the L0C->UB copy with a scalar
    //     // multiply on the drain, or (b) materialize ctile to UB and apply
    //     // a vector mul with broadcast scalar `scale`. Decision deferred.
    //     //
    //     // Option (a) sketch:
    //     //   TSTORE_SCALED(gO[mTile*kTileM:(mTile+1)*kTileM, :], ctile, scale);
    //     // Option (b) sketch:
    //     //   Tile<float, MemUB> ub;
    //     //   TCOPY(ub, ctile);
    //     //   TMULS(ub, scale);            // FP32 broadcast scalar mul
    //     //   TSTORE(gO[mTile*kTileM:(mTile+1)*kTileM, :], ub);
    //     //
    //     // For the prototype, scaffold option (b).
    //   }
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on z_gemm + the auto-mode UB scalar-multiply pattern from
    // add_tile_array; pick the scale-handling option after profiler feedback.
    (void)out; (void)X; (void)W_wproj; (void)scale;
}

// Host launcher (minimal scaffold).
void launch_weights_proj_gemm(uint8_t *out, uint8_t *X, uint8_t *W_wproj,
                              float scale,
                              uint64_t M, uint64_t N, uint64_t K, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch the kernel.
    (void)out; (void)X; (void)W_wproj; (void)scale;
    (void)M; (void)N; (void)K; (void)stream;
}
