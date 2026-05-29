// --------------------------------------------------------------------------------
// c_gemm — Compressor.wgate linear projection (auto-mode A3, cube).
//
// score[M, N] = X[M, K] @ Wgate[N, K]^T          (treat Wgate as transposed for cube)
//   M = B * S         (token count)
//   K = DIM           (model dim = 4096 default)
//   N = COFF*HEAD_DIM (1024 with overlap, 512 without)
//
// Reference: deepseek/model.py:325 (Compressor.forward)
//            deepseek/kernel.py: dispatched via `linear()` (model.py:109)
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

// Cube tile constants (tentative — see README.md memory-budget plan).
// Adjust after profiling; keep all multiples of 16 to respect the cube
// fractal granularity (16x16x16).
constexpr uint32_t kTileM = 128;
constexpr uint32_t kTileN = 128;
constexpr uint32_t kTileK = 64;

// Shape constants from the family manifest (../build/generated_cases.h).
//   kCompM    = B*S
//   kCompK    = DIM
//   kCompN    = COFF * HEAD_DIM
// Assumption: generated_cases.h emits kCompB/kCompS/kCompDim/kCompHeadDim/
// kCompCoff aliases — match the convention of `kMoeT/kMoeH/...` in
// MoE/scripts/generate_cases.py.

extern "C" __global__ __aicore__ void c_gemm_kernel(
    __gm__ uint8_t *out,    // FP32, (M, N) row-major
    __gm__ uint8_t *X,      // BF16, (M, K) row-major
    __gm__ uint8_t *Wgate)  // BF16, (N, K) row-major  (treat as Wgate^T from python)
{
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 cube GEMM, BF16 x BF16 -> FP32.
    // Pattern mirrors moe_segmented_gemm_relu / router_matmul (cube path).
    //
    //   GlobalTensor<bfloat16_t> gX  (X,     {M, K});
    //   GlobalTensor<bfloat16_t> gWg (Wgate, {N, K});
    //   GlobalTensor<float>      gS  (out,   {M, N});
    //
    //   for (mTile = 0; mTile < ceildiv(M, kTileM); ++mTile) {
    //     Tile<bfloat16_t, MemL1>   atile;        // L1 A panel
    //     Tile<bfloat16_t, MemL1>   btile;        // L1 B panel (Wgate tile)
    //     Tile<float,      MemL0C>  ctile;        // L0C accumulator
    //     TileLeft<bfloat16_t,  MemL0A>  aL0A;    // A fractal
    //     TileRight<bfloat16_t, MemL0B>  bL0B;    // B fractal (Wgate^T view)
    //
    //     for (nTile = 0; nTile < ceildiv(N, kTileN); ++nTile) {
    //       // Reload B every (mTile, nTile) — matches the moe_segmented
    //       // bMatTile pattern; cheap because Wgate is reused across M.
    //       TLOAD(btile, gWg[nTile * kTileN : (nTile+1)*kTileN, :]);
    //       TLOAD(atile, gX [mTile * kTileM : (mTile+1)*kTileM, :]);
    //
    //       TCLEAR(ctile);                       // zero L0C accumulator
    //       for (kTile = 0; kTile < ceildiv(K, kTileK); ++kTile) {
    //         TLOAD(aL0A, atile.slice(:, kTile*kTileK : (kTile+1)*kTileK));
    //         TLOAD(bL0B, btile.slice(:, kTile*kTileK : (kTile+1)*kTileK));
    //         TMATMUL(ctile, aL0A, bL0B);        // FMA on 16x16 fractals
    //       }
    //
    //       // Tail M handled by the auto-mode pass via valid-region; no
    //       // explicit branch in source.
    //       TSTORE(gS[mTile*kTileM : (mTile+1)*kTileM,
    //                 nTile*kTileN : (nTile+1)*kTileN], ctile);
    //     }
    //   }
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on kernels/automode/a2a3/MoE/router_matmul/router_matmul_kernel.cpp
    // for the cube reload pattern, and add_tile_array_kernel.cpp for the
    // baseline auto-mode call shape.
    (void)out; (void)X; (void)Wgate;
}

// Host launcher (kept minimal — the real launcher will go through pto-launch
// helpers similar to MoE/router_matmul/router_matmul_kernel.cpp).
void launch_c_gemm(uint8_t *out, uint8_t *X, uint8_t *Wgate,
                   uint64_t M, uint64_t N, uint64_t K, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch c_gemm_kernel.
    (void)out; (void)X; (void)Wgate; (void)M; (void)N; (void)K; (void)stream;
}
