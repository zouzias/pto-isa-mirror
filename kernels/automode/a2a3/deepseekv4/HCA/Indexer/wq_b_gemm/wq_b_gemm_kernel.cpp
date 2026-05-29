// --------------------------------------------------------------------------------
// wq_b_gemm — Indexer.wq_b Q low-rank expansion (auto-mode A3, cube).
//
// q[M, N] = qr[M, K] @ Wq_b[N, K]^T       (treat Wq_b as transposed for cube)
//   M = B * S
//   K = Q_LORA_RANK              (1024 default; small for the prototype)
//   N = N_HEADS * HEAD_DIM       (64*128 = 8192 default)
//
// Reference: deepseek/model.py:412 (Indexer.forward, `q = self.wq_b(qr)`)
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
constexpr uint32_t kTileM = 64;
constexpr uint32_t kTileN = 128;
constexpr uint32_t kTileK = 64;

// Shape constants from the family manifest (../build/generated_cases.h):
//   kIdxB, kIdxS, kIdxQLoraRank, kIdxNHeads, kIdxHeadDim, ...

extern "C" __global__ __aicore__ void wq_b_gemm_kernel(
    __gm__ uint8_t *out,    // FP32, (M, N) row-major
    __gm__ uint8_t *qr,     // BF16, (M, K) row-major
    __gm__ uint8_t *Wq_b)   // BF16, (N, K) row-major  (Wq_b^T from python)
{
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 cube GEMM, BF16 x BF16 -> FP32.
    // Pattern mirrors z_gemm / MoE/router_matmul (cube path).
    //
    //   GlobalTensor<bfloat16_t> gQr (qr,   {M, K});
    //   GlobalTensor<bfloat16_t> gWb (Wq_b, {N, K});
    //   GlobalTensor<float>      gQ  (out,  {M, N});
    //
    //   for (mTile = 0; mTile < ceildiv(M, kTileM); ++mTile) {
    //     Tile<bfloat16_t, MemL1>   atile;        // L1 A panel  (qr)
    //     Tile<bfloat16_t, MemL1>   btile;        // L1 B panel  (Wq_b row block)
    //     Tile<float,      MemL0C>  ctile;        // L0C accumulator
    //     TileLeft<bfloat16_t,  MemL0A>  aL0A;    // A fractal
    //     TileRight<bfloat16_t, MemL0B>  bL0B;    // B fractal (Wq_b^T view)
    //
    //     for (nTile = 0; nTile < ceildiv(N, kTileN); ++nTile) {
    //       // Reload B every (mTile, nTile) — bMatTile reload pattern
    //       // from MoE/router_matmul.
    //       TLOAD(btile, gWb[nTile * kTileN : (nTile+1)*kTileN, :]);
    //       TLOAD(atile, gQr[mTile * kTileM : (mTile+1)*kTileM, :]);
    //
    //       TCLEAR(ctile);                       // zero L0C accumulator
    //       for (kTile = 0; kTile < ceildiv(K, kTileK); ++kTile) {
    //         TLOAD(aL0A, atile.slice(:, kTile*kTileK : (kTile+1)*kTileK));
    //         TLOAD(bL0B, btile.slice(:, kTile*kTileK : (kTile+1)*kTileK));
    //         TMATMUL(ctile, aL0A, bL0B);        // FMA on 16x16 fractals
    //       }
    //
    //       // Tail M handled by the auto-mode pass via valid-region.
    //       TSTORE(gQ[mTile*kTileM : (mTile+1)*kTileM,
    //                 nTile*kTileN : (nTile+1)*kTileN], ctile);
    //     }
    //   }
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on HCA/Compressor/z_gemm/z_gemm_kernel.cpp once that builds,
    // and on MoE/router_matmul/router_matmul_kernel.cpp for the cube
    // reload pattern.
    (void)out; (void)qr; (void)Wq_b;
}

// Host launcher (kept minimal — the real launcher will go through pto-launch
// helpers similar to MoE/router_matmul/router_matmul_kernel.cpp).
void launch_wq_b_gemm(uint8_t *out, uint8_t *qr, uint8_t *Wq_b,
                      uint64_t M, uint64_t N, uint64_t K, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch wq_b_gemm_kernel.
    (void)out; (void)qr; (void)Wq_b; (void)M; (void)N; (void)K; (void)stream;
}
