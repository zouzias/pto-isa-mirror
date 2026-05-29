// --------------------------------------------------------------------------------
// gate_logits — DeepSeek-V4 router GEMM (auto-mode A3, cube, FP32).
//
// scores[T, N_ROUTED] = X[T, DIM] @ W_gate[N_ROUTED, DIM]^T
//
// All tensors are FP32 — DeepSeek-V4's Gate.forward explicitly casts to
// float() for numerical stability (deepseek/model.py:566).
//
// DESIGN / SKELETON ONLY — pseudocode body below. Real implementation is
// deferred. Do not claim compile or runtime success until run.sh produces
// `test data success` on the compiler server.
//
// Pattern source (treat W_gate analogously to W_router):
//   kernels/automode/a2a3/MoE/router_matmul/router_matmul_kernel.cpp
// --------------------------------------------------------------------------------

#include <cstdint>
#include "common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
#include "generated_cases.h"

using namespace pto;

// Cube tile constants (tentative — see README.md memory-budget plan).
// FP32 takes 2× the L0 bytes of BF16, so tiles shrink relative to
// MoE/router_matmul. Keep all multiples of 16 (cube fractal granularity).
constexpr uint32_t kTileT = 64;          // M (tokens) tile
constexpr uint32_t kTileE = 16;          // N (routed-experts) tile, rounded up from kDsmoeNRouted=8
constexpr uint32_t kTileK = 64;          // K (dim) tile

// Shape constants from the family manifest (../build/generated_cases.h):
//   kDsmoeT, kDsmoeDim, kDsmoeNRouted.

extern "C" __global__ __aicore__ void gate_logits_kernel(
    __gm__ uint8_t *scores,    // FP32, (T, N_ROUTED) row-major
    __gm__ uint8_t *X,         // FP32, (T, DIM)      row-major
    __gm__ uint8_t *W_gate)    // FP32, (N_ROUTED, DIM) row-major (treated as W^T from python)
{
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 cube GEMM, FP32 × FP32 → FP32.
    // Mirrors router_matmul (two-level K tiling, bMatTile reload).
    //
    //   GlobalTensor<float> gX  (X,      {T, DIM});
    //   GlobalTensor<float> gW  (W_gate, {N_ROUTED, DIM});
    //   GlobalTensor<float> gS  (scores, {T, N_ROUTED});
    //
    //   for (tTile = 0; tTile < ceildiv(T, kTileT); ++tTile) {
    //     Tile<float, MemL1>   xtile;          // L1 X panel
    //     Tile<float, MemL1>   wtile;          // L1 W panel
    //     Tile<float, MemL0C>  ctile;          // L0C accumulator
    //     TileLeft<float,  MemL0A>  xL0A;
    //     TileRight<float, MemL0B>  wL0B;
    //
    //     TCLEAR(ctile);
    //     for (kTile = 0; kTile < ceildiv(DIM, kTileK); ++kTile) {
    //       TLOAD(xtile, gX[tTile*kTileT : (tTile+1)*kTileT, kTile*kTileK : (kTile+1)*kTileK]);
    //       TLOAD(wtile, gW[0 : kTileE,                       kTile*kTileK : (kTile+1)*kTileK]);
    //       TLOAD(xL0A, xtile);
    //       TLOAD(wL0B, wtile);
    //       if (kTile == 0) {
    //         TMATMUL    (ctile, xL0A, wL0B);    // first FMA — overwrite
    //       } else {
    //         TMATMUL_ACC(ctile, xL0A, wL0B);    // accumulate
    //       }
    //     }
    //     TSTORE(gS[tTile*kTileT : (tTile+1)*kTileT, 0 : kDsmoeNRouted], ctile);
    //   }
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on MoE/router_matmul/router_matmul_kernel.cpp for the cube
    // reload pattern and tile-budget calculation.
    (void)scores; (void)X; (void)W_gate;
}

// Host launcher (kept minimal — real launcher will go through pto-launch
// helpers similar to MoE/router_matmul).
void launch_gate_logits(uint8_t *scores, uint8_t *X, uint8_t *W_gate,
                        uint64_t T, uint64_t N, uint64_t K, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch gate_logits_kernel.
    (void)scores; (void)X; (void)W_gate; (void)T; (void)N; (void)K; (void)stream;
}
