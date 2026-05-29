// --------------------------------------------------------------------------------
// expert_ffn — DeepSeek-V4 per-expert SwiGLU FFN (auto-mode A3, cube + vector).
//
//   B[m, :]    = W2 · ( silu(clamp_g(W1·A[m, :])) * clamp_u(W3·A[m, :]) * w[m] )
//
//   - clamp_g(x) = min(x, swiglu_limit)            (model.py:603)
//   - clamp_u(x) = clip(x, -swiglu_limit, swiglu_limit)   (model.py:602)
//   - w[m] = weights[m] when weights != nullptr    (model.py:606)
//
// DESIGN / SKELETON ONLY — pseudocode body below. Real implementation is
// deferred. Do not claim compile or runtime success until run.sh produces
// `test data success` on the compiler server.
//
// Pattern source: MoE/expert_ffn/expert_ffn_kernel.cpp (mirror exactly;
// only the dtype changes from half → bfloat16_t, and the FFN body fuses
// a second W3 GEMM + SiLU/clamp/weights composition into the L0C path
// before the second W2 GEMM).
// --------------------------------------------------------------------------------

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "generated_cases.h"

using namespace pto;

namespace expert_ffn_cfg {

constexpr unsigned kT          = kDsmoeT;
constexpr unsigned kDim        = kDsmoeDim;          // GEMM1 K, GEMM2 N
constexpr unsigned kInterDim   = kDsmoeInterDim;     // GEMM1 N, GEMM2 K
constexpr unsigned kNRouted    = kDsmoeNRouted;
constexpr unsigned kNActivated = kDsmoeNActivated;
constexpr unsigned kTileM      = 16;                 // expert-local token tile height

constexpr unsigned kPackedRows = kT * kNActivated;

constexpr int kWorkingSetBudgetBytes = 1 << 17;
constexpr int kL0BudgetBytes         = 1 << 14;

#ifndef DSMOE_SWIGLU_LIMIT
#define DSMOE_SWIGLU_LIMIT 0.0f
#endif
constexpr float kSwigluLimit = static_cast<float>(DSMOE_SWIGLU_LIMIT);

#ifndef DSMOE_APPLY_ROUTING_WEIGHT
#define DSMOE_APPLY_ROUTING_WEIGHT 0
#endif
constexpr bool  kApplyWeights = (DSMOE_APPLY_ROUTING_WEIGHT != 0);

}  // namespace expert_ffn_cfg

extern "C" __global__ AICORE void expert_ffn_kernel(
    __gm__ uint8_t *B_raw,           // FP32, (kAlloc, kDim) — out
    __gm__ uint8_t *A_raw,           // BF16, (kAlloc, kDim) — in (scatter output)
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ uint8_t *W1_raw,          // BF16, (kNRouted, kInterDim, kDim) — gate proj
    __gm__ uint8_t *W3_raw,          // BF16, (kNRouted, kInterDim, kDim) — up proj
    __gm__ uint8_t *W2_raw,          // BF16, (kNRouted, kDim, kInterDim) — out proj
    __gm__ float   *weights)         // FP32, (kAlloc,) — optional routing weights (or nullptr)
{
    using namespace expert_ffn_cfg;
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 SwiGLU FFN per expert.
    //
    // The cube body is structurally identical to MoE/expert_ffn but with a
    // parallel W3 GEMM and a SwiGLU compose step in UB before the W2 GEMM.
    //
    //   for (e = 0; e < kNRouted; ++e) {
    //     int32_t start = expert_start[e];
    //     int32_t count = expert_count[e];
    //     for (int32_t m0 = 0; m0 < count; m0 += kTileM) {
    //       unsigned currentM = min(count - m0, kTileM);
    //
    //       // ---- GEMM1: gate = A @ W1[e]^T  (BF16 × BF16 → FP32) ----
    //       // ---- GEMM1b: up  = A @ W3[e]^T  (BF16 × BF16 → FP32) ----
    //       // (interleaved K loop, same pattern as MoE/expert_ffn's W1 loop)
    //       //
    //       //   for (n0 over kInterDim)
    //       //     for (k loop)
    //       //       TMATMUL_ACC(gateAcc, aTile, w1Tile);
    //       //       TMATMUL_ACC(upAcc,   aTile, w3Tile);
    //       //
    //       // Bring both accumulators to BF16 in L1 via TMOV; before TMOV,
    //       // apply the SwiGLU compose in UB:
    //       //
    //       //   if (kSwigluLimit > 0) {
    //       //     TMINS(gate, gate, kSwigluLimit);                // clamp_g
    //       //     TMINS(up,   up,   kSwigluLimit);                // upper
    //       //     TMAXS(up,   up,  -kSwigluLimit);                // lower
    //       //   }
    //       //   TSILU(silu, gate);                                // SiLU
    //       //   TMUL (y,    silu, up);                            // gate*up
    //       //   if (kApplyWeights) {
    //       //     // weights[m] broadcast across F_l1 columns
    //       //     TROWEXPANDMUL(y, y, wTile);
    //       //   }
    //       //   TMOV<...>(yMatTile, y);                           // to L1 BF16
    //
    //       // ---- GEMM2: B[m, :] = y @ W2[e]^T  (BF16 × BF16 → FP32) ----
    //       //   for (n0 over kDim)
    //       //     for (k loop)
    //       //       TMATMUL_ACC(bAccTile, yTile, w2Tile);
    //       //   TSTORE(B[row, :], bAccTile);
    //     }
    //   }
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Mirror MoE/expert_ffn/expert_ffn_kernel.cpp tile-budget helpers
    // (chooseNPanel, chooseFPanel, chooseHPanel, chooseH0Panel) verbatim
    // — they generalize to BF16 by parameterizing TIn / TWeight.
    (void)B_raw; (void)A_raw;
    (void)expert_count; (void)expert_start;
    (void)W1_raw; (void)W3_raw; (void)W2_raw;
    (void)weights;
}

void launch_expert_ffn(uint8_t *B, uint8_t *A,
                      int32_t *expert_count, int32_t *expert_start,
                      uint8_t *W1, uint8_t *W3, uint8_t *W2,
                      float *weights, void *stream)
{
    // TODO(deepseekv4): launch expert_ffn_kernel via auto-mode triple-bracket
    // syntax once the cube body is implemented.
    expert_ffn_kernel<<<1, nullptr, stream>>>(
        B, A, expert_count, expert_start, W1, W3, W2, weights);
}
