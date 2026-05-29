// --------------------------------------------------------------------------------
// shared_expert_ffn — DeepSeek-V4 always-on shared expert (auto-mode A3, cube + vector).
//
//   Y[m, :] = W2 · ( silu(clamp_g(W1·X[m, :])) * clamp_u(W3·X[m, :]) )
//
// Structurally identical to expert_ffn but without the routing axis
// (single expert, all tokens, no per-row weight).
// Reference: deepseek/model.py:628 (constructor), model.py:644 (call site).
//
// DESIGN / SKELETON ONLY — pseudocode body below. Real implementation is
// deferred. Do not claim compile or runtime success until run.sh produces
// `test data success` on the compiler server.
//
// Pattern source:
//   - MoE/expert_ffn/expert_ffn_kernel.cpp (mirror the cube body; drop
//     the e outer loop, drop the expert-axis offsets on W1/W2/W3,
//     drop the per-row routing-weight broadcast.)
// --------------------------------------------------------------------------------

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "generated_cases.h"

using namespace pto;

namespace shared_expert_ffn_cfg {

constexpr unsigned kT        = kDsmoeT;
constexpr unsigned kDim      = kDsmoeDim;       // GEMM1 K, GEMM2 N
constexpr unsigned kInterDim = kDsmoeInterDim;  // GEMM1 N, GEMM2 K
constexpr unsigned kTileM    = 16;

constexpr int kWorkingSetBudgetBytes = 1 << 17;
constexpr int kL0BudgetBytes         = 1 << 14;

#ifndef DSMOE_SWIGLU_LIMIT
#define DSMOE_SWIGLU_LIMIT 0.0f
#endif
constexpr float kSwigluLimit = static_cast<float>(DSMOE_SWIGLU_LIMIT);

}  // namespace shared_expert_ffn_cfg

extern "C" __global__ AICORE void shared_expert_ffn_kernel(
    __gm__ uint8_t *Y_raw,    // BF16, (T, DIM) — out
    __gm__ uint8_t *X_raw,    // BF16, (T, DIM) — in
    __gm__ uint8_t *W1_raw,   // BF16, (INTER_DIM, DIM) — gate proj
    __gm__ uint8_t *W3_raw,   // BF16, (INTER_DIM, DIM) — up proj
    __gm__ uint8_t *W2_raw)   // BF16, (DIM, INTER_DIM) — out proj
{
    using namespace shared_expert_ffn_cfg;
    // ----------------------------------------------------------------------
    // PSEUDOCODE — same body as expert_ffn but with a flat single-segment
    // outer loop (start=0, count=kT) and no per-row routing-weight scale.
    //
    //   for (int32_t m0 = 0; m0 < kT; m0 += kTileM) {
    //       unsigned currentM = min(kT - m0, kTileM);
    //
    //       // GEMM1a:   gate = X @ W1^T  (BF16 × BF16 → FP32)
    //       // GEMM1b:   up   = X @ W3^T  (BF16 × BF16 → FP32)
    //       // (interleaved K loop)
    //
    //       // SwiGLU compose (UB):
    //       //   if (kSwigluLimit > 0) {
    //       //     TMINS(gate, gate, kSwigluLimit);
    //       //     TMINS(up,   up,   kSwigluLimit);
    //       //     TMAXS(up,   up,  -kSwigluLimit);
    //       //   }
    //       //   TSILU(silu, gate);
    //       //   TMUL (y,    silu, up);
    //       //   TMOV<...>(yMatTile, y);                      // BF16 to L1
    //
    //       // GEMM2:    Y[m0, :] = y @ W2^T  (BF16 × BF16 → FP32)
    //       //   TSTORE(Y[m0:m0+currentM, :], bAccTile);
    //   }
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Mirror MoE/expert_ffn/expert_ffn_kernel.cpp; drop the e outer loop
    // and drop the per-row routing-weight broadcast tile.
    (void)Y_raw; (void)X_raw; (void)W1_raw; (void)W3_raw; (void)W2_raw;
}

void launch_shared_expert_ffn(uint8_t *Y, uint8_t *X,
                              uint8_t *W1, uint8_t *W3, uint8_t *W2,
                              void *stream)
{
    shared_expert_ffn_kernel<<<1, nullptr, stream>>>(Y, X, W1, W3, W2);
}
