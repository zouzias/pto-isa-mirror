// --------------------------------------------------------------------------------
// gather — DeepSeek-V4 expert-output → token-position recombine
// (auto-mode A3, vector).
//
//   For each packed row r in [0, T*N_ACTIVATED):
//       t = A_id[r]
//       k = rank_id[r]
//       w = weights[t * N_ACTIVATED + k]    (FP32 scalar)
//       Y[t] += w * B[r]                    (BF16 row, FP32-scaled accumulate)
//
// Y is assumed host-zeroed before kernel launch.
//
// DESIGN / SKELETON ONLY — pseudocode body below. Real implementation is
// deferred. Do not claim compile or runtime success until run.sh produces
// `test data success` on the compiler server.
//
// Pattern source:
//   - MoE/gather/gather_kernel.cpp  (kTopK > 1 branch, pass 2 weighted
//     scatter-add; DeepSeek-V4 skips that file's pass 1 softmax because
//     upstream already normalizes the weights.)
// --------------------------------------------------------------------------------

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "generated_cases.h"

using namespace pto;

namespace gather_cfg {
constexpr unsigned kT          = kDsmoeT;
constexpr unsigned kDim        = kDsmoeDim;
constexpr unsigned kNActivated = kDsmoeNActivated;

constexpr unsigned kPackedRows   = kT * kNActivated;
constexpr unsigned kOverspillPad = 16;
constexpr unsigned kAlloc        = kPackedRows + kOverspillPad;
}  // namespace gather_cfg

extern "C" __global__ AICORE void gather_kernel(
    __gm__ uint8_t *Y_raw,            // BF16, (T, DIM) — in/out (zero-init by host)
    __gm__ uint8_t *B_raw,            // BF16, (kAlloc, DIM) — in (expert_ffn output)
    __gm__ int32_t *A_id,             // INT32, (kAlloc,)    — in
    __gm__ int32_t *rank_id,          // INT32, (kAlloc,)    — in
    __gm__ float   *weights)          // FP32, (T, N_ACTIVATED) — in
{
    using namespace gather_cfg;
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 vector weighted scatter-add.
    //
    //   __gm__ bfloat16_t *Y = reinterpret_cast<__gm__ bfloat16_t *>(Y_raw);
    //   __gm__ bfloat16_t *B = reinterpret_cast<__gm__ bfloat16_t *>(B_raw);
    //
    //   using RowShape  = Shape <1, 1, 1, 1, kDim>;
    //   using RowStride = Stride<1, 1, 1, kDim, 1>;
    //   using RowGlobal = GlobalTensor<bfloat16_t, RowShape, RowStride>;
    //   using RowTile   = Tile<TileType::Vec, bfloat16_t,
    //                          1, kDim,
    //                          BLayout::RowMajor,
    //                          1, kDim>;
    //
    //   RowTile bTile;
    //   RowTile yTile;
    //   RowTile scaledTile;
    //   RowTile sumTile;
    //
    //   for (unsigned r = 0; r < kPackedRows; ++r) {
    //       pipe_barrier(PIPE_ALL);                  // cross-iter sync guard
    //
    //       int32_t t = A_id   [r];                  // GM scalar reads
    //       int32_t k = rank_id[r];
    //       float   w = weights[t * kNActivated + k];
    //
    //       size_t src_off = static_cast<size_t>(r) * kDim;
    //       size_t dst_off = static_cast<size_t>(t) * kDim;
    //       RowGlobal bGlobal(B + src_off);
    //       RowGlobal yGlobal(Y + dst_off);
    //
    //       TLOAD (bTile, bGlobal);
    //       TMULS (scaledTile, bTile, w);            // BF16-out scaled by FP32 scalar
    //       TLOAD (yTile, yGlobal);
    //       TADD  (sumTile, yTile, scaledTile);
    //       TSTORE(yGlobal, sumTile);
    //   }
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on MoE/gather/gather_kernel.cpp pass 2 (kTopK > 1 branch).
    (void)Y_raw; (void)B_raw; (void)A_id; (void)rank_id; (void)weights;
}

void launch_gather(uint8_t *Y, uint8_t *B,
                   int32_t *A_id, int32_t *rank_id,
                   float *weights, void *stream)
{
    gather_kernel<<<1, nullptr, stream>>>(Y, B, A_id, rank_id, weights);
}
