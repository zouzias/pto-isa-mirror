// --------------------------------------------------------------------------------
// scatter — DeepSeek-V4 token-to-expert reorder (auto-mode A3, vector).
//
// Three-pass scatter (mirrors MoE/scatter/scatter_kernel.cpp; dtype is BF16
// instead of FP16 for the row data):
//   Pass 1: histogram  count[e] = #{(t, k) : expert_id[t, k] == e}
//   Pass 2: prefix sum start[e] = sum_{i<e} count[i]
//   Pass 3: pack       per (t, k) in row-major order, copy X[t] to
//                       A[start[expert_id[t,k]] + counter[e]++].
//                       Also record A_id[r]=t, rank_id[r]=k.
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

namespace scatter_cfg {
constexpr unsigned kT          = kDsmoeT;
constexpr unsigned kDim        = kDsmoeDim;
constexpr unsigned kNRouted    = kDsmoeNRouted;
constexpr unsigned kNActivated = kDsmoeNActivated;

constexpr unsigned kPackedRows   = kT * kNActivated;
constexpr unsigned kOverspillPad = 16;
constexpr unsigned kAlloc        = kPackedRows + kOverspillPad;
}  // namespace scatter_cfg

extern "C" __global__ __aicore__ void scatter_kernel(
    __gm__ uint8_t *A_raw,           // BF16, (kAlloc, kDim) — out
    __gm__ int32_t *A_id,            // INT32, (kAlloc,)     — out
    __gm__ int32_t *rank_id,         // INT32, (kAlloc,)     — out
    __gm__ int32_t *expert_count,    // INT32, (kNRouted,)   — out
    __gm__ int32_t *expert_start,    // INT32, (kNRouted,)   — out
    __gm__ uint8_t *X_raw,           // BF16, (kT, kDim)     — in
    __gm__ int32_t *expert_id)       // INT32, (kT, kNActivated) — in
{
    using namespace scatter_cfg;
    // ----------------------------------------------------------------------
    // PSEUDOCODE — mirrors MoE/scatter/scatter_kernel.cpp exactly; only
    // the data dtype changes from `half` to `bfloat16_t`.
    //
    //   __gm__ bfloat16_t *A = reinterpret_cast<__gm__ bfloat16_t *>(A_raw);
    //   __gm__ bfloat16_t *X = reinterpret_cast<__gm__ bfloat16_t *>(X_raw);
    //
    //   using RowShape  = Shape <1, 1, 1, 1, kDim>;
    //   using RowStride = Stride<1, 1, 1, kDim, 1>;
    //   using RowGlobal = GlobalTensor<bfloat16_t, RowShape, RowStride>;
    //   using RowTile   = Tile<TileType::Vec, bfloat16_t,
    //                          1, kDim,
    //                          BLayout::RowMajor,
    //                          1, kDim>;
    //   RowTile rowTile;
    //
    //   // Pass 1: histogram.
    //   int32_t count[kNRouted];
    //   for (e = 0; e < kNRouted; ++e) count[e] = 0;
    //   for (t = 0; t < kT; ++t)
    //     for (k = 0; k < kNActivated; ++k)
    //       count[expert_id[t * kNActivated + k]]++;
    //   for (e = 0; e < kNRouted; ++e) expert_count[e] = count[e];
    //
    //   // Pass 2: prefix sum.
    //   int32_t start[kNRouted];
    //   start[0] = 0;
    //   for (e = 1; e < kNRouted; ++e) start[e] = start[e-1] + count[e-1];
    //   for (e = 0; e < kNRouted; ++e) expert_start[e] = start[e];
    //
    //   // Pass 3: pack tokens.
    //   int32_t counter[kNRouted];
    //   for (e = 0; e < kNRouted; ++e) counter[e] = 0;
    //
    //   for (t = 0; t < kT; ++t) {
    //     for (k = 0; k < kNActivated; ++k) {
    //       pipe_barrier(PIPE_ALL);
    //       int32_t e          = expert_id[t * kNActivated + k];
    //       int32_t slot       = counter[e]++;
    //       int32_t packed_pos = start[e] + slot;
    //
    //       A_id   [packed_pos] = static_cast<int32_t>(t);
    //       rank_id[packed_pos] = static_cast<int32_t>(k);
    //
    //       size_t src_off = static_cast<size_t>(t)          * kDim;
    //       size_t dst_off = static_cast<size_t>(packed_pos) * kDim;
    //       RowGlobal srcGlobal(X + src_off);
    //       RowGlobal dstGlobal(A + dst_off);
    //       TLOAD (rowTile, srcGlobal);
    //       TSTORE(dstGlobal, rowTile);
    //     }
    //   }
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on MoE/scatter/scatter_kernel.cpp (already confirmed-built
    // for the half variant). The diff is only the row dtype.
    (void)A_raw; (void)A_id; (void)rank_id;
    (void)expert_count; (void)expert_start;
    (void)X_raw; (void)expert_id;
}

// Host launcher (kept minimal).
void launch_scatter(uint8_t *A, int32_t *A_id, int32_t *rank_id,
                    int32_t *expert_count, int32_t *expert_start,
                    uint8_t *X, int32_t *expert_id,
                    void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch scatter_kernel.
    (void)A; (void)A_id; (void)rank_id;
    (void)expert_count; (void)expert_start;
    (void)X; (void)expert_id; (void)stream;
}
