/**
 * router_matmul_kernel.cpp - auto-mode A3 prototype.
 *
 * Computes MoE router logits:  logits = X @ W_router
 * where X is (kT, kH) float16, W_router is (kH, kE) float16,
 * and logits is (kT, kE) float32 (FP32 accumulator).
 *
 * Pipeline: token-tiled GEMM, kTileM rows per cube tile.
 * Pattern: moe_segmented_gemm_relu with the expert outer loop removed.
 * W_router is shared for every token tile; bGlobal view is constructed
 * once before the m0 loop and reloaded every iteration
 * (same as moe_segmented_gemm_relu's per-m0 bMatTile reload, confirmed-working).
 *
 * Semantics:
 *   for m0 in range(0, kT, kTileM):
 *       logits[m0:min(m0+kTileM,kT), :] =
 *           X[m0:min(m0+kTileM,kT), :] @ W_router
 *
 * Parallelism (v2):
 *   - Multi-AICORE SPMD via `block_idx`, partitioned on the M (token) axis.
 *     The M-tile index space [0, kNumMTiles) is split into contiguous,
 *     even-sized ranges using `bid * kNumMTiles / kBlockDim`. Each core writes
 *     a disjoint row range of `logits`, so no cross-core sync is required.
 *   - kBlockDim = min(kNumMTiles, kMaxCubeCores) is chosen at compile time,
 *     so when kT yields fewer M-tiles than cube cores we launch only as many
 *     cores as there is work for (no idle cores).
 *   - The N (expert) axis is NOT split: kE is typically much smaller than kT
 *     and a full row of W_router already fits in the working-set budget.
 *   - K accumulation stays on one core per M-tile (no Split-K across cores).
 *
 * Host/AICORE separation: the launcher needs the same M / kBlockDim values
 * the kernel computes, but the kernel-side helpers carry AICORE qualifiers
 * and are not host-callable. router_matmul_host_cfg holds a duplicate of the
 * formulas with no AICORE attribute for use by launchRouterMatmul.
 *
 * Pipeline (v3 — MultiBuffered):
 *   - GM->L1 panel loads (aMatTile/bMatTile) ping-pong across the outer K loop
 *     via `MultiBuffered<2>::loop<Range<K_l1_blocks>>`.
 *   - L1->L0 extracts (aTile/bTile) ping-pong across the inner K loop via
 *     `MultiBuffered<2>::loop<Range<K_l0_segments>>`.
 *   - cTile (L0C accumulator) stays single-buffered; it must persist across
 *     all (k1, k0) iterations until TSTORE.
 *   - No TPipe / TPUSH / TPOP; the v_loop_barrier pragmas are emitted by the
 *     MultiBuffered helper, not by hand.
 *
 * Limitations (v3):
 *   - No ReLU / softmax (raw logits only).
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "generated_cases.h"
#include "multiBuffer.hpp"

using namespace pto;
using namespace pto_auto;

// Number of pipeline lanes for the L1 (GM->L1) and L0 (L1->L0) buffered loops.
// 2 = ping-pong; the MultiBuffered helper emits a #pragma pto v_loop_barrier
// between lanes so the auto-mode compiler allocates one buffer per lane.
constexpr int kNumBuffers = 2;

namespace router_matmul_cfg {
constexpr unsigned kH     = kMoeH;  // d_model (input hidden dim)
constexpr unsigned kE     = kMoeE;  // num_experts
constexpr unsigned kTileM = 128;    // max token tile height (cube M dimension) - kernel-internal
constexpr unsigned kT     = kMoeT;  // total tokens; tail rows are handled dynamically.

constexpr int kWorkingSetBudgetBytes = 1 << 19;  // X panel + full-row W_router panel + fp32 logits tile.
constexpr int kL0ABudgetBytes    = 64 * 1024;
constexpr int kL0BBudgetBytes    = 64 * 1024;
constexpr int kL0CBudgetBytes    = 128 * 1024;

// A3 cube core count. Matches kernels/manual/a2a3/gemm_performance L246
// (`constexpr uint32_t blockDim = 24;`).
constexpr unsigned kMaxCubeCores = 24;

AICORE inline constexpr int minInt(int lhs, int rhs)
{
    return lhs < rhs ? lhs : rhs;
}

AICORE inline constexpr int alignDownTo(int value, int align)
{
    return (value / align) * align;
}

AICORE inline constexpr int chooseDivisibleKBlock(int totalK, int maxK, int align)
{
    int block = alignDownTo(minInt(totalK, maxK), align);
    while (block >= align) {
        if (totalK % block == 0) {
            return block;
        }
        block -= align;
    }
    return 0;
}

AICORE inline constexpr int chooseMBlock(int maxM, int n, int kMin, int inBytes, int weightBytes, int outBytes,
                                         int budget, int align)
{
    const int minWeightBytes = kMin * n * weightBytes;
    if (minWeightBytes >= budget) {
        return 0;
    }
    const int maxByBudget = (budget - minWeightBytes) / (kMin * inBytes + n * outBytes);
    const int block = alignDownTo(minInt(maxM, maxByBudget), align);
    return block >= align ? block : 0;
}

AICORE inline constexpr int chooseKPanel(int totalK, int m, int n, int inBytes, int weightBytes, int outBytes,
                                         int budget, int align)
{
    const int cBytes = m * n * outBytes;
    if (cBytes >= budget) {
        return 0;
    }
    const int maxK = (budget - cBytes) / (m * inBytes + n * weightBytes);
    return chooseDivisibleKBlock(totalK, maxK, align);
}
}  // namespace router_matmul_cfg

// Host-callable mirror of the M / blockDim math. Identical formulas, no
// AICORE attribute, so launchRouterMatmul can use them. Kept structurally
// 1:1 with router_matmul_cfg so updates stay in lock-step.
namespace router_matmul_host_cfg {
inline constexpr int minInt(int lhs, int rhs)
{
    return lhs < rhs ? lhs : rhs;
}

inline constexpr int alignDownTo(int value, int align)
{
    return (value / align) * align;
}

inline constexpr int chooseMBlock(int maxM, int n, int kMin, int inBytes, int weightBytes, int outBytes,
                                  int budget, int align)
{
    const int minWeightBytes = kMin * n * weightBytes;
    if (minWeightBytes >= budget) {
        return 0;
    }
    const int maxByBudget = (budget - minWeightBytes) / (kMin * inBytes + n * outBytes);
    const int block = alignDownTo(minInt(maxM, maxByBudget), align);
    return block >= align ? block : 0;
}
}  // namespace router_matmul_host_cfg

// Two-level K tiling:
//   GM -> L1: load K_l1 panels while X + full-row W_router + logits tile is <= 2^16 bytes.
//   L1 -> L0: TEXTRACT K_l0 slices into L0A/L0B, then accumulate into one L0C tile.
//   GM store: write each M-tile's logits to GM before moving to the next M tile.
template <typename TOut, typename TIn, typename TWeight>
__global__ AICORE void runRouterMatmul(
    __gm__ uint8_t *logits_raw,
    __gm__ uint8_t *x_raw,
    __gm__ uint8_t *w_router_raw)
{
    using namespace router_matmul_cfg;

    __gm__ TOut    *logits   = reinterpret_cast<__gm__ TOut    *>(logits_raw);
    __gm__ TIn     *x        = reinterpret_cast<__gm__ TIn     *>(x_raw);
    __gm__ TWeight *w_router = reinterpret_cast<__gm__ TWeight *>(w_router_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int mAlign = 16;
    constexpr int M_max = ((kTileM + mAlign - 1) / mAlign) * mAlign;
    constexpr int K = ((kH + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kE + blockAlign - 1) / blockAlign) * blockAlign;

    static_assert(kH % blockAlign == 0, "router_matmul currently requires kH to be block-aligned.");

    constexpr int M_raw = chooseMBlock(M_max, N, blockAlign, static_cast<int>(sizeof(TIn)),
                                       static_cast<int>(sizeof(TWeight)), static_cast<int>(sizeof(TOut)),
                                       kWorkingSetBudgetBytes, mAlign);
    static_assert(M_raw >= mAlign,
                  "No valid M tile: full-row W_router plus one aligned X/logits tile exceeds the budget.");
    constexpr int M = (M_raw >= mAlign) ? M_raw : mAlign;
    static_assert(static_cast<size_t>(M) * N * sizeof(TOut) <= kL0CBudgetBytes,
                  "Accumulator tile exceeds L0C; reduce kTileM or kE.");

    constexpr int K_l1_raw =
        chooseKPanel(K, M, N, static_cast<int>(sizeof(TIn)), static_cast<int>(sizeof(TWeight)),
                     static_cast<int>(sizeof(TOut)), kWorkingSetBudgetBytes, blockAlign);
    static_assert(K_l1_raw >= blockAlign,
                  "No valid K_l1: X panel + full-row W_router panel + logits tile exceeds the budget.");
    constexpr int K_l1 = (K_l1_raw >= blockAlign) ? K_l1_raw : blockAlign;
    static_assert((static_cast<size_t>(M) * K_l1 * sizeof(TIn) +
                   static_cast<size_t>(K_l1) * N * sizeof(TWeight) +
                   static_cast<size_t>(M) * N * sizeof(TOut)) <= kWorkingSetBudgetBytes,
                  "Working-set budget exceeded: X + full-row W_router + logits must fit in budget.");

    constexpr int K_l0_max_L0A = kL0ABudgetBytes / (M * static_cast<int>(sizeof(TIn)));
    constexpr int K_l0_max_L0B = kL0BBudgetBytes / (N * static_cast<int>(sizeof(TWeight)));
    constexpr int K_l0_max = minInt(K_l0_max_L0A, K_l0_max_L0B);
    constexpr int K_l0_raw = chooseDivisibleKBlock(K_l1, K_l0_max, blockAlign);
    static_assert(K_l0_raw >= blockAlign,
                  "No valid K_l0: L0A/L0B cannot hold the minimum aligned K slice.");
    constexpr int K_l0 = (K_l0_raw >= blockAlign) ? K_l0_raw : blockAlign;
    static_assert(K_l1 % K_l0 == 0, "K_l1 must be divisible by K_l0 for L1-to-L0 extraction.");
    constexpr int K_l1_blocks = K / K_l1;
    constexpr int K_l0_segments = K_l1 / K_l0;

    // SPMD M-axis split. kNumMTiles covers all kT rows; kBlockDim is capped at
    // kMaxCubeCores so when kT is small we don't request idle cores. Each core
    // computes a contiguous slice [tStart, tEnd) of the M-tile index space.
    constexpr unsigned kNumMTiles = (kT + static_cast<unsigned>(M) - 1) / static_cast<unsigned>(M);
    constexpr unsigned kBlockDim  = (kNumMTiles < kMaxCubeCores) ? kNumMTiles : kMaxCubeCores;
    static_assert(kBlockDim >= 1, "kBlockDim must be at least 1 (kT must be > 0).");

    // GlobalDataA: shape (currentM, K_l1), row stride kH (full row width in GM).
    // GlobalDataB: shape (K_l1, kE), stride kE; this always loads full rows of W_router.
    // GlobalDataC: shape (currentM, kE), stride kE per row (unpadded logit output).
    using GlobalShapeA = Shape<1, 1, 1, DYNAMIC, K_l1>;
    using GlobalShapeC = Shape<1, 1, 1, DYNAMIC, kE>;
    using GlobalDataA =
        GlobalTensor<TIn,     GlobalShapeA,
                     Stride<M * kH, M * kH, M * kH, kH, 1>>;
    using GlobalDataB =
        GlobalTensor<TWeight, Shape<1, 1, 1, K_l1, kE>,
                     Stride<K_l1 * kE, K_l1 * kE, K_l1 * kE, kE, 1>>;
    using GlobalDataC =
        GlobalTensor<TOut,    GlobalShapeC,
                     Stride<M * kE, M * kE, M * kE, kE, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, K_l1, BLayout::ColMajor,
                              DYNAMIC, K_l1, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K_l1, N, BLayout::ColMajor,
                              K_l1, kE,    SLayout::RowMajor, 512>;

    using LeftTile  = TileLeft <TIn,     M, K_l0, DYNAMIC, K_l0>;
    using RightTile = TileRight<TWeight, K_l0, N, K_l0, kE>;
    using AccTile   = TileAcc  <TOut,    M, N, DYNAMIC, kE>;  // valid kE cols -> logits

    // cTile is the L0C accumulator and must persist across every (k1, k0)
    // iteration of the K reduction, so it stays as a single tile at function
    // scope (NOT multi-buffered). aMatTile/bMatTile (L1) and aTile/bTile (L0A/B)
    // are declared inside the MultiBuffered lambdas below so each lane gets
    // its own buffer.
    AccTile cTile(M);

    MultiBuffered<kNumBuffers> outer_db;

    // Even integer-split of [0, kNumMTiles) across kBlockDim cores. Pattern:
    // each core's range = [bid*N/D, (bid+1)*N/D); remainders fall in the last
    // few cores. Matches gemm_performance / MoE-manual-opt router_matmul.
    const unsigned bid    = static_cast<unsigned>(get_block_idx());
    const unsigned tStart = (bid * kNumMTiles) / kBlockDim;
    const unsigned tEnd   = ((bid + 1) * kNumMTiles) / kBlockDim;

    for (unsigned tIdx = tStart; tIdx < tEnd; ++tIdx) {
        const unsigned m0 = tIdx * static_cast<unsigned>(M);
        const unsigned currentM = ((m0 + M) <= kT) ? M : (kT - m0);
        cTile.SetValidRow(currentM);

        size_t cOff = static_cast<size_t>(m0) * kE;
        GlobalShapeC cShape(currentM);
        GlobalDataC cGlobal(logits + cOff, cShape);

        // Outer K-panel loop: GM -> L1 ping-pong. Each unrolled lane owns its
        // own aMatTile / bMatTile in L1, so MTE2 (TLOAD) of lane N+1 can
        // overlap the cube work on lane N's panel.
        outer_db.loop<Range<K_l1_blocks>>([&](auto outerCtx) {
            const int k1 = outerCtx.iter;
            size_t kBase = static_cast<size_t>(k1) * K_l1;
            size_t aOff = static_cast<size_t>(m0) * kH + kBase;
            size_t bOff = kBase * kE;

            TileMatAData aMatTile(M);
            TileMatBData bMatTile;
            aMatTile.SetValidRow(currentM);

            GlobalShapeA aShape(currentM);
            GlobalDataA aGlobal(x        + aOff, aShape);
            GlobalDataB bGlobal(w_router + bOff);

            TLOAD(aMatTile, aGlobal);
            TLOAD(bMatTile, bGlobal);

            // Inner K-segment loop: L1 -> L0A/L0B ping-pong. Each unrolled
            // lane owns its own aTile / bTile in L0, so MTE1 (TEXTRACT) of
            // segment N+1 can overlap the TMATMUL on segment N.
            MultiBuffered<kNumBuffers> inner_db;
            inner_db.loop<Range<K_l0_segments>>([&](auto innerCtx) {
                const int k0 = innerCtx.iter;
                const uint16_t kOff = static_cast<uint16_t>(k0 * K_l0);

                LeftTile  aTile(M);
                RightTile bTile;
                aTile.SetValidRow(currentM);

                TEXTRACT(aTile, aMatTile, 0, kOff);
                TEXTRACT(bTile, bMatTile, kOff, 0);
                if (k1 == 0 && k0 == 0) {
                    TMATMUL(cTile, aTile, bTile);
                } else {
                    TMATMUL_ACC(cTile, aTile, bTile);
                }
            });
        });

        TSTORE(cGlobal, cTile);
    }
}

template <typename TOut, typename TIn, typename TWeight>
void launchRouterMatmul(uint8_t *logits, uint8_t *x, uint8_t *w_router, void *stream)
{
    // Recompute M / kNumMTiles / kBlockDim with the host-side mirror of the
    // kernel's formulas. Cannot call the AICORE-qualified helpers from host.
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int mAlign = 16;
    constexpr int M_max = ((router_matmul_cfg::kTileM + mAlign - 1) / mAlign) * mAlign;
    constexpr int N = ((router_matmul_cfg::kE + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int M_raw = router_matmul_host_cfg::chooseMBlock(
        M_max, N, blockAlign,
        static_cast<int>(sizeof(TIn)),
        static_cast<int>(sizeof(TWeight)),
        static_cast<int>(sizeof(TOut)),
        router_matmul_cfg::kWorkingSetBudgetBytes, mAlign);
    constexpr int M = (M_raw >= mAlign) ? M_raw : mAlign;
    constexpr unsigned kNumMTiles =
        (router_matmul_cfg::kT + static_cast<unsigned>(M) - 1) / static_cast<unsigned>(M);
    constexpr unsigned kBlockDim =
        (kNumMTiles < router_matmul_cfg::kMaxCubeCores) ? kNumMTiles : router_matmul_cfg::kMaxCubeCores;

    runRouterMatmul<TOut, TIn, TWeight><<<kBlockDim, nullptr, stream>>>(logits, x, w_router);
}

template void launchRouterMatmul<float, half, half>(
    uint8_t *logits, uint8_t *x, uint8_t *w_router, void *stream);

// Non-template wrapper: host TU compiled with -xc++ lacks `half`.
// See compile_error_logbook.md §E13.
extern "C" void launchRouterMatmulFp16(uint8_t *logits, uint8_t *x, uint8_t *w_router, void *stream)
{
    launchRouterMatmul<float, half, half>(logits, x, w_router, stream);
}
