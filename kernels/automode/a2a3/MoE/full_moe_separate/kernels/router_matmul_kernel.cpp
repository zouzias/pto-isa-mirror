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
 *       logits[m0:m0+kTileM, :] = X[m0:m0+kTileM, :] @ W_router
 *
 * Limitations (v1):
 *   - Single AICORE; no block_idx work split.
 *   - kT must be a multiple of kTileM.
 *   - No ReLU / softmax (raw logits only).
 *   - No double-buffering, no TPipe / TPUSH / TPOP.
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace router_matmul_cfg {
constexpr unsigned kH     = 64;   // d_model (input hidden dim)
constexpr unsigned kE     = 32;   // num_experts
constexpr unsigned kTileM = 128;  // token tile height (cube M dimension)
constexpr unsigned kT     = 256;  // total tokens (must be kTileM-aligned)
}  // namespace router_matmul_cfg

// Split-K over kH: partition the contraction dimension into K_block-sized
// chunks constrained by both L0A (LeftTile M*K_block) and L0B (RightTile
// K_block*N).  Accumulate cTile across K_nblocks, then TSTORE once.
// When K ≤ K_block_max, K_nblocks=1 — identical to the old single-tile path.
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
    constexpr int M = ((kTileM + 15) / 16) * 16;
    constexpr int K = ((kH + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kE + blockAlign - 1) / blockAlign) * blockAlign;

    // Split-K: constrain K_block so LeftTile M*K_block ≤ L0A and RightTile K_block*N ≤ L0B.
    constexpr int kL0A_fp16       = 65536;
    constexpr int kL0B_fp16       = 32768;
    constexpr int K_block_max_L0A = ((kL0A_fp16 / M) / blockAlign) * blockAlign;
    constexpr int K_block_max_L0B = ((kL0B_fp16 / N) / blockAlign) * blockAlign;
    constexpr int K_block_max     = (K_block_max_L0A < K_block_max_L0B)
                                    ? K_block_max_L0A : K_block_max_L0B;
    constexpr int K_block   = (K <= K_block_max) ? K : K_block_max;
    constexpr int K_nblocks = K / K_block;
    static_assert(K_block >= blockAlign,
        "K_block < blockAlign: kH too large for L0A/L0B at this M/N; reduce kH or kTileM.");
    static_assert(K % K_block == 0,
        "K not divisible by K_block: adjust kH so kH_aligned % K_block == 0.");

    // GlobalDataA: shape (kTileM, K_block), row stride K (full row width in GM).
    // GlobalDataB: shape (K_block, N), stride N per row within the slice.
    // GlobalDataC: shape (kTileM, kE), stride kE per row (unpadded logit output).
    using GlobalDataA =
        GlobalTensor<TIn,     Shape<1, 1, 1, kTileM, K_block>,
                     Stride<kTileM * K, kTileM * K, kTileM * K, K, 1>>;
    using GlobalDataB =
        GlobalTensor<TWeight, Shape<1, 1, 1, K_block, N>,
                     Stride<K_block * N, K_block * N, K_block * N, N, 1>>;
    using GlobalDataC =
        GlobalTensor<TOut,    Shape<1, 1, 1, kTileM, kE>,
                     Stride<kTileM * kE, kTileM * kE, kTileM * kE, kE, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, K_block, BLayout::ColMajor,
                              kTileM, K_block, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K_block, N, BLayout::ColMajor,
                              K_block, N,     SLayout::RowMajor, 512>;

    using LeftTile  = TileLeft <TIn,     M, K_block, kTileM, K_block>;
    using RightTile = TileRight<TWeight, K_block, N, K_block, N>;
    using AccTile   = TileAcc  <TOut,    M, N, kTileM, kE>;  // valid kE cols -> logits

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    for (unsigned m0 = 0; m0 < kT; m0 += kTileM) {
        size_t cOff = static_cast<size_t>(m0) * kE;
        GlobalDataC cGlobal(logits + cOff);

        for (int kb = 0; kb < K_nblocks; ++kb) {
            size_t aOff = static_cast<size_t>(m0) * K + static_cast<size_t>(kb) * K_block;
            size_t bOff = static_cast<size_t>(kb) * K_block * N;

            GlobalDataA aGlobal(x        + aOff);
            GlobalDataB bGlobal(w_router + bOff);

            TLOAD(aMatTile, aGlobal);
            TLOAD(bMatTile, bGlobal);
            TMOV(aTile, aMatTile);
            TMOV(bTile, bMatTile);
            TMATMUL(cTile, aTile, bTile);
        }
        TSTORE(cGlobal, cTile);
    }
}

template <typename TOut, typename TIn, typename TWeight>
void launchRouterMatmul(uint8_t *logits, uint8_t *x, uint8_t *w_router, void *stream)
{
    runRouterMatmul<TOut, TIn, TWeight><<<1, nullptr, stream>>>(logits, x, w_router);
}

template void launchRouterMatmul<float, half, half>(
    uint8_t *logits, uint8_t *x, uint8_t *w_router, void *stream);

// Non-template wrapper: host TU compiled with -xc++ lacks `half`.
// See compile_error_logbook.md §E13.
extern "C" void launchRouterMatmulFp16(uint8_t *logits, uint8_t *x, uint8_t *w_router, void *stream)
{
    launchRouterMatmul<float, half, half>(logits, x, w_router, stream);
}
