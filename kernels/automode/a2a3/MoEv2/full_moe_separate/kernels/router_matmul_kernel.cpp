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

    using GlobalDataA =
        GlobalTensor<TIn,     Shape<1, 1, 1, kTileM, kH>,
                     Stride<kTileM * kH, kTileM * kH, kTileM * kH, kH, 1>>;
    using GlobalDataB =
        GlobalTensor<TWeight, Shape<1, 1, 1, kH, kE>,
                     Stride<kH * kE,     kH * kE,     kH * kE,     kE, 1>>;
    using GlobalDataC =
        GlobalTensor<TOut,    Shape<1, 1, 1, kTileM, kE>,
                     Stride<kTileM * kE, kTileM * kE, kTileM * kE, kE, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, K, BLayout::ColMajor,
                              kTileM, kH, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K, N, BLayout::ColMajor,
                              kH,     kE, SLayout::RowMajor, 512>;

    using LeftTile  = TileLeft <TIn,     M, K, kTileM, kH>;
    using RightTile = TileRight<TWeight, K, N, kH,     kE>;
    using AccTile   = TileAcc  <TOut,    M, N, kTileM, kE>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    // W_router is the same for all token tiles; view constructed once.
    GlobalDataB bGlobal(w_router);

    for (unsigned m0 = 0; m0 < kT; m0 += kTileM) {
        size_t aOff = static_cast<size_t>(m0) * kH;
        size_t cOff = static_cast<size_t>(m0) * kE;

        GlobalDataA aGlobal(x + aOff);
        GlobalDataC cGlobal(logits + cOff);

        TLOAD(aMatTile, aGlobal);
        TLOAD(bMatTile, bGlobal);
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);
        TMATMUL(cTile, aTile, bTile);
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
