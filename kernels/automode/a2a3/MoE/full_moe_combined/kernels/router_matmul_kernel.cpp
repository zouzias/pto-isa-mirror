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
 * Limitations (v1):
 *   - Single AICORE; no block_idx work split.
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
constexpr unsigned kT     = 256;  // total tokens; tail rows are handled dynamically.

constexpr int kL1LoadBudgetBytes = 1 << 17;  // X panel + full-row W_router panel, fp16.
constexpr int kL0ABudgetBytes    = 64 * 1024;
constexpr int kL0BBudgetBytes    = 64 * 1024;
constexpr int kL0CBudgetBytes    = 128 * 1024;

constexpr int minInt(int lhs, int rhs)
{
    return lhs < rhs ? lhs : rhs;
}

constexpr int alignDownTo(int value, int align)
{
    return (value / align) * align;
}

constexpr int chooseDivisibleKBlock(int totalK, int maxK, int align)
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
}  // namespace router_matmul_cfg

// Two-level K tiling:
//   GM -> L1: load K_l1 panels whose X + full-row W_router footprint is <= 2^17 bytes.
//   L1 -> L0: TEXTRACT K_l0 slices into L0A/L0B, then accumulate into one L0C tile.
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

    static_assert(kH % blockAlign == 0, "router_matmul currently requires kH to be block-aligned.");
    static_assert(static_cast<size_t>(M) * N * sizeof(TOut) <= kL0CBudgetBytes,
                  "Accumulator tile exceeds L0C; reduce kTileM or kE.");

    constexpr int K_l1_max =
        kL1LoadBudgetBytes / (M * static_cast<int>(sizeof(TIn)) + N * static_cast<int>(sizeof(TWeight)));
    constexpr int K_l1_raw = chooseDivisibleKBlock(K, K_l1_max, blockAlign);
    static_assert(K_l1_raw >= blockAlign,
                  "No valid K_l1: X panel + full-row W_router panel exceeds the 2^17-byte L1 cap.");
    constexpr int K_l1 = (K_l1_raw >= blockAlign) ? K_l1_raw : blockAlign;
    static_assert((static_cast<size_t>(M) * K_l1 * sizeof(TIn) +
                   static_cast<size_t>(K_l1) * N * sizeof(TWeight)) <= kL1LoadBudgetBytes,
                  "L1 panel budget exceeded: X + full-row W_router must be <= 2^17 bytes.");

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

    // GlobalDataA: shape (currentM, K_l1), row stride kH (full row width in GM).
    // GlobalDataB: shape (K_l1, kE), stride kE; this always loads full rows of W_router.
    // GlobalDataC: shape (currentM, kE), stride kE per row (unpadded logit output).
    using GlobalShapeA = Shape<1, 1, 1, DYNAMIC, K_l1>;
    using GlobalShapeC = Shape<1, 1, 1, DYNAMIC, kE>;
    using GlobalDataA =
        GlobalTensor<TIn,     GlobalShapeA,
                     Stride<kTileM * kH, kTileM * kH, kTileM * kH, kH, 1>>;
    using GlobalDataB =
        GlobalTensor<TWeight, Shape<1, 1, 1, K_l1, kE>,
                     Stride<K_l1 * kE, K_l1 * kE, K_l1 * kE, kE, 1>>;
    using GlobalDataC =
        GlobalTensor<TOut,    GlobalShapeC,
                     Stride<kTileM * kE, kTileM * kE, kTileM * kE, kE, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, K_l1, BLayout::ColMajor,
                              DYNAMIC, K_l1, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K_l1, N, BLayout::ColMajor,
                              K_l1, kE,    SLayout::RowMajor, 512>;

    using LeftTile  = TileLeft <TIn,     M, K_l0, DYNAMIC, K_l0>;
    using RightTile = TileRight<TWeight, K_l0, N, K_l0, kE>;
    using AccTile   = TileAcc  <TOut,    M, N, DYNAMIC, kE>;  // valid kE cols -> logits

    TileMatAData aMatTile(kTileM);
    TileMatBData bMatTile;
    LeftTile     aTile(kTileM);
    RightTile    bTile;
    AccTile      cTile(kTileM);

    for (unsigned m0 = 0; m0 < kT; m0 += kTileM) {
        const unsigned currentM = ((m0 + kTileM) <= kT) ? kTileM : (kT - m0);
        aMatTile.SetValidRow(currentM);
        aTile.SetValidRow(currentM);
        cTile.SetValidRow(currentM);

        size_t cOff = static_cast<size_t>(m0) * kE;
        GlobalShapeC cShape(currentM);
        GlobalDataC cGlobal(logits + cOff, cShape);

        for (int k1 = 0; k1 < K_l1_blocks; ++k1) {
            size_t kBase = static_cast<size_t>(k1) * K_l1;
            size_t aOff = static_cast<size_t>(m0) * kH + kBase;
            size_t bOff = kBase * kE;

            GlobalShapeA aShape(currentM);
            GlobalDataA aGlobal(x        + aOff, aShape);
            GlobalDataB bGlobal(w_router + bOff);

            TLOAD(aMatTile, aGlobal);
            TLOAD(bMatTile, bGlobal);

            for (int k0 = 0; k0 < K_l0_segments; ++k0) {
                const uint16_t kOff = static_cast<uint16_t>(k0 * K_l0);
                TEXTRACT(aTile, aMatTile, 0, kOff);
                TEXTRACT(bTile, bMatTile, kOff, 0);
                if (k1 == 0 && k0 == 0) {
                    TMATMUL(cTile, aTile, bTile);
                } else {
                    TMATMUL_ACC(cTile, aTile, bTile);
                }
            }
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
