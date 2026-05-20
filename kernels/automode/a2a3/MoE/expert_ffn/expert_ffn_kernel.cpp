/**
 * expert_ffn_kernel.cpp - auto-mode A3 prototype.
 *
 * Per-expert fused FFN over packed-by-expert tokens:
 *
 *   B_s[:, n:n+N_l1] += relu(A_s @ W1[e, h:h+H_l1, f:f+F_l1])
 *                       @ W2[e, f:f+F_l1, n:n+N_l1]
 *
 * for each expert-local row tile A_s.  The first GEMM materializes the
 * post-ReLU fp16 intermediate in L1 with TMOV Acc->Mat + NormalRelu, then the
 * second GEMM consumes that tile immediately.  Each B_s column panel is kept
 * in L0C across F_l1 panels and stored to GM before moving to the next panel.
 *
 * Working-set cap:
 *   A_s + W1_t + W2_t + relu(A_s @ W1_t) + B_s <= 2^17 bytes
 *
 * B_s is physically the fp32 L0C accumulator until TSTORE, but its footprint is
 * included in the cap so the tile choice remains conservative for local memory.
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace expert_ffn_cfg {

// v1 shape - must match scripts/gen_data.py and main.cpp.
constexpr unsigned kT     = 256;
constexpr unsigned kH     = 64;  // GEMM1 K = GEMM2 N
constexpr unsigned kF     = 64;  // GEMM1 N = GEMM2 K
constexpr unsigned kE     = 32;
constexpr unsigned kTopK  = 1;
constexpr unsigned kTileM = 16;  // max expert-local token tile height

constexpr unsigned kPackedRows = kT * kTopK;

constexpr int kWorkingSetBudgetBytes = 1 << 17;
constexpr int kL0BudgetBytes         = 1 << 14;

AICORE inline constexpr int minInt(int lhs, int rhs)
{
    return lhs < rhs ? lhs : rhs;
}

AICORE inline constexpr int alignDownTo(int value, int align)
{
    return (value / align) * align;
}

AICORE inline constexpr int chooseDivisibleBlock(int total, int maxBlock, int align)
{
    int block = alignDownTo(minInt(total, maxBlock), align);
    while (block >= align) {
        if (total % block == 0) {
            return block;
        }
        block -= align;
    }
    return 0;
}

AICORE inline constexpr int chooseNPanel(int totalN, int m, int fMin, int accBytes, int outBytes, int l0cBudget,
                                         int align)
{
    const int yAccBytes = m * fMin * accBytes;
    if (yAccBytes >= l0cBudget) {
        return 0;
    }
    const int maxN = (l0cBudget - yAccBytes) / (m * outBytes);
    return chooseDivisibleBlock(totalN, maxN, align);
}

AICORE inline constexpr int chooseFPanel(int totalF, int m, int hMin, int n, int inBytes, int weightBytes,
                                         int scratchBytes, int outBytes, int accBytes, int budget, int l0cBudget,
                                         int align)
{
    const int bAccBytes = m * n * outBytes;
    if (bAccBytes >= l0cBudget) {
        return 0;
    }
    const int maxByL0C = (l0cBudget - bAccBytes) / (m * accBytes);

    const int fixedBytes = m * hMin * inBytes + m * n * outBytes;
    const int bytesPerF = hMin * weightBytes + n * weightBytes + m * scratchBytes;
    if (fixedBytes >= budget) {
        return 0;
    }
    const int maxByBudget = (budget - fixedBytes) / bytesPerF;
    return chooseDivisibleBlock(totalF, minInt(maxByBudget, maxByL0C), align);
}

AICORE inline constexpr int chooseHPanel(int totalH, int m, int f, int n, int inBytes, int weightBytes,
                                         int scratchBytes, int outBytes, int budget, int align)
{
    const int fixedBytes = f * n * weightBytes + m * f * scratchBytes + m * n * outBytes;
    if (fixedBytes >= budget) {
        return 0;
    }
    const int bytesPerH = m * inBytes + f * weightBytes;
    const int maxH = (budget - fixedBytes) / bytesPerH;
    return chooseDivisibleBlock(totalH, maxH, align);
}

AICORE inline constexpr int chooseH0Panel(int hL1, int m, int fL1, int fL0, int nL1, int inBytes, int weightBytes,
                                          int scratchBytes, int l0Budget, int align)
{
    const int yBytes = m * fL0 * scratchBytes;
    const int w2Bytes = fL0 * nL1 * weightBytes;
    if (yBytes >= l0Budget || w2Bytes >= l0Budget) {
        return 0;
    }
    const int maxByL0A = (l0Budget - yBytes) / (m * inBytes);
    const int maxByL0B = (l0Budget - w2Bytes) / (fL1 * weightBytes);
    return chooseDivisibleBlock(hL1, minInt(maxByL0A, maxByL0B), align);
}

}  // namespace expert_ffn_cfg

// ----------------------------------------------------------------------------
// Fused expert FFN.
//
// GM -> L1:
//   A_s  : [currentM, H_l1]    H panel of full A rows for one expert only
//   W1_t : [H_l1, F_l1]        matching H rows, F_l1 columns of W1
//   W2_t : [F_l1, N_l1]        matching F_l1 rows, N_l1 columns of W2
//   Y_t  : [currentM, F_l1]    fp16 ReLU intermediate for one F_l1 panel
//
// L1 -> L0:
//   Stage 1 splits over H_l0 and accumulates Y_t in L0C.
//   Stage 2 splits over F_l0 and accumulates B_s in L0C.
// ----------------------------------------------------------------------------
template <typename TOut, typename TIn, typename TWeight, typename TScratch>
__global__ AICORE void runExpertFfn(
    __gm__ uint8_t *B_raw,
    __gm__ uint8_t *A_raw,
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ uint8_t *W1_raw,
    __gm__ uint8_t *W2_raw)
{
    using namespace expert_ffn_cfg;

    __gm__ TOut    *B  = reinterpret_cast<__gm__ TOut    *>(B_raw);
    __gm__ TIn     *A  = reinterpret_cast<__gm__ TIn     *>(A_raw);
    __gm__ TWeight *W1 = reinterpret_cast<__gm__ TWeight *>(W1_raw);
    __gm__ TWeight *W2 = reinterpret_cast<__gm__ TWeight *>(W2_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int mAlign = 16;
    constexpr int M = ((kTileM + mAlign - 1) / mAlign) * mAlign;
    constexpr int H = ((kH + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int F = ((kF + blockAlign - 1) / blockAlign) * blockAlign;

    constexpr int N_l1_raw =
        chooseNPanel(H, M, blockAlign, sizeof(float), static_cast<int>(sizeof(TOut)), kL0BudgetBytes, blockAlign);
    static_assert(N_l1_raw >= blockAlign,
                  "No valid N_l1 tile: minimal Y accumulator plus B panel exceeds the 2^14-byte L0C cap.");
    constexpr int N_l1 = (N_l1_raw >= blockAlign) ? N_l1_raw : blockAlign;
    static_assert(H % N_l1 == 0, "H must be divisible by N_l1.");

    constexpr int F_l1_raw = chooseFPanel(
        F, M, blockAlign, N_l1, static_cast<int>(sizeof(TIn)), static_cast<int>(sizeof(TWeight)),
        static_cast<int>(sizeof(TScratch)), static_cast<int>(sizeof(TOut)), sizeof(float), kWorkingSetBudgetBytes,
        kL0BudgetBytes, blockAlign);
    static_assert(F_l1_raw >= blockAlign,
                  "No valid F_l1 tile: A_s + W1_t + W2_t + Y_t + B_s exceeds the 2^17-byte cap.");
    constexpr int F_l1 = (F_l1_raw >= blockAlign) ? F_l1_raw : blockAlign;

    constexpr int H_l1_raw = chooseHPanel(
        H, M, F_l1, N_l1, static_cast<int>(sizeof(TIn)), static_cast<int>(sizeof(TWeight)),
        static_cast<int>(sizeof(TScratch)), static_cast<int>(sizeof(TOut)), kWorkingSetBudgetBytes, blockAlign);
    static_assert(H_l1_raw >= blockAlign,
                  "No valid H_l1 tile: minimal A_s/W1_t/W2_t/Y_t/B_s exceeds the 2^17-byte cap.");
    constexpr int H_l1 = (H_l1_raw >= blockAlign) ? H_l1_raw : blockAlign;

    constexpr int F_l0 = blockAlign;
    static_assert(F_l1 % F_l0 == 0, "F_l1 must be divisible by F_l0.");

    constexpr int H_l0_raw =
        chooseH0Panel(H_l1, M, F_l1, F_l0, N_l1, static_cast<int>(sizeof(TIn)),
                      static_cast<int>(sizeof(TWeight)), static_cast<int>(sizeof(TScratch)), kL0BudgetBytes,
                      blockAlign);
    static_assert(H_l0_raw >= blockAlign,
                  "No valid H_l0 tile: L0A/L0B cannot hold the minimum W1 contraction slice.");
    constexpr int H_l0 = (H_l0_raw >= blockAlign) ? H_l0_raw : blockAlign;
    static_assert(H % H_l1 == 0, "H must be divisible by H_l1.");
    static_assert(H_l1 % H_l0 == 0, "H_l1 must be divisible by H_l0.");

    static_assert((static_cast<size_t>(M) * H_l1 * sizeof(TIn) +
                   static_cast<size_t>(H_l1) * F_l1 * sizeof(TWeight) +
                   static_cast<size_t>(F_l1) * N_l1 * sizeof(TWeight) +
                   static_cast<size_t>(M) * F_l1 * sizeof(TScratch) +
                   static_cast<size_t>(M) * N_l1 * sizeof(TOut)) <= kWorkingSetBudgetBytes,
                  "Working-set budget exceeded: A_s + W1_t + W2_t + Y_t + B_s must be <= 2^17 bytes.");
    static_assert((static_cast<size_t>(M) * F_l1 * sizeof(float) +
                   static_cast<size_t>(M) * N_l1 * sizeof(TOut)) <= kL0BudgetBytes,
                  "Combined Y/B accumulators exceed L0C.");
    static_assert((static_cast<size_t>(M) * H_l0 * sizeof(TIn) +
                   static_cast<size_t>(M) * F_l0 * sizeof(TScratch)) <= kL0BudgetBytes,
                  "Combined A/Y L0A tiles exceed L0A.");
    static_assert((static_cast<size_t>(H_l0) * F_l1 * sizeof(TWeight) +
                   static_cast<size_t>(F_l0) * N_l1 * sizeof(TWeight)) <= kL0BudgetBytes,
                  "Combined W1/W2 L0B tiles exceed L0B.");

    constexpr int F_l1_blocks = F / F_l1;
    constexpr int H_l1_blocks = H / H_l1;
    constexpr int H_l0_segments = H_l1 / H_l0;
    constexpr int F_l0_segments = F_l1 / F_l0;

    using GlobalShapeA = Shape<1, 1, 1, DYNAMIC, H_l1>;
    using GlobalShapeW2 = Shape<1, 1, 1, F_l1, DYNAMIC>;
    using GlobalShapeB = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using GlobalDataA =
        GlobalTensor<TIn,     GlobalShapeA,
                     Stride<M * H, M * H, M * H, H, 1>>;
    using GlobalDataW1 =
        GlobalTensor<TWeight, Shape<1, 1, 1, H_l1, F_l1>,
                     Stride<H_l1 * F, H_l1 * F, H_l1 * F, F, 1>>;
    using GlobalDataW2 =
        GlobalTensor<TWeight, GlobalShapeW2,
                     Stride<F_l1 * H, F_l1 * H, F_l1 * H, H, 1>>;
    using GlobalDataB =
        GlobalTensor<TOut,    GlobalShapeB,
                     Stride<M * kH, M * kH, M * kH, kH, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, H_l1, BLayout::ColMajor,
                              DYNAMIC, H_l1, SLayout::RowMajor, 512>;
    using TileMatW1Data = Tile<TileType::Mat, TWeight, H_l1, F_l1, BLayout::ColMajor,
                               H_l1,    F_l1, SLayout::RowMajor, 512>;
    using TileMatW2Data = Tile<TileType::Mat, TWeight, F_l1, N_l1, BLayout::ColMajor,
                               F_l1,    DYNAMIC, SLayout::RowMajor, 512>;
    using TileMatYData = Tile<TileType::Mat, TScratch, M, F_l1, BLayout::ColMajor,
                              DYNAMIC, F_l1, SLayout::RowMajor, 512>;

    using ALeftTile = TileLeft <TIn,     M,    H_l0, DYNAMIC, H_l0>;
    using W1RightTile = TileRight<TWeight, H_l0, F_l1, H_l0, F_l1>;
    using YAccTile = TileAcc<float,      M,    F_l1, DYNAMIC, F_l1>;

    using YLeftTile = TileLeft <TScratch, M,    F_l0, DYNAMIC, F_l0>;
    using W2RightTile = TileRight<TWeight, F_l0, N_l1, F_l0,   DYNAMIC>;
    using BAccTile = TileAcc<TOut,       M,    N_l1, DYNAMIC, DYNAMIC>;

    TileMatAData  aMatTile(M);
    TileMatW1Data w1MatTile;
    TileMatW2Data w2MatTile(N_l1);
    TileMatYData  yMatTile(M);

    ALeftTile   aTile(M);
    W1RightTile w1Tile;
    YAccTile    yAccTile(M);

    YLeftTile   yTile(M);
    W2RightTile w2Tile(N_l1);
    BAccTile    bAccTile(M, N_l1);

    for (unsigned e = 0; e < kE; ++e) {
        const int32_t start = expert_start[e];
        const int32_t count = expert_count[e];

        for (int32_t m0 = 0; m0 < count; m0 += M) {
            const unsigned remaining = static_cast<unsigned>(count - m0);
            const unsigned currentM = (remaining < static_cast<unsigned>(M)) ? remaining : static_cast<unsigned>(M);
            const size_t row = static_cast<size_t>(start) + static_cast<size_t>(m0);

            aMatTile.SetValidRow(currentM);
            yMatTile.SetValidRow(currentM);
            aTile.SetValidRow(currentM);
            yAccTile.SetValidRow(currentM);
            yTile.SetValidRow(currentM);

            for (unsigned n0 = 0; n0 < static_cast<unsigned>(H); n0 += N_l1) {
                const unsigned remainingN = static_cast<unsigned>(H) - n0;
                const unsigned currentN =
                    (remainingN < static_cast<unsigned>(N_l1)) ? remainingN : static_cast<unsigned>(N_l1);
                const unsigned storeN =
                    (n0 < kH) ? ((kH - n0 < currentN) ? (kH - n0) : currentN) : 0;
                w2MatTile.SetValidCol(currentN);
                w2Tile.SetValidCol(currentN);
                bAccTile.SetValidShape(currentM, currentN);

                for (int f1 = 0; f1 < F_l1_blocks; ++f1) {
                    const size_t fBase = static_cast<size_t>(f1) * F_l1;

                    for (int h1 = 0; h1 < H_l1_blocks; ++h1) {
                        const size_t hBase = static_cast<size_t>(h1) * H_l1;
                        const size_t aOff = row * H + hBase;
                        const size_t w1Off = static_cast<size_t>(e) * H * F + hBase * F + fBase;

                        GlobalShapeA aShape(currentM);
                        GlobalDataA aGlobal(A + aOff, aShape);
                        GlobalDataW1 w1Global(W1 + w1Off);

                        TLOAD(aMatTile, aGlobal);
                        TLOAD(w1MatTile, w1Global);

                        for (int h0 = 0; h0 < H_l0_segments; ++h0) {
                            const uint16_t hOff = static_cast<uint16_t>(h0 * H_l0);
                            TEXTRACT(aTile,  aMatTile,  0, hOff);
                            TEXTRACT(w1Tile, w1MatTile, hOff, 0);
                            if (h1 == 0 && h0 == 0) {
                                TMATMUL(yAccTile, aTile, w1Tile);
                            } else {
                                TMATMUL_ACC(yAccTile, aTile, w1Tile);
                            }
                        }
                    }

                    TMOV<TileMatYData, YAccTile, ReluPreMode::NormalRelu>(yMatTile, yAccTile);

                    const size_t w2Off = static_cast<size_t>(e) * F * H + fBase * H + n0;
                    GlobalShapeW2 w2Shape(currentN);
                    GlobalDataW2 w2Global(W2 + w2Off, w2Shape);
                    TLOAD(w2MatTile, w2Global);

                    for (int f0 = 0; f0 < F_l0_segments; ++f0) {
                        const uint16_t fOff = static_cast<uint16_t>(f0 * F_l0);
                        TEXTRACT(yTile,  yMatTile,  0, fOff);
                        TEXTRACT(w2Tile, w2MatTile, fOff, 0);
                        if (f1 == 0 && f0 == 0) {
                            TMATMUL(bAccTile, yTile, w2Tile);
                        } else {
                            TMATMUL_ACC(bAccTile, yTile, w2Tile);
                        }
                    }
                }

                if (storeN > 0) {
                    bAccTile.SetValidShape(currentM, storeN);
                    size_t bOff = row * kH + n0;
                    GlobalShapeB bShape(currentM, storeN);
                    GlobalDataB bGlobal(B + bOff, bShape);
                    TSTORE(bGlobal, bAccTile);
                }
            }
        }
    }
}

template <typename TOut, typename TIn, typename TWeight, typename TScratch>
void launchExpertFfn(uint8_t *B,
                     uint8_t *A,
                     int32_t *expert_count,
                     int32_t *expert_start,
                     uint8_t *W1,
                     uint8_t *W2,
                     void    *stream)
{
    runExpertFfn<TOut, TIn, TWeight, TScratch><<<1, nullptr, stream>>>(
        B, A, expert_count, expert_start, W1, W2);
}

template void launchExpertFfn<float, half, half, half>(
    uint8_t *B, uint8_t *A,
    int32_t *expert_count, int32_t *expert_start,
    uint8_t *W1, uint8_t *W2, void *stream);

extern "C" void launchExpertFfnFp16(uint8_t *B,
                                    uint8_t *A,
                                    int32_t *expert_count,
                                    int32_t *expert_start,
                                    uint8_t *W1,
                                    uint8_t *W2,
                                    uint8_t *Y_scratch,
                                    void    *stream)
{
    (void)Y_scratch;
    launchExpertFfn<float, half, half, half>(
        B, A, expert_count, expert_start, W1, W2, stream);
}
