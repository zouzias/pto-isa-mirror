/**
 * expert_ffn_kernel.cpp - auto-mode A3 prototype.
 *
 * Two-stage per-expert FFN over packed-by-expert tokens. Both stages run on
 * the cube target (`--cce-aicore-arch=dav-c220-cube`); ReLU is fused into
 * Stage 1's TSTORE FixPipe path (no vector hop). Generic over kTopK
 * (the kernel itself doesn't see kTopK — it only sees the packed
 * row count via expert_count / expert_start).
 *
 * Inputs   (GM): A         [kT*kTopK + 16, kH] fp16
 *                expert_count [kE]             int32
 *                expert_start [kE]             int32
 *                W1        [kE, kH, kF]        fp16
 *                W2        [kE, kF, kH]        fp16
 * Scratch  (GM): Y         [kT*kTopK + 16, kF] fp16  (post-ReLU, fp32->fp16 in FixPipe)
 * Outputs  (GM): B         [kT*kTopK + 16, kH] fp32  (first kT*kTopK rows valid)
 *
 * Per-expert overspill scheme (mirrors §11.9 §A18, with smaller kTileM):
 *   for each expert e:
 *     for m0 = 0; m0 < count[e]; m0 += kTileM:
 *       Stage 1: A_tile[start[e]+m0 : +kTileM] @ W1[e] -> Y[start[e]+m0 : +kTileM]
 *                with fused ReLU + fp32->fp16 in TSTORE FixPipe.
 *       Stage 2: Y_tile[start[e]+m0 : +kTileM] @ W2[e] -> B[start[e]+m0 : +kTileM]
 *                with fp32 accumulator written directly to GM.
 *
 * The loop writes kTileM rows per iteration even when count[e] is not a
 * multiple of kTileM. Rows past start[e]+count[e] are "overspill" — the next
 * non-empty expert e' starts at start[e']=start[e]+count[e] and overwrites
 * them. The last non-empty expert's overspill lands in the 16-row trailing
 * pad of A/Y/B (which the host allocates and the gather kernel ignores).
 *
 * Auto-mode constraints honored (per stage; mirrors §A17 / §A16 / §11.9):
 *   - Single AICORE per kernel (<<<1, nullptr, stream>>>); no block_idx split.
 *   - Static tile shapes (M = ceil(kTileM/16)*16, K/N rounded to blockAlign).
 *   - 5 cube tiles per stage (Mat A/B, Left, Right, Acc), declared OUTSIDE
 *     both loops (single auto-allocator analysis pin).
 *   - No TASSIGN literal addresses, no `#ifndef __PTO_AUTO__` manual-sync,
 *     no Tile::data() in kernel code, no *_IMPL calls, no raw CCE intrinsics,
 *     no Event<>, no TPipe / TPUSH / TPOP, no double buffering, no A5-only ops.
 *   - Only ReluPreMode::NormalRelu (the in-tree-proven activation enum value).
 *
 * Two stream-serialised __global__ AICORE kernels. ACL stream ordering
 * guarantees stage 2 starts only after stage 1's TSTORE-to-Y is fully
 * drained to GM — no within-kernel cross-GEMM auto-sync.
 *
 * Host boundary: uint8_t* for the typed fp16/fp32 buffers; non-template
 * `…Fp16` wrapper hides `half` from main.cpp (compile_error_logbook.md §E13).
 *
 * Pattern source:
 *   - kernels/automode/a2a3/moe_segmented_ffn_top1/  (§11.9; same two-stage
 *     skeleton, with kTileM=128 there vs kTileM=16 here, and an overspill
 *     `m0 < count` loop in both)
 *   - kernels/automode/a2a3/moe_segmented_gemm_one_layer/  (§A16)
 *   - kernels/automode/a2a3/moe_segmented_gemm_relu/       (§A17)
 *
 * Limitations (v1):
 *   - kTileM=16 (cube minimum) — smaller than §11.9's 128. §11.9 was the
 *     proven shape; M=16 is below it but within the documented cube
 *     M-alignment rule. If hardware fails, raise to 128 and grow the
 *     trailing pad to 128.
 *   - kH = kF = 64; both 64-wide K/N dimensions fit comfortably in L0A/L0B
 *     at fp16. Larger shapes need Split-K (postponed; see
 *     pto_auto_mode_hw_optimization_guide.md §1.3).
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace expert_ffn_cfg {

// v1 shape — must match scripts/gen_data.py and main.cpp.
constexpr unsigned kT     = 256;
constexpr unsigned kH     = 64;   // GEMM1 K = GEMM2 N
constexpr unsigned kF     = 64;   // GEMM1 N = GEMM2 K
constexpr unsigned kE     = 32;
constexpr unsigned kTopK  = 1;
constexpr unsigned kTileM = 16;   // overspill tile granularity (cube min M)

constexpr unsigned kPackedRows   = kT * kTopK;
constexpr unsigned kOverspillPad = kTileM;
constexpr unsigned kAlloc        = kPackedRows + kOverspillPad;

}  // namespace expert_ffn_cfg

// ============================================================================
// Stage 1 — GEMM1 + fused ReLU + fp32->fp16 in TSTORE.
//   A_tile (rows start[e]+m0..+kTileM, kH cols) @ W1[e] (kH×kF)
//   -> Y[same rows, kF cols] with ReLU + fp32->fp16 fused in FixPipe.
// ============================================================================
template <typename TIn, typename TWeight, typename TScratch>
__global__ AICORE void runFfnStage1Gemm1Relu(
    __gm__ uint8_t *Y_raw,
    __gm__ uint8_t *A_raw,
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ uint8_t *W1_raw)
{
    using namespace expert_ffn_cfg;

    __gm__ TIn      *A  = reinterpret_cast<__gm__ TIn      *>(A_raw);
    __gm__ TWeight  *W1 = reinterpret_cast<__gm__ TWeight  *>(W1_raw);
    __gm__ TScratch *Y  = reinterpret_cast<__gm__ TScratch *>(Y_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM + 15) / 16) * 16;
    constexpr int K = ((kH      + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kF      + blockAlign - 1) / blockAlign) * blockAlign;

    using GlobalDataA =
        GlobalTensor<TIn,      Shape<1, 1, 1, kTileM, kH>,
                     Stride<1 * kTileM * kH, 1 * kTileM * kH, kTileM * kH, kH, 1>>;
    using GlobalDataB =
        GlobalTensor<TWeight,  Shape<1, 1, 1, kH,     kF>,
                     Stride<1 * kH * kF,     1 * kH * kF,     kH * kF,     kF, 1>>;
    using GlobalDataC =
        GlobalTensor<TScratch, Shape<1, 1, 1, kTileM, kF>,
                     Stride<1 * kTileM * kF, 1 * kTileM * kF, kTileM * kF, kF, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, K, BLayout::ColMajor,
                              kTileM, kH, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K, N, BLayout::ColMajor,
                              kH,     kF, SLayout::RowMajor, 512>;
    using LeftTile     = TileLeft <TIn,     M, K, kTileM, kH>;
    using RightTile    = TileRight<TWeight, K, N, kH,     kF>;
    using AccTile      = TileAcc  <float,   M, N, kTileM, kF>;  // FP32 acc

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    for (unsigned e = 0; e < kE; ++e) {
        int32_t start = expert_start[e];
        int32_t count = expert_count[e];

        GlobalDataB bGlobal(W1 + static_cast<size_t>(e) * kH * kF);

        for (int32_t m0 = 0; m0 < count; m0 += static_cast<int32_t>(kTileM)) {
            size_t row  = static_cast<size_t>(start) + static_cast<size_t>(m0);
            size_t aOff = row * kH;
            size_t cOff = row * kF;

            GlobalDataA aGlobal(A + aOff);
            GlobalDataC cGlobal(Y + cOff);

            TLOAD(aMatTile, aGlobal);
            TLOAD(bMatTile, bGlobal);
            TMOV(aTile, aMatTile);
            TMOV(bTile, bMatTile);
            TMATMUL(cTile, aTile, bTile);

            // L0C -> GM with ReLU + FP32->FP16 fused in FixPipe (§A17 / §11.8).
            TSTORE<AccTile, GlobalDataC, AtomicType::AtomicNone,
                   ReluPreMode::NormalRelu>(cGlobal, cTile);
        }
    }
}

// ============================================================================
// Stage 2 — GEMM2 only.
//   Y_tile (rows start[e]+m0..+kTileM, kF cols) @ W2[e] (kF×kH)
//   -> B[same rows, kH cols] as fp32 accumulator.
// ============================================================================
template <typename TOut, typename TScratch, typename TWeight>
__global__ AICORE void runFfnStage2Gemm2(
    __gm__ uint8_t *B_raw,
    __gm__ uint8_t *Y_raw,
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ uint8_t *W2_raw)
{
    using namespace expert_ffn_cfg;

    __gm__ TOut     *B  = reinterpret_cast<__gm__ TOut     *>(B_raw);
    __gm__ TScratch *Y  = reinterpret_cast<__gm__ TScratch *>(Y_raw);
    __gm__ TWeight  *W2 = reinterpret_cast<__gm__ TWeight  *>(W2_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TScratch);
    constexpr int M = ((kTileM + 15) / 16) * 16;
    constexpr int K = ((kF      + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kH      + blockAlign - 1) / blockAlign) * blockAlign;

    using GlobalDataA =
        GlobalTensor<TScratch, Shape<1, 1, 1, kTileM, kF>,
                     Stride<1 * kTileM * kF, 1 * kTileM * kF, kTileM * kF, kF, 1>>;
    using GlobalDataB =
        GlobalTensor<TWeight,  Shape<1, 1, 1, kF,     kH>,
                     Stride<1 * kF * kH,     1 * kF * kH,     kF * kH,     kH, 1>>;
    using GlobalDataC =
        GlobalTensor<TOut,     Shape<1, 1, 1, kTileM, kH>,
                     Stride<1 * kTileM * kH, 1 * kTileM * kH, kTileM * kH, kH, 1>>;

    using TileMatAData = Tile<TileType::Mat, TScratch, M, K, BLayout::ColMajor,
                              kTileM, kF, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight,  K, N, BLayout::ColMajor,
                              kF,     kH, SLayout::RowMajor, 512>;
    using LeftTile     = TileLeft <TScratch, M, K, kTileM, kF>;
    using RightTile    = TileRight<TWeight,  K, N, kF,     kH>;
    using AccTile      = TileAcc  <TOut,     M, N, kTileM, kH>;  // FP32 acc

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    for (unsigned e = 0; e < kE; ++e) {
        int32_t start = expert_start[e];
        int32_t count = expert_count[e];

        GlobalDataB bGlobal(W2 + static_cast<size_t>(e) * kF * kH);

        for (int32_t m0 = 0; m0 < count; m0 += static_cast<int32_t>(kTileM)) {
            size_t row  = static_cast<size_t>(start) + static_cast<size_t>(m0);
            size_t aOff = row * kF;
            size_t cOff = row * kH;

            GlobalDataA aGlobal(Y + aOff);
            GlobalDataC cGlobal(B + cOff);

            TLOAD(aMatTile, aGlobal);
            TLOAD(bMatTile, bGlobal);
            TMOV(aTile, aMatTile);
            TMOV(bTile, bMatTile);
            TMATMUL(cTile, aTile, bTile);

            TSTORE(cGlobal, cTile);
        }
    }
}

// ----------------------------------------------------------------------------
// Templated host launchers. The non-template `…Fp16` wrapper at the bottom
// hides `half` from main.cpp (compile_error_logbook.md §E13).
// ----------------------------------------------------------------------------

template <typename TIn, typename TWeight, typename TScratch>
void launchFfnStage1Gemm1Relu(uint8_t *Y,
                              uint8_t *A,
                              int32_t *expert_count,
                              int32_t *expert_start,
                              uint8_t *W1,
                              void    *stream)
{
    runFfnStage1Gemm1Relu<TIn, TWeight, TScratch><<<1, nullptr, stream>>>(
        Y, A, expert_count, expert_start, W1);
}

template <typename TOut, typename TScratch, typename TWeight>
void launchFfnStage2Gemm2(uint8_t *B,
                          uint8_t *Y,
                          int32_t *expert_count,
                          int32_t *expert_start,
                          uint8_t *W2,
                          void    *stream)
{
    runFfnStage2Gemm2<TOut, TScratch, TWeight><<<1, nullptr, stream>>>(
        B, Y, expert_count, expert_start, W2);
}

template void launchFfnStage1Gemm1Relu<half, half, half>(
    uint8_t *Y, uint8_t *A,
    int32_t *expert_count, int32_t *expert_start,
    uint8_t *W1, void *stream);

template void launchFfnStage2Gemm2<float, half, half>(
    uint8_t *B, uint8_t *Y,
    int32_t *expert_count, int32_t *expert_start,
    uint8_t *W2, void *stream);

// Non-template host-boundary wrapper. Fires both stage kernels on the same
// stream; ACL stream-order semantics guarantee stage 2 only starts after
// stage 1's TSTORE-to-Y is fully drained to GM.
extern "C" void launchExpertFfnFp16(uint8_t *B,
                                    uint8_t *A,
                                    int32_t *expert_count,
                                    int32_t *expert_start,
                                    uint8_t *W1,
                                    uint8_t *W2,
                                    uint8_t *Y_scratch,
                                    void    *stream)
{
    launchFfnStage1Gemm1Relu<half, half, half>(
        Y_scratch, A, expert_count, expert_start, W1, stream);

    launchFfnStage2Gemm2<float, half, half>(
        B, Y_scratch, expert_count, expert_start, W2, stream);
}
