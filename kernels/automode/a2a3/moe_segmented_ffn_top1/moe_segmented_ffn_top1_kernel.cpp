/**
 * moe_segmented_ffn_top1_kernel.cpp - auto-mode A3 prototype.
 *
 * SPLIT INTO TWO KERNELS (revision 2). The first attempt fused both GEMMs
 * into one __global__ body so they could share five cube tiles; that hung on
 * device, presumably because auto-sync's flag-pairing for cross-GEMM tile
 * reuse (cTile drain to scratch GM, then aMatTile reload from the same
 * scratch GM in the same iteration) has no in-tree precedent and produced
 * a wait_flag whose matching set_flag was never emitted.
 *
 * This revision splits the FFN into two `__global__ AICORE` functions, each
 * a verbatim copy of an already-confirmed-built milestone:
 *
 *   Stage 1 — `runFfnStage1Gemm1Relu` : §A17 moe_segmented_gemm_relu
 *             shape exactly. GEMM1 (FP16 × FP16 → FP32 acc) + fused ReLU +
 *             FP32→FP16 downcast in the L0C → GM TSTORE. The only delta vs
 *             §A17 is the GM destination dtype (half instead of float),
 *             which the previous run already confirmed works.
 *
 *   Stage 2 — `runFfnStage2Gemm2` : §A16 moe_segmented_gemm_one_layer shape
 *             exactly. GEMM2 (FP16 × FP16 → FP32 acc) on the scratch from
 *             stage 1 and w2[e], writing FP32 to packed_output.
 *
 * The host launches the two kernels back-to-back on the SAME stream. ACL
 * guarantees stream-order execution, so the second launch only starts after
 * stage 1 has fully drained to GM. There is therefore NO within-kernel
 * cross-GEMM auto-sync — each kernel only has to handle the §A16/§A17
 * known-good sync pattern.
 *
 * Why this matters:
 *   - Each kernel uses exactly 5 cube tiles (Mat A/B, Left, Right, Acc),
 *     matching the §A16/§A17 budget byte-for-byte.
 *   - Each TLOAD/TMOV/TMATMUL/TSTORE chain operates on independent GM
 *     regions within its own kernel — no cross-tile GM dependency through
 *     scratch.
 *   - The "scratch handoff" between the two GEMMs becomes an ACL
 *     stream-level dependency, not a PTO auto-sync dependency.
 *
 *   for each expert e, each microtile m0:
 *     Stage 1 kernel:
 *       scratch[row:row+kTileM, 0:kF] = max(packed_tokens[…] @ w1[e], 0)   (FP16)
 *     Stage 2 kernel:
 *       packed_output[row:row+kTileM, 0:kH] = scratch[…] @ w2[e]            (FP32)
 *
 * Auto-mode constraints honored (per stage; identical to §A16/§A17):
 *   - Single AICORE per kernel; no block_idx work split.
 *   - Static tile shapes; no SetValidRow / partial stores.
 *   - All tiles declared once outside both loops; auto allocator pins each.
 *   - No TASSIGN literal addresses, no `#ifndef __PTO_AUTO__` manual-sync,
 *     no Tile::data() in kernel code, no *_IMPL calls, no raw CCE intrinsics,
 *     no Event<>, no TPipe/TPUSH/TPOP, no double buffering, no A5-only ops.
 *   - Only ReluPreMode::NormalRelu; no GELU/SiLU/LeakyReLU.
 *
 * Host boundary: uint8_t* for the typed FP16/FP32 buffers and int32_t* for
 * metadata; non-template `…Fp16` wrapper hides `half` from main.cpp (see
 * compile_error_logbook.md §E13 / known_good_kernel_examples.md §A16).
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace moe_segmented_ffn_top1_cfg {

// Static configuration. Must match scripts/gen_data.py and main.cpp.
constexpr unsigned kH          = 64;   // hidden dim (GEMM1 K = GEMM2 N)
constexpr unsigned kF          = 64;   // FFN intermediate dim (GEMM1 N = GEMM2 K)
constexpr unsigned kTileM      = 128;  // M dim per cube tile
constexpr unsigned kNumExperts = 4;    // experts (compile-time)

}  // namespace moe_segmented_ffn_top1_cfg

// ============================================================================
// Stage 1 — GEMM1 + fused ReLU + FP32->FP16 in TSTORE.
//   In  : packed_tokens [T_PADDED, kH] half, w1 [kE, kH, kF] half.
//   Out : scratch       [T_PADDED, kF] half (post-ReLU; written by FIX pipe).
// Shape source: byte-for-byte the §A17 moe_segmented_gemm_relu kernel, with
// the GM dest dtype changed to half. That dtype change is the one new piece
// this milestone needed; the previous run confirmed it works.
// ============================================================================
template <typename TIn, typename TWeight, typename TScratch>
__global__ AICORE void runFfnStage1Gemm1Relu(
    __gm__ uint8_t *scratch_raw,
    __gm__ uint8_t *packed_tokens_raw,
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ uint8_t *w1_raw)
{
    using namespace moe_segmented_ffn_top1_cfg;

    __gm__ TIn      *packed_tokens = reinterpret_cast<__gm__ TIn      *>(packed_tokens_raw);
    __gm__ TWeight  *w1            = reinterpret_cast<__gm__ TWeight  *>(w1_raw);
    __gm__ TScratch *scratch       = reinterpret_cast<__gm__ TScratch *>(scratch_raw);

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

    for (unsigned e = 0; e < kNumExperts; ++e) {
        int32_t start = expert_start[e];
        int32_t count = expert_count[e];

        GlobalDataB bGlobal(w1 + static_cast<size_t>(e) * kH * kF);

        for (int32_t m0 = 0; m0 < count; m0 += static_cast<int32_t>(kTileM)) {
            size_t row  = static_cast<size_t>(start) + static_cast<size_t>(m0);
            size_t aOff = row * kH;
            size_t cOff = row * kF;

            GlobalDataA aGlobal(packed_tokens + aOff);
            GlobalDataC cGlobal(scratch       + cOff);

            TLOAD(aMatTile, aGlobal);
            TLOAD(bMatTile, bGlobal);
            TMOV(aTile, aMatTile);
            TMOV(bTile, bMatTile);
            TMATMUL(cTile, aTile, bTile);

            // L0C -> GM with ReLU + FP32->FP16 downcast fused into the FIX
            // pipe. Confirmed working in the previous run.
            TSTORE<AccTile, GlobalDataC, AtomicType::AtomicNone,
                   ReluPreMode::NormalRelu>(cGlobal, cTile);
        }
    }
}

// ============================================================================
// Stage 2 — GEMM2 only.
//   In  : scratch [T_PADDED, kF] half (produced by stage 1), w2 [kE, kF, kH] half.
//   Out : packed_output [T_PADDED, kH] float32.
// Shape source: byte-for-byte the §A16 moe_segmented_gemm_one_layer kernel,
// with the A buffer pointed at scratch instead of packed_tokens.
// ============================================================================
template <typename TOut, typename TScratch, typename TWeight>
__global__ AICORE void runFfnStage2Gemm2(
    __gm__ uint8_t *packed_output_raw,
    __gm__ uint8_t *scratch_raw,
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ uint8_t *w2_raw)
{
    using namespace moe_segmented_ffn_top1_cfg;

    __gm__ TOut     *packed_output = reinterpret_cast<__gm__ TOut     *>(packed_output_raw);
    __gm__ TScratch *scratch       = reinterpret_cast<__gm__ TScratch *>(scratch_raw);
    __gm__ TWeight  *w2            = reinterpret_cast<__gm__ TWeight  *>(w2_raw);

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

    for (unsigned e = 0; e < kNumExperts; ++e) {
        int32_t start = expert_start[e];
        int32_t count = expert_count[e];

        GlobalDataB bGlobal(w2 + static_cast<size_t>(e) * kF * kH);

        for (int32_t m0 = 0; m0 < count; m0 += static_cast<int32_t>(kTileM)) {
            size_t row  = static_cast<size_t>(start) + static_cast<size_t>(m0);
            size_t aOff = row * kF;
            size_t cOff = row * kH;

            GlobalDataA aGlobal(scratch       + aOff);
            GlobalDataC cGlobal(packed_output + cOff);

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
void launchFfnStage1Gemm1Relu(uint8_t *scratch,
                              uint8_t *packed_tokens,
                              int32_t *expert_count,
                              int32_t *expert_start,
                              uint8_t *w1,
                              void    *stream)
{
    runFfnStage1Gemm1Relu<TIn, TWeight, TScratch><<<1, nullptr, stream>>>(
        scratch, packed_tokens, expert_count, expert_start, w1);
}

template <typename TOut, typename TScratch, typename TWeight>
void launchFfnStage2Gemm2(uint8_t *packed_output,
                          uint8_t *scratch,
                          int32_t *expert_count,
                          int32_t *expert_start,
                          uint8_t *w2,
                          void    *stream)
{
    runFfnStage2Gemm2<TOut, TScratch, TWeight><<<1, nullptr, stream>>>(
        packed_output, scratch, expert_count, expert_start, w2);
}

template void launchFfnStage1Gemm1Relu<half, half, half>(
    uint8_t *scratch, uint8_t *packed_tokens,
    int32_t *expert_count, int32_t *expert_start,
    uint8_t *w1, void *stream);

template void launchFfnStage2Gemm2<float, half, half>(
    uint8_t *packed_output, uint8_t *scratch,
    int32_t *expert_count, int32_t *expert_start,
    uint8_t *w2, void *stream);

// Non-template host-boundary wrapper. Fires both stage kernels on the same
// stream; ACL stream-order semantics guarantee stage 2 only starts after
// stage 1's TSTORE-to-scratch is fully drained to GM. No within-kernel
// cross-GEMM auto-sync is involved.
//
// If you need to debug only stage 1 (scratch path), set kSkipStage2 = true.
// Stage 1 still runs; stage 2 launch is skipped, so packed_output keeps
// whatever the host poisoned it with — compare_outputs.py will detect that.
extern "C" void launchMoeSegmentedFfnTop1Fp16(uint8_t *packed_output,
                                              uint8_t *packed_tokens,
                                              int32_t *expert_count,
                                              int32_t *expert_start,
                                              uint8_t *w1,
                                              uint8_t *w2,
                                              uint8_t *scratch,
                                              void    *stream)
{
    // Toggle this to true to narrow the milestone to "GEMM1 + ReLU + FP16
    // scratch only" (skip stage 2 entirely from the host side).
    constexpr bool kSkipStage2 = false;

    launchFfnStage1Gemm1Relu<half, half, half>(
        scratch, packed_tokens, expert_count, expert_start, w1, stream);

    if constexpr (!kSkipStage2) {
        launchFfnStage2Gemm2<float, half, half>(
            packed_output, scratch, expert_count, expert_start, w2, stream);
    }
}
