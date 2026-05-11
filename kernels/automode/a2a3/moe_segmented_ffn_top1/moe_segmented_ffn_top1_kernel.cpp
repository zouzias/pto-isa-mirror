/**
 * moe_segmented_ffn_top1_kernel.cpp - auto-mode A3 prototype.
 *
 * Full top-1 segmented FFN inside the already-proven per-expert microtile
 * loop: GEMM1 -> ReLU -> GEMM2. Single AICORE, single cube-arch kernel.
 *
 * Per inner microtile (one expert e, one (start + m0) row range):
 *
 *   A1 = packed_tokens[row : row + TILE_M, 0:H]   FP16  (load from GM into Mat tile)
 *   B1 = w1[e, 0:H, 0:F]                          FP16  (load from GM into Mat tile)
 *   C1_fp32 = A1 @ B1                                    (FP32 in L0C via TMATMUL)
 *
 *   // Single TSTORE fuses: ReLU on FP32 accumulator + down-cast to FP16
 *   // GM scratch (per-tile reuse). See "Assumption A.combined" note below.
 *   hidden_gm[0 : TILE_M, 0:F] = ReLU(C1_fp32) cast to FP16
 *
 *   A2 = hidden_gm[0 : TILE_M, 0:F]               FP16  (load back into the same Mat tile)
 *   B2 = w2[e, 0:F, 0:O]                          FP16  (load into the same Mat tile slot)
 *   C2_fp32 = A2 @ B2                                    (FP32 in L0C via TMATMUL)
 *
 *   packed_output[row : row + TILE_M, 0:O] = C2_fp32     (plain TSTORE; no ReLU here)
 *
 * Reuse of cube tiles across the two GEMMs:
 *   With our shape (kH = kF = kO = 64, kTileM = 128), every cube tile in
 *   GEMM1 has the same dimensions as the corresponding cube tile in GEMM2:
 *     - Mat A side : [kTileM, kH=64]  ==  [kTileM, kF=64]  (reused)
 *     - Mat B side : [kH,     kO=64]  ==  [kF,     kO=64]  (reused)
 *     - LeftTile   : [M=128,  K=64]   ==  same             (reused)
 *     - RightTile  : [K=64,   N=64]   ==  same             (reused)
 *     - AccTile    : [M=128,  N=64]   ==  same             (reused)
 *   So the auto allocator only needs one of each (no L0 budget pressure).
 *   This assumes the auto-sync analysis allows the second TLOAD/TMOV pair to
 *   reuse the same tile after the first TSTORE completes (Assumption A.reuse).
 *
 * Hidden scratch buffer:
 *   Sized at TILE_M * F * sizeof(half) = 128 * 64 * 2 = 16 KB on GM.
 *   Allocated once by main.cpp, reused across all (expert, microtile) iters.
 *   The kernel reconstructs the same GlobalTensor for it each inner iter.
 *
 * Reference patterns:
 *   - §A16 moe_segmented_gemm_one_layer: one TMATMUL per inner iter; tile
 *     aliases lifted from tmatmul_kernel.cpp RunTMATMUL<float, half, half, ...>.
 *   - §A17 moe_segmented_gemm_relu: ReLU fused into TSTORE via
 *     ReluPreMode::NormalRelu (cube path; no UB roundtrip).
 *   - tstore_acc2gm_kernel.cpp:
 *       LaunchTStoreAcc2gmNz2nd<4>: FP32 Acc -> FP16 GM dst (no ReLU).
 *       LaunchTStoreAcc2gmNz2nz<21>: FP32 Acc -> FP32 GM dst (with NormalRelu).
 *     The combination (FP32 Acc -> FP16 GM dst + NormalRelu in the same
 *     TSTORE call) has no in-tree precedent — see Assumption A.combined.
 *
 * Open assumptions this kernel exercises (will be promoted to Known on PASS):
 *   A.combined: TSTORE template arg combination
 *       <AccTile<float>, GlobalDataHiddenHalf, AtomicNone, NormalRelu>
 *     simultaneously applies ReLU (FIX-pipe) AND down-casts FP32 to FP16
 *     on the GM-store path. The two features are individually proven in
 *     `tstore_acc2gm` (tilingKey=4 has FP32->FP16 no-ReLU; tilingKey=21 has
 *     ReLU no-cast). Combination is new.
 *   A.reuse:    Reusing the same Mat / Left / Right / Acc tiles across two
 *               back-to-back TMATMUL calls in the same inner iter, with a
 *               TSTORE-then-TLOAD chain in between, is auto-sync-safe.
 *
 * Auto-mode constraints honored (mirror §A16 / §A17):
 *   - Single AICORE (<<<1, nullptr, stream>>>); no block_idx work split.
 *   - Static tile shapes everywhere; no SetValidRow / partial stores.
 *   - Tile declarations once outside both loops; auto allocator pins addresses.
 *   - No TASSIGN literal addresses; no #ifndef __PTO_AUTO__ manual-sync blocks
 *     (auto-sync inserts MTE2 -> MTE1 -> M -> FIX fences for both GEMMs);
 *     no Tile::data(); no *_IMPL; no raw CCE intrinsics; no Event<>; no
 *     TPipe/TPUSH/TPOP; no double buffering; no A5-only instructions.
 *   - uint8_t* host boundary for typed buffers + bare int32_t* for metadata
 *     + non-template `…Fp16` wrapper hiding `half` from main.cpp
 *     (compile_error_logbook.md §E13).
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace moe_segmented_ffn_top1_cfg {
constexpr unsigned kH          = 64;     // GEMM1 K dim (token hidden)
constexpr unsigned kF          = 64;     // GEMM1 N dim / GEMM2 K dim (FFN hidden)
constexpr unsigned kO          = 64;     // GEMM2 N dim (FFN output)
constexpr unsigned kTileM      = 128;
constexpr unsigned kNumExperts = 4;
}  // namespace moe_segmented_ffn_top1_cfg

template <typename TOut, typename TIn, typename TWeight>
__global__ AICORE void runMoeSegmentedFfnTop1(
    __gm__ uint8_t *packed_output_raw,
    __gm__ uint8_t *hidden_scratch_raw,   // device-resident FP16 scratch, kTileM*kF half (reused)
    __gm__ uint8_t *packed_tokens_raw,
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ uint8_t *w1_raw,
    __gm__ uint8_t *w2_raw)
{
    using namespace moe_segmented_ffn_top1_cfg;

    __gm__ TOut    *packed_output  = reinterpret_cast<__gm__ TOut    *>(packed_output_raw);
    __gm__ TIn     *hidden_scratch = reinterpret_cast<__gm__ TIn     *>(hidden_scratch_raw);
    __gm__ TIn     *packed_tokens  = reinterpret_cast<__gm__ TIn     *>(packed_tokens_raw);
    __gm__ TWeight *w1             = reinterpret_cast<__gm__ TWeight *>(w1_raw);
    __gm__ TWeight *w2             = reinterpret_cast<__gm__ TWeight *>(w2_raw);

    // Alignment (matches §A16 derivation for FP16 inputs):
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);   // FP16 -> 16
    constexpr int M  = ((kTileM + 15) / 16) * 16;            // 128
    constexpr int KH = ((kH + blockAlign - 1) / blockAlign) * blockAlign;  // 64
    constexpr int KF = ((kF + blockAlign - 1) / blockAlign) * blockAlign;  // 64
    constexpr int N  = ((kO + blockAlign - 1) / blockAlign) * blockAlign;  // 64
    // For our (64, 64, 64) shape, KH == KF and N == kO == kF, so the second
    // GEMM's tile dimensions match the first GEMM's, and one set of tile
    // declarations covers both.

    using GlobalDataA1 =
        GlobalTensor<TIn, Shape<1, 1, 1, kTileM, kH>,
                     Stride<1 * kTileM * kH, 1 * kTileM * kH, kTileM * kH, kH, 1>>;
    using GlobalDataB1 =
        GlobalTensor<TWeight, Shape<1, 1, 1, kH, kF>,
                     Stride<1 * kH * kF, 1 * kH * kF, kH * kF, kF, 1>>;
    // Hidden scratch is a fixed [kTileM, kF] FP16 buffer on GM. Strides treat
    // it as a standalone [kTileM, kF] block (row stride = kF). Same shape as
    // the GEMM1 output slice and the GEMM2 input slice.
    using GlobalDataHiddenFp16 =
        GlobalTensor<TIn, Shape<1, 1, 1, kTileM, kF>,
                     Stride<1 * kTileM * kF, 1 * kTileM * kF, kTileM * kF, kF, 1>>;
    using GlobalDataB2 =
        GlobalTensor<TWeight, Shape<1, 1, 1, kF, kO>,
                     Stride<1 * kF * kO, 1 * kF * kO, kF * kO, kO, 1>>;
    using GlobalDataC2 =
        GlobalTensor<TOut, Shape<1, 1, 1, kTileM, kO>,
                     Stride<1 * kTileM * kO, 1 * kTileM * kO, kTileM * kO, kO, 1>>;

    // Tile aliases — same form as §A16. Reused across both GEMMs because the
    // dimensions are identical (kH = kF = kO = 64).
    using TileMatAData = Tile<TileType::Mat, TIn,     M, KH, BLayout::ColMajor,
                              kTileM, kH, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, KH, N, BLayout::ColMajor,
                              kH,     kF, SLayout::RowMajor, 512>;

    using LeftTile  = TileLeft <TIn,     M,  KH, kTileM, kH>;
    using RightTile = TileRight<TWeight, KH, N,  kH,     kF>;
    using AccTile   = TileAcc  <TOut,    M,  N,  kTileM, kF>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    for (unsigned e = 0; e < kNumExperts; ++e) {
        int32_t start = expert_start[e];        // GM scalar read
        int32_t count = expert_count[e];        // GM scalar read

        // Per-expert weight offsets.
        GlobalDataB1 b1Global(w1 + static_cast<size_t>(e) * kH * kF);
        GlobalDataB2 b2Global(w2 + static_cast<size_t>(e) * kF * kO);

        for (int32_t m0 = 0; m0 < count; m0 += static_cast<int32_t>(kTileM)) {
            size_t row  = static_cast<size_t>(start) + static_cast<size_t>(m0);
            size_t aOff = row * kH;
            size_t cOff = row * kO;

            GlobalDataA1            a1Global(packed_tokens  + aOff);
            GlobalDataC2            c2Global(packed_output  + cOff);
            GlobalDataHiddenFp16    hGlobal (hidden_scratch);   // always offset 0; reused per inner iter

            // ===== GEMM1 ============================================================
            TLOAD(aMatTile, a1Global);
            TLOAD(bMatTile, b1Global);
            TMOV (aTile,  aMatTile);
            TMOV (bTile,  bMatTile);
            TMATMUL(cTile, aTile, bTile);
            // FIX-pipe TSTORE: applies ReLU AND down-casts FP32 Acc -> FP16 GM.
            // Combined-mode (Assumption A.combined). See header comment.
            TSTORE<AccTile, GlobalDataHiddenFp16,
                   AtomicType::AtomicNone, ReluPreMode::NormalRelu>(hGlobal, cTile);

            // ===== GEMM2 ============================================================
            // Reuse aMatTile / bMatTile / aTile / bTile / cTile (same shapes).
            TLOAD(aMatTile, hGlobal);
            TLOAD(bMatTile, b2Global);
            TMOV (aTile,  aMatTile);
            TMOV (bTile,  bMatTile);
            TMATMUL(cTile, aTile, bTile);
            // Plain L0C -> GM TSTORE (no ReLU on final output; FP32).
            TSTORE(c2Global, cTile);
        }
    }
}

template <typename TOut, typename TIn, typename TWeight>
void launchMoeSegmentedFfnTop1(uint8_t *packed_output,
                               uint8_t *hidden_scratch,
                               uint8_t *packed_tokens,
                               int32_t *expert_count,
                               int32_t *expert_start,
                               uint8_t *w1,
                               uint8_t *w2,
                               void *stream)
{
    runMoeSegmentedFfnTop1<TOut, TIn, TWeight><<<1, nullptr, stream>>>(
        packed_output, hidden_scratch, packed_tokens,
        expert_count, expert_start, w1, w2);
}

template void launchMoeSegmentedFfnTop1<float, half, half>(
    uint8_t *packed_output, uint8_t *hidden_scratch, uint8_t *packed_tokens,
    int32_t *expert_count, int32_t *expert_start,
    uint8_t *w1, uint8_t *w2, void *stream);

// Non-template host-boundary wrapper (compile_error_logbook.md §E13).
extern "C" void launchMoeSegmentedFfnTop1Fp16(uint8_t *packed_output,
                                               uint8_t *hidden_scratch,
                                               uint8_t *packed_tokens,
                                               int32_t *expert_count,
                                               int32_t *expert_start,
                                               uint8_t *w1,
                                               uint8_t *w2,
                                               void *stream)
{
    launchMoeSegmentedFfnTop1<float, half, half>(
        packed_output, hidden_scratch, packed_tokens,
        expert_count, expert_start, w1, w2, stream);
}
