/**
 * moe_segmented_gemm_one_layer_kernel.cpp - auto-mode A3 prototype.
 *
 * First cube/GEMM milestone inside the working MoE expert-segment loop. Reuses
 * the host-padded segment layout from
 * [moe_segmented_identity](../moe_segmented_identity/) — same nested
 * (expert, microtile) loop shape, same v1 tail policy (host pads counts to a
 * multiple of TILE_M; padded rows are zero so their GEMM output is zero) —
 * but replaces the elementwise TADDS with one expert-specific matrix multiply
 * per inner iteration:
 *
 *   packed_output[row : row + TILE_M, 0:O]
 *     = packed_tokens[row : row + TILE_M, 0:H]  @  expert_weight[e, 0:H, 0:O]
 *
 * Pattern source: tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp
 * `RunTMATMUL<float, half, half, float, validM, validK, validN, false>`
 * (isBias = false). This is the canonical A3 auto-mode-eligible cube combo
 * (FP16xFP16 -> FP32 accumulator, no bias). The same tile aliases,
 * GlobalTensor shape/stride, and TLOAD/TMOV/TMATMUL/TSTORE sequence are
 * copied here; the only structural delta is the (expert, microtile) outer
 * loop wrapping the body.
 *
 * Inputs   (GM, uint8_t* at the host boundary; cast inside):
 *   packed_output  [T_PADDED, O]   float32   (cube FP32 accumulator)
 *   packed_tokens  [T_PADDED, H]   float16
 *   expert_count   [kNumExperts]   int32     (PADDED counts; multiples of TILE_M)
 *   expert_start   [kNumExperts]   int32     (PADDED starts; prefix sum)
 *   expert_weight  [kNumExperts, H, O]  float16
 *
 * Auto-mode constraints honored:
 *   - Single AICORE (<<<1, nullptr, stream>>>); no block_idx work split.
 *   - Static tile shapes only inside the inner loop (no SetValidRow, no
 *     partial-tile stores, no dynamic valid region). Host padding makes
 *     every inner iter a full TILE_M-row tile.
 *   - Tile aliases lifted verbatim from RunTMATMUL: TileMatA/B/Acc + Left/
 *     Right/Acc, both A and B as BLayout::ColMajor in L1 with SLayout::
 *     RowMajor inner box and SFractalSize = 512 (the FP16-shaped value
 *     used by the reference for FP16 A/B).
 *   - No TASSIGN literal addresses (auto allocator pins each tile once).
 *   - No #ifndef __PTO_AUTO__ manual sync blocks (auto-sync inserts the
 *     MTE2->MTE1->M->FIX fences).
 *   - No Tile::data() in kernel; no *_IMPL calls; no raw CCE intrinsics;
 *     no Event<>; no TPipe / TPUSH / TPOP; no double buffering; no manual
 *     hard-coded L0/L1 buffer macros; no A5-only instructions.
 *
 * Patterns reused (all already user-confirmed):
 *   - scalar GM read of int32_t metadata             — §11.4 / §11.6
 *   - runtime scalar GM row offset                   — §11.4 / §11.5 / §11.6
 *   - nested (expert, microtile) loop                — §11.6
 *   - host-padded expert segment layout              — §11.6
 *
 * NEW (not yet confirmed by experiment; this kernel's verification target):
 *   - Cube path (TMATMUL) inside the segmented loop.
 *   - Three concurrently live tile types: Mat (L1), Left/Right (L0A/L0B),
 *     Acc (L0C).
 *   - GM input slice driven by a runtime row offset for both A (per inner
 *     iter) and per-expert weight B (per outer iter).
 *   - FP16 A/B -> FP32 C dtype combo in auto mode (matches tmatmul tilingKey=1).
 *
 * Host boundary: uint8_t* for the typed buffers and int32_t* for metadata —
 * mirrors the kernels/automode/a2a3/topk/topk_kernel.cpp pattern that
 * resolved compile-error E11 (host-side `__gm__` cast).
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace moe_segmented_gemm_one_layer_cfg {

// Static configuration. Must match scripts/gen_data.py and main.cpp.
constexpr unsigned kH          = 64;    // K dim (tokens hidden / weight rows)
constexpr unsigned kO          = 64;    // N dim (weight cols / output cols)
constexpr unsigned kTileM      = 128;   // M dim per cube tile
constexpr unsigned kNumExperts = 4;     // experts (compile-time)

}  // namespace moe_segmented_gemm_one_layer_cfg

template <typename TOut, typename TIn, typename TWeight>
__global__ AICORE void runMoeSegmentedGemmOneLayer(
    __gm__ uint8_t *packed_output_raw,
    __gm__ uint8_t *packed_tokens_raw,
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ uint8_t *expert_weight_raw)
{
    using namespace moe_segmented_gemm_one_layer_cfg;

    // Apply __gm__ + typed view inside the kernel (E11 pattern from topk).
    __gm__ TOut    *packed_output = reinterpret_cast<__gm__ TOut    *>(packed_output_raw);
    __gm__ TIn     *packed_tokens = reinterpret_cast<__gm__ TIn     *>(packed_tokens_raw);
    __gm__ TWeight *expert_weight = reinterpret_cast<__gm__ TWeight *>(expert_weight_raw);

    // ---- Tile alignment per the A3 tmatmul reference -----------------------
    // For FP16 inputs: blockAlign = C0_SIZE_BYTE / sizeof(TIn) = 32 / 2 = 16.
    // M aligned to 16, N/K aligned to blockAlign. With our shape:
    //   validM = kTileM = 128  -> M = 128
    //   validK = kH     = 64   -> K = 64
    //   validN = kO     = 64   -> N = 64
    // All are already aligned; M = validM, K = validK, N = validN.
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM + 15) / 16) * 16;
    constexpr int K = ((kH + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kO + blockAlign - 1) / blockAlign) * blockAlign;

    // ---- GM tensor views ---------------------------------------------------
    // Each view is a [validM, validK] / [validK, validN] / [validM, validN]
    // slice of the underlying buffer. Because validK == kH (the buffer's
    // row width) and validN == kO (the buffer's row width), the slice's row
    // stride equals the buffer's row stride and we can use the tmatmul-style
    // dense stride pattern.
    using GlobalDataA =
        GlobalTensor<TIn,     Shape<1, 1, 1, kTileM, kH>,
                     Stride<1 * kTileM * kH, 1 * kTileM * kH, kTileM * kH, kH, 1>>;
    using GlobalDataB =
        GlobalTensor<TWeight, Shape<1, 1, 1, kH,     kO>,
                     Stride<1 * kH * kO,     1 * kH * kO,     kH * kO,     kO, 1>>;
    using GlobalDataC =
        GlobalTensor<TOut,    Shape<1, 1, 1, kTileM, kO>,
                     Stride<1 * kTileM * kO, 1 * kTileM * kO, kTileM * kO, kO, 1>>;

    // ---- Tile aliases (exact copy from tmatmul_kernel.cpp RunTMATMUL) -----
    using TileMatAData = Tile<TileType::Mat, TIn,     M, K, BLayout::ColMajor,
                              kTileM, kH, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K, N, BLayout::ColMajor,
                              kH,     kO, SLayout::RowMajor, 512>;

    using LeftTile  = TileLeft <TIn,     M, K, kTileM, kH>;  // L0A
    using RightTile = TileRight<TWeight, K, N, kH,     kO>;  // L0B
    using AccTile   = TileAcc  <TOut,    M, N, kTileM, kO>;  // L0C

    // Declare each tile once outside the loops. Auto allocator pins L1/L0
    // addresses; reused across all inner iterations.
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    for (unsigned e = 0; e < kNumExperts; ++e) {
        int32_t start = expert_start[e];      // GM scalar read (resolved §11.4)
        int32_t count = expert_count[e];      // GM scalar read (resolved §11.4)

        // Per-expert weight is at offset e * (kH * kO) into the flat weight buffer.
        GlobalDataB bGlobal(expert_weight + static_cast<size_t>(e) * kH * kO);

        for (int32_t m0 = 0; m0 < count; m0 += static_cast<int32_t>(kTileM)) {
            size_t row  = static_cast<size_t>(start) + static_cast<size_t>(m0);
            size_t aOff = row * kH;
            size_t cOff = row * kO;

            GlobalDataA aGlobal(packed_tokens + aOff);
            GlobalDataC cGlobal(packed_output + cOff);

            // GM -> L1 (Mat tile, both A and B).
            TLOAD(aMatTile, aGlobal);
            TLOAD(bMatTile, bGlobal);

            // L1 -> L0A / L0B.
            TMOV(aTile, aMatTile);
            TMOV(bTile, bMatTile);

            // Cube op (no bias). cTile is overwritten each inner iter;
            // there is no across-iter accumulation. Auto-sync inserts the
            // MTE2 -> MTE1 -> M -> FIX fences.
            TMATMUL(cTile, aTile, bTile);

            // L0C -> GM.
            TSTORE(cGlobal, cTile);
        }
    }
}

template <typename TOut, typename TIn, typename TWeight>
void launchMoeSegmentedGemmOneLayer(uint8_t *packed_output,
                                    uint8_t *packed_tokens,
                                    int32_t *expert_count,
                                    int32_t *expert_start,
                                    uint8_t *expert_weight,
                                    void *stream)
{
    runMoeSegmentedGemmOneLayer<TOut, TIn, TWeight><<<1, nullptr, stream>>>(
        packed_output, packed_tokens, expert_count, expert_start, expert_weight);
}

// FP16 x FP16 -> FP32 instantiation (matches tmatmul tilingKey=1, the
// canonical A3 auto-mode-eligible cube combo).
template void launchMoeSegmentedGemmOneLayer<float, half, half>(
    uint8_t *packed_output, uint8_t *packed_tokens,
    int32_t *expert_count, int32_t *expert_start,
    uint8_t *expert_weight, void *stream);
