/**
 * moe_segmented_ffn_top1_kernel.cpp - auto-mode A3 prototype.
 *
 * Per-expert top-1 segmented FFN: GEMM1 -> ReLU -> GEMM2, all inside the
 * per-expert / per-microtile loop already validated by §A15 / §A16 / §A17.
 *
 *   for (e = 0; e < kNumExperts; ++e) {
 *       start = expert_start[e];     // PADDED
 *       count = expert_count[e];     // PADDED; multiple of kTileM
 *       for (m0 = 0; m0 < count; m0 += kTileM) {
 *           row = start + m0;
 *           // GEMM1: [kTileM, kH] x [kH, kF] -> [kTileM, kF]
 *           A1   = packed_tokens[row : row + kTileM, 0:kH]            // FP16
 *           B1   = w1[e, 0:kH, 0:kF]                                  // FP16
 *           Acc1 = A1 @ B1                                            // FP32 (cube)
 *           // ReLU + down-cast fused into the L0C -> GM TSTORE.
 *           scratch[row : row + kTileM, 0:kF] = max(Acc1, 0)          // FP16
 *
 *           // GEMM2: [kTileM, kF] x [kF, kH] -> [kTileM, kH]
 *           A2   = scratch[row : row + kTileM, 0:kF]                  // FP16
 *           B2   = w2[e, 0:kF, 0:kH]                                  // FP16
 *           Acc2 = A2 @ B2                                            // FP32 (cube)
 *           packed_output[row : row + kTileM, 0:kH] = Acc2            // FP32
 *       }
 *   }
 *
 * Inputs (GM, uint8_t* at the host boundary; cast inside):
 *   packed_tokens   [T_PADDED, kH]            float16
 *   w1              [kNumExperts, kH, kF]     float16
 *   w2              [kNumExperts, kF, kH]     float16
 *   expert_count    [kNumExperts]             int32   (PADDED counts; multiples of kTileM)
 *   expert_start    [kNumExperts]             int32   (PADDED starts; prefix sum)
 *   scratch         [T_PADDED, kF]            float16 (kernel-managed temporary; host allocates)
 *
 * Outputs (GM):
 *   packed_output   [T_PADDED, kH]            float32
 *
 * Composition vs known-good milestones:
 *   GEMM1: byte-for-byte the §A17 moe_segmented_gemm_relu shape, except the
 *          TSTORE destination dtype changes from float (FP32 GM) to half
 *          (FP16 GM). The TileData (AccTile<float>), the TMATMUL, the
 *          ReluPreMode::NormalRelu template arg, and the cube tile aliases
 *          are unchanged.
 *   GEMM2: byte-for-byte the §A16 moe_segmented_gemm_one_layer shape. A
 *          comes from the GEMM1 FP16 scratch; B from w2[e]; C is FP32.
 *
 * NEW (single isolated assumption): fused FP32 Acc -> FP16 GM + ReLU in a
 * single TSTORE call, in ND layout. The same dtype combo + ReLU is validated
 * in NZ layout via tstore_acc2gm Nz2nz tilingKey=21
 *   `<0, float, float, half, ..., 1>` at
 *   tests/npu/a2a3/src/st/testcase/tstore_acc2gm/tstore_acc2gm_kernel.cpp:627.
 * ND-layout variant of this combo is the one new piece this milestone tests.
 * If it fails, the documented fallback (see README) is to split the kernel:
 * (1) §A17 gemm_relu writing FP32 scratch, (2) a separate FP32->FP16 cast
 * pass, (3) §A16 gemm_one_layer reading the FP16 scratch.
 *
 * Auto-mode constraints honored (mirrors §A16 / §A17):
 *   - Single AICORE (<<<1, nullptr, stream>>>); no block_idx work split.
 *   - Static tile shapes inside the inner loop (no SetValidRow / partial
 *     stores). Host-padded counts make every inner iter a full kTileM tile.
 *   - Tile aliases lifted verbatim from RunTMATMUL<float, half, half, float>
 *     (the canonical A3 auto-mode-eligible FP16xFP16->FP32 cube combo).
 *   - All tiles declared once outside both loops; auto allocator pins each
 *     address; auto-sync inserts the MTE2 -> MTE1 -> M -> FIX fences between
 *     GEMM1 -> GEMM2 on the same scratch row.
 *   - No TASSIGN literal addresses, no #ifndef __PTO_AUTO__ manual-sync, no
 *     Tile::data() in kernel code, no *_IMPL calls, no raw CCE intrinsics,
 *     no Event<>, no TPipe / TPUSH / TPOP, no double buffering, no A5-only
 *     instructions, no GELU / SiLU / LeakyReLU (only NormalRelu is exercised
 *     by §A17).
 *
 * Host boundary: uint8_t* for the typed FP16 / FP32 buffers and int32_t* for
 * metadata; non-template `...Fp16` wrapper hides `half` from main.cpp (see
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

template <typename TOut, typename TIn, typename TWeight, typename TScratch>
__global__ AICORE void runMoeSegmentedFfnTop1(
    __gm__ uint8_t *packed_output_raw,
    __gm__ uint8_t *packed_tokens_raw,
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ uint8_t *w1_raw,
    __gm__ uint8_t *w2_raw,
    __gm__ uint8_t *scratch_raw)
{
    using namespace moe_segmented_ffn_top1_cfg;

    __gm__ TOut     *packed_output = reinterpret_cast<__gm__ TOut     *>(packed_output_raw);
    __gm__ TIn      *packed_tokens = reinterpret_cast<__gm__ TIn      *>(packed_tokens_raw);
    __gm__ TWeight  *w1            = reinterpret_cast<__gm__ TWeight  *>(w1_raw);
    __gm__ TWeight  *w2            = reinterpret_cast<__gm__ TWeight  *>(w2_raw);
    __gm__ TScratch *scratch       = reinterpret_cast<__gm__ TScratch *>(scratch_raw);

    // ---- Tile alignment (FP16 path, per §A16 / §A17) -----------------------
    // For FP16 inputs: blockAlign = C0_SIZE_BYTE / sizeof(TIn) = 32 / 2 = 16.
    // With kTileM=128, kH=kF=64 every dim is already aligned; M=128, K=N=64.
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M  = ((kTileM + 15) / 16) * 16;
    constexpr int K1 = ((kH      + blockAlign - 1) / blockAlign) * blockAlign;  // GEMM1 K
    constexpr int N1 = ((kF      + blockAlign - 1) / blockAlign) * blockAlign;  // GEMM1 N
    constexpr int K2 = ((kF      + blockAlign - 1) / blockAlign) * blockAlign;  // GEMM2 K
    constexpr int N2 = ((kH      + blockAlign - 1) / blockAlign) * blockAlign;  // GEMM2 N

    // ---- GM tensor views ---------------------------------------------------
    // GEMM1:  A1 = packed_tokens slice [kTileM, kH] half
    //         B1 = w1[e]                [kH, kF]    half
    //         C1 = scratch slice         [kTileM, kF] half  (post-ReLU)
    // GEMM2:  A2 = scratch slice         [kTileM, kF] half
    //         B2 = w2[e]                [kF, kH]    half
    //         C2 = packed_output slice  [kTileM, kH] float
    using GlobalDataA1 =
        GlobalTensor<TIn,      Shape<1, 1, 1, kTileM, kH>,
                     Stride<1 * kTileM * kH, 1 * kTileM * kH, kTileM * kH, kH, 1>>;
    using GlobalDataB1 =
        GlobalTensor<TWeight,  Shape<1, 1, 1, kH,     kF>,
                     Stride<1 * kH * kF,     1 * kH * kF,     kH * kF,     kF, 1>>;
    using GlobalDataC1 =
        GlobalTensor<TScratch, Shape<1, 1, 1, kTileM, kF>,
                     Stride<1 * kTileM * kF, 1 * kTileM * kF, kTileM * kF, kF, 1>>;
    using GlobalDataA2 =
        GlobalTensor<TScratch, Shape<1, 1, 1, kTileM, kF>,
                     Stride<1 * kTileM * kF, 1 * kTileM * kF, kTileM * kF, kF, 1>>;
    using GlobalDataB2 =
        GlobalTensor<TWeight,  Shape<1, 1, 1, kF,     kH>,
                     Stride<1 * kF * kH,     1 * kF * kH,     kF * kH,     kH, 1>>;
    using GlobalDataC2 =
        GlobalTensor<TOut,     Shape<1, 1, 1, kTileM, kH>,
                     Stride<1 * kTileM * kH, 1 * kTileM * kH, kTileM * kH, kH, 1>>;

    // ---- Tile aliases (exact RunTMATMUL pattern; see §A16) ----------------
    // GEMM1
    using TileMatA1Data = Tile<TileType::Mat, TIn,      M,  K1, BLayout::ColMajor,
                               kTileM, kH, SLayout::RowMajor, 512>;
    using TileMatB1Data = Tile<TileType::Mat, TWeight,  K1, N1, BLayout::ColMajor,
                               kH,     kF, SLayout::RowMajor, 512>;
    using LeftTile1     = TileLeft <TIn,      M,  K1, kTileM, kH>;  // L0A
    using RightTile1    = TileRight<TWeight,  K1, N1, kH,     kF>;  // L0B
    using AccTile1      = TileAcc  <TOut,     M,  N1, kTileM, kF>;  // L0C (FP32)

    // GEMM2
    using TileMatA2Data = Tile<TileType::Mat, TScratch, M,  K2, BLayout::ColMajor,
                               kTileM, kF, SLayout::RowMajor, 512>;
    using TileMatB2Data = Tile<TileType::Mat, TWeight,  K2, N2, BLayout::ColMajor,
                               kF,     kH, SLayout::RowMajor, 512>;
    using LeftTile2     = TileLeft <TScratch, M,  K2, kTileM, kF>;  // L0A
    using RightTile2    = TileRight<TWeight,  K2, N2, kF,     kH>;  // L0B
    using AccTile2      = TileAcc  <TOut,     M,  N2, kTileM, kH>;  // L0C (FP32)

    // Declare each tile once outside both loops. Auto allocator pins L1 / L0
    // addresses; reused across all inner iterations.
    TileMatA1Data a1MatTile;
    TileMatB1Data b1MatTile;
    LeftTile1     a1Tile;
    RightTile1    b1Tile;
    AccTile1      c1Tile;

    TileMatA2Data a2MatTile;
    TileMatB2Data b2MatTile;
    LeftTile2     a2Tile;
    RightTile2    b2Tile;
    AccTile2      c2Tile;

    for (unsigned e = 0; e < kNumExperts; ++e) {
        int32_t start = expert_start[e];      // GM scalar read (§11.4 / §A13)
        int32_t count = expert_count[e];      // GM scalar read (§11.4 / §A13)

        // Per-expert weight pointers; hoisted out of the inner loop.
        GlobalDataB1 b1Global(w1 + static_cast<size_t>(e) * kH * kF);
        GlobalDataB2 b2Global(w2 + static_cast<size_t>(e) * kF * kH);

        for (int32_t m0 = 0; m0 < count; m0 += static_cast<int32_t>(kTileM)) {
            size_t row    = static_cast<size_t>(start) + static_cast<size_t>(m0);
            size_t aOff   = row * kH;   // packed_tokens / packed_output stride
            size_t sOff   = row * kF;   // scratch stride

            GlobalDataA1 a1Global(packed_tokens + aOff);
            GlobalDataC1 c1Global(scratch       + sOff);
            GlobalDataA2 a2Global(scratch       + sOff);
            GlobalDataC2 c2Global(packed_output + aOff);

            // ============================================================
            // GEMM1: packed_tokens (FP16) @ w1[e] (FP16) -> Acc<float> -> ReLU -> scratch (FP16)
            // ============================================================
            TLOAD(a1MatTile, a1Global);
            TLOAD(b1MatTile, b1Global);
            TMOV(a1Tile, a1MatTile);
            TMOV(b1Tile, b1MatTile);
            TMATMUL(c1Tile, a1Tile, b1Tile);

            // L0C -> GM with ReLU AND FP32 -> FP16 downcast fused into the
            // FIX-pipe store. Template args order: <TileData, GlobalData,
            // AtomicType, ReluPreMode>. Same call shape as §A17 except the
            // GM destination dtype is half instead of float. See file
            // header for the "NEW (single isolated assumption)" note.
            TSTORE<AccTile1, GlobalDataC1, AtomicType::AtomicNone,
                   ReluPreMode::NormalRelu>(c1Global, c1Tile);

            // ============================================================
            // GEMM2: scratch (FP16) @ w2[e] (FP16) -> Acc<float> -> packed_output (FP32)
            // ============================================================
            TLOAD(a2MatTile, a2Global);
            TLOAD(b2MatTile, b2Global);
            TMOV(a2Tile, a2MatTile);
            TMOV(b2Tile, b2MatTile);
            TMATMUL(c2Tile, a2Tile, b2Tile);

            // L0C -> GM, plain FP32 store (no ReLU on the FFN output).
            TSTORE(c2Global, c2Tile);
        }
    }
}

template <typename TOut, typename TIn, typename TWeight, typename TScratch>
void launchMoeSegmentedFfnTop1(uint8_t *packed_output,
                               uint8_t *packed_tokens,
                               int32_t *expert_count,
                               int32_t *expert_start,
                               uint8_t *w1,
                               uint8_t *w2,
                               uint8_t *scratch,
                               void    *stream)
{
    runMoeSegmentedFfnTop1<TOut, TIn, TWeight, TScratch><<<1, nullptr, stream>>>(
        packed_output, packed_tokens, expert_count, expert_start,
        w1, w2, scratch);
}

// FP16 x FP16 -> FP32 (output) with FP16 intermediate scratch. Matches the
// dtype contract documented in the file header and README.
template void launchMoeSegmentedFfnTop1<float, half, half, half>(
    uint8_t *packed_output, uint8_t *packed_tokens,
    int32_t *expert_count, int32_t *expert_start,
    uint8_t *w1, uint8_t *w2, uint8_t *scratch, void *stream);

// Non-template host-boundary wrapper. main.cpp cannot see `half` (the host
// TU is compiled with plain `-xc++`, which lacks bisheng-CCE built-ins);
// see compile_error_logbook.md §E13 / known_good_kernel_examples.md §A16.
extern "C" void launchMoeSegmentedFfnTop1Fp16(uint8_t *packed_output,
                                              uint8_t *packed_tokens,
                                              int32_t *expert_count,
                                              int32_t *expert_start,
                                              uint8_t *w1,
                                              uint8_t *w2,
                                              uint8_t *scratch,
                                              void    *stream)
{
    launchMoeSegmentedFfnTop1<float, half, half, half>(
        packed_output, packed_tokens, expert_count, expert_start,
        w1, w2, scratch, stream);
}
