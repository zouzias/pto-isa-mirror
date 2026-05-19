/**
 * mla_basic_cube_kernel.cpp - auto-mode A3 prototype, cube-arch stages.
 *
 * Multi-Head Latent Attention (DeepSeek V2/V3 style) v1 - correctness first,
 * NO double buffering, NO multi-core, NO block_idx, NO TPipe/TPUSH/TPOP,
 * NO TASSIGN, NO manual sync.
 *
 * Cube-arch (`--cce-aicore-arch=dav-c220-cube`) translation unit. Five
 * `__global__ AICORE` entries, each launched independently by the host on the
 * same stream (ACL stream-order is the cross-kernel sync, exactly the §A18
 * recipe in docs_for_ai/known_good_kernel_examples.md).
 *
 * Stage 1 -- runQProjection
 *      Q[B,S,Nh,Hd]   = X[B,S,H]  @  W_q[H, Nh*Hd]
 *      M=128  K=4096  N=4096  (split-K x 64 then split-N x 64)
 *
 * Stage 2 -- runKVCompression
 *      C_kv[B,S,L]    = X[B,S,H]  @  W_dkv[H, L]
 *      M=128  K=4096  N=64   (split-K x 64, single N tile)
 *
 * (Stage 3 -- cache store -- is in the vec TU.)
 *
 * Stage 4 -- runKVReconstruction
 *      K[B,S,Nh,Hd]   = C_cache[B,S,L]  @  W_uk[L, Nh*Hd]
 *      V[B,S,Nh,Hd]   = C_cache[B,S,L]  @  W_uv[L, Nh*Hd]
 *      M=128  K=64  N=4096   (single K tile, split-N x 64) -- twice.
 *
 * Stage 5a -- runAttnQK
 *      scores[h][S,S] = Q_h[S,Hd] @ K_h_T[Hd,S]    per head h in [0, Nh)
 *      K^T view is built from K_h via a GlobalTensor with swapped strides
 *      (row stride 1, col stride Nh*Hd = 4096) -- no physical transpose
 *      kernel needed. See tile_type_reference.md section 1.3 (GlobalTensor
 *      Stride) and the row-softmax tutorial's use of arbitrary 5-D strides.
 *      M=128  K=128  N=128  (split-K x 2, split-N x 2)
 *
 * Stage 5c -- runAttnPV
 *      Out_h[S,Hd]    = probs_h[S,S] @ V_h[S,Hd]    per head h
 *      M=128  K=128  N=128  (split-K x 2, split-N x 2)
 *
 * Tile shapes inside every cube body match the §A16 / §A17 / §A18 references
 * exactly: kTileM=128, kInnerK=64, kInnerN=64, FP16 inputs, FP32 accumulator.
 * The §A18 fused FP32->FP16 TSTORE (without ReLU) is used to write all cube
 * outputs as FP16; this is an Assumption (per CLAUDE.md vocabulary): §A17
 * proves ReLU+downcast, §A18 proves ReLU+downcast in another context, and the
 * underlying TSTORE template covers atomicType+reluPreMode template pairs
 * (include/pto/common/pto_instr.hpp:251-258) -- combining "downcast" with
 * "ReluPreMode::NoRelu" is the natural composition. If the assumption fails
 * at compile or runtime we will add a separate vec downcast kernel.
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>
#include "generated_cases.h"

using namespace pto;

namespace mla_basic_cfg {

// Static configuration. Must match scripts/gen_data.py and main.cpp.
constexpr unsigned kBatch    = 1;
constexpr unsigned kSeqLen   = 128;   // matches kTileM exactly -> single M tile
constexpr unsigned kHidden   = 4096;
constexpr unsigned kNumHeads = 32;
constexpr unsigned kHeadDim  = 128;
constexpr unsigned kLatent   = 64;

constexpr unsigned kTileM    = 128;
constexpr unsigned kInnerK   = 64;
constexpr unsigned kInnerN   = 64;

// Derived
constexpr unsigned kQKVHidden = kNumHeads * kHeadDim;   // 4096
static_assert(kQKVHidden == kHidden, "MLA v1 assumes num_heads*head_dim == hidden");

constexpr unsigned kHiddenKIter = kHidden  / kInnerK;   // 64
constexpr unsigned kQKVNIter    = kQKVHidden / kInnerN; // 64
constexpr unsigned kHeadDimKIter = kHeadDim / kInnerK;  // 2
constexpr unsigned kHeadDimNIter = kHeadDim / kInnerN;  // 2
constexpr unsigned kSeqNIter     = kSeqLen  / kInnerN;  // 2
constexpr unsigned kSeqKIter     = kSeqLen  / kInnerK;  // 2

}  // namespace mla_basic_cfg

// =============================================================================
// Stage 1 -- Query projection: Q = X @ W_q
//   X    : [kSeqLen, kHidden]              half     (M=128, K_full=4096)
//   W_q  : [kHidden, kNumHeads*kHeadDim]   half     (K_full=4096, N_full=4096)
//   Q    : [kSeqLen, kNumHeads, kHeadDim]  half     (M=128, N_full=4096)
//
// Single AICORE. Outer split-N loop (64 iters of 64 cols), inner split-K loop
// (64 iters of 64 cols). Pattern: §A16 moe_segmented_gemm_one_layer +
// §A6 tmatmul SplitK (TMATMUL on i==0, TMATMUL_ACC thereafter).
// =============================================================================
template <typename TIn, typename TWeight, typename TOut>
__global__ AICORE void runQProjection(__gm__ uint8_t *q_raw,
                                      __gm__ uint8_t *x_raw,
                                      __gm__ uint8_t *w_q_raw)
{
    using namespace mla_basic_cfg;

    __gm__ TIn     *x   = reinterpret_cast<__gm__ TIn     *>(x_raw);
    __gm__ TWeight *w_q = reinterpret_cast<__gm__ TWeight *>(w_q_raw);
    __gm__ TOut    *q   = reinterpret_cast<__gm__ TOut    *>(q_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM   + 15) / 16) * 16;                          // 128
    constexpr int K = ((kInnerK  + blockAlign - 1) / blockAlign) * blockAlign; // 64
    constexpr int N = ((kInnerN  + blockAlign - 1) / blockAlign) * blockAlign; // 64

    // GM views sliced down to a (M x K) or (K x N) cube tile. Row stride is
    // always the FULL parent column count -- we are addressing a sub-tile of
    // a wider matrix, not a self-contained tile.
    using GlobalDataA = GlobalTensor<TIn,     Shape<1, 1, 1, kTileM,  kInnerK>,
                                     Stride<1, 1, 1, kHidden,   1>>;
    using GlobalDataB = GlobalTensor<TWeight, Shape<1, 1, 1, kInnerK, kInnerN>,
                                     Stride<1, 1, 1, kQKVHidden, 1>>;
    using GlobalDataC = GlobalTensor<TOut,    Shape<1, 1, 1, kTileM,  kInnerN>,
                                     Stride<1, 1, 1, kQKVHidden, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, K, BLayout::ColMajor,
                              kTileM, kInnerK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K, N, BLayout::ColMajor,
                              kInnerK, kInnerN, SLayout::RowMajor, 512>;
    using LeftTile     = TileLeft <TIn,     M, K, kTileM,  kInnerK>;
    using RightTile    = TileRight<TWeight, K, N, kInnerK, kInnerN>;
    using AccTile      = TileAcc  <float,   M, N, kTileM,  kInnerN>;  // FP32 accumulator

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    for (unsigned nIter = 0; nIter < kQKVNIter; ++nIter) {
        const size_t nOffset = static_cast<size_t>(nIter) * kInnerN;

        for (unsigned kIter = 0; kIter < kHiddenKIter; ++kIter) {
            const size_t kOffset = static_cast<size_t>(kIter) * kInnerK;

            GlobalDataA aGlobal(x   + kOffset);                            // X[:, kOffset:kOffset+64]
            GlobalDataB bGlobal(w_q + kOffset * kQKVHidden + nOffset);     // W_q[kOffset:..,nOffset:..]

            TLOAD(aMatTile, aGlobal);
            TLOAD(bMatTile, bGlobal);
            TMOV (aTile, aMatTile);
            TMOV (bTile, bMatTile);

            if (kIter == 0) {
                TMATMUL    (cTile, aTile, bTile);
            } else {
                TMATMUL_ACC(cTile, aTile, bTile);
            }
        }

        GlobalDataC cGlobal(q + nOffset);
        // Fused FP32 acc -> FP16 GM via FIX pipe (no ReLU). Same template
        // shape as §A18 stage 1 minus the activation. Assumption: this combo
        // works without ReluPreMode::NormalRelu.
        TSTORE<AccTile, GlobalDataC,
               AtomicType::AtomicNone,
               ReluPreMode::NoRelu>(cGlobal, cTile);
    }
}

// =============================================================================
// Stage 2 -- KV compression: C_kv = X @ W_dkv
//   X     : [kSeqLen, kHidden]   half   (M=128, K_full=4096)
//   W_dkv : [kHidden, kLatent]   half   (K_full=4096, N=64)
//   C_kv  : [kSeqLen, kLatent]   half   (M=128, N=64)
//
// Single split-K loop, single N tile (kInnerN = kLatent = 64).
// =============================================================================
template <typename TIn, typename TWeight, typename TOut>
__global__ AICORE void runKVCompression(__gm__ uint8_t *c_kv_raw,
                                        __gm__ uint8_t *x_raw,
                                        __gm__ uint8_t *w_dkv_raw)
{
    using namespace mla_basic_cfg;

    __gm__ TIn     *x     = reinterpret_cast<__gm__ TIn     *>(x_raw);
    __gm__ TWeight *w_dkv = reinterpret_cast<__gm__ TWeight *>(w_dkv_raw);
    __gm__ TOut    *c_kv  = reinterpret_cast<__gm__ TOut    *>(c_kv_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM  + 15) / 16) * 16;
    constexpr int K = ((kInnerK + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kLatent + blockAlign - 1) / blockAlign) * blockAlign;
    static_assert(kLatent == kInnerN, "MLA v1 assumes kLatent == kInnerN so N fits one tile");

    using GlobalDataA = GlobalTensor<TIn,     Shape<1, 1, 1, kTileM,  kInnerK>,
                                     Stride<1, 1, 1, kHidden, 1>>;
    using GlobalDataB = GlobalTensor<TWeight, Shape<1, 1, 1, kInnerK, kLatent>,
                                     Stride<1, 1, 1, kLatent, 1>>;
    using GlobalDataC = GlobalTensor<TOut,    Shape<1, 1, 1, kTileM,  kLatent>,
                                     Stride<1, 1, 1, kLatent, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, K, BLayout::ColMajor,
                              kTileM, kInnerK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K, N, BLayout::ColMajor,
                              kInnerK, kLatent, SLayout::RowMajor, 512>;
    using LeftTile     = TileLeft <TIn,     M, K, kTileM,  kInnerK>;
    using RightTile    = TileRight<TWeight, K, N, kInnerK, kLatent>;
    using AccTile      = TileAcc  <float,   M, N, kTileM,  kLatent>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    for (unsigned kIter = 0; kIter < kHiddenKIter; ++kIter) {
        const size_t kOffset = static_cast<size_t>(kIter) * kInnerK;

        GlobalDataA aGlobal(x     + kOffset);
        GlobalDataB bGlobal(w_dkv + kOffset * kLatent);

        TLOAD(aMatTile, aGlobal);
        TLOAD(bMatTile, bGlobal);
        TMOV (aTile, aMatTile);
        TMOV (bTile, bMatTile);

        if (kIter == 0) {
            TMATMUL    (cTile, aTile, bTile);
        } else {
            TMATMUL_ACC(cTile, aTile, bTile);
        }
    }

    GlobalDataC cGlobal(c_kv);
    TSTORE<AccTile, GlobalDataC,
           AtomicType::AtomicNone,
           ReluPreMode::NoRelu>(cGlobal, cTile);
}

// =============================================================================
// Stage 4 -- KV reconstruction: K = C_cache @ W_uk; V = C_cache @ W_uv
//   C_cache : [kSeqLen, kLatent]              half   (M=128, K=64)
//   W_uk    : [kLatent, kNumHeads*kHeadDim]   half   (K=64,  N_full=4096)
//   W_uv    : [kLatent, kNumHeads*kHeadDim]   half   (same shape as W_uk)
//   K       : [kSeqLen, kNumHeads, kHeadDim]  half
//   V       : [kSeqLen, kNumHeads, kHeadDim]  half
//
// Two GEMMs in a single __global__: the second one starts after the first one
// completes via auto-sync within the same kernel (matches §A18 split-kernel
// approach but consolidated because each GEMM has a single cube-tile working
// set; no L0C reuse hazard like the abandoned §A18 fused-single-kernel form).
// Inside each GEMM: single K tile (kLatent = kInnerK = 64), outer split-N.
// =============================================================================
template <typename TIn, typename TWeight, typename TOut>
__global__ AICORE void runKVReconstruction(__gm__ uint8_t *k_raw,
                                           __gm__ uint8_t *v_raw,
                                           __gm__ uint8_t *c_cache_raw,
                                           __gm__ uint8_t *w_uk_raw,
                                           __gm__ uint8_t *w_uv_raw)
{
    using namespace mla_basic_cfg;

    __gm__ TIn     *c_cache = reinterpret_cast<__gm__ TIn     *>(c_cache_raw);
    __gm__ TWeight *w_uk    = reinterpret_cast<__gm__ TWeight *>(w_uk_raw);
    __gm__ TWeight *w_uv    = reinterpret_cast<__gm__ TWeight *>(w_uv_raw);
    __gm__ TOut    *k       = reinterpret_cast<__gm__ TOut    *>(k_raw);
    __gm__ TOut    *v       = reinterpret_cast<__gm__ TOut    *>(v_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM   + 15) / 16) * 16;
    constexpr int K = ((kLatent  + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kInnerN  + blockAlign - 1) / blockAlign) * blockAlign;

    using GlobalDataA = GlobalTensor<TIn,     Shape<1, 1, 1, kTileM,  kLatent>,
                                     Stride<1, 1, 1, kLatent, 1>>;
    using GlobalDataB = GlobalTensor<TWeight, Shape<1, 1, 1, kLatent, kInnerN>,
                                     Stride<1, 1, 1, kQKVHidden, 1>>;
    using GlobalDataC = GlobalTensor<TOut,    Shape<1, 1, 1, kTileM,  kInnerN>,
                                     Stride<1, 1, 1, kQKVHidden, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, K, BLayout::ColMajor,
                              kTileM, kLatent, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K, N, BLayout::ColMajor,
                              kLatent, kInnerN, SLayout::RowMajor, 512>;
    using LeftTile     = TileLeft <TIn,     M, K, kTileM,  kLatent>;
    using RightTile    = TileRight<TWeight, K, N, kLatent, kInnerN>;
    using AccTile      = TileAcc  <float,   M, N, kTileM,  kInnerN>;

    // A is read-only across both GEMMs (C_cache); B and C are per-GEMM but
    // the tile *shapes* match so we declare them once and reuse. The
    // GlobalTensor base pointer differs per GEMM/per nIter.
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    // --- GEMM 1 : K = C_cache @ W_uk ----------------------------------------
    {
        GlobalDataA aGlobal(c_cache);
        TLOAD(aMatTile, aGlobal);
        TMOV (aTile, aMatTile);

        for (unsigned nIter = 0; nIter < kQKVNIter; ++nIter) {
            const size_t nOffset = static_cast<size_t>(nIter) * kInnerN;

            GlobalDataB bGlobal(w_uk + nOffset);
            TLOAD(bMatTile, bGlobal);
            TMOV (bTile, bMatTile);

            TMATMUL(cTile, aTile, bTile);

            GlobalDataC cGlobal(k + nOffset);
            TSTORE<AccTile, GlobalDataC,
                   AtomicType::AtomicNone,
                   ReluPreMode::NoRelu>(cGlobal, cTile);
        }
    }

    // --- GEMM 2 : V = C_cache @ W_uv ----------------------------------------
    {
        // A (c_cache) is unchanged; reuse aMatTile / aTile contents.
        for (unsigned nIter = 0; nIter < kQKVNIter; ++nIter) {
            const size_t nOffset = static_cast<size_t>(nIter) * kInnerN;

            GlobalDataB bGlobal(w_uv + nOffset);
            TLOAD(bMatTile, bGlobal);
            TMOV (bTile, bMatTile);

            TMATMUL(cTile, aTile, bTile);

            GlobalDataC cGlobal(v + nOffset);
            TSTORE<AccTile, GlobalDataC,
                   AtomicType::AtomicNone,
                   ReluPreMode::NoRelu>(cGlobal, cTile);
        }
    }
}

// =============================================================================
// Stage 5a -- Attention QK^T:
//      scores[h] = Q_h @ K_h^T,  per head h in [0, kNumHeads)
//
//   Q       : [kSeqLen, kNumHeads, kHeadDim]   half
//   K       : [kSeqLen, kNumHeads, kHeadDim]   half
//   scores  : [kNumHeads, kSeqLen, kSeqLen]    half     (FP16 to fit UB)
//
// Per-head GEMM with M=kSeqLen=128, K=kHeadDim=128, N=kSeqLen=128. Split-K
// and split-N both x 2 (kInnerK=kInnerN=64).
//
// K^T view: we want B = K_h^T of shape (kHeadDim, kSeqLen). K_h's storage in
// GM is (kSeqLen rows of stride kQKVHidden, kHeadDim cols of stride 1) at
// base K + h*kHeadDim. Swapping the GlobalTensor strides to (col stride =
// kQKVHidden, row stride = 1) for shape (kHeadDim, kSeqLen) produces the
// transposed read pattern with NO physical transpose pass. The cube TLOAD
// handles arbitrary stride by walking the GM in that pattern.
//
//   row stride 1         -> moving down the "rows" of B advances by 1 byte
//                           (= advances K_h column index by 1)
//   col stride kQKVHidden -> moving across the "cols" of B advances by
//                           kQKVHidden (= advances K_h row index by 1)
// =============================================================================
template <typename TIn, typename TOut>
__global__ AICORE void runAttnQK(__gm__ uint8_t *scores_raw,
                                 __gm__ uint8_t *q_raw,
                                 __gm__ uint8_t *k_raw)
{
    using namespace mla_basic_cfg;

    __gm__ TIn  *q      = reinterpret_cast<__gm__ TIn  *>(q_raw);
    __gm__ TIn  *k      = reinterpret_cast<__gm__ TIn  *>(k_raw);
    __gm__ TOut *scores = reinterpret_cast<__gm__ TOut *>(scores_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM   + 15) / 16) * 16;
    constexpr int K = ((kInnerK  + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kInnerN  + blockAlign - 1) / blockAlign) * blockAlign;

    // A : Q_h sub-tile (kTileM, kInnerK), row stride = kQKVHidden (= 4096).
    using GlobalDataA = GlobalTensor<TIn, Shape<1, 1, 1, kTileM,  kInnerK>,
                                     Stride<1, 1, 1, kQKVHidden, 1>>;
    // B : K_h^T view sub-tile (kInnerK, kInnerN).
    //   Stored in K as [S, Nh, Hd]: element K[s, h, d] = k + s*kQKVHidden + h*kHeadDim + d.
    //   For K_h^T[d, s]: row = d (stride-1 direction), col = s (stride-kQKVHidden direction).
    //   Layout::DN signals the DN (col-major) tensor layout to the DMA, enabling
    //   the DN->ZN TLOAD path (include/pto/npu/a2a3/TLoad.hpp:471).
    //   Matches the flash_atten GlobalDataK pattern (fa_performance_kernel.cpp:161-166).
    using GlobalDataBT = GlobalTensor<TIn, Shape<1, 1, 1, kInnerK, kInnerN>,
                                      Stride<1, 1, 1, 1, kQKVHidden>, Layout::DN>;
    // C : scores_h sub-tile (kTileM, kInnerN), row stride = kSeqLen.
    using GlobalDataC  = GlobalTensor<TOut, Shape<1, 1, 1, kTileM, kInnerN>,
                                      Stride<1, 1, 1, kSeqLen, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn, M, K, BLayout::ColMajor,
                              kTileM, kInnerK, SLayout::RowMajor, 512>;
    // BLayout::RowMajor + SLayout::ColMajor -> GetTileLayoutCustom() = ZN,
    // which pairs with Layout::DN GlobalTensor for the DN->ZN TLOAD path.
    // Mirrors fa_performance_kernel.cpp:444 TileMatKData.
    using TileMatBData = Tile<TileType::Mat, TIn, K, N, BLayout::RowMajor,
                              kInnerK, kInnerN, SLayout::ColMajor, 512>;
    using LeftTile     = TileLeft <TIn,   M, K, kTileM,  kInnerK>;
    using RightTile    = TileRight<TIn,   K, N, kInnerK, kInnerN>;
    using AccTile      = TileAcc  <float, M, N, kTileM,  kInnerN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    for (unsigned h = 0; h < kNumHeads; ++h) {
        const size_t qBase     = static_cast<size_t>(h) * kHeadDim;
        const size_t kBase     = static_cast<size_t>(h) * kHeadDim;
        const size_t scoresBase = static_cast<size_t>(h) * kSeqLen * kSeqLen;

        for (unsigned nIter = 0; nIter < kSeqNIter; ++nIter) {
            const size_t nOffset = static_cast<size_t>(nIter) * kInnerN;

            for (unsigned kIter = 0; kIter < kHeadDimKIter; ++kIter) {
                const size_t kOffset = static_cast<size_t>(kIter) * kInnerK;

                GlobalDataA  aGlobal(q + qBase + kOffset);
                // K^T view: outer kOffset is rows-of-B (= cols-of-K_h, stride 1).
                //           outer nOffset is cols-of-B (= rows-of-K_h, stride kQKVHidden).
                GlobalDataBT bGlobal(k + kBase + kOffset
                                       + nOffset * kQKVHidden);

                TLOAD(aMatTile, aGlobal);
                TLOAD(bMatTile, bGlobal);
                TMOV (aTile, aMatTile);
                TMOV (bTile, bMatTile);

                if (kIter == 0) {
                    TMATMUL    (cTile, aTile, bTile);
                } else {
                    TMATMUL_ACC(cTile, aTile, bTile);
                }
            }

            GlobalDataC cGlobal(scores + scoresBase + nOffset);
            TSTORE<AccTile, GlobalDataC,
                   AtomicType::AtomicNone,
                   ReluPreMode::NoRelu>(cGlobal, cTile);
        }
    }
}

// =============================================================================
// Stage 5c -- Attention PV:
//      Out_h = probs_h @ V_h,   per head h
//
//   probs : [kNumHeads, kSeqLen, kSeqLen]    half     (from softmax stage)
//   V     : [kSeqLen, kNumHeads, kHeadDim]   half
//   Out   : [kSeqLen, kNumHeads, kHeadDim]   half     (re-interleaved per-head)
//
// Per-head GEMM with M=kSeqLen=128, K=kSeqLen=128, N=kHeadDim=128. Split-K
// and split-N both x 2.
// =============================================================================
template <typename TIn, typename TOut>
__global__ AICORE void runAttnPV(__gm__ uint8_t *out_raw,
                                 __gm__ uint8_t *probs_raw,
                                 __gm__ uint8_t *v_raw)
{
    using namespace mla_basic_cfg;

    __gm__ TIn  *probs = reinterpret_cast<__gm__ TIn  *>(probs_raw);
    __gm__ TIn  *v     = reinterpret_cast<__gm__ TIn  *>(v_raw);
    __gm__ TOut *out   = reinterpret_cast<__gm__ TOut *>(out_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM   + 15) / 16) * 16;
    constexpr int K = ((kInnerK  + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kInnerN  + blockAlign - 1) / blockAlign) * blockAlign;

    // A : probs_h sub-tile (kTileM, kInnerK), row stride = kSeqLen.
    using GlobalDataA = GlobalTensor<TIn,  Shape<1, 1, 1, kTileM, kInnerK>,
                                     Stride<1, 1, 1, kSeqLen, 1>>;
    // B : V_h sub-tile (kInnerK, kInnerN), row stride = kQKVHidden.
    using GlobalDataB = GlobalTensor<TIn,  Shape<1, 1, 1, kInnerK, kInnerN>,
                                     Stride<1, 1, 1, kQKVHidden, 1>>;
    // C : Out_h sub-tile (kTileM, kInnerN), row stride = kQKVHidden.
    using GlobalDataC = GlobalTensor<TOut, Shape<1, 1, 1, kTileM,  kInnerN>,
                                     Stride<1, 1, 1, kQKVHidden, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn, M, K, BLayout::ColMajor,
                              kTileM, kInnerK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TIn, K, N, BLayout::ColMajor,
                              kInnerK, kInnerN, SLayout::RowMajor, 512>;
    using LeftTile     = TileLeft <TIn,   M, K, kTileM,  kInnerK>;
    using RightTile    = TileRight<TIn,   K, N, kInnerK, kInnerN>;
    using AccTile      = TileAcc  <float, M, N, kTileM,  kInnerN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    for (unsigned h = 0; h < kNumHeads; ++h) {
        const size_t probsBase = static_cast<size_t>(h) * kSeqLen * kSeqLen;
        const size_t vBase     = static_cast<size_t>(h) * kHeadDim;
        const size_t outBase   = static_cast<size_t>(h) * kHeadDim;

        for (unsigned nIter = 0; nIter < kHeadDimNIter; ++nIter) {
            const size_t nOffset = static_cast<size_t>(nIter) * kInnerN;

            for (unsigned kIter = 0; kIter < kSeqKIter; ++kIter) {
                const size_t kOffset = static_cast<size_t>(kIter) * kInnerK;

                GlobalDataA aGlobal(probs + probsBase + kOffset);
                GlobalDataB bGlobal(v     + vBase
                                          + kOffset * kQKVHidden
                                          + nOffset);

                TLOAD(aMatTile, aGlobal);
                TLOAD(bMatTile, bGlobal);
                TMOV (aTile, aMatTile);
                TMOV (bTile, bMatTile);

                if (kIter == 0) {
                    TMATMUL    (cTile, aTile, bTile);
                } else {
                    TMATMUL_ACC(cTile, aTile, bTile);
                }
            }

            GlobalDataC cGlobal(out + outBase + nOffset);
            TSTORE<AccTile, GlobalDataC,
                   AtomicType::AtomicNone,
                   ReluPreMode::NoRelu>(cGlobal, cTile);
        }
    }
}

// ----------------------------------------------------------------------------
// Templated host-side launchers + explicit instantiations + non-template
// `…Fp16` wrappers that hide `half` from main.cpp.
// (Compile error pattern §E13 in docs_for_ai/compile_error_logbook.md: host
// `-xc++` TU cannot name `half`; expose only uint8_t* / int32_t* across the
// boundary.)
// ----------------------------------------------------------------------------

template <typename TIn, typename TWeight, typename TOut>
void launchQProjection(uint8_t *q, uint8_t *x, uint8_t *w_q, void *stream)
{
    runQProjection<TIn, TWeight, TOut><<<1, nullptr, stream>>>(q, x, w_q);
}

template <typename TIn, typename TWeight, typename TOut>
void launchKVCompression(uint8_t *c_kv, uint8_t *x, uint8_t *w_dkv, void *stream)
{
    runKVCompression<TIn, TWeight, TOut><<<1, nullptr, stream>>>(c_kv, x, w_dkv);
}

template <typename TIn, typename TWeight, typename TOut>
void launchKVReconstruction(uint8_t *k, uint8_t *v,
                            uint8_t *c_cache, uint8_t *w_uk, uint8_t *w_uv,
                            void *stream)
{
    runKVReconstruction<TIn, TWeight, TOut><<<1, nullptr, stream>>>(
        k, v, c_cache, w_uk, w_uv);
}

template <typename TIn, typename TOut>
void launchAttnQK(uint8_t *scores, uint8_t *q, uint8_t *k, void *stream)
{
    runAttnQK<TIn, TOut><<<1, nullptr, stream>>>(scores, q, k);
}

template <typename TIn, typename TOut>
void launchAttnPV(uint8_t *out, uint8_t *probs, uint8_t *v, void *stream)
{
    runAttnPV<TIn, TOut><<<1, nullptr, stream>>>(out, probs, v);
}

template void launchQProjection<half, half, half>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchKVCompression<half, half, half>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchKVReconstruction<half, half, half>(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, void *);
template void launchAttnQK<half, half>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchAttnPV<half, half>(uint8_t *, uint8_t *, uint8_t *, void *);

// Non-template wrappers consumed by main.cpp.
extern "C" void launchMlaQProjectionFp16(uint8_t *q, uint8_t *x, uint8_t *w_q, void *stream)
{
    launchQProjection<half, half, half>(q, x, w_q, stream);
}

extern "C" void launchMlaKVCompressionFp16(uint8_t *c_kv, uint8_t *x, uint8_t *w_dkv, void *stream)
{
    launchKVCompression<half, half, half>(c_kv, x, w_dkv, stream);
}

extern "C" void launchMlaKVReconstructionFp16(uint8_t *k, uint8_t *v,
                                              uint8_t *c_cache,
                                              uint8_t *w_uk, uint8_t *w_uv,
                                              void *stream)
{
    launchKVReconstruction<half, half, half>(k, v, c_cache, w_uk, w_uv, stream);
}

extern "C" void launchMlaAttnQKFp16(uint8_t *scores, uint8_t *q, uint8_t *k, void *stream)
{
    launchAttnQK<half, half>(scores, q, k, stream);
}

extern "C" void launchMlaAttnPVFp16(uint8_t *out, uint8_t *probs, uint8_t *v, void *stream)
{
    launchAttnPV<half, half>(out, probs, v, stream);
}
