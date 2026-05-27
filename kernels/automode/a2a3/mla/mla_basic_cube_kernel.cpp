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
 * DeepSeek-V2 MLA path: Q is compressed (W_dq) and reconstructed (W_uq);
 * head_dim is partitioned into nope_dim + rope_dim. Stages 5d/5e/5f and the
 * vec runRoPE add the decoupled RoPE branch.
 *
 * Stage 1a -- runQCompression
 *      C_q[B,S,qL]    = X[B,S,H]  @  W_dq[H, qL]              qL=kQLatent=64
 *      M=128  K=4096  N=64   (split-K x 64, single N tile)
 *
 * Stage 1b -- runQAbsorb  (DeepSeek-V2 weight absorption)
 *      Q_absorbed[B,S,Nh,L] = C_q[B,S,qL]  @  W_qk[qL, Nh*L]
 *      where W_qk[h] = W_uq[h] @ W_uk[h]^T  is precomputed on the host.
 *      M=128  K=64  N=2048   (single K tile, split-N x 32)
 *
 * Stage 2 -- runKVCompression
 *      C_kv[B,S,L]    = X[B,S,H]  @  W_dkv[H, L]
 *      M=128  K=4096  N=64   (split-K x 64, single N tile)
 *
 * (Stage 3 -- cache store -- is in the vec TU.)
 *
 * Stage 4 -- runVReconstruction  (K reconstruction is GONE — absorbed into W_qk)
 *      V[B,S,Nh,head_d] = C_cache[B,S,L]  @  W_uv[L, Nh*head_d]
 *      M=128  K=64  N=4096
 *
 * Stage 5a -- runAttnQK (nope branch, absorbed)
 *      scores_nope[h][S,S] = Q_absorbed_h[S,L] @ C_cache^T[L,S]
 *      C_cache is SHARED across heads (no per-head offset on B).
 *      K^T view via Layout::DN + ZN tile (no physical transpose).
 *      M=128  K=64  N=128  (single K iter, split-N x 2)
 *
 * Stage 5c -- runAttnPV
 *      Out_h[S,head_d] = probs_h[S,S] @ V_h[S,head_d]
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

#include "generated_cases.h"   // emitted by scripts/generate_cases.py

using namespace pto;

namespace mla_basic_cfg {

// Static configuration. Sourced from the case selected at build time;
// generate_cases.py + run.sh control which case the binary is built against.
constexpr unsigned kBatch    = 1;
constexpr unsigned kSeqLen   = kMlaSeqLen;
constexpr unsigned kHidden   = kMlaHidden;
constexpr unsigned kNumHeads = kMlaNumHeads;
constexpr unsigned kHeadDim  = kMlaHeadDim;    // total per-head dim (nope + rope) for V/Out
constexpr unsigned kNopeDim  = kMlaNopeDim;    // = kHeadDim - kRopeDim
constexpr unsigned kRopeDim  = kMlaRopeDim;
constexpr unsigned kLatent   = kMlaLatent;
constexpr unsigned kQLatent  = kMlaQLatent;

constexpr unsigned kTileM    = 128;
constexpr unsigned kInnerK   = 64;
constexpr unsigned kInnerN   = 64;

// Multi-core launch dimension. A3 (Ascend 910B1) has 24 cube AI cores;
// gemm/flash_atten use 24 too. Each kernel uses `get_block_idx()` to claim a
// stride-`kBlockDim` slice of its output work items. Kernels with fewer than
// `kBlockDim` work items leave the remaining cores idle (cheap).
constexpr unsigned kBlockDim = 24;

// Derived
constexpr unsigned kQKVHidden   = kNumHeads * kHeadDim;   // 4096  (V row stride; Out row stride)
constexpr unsigned kQNopeWidth  = kNumHeads * kNopeDim;   // 2048  (Q_nope, K_nope row stride)
constexpr unsigned kQRopeWidth  = kNumHeads * kRopeDim;   // 2048  (flat W_q_rope width)
static_assert(kQKVHidden == kHidden,           "MLA assumes num_heads*head_dim == hidden");
static_assert(kNopeDim + kRopeDim == kHeadDim, "head_dim must split into nope+rope");
static_assert(kNopeDim == kInnerK,             "Assumes nope dim fits a single inner-K tile");
static_assert(kRopeDim == kInnerN,             "Assumes rope dim fits a single inner-N tile");

constexpr unsigned kHiddenKIter   = kHidden     / kInnerK;   // 64
constexpr unsigned kQKVNIter      = kQKVHidden  / kInnerN;   // 64  (V-side N iters)
constexpr unsigned kQNopeNIter    = kQNopeWidth / kInnerN;   // 32  (K_nope, Q_nope reconstruction)
constexpr unsigned kQRopeNIter    = kQRopeWidth / kInnerN;   // 32  (== kNumHeads)
constexpr unsigned kHeadDimNIter  = kHeadDim    / kInnerN;   // 2   (runAttnPV N iters)
constexpr unsigned kSeqNIter      = kSeqLen     / kInnerN;   // S / 64
constexpr unsigned kSeqKIter      = kSeqLen     / kInnerK;   // S / 64
constexpr unsigned kSeqMIter      = kSeqLen     / kTileM;    // S / 128  (outer M-chunk loop)
static_assert(kSeqLen % kTileM == 0,
              "kSeqLen must be a multiple of kTileM (cube M-axis chunk size)");

}  // namespace mla_basic_cfg

// =============================================================================
// Stage 1a (DeepSeek-V2) -- Q compression: C_q = X @ W_dq
//   X    : [kSeqLen, kHidden]                  half     (M=128, K_full=4096)
//   W_dq : [kHidden, kQLatent]                 half     (K_full=4096, N=64)
//   C_q  : [kSeqLen, kQLatent]                 half     (M=128, N=64)
//
// Structurally identical to runKVCompression (single N tile, split-K x 64).
// =============================================================================
template <typename TIn, typename TWeight, typename TOut>
__global__ AICORE void runQCompression(__gm__ uint8_t *c_q_raw,
                                       __gm__ uint8_t *x_raw,
                                       __gm__ uint8_t *w_dq_raw)
{
    using namespace mla_basic_cfg;

    __gm__ TIn     *x    = reinterpret_cast<__gm__ TIn     *>(x_raw);
    __gm__ TWeight *w_dq = reinterpret_cast<__gm__ TWeight *>(w_dq_raw);
    __gm__ TOut    *c_q  = reinterpret_cast<__gm__ TOut    *>(c_q_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM   + 15) / 16) * 16;
    constexpr int K = ((kInnerK  + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kQLatent + blockAlign - 1) / blockAlign) * blockAlign;
    static_assert(kQLatent == kInnerN, "MLA assumes q_latent == kInnerN so N fits one tile");

    using GlobalDataA = GlobalTensor<TIn,     Shape<1, 1, 1, kTileM,  kInnerK>,
                                     Stride<1, 1, 1, kHidden,  1>>;
    using GlobalDataB = GlobalTensor<TWeight, Shape<1, 1, 1, kInnerK, kQLatent>,
                                     Stride<1, 1, 1, kQLatent, 1>>;
    using GlobalDataC = GlobalTensor<TOut,    Shape<1, 1, 1, kTileM,  kQLatent>,
                                     Stride<1, 1, 1, kQLatent, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, K, BLayout::ColMajor,
                              kTileM,  kInnerK,  SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K, N, BLayout::ColMajor,
                              kInnerK, kQLatent, SLayout::RowMajor, 512>;
    using LeftTile     = TileLeft <TIn,     M, K, kTileM,  kInnerK>;
    using RightTile    = TileRight<TWeight, K, N, kInnerK, kQLatent>;
    using AccTile      = TileAcc  <float,   M, N, kTileM,  kQLatent>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    // Multi-core: each core takes a strided slice of the M-chunks
    // (kSeqMIter chunks of kTileM rows each). Splitting the K-axis would
    // require atomic-add TSTORE which we skip here.
    const unsigned core_id = get_block_idx();

    for (unsigned mIter = core_id; mIter < kSeqMIter; mIter += kBlockDim) {
        const size_t mOffset = static_cast<size_t>(mIter) * kTileM;

        for (unsigned kIter = 0; kIter < kHiddenKIter; ++kIter) {
            const size_t kOffset = static_cast<size_t>(kIter) * kInnerK;

            GlobalDataA aGlobal(x    + mOffset * kHidden + kOffset);
            GlobalDataB bGlobal(w_dq + kOffset * kQLatent);

            TLOAD(aMatTile, aGlobal);
            TLOAD(bMatTile, bGlobal);
            TMOV (aTile, aMatTile);
            TMOV (bTile, bMatTile);

            if (kIter == 0) { TMATMUL    (cTile, aTile, bTile); }
            else            { TMATMUL_ACC(cTile, aTile, bTile); }
        }

        GlobalDataC cGlobal(c_q + mOffset * kQLatent);
        TSTORE<AccTile, GlobalDataC,
               AtomicType::AtomicNone,
               ReluPreMode::NoRelu>(cGlobal, cTile);
    }
}

// =============================================================================
// Stage 1b (DeepSeek-V2, weight-absorbed) -- Q absorption: Q_absorbed = C_q @ W_qk
//   C_q        : [kSeqLen, kQLatent]                  half  (M=128, K=64)
//   W_qk       : [kQLatent, kNumHeads*kLatent]        half  (K=64, N_full=2048)
//   Q_absorbed : [kSeqLen, kNumHeads, kLatent]        half  (M=128, N_full=2048)
//
// W_qk is the host-precomputed absorbed weight:  W_qk[h] = W_uq[h] @ W_uk[h]^T,
// shape [qL, L] per head, packed to [qL, Nh*L]. This eliminates the separate
// K_nope reconstruction:  Q_nope @ K_nope^T == C_q @ W_qk @ C_kv^T  exactly
// (matrix associativity).
//
// Shape-wise this kernel is identical to the previous Q-reconstruction GEMM
// (kQLatent == kInnerK == 64, output width kNumHeads*kLatent == 2048).
// =============================================================================
template <typename TIn, typename TWeight, typename TOut>
__global__ AICORE void runQAbsorb(__gm__ uint8_t *q_absorbed_raw,
                                  __gm__ uint8_t *c_q_raw,
                                  __gm__ uint8_t *w_qk_raw)
{
    using namespace mla_basic_cfg;

    __gm__ TIn     *c_q        = reinterpret_cast<__gm__ TIn     *>(c_q_raw);
    __gm__ TWeight *w_qk       = reinterpret_cast<__gm__ TWeight *>(w_qk_raw);
    __gm__ TOut    *q_absorbed = reinterpret_cast<__gm__ TOut    *>(q_absorbed_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM   + 15) / 16) * 16;
    constexpr int K = ((kQLatent + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kInnerN  + blockAlign - 1) / blockAlign) * blockAlign;

    // Output column count = kNumHeads * kLatent  (== kQNopeWidth since
    // L == Nope in the current config; keeping kQNopeWidth as the alias).
    using GlobalDataA = GlobalTensor<TIn,     Shape<1, 1, 1, kTileM,   kQLatent>,
                                     Stride<1, 1, 1, kQLatent,    1>>;
    using GlobalDataB = GlobalTensor<TWeight, Shape<1, 1, 1, kQLatent, kInnerN>,
                                     Stride<1, 1, 1, kQNopeWidth, 1>>;
    using GlobalDataC = GlobalTensor<TOut,    Shape<1, 1, 1, kTileM,   kInnerN>,
                                     Stride<1, 1, 1, kQNopeWidth, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, K, BLayout::ColMajor,
                              kTileM,   kQLatent, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K, N, BLayout::ColMajor,
                              kQLatent, kInnerN,  SLayout::RowMajor, 512>;
    using LeftTile     = TileLeft <TIn,     M, K, kTileM,   kQLatent>;
    using RightTile    = TileRight<TWeight, K, N, kQLatent, kInnerN>;
    using AccTile      = TileAcc  <float,   M, N, kTileM,   kInnerN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    // Multi-core: flat (mIter, nIter) work item -> one [kTileM, kInnerN] tile.
    const unsigned core_id   = get_block_idx();
    constexpr unsigned kWork = kSeqMIter * kQNopeNIter;

    for (unsigned w = core_id; w < kWork; w += kBlockDim) {
        const unsigned mIter = w / kQNopeNIter;
        const unsigned nIter = w % kQNopeNIter;
        const size_t mOffset = static_cast<size_t>(mIter) * kTileM;
        const size_t nOffset = static_cast<size_t>(nIter) * kInnerN;

        GlobalDataA aGlobal(c_q + mOffset * kQLatent);
        TLOAD(aMatTile, aGlobal);
        TMOV (aTile, aMatTile);

        GlobalDataB bGlobal(w_qk + nOffset);
        TLOAD(bMatTile, bGlobal);
        TMOV (bTile, bMatTile);

        TMATMUL(cTile, aTile, bTile);

        GlobalDataC cGlobal(q_absorbed + mOffset * kQNopeWidth + nOffset);
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

    // Multi-core: M-chunks of kTileM rows, strided across cores.
    const unsigned core_id = get_block_idx();

    for (unsigned mIter = core_id; mIter < kSeqMIter; mIter += kBlockDim) {
        const size_t mOffset = static_cast<size_t>(mIter) * kTileM;

        for (unsigned kIter = 0; kIter < kHiddenKIter; ++kIter) {
            const size_t kOffset = static_cast<size_t>(kIter) * kInnerK;

            GlobalDataA aGlobal(x     + mOffset * kHidden + kOffset);
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

        GlobalDataC cGlobal(c_kv + mOffset * kLatent);
        TSTORE<AccTile, GlobalDataC,
               AtomicType::AtomicNone,
               ReluPreMode::NoRelu>(cGlobal, cTile);
    }
}

// =============================================================================
// Stage 4 -- V reconstruction (DeepSeek-V2, weight-absorbed):
//   V = C_cache @ W_uv     -> [S, Nh*kHeadDim]
//
//   C_cache : [kSeqLen, kLatent]                half  (M=128, K=64)
//   W_uv    : [kLatent, kNumHeads*kHeadDim]     half  (K=64, N_full=4096)
//   V       : [kSeqLen, kNumHeads, kHeadDim]    half
//
// K reconstruction is GONE — the W_uk^T factor has been absorbed into W_qk
// (see Stage 1b). The attention QK^T kernel now reads C_cache directly.
// V is still materialised because the PV stage still consumes it as-is.
// Single K iter (kLatent == kInnerK == 64), split-N x kQKVNIter (= 64 at H=4096).
// =============================================================================
template <typename TIn, typename TWeight, typename TOut>
__global__ AICORE void runVReconstruction(__gm__ uint8_t *v_raw,
                                          __gm__ uint8_t *c_cache_raw,
                                          __gm__ uint8_t *w_uv_raw)
{
    using namespace mla_basic_cfg;

    __gm__ TIn     *c_cache = reinterpret_cast<__gm__ TIn     *>(c_cache_raw);
    __gm__ TWeight *w_uv    = reinterpret_cast<__gm__ TWeight *>(w_uv_raw);
    __gm__ TOut    *v       = reinterpret_cast<__gm__ TOut    *>(v_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM   + 15) / 16) * 16;
    constexpr int K = ((kLatent  + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kInnerN  + blockAlign - 1) / blockAlign) * blockAlign;

    using GlobalDataA = GlobalTensor<TIn, Shape<1, 1, 1, kTileM, kLatent>,
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

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    // Multi-core: flat (mIter, nIter) -> one [kTileM, kInnerN] V tile.
    const unsigned core_id   = get_block_idx();
    constexpr unsigned kWork = kSeqMIter * kQKVNIter;

    for (unsigned w = core_id; w < kWork; w += kBlockDim) {
        const unsigned mIter = w / kQKVNIter;
        const unsigned nIter = w % kQKVNIter;
        const size_t mOffset = static_cast<size_t>(mIter) * kTileM;
        const size_t nOffset = static_cast<size_t>(nIter) * kInnerN;

        GlobalDataA aGlobal(c_cache + mOffset * kLatent);
        TLOAD(aMatTile, aGlobal);
        TMOV (aTile, aMatTile);

        GlobalDataB bGlobal(w_uv + nOffset);
        TLOAD(bMatTile, bGlobal);
        TMOV (bTile, bMatTile);

        TMATMUL(cTile, aTile, bTile);

        GlobalDataC cGlobal(v + mOffset * kQKVHidden + nOffset);
        TSTORE<AccTile, GlobalDataC,
               AtomicType::AtomicNone,
               ReluPreMode::NoRelu>(cGlobal, cTile);
    }
}

// =============================================================================
// Stage 5a -- Attention QK^T (DeepSeek-V2 nope branch):
//      scores_nope[h] = Q_nope_h @ K_nope_h^T,  per head h in [0, kNumHeads)
//
//   Q_nope     : [kSeqLen, kNumHeads, kNopeDim]    half
//   K_nope     : [kSeqLen, kNumHeads, kNopeDim]    half
//   scores_nope: [kNumHeads, kSeqLen, kSeqLen]     half     (FP16 to fit UB)
//
// Per-head GEMM with M=kSeqLen=128, K=kNopeDim=64, N=kSeqLen=128. Since
// kNopeDim == kInnerK, there is a SINGLE K iter (no TMATMUL_ACC). Split-N
// stays at 2 (kSeqNIter).
//
// Row stride changes from kQKVHidden (4096, was head_dim layout) to
// kQNopeWidth (2048, new nope-only width). Per-head base offset is h*kNopeDim
// (was h*kHeadDim).
//
// K^T view (DeepSeek-V2, weight-absorbed): A is the per-head Q_absorbed slice,
// B is C_cache^T — SHARED across heads (no per-head offset, col stride kLatent
// instead of kQNopeWidth).
// =============================================================================
template <typename TIn, typename TOut>
__global__ AICORE void runAttnQK(__gm__ uint8_t *scores_raw,
                                 __gm__ uint8_t *q_absorbed_raw,
                                 __gm__ uint8_t *c_cache_raw)
{
    using namespace mla_basic_cfg;

    __gm__ TIn  *q_absorbed = reinterpret_cast<__gm__ TIn  *>(q_absorbed_raw);
    __gm__ TIn  *c_cache    = reinterpret_cast<__gm__ TIn  *>(c_cache_raw);
    __gm__ TOut *scores     = reinterpret_cast<__gm__ TOut *>(scores_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM   + 15) / 16) * 16;
    constexpr int K = ((kInnerK  + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kInnerN  + blockAlign - 1) / blockAlign) * blockAlign;

    // A : Q_absorbed_h sub-tile (kTileM, kLatent==kInnerK), row stride = kQNopeWidth.
    //     Q_absorbed layout [S, Nh, kLatent] (== old Q_nope layout).
    using GlobalDataA = GlobalTensor<TIn, Shape<1, 1, 1, kTileM,  kInnerK>,
                                     Stride<1, 1, 1, kQNopeWidth, 1>>;
    // B : C_cache^T view sub-tile (kInnerK, kInnerN).
    //   C_cache storage: [S, kLatent] (NO head dim — shared across heads).
    //   C_cache^T[d, s] base = c_cache + 0; row stride 1 (d direction),
    //   col stride kLatent (s direction). Layout::DN + ZN tile.
    //   ** Key change vs old K_nope^T: col stride is kLatent=64 not kQNopeWidth=2048,
    //   ** and there is NO per-head base offset.
    using GlobalDataBT = GlobalTensor<TIn, Shape<1, 1, 1, kInnerK, kInnerN>,
                                      Stride<1, 1, 1, 1, kLatent>, Layout::DN>;
    // C : scores_h sub-tile (kTileM, kInnerN), row stride = kSeqLen.
    using GlobalDataC  = GlobalTensor<TOut, Shape<1, 1, 1, kTileM, kInnerN>,
                                      Stride<1, 1, 1, kSeqLen, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn, M, K, BLayout::ColMajor,
                              kTileM, kInnerK, SLayout::RowMajor, 512>;
    // BLayout::RowMajor + SLayout::ColMajor -> ZN, pairs with Layout::DN.
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

    // Multi-core: flatten (h, mIter, nIter) work-item space and stride
    // across cores. Each work item is one [kTileM, kInnerN] score output.
    const unsigned core_id   = get_block_idx();
    constexpr unsigned kWork = kNumHeads * kSeqMIter * kSeqNIter;

    for (unsigned w = core_id; w < kWork; w += kBlockDim) {
        const unsigned h     = w / (kSeqMIter * kSeqNIter);
        const unsigned rem   = w % (kSeqMIter * kSeqNIter);
        const unsigned mIter = rem / kSeqNIter;
        const unsigned nIter = rem % kSeqNIter;

        const size_t qBase      = static_cast<size_t>(h) * kLatent;
        const size_t scoresBase = static_cast<size_t>(h) * kSeqLen * kSeqLen;
        const size_t mOffset    = static_cast<size_t>(mIter) * kTileM;
        const size_t nOffset    = static_cast<size_t>(nIter) * kInnerN;

        // Single K iter: kLatent == kInnerK == 64.
        // A: Q_absorbed sub-tile starting at row mOffset of head h.
        GlobalDataA  aGlobal(q_absorbed + mOffset * kQNopeWidth + qBase);
        // B: C_cache^T sub-tile (shared across heads — no per-head offset).
        //    Column index in the transposed view = j (sequence), starting at nOffset.
        GlobalDataBT bGlobal(c_cache + nOffset * kLatent);

        TLOAD(aMatTile, aGlobal);
        TLOAD(bMatTile, bGlobal);
        TMOV (aTile, aMatTile);
        TMOV (bTile, bMatTile);

        TMATMUL(cTile, aTile, bTile);

        // scores[h, mOffset:mOffset+M, nOffset:nOffset+N]; row stride kSeqLen.
        GlobalDataC cGlobal(scores + scoresBase + mOffset * kSeqLen + nOffset);
        TSTORE<AccTile, GlobalDataC,
               AtomicType::AtomicNone,
               ReluPreMode::NoRelu>(cGlobal, cTile);
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

    // Multi-core: flatten (h, mIter, nIter) — each work item is one
    // [kTileM, kInnerN] output tile. Inner K-iter loop (kSeqKIter) still
    // does TMATMUL + TMATMUL_ACC because K=kSeqLen > kInnerK.
    const unsigned core_id   = get_block_idx();
    constexpr unsigned kWork = kNumHeads * kSeqMIter * kHeadDimNIter;

    for (unsigned w = core_id; w < kWork; w += kBlockDim) {
        const unsigned h     = w / (kSeqMIter * kHeadDimNIter);
        const unsigned rem   = w % (kSeqMIter * kHeadDimNIter);
        const unsigned mIter = rem / kHeadDimNIter;
        const unsigned nIter = rem % kHeadDimNIter;

        const size_t probsBase = static_cast<size_t>(h) * kSeqLen * kSeqLen;
        const size_t vBase     = static_cast<size_t>(h) * kHeadDim;
        const size_t outBase   = static_cast<size_t>(h) * kHeadDim;
        const size_t mOffset   = static_cast<size_t>(mIter) * kTileM;
        const size_t nOffset   = static_cast<size_t>(nIter) * kInnerN;

        for (unsigned kIter = 0; kIter < kSeqKIter; ++kIter) {
            const size_t kOffset = static_cast<size_t>(kIter) * kInnerK;

            // A: probs[h, mOffset:mOffset+M, kOffset:kOffset+K]
            GlobalDataA aGlobal(probs + probsBase + mOffset * kSeqLen + kOffset);
            // B: V[kOffset:kOffset+K, h, nOffset:nOffset+N]  (row stride kQKVHidden)
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

        GlobalDataC cGlobal(out + outBase + mOffset * kQKVHidden + nOffset);
        TSTORE<AccTile, GlobalDataC,
               AtomicType::AtomicNone,
               ReluPreMode::NoRelu>(cGlobal, cTile);
    }
}

// =============================================================================
// Stage 5d (NEW) -- Q_rope projection (DeepSeek-V2 decoupled RoPE).
//   X        : [kSeqLen, kHidden]                   half     (M=128, K_full=4096)
//   W_q_rope : [kHidden, kNumHeads*kRopeDim]        half     (K_full=4096, N_full=2048)
//   Q_rope   : [kNumHeads, kSeqLen, kRopeDim]       half     <-- head-major layout
//
// The GEMM naturally produces cols of [S, Nh*Rd]. Each nIter covers
// kInnerN=kRopeDim=64 cols, which is exactly one head's worth. So we
// reinterpret nIter as a head index and write the tile to
//   q_rope + head_h * kSeqLen * kRopeDim
// with row stride kRopeDim. This puts Q_rope into [Nh, S, Rd] head-major
// layout *during the GEMM store*, no separate transpose kernel needed.
// =============================================================================
template <typename TIn, typename TWeight, typename TOut>
__global__ AICORE void runQRopeProjection(__gm__ uint8_t *q_rope_raw,
                                          __gm__ uint8_t *x_raw,
                                          __gm__ uint8_t *w_q_rope_raw)
{
    using namespace mla_basic_cfg;

    __gm__ TIn     *x        = reinterpret_cast<__gm__ TIn     *>(x_raw);
    __gm__ TWeight *w_q_rope = reinterpret_cast<__gm__ TWeight *>(w_q_rope_raw);
    __gm__ TOut    *q_rope   = reinterpret_cast<__gm__ TOut    *>(q_rope_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM  + 15) / 16) * 16;
    constexpr int K = ((kInnerK + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kInnerN + blockAlign - 1) / blockAlign) * blockAlign;

    using GlobalDataA = GlobalTensor<TIn,     Shape<1, 1, 1, kTileM,  kInnerK>,
                                     Stride<1, 1, 1, kHidden, 1>>;
    using GlobalDataB = GlobalTensor<TWeight, Shape<1, 1, 1, kInnerK, kInnerN>,
                                     Stride<1, 1, 1, kQRopeWidth, 1>>;
    // C: head-major destination. Row stride = kRopeDim (NOT kQRopeWidth);
    // base pointer carries the per-head offset.
    using GlobalDataC = GlobalTensor<TOut,    Shape<1, 1, 1, kTileM,  kInnerN>,
                                     Stride<1, 1, 1, kRopeDim, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, K, BLayout::ColMajor,
                              kTileM, kInnerK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K, N, BLayout::ColMajor,
                              kInnerK, kInnerN, SLayout::RowMajor, 512>;
    using LeftTile     = TileLeft <TIn,     M, K, kTileM,  kInnerK>;
    using RightTile    = TileRight<TWeight, K, N, kInnerK, kInnerN>;
    using AccTile      = TileAcc  <float,   M, N, kTileM,  kInnerN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    // Multi-core: flatten (mIter, nIter) — nIter == head index. Each work
    // item is one [kTileM, kRopeDim] sub-tile of Q_rope (head-major).
    const unsigned core_id   = get_block_idx();
    constexpr unsigned kWork = kSeqMIter * kQRopeNIter;

    for (unsigned w = core_id; w < kWork; w += kBlockDim) {
        const unsigned mIter = w / kQRopeNIter;
        const unsigned nIter = w % kQRopeNIter;     // == head index
        const size_t mOffset = static_cast<size_t>(mIter) * kTileM;
        const size_t nOffset = static_cast<size_t>(nIter) * kInnerN;

        for (unsigned kIter = 0; kIter < kHiddenKIter; ++kIter) {
            const size_t kOffset = static_cast<size_t>(kIter) * kInnerK;

            GlobalDataA aGlobal(x        + mOffset * kHidden + kOffset);
            GlobalDataB bGlobal(w_q_rope + kOffset * kQRopeWidth + nOffset);

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

        // Head-major output: head nIter's [S, kRopeDim] block, this M-chunk.
        GlobalDataC cGlobal(q_rope + static_cast<size_t>(nIter) * kSeqLen * kRopeDim
                                   + mOffset * kRopeDim);
        TSTORE<AccTile, GlobalDataC,
               AtomicType::AtomicNone,
               ReluPreMode::NoRelu>(cGlobal, cTile);
    }
}

// =============================================================================
// Stage 5e (NEW) -- K_rope projection (shared across heads).
//   X        : [kSeqLen, kHidden]    half   (M=128, K_full=4096)
//   W_k_rope : [kHidden, kRopeDim]   half   (K_full=4096, N=64)
//   K_rope   : [kSeqLen, kRopeDim]   half   (M=128, N=64)
//
// Same structure as runKVCompression with kRopeDim in place of kLatent.
// =============================================================================
template <typename TIn, typename TWeight, typename TOut>
__global__ AICORE void runKRopeProjection(__gm__ uint8_t *k_rope_raw,
                                          __gm__ uint8_t *x_raw,
                                          __gm__ uint8_t *w_k_rope_raw)
{
    using namespace mla_basic_cfg;

    __gm__ TIn     *x        = reinterpret_cast<__gm__ TIn     *>(x_raw);
    __gm__ TWeight *w_k_rope = reinterpret_cast<__gm__ TWeight *>(w_k_rope_raw);
    __gm__ TOut    *k_rope   = reinterpret_cast<__gm__ TOut    *>(k_rope_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM   + 15) / 16) * 16;
    constexpr int K = ((kInnerK  + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kRopeDim + blockAlign - 1) / blockAlign) * blockAlign;

    using GlobalDataA = GlobalTensor<TIn,     Shape<1, 1, 1, kTileM,  kInnerK>,
                                     Stride<1, 1, 1, kHidden, 1>>;
    using GlobalDataB = GlobalTensor<TWeight, Shape<1, 1, 1, kInnerK, kRopeDim>,
                                     Stride<1, 1, 1, kRopeDim, 1>>;
    using GlobalDataC = GlobalTensor<TOut,    Shape<1, 1, 1, kTileM,  kRopeDim>,
                                     Stride<1, 1, 1, kRopeDim, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, K, BLayout::ColMajor,
                              kTileM, kInnerK,  SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K, N, BLayout::ColMajor,
                              kInnerK, kRopeDim, SLayout::RowMajor, 512>;
    using LeftTile     = TileLeft <TIn,     M, K, kTileM,  kInnerK>;
    using RightTile    = TileRight<TWeight, K, N, kInnerK, kRopeDim>;
    using AccTile      = TileAcc  <float,   M, N, kTileM,  kRopeDim>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    // Multi-core: M-chunks of kTileM rows, strided across cores.
    const unsigned core_id = get_block_idx();

    for (unsigned mIter = core_id; mIter < kSeqMIter; mIter += kBlockDim) {
        const size_t mOffset = static_cast<size_t>(mIter) * kTileM;

        for (unsigned kIter = 0; kIter < kHiddenKIter; ++kIter) {
            const size_t kOffset = static_cast<size_t>(kIter) * kInnerK;

            GlobalDataA aGlobal(x        + mOffset * kHidden + kOffset);
            GlobalDataB bGlobal(w_k_rope + kOffset * kRopeDim);

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

        GlobalDataC cGlobal(k_rope + mOffset * kRopeDim);
        TSTORE<AccTile, GlobalDataC,
               AtomicType::AtomicNone,
               ReluPreMode::NoRelu>(cGlobal, cTile);
    }
}

// =============================================================================
// Stage 5f (NEW) -- Attention QK rope:
//      scores_rope[h] = Q_rope_rot_h @ K_rope_rot^T,  per head h
//
//   Q_rope_rot : [kNumHeads, kSeqLen, kRopeDim]   half   (head-major)
//   K_rope_rot : [kSeqLen, kRopeDim]              half   (shared across heads)
//   scores_rope: [kNumHeads, kSeqLen, kSeqLen]    half
//
// M=kSeqLen=128, K=kRopeDim=64 (single K iter), N=kSeqLen=128 (split-N x 2).
// Reuses the Layout::DN / ZN trick from runAttnQK for the transposed B.
// =============================================================================
template <typename TIn, typename TOut>
__global__ AICORE void runAttnQKRope(__gm__ uint8_t *scores_raw,
                                     __gm__ uint8_t *q_rope_raw,
                                     __gm__ uint8_t *k_rope_raw)
{
    using namespace mla_basic_cfg;

    __gm__ TIn  *q_rope = reinterpret_cast<__gm__ TIn  *>(q_rope_raw);
    __gm__ TIn  *k_rope = reinterpret_cast<__gm__ TIn  *>(k_rope_raw);
    __gm__ TOut *scores = reinterpret_cast<__gm__ TOut *>(scores_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM   + 15) / 16) * 16;
    constexpr int K = ((kRopeDim + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kInnerN  + blockAlign - 1) / blockAlign) * blockAlign;

    // A : Q_rope_rot_h sub-tile (kTileM, kRopeDim), head-major layout with row stride kRopeDim.
    using GlobalDataA = GlobalTensor<TIn, Shape<1, 1, 1, kTileM,   kRopeDim>,
                                     Stride<1, 1, 1, kRopeDim, 1>>;
    // B : K_rope_rot^T view (kRopeDim, kInnerN). Same DN trick as runAttnQK.
    using GlobalDataBT = GlobalTensor<TIn, Shape<1, 1, 1, kRopeDim, kInnerN>,
                                      Stride<1, 1, 1, 1, kRopeDim>, Layout::DN>;
    // C : scores_rope[h] sub-tile (kTileM, kInnerN), row stride kSeqLen.
    using GlobalDataC  = GlobalTensor<TOut, Shape<1, 1, 1, kTileM, kInnerN>,
                                      Stride<1, 1, 1, kSeqLen, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn, M, K, BLayout::ColMajor,
                              kTileM,   kRopeDim, SLayout::RowMajor, 512>;
    // BLayout::RowMajor + SLayout::ColMajor -> ZN, matches Layout::DN GlobalTensor.
    using TileMatBData = Tile<TileType::Mat, TIn, K, N, BLayout::RowMajor,
                              kRopeDim, kInnerN,  SLayout::ColMajor, 512>;
    using LeftTile     = TileLeft <TIn,   M, K, kTileM,   kRopeDim>;
    using RightTile    = TileRight<TIn,   K, N, kRopeDim, kInnerN>;
    using AccTile      = TileAcc  <float, M, N, kTileM,   kInnerN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    // Multi-core: flatten (h, mIter, nIter) — each work item is one
    // [kTileM, kInnerN] scores_rope sub-tile.
    const unsigned core_id   = get_block_idx();
    constexpr unsigned kWork = kNumHeads * kSeqMIter * kSeqNIter;

    for (unsigned w = core_id; w < kWork; w += kBlockDim) {
        const unsigned h     = w / (kSeqMIter * kSeqNIter);
        const unsigned rem   = w % (kSeqMIter * kSeqNIter);
        const unsigned mIter = rem / kSeqNIter;
        const unsigned nIter = rem % kSeqNIter;

        const size_t qBase      = static_cast<size_t>(h) * kSeqLen * kRopeDim;
        const size_t scoresBase = static_cast<size_t>(h) * kSeqLen * kSeqLen;
        const size_t mOffset    = static_cast<size_t>(mIter) * kTileM;
        const size_t nOffset    = static_cast<size_t>(nIter) * kInnerN;

        // Single K iter (kRopeDim == kInnerN == 64). A is the [kTileM, kRopeDim]
        // slice of Q_rope_rot for head h starting at row mOffset.
        GlobalDataA  aGlobal(q_rope + qBase + mOffset * kRopeDim);
        GlobalDataBT bGlobal(k_rope + nOffset * kRopeDim);

        TLOAD(aMatTile, aGlobal);
        TLOAD(bMatTile, bGlobal);
        TMOV (aTile, aMatTile);
        TMOV (bTile, bMatTile);

        TMATMUL(cTile, aTile, bTile);

        GlobalDataC cGlobal(scores + scoresBase + mOffset * kSeqLen + nOffset);
        TSTORE<AccTile, GlobalDataC,
               AtomicType::AtomicNone,
               ReluPreMode::NoRelu>(cGlobal, cTile);
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
void launchQCompression(uint8_t *c_q, uint8_t *x, uint8_t *w_dq, void *stream)
{
    runQCompression<TIn, TWeight, TOut><<<mla_basic_cfg::kBlockDim, nullptr, stream>>>(c_q, x, w_dq);
}

template <typename TIn, typename TWeight, typename TOut>
void launchQAbsorb(uint8_t *q_absorbed, uint8_t *c_q, uint8_t *w_qk, void *stream)
{
    runQAbsorb<TIn, TWeight, TOut><<<mla_basic_cfg::kBlockDim, nullptr, stream>>>(q_absorbed, c_q, w_qk);
}

template <typename TIn, typename TWeight, typename TOut>
void launchKVCompression(uint8_t *c_kv, uint8_t *x, uint8_t *w_dkv, void *stream)
{
    runKVCompression<TIn, TWeight, TOut><<<mla_basic_cfg::kBlockDim, nullptr, stream>>>(c_kv, x, w_dkv);
}

template <typename TIn, typename TWeight, typename TOut>
void launchVReconstruction(uint8_t *v, uint8_t *c_cache, uint8_t *w_uv, void *stream)
{
    runVReconstruction<TIn, TWeight, TOut><<<mla_basic_cfg::kBlockDim, nullptr, stream>>>(
        v, c_cache, w_uv);
}

template <typename TIn, typename TOut>
void launchAttnQK(uint8_t *scores, uint8_t *q, uint8_t *k, void *stream)
{
    runAttnQK<TIn, TOut><<<mla_basic_cfg::kBlockDim, nullptr, stream>>>(scores, q, k);
}

template <typename TIn, typename TOut>
void launchAttnPV(uint8_t *out, uint8_t *probs, uint8_t *v, void *stream)
{
    runAttnPV<TIn, TOut><<<mla_basic_cfg::kBlockDim, nullptr, stream>>>(out, probs, v);
}

template <typename TIn, typename TWeight, typename TOut>
void launchQRopeProjection(uint8_t *q_rope, uint8_t *x, uint8_t *w_q_rope, void *stream)
{
    runQRopeProjection<TIn, TWeight, TOut><<<mla_basic_cfg::kBlockDim, nullptr, stream>>>(q_rope, x, w_q_rope);
}

template <typename TIn, typename TWeight, typename TOut>
void launchKRopeProjection(uint8_t *k_rope, uint8_t *x, uint8_t *w_k_rope, void *stream)
{
    runKRopeProjection<TIn, TWeight, TOut><<<mla_basic_cfg::kBlockDim, nullptr, stream>>>(k_rope, x, w_k_rope);
}

template <typename TIn, typename TOut>
void launchAttnQKRope(uint8_t *scores_rope, uint8_t *q_rope, uint8_t *k_rope, void *stream)
{
    runAttnQKRope<TIn, TOut><<<mla_basic_cfg::kBlockDim, nullptr, stream>>>(scores_rope, q_rope, k_rope);
}

template void launchQCompression<half, half, half>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchQAbsorb<half, half, half>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchKVCompression<half, half, half>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchVReconstruction<half, half, half>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchAttnQK<half, half>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchAttnPV<half, half>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchQRopeProjection<half, half, half>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchKRopeProjection<half, half, half>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchAttnQKRope<half, half>(uint8_t *, uint8_t *, uint8_t *, void *);

// Non-template wrappers consumed by main.cpp.
extern "C" void launchMlaQCompressionFp16(uint8_t *c_q, uint8_t *x, uint8_t *w_dq, void *stream)
{
    launchQCompression<half, half, half>(c_q, x, w_dq, stream);
}

extern "C" void launchMlaQAbsorbFp16(uint8_t *q_absorbed, uint8_t *c_q,
                                     uint8_t *w_qk, void *stream)
{
    launchQAbsorb<half, half, half>(q_absorbed, c_q, w_qk, stream);
}

extern "C" void launchMlaKVCompressionFp16(uint8_t *c_kv, uint8_t *x, uint8_t *w_dkv, void *stream)
{
    launchKVCompression<half, half, half>(c_kv, x, w_dkv, stream);
}

extern "C" void launchMlaVReconstructionFp16(uint8_t *v, uint8_t *c_cache,
                                             uint8_t *w_uv, void *stream)
{
    launchVReconstruction<half, half, half>(v, c_cache, w_uv, stream);
}

extern "C" void launchMlaAttnQKFp16(uint8_t *scores, uint8_t *q_absorbed,
                                    uint8_t *c_cache, void *stream)
{
    launchAttnQK<half, half>(scores, q_absorbed, c_cache, stream);
}

extern "C" void launchMlaAttnPVFp16(uint8_t *out, uint8_t *probs, uint8_t *v, void *stream)
{
    launchAttnPV<half, half>(out, probs, v, stream);
}

extern "C" void launchMlaQRopeProjectionFp16(uint8_t *q_rope, uint8_t *x,
                                             uint8_t *w_q_rope, void *stream)
{
    launchQRopeProjection<half, half, half>(q_rope, x, w_q_rope, stream);
}

extern "C" void launchMlaKRopeProjectionFp16(uint8_t *k_rope, uint8_t *x,
                                             uint8_t *w_k_rope, void *stream)
{
    launchKRopeProjection<half, half, half>(k_rope, x, w_k_rope, stream);
}

extern "C" void launchMlaAttnQKRopeFp16(uint8_t *scores_rope, uint8_t *q_rope_rot,
                                        uint8_t *k_rope_rot, void *stream)
{
    launchAttnQKRope<half, half>(scores_rope, q_rope_rot, k_rope_rot, stream);
}
