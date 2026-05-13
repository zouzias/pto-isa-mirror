/**
 * moe_router_top1_kernel.cpp - auto-mode A3 prototype.
 *
 * SKELETON. Tile types, loop shape, and host wrapper are final. The bodies
 * of stage 1 (cube GEMM) and stage 2 (vec TROWARGMAX) are marked
 * TODO(body) and will be filled in the implementation pass.
 *
 * Composition (mirrors §A18 split-kernel pattern):
 *
 *   Stage 1 (cube) — runRouterStage1Gemm
 *     X[T, H] @ W_router[H, E] -> logits[T, E] FP32   (microtiled: M=kTileM)
 *
 *   Stage 2 (vec)  — runRouterStage2Argmax
 *     TROWARGMAX(expert_id[T, 1], logits[T, E], tmp)  (no values output)
 *
 *   Host fires both on the same ACL stream; ACL stream-order guarantees
 *   stage 2 only starts after stage 1's TSTORE has fully drained to GM.
 *
 * Why split:
 *   - Stage 1 is cube arch (--cce-aicore-arch=dav-c220-cube).
 *   - Stage 2 is vec arch. The TU compiles with the cube arch flag, and
 *     §A18 confirmed that vec-only operations also run on cube builds
 *     (no in-tree precedent for mixing real vec passes with cube TMATMUL
 *     in the SAME __global__ AICORE body — split avoids touching that).
 *
 * Risks (per the §A18 / TROWARGMAX investigation):
 *   - `TRowReduceIdxOps.hpp` PR-852 sync bug ([auto_mode_bad_patterns.md §2.7]).
 *     If stage 1 logits match the golden but stage 2 expert_id does not, this
 *     is the most likely cause; log as a new occurrence of §E2.
 *   - Cube GEMM at N=E=16: blockAlign for FP16 is 16, so N=16 sits exactly on
 *     the alignment boundary. No `validN < N` assumption needed.
 *
 * Auto-mode constraints honored (each stage):
 *   - Single AICORE; no block_idx work split.
 *   - Static tile shapes; no SetValidRow / partial stores.
 *   - All tiles declared once outside the loops; auto allocator pins each.
 *   - No TASSIGN literal addresses, no `#ifndef __PTO_AUTO__` manual-sync,
 *     no Tile::data() in kernel code, no *_IMPL calls, no raw CCE intrinsics,
 *     no Event<>, no TPipe/TPUSH/TPOP, no double buffering.
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace moe_router_top1_cfg {

// Static configuration. Must match scripts/gen_data.py and main.cpp.
constexpr unsigned kT      = 256;   // tokens (rows of X)
constexpr unsigned kH      = 64;    // hidden dim (cols of X, rows of W_router)
constexpr unsigned kE      = 16;    // experts (cols of W_router, cols of logits)
constexpr unsigned kTileM  = 128;   // M dim per cube microtile (matches §A18)

static_assert(kT % kTileM == 0, "kT must be a multiple of kTileM for the v1 static-tail layout.");

}  // namespace moe_router_top1_cfg

// ============================================================================
// Stage 1 — cube GEMM:  logits[T, E] = X[T, H] @ W_router[H, E]   (FP32 acc).
// Shape source: structurally identical to §A16 / §A18 stage 2 — only the
// dimension labels differ (M=T, K=H, N=E, no per-expert outer loop).
// ============================================================================
template <typename TIn, typename TWeight, typename TOut>
__global__ AICORE void runRouterStage1Gemm(
    __gm__ uint8_t *logits_raw,
    __gm__ uint8_t *X_raw,
    __gm__ uint8_t *W_raw)
{
    using namespace moe_router_top1_cfg;

    __gm__ TIn     *X      = reinterpret_cast<__gm__ TIn     *>(X_raw);
    __gm__ TWeight *W      = reinterpret_cast<__gm__ TWeight *>(W_raw);
    __gm__ TOut    *logits = reinterpret_cast<__gm__ TOut    *>(logits_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM + 15) / 16) * 16;
    constexpr int K = ((kH      + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kE      + blockAlign - 1) / blockAlign) * blockAlign;

    using GlobalDataA =
        GlobalTensor<TIn,     Shape<1, 1, 1, kTileM, kH>,
                     Stride<1 * kTileM * kH, 1 * kTileM * kH, kTileM * kH, kH, 1>>;
    using GlobalDataB =
        GlobalTensor<TWeight, Shape<1, 1, 1, kH,     kE>,
                     Stride<1 * kH * kE,     1 * kH * kE,     kH * kE,     kE, 1>>;
    using GlobalDataC =
        GlobalTensor<TOut,    Shape<1, 1, 1, kTileM, kE>,
                     Stride<1 * kTileM * kE, 1 * kTileM * kE, kTileM * kE, kE, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, K, BLayout::ColMajor,
                              kTileM, kH, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K, N, BLayout::ColMajor,
                              kH,     kE, SLayout::RowMajor, 512>;
    using LeftTile     = TileLeft <TIn,     M, K, kTileM, kH>;
    using RightTile    = TileRight<TWeight, K, N, kH,     kE>;
    using AccTile      = TileAcc  <TOut,    M, N, kTileM, kE>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    // W_router is invariant across token microtiles; load it once outside the loop.
    GlobalDataB bGlobal(W);

    // TODO(body): port the §A18 stage-2 GEMM body to this microtile loop.
    //             Sketch:
    //               for (m0 = 0; m0 < kT; m0 += kTileM) {
    //                   GlobalDataA aGlobal(X      + m0 * kH);
    //                   GlobalDataC cGlobal(logits + m0 * kE);
    //                   TLOAD(aMatTile, aGlobal);
    //                   TLOAD(bMatTile, bGlobal);
    //                   TMOV(aTile, aMatTile);
    //                   TMOV(bTile, bMatTile);
    //                   TMATMUL(cTile, aTile, bTile);
    //                   TSTORE(cGlobal, cTile);    // FP32 ND store; no activation.
    //               }
    (void)aMatTile; (void)bMatTile; (void)aTile; (void)bTile; (void)cTile;
    (void)bGlobal;  (void)X;        (void)logits;
}

// ============================================================================
// Stage 2 — vec TROWARGMAX over logits.
//   In  : logits     [T, E] float32.
//   Out : expert_id  [T, 1] uint32 (= argmax_e logits[t, e]).
// ============================================================================
template <typename T_, typename TIdx_>
__global__ AICORE void runRouterStage2Argmax(
    __gm__ uint8_t *expert_id_raw,
    __gm__ uint8_t *logits_raw)
{
    using namespace moe_router_top1_cfg;

    __gm__ T_    *logits    = reinterpret_cast<__gm__ T_    *>(logits_raw);
    __gm__ TIdx_ *expert_id = reinterpret_cast<__gm__ TIdx_ *>(expert_id_raw);

    using LogitsGlobal   = GlobalTensor<T_,    Shape<1, 1, 1, kT, kE>, Stride<1, 1, 1, kE, 1>>;
    using ExpertIdGlobal = GlobalTensor<TIdx_, Shape<1, 1, 1, kT, 1 >, Stride<1, 1, 1, 1 , 1>>;

    using LogitsTile    = Tile<TileType::Vec, T_,    kT, kE, BLayout::RowMajor, -1, -1>;
    using TmpTile       = Tile<TileType::Vec, T_,    kT, kE, BLayout::RowMajor, -1, -1>;
    using ExpertIdTile  = Tile<TileType::Vec, TIdx_, kT, 1,  BLayout::RowMajor, -1, -1>;

    LogitsTile   logitsTile(kT, kE);
    TmpTile      tmpTile(kT, kE);
    ExpertIdTile expertIdTile(kT, 1);

    LogitsGlobal   logitsGlobal(logits);
    ExpertIdGlobal expertIdGlobal(expert_id);

    // TODO(body): TLOAD logits; call single-output TROWARGMAX(expertIdTile,
    //             logitsTile, tmpTile); TSTORE expertIdTile.
    (void)logitsTile; (void)tmpTile; (void)expertIdTile;
    (void)logitsGlobal; (void)expertIdGlobal;
}

// ----------------------------------------------------------------------------
// Templated host launchers + non-template wrapper. Mirrors §A18.
// ----------------------------------------------------------------------------
template <typename TIn, typename TWeight, typename TOut>
void launchRouterStage1Gemm(uint8_t *logits, uint8_t *X, uint8_t *W, void *stream)
{
    runRouterStage1Gemm<TIn, TWeight, TOut><<<1, nullptr, stream>>>(logits, X, W);
}

template <typename T_, typename TIdx_>
void launchRouterStage2Argmax(uint8_t *expert_id, uint8_t *logits, void *stream)
{
    runRouterStage2Argmax<T_, TIdx_><<<1, nullptr, stream>>>(expert_id, logits);
}

template void launchRouterStage1Gemm<half, half, float>(
    uint8_t *logits, uint8_t *X, uint8_t *W, void *stream);
template void launchRouterStage2Argmax<float, uint32_t>(
    uint8_t *expert_id, uint8_t *logits, void *stream);

extern "C" void launchMoeRouterTop1Fp16(uint8_t *logits_fp32,
                                        uint8_t *expert_id_u32,
                                        uint8_t *X_fp16,
                                        uint8_t *W_fp16,
                                        void    *stream)
{
    launchRouterStage1Gemm<half, half, float>(logits_fp32, X_fp16, W_fp16, stream);
    launchRouterStage2Argmax<float, uint32_t>(expert_id_u32, logits_fp32, stream);
}
