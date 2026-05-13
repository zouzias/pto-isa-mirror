/**
 * moe_top1_full_kernel.cpp - auto-mode A3 prototype (SKELETON).
 *
 * Integrated top-1 MoE forward path. Five `__global__ AICORE` functions in
 * one TU, fired back-to-back on the same ACL stream by one extern "C" host
 * wrapper. Each stage reuses an already-proven §A pattern; only stage 2
 * (argmax + permute) introduces a small new composition.
 *
 *   Stage 1 (cube) — runMoeTop1Stage1RouterGemm
 *     X @ W_router -> logits[T, E] FP32
 *     Body: byte-for-byte the moe_router_top1 stage-1 GEMM.
 *
 *   Stage 2 (vec)  — runMoeTop1Stage2ArgmaxPermute   <-- the NEW composition
 *     TROWARGMAX over logits -> expert_id
 *     histogram + (padded) prefix-sum + pack tokens into packed_tokens
 *     Writes: expert_id, expert_count (padded), expert_start (padded),
 *             token_to_packed, packed_tokens.
 *     Bodies fused from: moe_router_top1 stage 2 + §A13 moe_top1_permute.
 *
 *   Stage 3 (cube) — runMoeTop1Stage3Ffn1Gemm1Relu
 *     Per-expert microtile loop. GEMM1 + fused ReLU + FP32->FP16 in TSTORE.
 *     Body: byte-for-byte the §A18 stage-1 kernel
 *           (runFfnStage1Gemm1Relu).
 *
 *   Stage 4 (cube) — runMoeTop1Stage4Ffn2Gemm2
 *     Per-expert microtile loop. GEMM2 reads ffn_scratch, writes
 *     packed_output. Body: byte-for-byte the §A18 stage-2 kernel
 *           (runFfnStage2Gemm2).
 *
 *   Stage 5 (vec)  — runMoeTop1Stage5Unpermute
 *     Inverse of the permute: row-copy packed_output[token_to_packed[t], :]
 *     into Y[t, :]. Body: byte-for-byte §A14 moe_top1_unpermute.
 *
 * Cross-stage data flow (all in GM, between kernel launches):
 *
 *   X, W_router -> [stage 1] -> logits
 *   logits      -> [stage 2] -> expert_id, expert_count, expert_start,
 *                                token_to_packed, packed_tokens
 *   packed_tokens, W1, expert_*  -> [stage 3] -> ffn_scratch
 *   ffn_scratch,   W2, expert_*  -> [stage 4] -> packed_output
 *   packed_output, token_to_packed -> [stage 5] -> Y
 *
 * Why this composition:
 *   - All five stages map to proven §A patterns; the only genuinely new
 *     piece is stage 2's argmax + permute fusion, which is just §A13's
 *     permute with its expert_id input replaced by a freshly-computed
 *     TROWARGMAX result on the same kernel's logits buffer.
 *   - Cube and vec arches do not mix inside one __global__ in any in-tree
 *     evidence; the five-kernel split respects that.
 *
 * Risks:
 *   - TROWARGMAX in `TRowReduceIdxOps.hpp` is pre-PR-852 (auto_mode_bad_
 *     patterns.md §2.7). If output_logits matches the golden but
 *     output_expert_id does not, that is the most likely cause.
 *   - Worst-case T_PADDED scratch sizing is coarse (T + kE*kTileM rows). For
 *     T=256, kE=16, kTileM=128 this is 2304 rows of FP16/FP32 GM, host-
 *     allocated. Future iteration may tighten with a tighter on-device bound.
 *
 * Auto-mode constraints honored per stage: identical to §A13 / §A14 / §A16 /
 * §A17 / §A18 — single AICORE, static tile shapes, no SetValidRow, no
 * TASSIGN literal addresses, no manual sync, no Tile::data() in kernel code,
 * no *_IMPL calls, no raw CCE intrinsics, no double buffering.
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace moe_top1_full_cfg {

// Static configuration. Must match scripts/gen_data.py and main.cpp.
constexpr unsigned kT      = 256;
constexpr unsigned kH      = 64;
constexpr unsigned kF      = 64;
constexpr unsigned kE      = 16;
constexpr unsigned kTileM  = 128;
constexpr unsigned kT_PADDED_MAX = kT + kE * kTileM;   // host-side scratch bound

static_assert(kT % kTileM == 0 || kT_PADDED_MAX % kTileM == 0,
              "padded scratch upper bound must be kTileM-aligned for the FFN microtile loop.");

}  // namespace moe_top1_full_cfg

// ============================================================================
// Stage 1 (cube) — router GEMM. Identical shape to moe_router_top1 stage 1.
// ============================================================================
template <typename TIn, typename TWeight, typename TOut>
__global__ AICORE void runMoeTop1Stage1RouterGemm(
    __gm__ uint8_t *logits_raw,
    __gm__ uint8_t *X_raw,
    __gm__ uint8_t *W_router_raw)
{
    using namespace moe_top1_full_cfg;
    (void)logits_raw; (void)X_raw; (void)W_router_raw;

    // TODO(body): copy from moe_router_top1_kernel.cpp::runRouterStage1Gemm.
    //             Static shape: M=kTileM, K=kH, N=kE; cube tile budget = 5.
    //             Outer microtile loop is over m0 = 0..kT step kTileM.
}

// ============================================================================
// Stage 2 (vec) — argmax + (padded) permute. Fuses moe_router_top1 stage 2
// with the §A13 permute kernel. This is the only stage with NEW composition;
// stages 3 / 4 / 5 are byte-for-byte ports of existing milestones.
//
// Compact algorithm:
//   1. TLOAD logits; TROWARGMAX(expert_id_tile, logits_tile, tmp).
//   2. TSTORE expert_id to GM (for the comparator).
//   3. histogram: int32_t real_count[kE] = 0; for t in 0..T: real_count[e]++;
//   4. pad each count up to the next multiple of kTileM; prefix-sum -> starts.
//   5. write expert_count (padded), expert_start (padded).
//   6. pack pass: counter[E] = 0; for t in 0..T:
//        e = expert_id_tile[t]; pos = expert_start[e] + counter[e]; counter[e]++;
//        token_to_packed[t] = pos;
//        TLOAD row X[t]; TSTORE packed_tokens[pos] = row.
//      (§A13 already proved this scalar-GM-read-write + row TLOAD/TSTORE shape.)
// ============================================================================
template <typename T_, typename TIn_, typename TIdx_>
__global__ AICORE void runMoeTop1Stage2ArgmaxPermute(
    __gm__ uint8_t *expert_id_raw,
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ int32_t *token_to_packed,
    __gm__ uint8_t *packed_tokens_raw,
    __gm__ uint8_t *logits_raw,
    __gm__ uint8_t *X_raw)
{
    using namespace moe_top1_full_cfg;
    (void)expert_id_raw; (void)expert_count; (void)expert_start;
    (void)token_to_packed; (void)packed_tokens_raw; (void)logits_raw; (void)X_raw;

    // TODO(body):
    //   - logits TLOAD, TROWARGMAX -> expert_id_tile, TSTORE expert_id.
    //   - On-device histogram into int32_t real_count[kE] (scalar GM reads
    //     of expert_id_tile entries — pattern proven by §A13).
    //   - Pad each count to kTileM; compute padded prefix-sum starts.
    //   - Scalar-GM writes expert_count[e], expert_start[e].
    //   - Pack pass: per t, scalar GM write token_to_packed[t]; row TLOAD
    //     X[t] into a single 1xH Vec tile; row TSTORE to packed_tokens[pos].
}

// ============================================================================
// Stage 3 (cube) — FFN GEMM1 + fused ReLU. Byte-for-byte §A18 stage 1.
// ============================================================================
template <typename TIn, typename TWeight, typename TScratch>
__global__ AICORE void runMoeTop1Stage3Ffn1Gemm1Relu(
    __gm__ uint8_t *ffn_scratch_raw,
    __gm__ uint8_t *packed_tokens_raw,
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ uint8_t *W1_raw)
{
    using namespace moe_top1_full_cfg;
    (void)ffn_scratch_raw; (void)packed_tokens_raw;
    (void)expert_count; (void)expert_start; (void)W1_raw;

    // TODO(body): copy from
    //   kernels/automode/a2a3/moe_segmented_ffn_top1/moe_segmented_ffn_top1_kernel.cpp::runFfnStage1Gemm1Relu
    // verbatim. No new auto-mode surface vs §A18.
}

// ============================================================================
// Stage 4 (cube) — FFN GEMM2. Byte-for-byte §A18 stage 2.
// ============================================================================
template <typename TOut, typename TScratch, typename TWeight>
__global__ AICORE void runMoeTop1Stage4Ffn2Gemm2(
    __gm__ uint8_t *packed_output_raw,
    __gm__ uint8_t *ffn_scratch_raw,
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ uint8_t *W2_raw)
{
    using namespace moe_top1_full_cfg;
    (void)packed_output_raw; (void)ffn_scratch_raw;
    (void)expert_count; (void)expert_start; (void)W2_raw;

    // TODO(body): copy from
    //   kernels/automode/a2a3/moe_segmented_ffn_top1/moe_segmented_ffn_top1_kernel.cpp::runFfnStage2Gemm2
    // verbatim.
}

// ============================================================================
// Stage 5 (vec) — unpermute. Byte-for-byte §A14.
// ============================================================================
template <typename T_, typename TIdx_>
__global__ AICORE void runMoeTop1Stage5Unpermute(
    __gm__ uint8_t *Y_raw,
    __gm__ uint8_t *packed_output_raw,
    __gm__ int32_t *token_to_packed)
{
    using namespace moe_top1_full_cfg;
    (void)Y_raw; (void)packed_output_raw; (void)token_to_packed;

    // TODO(body): copy from kernels/automode/a2a3/moe_top1_unpermute/.
    //   for (t = 0; t < kT; ++t) {
    //       int32_t pos = token_to_packed[t];
    //       GlobalDataSrc srcG(packed_output + pos * kH);
    //       GlobalDataDst dstG(Y             + t   * kH);
    //       TLOAD(rowTile, srcG); TSTORE(dstG, rowTile);
    //   }
}

// ----------------------------------------------------------------------------
// Templated host launchers + non-template wrapper.
// ----------------------------------------------------------------------------
template <typename TIn, typename TWeight, typename TOut>
void launchMoeTop1Stage1RouterGemm(uint8_t *logits, uint8_t *X, uint8_t *W_router, void *stream)
{
    runMoeTop1Stage1RouterGemm<TIn, TWeight, TOut><<<1, nullptr, stream>>>(logits, X, W_router);
}

template <typename T_, typename TIn_, typename TIdx_>
void launchMoeTop1Stage2ArgmaxPermute(uint8_t *expert_id,
                                       int32_t *expert_count,
                                       int32_t *expert_start,
                                       int32_t *token_to_packed,
                                       uint8_t *packed_tokens,
                                       uint8_t *logits,
                                       uint8_t *X,
                                       void    *stream)
{
    runMoeTop1Stage2ArgmaxPermute<T_, TIn_, TIdx_><<<1, nullptr, stream>>>(
        expert_id, expert_count, expert_start, token_to_packed, packed_tokens, logits, X);
}

template <typename TIn, typename TWeight, typename TScratch>
void launchMoeTop1Stage3Ffn1Gemm1Relu(uint8_t *ffn_scratch,
                                       uint8_t *packed_tokens,
                                       int32_t *expert_count,
                                       int32_t *expert_start,
                                       uint8_t *W1,
                                       void    *stream)
{
    runMoeTop1Stage3Ffn1Gemm1Relu<TIn, TWeight, TScratch><<<1, nullptr, stream>>>(
        ffn_scratch, packed_tokens, expert_count, expert_start, W1);
}

template <typename TOut, typename TScratch, typename TWeight>
void launchMoeTop1Stage4Ffn2Gemm2(uint8_t *packed_output,
                                   uint8_t *ffn_scratch,
                                   int32_t *expert_count,
                                   int32_t *expert_start,
                                   uint8_t *W2,
                                   void    *stream)
{
    runMoeTop1Stage4Ffn2Gemm2<TOut, TScratch, TWeight><<<1, nullptr, stream>>>(
        packed_output, ffn_scratch, expert_count, expert_start, W2);
}

template <typename T_, typename TIdx_>
void launchMoeTop1Stage5Unpermute(uint8_t *Y,
                                   uint8_t *packed_output,
                                   int32_t *token_to_packed,
                                   void    *stream)
{
    runMoeTop1Stage5Unpermute<T_, TIdx_><<<1, nullptr, stream>>>(Y, packed_output, token_to_packed);
}

template void launchMoeTop1Stage1RouterGemm   <half, half, float>(
    uint8_t *logits, uint8_t *X, uint8_t *W_router, void *stream);
template void launchMoeTop1Stage2ArgmaxPermute<float, half, uint32_t>(
    uint8_t *expert_id, int32_t *expert_count, int32_t *expert_start,
    int32_t *token_to_packed, uint8_t *packed_tokens,
    uint8_t *logits, uint8_t *X, void *stream);
template void launchMoeTop1Stage3Ffn1Gemm1Relu<half, half, half>(
    uint8_t *ffn_scratch, uint8_t *packed_tokens,
    int32_t *expert_count, int32_t *expert_start,
    uint8_t *W1, void *stream);
template void launchMoeTop1Stage4Ffn2Gemm2    <float, half, half>(
    uint8_t *packed_output, uint8_t *ffn_scratch,
    int32_t *expert_count, int32_t *expert_start,
    uint8_t *W2, void *stream);
template void launchMoeTop1Stage5Unpermute    <float, uint32_t>(
    uint8_t *Y, uint8_t *packed_output, int32_t *token_to_packed, void *stream);

// Non-template host-boundary wrapper. Fires all five stages on the same
// stream; ACL stream-order does the cross-kernel sequencing. There is no
// within-kernel cross-stage auto-sync in this design.
extern "C" void launchMoeTop1FullFp16(uint8_t *Y,
                                       uint8_t *expert_id,
                                       uint8_t *logits,
                                       uint8_t *X,
                                       uint8_t *W_router,
                                       uint8_t *W1,
                                       uint8_t *W2,
                                       uint8_t *packed_tokens,
                                       int32_t *expert_count,
                                       int32_t *expert_start,
                                       int32_t *token_to_packed,
                                       uint8_t *ffn_scratch,
                                       uint8_t *packed_output,
                                       void    *stream)
{
    launchMoeTop1Stage1RouterGemm   <half,  half, float   >(logits, X, W_router, stream);
    launchMoeTop1Stage2ArgmaxPermute<float, half, uint32_t>(expert_id, expert_count, expert_start,
                                                            token_to_packed, packed_tokens,
                                                            logits, X, stream);
    launchMoeTop1Stage3Ffn1Gemm1Relu<half,  half, half    >(ffn_scratch, packed_tokens,
                                                            expert_count, expert_start, W1, stream);
    launchMoeTop1Stage4Ffn2Gemm2    <float, half, half    >(packed_output, ffn_scratch,
                                                            expert_count, expert_start, W2, stream);
    launchMoeTop1Stage5Unpermute    <float, uint32_t      >(Y, packed_output, token_to_packed, stream);
}
