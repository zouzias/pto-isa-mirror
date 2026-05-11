/**
 * moe_segmented_gemm_relu_kernel.cpp - auto-mode A3 prototype.
 *
 * One expert-specific GEMM per microtile (same as §A16
 * moe_segmented_gemm_one_layer), but with ReLU fused into the L0C -> GM store
 * via the `ReluPreMode::NormalRelu` overload of the public `TSTORE`
 * wrapper. The pre-ReLU accumulator stays on L0C; the FIX pipe applies
 * `max(x, 0)` while writing to GM. **No** cube -> vec handoff, **no** separate
 * vector kernel, **no** mix-arch build.
 *
 * Activation reference / proof-of-existence:
 *   - [include/pto/common/pto_instr.hpp:251-258](../../../../include/pto/common/pto_instr.hpp#L251-L258)
 *     defines
 *       template <typename TileData, typename GlobalData,
 *                 AtomicType atomicType = AtomicType::AtomicNone,
 *                 ReluPreMode reluPreMode, typename... WaitEvents>
 *       TSTORE(GlobalData &dst, TileData &src, WaitEvents &...events);
 *   - `ReluPreMode::NormalRelu` is enum value 1 at
 *     [include/pto/common/type.hpp:255-259](../../../../include/pto/common/type.hpp#L255-L259).
 *   - Confirmed-build A3 cube reference using this exact form:
 *     [tests/npu/a2a3/src/st/testcase/tstore_acc2gm/tstore_acc2gm_kernel.cpp:88-93](../../../../tests/npu/a2a3/src/st/testcase/tstore_acc2gm/tstore_acc2gm_kernel.cpp#L88-L93)
 *     —
 *       TSTORE<AccTile, GlobalDataOut, atomicTypeEnum, ReluPreMode::NormalRelu>(
 *           dstGlobal, cTile);
 *     `tstore_acc2gm` is in `ALL_TESTCASES` and builds under
 *     `pto_cube_st` (same cube-arch + auto-mode recipe as this project).
 *
 * Semantics (host-visible):
 *   for (e = 0; e < kNumExperts; ++e) {
 *       for (m0 = 0; m0 < count; m0 += kTileM) {
 *           A = packed_tokens[start+m0 : start+m0+kTileM, 0:kH]
 *           B = expert_weight[e, 0:kH, 0:kO]
 *           C = max(A @ B, 0)   // ReLU fused into the TSTORE
 *           packed_output[start+m0 : start+m0+kTileM, 0:kO] = C
 *       }
 *   }
 *
 * The only delta vs §A16 moe_segmented_gemm_one_layer is the explicit
 * TSTORE template-arg block adding `ReluPreMode::NormalRelu`. Everything
 * else — tile aliases, GlobalTensor shapes, nested loop, GM offset
 * arithmetic, host boundary, FP16 x FP16 -> FP32 dtype combo, host
 * padding — is identical and the same comments apply.
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace moe_segmented_gemm_relu_cfg {
constexpr unsigned kH          = 64;
constexpr unsigned kO          = 64;
constexpr unsigned kTileM      = 128;
constexpr unsigned kNumExperts = 4;
}  // namespace moe_segmented_gemm_relu_cfg

template <typename TOut, typename TIn, typename TWeight>
__global__ AICORE void runMoeSegmentedGemmRelu(
    __gm__ uint8_t *packed_output_raw,
    __gm__ uint8_t *packed_tokens_raw,
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ uint8_t *expert_weight_raw)
{
    using namespace moe_segmented_gemm_relu_cfg;

    __gm__ TOut    *packed_output = reinterpret_cast<__gm__ TOut    *>(packed_output_raw);
    __gm__ TIn     *packed_tokens = reinterpret_cast<__gm__ TIn     *>(packed_tokens_raw);
    __gm__ TWeight *expert_weight = reinterpret_cast<__gm__ TWeight *>(expert_weight_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM + 15) / 16) * 16;
    constexpr int K = ((kH + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kO + blockAlign - 1) / blockAlign) * blockAlign;

    using GlobalDataA =
        GlobalTensor<TIn,     Shape<1, 1, 1, kTileM, kH>,
                     Stride<1 * kTileM * kH, 1 * kTileM * kH, kTileM * kH, kH, 1>>;
    using GlobalDataB =
        GlobalTensor<TWeight, Shape<1, 1, 1, kH,     kO>,
                     Stride<1 * kH * kO,     1 * kH * kO,     kH * kO,     kO, 1>>;
    using GlobalDataC =
        GlobalTensor<TOut,    Shape<1, 1, 1, kTileM, kO>,
                     Stride<1 * kTileM * kO, 1 * kTileM * kO, kTileM * kO, kO, 1>>;

    using TileMatAData = Tile<TileType::Mat, TIn,     M, K, BLayout::ColMajor,
                              kTileM, kH, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, TWeight, K, N, BLayout::ColMajor,
                              kH,     kO, SLayout::RowMajor, 512>;

    using LeftTile  = TileLeft <TIn,     M, K, kTileM, kH>;
    using RightTile = TileRight<TWeight, K, N, kH,     kO>;
    using AccTile   = TileAcc  <TOut,    M, N, kTileM, kO>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile     aTile;
    RightTile    bTile;
    AccTile      cTile;

    for (unsigned e = 0; e < kNumExperts; ++e) {
        int32_t start = expert_start[e];
        int32_t count = expert_count[e];

        GlobalDataB bGlobal(expert_weight + static_cast<size_t>(e) * kH * kO);

        for (int32_t m0 = 0; m0 < count; m0 += static_cast<int32_t>(kTileM)) {
            size_t row  = static_cast<size_t>(start) + static_cast<size_t>(m0);
            size_t aOff = row * kH;
            size_t cOff = row * kO;

            GlobalDataA aGlobal(packed_tokens + aOff);
            GlobalDataC cGlobal(packed_output + cOff);

            TLOAD(aMatTile, aGlobal);
            TLOAD(bMatTile, bGlobal);
            TMOV(aTile, aMatTile);
            TMOV(bTile, bMatTile);
            TMATMUL(cTile, aTile, bTile);

            // L0C -> GM with ReLU fused into the FIX-pipe store.
            // Template args, in order: <TileData, GlobalData, AtomicType,
            // ReluPreMode>. `AtomicType::AtomicNone` is the explicit default;
            // `ReluPreMode::NormalRelu` is the new piece (default is NoRelu).
            // Matches the shape used by tstore_acc2gm_kernel.cpp lines 88-93.
            TSTORE<AccTile, GlobalDataC, AtomicType::AtomicNone,
                   ReluPreMode::NormalRelu>(cGlobal, cTile);
        }
    }
}

template <typename TOut, typename TIn, typename TWeight>
void launchMoeSegmentedGemmRelu(uint8_t *packed_output,
                                uint8_t *packed_tokens,
                                int32_t *expert_count,
                                int32_t *expert_start,
                                uint8_t *expert_weight,
                                void *stream)
{
    runMoeSegmentedGemmRelu<TOut, TIn, TWeight><<<1, nullptr, stream>>>(
        packed_output, packed_tokens, expert_count, expert_start, expert_weight);
}

template void launchMoeSegmentedGemmRelu<float, half, half>(
    uint8_t *packed_output, uint8_t *packed_tokens,
    int32_t *expert_count, int32_t *expert_start,
    uint8_t *expert_weight, void *stream);

// Non-template host-boundary wrapper. main.cpp cannot see `half` (the host
// TU is compiled with plain `-xc++`, which lacks bisheng-CCE built-ins);
// see compile_error_logbook.md §E13 and known_good_kernel_examples.md §A16.
extern "C" void launchMoeSegmentedGemmReluFp16(uint8_t *packed_output,
                                                uint8_t *packed_tokens,
                                                int32_t *expert_count,
                                                int32_t *expert_start,
                                                uint8_t *expert_weight,
                                                void *stream)
{
    launchMoeSegmentedGemmRelu<float, half, half>(
        packed_output, packed_tokens, expert_count, expert_start,
        expert_weight, stream);
}
