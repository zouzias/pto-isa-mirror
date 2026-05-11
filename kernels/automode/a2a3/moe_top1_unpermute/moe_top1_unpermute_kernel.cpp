/**
 * moe_top1_unpermute_kernel.cpp - auto-mode A3 prototype.
 *
 * Reverse token movement after the forward top-1 MoE permute. Restores packed
 * expert output back to original token order.
 *
 * Semantics:
 *   for (t = 0; t < T; ++t) {
 *       packed_pos = token_to_packed[t];
 *       output[t, :] = packed_output[packed_pos, :];
 *   }
 *
 * Inputs   (GM): packed_output [T, H] float32, token_to_packed [T] int32
 * Outputs  (GM): output        [T, H] float32
 *
 * `token_to_packed` is the same mapping the passing forward kernel
 * [kernels/automode/a2a3/moe_top1_permute/](../moe_top1_permute/) emits:
 *   token_to_packed[t] = packed slot at which token t lives after permute.
 *
 * Auto-mode constraints honored (mirror moe_top1_permute / add_tile_array):
 *   - Single AICORE (<<<1, nullptr, stream>>>); no block_idx work split.
 *   - Static row tile (Tile<Vec, T, 1, H, RowMajor, 1, H>) declared once,
 *     reused across iterations — auto allocator pins its UB address.
 *   - GlobalTensor reconstructed per iteration with (base + runtime offset).
 *   - No TASSIGN aliasing; no Tile::data() in kernel; no *_IMPL calls;
 *     no raw CCE intrinsics; no Event<>; no manual sync; no TPipe / TPUSH /
 *     TPOP; no double buffering.
 *
 * Status: NOT yet confirmed-built. Pattern follows
 *   - kernels/automode/a2a3/moe_top1_permute/moe_top1_permute_kernel.cpp
 *     (user-confirmed PASS on Ascend910B1; see
 *     docs_for_ai/known_good_kernel_examples.md §A13 and
 *     docs_for_ai/assumptions_to_verify.md §11.4 for the resolved patterns
 *     this kernel reuses)
 *   - kernels/automode/a2a3/add_tile_array/ (the original auto-mode A3
 *     baseline shape).
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace moe_top1_unpermute_cfg {

// Static configuration. Must match scripts/gen_data.py and main.cpp, and the
// shape used by moe_top1_permute.
constexpr unsigned kT = 256;    // number of tokens
constexpr unsigned kH = 64;     // hidden dim per token

}  // namespace moe_top1_unpermute_cfg

template <typename T>
__global__ AICORE void runMoeTop1Unpermute(__gm__ T       __out__ *output,
                                            __gm__ T       __in__  *packed_output,
                                            __gm__ int32_t __in__  *token_to_packed)
{
    using namespace moe_top1_unpermute_cfg;

    using RowShape  = Shape <1, 1, 1, 1, kH>;
    using RowStride = Stride<1, 1, 1, kH, 1>;
    using RowGlobal = GlobalTensor<T, RowShape, RowStride>;

    using RowTile = Tile<TileType::Vec, T,
                         1, kH,
                         BLayout::RowMajor,
                         1, kH>;

    RowTile rowTile;

    for (unsigned t = 0; t < kT; ++t) {
        int32_t packed_pos = token_to_packed[t];      // GM scalar read (resolved §11.4)

        size_t src_off = static_cast<size_t>(packed_pos) * kH;
        size_t dst_off = static_cast<size_t>(t)          * kH;

        RowGlobal srcGlobal(packed_output + src_off);
        RowGlobal dstGlobal(output        + dst_off);

        TLOAD (rowTile, srcGlobal);
        TSTORE(dstGlobal, rowTile);
    }
}

template <typename T>
void launchMoeTop1Unpermute(T *output, T *packed_output,
                            int32_t *token_to_packed, void *stream)
{
    runMoeTop1Unpermute<T><<<1, nullptr, stream>>>(output, packed_output, token_to_packed);
}

template void launchMoeTop1Unpermute<float>(float *output, float *packed_output,
                                            int32_t *token_to_packed, void *stream);
