/**
 * moe_top1_gather_precomp_kernel.cpp - auto-mode A3 prototype (Milestone 1a).
 *
 * Host-precomputed packed gather sanity test. NOT real MoE permute; it
 * verifies two A3 auto-mode capabilities the real M1b kernel will depend on:
 *   1. runtime GM scalar reads of an index array from kernel code, and
 *   2. runtime-offset GM-row TLOAD/TSTORE driven by that scalar.
 *
 * Semantics (host supplies the permutation):
 *   for (p = 0; p < T; ++p) {
 *       src_row = packed_to_token[p];        // GM scalar read
 *       packed_tokens[p, :] = tokens[src_row, :];
 *   }
 *
 * Pattern source:
 *   - kernels/automode/a2a3/add_tile_array/add_tile_array_kernel.cpp
 *     (single AICORE, in-kernel serial loop, runtime GM offset per iter)
 *   - kernels/automode/a2a3/topk/topk_kernel.cpp (auto-mode A3 confirmed-built)
 *
 * Auto-mode constraints honored:
 *   - Single AICORE (<<<1, nullptr, stream>>>); no block_idx split.
 *   - Static row tile (Tile<Vec, T, 1, H, RowMajor, 1, H>) declared once,
 *     reused across iterations (auto allocator pins its UB address).
 *   - No TASSIGN aliasing tricks; no Tile::data() in kernel code;
 *     no *_IMPL calls; no raw CCE intrinsics; no Event<>; no manual sync.
 *   - GlobalTensor reconstructed per iteration with (base + offset).
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace moe_top1_gather_precomp_cfg {

// Static configuration. Must match scripts/gen_data.py and main.cpp.
constexpr unsigned kT = 256;    // number of tokens
constexpr unsigned kH = 64;     // hidden dim per token

}  // namespace moe_top1_gather_precomp_cfg

template <typename T>
__global__ AICORE void runMoeTop1GatherPrecomp(__gm__ T        __out__ *packed_tokens,
                                                __gm__ T        __in__  *tokens,
                                                __gm__ uint32_t __in__  *packed_to_token)
{
    using namespace moe_top1_gather_precomp_cfg;

    using RowShape  = Shape <1, 1, 1, 1, kH>;
    using RowStride = Stride<1, 1, 1, kH, 1>;
    using RowGlobal = GlobalTensor<T, RowShape, RowStride>;

    using RowTile = Tile<TileType::Vec, T,
                         1, kH,
                         BLayout::RowMajor,
                         1, kH>;

    RowTile rowTile;

    for (unsigned p = 0; p < kT; ++p) {
        uint32_t src_row = packed_to_token[p];                          // GM scalar read (Assumption A1)
        size_t   src_off = static_cast<size_t>(src_row) * kH;
        size_t   dst_off = static_cast<size_t>(p)       * kH;

        RowGlobal srcGlobal(tokens        + src_off);
        RowGlobal dstGlobal(packed_tokens + dst_off);

        TLOAD (rowTile, srcGlobal);
        TSTORE(dstGlobal, rowTile);
    }
}

template <typename T>
void launchMoeTop1GatherPrecomp(T *packed_tokens, T *tokens,
                                uint32_t *packed_to_token, void *stream)
{
    runMoeTop1GatherPrecomp<T><<<1, nullptr, stream>>>(packed_tokens, tokens, packed_to_token);
}

template void launchMoeTop1GatherPrecomp<float>(float *packed_tokens, float *tokens,
                                                uint32_t *packed_to_token, void *stream);
