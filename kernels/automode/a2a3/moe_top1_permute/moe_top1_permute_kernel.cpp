/**
 * moe_top1_permute_kernel.cpp - auto-mode A3 prototype (Milestone 1b).
 *
 * Device-side top-1 MoE permute, single AICORE, single-core.
 *
 * Inputs   (GM): tokens [T, H], expert_id [T]
 * Outputs  (GM): packed_tokens [T, H], expert_count [E],
 *                expert_start  [E],    token_to_packed [T]
 *
 * Semantics (three passes; same shape as the TileLang Ascend forward permute
 * recipe, simplified to single-core, top-1, unlimited capacity):
 *
 *   1. histogram         : count[e] = #{t : expert_id[t] == e}
 *   2. prefix sum        : start[0] = 0;
 *                          start[e] = start[e-1] + count[e-1]
 *   3. pack              : for each token t in original order,
 *                          slot = counter[e]++;
 *                          packed_pos = start[e] + slot;
 *                          token_to_packed[t] = packed_pos;
 *                          packed_tokens[packed_pos, :] = tokens[t, :];
 *
 * Auto-mode constraints honored:
 *   - Single AICORE (<<<1, nullptr, stream>>>); no block_idx split.
 *   - Static row tile (Tile<Vec, T, 1, H, RowMajor, 1, H>) declared once,
 *     reused across iterations (auto allocator pins its UB address).
 *   - Per-expert histogram / start / counter held in small local int32_t
 *     arrays (NUM_EXPERTS == 4) — register/stack resident, no UB tile.
 *   - GlobalTensor reconstructed per iteration with (base + runtime offset).
 *   - No TASSIGN aliasing tricks; no Tile::data() in kernel code;
 *     no *_IMPL calls; no raw CCE intrinsics; no Event<>; no manual sync.
 *
 * Pattern source:
 *   - kernels/automode/a2a3/add_tile_array/add_tile_array_kernel.cpp
 *     (single AICORE, in-kernel serial loop, runtime GM offset per iter)
 *   - kernels/automode/a2a3/moe_top1_gather_precomp/ (M1a — same row TLOAD/TSTORE
 *     shape; this kernel adds the device-side histogram + counter logic)
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace moe_top1_permute_cfg {

// Static configuration. Must match scripts/gen_data.py and main.cpp.
constexpr unsigned kT           = 256;   // number of tokens
constexpr unsigned kH           = 64;    // hidden dim per token
constexpr unsigned kNumExperts  = 4;     // experts (small, compile-time)

}  // namespace moe_top1_permute_cfg

template <typename T>
__global__ AICORE void runMoeTop1Permute(__gm__ T       __out__ *packed_tokens,
                                          __gm__ int32_t __out__ *expert_count,
                                          __gm__ int32_t __out__ *expert_start,
                                          __gm__ int32_t __out__ *token_to_packed,
                                          __gm__ T       __in__  *tokens,
                                          __gm__ int32_t __in__  *expert_id)
{
    using namespace moe_top1_permute_cfg;

    using RowShape  = Shape <1, 1, 1, 1, kH>;
    using RowStride = Stride<1, 1, 1, kH, 1>;
    using RowGlobal = GlobalTensor<T, RowShape, RowStride>;

    using RowTile = Tile<TileType::Vec, T,
                         1, kH,
                         BLayout::RowMajor,
                         1, kH>;

    RowTile rowTile;

    // ============================================================
    // Pass 1: histogram. count[e] = #{t : expert_id[t] == e}.
    // Held in a small local int32_t array; no UB tile involved.
    // ============================================================
    int32_t count[kNumExperts];
    for (unsigned e = 0; e < kNumExperts; ++e) {
        count[e] = 0;
    }
    for (unsigned t = 0; t < kT; ++t) {
        int32_t e = expert_id[t];               // GM scalar read
        count[e]++;
    }
    for (unsigned e = 0; e < kNumExperts; ++e) {
        expert_count[e] = count[e];             // GM scalar write
    }

    // ============================================================
    // Pass 2: prefix sum. start[e] = sum_{i<e} count[i].
    // ============================================================
    int32_t start[kNumExperts];
    start[0] = 0;
    for (unsigned e = 1; e < kNumExperts; ++e) {
        start[e] = start[e - 1] + count[e - 1];
    }
    for (unsigned e = 0; e < kNumExperts; ++e) {
        expert_start[e] = start[e];             // GM scalar write
    }

    // ============================================================
    // Pass 3: pack tokens into expert-grouped order, preserving original
    // within-expert order. counter[e] is the running slot index within
    // expert e's segment.
    // ============================================================
    int32_t counter[kNumExperts];
    for (unsigned e = 0; e < kNumExperts; ++e) {
        counter[e] = 0;
    }

    for (unsigned t = 0; t < kT; ++t) {
        int32_t e          = expert_id[t];      // GM scalar read (re-read; cheap, same address)
        int32_t slot       = counter[e]++;
        int32_t packed_pos = start[e] + slot;

        token_to_packed[t] = packed_pos;        // GM scalar write

        size_t src_off = static_cast<size_t>(t)          * kH;
        size_t dst_off = static_cast<size_t>(packed_pos) * kH;

        RowGlobal srcGlobal(tokens        + src_off);
        RowGlobal dstGlobal(packed_tokens + dst_off);

        TLOAD (rowTile, srcGlobal);
        TSTORE(dstGlobal, rowTile);
    }
}

template <typename T>
void launchMoeTop1Permute(T *packed_tokens,
                          int32_t *expert_count, int32_t *expert_start,
                          int32_t *token_to_packed,
                          T *tokens, int32_t *expert_id, void *stream)
{
    runMoeTop1Permute<T><<<1, nullptr, stream>>>(
        packed_tokens, expert_count, expert_start, token_to_packed,
        tokens, expert_id);
}

template void launchMoeTop1Permute<float>(float *packed_tokens,
                                          int32_t *expert_count, int32_t *expert_start,
                                          int32_t *token_to_packed,
                                          float *tokens, int32_t *expert_id, void *stream);
